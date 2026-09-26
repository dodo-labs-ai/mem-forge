"""Each Observation Revision records the source's own time for its content (design 0.9)."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from memforge import main
from memforge.agent_sessions import _primary_event_source_time
from memforge.config import AppConfig
from memforge.genes.github_pages_gene import _content_item_from_url
from memforge.genes.jira_gene import JiraGene
from memforge.genes.teams_gene import TeamsGene
from memforge.local_adapter import submit_local_markdown_document
from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_artifacts import StoredSourceArtifact
from memforge.source_representation import UNIT_TITLE_OBSERVATION_TYPE
from memforge.storage.database import Database

# The time MemForge fetched or received the content; never a source time.
SYNC_TIME = datetime(2026, 9, 26, 8, 0, tzinfo=timezone.utc)
SOURCE_TIME = "2026-03-02T23:30:00-05:00"
SOURCE_TIME_UTC = "2026-03-03T04:30:00+00:00"


def _item(item_id: str = "doc-1", **extra) -> ContentItem:
    return ContentItem(
        item_id=item_id,
        title="Title",
        source_url=f"https://example.test/{item_id}",
        last_modified=SYNC_TIME,
        version="1",
        extra=dict(extra),
    )


def _project(source_type: str, item: ContentItem, body: bytes, semantics: dict, **kwargs):
    raw = RawContent(item=item, body=body, content_type=kwargs.pop("content_type", "text/markdown"))
    normalized = NormalizedContent(item=item, markdown_body=body.decode(), source_semantics=semantics)
    return project_source_item(
        source_id="src-1",
        source_type=source_type,
        run_id=kwargs.pop("run_id", "run-1"),
        item=item,
        raw=raw,
        normalized=normalized,
        **kwargs,
    )


def _revisions_by_type(projection) -> dict[str, object]:
    types = {observation.id: observation.observation_type for observation in projection.observations}
    return {types[revision.observation_id]: revision for revision in projection.observation_revisions}


BODY_SOURCES = [
    ("confluence", "page_body", {"page_id": "123", "space_key": "ENG"}),
    ("github_repo", "file_content", {"relative_path": "docs/a.md", "repo_owner": "org", "repo_name": "repo"}),
    ("github_pages", "page_content", {"canonical_url": "https://org.github.io/docs/a"}),
    ("local_markdown", "file_content", {"relative_path": "notes/a.md"}),
    ("agent_session", "session_summary", {}),
    ("extension_docs", "document_content", {}),
]


@pytest.mark.parametrize(("source_type", "body_type", "extra"), BODY_SOURCES)
def test_body_observation_takes_the_source_time_the_gene_reports(source_type, body_type, extra) -> None:
    projection = _project(source_type, _item(**extra), b"# Body", {"source_updated_at": SOURCE_TIME})

    revisions = _revisions_by_type(projection)
    assert revisions[body_type].observed_at == SOURCE_TIME_UTC
    # The Unit Title is the provider's name for the Unit, not content with a time of its own.
    assert revisions[UNIT_TITLE_OBSERVATION_TYPE].observed_at is None
    assert projection.source_unit_revisions[0].observed_at == SOURCE_TIME_UTC


@pytest.mark.parametrize(("source_type", "body_type", "extra"), BODY_SOURCES)
def test_body_without_a_source_time_never_takes_the_sync_time(source_type, body_type, extra) -> None:
    projection = _project(source_type, _item(**extra), b"# Body", {})

    assert _revisions_by_type(projection)[body_type].observed_at is None
    assert projection.source_unit_revisions[0].observed_at is None


def test_artifact_carries_its_parent_observation_time() -> None:
    artifact = StoredSourceArtifact(
        id="artifact-1",
        provider_key="docs/diagram.png",
        parent_observation_type="file_content",
        parent_provider_key="content",
        provider_revision="blob-1",
        filename="diagram.png",
        media_type="image/png",
        size_bytes=17,
        sha256="a" * 64,
        uri="/store/artifact-1/diagram.png",
        inference_eligible=True,
    )

    projection = _project(
        "github_repo",
        _item(relative_path="docs/diagram.png"),
        b"",
        {"source_updated_at": SOURCE_TIME},
        artifacts=(artifact,),
    )

    assert _revisions_by_type(projection)["binary_artifact"].observed_at == SOURCE_TIME_UTC


def _jira_payload(histories: list[dict], *, total: int | None = None) -> bytes:
    return json.dumps(
        {
            "id": "10001",
            "key": "PAY-1",
            "fields": {
                "summary": "Retain A7",
                "description": "A7 stays for regular payroll.",
                "status": {"name": "Open"},
                "priority": {"name": "Major"},
                "assignee": None,
                "labels": [],
                "resolution": None,
                "created": "2026-01-05T09:00:00.000+0000",
                "updated": "2026-02-20T09:00:00.000+0000",
            },
            "_comments": [
                {"id": "c1", "body": "Agreed.", "created": "2026-02-01T09:00:00.000+0000",
                 "updated": "2026-02-20T09:00:00.000+0000"},
            ],
            "_comments_included": True,
            "_comments_total": 1,
            "changelog": {
                "startAt": 0,
                "histories": histories,
                "total": len(histories) if total is None else total,
            },
        }
    ).encode()


def _history(history_id: str, created: str, field: str) -> dict:
    return {"id": history_id, "created": created, "items": [{"field": field, "fromString": "a", "toString": "b"}]}


def _jira_core_time(payload: bytes) -> str | None:
    projection = _project("jira", _item("jira-PAY-1"), payload, {}, content_type="application/json")
    return _revisions_by_type(projection)["issue_core"].observed_at


def test_jira_core_time_is_its_latest_core_field_change() -> None:
    payload = _jira_payload([
        _history("h1", "2026-01-10T09:00:00.000+0000", "status"),
        # A later change to a field outside the core does not change the core.
        _history("h2", "2026-02-15T09:00:00.000+0000", "Sprint"),
    ])

    assert _jira_core_time(payload) == "2026-01-10T09:00:00+00:00"


def test_jira_core_unchanged_since_creation_takes_the_creation_time() -> None:
    payload = _jira_payload([_history("h1", "2026-02-15T09:00:00.000+0000", "Sprint")])

    assert _jira_core_time(payload) == "2026-01-05T09:00:00+00:00"


def test_jira_core_time_is_unknown_when_the_changelog_is_truncated() -> None:
    payload = _jira_payload([_history("h1", "2026-01-10T09:00:00.000+0000", "status")], total=150)

    # ``fields.updated`` moves with comments too, so it is not the core's time.
    assert _jira_core_time(payload) is None


def test_jira_comments_and_changelog_keep_their_own_times() -> None:
    projection = _project(
        "jira",
        _item("jira-PAY-1"),
        _jira_payload([_history("h1", "2026-01-10T09:00:00.000+0000", "status")]),
        {},
        content_type="application/json",
    )

    revisions = _revisions_by_type(projection)
    assert revisions["comment"].observed_at == "2026-02-20T09:00:00.000+0000"
    assert revisions["changelog"].observed_at == "2026-01-10T09:00:00.000+0000"
    # The Unit's time is the latest time of its Observations.
    assert projection.source_unit_revisions[0].observed_at == "2026-02-20T09:00:00+00:00"


def test_a_revision_recorded_without_a_time_takes_the_source_time_once() -> None:
    item = _item("confluence-123", page_id="123", space_key="ENG")
    first = _project("confluence", item, b"# Body", {}, run_id="run-1")
    prior_revisions = {revision.observation_id: revision for revision in first.observation_revisions}

    second = _project(
        "confluence",
        item,
        b"# Body",
        {"source_updated_at": SOURCE_TIME},
        run_id="run-2",
        prior_unit_revision=first.source_unit_revisions[0],
        prior_observation_revisions=prior_revisions,
    )

    body = _revisions_by_type(second)["page_body"]
    assert body.id == _revisions_by_type(first)["page_body"].id
    assert body.observed_at == SOURCE_TIME_UTC
    # A time correction is no content change.
    assert second.deltas[0].axes == frozenset()
    assert second.deltas[0].changed_anchors == ()
    assert second.source_unit_revisions[0].id == first.source_unit_revisions[0].id

    third = _project(
        "confluence",
        item,
        b"# Body",
        {"source_updated_at": "2026-04-01T00:00:00+00:00"},
        run_id="run-3",
        prior_unit_revision=second.source_unit_revisions[0],
        prior_observation_revisions={revision.observation_id: revision for revision in second.observation_revisions},
    )

    # A recorded source time is never replaced.
    assert _revisions_by_type(third)["page_body"].observed_at == SOURCE_TIME_UTC


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "source-time.db"))
    await database.connect()
    await database.upsert_source(
        id="src-1",
        type="confluence",
        name="Engineering",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    try:
        yield database
    finally:
        await database.close()


async def _stored_times(db: Database, unit_id: str) -> dict[str, str | None]:
    revisions = await db.get_current_source_observation_revisions(unit_id)
    return {revision.id: revision.observed_at for revision in revisions.values()}


@pytest.mark.asyncio
async def test_storage_writes_a_missing_source_time_once(db: Database) -> None:
    item = _item("confluence-123", page_id="123", space_key="ENG")
    untimed = _project("confluence", item, b"# Body", {}, run_id="run-1")
    await db.record_source_projection(untimed)
    unit_id = untimed.source_units[0].id
    body_id = _revisions_by_type(untimed)["page_body"].id
    assert (await _stored_times(db, unit_id))[body_id] is None

    timed = replace(
        untimed,
        run_id="run-2",
        observation_revisions=tuple(
            replace(revision, observed_at=SOURCE_TIME_UTC) if revision.id == body_id else revision
            for revision in untimed.observation_revisions
        ),
    )
    await db.record_source_projection(timed)
    assert (await _stored_times(db, unit_id))[body_id] == SOURCE_TIME_UTC

    retimed = replace(
        timed,
        run_id="run-3",
        observation_revisions=tuple(
            replace(revision, observed_at="2026-04-01T00:00:00+00:00") if revision.id == body_id else revision
            for revision in timed.observation_revisions
        ),
    )
    await db.record_source_projection(retimed)
    assert (await _stored_times(db, unit_id))[body_id] == SOURCE_TIME_UTC


@pytest.mark.asyncio
async def test_migration_gives_current_confluence_bodies_their_page_version_time(tmp_path) -> None:
    path = tmp_path / "confluence-time.db"
    legacy = Database(str(path))
    await legacy.connect()
    await legacy.upsert_source(
        id="src-1",
        type="confluence",
        name="Engineering",
        config_json="{}",
        access_policy="workspace",
        owner_user_id="owner-1",
    )
    page = _project("confluence", _item("confluence-123", page_id="123", space_key="ENG"), b"# Body", {})
    await legacy.record_source_projection(page)
    now = SYNC_TIME.isoformat()
    await legacy.db.execute(
        """INSERT INTO documents (
               doc_id, source, source_url, title, space_or_project, last_modified,
               version, content_hash, last_synced, created_at, updated_at
           ) VALUES ('confluence-123', 'src-1', 'https://example.test', 'Title', 'ENG', ?, '7', 'h', ?, ?, ?)""",
        ("2026-02-11T10:00:00+00:00", now, now, now),
    )
    await legacy.db.execute("DELETE FROM schema_migrations WHERE version = 101")
    await legacy.db.commit()
    await legacy.close()

    migrated = Database(str(path))
    await migrated.connect()
    try:
        times = {
            revision.observation_id: revision.observed_at
            for revision in (
                await migrated.get_current_source_observation_revisions(page.source_units[0].id)
            ).values()
        }
        types = {observation.id: observation.observation_type for observation in page.observations}
        assert {types[observation_id]: time for observation_id, time in times.items()} == {
            "page_body": "2026-02-11T10:00:00+00:00",
            UNIT_TITLE_OBSERVATION_TYPE: None,
        }
    finally:
        await migrated.close()


def _local_markdown_source(tmp_path: Path) -> tuple[AppConfig, dict]:
    config = AppConfig(base_dir=tmp_path / "mem")
    source = {"id": "src-notes", "type": "local_markdown", "config": {"vault_id": "notes"}}
    return config, source


@pytest.mark.asyncio
async def test_local_package_records_the_file_time_it_is_given(tmp_path) -> None:
    config, source = _local_markdown_source(tmp_path)

    timed = await submit_local_markdown_document(
        db=None, config=config, source=source, vault_id="notes", relative_path="a.md",
        markdown_body="# A", source_updated_at=SOURCE_TIME,
    )
    untimed = await submit_local_markdown_document(
        db=None, config=config, source=source, vault_id="notes", relative_path="b.md", markdown_body="# B",
    )

    assert json.loads(Path(timed["package_path"]).read_text())["source_updated_at"] == SOURCE_TIME_UTC
    # An older local agent sends no file time; the submission time never stands in for it.
    assert "source_updated_at" not in json.loads(Path(untimed["package_path"]).read_text())


@pytest.mark.asyncio
async def test_local_package_rejects_a_file_time_without_offset(tmp_path) -> None:
    config, source = _local_markdown_source(tmp_path)

    with pytest.raises(ValueError, match="timezone offset"):
        await submit_local_markdown_document(
            db=None, config=config, source=source, vault_id="notes", relative_path="a.md",
            markdown_body="# A", source_updated_at="2026-03-02T23:30:00",
        )


def _git(root: Path, *args: str, date: str | None = None) -> None:
    env = {**os.environ, "GIT_AUTHOR_DATE": date or "", "GIT_COMMITTER_DATE": date or ""}
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.test", *args],
        check=True,
        capture_output=True,
        env=env if date else os.environ,
    )


def test_local_file_time_is_its_commit_time_unless_changed_locally(tmp_path) -> None:
    root = tmp_path / "vault"
    (root / "docs").mkdir(parents=True)
    _git(root, "init", "-q")
    for name in ("clean.md", "edited.md"):
        (root / "docs" / name).write_text(f"# {name}\n")
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "notes", date="2026-01-15T12:00:00+00:00")
    (root / "docs" / "edited.md").write_text("# edited locally\n")
    (root / "docs" / "new.md").write_text("# new\n")
    modified_epoch = datetime(2026, 5, 6, 7, 8, tzinfo=timezone.utc).timestamp()
    for name in ("clean.md", "edited.md", "new.md"):
        # A checkout sets every file's modification time to when it ran.
        os.utime(root / "docs" / name, (modified_epoch, modified_epoch))
    counts = {"included": 0, "ignored": 0, "too_large": 0, "invalid_utf8": 0, "unreadable": 0}
    # The collection root may be a subdirectory of the checkout.
    collection_root = root / "docs"
    entries = {
        entry["relative_path"]: entry
        for entry in main._scan_kb_profile(collection_root, include=["**/*.md", "*.md"], exclude=[], counts=counts)
    }
    worktree = main._git_worktree(collection_root)

    times = {path: main._local_file_source_time(collection_root, entry, worktree) for path, entry in entries.items()}

    assert times == {
        "clean.md": "2026-01-15T12:00:00+00:00",
        "edited.md": "2026-05-06T07:08:00+00:00",
        "new.md": "2026-05-06T07:08:00+00:00",
    }


def test_local_file_outside_git_takes_its_modification_time(tmp_path) -> None:
    root = tmp_path / "plain"
    root.mkdir()
    (root / "a.md").write_text("# A\n")
    modified_epoch = datetime(2026, 5, 6, 7, 8, tzinfo=timezone.utc).timestamp()
    os.utime(root / "a.md", (modified_epoch, modified_epoch))
    counts = {"included": 0, "ignored": 0, "too_large": 0, "invalid_utf8": 0, "unreadable": 0}
    [entry] = list(main._scan_kb_profile(root, include=["*.md"], exclude=[], counts=counts))

    assert main._git_worktree(root) is None
    assert main._local_file_source_time(root, entry, None) == "2026-05-06T07:08:00+00:00"


def test_agent_patch_time_is_its_authorizing_event_time() -> None:
    events = [
        {"evidence_id": "E1", "timestamp": "2026-06-01T09:00:00Z"},
        {"evidence_id": "E2", "timestamp": "2026-06-01T09:05:00+02:00"},
        {"evidence_id": "E3"},
        {"evidence_id": "E4", "timestamp": "2026-06-01T09:10:00"},
    ]

    def time_of(event_id: str | None):
        return _primary_event_source_time(SimpleNamespace(primary_event_id=event_id), events)

    assert time_of("E2") == datetime(2026, 6, 1, 7, 5, tzinfo=timezone.utc)
    # An event without an offset-aware timestamp has no source time.
    assert time_of("E3") is None
    assert time_of("E4") is None
    assert time_of(None) is None


@pytest.mark.asyncio
async def test_jira_issue_time_is_its_updated_field() -> None:
    gene = JiraGene(config={"base_url": "https://jira.example.test"}, source_id="src-jira")
    item = _item("jira-PAY-1", issue_key="PAY-1")

    normalized = await gene.normalize(RawContent(item=item, body=_jira_payload([]), content_type="application/json"))

    assert normalized.source_semantics["source_updated_at"] == "2026-02-20T09:00:00+00:00"


@pytest.mark.asyncio
async def test_teams_window_time_is_its_latest_message_time() -> None:
    gene = TeamsGene(config={"conversation_ids": ["19:conversation@thread.v2"]}, source_id="src-teams")
    item = _item("teams-window-1")
    payload = {
        "messages": [
            {"id": "m1", "content": "First", "time": "2026-07-30T10:00:00Z"},
            {"id": "m2", "content": "Edited", "time": "2026-07-30T10:01:00Z",
             "lastModifiedDateTime": "2026-07-30T11:30:00Z"},
            {"id": "m3", "content": "Last", "time": "2026-07-30T10:05:00Z"},
        ]
    }

    normalized = await gene.normalize(
        RawContent(item=item, body=json.dumps(payload).encode(), content_type="application/json")
    )

    assert normalized.source_semantics["source_updated_at"] == "2026-07-30T11:30:00+00:00"



def test_pages_http_time_is_the_last_modified_header_and_never_the_fetch_time() -> None:
    reported = _content_item_from_url(
        "https://org.github.io/docs/a", {"Last-Modified": "Wed, 06 May 2026 07:08:00 GMT"}
    )
    unreported = _content_item_from_url("https://org.github.io/docs/b", {})

    assert reported.extra["content_updated_at"] == "2026-05-06T07:08:00+00:00"
    assert unreported.extra["content_updated_at"] is None
