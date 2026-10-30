"""The evaluation snapshot acquisition: off by default, confined to its root, chosen at the construction seams.

A Source that names an evaluation snapshot set replays it instead of reading
its provider. The deployment turns replay on by naming the snapshot root; the
Gene construction seam, the conversation provider seam and the executor choice
read the Source, so the sync pipeline never does.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.conversations.port import ConversationAccess, Interval, PollOutcome
from memforge.evals.source_snapshots import (
    SNAPSHOT_CONFIG_FIELD,
    SNAPSHOT_ROOT_VARIABLE,
    SnapshotConversationProvider,
    SnapshotsDisabled,
    SnapshotUnavailable,
    conversation_snapshots,
    replay_offset,
)
from memforge.genes import create_gene, source_lists_scope, source_rediscovers_documents
from memforge.genes.jira_gene import JiraGene
from memforge.genes.teams_gene import TeamsGene
from memforge.local_agent.source_contract import local_agent_job_config, local_agent_sync_operation
from memforge.models import AbsenceAnswer, ContentItem, ScopeListingKind
from memforge.source_types import source_type_descriptor
from tests.test_source_authority_api import _app, _connect_database
from tests.test_startup_runtime import TEST_SOURCE_KEY

SNAPSHOTS = Path(__file__).resolve().parents[1] / "evals" / "quality" / "quick" / "snapshots"
REPLAYED_AT = "2026-10-30T12:00:00+00:00"


def _replaying(source_type: str, revision: int = 1, **config) -> dict:
    spec = json.loads((SNAPSHOTS / source_type / "source.json").read_text(encoding="utf-8"))["config"]
    return {
        **spec,
        **config,
        SNAPSHOT_CONFIG_FIELD: {"set": source_type, "revision": revision, "replayed_at": REPLAYED_AT},
    }


@pytest.fixture
def replay_on(monkeypatch):
    monkeypatch.setenv(SNAPSHOT_ROOT_VARIABLE, str(SNAPSHOTS))


@pytest.fixture
def replay_off(monkeypatch):
    monkeypatch.delenv(SNAPSHOT_ROOT_VARIABLE, raising=False)


def test_a_snapshot_source_gets_no_gene_while_the_deployment_does_not_replay_snapshots(replay_off) -> None:
    with pytest.raises(SnapshotsDisabled, match=SNAPSHOT_ROOT_VARIABLE):
        create_gene("jira", _replaying("jira"), "src-jira")
    with pytest.raises(SnapshotsDisabled, match=SNAPSHOT_ROOT_VARIABLE):
        create_gene("teams", _replaying("teams"), "src-teams")


async def test_a_conversation_snapshot_opens_no_provider_while_the_deployment_does_not_replay_snapshots(
    replay_off,
) -> None:
    reference = source_type_descriptor("teams").conversation_provider
    with pytest.raises(SnapshotsDisabled):
        async with reference.open(_replaying("teams")):
            pass


@pytest.mark.parametrize("set_name", ["../snapshots/jira/../..", "/etc", "missing"])
def test_a_source_reads_only_a_set_below_the_snapshot_root(replay_on, set_name) -> None:
    config = {**_replaying("jira"), SNAPSHOT_CONFIG_FIELD: {"set": set_name, "revision": 1}}

    with pytest.raises(SnapshotUnavailable):
        create_gene("jira", config, "src-jira")


def test_a_source_that_reads_its_provider_keeps_its_gene(replay_off) -> None:
    gene = create_gene("jira", {"base_url": "https://jira.example.test", "projects": ["KES"]}, "src-jira")

    assert type(gene) is JiraGene


async def test_a_document_snapshot_gene_is_the_real_gene_acquiring_from_the_set(replay_on) -> None:
    gene = create_gene("jira", _replaying("jira", revision=2), "src-jira")
    await gene.authenticate()

    items = [item async for item in gene.discover(datetime(2027, 1, 1, tzinfo=timezone.utc))]
    listing = await gene.list_scope()
    stray = ContentItem(item_id="jira-KES-1", title="", source_url="", last_modified=datetime.now(timezone.utc))
    answers = await gene.confirm_absent([items[0], stray])
    raw = await gene.fetch(items[0])

    assert isinstance(gene, JiraGene) and type(gene).__name__ == "SnapshotJiraGene"
    # Discovery yields the whole set whatever the cursor; an unchanged item is not extracted again.
    assert [item.item_id for item in items] == ["jira-KES-42"]
    assert listing.doc_ids == {"jira-KES-42"} and listing.kind is ScopeListingKind.QUERY
    assert answers == {"jira-KES-42": AbsenceAnswer.EXISTS, "jira-KES-1": AbsenceAnswer.ABSENT}
    assert len(json.loads(raw.body)["_comments"]) == 3
    assert [artifact.media_type for artifact in raw.artifacts] == ["image/png"]
    async with gene.open_source_artifact(raw.artifacts[0]) as download:
        assert b"".join([chunk async for chunk in download.chunks]).startswith(b"\x89PNG")
    # A Document the run did not discover is asked of the set, never of the provider.
    assert (await gene.rediscover(items[0])).item_id == "jira-KES-42"
    assert await gene.rediscover(stray) is None
    assert source_rediscovers_documents("confluence", _replaying("confluence"))
    assert source_lists_scope("github_pages", _replaying("github_pages"))


@pytest.mark.parametrize(
    ("layout", "reason"),
    [
        ({"KES-42": "outside"}, "is a link"),
        ({"KES-42/1": "outside"}, "is a link"),
        ({"KES-42/1": "real", "ALIAS": "KES-42"}, "is a link"),
        ({"KES-42/1": "real", "KES-42/2": "KES-42/1"}, "is a link"),
        ({"KES-42/1": "real", "KES-42/02": "real"}, "is not a revision"),
    ],
)
async def test_a_set_reads_only_its_own_numbered_directories(tmp_path, monkeypatch, layout, reason) -> None:
    source = SNAPSHOTS / "jira" / "KES-42" / "1"
    set_dir = tmp_path / "root" / "jira"
    set_dir.mkdir(parents=True)
    for name, target in layout.items():
        path = set_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if target == "real":
            shutil.copytree(source, path)
        elif target == "outside":
            path.symlink_to(source if "/" in name else source.parent, target_is_directory=True)
        else:
            path.symlink_to(set_dir / target, target_is_directory=True)
    monkeypatch.setenv(SNAPSHOT_ROOT_VARIABLE, str(tmp_path / "root"))

    with pytest.raises(SnapshotUnavailable, match=reason):
        await create_gene("jira", _replaying("jira"), "src-jira").authenticate()


async def test_an_item_stands_at_its_highest_revision_not_above_the_one_replayed(replay_on, tmp_path, monkeypatch) -> None:
    set_dir = tmp_path / "jira"
    for item, revisions in (("KES-42", (1, 2)), ("KES-43", (2,))):
        for revision in revisions:
            source = SNAPSHOTS / "jira" / "KES-42" / str(min(revision, 2))
            target = set_dir / item / str(revision)
            target.mkdir(parents=True)
            spec = json.loads((source / "item.json").read_text())
            spec["item"]["item_id"] = f"jira-{item}"
            (target / "item.json").write_text(json.dumps({**spec, "artifacts": []}))
            (target / "raw.json").write_bytes((source / "raw.json").read_bytes())
    monkeypatch.setenv(SNAPSHOT_ROOT_VARIABLE, str(tmp_path))

    async def items(revision: int) -> list[str]:
        gene = create_gene("jira", _replaying("jira", revision=revision), "src-jira")
        return [item.item_id async for item in gene.discover(None)]

    assert await items(1) == ["jira-KES-42"]
    assert await items(2) == ["jira-KES-42", "jira-KES-43"]
    assert await items(3) == ["jira-KES-42", "jira-KES-43"]


def test_a_document_snapshot_source_runs_on_the_server_and_a_conversation_one_on_the_device() -> None:
    # local_markdown is always collected by the agent; replaying a set, the server reads it.
    assert local_agent_sync_operation("local_markdown", {"root": "/notes"}) == "local_markdown_sync"
    assert local_agent_sync_operation("local_markdown", _replaying("local_markdown")) is None
    assert local_agent_sync_operation("jira", _replaying("jira", sync_mode="local_agent")) is None
    assert local_agent_sync_operation("teams", _replaying("teams")) == "teams_sync"
    assert local_agent_job_config("teams", _replaying("teams"))[SNAPSHOT_CONFIG_FIELD] == {
        "set": "teams", "revision": 1, "replayed_at": REPLAYED_AT,
    }


async def test_the_device_executor_opens_the_snapshot_provider_for_a_snapshot_job(replay_on) -> None:
    reference = source_type_descriptor("teams").conversation_provider
    replayed = datetime.fromisoformat(REPLAYED_AT)

    async with reference.open(_replaying("teams", revision=2)) as provider:
        listing = await provider.list_conversations()
        ref = listing.conversations[0]
        everything = await provider.poll(ref, Interval(start=replayed - timedelta(days=30)))
        recent = await provider.poll(ref, Interval(start=replayed - timedelta(minutes=95)))
        unknown, _ = await provider.get_conversation("19:other@thread.v2")

    assert isinstance(provider, SnapshotConversationProvider)
    assert provider.record_schema is reference.provider_class().record_schema
    assert listing.statuses[ref.conversation_id].access is ConversationAccess.READABLE
    assert [message.created_at for message in everything.messages] == sorted(
        (message.created_at for message in everything.messages), reverse=True
    )
    assert len(everything.messages) == 4 and len(recent.messages) == 1
    assert everything.outcome is PollOutcome.REACHED_START
    assert everything.earliest_complete_page_time < replayed - timedelta(days=30)
    assert unknown.access is ConversationAccess.GONE
    # The newest message moved by whole days to just before the replay began, keeping its time of day.
    newest = everything.messages[0]
    assert newest.created_at == datetime(2026, 10, 30, 10, 40, tzinfo=timezone.utc)
    assert newest.record["time"] == newest.created_at.isoformat()


def test_replay_moves_a_conversation_by_whole_days() -> None:
    newest = datetime(2026, 10, 21, 10, 40, tzinfo=timezone.utc)

    assert replay_offset(newest, datetime(2026, 10, 30, 12, tzinfo=timezone.utc)) == timedelta(days=9)
    assert replay_offset(newest, datetime(2026, 10, 30, 11, 30, tzinfo=timezone.utc)) == timedelta(days=8)
    assert all(
        snapshot.ref.conversation_id
        for snapshot in conversation_snapshots(SNAPSHOTS / "teams", 1, replayed_at=datetime.now(timezone.utc))
    )


def test_a_conversation_snapshot_source_keeps_its_own_gene(replay_on) -> None:
    assert type(create_gene("teams", _replaying("teams"), "src-teams")) is TeamsGene


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
    database = _connect_database(tmp_path)
    try:
        with TestClient(_app(tmp_path, database)) as test_client:
            yield test_client, database
    finally:
        asyncio.run(database.close())


def _create(client: TestClient, source_type: str, config: dict):
    return client.post(
        "/api/v1/sources",
        json={"type": source_type, "name": f"Snapshot {source_type}", "access_policy": "private", "config": config},
    )


def test_creating_a_snapshot_source_is_refused_while_the_deployment_does_not_replay_snapshots(
    client, replay_off
) -> None:
    test_client, _ = client

    response = _create(test_client, "jira", _replaying("jira"))

    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "config_invalid"
    assert detail["fields"][0]["key"] == SNAPSHOT_CONFIG_FIELD
    assert SNAPSHOT_ROOT_VARIABLE in detail["message"]


def test_a_snapshot_source_keeps_its_set_and_moves_only_its_revision(client, replay_on) -> None:
    test_client, database = client
    config = _replaying("jira")
    del config[SNAPSHOT_CONFIG_FIELD]["replayed_at"]

    created = _create(test_client, "jira", config)
    assert created.status_code == 200, created.text
    source_id = created.json()["id"]
    stored = asyncio.run(database.get_source(source_id))
    replayed_at = stored["config"][SNAPSHOT_CONFIG_FIELD]["replayed_at"]

    def save(reference: dict | None):
        changed = {key: value for key, value in _replaying("jira").items() if key != SNAPSHOT_CONFIG_FIELD}
        if reference is not None:
            changed[SNAPSHOT_CONFIG_FIELD] = reference
        return test_client.put(f"/api/v1/sources/{source_id}", json={"config": changed})

    assert save({"set": "jira", "revision": 2}).status_code == 200
    moved = asyncio.run(database.get_source(source_id))["config"][SNAPSHOT_CONFIG_FIELD]
    assert moved == {"set": "jira", "revision": 2, "replayed_at": replayed_at}
    assert save({"set": "jira", "revision": 3}).status_code == 400
    assert save({"set": "confluence", "revision": 1}).status_code == 400
    assert save(None).status_code == 400
    # The document type's snapshot Source syncs on the server, so it has no device to run on.
    assert stored["execution_owner_user_id"] is None


def test_a_configuration_that_names_no_set_is_refused(client, replay_on) -> None:
    test_client, _ = client
    config = {**_replaying("local_markdown"), SNAPSHOT_CONFIG_FIELD: None}

    response = _create(test_client, "local_markdown", config)

    assert response.status_code == 400, response.text
    assert response.json()["detail"]["fields"][0]["key"] == SNAPSHOT_CONFIG_FIELD


def test_syncing_a_snapshot_source_is_refused_once_the_deployment_stops_replaying(client, monkeypatch) -> None:
    test_client, _ = client
    monkeypatch.setenv(SNAPSHOT_ROOT_VARIABLE, str(SNAPSHOTS))
    jira = _create(test_client, "jira", _replaying("jira")).json()["id"]
    teams = _create(test_client, "teams", _replaying("teams")).json()["id"]
    monkeypatch.delenv(SNAPSHOT_ROOT_VARIABLE)

    responses = [
        test_client.post(f"/api/v1/sources/{jira}/sync"),
        test_client.post(f"/api/v1/sources/{teams}/sync"),
        test_client.post(f"/api/v1/sources/{teams}/sync", json={"dry_run": True}),
    ]
    preview = test_client.post("/api/v1/genes/jira/preview-discovery", json={"config": _replaying("jira"), "limit": 5})

    assert [response.status_code for response in responses] == [409] * 3, [r.text for r in responses]
    assert all(SNAPSHOT_ROOT_VARIABLE in response.json()["detail"] for response in responses)
    # A preview decides as saving would: the configuration is invalid while replay is off.
    assert preview.status_code == 400, preview.text
    assert preview.json()["detail"]["fields"][0]["key"] == SNAPSHOT_CONFIG_FIELD


def test_a_source_that_reads_its_provider_cannot_start_replaying(client, replay_on) -> None:
    test_client, _ = client
    plain = {key: value for key, value in _replaying("jira").items() if key != SNAPSHOT_CONFIG_FIELD}
    source_id = _create(test_client, "jira", plain).json()["id"]

    response = test_client.put(f"/api/v1/sources/{source_id}", json={"config": _replaying("jira")})

    assert response.status_code == 400, response.text
