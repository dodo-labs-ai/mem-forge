"""ADR 0040: stored input is current, reprocess reads the provider, revisions are content-addressed,
and recovery isolates Source Units."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import httpx
import pytest

from memforge.llm.structured import failure_retryable
from memforge.memory.engine import SourceUnitLifecycleExecutionError
from memforge.models import ContentItem, GeneMetadata, NormalizedContent, RawContent
from memforge.pipeline.source_projection_adapters import jira_changelog_semantic_class
from memforge.pipeline.sync import GeneSyncOrchestrator, SourceSyncMode
from memforge.source_activity import SourceActivityConflict
from memforge.source_derivation import DERIVATION_DETERMINISTIC_FAILURE
from memforge.storage.database import Database
from tests.test_source_reprocess import RawKeepingDocumentStore
from tests.test_sync_bookkeeping import (
    FailingMemoryExtractor,
    NoopMemoryEngine,
    ProjectionFragmentRecordingExtractor,
    RecordingMemoryEngine,
)

SOURCE_ID = "src-jira"
ISSUES = {"PAY-1": "400001", "PAY-2": "400002"}
SYNCED_AT = datetime(2026, 9, 20, tzinfo=timezone.utc)
DOMAIN_HISTORY = {"id": "9001", "created": "2026-09-01T08:00:00.000+0000", "items": [{"field": "summary"}]}
SPRINT_HISTORY = {"id": "9002", "created": "2026-09-02T08:00:00.000+0000", "items": [{"field": "Sprint"}]}


def doc_id(issue_key: str) -> str:
    return f"jira-{issue_key}"


def authored(history: dict, author: str) -> dict:
    return {**history, "author": {"displayName": author}}


class JiraProvider:
    """A Jira provider whose issues carry changelog entries that never reach the markdown."""

    discovery_complete = True

    def __init__(self) -> None:
        self.histories: dict[str, list[dict]] = {key: [authored(DOMAIN_HISTORY, "Ann")] for key in ISSUES}
        self.removed: set[str] = set()
        self.rediscovered: list[str] = []
        self.urls = {key: f"https://jira.example/browse/{key}" for key in ISSUES}

    @classmethod
    def metadata(cls):
        return GeneMetadata(
            name="jira", display_name="Jira", description="", default_sync_interval_minutes=60,
            auth_method="pat", data_shape="ticket",
        )

    def requires_pdf_artifact(self, **kwargs) -> bool:
        return False

    async def authenticate(self) -> None:
        return None

    def _item(self, issue_key: str) -> ContentItem:
        return ContentItem(
            item_id=doc_id(issue_key), title=f"{issue_key}: Payroll run",
            source_url=self.urls[issue_key], last_modified=SYNCED_AT,
            content_type="application/json", space_or_project="PAY", version="1",
            extra={"issue_id": ISSUES[issue_key], "issue_key": issue_key},
        )

    async def discover(self, since=None):
        for issue_key in ISSUES:
            if issue_key not in self.removed:
                yield self._item(issue_key)

    async def rediscover(self, item):
        issue_key = item.extra["issue_key"]
        self.rediscovered.append(issue_key)
        return None if issue_key in self.removed else self._item(issue_key)

    def payload(self, issue_key: str) -> bytes:
        histories = self.histories[issue_key]
        return json.dumps({
            "id": ISSUES[issue_key],
            "key": issue_key,
            "fields": {
                "summary": "Payroll run", "description": "Body", "status": None, "priority": None,
                "assignee": None, "labels": [], "resolution": None, "updated": SYNCED_AT.isoformat(),
            },
            "_comments": [], "_comments_included": True, "_comments_total": 0,
            "changelog": {"startAt": 0, "histories": histories, "total": len(histories)},
        }).encode("utf-8")

    async def fetch(self, item):
        return RawContent(item=item, body=self.payload(item.extra["issue_key"]), content_type="application/json")

    async def normalize(self, raw):
        return NormalizedContent(item=raw.item, markdown_body=f"# {raw.item.title}\n\nBody")


class Harness:
    def __init__(self, db: Database) -> None:
        self.db = db
        self.store = RawKeepingDocumentStore()
        self.provider = JiraProvider()
        self.extractor = ProjectionFragmentRecordingExtractor()
        self.engine = RecordingMemoryEngine()

    def orchestrator(self, extractor=None) -> GeneSyncOrchestrator:
        return GeneSyncOrchestrator(
            db=self.db, doc_store=self.store, memory_extractor=extractor or self.extractor,
            memory_engine=self.engine, memory_store=None, max_concurrent=1,
        )

    async def sync(self, extractor=None):
        return await self.orchestrator(extractor).sync_gene(
            gene=self.provider, source_name="Jira", source_id=SOURCE_ID,
        )

    async def reprocess(self, *issue_keys: str, extractor=None):
        return await self.orchestrator(extractor).sync_gene(
            gene=self.provider, source_name="Jira", source_id=SOURCE_ID,
            execution_mode=SourceSyncMode.REPROCESS,
            reprocess_doc_ids=frozenset(doc_id(key) for key in issue_keys),
        )

    async def unit_id(self, issue_key: str) -> str:
        return (await self.db.find_source_unit_by_document_id(SOURCE_ID, doc_id(issue_key), current_only=True)).id

    async def current_changelog(self, issue_key: str) -> dict[str, str]:
        """Current changelog revisions of one issue, by provider key."""
        projection = await self.db.get_current_source_unit_projection(await self.unit_id(issue_key))
        observations = {observation.id: observation for observation in projection.observations}
        return {
            observations[revision.observation_id].provider_key: revision.id
            for revision in projection.observation_revisions
            if observations[revision.observation_id].observation_type == "changelog"
        }

    def stored_raw(self, uri: str) -> dict:
        return json.loads(self.store.source_artifacts[uri])


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "currency.db"))
    await database.connect()
    NoopMemoryEngine.db = database
    try:
        yield database
    finally:
        NoopMemoryEngine.db = None
        await database.close()


async def synced(db: Database) -> Harness:
    await db.upsert_source(
        id=SOURCE_ID, type="jira", name="Jira", config_json="{}", access_policy="workspace", owner_user_id="dev",
    )
    harness = Harness(db)
    assert (await harness.sync()).last_sync_status == "success"
    return harness


@pytest.mark.asyncio
async def test_a_new_revision_stores_its_raw_input_although_the_markdown_is_unchanged(db):
    harness = await synced(db)
    before = await db.get_document(doc_id("PAY-1"))
    harness.provider.histories["PAY-1"].append(authored(SPRINT_HISTORY, "Ann"))

    assert (await harness.sync()).last_sync_status == "success"

    after = await db.get_document(doc_id("PAY-1"))
    assert after.content_hash == before.content_hash
    assert set(await harness.current_changelog("PAY-1")) == {"9001", "9002"}
    assert [history["id"] for history in harness.stored_raw(after.raw_content_uri)["changelog"]["histories"]] == [
        "9001", "9002",
    ]


@pytest.mark.asyncio
async def test_reprocess_reads_the_provider_and_keeps_entries_missing_only_from_stale_storage(db):
    harness = await synced(db)
    stale_body = harness.provider.payload("PAY-1")
    harness.provider.histories["PAY-1"].append(authored(SPRINT_HISTORY, "Ann"))
    assert (await harness.sync()).last_sync_status == "success"
    committed = await harness.current_changelog("PAY-1")
    # Stored input written before a sync kept it current lacks the newer entry.
    harness.store.source_artifacts[(await db.get_document(doc_id("PAY-1"))).raw_content_uri] = stale_body

    state = await harness.reprocess("PAY-1")

    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    assert harness.provider.rediscovered == ["PAY-1"]
    assert await harness.current_changelog("PAY-1") == committed
    [lifecycle] = harness.engine.projected_lifecycle_calls[-1:]
    assert lifecycle["projection"].deltas[0].removed_observation_ids == ()
    assert lifecycle["derivation_support_without_baseline"] is True


@pytest.mark.asyncio
async def test_a_document_the_provider_no_longer_returns_is_reported_and_keeps_its_unit(db):
    harness = await synced(db)
    unit_id = await harness.unit_id("PAY-1")
    committed = await db.get_current_source_unit_revision(unit_id)
    harness.provider.removed.add("PAY-1")
    harness.engine.projected_lifecycle_calls.clear()

    state = await harness.reprocess("PAY-1", "PAY-2")

    assert (state.docs_processed, state.docs_failed) == (1, 1)
    [failed] = state.failed_docs
    assert (failed.doc_id, failed.error.split(":")[0]) == (doc_id("PAY-1"), "provider_document_missing")
    assert await db.get_current_source_unit_revision(unit_id) == committed
    assert [call["doc_id"] for call in harness.engine.projected_lifecycle_calls] == [doc_id("PAY-2")]


@pytest.mark.asyncio
async def test_a_value_that_returns_to_a_stored_revision_reuses_that_row_as_stored(db):
    harness = await synced(db)
    [first_id] = (await harness.current_changelog("PAY-1")).values()
    # The first revision was recorded before its changelog class was derived.
    await db.db.execute(
        "UPDATE source_observation_revisions SET metadata_json = ? WHERE id = ?",
        (json.dumps({"provider_key": "9001"}), first_id),
    )
    await db.db.commit()
    harness.provider.histories["PAY-1"] = [authored(DOMAIN_HISTORY, "Ann Lee")]
    assert (await harness.sync()).last_sync_status == "success"
    assert (await harness.current_changelog("PAY-1"))["9001"] != first_id

    harness.provider.histories["PAY-1"] = [authored(DOMAIN_HISTORY, "Ann")]
    state = await harness.sync()

    assert (state.last_sync_status, state.docs_failed) == ("success", 0)
    assert (await harness.current_changelog("PAY-1"))["9001"] == first_id
    stored = await db.get_source_observation_revisions((first_id,))
    assert dict(stored[first_id].metadata) == {"provider_key": "9001"}
    [lifecycle] = harness.engine.projected_lifecycle_calls[-1:]
    [projected] = [revision for revision in lifecycle["projection"].observation_revisions if revision.id == first_id]
    assert dict(projected.metadata) == {"provider_key": "9001"}


@pytest.mark.asyncio
async def test_the_migration_classifies_stored_changelog_revisions_without_a_class(tmp_path):
    path = tmp_path / "legacy.db"
    legacy = Database(str(path))
    await legacy.connect()
    NoopMemoryEngine.db = legacy
    try:
        await legacy.upsert_source(
            id=SOURCE_ID, type="jira", name="Jira", config_json="{}", access_policy="workspace",
            owner_user_id="dev",
        )
        harness = Harness(legacy)
        harness.provider.histories["PAY-1"].append(authored(SPRINT_HISTORY, "Ann"))
        assert (await harness.sync()).last_sync_status == "success"
        revisions = await harness.current_changelog("PAY-1")
        classified = {
            revision_id: {**dict(revision.metadata)}
            for revision_id, revision in (await legacy.get_source_observation_revisions(revisions.values())).items()
        }
        await legacy.db.execute(
            "UPDATE source_observation_revisions SET metadata_json = json_remove(metadata_json, '$.semantic_class')"
        )
        await legacy.db.execute("DELETE FROM schema_migrations WHERE version = 102")
        await legacy.db.commit()
    finally:
        NoopMemoryEngine.db = None
        await legacy.close()

    migrated = Database(str(path))
    await migrated.connect()
    try:
        stored = await migrated.get_source_observation_revisions(revisions.values())
        assert {revision_id: dict(revision.metadata) for revision_id, revision in stored.items()} == classified
        assert {
            revision.metadata["provider_key"]: revision.metadata["semantic_class"] for revision in stored.values()
        } == {
            "9001": jira_changelog_semantic_class(DOMAIN_HISTORY),
            "9002": jira_changelog_semantic_class(SPRINT_HISTORY),
        }
        assert jira_changelog_semantic_class(SPRINT_HISTORY) == "operational_transition"
    finally:
        await migrated.close()


async def _staged_changes(harness: Harness) -> dict[str, object]:
    """Stage a derivation for a new revision of each issue whose extraction fails, keyed by issue."""
    for issue_key in ISSUES:
        harness.provider.histories[issue_key].append(authored(SPRINT_HISTORY, "Ann"))
    failed = await harness.sync(FailingMemoryExtractor())
    assert failed.docs_failed == len(ISSUES)
    attempts = await harness.db.list_source_derivation_attempts(source_id=SOURCE_ID, statuses=("retryable_failure",))
    return {attempt.context.document.doc_id.removeprefix("jira-"): attempt for attempt in attempts}


async def _conflict_with_stored_identity(db: Database, revision_id: str) -> None:
    await db.db.execute(
        "UPDATE source_observation_revisions SET semantic_hash = 'another-semantic-value' WHERE id = ?",
        (revision_id,),
    )
    await db.db.commit()


@pytest.mark.asyncio
async def test_recovery_ends_a_derivation_that_cannot_commit_before_any_model_call_and_continues(db):
    harness = await synced(db)
    staged = await _staged_changes(harness)
    # PAY-1's staged projection carries a stored revision whose identity no longer matches.
    projection = staged["PAY-1"].projection
    observations = {observation.id: observation for observation in projection.observations}
    [issue_core] = [
        revision.id
        for revision in projection.observation_revisions
        if observations[revision.observation_id].observation_type == "issue_core"
    ]
    await _conflict_with_stored_identity(db, issue_core)
    harness.provider.removed.update(ISSUES)
    harness.extractor.fragment_calls.clear()

    state = await harness.sync()

    assert state.docs_failed == 1
    [failed] = state.failed_docs
    assert (failed.doc_id, failed.error) == (
        doc_id("PAY-1"), f"immutable projection identity mismatch: source_observation_revisions:{issue_core}",
    )
    assert state.failure_retryable is False
    ended = await db.list_source_derivation_attempts(source_id=SOURCE_ID, statuses=("superseded",))
    assert [(attempt.id, attempt.terminal_reason_code) for attempt in ended] == [
        (staged["PAY-1"].id, DERIVATION_DETERMINISTIC_FAILURE),
    ]
    applied = await db.list_source_derivation_attempts(source_id=SOURCE_ID, statuses=("applied",))
    assert staged["PAY-2"].id in {attempt.id for attempt in applied}
    # Only PAY-2's pending extraction ran.
    assert harness.extractor.fragment_calls
    assert all(
        fragment.anchor.observation_id in {item.id for item in staged["PAY-2"].projection.observations}
        for catalog in harness.extractor.fragment_calls
        for fragment in catalog.fragments
    )


class _WrappingLifecycleEngine(RecordingMemoryEngine):
    """Fails the first lifecycle commit as the lifecycle engine reports a failure: wrapped, with its retry rule."""

    def __init__(self, cause: Exception) -> None:
        super().__init__()
        self.cause = cause

    async def prepare_and_commit_projected_lifecycle(self, **kwargs):
        if self.cause is not None:
            cause, self.cause = self.cause, None
            raise SourceUnitLifecycleExecutionError(
                "projected lifecycle commit failed", None, retryable=failure_retryable(cause),
            ) from cause
        return await super().prepare_and_commit_projected_lifecycle(**kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cause",
    [
        SourceActivityConflict("source activity epoch changed: expected 1, current 2"),
        sqlite3.OperationalError("database is locked"),
    ],
    ids=["lost_fence", "storage"],
)
async def test_a_retryable_recovery_failure_stops_the_run_and_keeps_every_derivation_staged(db, cause):
    harness = await synced(db)
    staged = await _staged_changes(harness)
    harness.engine = _WrappingLifecycleEngine(cause)

    state = await harness.sync()

    assert (state.last_sync_status, state.error_message) == ("failed", "projected lifecycle commit failed")
    statuses = {attempt.id: attempt.status for attempt in await db.list_source_derivation_attempts(source_id=SOURCE_ID)}
    # PAY-1's extraction completed before its commit failed; PAY-2 was not reached. Both stay staged.
    assert (statuses[staged["PAY-1"].id], statuses[staged["PAY-2"].id]) == ("completed", "retryable_failure")


@pytest.mark.asyncio
async def test_a_raw_save_failure_fails_the_document_before_its_revision_commits(db):
    harness = await synced(db)
    unit_id = await harness.unit_id("PAY-1")
    committed = await db.get_current_source_unit_revision(unit_id)
    before = await db.get_document(doc_id("PAY-1"))
    # A location-only change moves the Unit to a new revision without semantic work.
    harness.provider.urls["PAY-1"] = "https://jira.example/browse/PAY-1?moved"
    store_raw = harness.store.store_raw

    def unavailable(**kwargs):
        raise OSError("object store unavailable")

    harness.store.store_raw = unavailable
    failed = await harness.sync()

    assert failed.docs_failed == 1
    assert await db.get_current_source_unit_revision(unit_id) == committed
    assert (await db.get_document(doc_id("PAY-1"))).raw_content_uri == before.raw_content_uri

    harness.store.store_raw = store_raw
    assert (await harness.sync()).last_sync_status == "success"
    moved = await db.get_current_source_unit_revision(unit_id)
    assert moved.id != committed.id and moved.semantic_hash == committed.semantic_hash
    stored = await db.get_document(doc_id("PAY-1"))
    assert stored.source_url == harness.provider.urls["PAY-1"]
    assert harness.stored_raw(stored.raw_content_uri)["key"] == "PAY-1"


@pytest.mark.asyncio
async def test_the_migration_supersedes_unapplied_reprocess_derivations_staged_from_stored_input(tmp_path):
    path = tmp_path / "staged.db"
    legacy = Database(str(path))
    await legacy.connect()
    NoopMemoryEngine.db = legacy
    try:
        await legacy.upsert_source(
            id=SOURCE_ID, type="jira", name="Jira", config_json="{}", access_policy="workspace",
            owner_user_id="dev",
        )
        harness = Harness(legacy)
        assert (await harness.sync()).last_sync_status == "success"
        await _staged_changes(harness)
        assert (await harness.reprocess("PAY-1", extractor=FailingMemoryExtractor())).docs_failed == 1
        unapplied = await legacy.list_source_derivation_attempts(
            source_id=SOURCE_ID, statuses=("pending", "retryable_failure", "completed"),
        )
        reprocessed = {attempt.id: attempt for attempt in unapplied if attempt.context.support_without_baseline}
        ordinary = {attempt.id: attempt.status for attempt in unapplied if attempt.id not in reprocessed}
        assert reprocessed and ordinary
        await legacy.db.execute("DELETE FROM schema_migrations WHERE version = 103")
        await legacy.db.commit()
    finally:
        NoopMemoryEngine.db = None
        await legacy.close()

    migrated = Database(str(path))
    await migrated.connect()
    try:
        attempts = {attempt.id: attempt for attempt in await migrated.list_source_derivation_attempts(source_id=SOURCE_ID)}
        for attempt_id, before in reprocessed.items():
            after = attempts[attempt_id]
            assert (after.status, after.terminal_reason_code) == ("superseded", "DERIVATION_INPUT_SUPERSEDED")
            assert after.updated_at > before.updated_at
        assert {attempt_id: attempts[attempt_id].status for attempt_id in ordinary} == ordinary
    finally:
        await migrated.close()


def _jira_gene(handler):
    from memforge.genes.jira_gene import JiraGene

    gene = JiraGene(config={"base_url": "https://jira.example.test", "pat": "token"}, source_id=SOURCE_ID)
    gene._client = httpx.AsyncClient(base_url="https://jira.example.test", transport=httpx.MockTransport(handler))
    gene._request_limiter = None
    gene._auth_mode = "pat"
    gene._base_url = "https://jira.example.test"
    gene._hydrated_issues = {}
    return gene


@pytest.mark.asyncio
async def test_jira_rediscovers_one_issue_by_its_id_and_fetches_it_as_discovered():
    requests: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request.url.path)
        if request.url.path != f"/rest/api/2/issue/{ISSUES['PAY-1']}":
            return httpx.Response(httpx.codes.NOT_FOUND, json={"errorMessages": ["Issue does not exist"]})
        return httpx.Response(httpx.codes.OK, json={
            "id": ISSUES["PAY-1"], "key": "PAY-1",
            "fields": {
                "summary": "Payroll run", "description": "Body", "status": None, "priority": None,
                "assignee": None, "resolution": None, "updated": "2026-09-21T08:00:00.000+0000",
                "project": {"key": "PAY"}, "labels": [], "comment": {"startAt": 0, "total": 0, "comments": []},
            },
            "changelog": {"startAt": 0, "total": 1, "histories": [DOMAIN_HISTORY]},
        })

    gene = _jira_gene(handler)
    stored = ContentItem(
        item_id=doc_id("PAY-1"), title="PAY-1: Old summary", source_url="https://jira.example.test/browse/PAY-1",
        last_modified=SYNCED_AT, extra={"issue_id": ISSUES["PAY-1"], "issue_key": "PAY-1"},
    )
    missing = ContentItem(
        item_id=doc_id("PAY-2"), title="PAY-2", source_url="https://jira.example.test/browse/PAY-2",
        last_modified=SYNCED_AT, extra={"issue_id": ISSUES["PAY-2"], "issue_key": "PAY-2"},
    )
    try:
        current = await gene.rediscover(stored)
        raw = await gene.fetch(current)
        assert await gene.rediscover(missing) is None
    finally:
        await gene._client.aclose()

    assert (current.item_id, current.title, current.version) == (
        doc_id("PAY-1"), "PAY-1: Payroll run", "2026-09-21T08:00:00.000+0000",
    )
    assert [history["id"] for history in json.loads(raw.body)["changelog"]["histories"]] == ["9001"]
    # The fetch reads the issue rediscovery returned; the provider is asked once per Document.
    assert requests == [f"/rest/api/2/issue/{ISSUES['PAY-1']}", f"/rest/api/2/issue/{ISSUES['PAY-2']}"]


def _confluence_page(page_id: str, **fields) -> dict:
    return {
        "id": page_id, "title": "Payroll runbook", "status": "current", "space": {"key": "PAY"},
        "version": {"number": 4, "when": "2026-09-21T08:00:00.000Z", "by": {"displayName": "Ann"}},
        "metadata": {"labels": {"results": []}}, "_links": {"webui": f"/pages/{page_id}"},
        **fields,
    }


def _labelled(page_id: str, *labels: str) -> dict:
    return {"id": page_id, "status": "current", "metadata": {"labels": {"results": [{"name": name} for name in labels]}}}


async def _rediscover_confluence(config: dict, pages: dict[str, dict]) -> dict[str, ContentItem | None]:
    from memforge.genes.confluence_gene import ConfluenceGene

    expands: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        expands.append(request.url.params["expand"])
        page = pages.get(request.url.path.rsplit("/", 1)[-1])
        if page is None:
            return httpx.Response(httpx.codes.NOT_FOUND, json={"message": "No content found"})
        return httpx.Response(httpx.codes.OK, json=page)

    gene = ConfluenceGene(config={"base_url": "https://wiki.example.test", "pat": "token", **config}, source_id=SOURCE_ID)
    gene._client = httpx.AsyncClient(base_url="https://wiki.example.test", transport=httpx.MockTransport(handler))
    gene._request_limiter = None
    gene._api_prefix = ""
    gene._base_url = "https://wiki.example.test"
    try:
        found = {}
        for page_id in (*pages, "404"):
            stored = ContentItem(
                item_id=f"confluence-{page_id}", title="Payroll runbook", source_url="https://wiki.example.test",
                last_modified=SYNCED_AT, version="3", extra={"page_id": page_id, "space_key": "PAY"},
            )
            found[page_id] = await gene.rediscover(stored)
    finally:
        await gene._client.aclose()
    assert all(expand.startswith("version,metadata.labels,space") for expand in expands)
    return found


@pytest.mark.asyncio
async def test_confluence_rediscovers_the_current_pages_of_its_spaces():
    found = await _rediscover_confluence(
        {"spaces": ["PAY"], "exclude_labels": ["draft"]},
        {
            "42": _confluence_page("42"),
            "43": _confluence_page("43", status="archived"),
            "44": _confluence_page("44", space={"key": "HR"}),
            "45": _confluence_page("45", metadata={"labels": {"results": [{"name": "draft"}]}}),
        },
    )

    assert (found["42"].item_id, found["42"].version, found["42"].space_or_project) == ("confluence-42", "4", "PAY")
    assert {page_id for page_id, item in found.items() if item is None} == {"43", "44", "45", "404"}


@pytest.mark.asyncio
async def test_confluence_rediscovers_the_pages_its_tree_discovery_reaches():
    root = "7"
    found = await _rediscover_confluence(
        {"page_tree_root": root, "exclude_labels": ["draft"]},
        {
            root: _confluence_page(root, status="archived"),
            "42": _confluence_page("42", ancestors=[_labelled("1"), _labelled(root), _labelled("8")]),
            "43": _confluence_page("43", ancestors=[_labelled("1")]),
            "44": _confluence_page("44", ancestors=[_labelled(root), _labelled("9", "draft")]),
            "45": _confluence_page("45", ancestors=[_labelled(root), {**_labelled("10"), "status": "archived"}]),
            "46": _confluence_page("46", status="trashed", ancestors=[_labelled(root)]),
        },
    )

    # The root is listed as discovery fetches it; the other pages only when discovery reaches them.
    assert {page_id for page_id, item in found.items() if item is not None} == {root, "42"}
    assert found["42"].space_or_project == "PAY"


def test_only_sources_with_a_provider_to_ask_rediscover_their_documents():
    from memforge.genes import source_rediscovers_documents

    assert source_rediscovers_documents("jira", {})
    assert not source_rediscovers_documents("jira", {"sync_mode": "local_agent"})
    assert source_rediscovers_documents("confluence", {})
    for source_type in ("github_repo", "github_pages", "teams", "agent_session", "local_markdown"):
        assert not source_rediscovers_documents(source_type, {})
