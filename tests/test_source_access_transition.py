from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.models import DocumentRecord, Memory, content_hash
from memforge.source_access import source_is_discoverable
from memforge.source_access_transition import SourceAccessTransitionService
from memforge.storage.adapters.context import AccessScope
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


class RecordingMemoryStore:
    def __init__(
        self,
        *,
        fail_once: bool = False,
        fail_on_calls: set[int] | None = None,
    ) -> None:
        self.fail_once = fail_once
        self.fail_on_calls = set(fail_on_calls or ())
        self.call_count = 0
        self.reindexed: list[str] = []

    async def reindex_memory_access(self, memory_id: str) -> None:
        self.call_count += 1
        self.reindexed.append(memory_id)
        if self.fail_once or self.call_count in self.fail_on_calls:
            self.fail_once = False
            raise RuntimeError("synthetic vector failure")


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "source-access-transition.db"))
    await database.connect()
    yield database
    await database.close()


async def _source(
    db: Database,
    source_id: str,
    *,
    policy: str,
    owner: str = "alice",
) -> None:
    await db.upsert_source(
        source_id,
        "confluence",
        source_id,
        "{}",
        policy,
        owner,
        created_by_user_id=owner,
    )


async def _document(db: Database, source_id: str, doc_id: str) -> None:
    now = datetime.now(timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source=source_id,
            source_url=f"https://example.test/{doc_id}",
            title=doc_id,
            space_or_project="TEST",
            author=None,
            last_modified=now,
            labels=[],
            version="1",
            content_hash=content_hash(doc_id),
            token_count=1,
            raw_content_uri=None,
            raw_content_type="text/plain",
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )


async def _memory(
    db: Database,
    memory_id: str,
    *,
    visibility: str,
    owner_user_id: str | None,
) -> None:
    await db.insert_memory(
        Memory(
            id=memory_id,
            memory_type="fact",
            content=f"memory {memory_id}",
            content_hash=content_hash(memory_id),
            visibility=visibility,
            owner_user_id=owner_user_id,
            project_key="TEST",
        )
    )


@pytest.mark.asyncio
async def test_transition_is_fail_closed_then_activates_all_history(db: Database):
    await _source(db, "src-private", policy="private")
    await _document(db, "src-private", "doc-private")
    await _memory(db, "mem-private", visibility="private", owner_user_id="alice")
    await db.add_memory_source("mem-private", "doc-private", "confluence", source_updated_at=None)
    store = RecordingMemoryStore()
    service = SourceAccessTransitionService(db=db, memory_store=store)

    transition = await service.start(
        source_id="src-private",
        actor_user_id="alice",
        target_policy="workspace",
        idempotency_key="share-once",
    )

    changing = await db.get_source("src-private")
    assert changing["access_state"] == "changing"
    assert changing["status"] == "paused"
    assert source_is_discoverable(changing, viewer_id="alice") is True
    assert source_is_discoverable(changing, viewer_id="bob") is False

    completed = await service.run(transition["operation_id"])

    source = await db.get_source("src-private")
    memory = await db.get_memory("mem-private")
    assert completed["status"] == "completed"
    assert source["access_policy"] == "workspace"
    assert source["access_state"] == "active"
    assert source["status"] == "active"
    assert memory.visibility == "workspace"
    assert memory.owner_user_id is None
    assert store.reindexed == ["mem-private"]


@pytest.mark.asyncio
async def test_transition_splits_only_the_changing_sources_support(db: Database):
    await _source(db, "src-a", policy="workspace")
    await _source(db, "src-b", policy="workspace", owner="bob")
    await _document(db, "src-a", "doc-a")
    await _document(db, "src-b", "doc-b")
    await _memory(db, "mem-shared", visibility="workspace", owner_user_id=None)
    await db.add_memory_source("mem-shared", "doc-a", "confluence", source_updated_at=None)
    await db.add_memory_source("mem-shared", "doc-b", "confluence", source_updated_at=None)
    await db.db.execute(
        """INSERT INTO evidence_units (
               id, source_id, doc_id, source_type, source_metadata_json,
               project_key, visibility, content, evidence_provenance
           ) VALUES ('eu-a', 'src-a', 'doc-a', 'confluence', '{}',
                     'TEST', 'workspace', 'evidence a', 'source_content')"""
    )
    await db.db.execute(
        """INSERT INTO evidence_relations (
               evidence_unit_id, memory_id, relation_type, relation_direction, authority_case,
               is_authoritative_support, classifier_version, relation_run_id
           ) VALUES ('eu-a', 'mem-shared', 'refines', 'challenger_to_candidate',
                     'same_source_lineage', 1, 'test-v1', 'run-a')"""
    )
    await db.db.commit()
    service = SourceAccessTransitionService(db=db, memory_store=RecordingMemoryStore())

    transition = await service.start(
        source_id="src-a",
        actor_user_id="alice",
        target_policy="private",
        idempotency_key="make-private",
    )
    await service.run(transition["operation_id"])

    old_support = await db.get_memory_sources("mem-shared")
    async with db.db.execute("SELECT DISTINCT memory_id FROM memory_sources WHERE source_id = 'src-a'") as cursor:
        moved_id = str((await cursor.fetchone())["memory_id"])
    moved = await db.get_memory(moved_id)
    assert moved_id != "mem-shared"
    assert {support.source_id for support in old_support} == {"src-b"}
    assert moved.visibility == "private"
    assert moved.owner_user_id == "alice"
    assert {support.source_id for support in await db.get_memory_sources(moved_id)} == {"src-a"}
    async with db.db.execute(
        "SELECT memory_id, relation_direction, classifier_version, relation_run_id "
        "FROM evidence_relations WHERE evidence_unit_id = 'eu-a'"
    ) as cursor:
        evidence_relation = await cursor.fetchone()
    assert evidence_relation["memory_id"] == moved_id
    assert evidence_relation["relation_direction"] == "challenger_to_candidate"
    assert evidence_relation["classifier_version"] == "test-v1"
    assert evidence_relation["relation_run_id"] == "run-a"


@pytest.mark.asyncio
async def test_failed_projection_stays_owner_only_and_retry_converges(db: Database):
    await _source(db, "src-retry", policy="private")
    await _document(db, "src-retry", "doc-retry")
    await _memory(db, "mem-retry", visibility="private", owner_user_id="alice")
    await db.add_memory_source("mem-retry", "doc-retry", "confluence", source_updated_at=None)
    store = RecordingMemoryStore(fail_once=True)
    service = SourceAccessTransitionService(db=db, memory_store=store)
    transition = await service.start(
        source_id="src-retry",
        actor_user_id="alice",
        target_policy="workspace",
        idempotency_key="retryable",
    )

    with pytest.raises(RuntimeError, match="synthetic vector failure"):
        await service.run(transition["operation_id"])

    failed = await db.get_source_access_transition(transition["operation_id"])
    source = await db.get_source("src-retry")
    assert failed["status"] == "failed"
    assert source["access_state"] == "changing"
    assert source_is_discoverable(source, viewer_id="bob") is False
    relational = build_sqlite_adapters(db, memory_collection=None).relational
    bob_visible = await relational.filter_visible_ids(
        ["mem-retry"],
        AccessScope(
            user_id="bob",
            include_private=False,
            allowed_statuses=("active",),
            active_project=None,
            scope_mode="project-first",
        ),
    )
    alice_visible = await relational.filter_visible_ids(
        ["mem-retry"],
        AccessScope(
            user_id="alice",
            include_private=True,
            allowed_statuses=("active",),
            active_project=None,
            scope_mode="project-first",
        ),
    )
    assert bob_visible == set()
    assert alice_visible == {"mem-retry"}

    completed = await service.retry(transition["operation_id"])
    assert completed["status"] == "completed"
    assert completed["processed_memories"] == completed["total_memories"] == 1
    assert (await db.get_source("src-retry"))["access_policy"] == "workspace"
    assert store.reindexed == ["mem-retry", "mem-retry"]


@pytest.mark.asyncio
async def test_failed_transition_can_restore_the_complete_previous_policy(db: Database):
    await _source(db, "src-revert", policy="private")
    await _document(db, "src-revert", "doc-revert")
    await _memory(db, "mem-revert", visibility="private", owner_user_id="alice")
    await db.add_memory_source("mem-revert", "doc-revert", "confluence", source_updated_at=None)
    store = RecordingMemoryStore(fail_once=True)
    service = SourceAccessTransitionService(db=db, memory_store=store)
    transition = await service.start(
        source_id="src-revert",
        actor_user_id="alice",
        target_policy="workspace",
        idempotency_key="revertable",
    )
    with pytest.raises(RuntimeError):
        await service.run(transition["operation_id"])

    reverted = await service.revert(transition["operation_id"])

    source = await db.get_source("src-revert")
    memory = await db.get_memory("mem-revert")
    assert reverted["status"] == "reverted"
    assert source["access_policy"] == "private"
    assert source["access_state"] == "active"
    assert source["status"] == "active"
    assert memory.visibility == "private"
    assert memory.owner_user_id == "alice"


@pytest.mark.asyncio
async def test_failed_split_revert_merges_source_support_without_duplicate(db: Database):
    await _source(db, "src-a", policy="workspace")
    await _source(db, "src-b", policy="workspace", owner="bob")
    await _document(db, "src-a", "doc-a")
    await _document(db, "src-b", "doc-b")
    await _memory(db, "mem-shared", visibility="workspace", owner_user_id=None)
    await db.add_memory_source("mem-shared", "doc-a", "confluence", source_updated_at=None)
    await db.add_memory_source("mem-shared", "doc-b", "confluence", source_updated_at=None)
    await db.db.execute(
        """INSERT INTO evidence_units (
               id, source_id, doc_id, source_type, source_metadata_json,
               project_key, visibility, content, evidence_provenance
           ) VALUES ('eu-a', 'src-a', 'doc-a', 'confluence', '{}',
                     'TEST', 'workspace', 'evidence a', 'source_content')"""
    )
    await db.db.execute(
        """INSERT INTO evidence_relations (
               evidence_unit_id, memory_id, relation_type, relation_direction, authority_case,
               is_authoritative_support, classifier_version, relation_run_id
           ) VALUES ('eu-a', 'mem-shared', 'refines', 'challenger_to_candidate',
                     'same_source_lineage', 1, 'test-v1', 'run-a')"""
    )
    await db.db.commit()
    store = RecordingMemoryStore(fail_once=True)
    service = SourceAccessTransitionService(db=db, memory_store=store)
    transition = await service.start(
        source_id="src-a",
        actor_user_id="alice",
        target_policy="private",
        idempotency_key="split-then-revert",
    )

    with pytest.raises(RuntimeError, match="synthetic vector failure"):
        await service.run(transition["operation_id"])

    async with db.db.execute(
        "SELECT target_memory_id FROM source_access_transition_memory_map WHERE operation_id = ?",
        (transition["operation_id"],),
    ) as cursor:
        split_memory_id = str((await cursor.fetchone())["target_memory_id"])
    reverted = await service.revert(transition["operation_id"])

    assert reverted["status"] == "reverted"
    assert {support.source_id for support in await db.get_memory_sources("mem-shared")} == {
        "src-a",
        "src-b",
    }
    assert await db.get_memory_sources(split_memory_id) == []
    split_memory = await db.get_memory(split_memory_id)
    assert split_memory is not None and split_memory.status == "retired"
    original = await db.get_memory("mem-shared")
    assert original is not None and original.status == "active"
    assert original.visibility == "workspace"
    async with db.db.execute(
        "SELECT memory_id, relation_direction FROM evidence_relations WHERE evidence_unit_id = 'eu-a'"
    ) as cursor:
        restored_relation = await cursor.fetchone()
    assert restored_relation["memory_id"] == "mem-shared"
    assert restored_relation["relation_direction"] == "challenger_to_candidate"


@pytest.mark.asyncio
async def test_failed_split_retry_reindexes_the_stable_original_and_split_pair(db: Database):
    await _source(db, "src-a", policy="workspace")
    await _source(db, "src-b", policy="workspace", owner="bob")
    await _document(db, "src-a", "doc-a")
    await _document(db, "src-b", "doc-b")
    await _memory(db, "mem-shared", visibility="workspace", owner_user_id=None)
    await db.add_memory_source("mem-shared", "doc-a", "confluence", source_updated_at=None)
    await db.add_memory_source("mem-shared", "doc-b", "confluence", source_updated_at=None)
    store = RecordingMemoryStore(fail_once=True)
    service = SourceAccessTransitionService(db=db, memory_store=store)
    transition = await service.start(
        source_id="src-a",
        actor_user_id="alice",
        target_policy="private",
        idempotency_key="split-then-retry",
    )

    with pytest.raises(RuntimeError, match="synthetic vector failure"):
        await service.run(transition["operation_id"])
    completed = await service.retry(transition["operation_id"])

    async with db.db.execute(
        "SELECT target_memory_id FROM source_access_transition_memory_map WHERE operation_id = ?",
        (transition["operation_id"],),
    ) as cursor:
        split_memory_id = str((await cursor.fetchone())["target_memory_id"])
    assert completed["status"] == "completed"
    assert completed["processed_memories"] == completed["total_memories"] == 2
    assert store.reindexed == ["mem-shared", "mem-shared", split_memory_id]


@pytest.mark.asyncio
async def test_failed_split_revert_retry_keeps_the_same_reindex_pair(db: Database):
    await _source(db, "src-a", policy="workspace")
    await _source(db, "src-b", policy="workspace", owner="bob")
    await _document(db, "src-a", "doc-a")
    await _document(db, "src-b", "doc-b")
    await _memory(db, "mem-shared", visibility="workspace", owner_user_id=None)
    await db.add_memory_source("mem-shared", "doc-a", "confluence", source_updated_at=None)
    await db.add_memory_source("mem-shared", "doc-b", "confluence", source_updated_at=None)
    store = RecordingMemoryStore(fail_on_calls={1, 2})
    service = SourceAccessTransitionService(db=db, memory_store=store)
    transition = await service.start(
        source_id="src-a",
        actor_user_id="alice",
        target_policy="private",
        idempotency_key="split-revert-retry",
    )

    with pytest.raises(RuntimeError, match="synthetic vector failure"):
        await service.run(transition["operation_id"])
    with pytest.raises(RuntimeError, match="synthetic vector failure"):
        await service.revert(transition["operation_id"])
    reverted = await service.revert(transition["operation_id"])

    async with db.db.execute(
        "SELECT target_memory_id FROM source_access_transition_memory_map WHERE operation_id = ?",
        (transition["operation_id"],),
    ) as cursor:
        split_memory_id = str((await cursor.fetchone())["target_memory_id"])
    assert reverted["status"] == "reverted"
    assert reverted["processed_memories"] == reverted["total_memories"] == 2
    assert store.reindexed == [
        "mem-shared",
        "mem-shared",
        "mem-shared",
        split_memory_id,
    ]
