"""Reading, ordering and dismissing Cross-Document Relations (ADR 0037)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.memory.cross_document_relation import (
    MAX_RELATIONS_PER_MEMORY,
    CrossDocumentRelationDecider,
    CrossDocumentRelationLabel,
    CurrentCrossDocumentRelation,
    RelationDismissalConflict,
    pair_key,
    recorded_relation_label,
)
from memforge.memory.cross_document_relation_reader import (
    RelationGraph,
    RelationRole,
    dismiss_relation,
    list_relation_pairs,
    load_relation_graph,
    order_by_relations,
    read_memory_relations,
    relation_notice,
    relation_role,
)
from memforge.models import DocumentRecord, Memory, MemoryRelationContext, MemorySourceRef, RelatedMemory, content_hash
from memforge.retrieval.search import SearchEngine
from memforge.storage.adapters.context import LOCAL_DEV_USER_ID, AccessScope
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.adapters.sqlite.relational import SqliteRelationalStore
from memforge.storage.database import Database

OTHER_USER_ID = "someone-else"
EARLIER = "2026-07-01"
LATER = "2026-08-01"
CONTRADICTS = CrossDocumentRelationLabel.CONTRADICTS
UPDATES = CrossDocumentRelationLabel.UPDATES
EQUIVALENT = CrossDocumentRelationLabel.EQUIVALENT


def _scope(user_id: str = LOCAL_DEV_USER_ID, *, include_private: bool = True) -> AccessScope:
    return AccessScope(
        user_id=user_id,
        include_private=include_private,
        allowed_statuses=("active",),
        active_project=None,
        scope_mode="workspace",
    )


def _memory(memory_id: str, content: str | None = None, **overrides) -> Memory:
    now = datetime.now(timezone.utc)
    text = content or f"statement of {memory_id}"
    return replace(
        Memory(
            id=memory_id,
            memory_type="fact",
            content=text,
            content_hash=content_hash(text),
            confidence=0.9,
            created_at=now,
            updated_at=now,
        ),
        **overrides,
    )


async def _relate(
    db: Database,
    first: Memory,
    second: Memory,
    label: CrossDocumentRelationLabel,
    *,
    times: Mapping[str, str] | None = None,
    decided_at: str = EARLIER,
    reason: str = "Both statements govern the same case.",
) -> None:
    """Store a relation with the Evidence time of each Memory in ``times``."""

    by_id = {first.id: first, second.id: second}
    low_id, high_id = pair_key(first.id, second.id)
    evidence_times = times or {}
    await db.db.execute(
        """INSERT INTO cross_document_relations (
               memory_low_id, memory_high_id, label, low_content_hash, high_content_hash,
               low_evidence_time, high_evidence_time,
               reason, classifier_version, relation_run_id, discovery_work_id, decided_by, decided_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?, ?)""",
        (
            low_id,
            high_id,
            label.value,
            by_id[low_id].content_hash,
            by_id[high_id].content_hash,
            evidence_times.get(low_id),
            evidence_times.get(high_id),
            reason,
            CrossDocumentRelationDecider.CLASSIFIER.value,
            decided_at,
        ),
    )
    await db.db.commit()


async def _change_content(db: Database, memory: Memory, content: str) -> Memory:
    changed = replace(memory, content=content, content_hash=content_hash(content))
    await db.db.execute(
        "UPDATE memories SET content = ?, content_hash = ? WHERE id = ?",
        (changed.content, changed.content_hash, memory.id),
    )
    await db.db.commit()
    return changed


async def _document(db: Database, doc_id: str, source_id: str) -> None:
    now = datetime.now(timezone.utc)
    await db.upsert_document(
        DocumentRecord(
            doc_id=doc_id,
            source=source_id,
            source_url=f"https://example/{doc_id}",
            title=f"Title {doc_id}",
            space_or_project="PAY",
            author="A",
            last_modified=now,
            labels=[],
            version="1",
            content_hash=f"hash-{doc_id}",
            token_count=10,
            raw_content_uri=None,
            raw_content_type="text/html",
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=now,
        )
    )


async def _attach_source(db: Database, memory_id: str, source_id: str, source_type: str) -> None:
    doc_id = f"doc-{memory_id}-{source_id}"
    await _document(db, doc_id, source_id)
    await db.db.execute(
        """INSERT INTO memory_sources (memory_id, doc_id, source_id, source_type)
           VALUES (?, ?, ?, ?)""",
        (memory_id, doc_id, source_id, source_type),
    )
    await db.db.commit()


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "relations.db"))
    await database.connect()
    await database.upsert_source("src-wiki", "confluence", "Payroll wiki", "{}", "workspace", LOCAL_DEV_USER_ID)
    await database.upsert_source("src-jira", "jira", "Payroll Jira", "{}", "workspace", LOCAL_DEV_USER_ID)
    await database.upsert_source("src-private", "teams", "Private chat", "{}", "private", OTHER_USER_ID)
    yield database
    await database.close()


async def _pair(db: Database, label=CONTRADICTS) -> tuple[Memory, Memory]:
    """Two related Memories; the first one's Evidence is the later one."""

    first = _memory("mem-rel-a", "Payroll closes on the 20th.")
    second = _memory("mem-rel-b", "Payroll closes on the 25th.")
    await db.insert_memory(first)
    await db.insert_memory(second)
    await _relate(db, first, second, label, times={first.id: LATER, second.id: EARLIER})
    return first, second


# ---------------------------------------------------------------- storage


@pytest.mark.asyncio
async def test_relation_is_current_from_both_memories(db):
    first, second = await _pair(db)

    relations = await SqliteRelationalStore(db).list_cross_document_relations((first.id, second.id), _scope())

    assert [item.counterpart_of(first.id) for item in relations[first.id]] == [second.id]
    assert [item.counterpart_of(second.id) for item in relations[second.id]] == [first.id]
    assert relations[first.id][0].label is CONTRADICTS


@pytest.mark.asyncio
async def test_relation_needs_both_memories_visible_to_the_caller(db):
    visible = _memory("mem-rel-visible")
    private = _memory("mem-rel-private", visibility="private", owner_user_id=OTHER_USER_ID)
    await db.insert_memory(visible)
    await db.insert_memory(private)
    await _relate(db, visible, private, CONTRADICTS)

    as_caller = await db.list_cross_document_relations((visible.id,), _scope())
    as_owner = await db.list_cross_document_relations((visible.id,), _scope(OTHER_USER_ID))

    assert as_caller == {visible.id: ()}
    assert len(as_owner[visible.id]) == 1


@pytest.mark.asyncio
async def test_relation_stops_being_current_when_either_memory_changes(db):
    first, second = await _pair(db)

    await _change_content(db, second, "Payroll closes on the 26th.")
    after_content_change = await db.list_cross_document_relations((first.id,), _scope())
    await _change_content(db, second, second.content)
    await db.db.execute("UPDATE memories SET status = 'retired' WHERE id = ?", (first.id,))
    after_retirement = await db.list_cross_document_relations((second.id,), _scope())

    assert after_content_change == {first.id: ()}
    assert after_retirement == {second.id: ()}


@pytest.mark.asyncio
async def test_relations_per_memory_are_bounded_most_consequential_first(db):
    center = _memory("mem-center")
    await db.insert_memory(center)
    for index in range(MAX_RELATIONS_PER_MEMORY):
        other = _memory(f"mem-eq-{index:02d}")
        await db.insert_memory(other)
        await _relate(db, center, other, EQUIVALENT, decided_at=LATER)
    conflicting = _memory("mem-zz-conflict")
    await db.insert_memory(conflicting)
    await _relate(db, center, conflicting, CONTRADICTS)

    [*relations] = (await db.list_cross_document_relations((center.id,), _scope()))[center.id]

    assert len(relations) == MAX_RELATIONS_PER_MEMORY
    assert relations[0].label is CONTRADICTS


@pytest.mark.asyncio
async def test_relation_view_filters_by_label_and_pages(db):
    center = _memory("mem-view-center")
    await db.insert_memory(center)
    others = []
    for index, label in enumerate((CONTRADICTS, UPDATES, CONTRADICTS, EQUIVALENT)):
        other = _memory(f"mem-view-{index}")
        others.append(other)
        await db.insert_memory(other)
        await _relate(
            db,
            center,
            other,
            label,
            times={center.id: EARLIER, other.id: LATER},
            decided_at=f"2026-07-0{index + 1}T00:00:00+00:00",
        )

    conflicts, conflict_total = await list_relation_pairs(db, _scope(), label=CONTRADICTS, limit=1, offset=0)
    next_conflicts, _ = await list_relation_pairs(db, _scope(), label=CONTRADICTS, limit=1, offset=1)
    updates, update_total = await list_relation_pairs(db, _scope(), label=UPDATES, limit=10, offset=0)
    _, total = await list_relation_pairs(db, _scope(), label=None, limit=10, offset=0)

    assert (conflict_total, update_total, total) == (2, 1, 4)
    assert [view.memories[0].memory_id for view in (*conflicts, *next_conflicts)] == ["mem-view-2", "mem-view-0"]
    assert {view.label for view in (*conflicts, *next_conflicts)} == {CONTRADICTS.value}
    [update] = updates
    assert (update.label, update.newer_memory_id) == (UPDATES.value, others[1].id)
    assert [(memory.memory_id, memory.evidence_time) for memory in update.memories] == [
        (others[1].id, LATER),
        (center.id, EARLIER),
    ]


@pytest.mark.asyncio
async def test_source_refs_name_only_sources_the_caller_may_read(db):
    memory = _memory("mem-sourced")
    await db.insert_memory(memory)
    await _attach_source(db, memory.id, "src-wiki", "confluence")
    await _attach_source(db, memory.id, "src-private", "teams")
    await _attach_source(db, memory.id, "src-jira", "jira")
    await db.set_source_subscription("src-jira", LOCAL_DEV_USER_ID, False)

    refs = await SqliteRelationalStore(db).get_memory_source_refs_many((memory.id,), _scope())

    assert refs == {memory.id: (MemorySourceRef("src-wiki", "confluence", "Payroll wiki"),)}


@pytest.mark.asyncio
async def test_dismissal_hides_the_relation_until_either_memory_changes(db):
    first, second = await _pair(db)

    dismissal = await db.record_cross_document_relation_dismissal(
        memory_id=first.id,
        counterpart_memory_id=second.id,
        label=CONTRADICTS,
        expected_content_hash=first.content_hash,
        counterpart_expected_content_hash=second.content_hash,
        actor="reader-1",
        note="Different payroll areas.",
    )
    again = await db.record_cross_document_relation_dismissal(
        memory_id=second.id,
        counterpart_memory_id=first.id,
        label=CONTRADICTS,
        expected_content_hash=second.content_hash,
        counterpart_expected_content_hash=first.content_hash,
        actor="reader-2",
    )
    hidden = await db.list_cross_document_relations((first.id,), _scope())
    in_force = await db.list_active_cross_document_relation_dismissals(first.id, _scope(), limit=10)
    changed = await _change_content(db, second, "Payroll closes on the 26th.")
    await _relate_replacing(db, first, changed, CONTRADICTS)
    shown_again = await db.list_cross_document_relations((first.id,), _scope())

    assert again == dismissal
    assert hidden == {first.id: ()}
    assert [item.id for item in in_force] == [dismissal.id]
    assert len(shown_again[first.id]) == 1
    assert await db.list_active_cross_document_relation_dismissals(first.id, _scope(), limit=10) == ()
    [event] = await db.list_memory_audit_events(event_type="cross_document_relation_dismissed")
    assert event.actor_id == "reader-1"
    assert event.payload["dismissal_id"] == dismissal.id
    assert event.payload["label"] == CONTRADICTS.value


async def _relate_replacing(db: Database, first: Memory, second: Memory, label) -> None:
    low_id, high_id = pair_key(first.id, second.id)
    await db.db.execute(
        "DELETE FROM cross_document_relations WHERE memory_low_id = ? AND memory_high_id = ?",
        (low_id, high_id),
    )
    await _relate(db, first, second, label, times={first.id: LATER, second.id: EARLIER})


@pytest.mark.asyncio
async def test_dismissal_of_one_label_does_not_hide_another(db):
    first, second = await _pair(db)
    await db.record_cross_document_relation_dismissal(
        memory_id=first.id,
        counterpart_memory_id=second.id,
        label=CONTRADICTS,
        expected_content_hash=first.content_hash,
        counterpart_expected_content_hash=second.content_hash,
        actor="reader-1",
    )
    await _relate_replacing(db, first, second, UPDATES)

    relations = await db.list_cross_document_relations((first.id,), _scope())

    assert [item.label for item in relations[first.id]] == [UPDATES]


@pytest.mark.asyncio
async def test_dismissal_rejects_content_the_caller_did_not_see(db):
    first, second = await _pair(db)

    with pytest.raises(RelationDismissalConflict):
        await db.record_cross_document_relation_dismissal(
            memory_id=first.id,
            counterpart_memory_id=second.id,
            label=CONTRADICTS,
            expected_content_hash=first.content_hash,
            counterpart_expected_content_hash="stale-hash",
            actor="reader-1",
        )
    assert await db.db.execute_fetchall("SELECT 1 FROM cross_document_relation_dismissals") == []


@pytest.mark.asyncio
async def test_restore_undoes_the_dismissal_and_is_audited(db):
    first, second = await _pair(db)
    dismissal = await db.record_cross_document_relation_dismissal(
        memory_id=first.id,
        counterpart_memory_id=second.id,
        label=CONTRADICTS,
        expected_content_hash=first.content_hash,
        counterpart_expected_content_hash=second.content_hash,
        actor="reader-1",
    )

    restored = await db.restore_cross_document_relation_dismissals(
        memory_id=second.id,
        counterpart_memory_id=first.id,
        actor="reader-2",
    )
    restored_again = await db.restore_cross_document_relation_dismissals(
        memory_id=second.id,
        counterpart_memory_id=first.id,
        actor="reader-2",
    )

    assert [item.id for item in restored] == [dismissal.id]
    assert restored_again == ()
    assert len((await db.list_cross_document_relations((first.id,), _scope()))[first.id]) == 1
    [event] = await db.list_memory_audit_events(event_type="cross_document_relation_restored")
    assert event.actor_id == "reader-2"
    assert event.payload["dismissal_id"] == dismissal.id


# ---------------------------------------------------------------- reader


def _stored(
    first: str,
    second: str,
    label: CrossDocumentRelationLabel,
    times: Mapping[str, str] | None = None,
) -> CurrentCrossDocumentRelation:
    low_id, high_id = pair_key(first, second)
    evidence_times = times or {}
    return CurrentCrossDocumentRelation(
        memory_low_id=low_id,
        memory_high_id=high_id,
        label=label,
        low_content_hash="low",
        high_content_hash="high",
        low_evidence_time=evidence_times.get(low_id),
        high_evidence_time=evidence_times.get(high_id),
        reason="",
        decided_by=CrossDocumentRelationDecider.CLASSIFIER,
        decided_at=EARLIER,
    )


def _graph(relations: list[CurrentCrossDocumentRelation]) -> RelationGraph:
    by_memory: dict[str, list] = {}
    for relation in relations:
        for memory_id in (relation.memory_low_id, relation.memory_high_id):
            by_memory.setdefault(memory_id, []).append(relation)
    return RelationGraph(relations={key: tuple(value) for key, value in by_memory.items()})


def test_updates_is_ordered_by_evidence_time_and_otherwise_recorded_as_contradicts():
    ordered = _stored("mem-a", "mem-b", UPDATES, {"mem-a": LATER, "mem-b": EARLIER})

    assert ordered.newer_memory_id == "mem-a"
    assert relation_role(ordered, "mem-b") is RelationRole.OLDER
    assert _stored("mem-a", "mem-b", CONTRADICTS, {"mem-a": LATER, "mem-b": EARLIER}).newer_memory_id is None
    for low_time, high_time in ((EARLIER, EARLIER), (LATER, None), ("Tue, 03 Jun 2025", EARLIER)):
        assert recorded_relation_label(UPDATES, "mem-a", "mem-b", low_time, high_time) is CONTRADICTS
        with pytest.raises(ValueError, match="order the pair"):
            _stored("mem-a", "mem-b", UPDATES, {"mem-a": low_time, "mem-b": high_time})
    assert recorded_relation_label(EQUIVALENT, "mem-a", "mem-b", None, None) is EQUIVALENT


def test_window_order_puts_each_memory_after_every_newer_one_in_an_updates_chain():
    times = {"mem-1": "2026-06-01", "mem-2": EARLIER, "mem-3": LATER}
    graph = _graph(
        [
            _stored("mem-1", "mem-2", UPDATES, times),
            _stored("mem-2", "mem-3", UPDATES, times),
            _stored("mem-1", "mem-3", UPDATES, times),
        ]
    )

    assert order_by_relations(["mem-1", "mem-0", "mem-2", "mem-3"], graph) == ["mem-3", "mem-2", "mem-1", "mem-0"]


def test_window_order_ends_when_relations_decided_at_different_times_form_a_cycle():
    graph = _graph(
        [
            _stored("mem-1", "mem-2", UPDATES, {"mem-1": EARLIER, "mem-2": LATER}),
            _stored("mem-2", "mem-3", UPDATES, {"mem-2": EARLIER, "mem-3": LATER}),
            _stored("mem-1", "mem-3", UPDATES, {"mem-3": EARLIER, "mem-1": LATER}),
        ]
    )

    ordered = order_by_relations(["mem-1", "mem-2", "mem-3"], graph)

    assert sorted(ordered) == ["mem-1", "mem-2", "mem-3"]


def test_window_order_drops_equivalents_of_kept_memories_and_puts_newer_first():
    graph = _graph(
        [
            _stored("mem-1", "mem-2", EQUIVALENT),
            _stored("mem-3", "mem-4", UPDATES, {"mem-3": EARLIER, "mem-4": LATER}),
            _stored("mem-2", "mem-5", EQUIVALENT),
        ]
    )

    assert order_by_relations(["mem-1", "mem-2", "mem-3", "mem-4", "mem-5"], graph) == [
        "mem-1",
        "mem-4",
        "mem-3",
        "mem-5",
    ]


def test_notice_names_conflicts_and_newer_memories_only():
    counterpart = RelatedMemory(
        memory_id="mem-new",
        summary="Payroll closes on the 25th.",
        content_hash="hash",
        sources=(MemorySourceRef("src-jira", "jira", "Payroll Jira"),),
        evidence_time=LATER,
    )

    def context(label: str, role: str) -> MemoryRelationContext:
        return MemoryRelationContext(label=label, role=role, counterpart=counterpart, reason="", decided_by="classifier")

    assert relation_notice([context("equivalent", "peer"), context("updates", "newer")]) is None
    assert relation_notice([context("updates", "older")]) == (
        "Updated by the newer Memory mem-new from Payroll Jira (recorded 2026-08-01)."
    )
    assert relation_notice([context("contradicts", "peer")]) == (
        "Conflicts with Memory mem-new from Payroll Jira (recorded 2026-08-01)."
    )


@pytest.mark.asyncio
async def test_updates_reads_in_the_order_and_dates_it_was_decided_on(db):
    first, second = await _pair(db, UPDATES)

    graph = await load_relation_graph(SqliteRelationalStore(db), (first.id,), _scope())
    contexts = await read_memory_relations(db, (second.id,), _scope())

    assert graph.relations_of(first.id)[0].newer_memory_id == first.id
    [context] = contexts[second.id]
    assert (context.label, context.role) == (UPDATES.value, RelationRole.OLDER.value)
    assert context.counterpart.evidence_time == LATER


@pytest.mark.asyncio
async def test_dismissal_names_the_label_the_reader_sees(db):
    first, second = await _pair(db, UPDATES)

    with pytest.raises(RelationDismissalConflict):
        await dismiss_relation(
            db,
            memory_id=first.id,
            counterpart_memory_id=second.id,
            label=CONTRADICTS,
            expected_content_hash=first.content_hash,
            counterpart_expected_content_hash=second.content_hash,
            actor="reader-1",
            scope=_scope(),
        )
    dismissal = await dismiss_relation(
        db,
        memory_id=first.id,
        counterpart_memory_id=second.id,
        label=UPDATES,
        expected_content_hash=first.content_hash,
        counterpart_expected_content_hash=second.content_hash,
        actor="reader-1",
        scope=_scope(),
    )

    assert dismissal.label is UPDATES
    assert await db.list_cross_document_relations((first.id,), _scope()) == {first.id: ()}


# ---------------------------------------------------------------- search


class _VectorHits:
    def __init__(self, ids: list[str]) -> None:
        self.ids = ids

    def query(self, **_kwargs):
        return {"ids": [self.ids], "distances": [[0.01 for _ in self.ids]]}


def _engine(db: Database, tmp_path: Path, vector_ids: list[str]) -> SearchEngine:
    adapters = build_sqlite_adapters(db, _VectorHits(vector_ids))
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=AppConfig(base_dir=tmp_path / "memforge").retrieval,
    )
    engine._get_or_compute_embedding = lambda _query: [0.1]
    return engine


@pytest.mark.asyncio
async def test_search_attaches_a_conflict_to_both_memories(db, tmp_path):
    first, second = await _pair(db)
    await _attach_source(db, second.id, "src-jira", "jira")
    engine = _engine(db, tmp_path, [first.id, second.id])

    result = await engine.search("Payroll closes", top_k=2)

    by_id = {item.memory_id: item for item in result["results"]}
    [context] = by_id[first.id].relations
    assert (context.label, context.role) == ("contradicts", "peer")
    assert context.counterpart.memory_id == second.id
    assert context.counterpart.sources == (MemorySourceRef("src-jira", "jira", "Payroll Jira"),)
    assert context.counterpart.evidence_time == EARLIER
    assert by_id[second.id].relations[0].counterpart.evidence_time == LATER
    assert by_id[first.id].relation_notice.startswith(f"Conflicts with Memory {second.id} from Payroll Jira")
    assert by_id[first.id].follow_up == {"suggested_tool": "get_memory", "reason": "result_has_relation_notice"}
    assert by_id[second.id].relations[0].counterpart.memory_id == first.id


@pytest.mark.asyncio
async def test_search_ranks_the_newer_memory_ahead_and_notes_the_older(db, tmp_path):
    older = _memory("mem-older", "Payroll closes on the 20th.")
    newer = _memory("mem-newer", "Salary cutoff moved to the 25th.")
    await db.insert_memory(older)
    await db.insert_memory(newer)
    await _relate(db, older, newer, UPDATES, times={older.id: EARLIER, newer.id: LATER})
    engine = _engine(db, tmp_path, [older.id, newer.id])

    result = await engine.search("Payroll closes", top_k=2)

    assert [item.memory_id for item in result["results"]] == [newer.id, older.id]
    newer_result, older_result = result["results"]
    assert newer_result.relation_notice is None
    assert newer_result.relations[0].role == "newer"
    assert older_result.relation_notice.startswith(f"Updated by the newer Memory {newer.id}")


@pytest.mark.asyncio
async def test_search_returns_one_of_an_equivalent_pair_and_pages_stably(db, tmp_path):
    kept = _memory("mem-kept", "Payroll closes on the 20th.")
    same = _memory("mem-same", "Payroll closing day is the 20th.")
    other = _memory("mem-other", "Payroll approvals close on the 18th.")
    for memory in (kept, same, other):
        await db.insert_memory(memory)
    await _relate(db, kept, same, EQUIVALENT)
    engine = _engine(db, tmp_path, [kept.id, same.id, other.id])

    first_page = await engine.search("Payroll", top_k=1)
    second_page = await engine.search("Payroll", top_k=1, offset=1)

    returned = [item.memory_id for page in (first_page, second_page) for item in page["results"]]
    assert same.id not in returned
    assert sorted(returned) == sorted([kept.id, other.id])
    assert first_page["has_more"] is True
    assert second_page["has_more"] is False
    kept_result = next(item for page in (first_page, second_page) for item in page["results"] if item.memory_id == kept.id)
    assert [(item.label, item.counterpart.memory_id) for item in kept_result.relations] == [("equivalent", same.id)]


# ---------------------------------------------------------------- admin API


def _config(tmp_path: Path) -> AppConfig:
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.worker_enabled = False
    return config


def _dismissal_path(memory_id: str, counterpart_id: str) -> str:
    return f"/api/v1/memories/{memory_id}/relations/{counterpart_id}/dismissal"


@pytest.mark.asyncio
async def test_admin_memory_detail_dismissal_and_restore(db, tmp_path):
    from memforge.server.admin_api import create_admin_app

    first, second = await _pair(db)
    body = {
        "label": "contradicts",
        "expected_content_hash": first.content_hash,
        "counterpart_expected_content_hash": second.content_hash,
        "note": "Different payroll areas.",
    }
    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        detail = client.get(f"/api/v1/memories/{first.id}").json()
        stale = client.post(
            _dismissal_path(first.id, second.id),
            json={**body, "counterpart_expected_content_hash": "stale"},
        )
        wrong_label = client.post(_dismissal_path(first.id, second.id), json={**body, "label": "updates"})
        dismissed = client.post(_dismissal_path(first.id, second.id), json=body)
        hidden = client.get(f"/api/v1/memories/{first.id}").json()
        restored = client.delete(_dismissal_path(second.id, first.id))
        restored_again = client.delete(_dismissal_path(second.id, first.id))
        shown = client.get(f"/api/v1/memories/{first.id}").json()

    [relation] = detail["relations"]
    assert relation["label"] == "contradicts"
    assert relation["counterpart"]["memory_id"] == second.id
    assert relation["counterpart"]["content_hash"] == second.content_hash
    assert detail["relation_notice"].startswith(f"Conflicts with Memory {second.id}")
    assert stale.status_code == 409
    assert wrong_label.status_code == 409
    assert dismissed.status_code == 200
    assert dismissed.json()["label"] == "contradicts"
    assert hidden["relations"] == [] and hidden["relation_notice"] is None
    [undoable] = hidden["dismissed_relations"]
    assert (undoable["labels"], undoable["note"]) == (["contradicts"], "Different payroll areas.")
    assert undoable["counterpart"]["memory_id"] == second.id
    assert detail["dismissed_relations"] == []
    assert restored.json() == {"restored_dismissal_ids": [dismissed.json()["dismissal_id"]]}
    assert restored_again.status_code == 404
    assert len(shown["relations"]) == 1
    assert shown["dismissed_relations"] == []


@pytest.mark.asyncio
async def test_admin_detail_lists_dismissals_only_with_a_visible_other_memory(db, tmp_path):
    from memforge.server.admin_api import create_admin_app

    visible = _memory("mem-dismissed-visible")
    private = _memory("mem-dismissed-private", visibility="private", owner_user_id=OTHER_USER_ID)
    await db.insert_memory(visible)
    await db.insert_memory(private)
    await _relate(db, visible, private, CONTRADICTS)
    await db.record_cross_document_relation_dismissal(
        memory_id=visible.id,
        counterpart_memory_id=private.id,
        label=CONTRADICTS,
        expected_content_hash=visible.content_hash,
        counterpart_expected_content_hash=private.content_hash,
        actor=OTHER_USER_ID,
    )
    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        detail = client.get(f"/api/v1/memories/{visible.id}").json()

    assert detail["dismissed_relations"] == []
    assert await db.list_active_cross_document_relation_dismissals(visible.id, _scope(), limit=10) == ()
    assert len(
        await db.list_active_cross_document_relation_dismissals(visible.id, _scope(OTHER_USER_ID), limit=10)
    ) == 1


@pytest.mark.asyncio
async def test_admin_dismissal_needs_both_memories_visible(db, tmp_path):
    from memforge.server.admin_api import create_admin_app

    visible = _memory("mem-admin-visible")
    private = _memory("mem-admin-private", visibility="private", owner_user_id=OTHER_USER_ID)
    await db.insert_memory(visible)
    await db.insert_memory(private)
    await _relate(db, visible, private, CONTRADICTS)
    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.post(
            _dismissal_path(visible.id, private.id),
            json={
                "label": "contradicts",
                "expected_content_hash": visible.content_hash,
                "counterpart_expected_content_hash": private.content_hash,
            },
        )

    assert response.status_code == 404
    assert await db.db.execute_fetchall("SELECT 1 FROM cross_document_relation_dismissals") == []


@pytest.mark.asyncio
async def test_admin_relation_view_lists_current_pairs(db, tmp_path):
    from memforge.server.admin_api import create_admin_app

    first, second = await _pair(db)
    third = _memory("mem-rel-c")
    await db.insert_memory(third)
    await _relate(db, first, third, EQUIVALENT, decided_at=LATER)
    app = create_admin_app(db=db, config=_config(tmp_path))
    with TestClient(app) as client:
        everything = client.get("/api/v1/memories/relations", params={"limit": 1}).json()
        conflicts = client.get("/api/v1/memories/relations", params={"label": "contradicts"}).json()

    assert everything["total"] == 2
    [newest] = everything["data"]
    assert newest["label"] == "equivalent"
    assert [item["memory_id"] for item in newest["memories"]] == [first.id, third.id]
    assert conflicts["total"] == 1
    assert [item["memory_id"] for item in conflicts["data"][0]["memories"]] == [first.id, second.id]
