from __future__ import annotations

from dataclasses import replace
from datetime import datetime

import pytest
import pytest_asyncio

from memforge.memory.evidence import (
    AuthorityCase,
    CandidateBucket,
    CandidateMemory,
    EvidenceContentProvenance,
    EvidenceRelationRecord,
    EvidenceUnit,
    LifecycleAction,
    RelationOutcomeBundle,
    RelationCandidateRecord,
    RelationRunRecord,
    RelationType,
    ReviewCase,
    relation_bundle_snapshot_audit,
)
from memforge.models import DocumentRecord, Memory, MemoryReview, content_hash
from memforge.storage.database import Database


@pytest_asyncio.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "evidence-store.db"))
    await database.connect()
    try:
        yield database
    finally:
        await database.close()


def _unit() -> EvidenceUnit:
    return EvidenceUnit(
        id="eu-1",
        source_id="src-1",
        doc_id="doc-1",
        doc_revision_id="rev-1",
        source_type="confluence",
        client=None,
        repo_identifier=None,
        source_anchor="page-1#claim-1",
        source_lineage_id="lineage-1",
        source_metadata={"space": "SFPAY"},
        project_key="SFPAY",
        visibility="workspace",
        owner_user_id=None,
        observed_at="2026-06-22T12:00:00+00:00",
        extractor_run_id="run-extract-1",
        access_context_hash="ctx-1",
        content="The payroll validator rejects stale lifecycle groups.",
        excerpt="Validator rejects stale lifecycle groups.",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
    )


def _memory(memory_id: str) -> Memory:
    content = f"Memory content for {memory_id}"
    return Memory(id=memory_id, memory_type="fact", content=content, content_hash=content_hash(content))


def _document(doc_id: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        source="src-1",
        source_url=f"https://example.test/{doc_id}",
        title=f"Document {doc_id}",
        space_or_project="SFPAY",
        author="tester",
        last_modified=datetime.fromisoformat("2026-06-22T12:00:00+00:00"),
        labels=[],
        version=f"v-{doc_id}",
        content_hash=f"hash-{doc_id}",
        token_count=None,
        raw_content_uri=None,
        raw_content_type=None,
        normalized_content_uri=None,
        pdf_content_uri=None,
        last_synced=datetime.fromisoformat("2026-06-22T12:00:00+00:00"),
    )


async def _record_run(
    db: Database,
    run_id: str = "rel-run-1",
    *,
    action: LifecycleAction = LifecycleAction.NONE,
    result_memory_id: str | None = None,
) -> None:
    await db.record_relation_run(
        RelationRunRecord(
            id=run_id,
            evidence_unit_id="eu-1",
            access_context_hash="ctx-1",
            candidate_count=0,
            mandatory_candidate_count=0,
            checked_candidate_count=0,
            incomplete_mandatory_buckets=(),
            classifier_version="test-v1",
            lifecycle_action=action,
            review_case=None,
            status="candidate_audit",
            result_memory_id=result_memory_id,
            audit={},
        )
    )


def _relation(memory_id: str, *, run_id: str = "rel-run-1") -> EvidenceRelationRecord:
    return EvidenceRelationRecord(
        evidence_unit_id="eu-1",
        memory_id=memory_id,
        relation_type=RelationType.SUPPORTS,
        authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
        is_authoritative_support=True,
        source_lineage_id="lineage-1",
        confidence=1.0,
        reason="Memory created from this source evidence unit.",
        excerpt="Validator rejects stale lifecycle groups.",
        classifier_version="test-v1",
        relation_run_id=run_id,
        created_at="2026-06-22T12:00:00+00:00",
    )


@pytest.mark.asyncio
async def test_evidence_unit_round_trips_source_scope_and_metadata(db: Database) -> None:
    await db.upsert_evidence_unit(_unit())

    stored = await db.get_evidence_unit("eu-1")

    assert stored == _unit()


@pytest.mark.asyncio
async def test_evidence_unit_retry_preserves_identity_fields(db: Database) -> None:
    original = _unit()
    retry = replace(
        original,
        source_id="other-source",
        doc_id="other-doc",
        doc_revision_id="other-rev",
        source_type="jira",
        source_anchor="other-anchor",
        source_lineage_id="other-lineage",
        source_metadata={"space": "OTHER"},
        project_key="OTHER",
        visibility="private",
        owner_user_id="andrew.sun01@sap.com",
        content="Different content under the same id must not rewrite canonical evidence.",
        excerpt="Different excerpt",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        observed_at="2026-06-22T13:00:00+00:00",
        extractor_run_id="run-extract-2",
        access_context_hash="ctx-2",
    )

    await db.upsert_evidence_unit(original)
    await db.upsert_evidence_unit(retry)

    stored = await db.get_evidence_unit(original.id)

    assert stored is not None
    assert stored.source_id == original.source_id
    assert stored.doc_id == original.doc_id
    assert stored.doc_revision_id == original.doc_revision_id
    assert stored.source_type == original.source_type
    assert stored.source_anchor == original.source_anchor
    assert stored.source_lineage_id == original.source_lineage_id
    assert stored.source_metadata == original.source_metadata
    assert stored.project_key == original.project_key
    assert stored.visibility == original.visibility
    assert stored.owner_user_id == original.owner_user_id
    assert stored.content == original.content
    assert stored.excerpt == original.excerpt
    assert stored.evidence_provenance == original.evidence_provenance
    assert stored.observed_at == retry.observed_at
    assert stored.extractor_run_id == retry.extractor_run_id
    assert stored.access_context_hash == retry.access_context_hash


@pytest.mark.asyncio
async def test_replace_evidence_relations_keeps_one_current_set_per_unit(db: Database) -> None:
    await db.upsert_evidence_unit(_unit())
    await db.insert_memory(_memory("mem-old"))
    await db.insert_memory(_memory("mem-new"))
    first = EvidenceRelationRecord(
        evidence_unit_id="eu-1",
        memory_id="mem-old",
        relation_type=RelationType.EQUIVALENT,
        authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
        is_authoritative_support=True,
        source_lineage_id="lineage-1",
        confidence=0.91,
        reason="Same source lineage revised the claim.",
        proposed_memory_content="The payroll validator rejects stale lifecycle groups.",
        excerpt="Validator rejects stale lifecycle groups.",
        classifier_version="test-v1",
        relation_run_id="rel-run-1",
        created_at="2026-06-22T12:00:00+00:00",
    )
    second = EvidenceRelationRecord(
        evidence_unit_id="eu-1",
        memory_id="mem-new",
        relation_type=RelationType.SUPPORTS,
        authority_case=AuthorityCase.INDEPENDENT_SUPPORT,
        is_authoritative_support=False,
        source_lineage_id=None,
        confidence=0.72,
        reason="Independent corroborating source.",
        classifier_version="test-v1",
        relation_run_id="rel-run-2",
        created_at="2026-06-22T12:00:01+00:00",
    )

    await db.replace_evidence_relations("eu-1", [first])
    await db.replace_evidence_relations("eu-1", [second])

    assert await db.get_evidence_relations("eu-1") == [second]


@pytest.mark.asyncio
async def test_evidence_relations_reject_non_persisted_relation_types(db: Database) -> None:
    await db.upsert_evidence_unit(_unit())
    await db.insert_memory(_memory("mem-old"))
    relation = EvidenceRelationRecord(
        evidence_unit_id="eu-1",
        memory_id="mem-old",
        relation_type=RelationType.NO_RELATION,
        authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
        is_authoritative_support=False,
        source_lineage_id="lineage-1",
        confidence=0.1,
        reason="Classifier decided there is no relation.",
        classifier_version="test-v1",
        relation_run_id="rel-run-1",
        created_at="2026-06-22T12:00:00+00:00",
    )

    with pytest.raises(ValueError, match="not a persisted evidence relation"):
        await db.replace_evidence_relations("eu-1", [relation])
    with pytest.raises(ValueError, match="not a persisted evidence relation"):
        await db.restore_evidence_relation_snapshot(relation)


@pytest.mark.asyncio
async def test_relation_run_round_trips_candidate_completeness_and_apply_outcome(db: Database) -> None:
    await db.upsert_evidence_unit(_unit())
    run = RelationRunRecord(
        id="rel-run-1",
        evidence_unit_id="eu-1",
        access_context_hash="ctx-1",
        candidate_count=12,
        mandatory_candidate_count=3,
        checked_candidate_count=12,
        incomplete_mandatory_buckets=("same_doc_lineage",),
        classifier_version="test-v1",
        lifecycle_action=LifecycleAction.CREATE_REVIEW,
        review_case=ReviewCase.MANDATORY_INCOMPLETE,
        status="review_required",
        audit={"classifier_batch_keys": ["batch-1"]},
        started_at="2026-06-22T12:00:00+00:00",
        completed_at="2026-06-22T12:00:01+00:00",
    )

    await db.record_relation_run(run)

    assert await db.get_relation_run("rel-run-1") == replace(
        run,
        audit={
            **run.audit,
            **relation_bundle_snapshot_audit(candidates=(), relations=()),
        },
    )


@pytest.mark.asyncio
async def test_relation_run_exact_retry_is_idempotent(db: Database) -> None:
    await db.upsert_evidence_unit(_unit())
    run = RelationRunRecord(
        id="rel-run-1",
        evidence_unit_id="eu-1",
        access_context_hash="ctx-1",
        candidate_count=12,
        mandatory_candidate_count=3,
        checked_candidate_count=12,
        incomplete_mandatory_buckets=("same_doc_lineage",),
        classifier_version="test-v1",
        lifecycle_action=LifecycleAction.CREATE_REVIEW,
        review_case=ReviewCase.MANDATORY_INCOMPLETE,
        status="review_required",
        audit={"classifier_batch_keys": ["batch-1"]},
        started_at="2026-06-22T12:00:00+00:00",
        completed_at="2026-06-22T12:00:01+00:00",
    )

    await db.record_relation_run(run)
    await db.record_relation_run(run)

    assert await db.get_relation_run("rel-run-1") == replace(
        run,
        audit={
            **run.audit,
            **relation_bundle_snapshot_audit(candidates=(), relations=()),
        },
    )


@pytest.mark.asyncio
async def test_relation_run_rejects_id_collision_with_different_payload(db: Database) -> None:
    await db.upsert_evidence_unit(_unit())
    original = RelationRunRecord(
        id="rel-run-1",
        evidence_unit_id="eu-1",
        access_context_hash="ctx-1",
        candidate_count=12,
        mandatory_candidate_count=3,
        checked_candidate_count=12,
        incomplete_mandatory_buckets=("same_doc_lineage",),
        classifier_version="test-v1",
        lifecycle_action=LifecycleAction.CREATE_REVIEW,
        review_case=ReviewCase.MANDATORY_INCOMPLETE,
        status="review_required",
        audit={"classifier_batch_keys": ["batch-1"]},
        started_at="2026-06-22T12:00:00+00:00",
        completed_at="2026-06-22T12:00:01+00:00",
    )
    collision = replace(
        original,
        candidate_count=99,
        mandatory_candidate_count=99,
        checked_candidate_count=99,
        incomplete_mandatory_buckets=("semantic_vector_neighbors",),
        status="success",
        audit={"rewritten": True},
        completed_at="2026-06-22T12:05:00+00:00",
    )

    await db.record_relation_run(original)

    with pytest.raises(RuntimeError, match="relation_run_id collision"):
        await db.record_relation_run(collision)

    assert await db.get_relation_run("rel-run-1") == replace(
        original,
        audit={
            **original.audit,
            **relation_bundle_snapshot_audit(candidates=(), relations=()),
        },
    )


@pytest.mark.asyncio
async def test_replace_relation_candidates_keeps_auditable_checked_universe(
    db: Database,
) -> None:
    await db.upsert_evidence_unit(_unit())
    await db.insert_memory(_memory("mem-anchor"))
    await db.insert_memory(_memory("mem-vector"))
    await _record_run(db)
    anchor = RelationCandidateRecord(
        relation_run_id="rel-run-1",
        evidence_unit_id="eu-1",
        memory_id="mem-anchor",
        bucket=CandidateBucket.EXACT_SOURCE_ANCHOR,
        bucket_rank=0,
        candidate_rank=0,
        score=None,
        is_mandatory=True,
        bucket_complete=True,
        was_checked=True,
        reason="Exact source anchor matched.",
    )
    vector = RelationCandidateRecord(
        relation_run_id="rel-run-1",
        evidence_unit_id="eu-1",
        memory_id="mem-vector",
        bucket=CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS,
        bucket_rank=6,
        candidate_rank=1,
        score=0.82,
        is_mandatory=False,
        bucket_complete=False,
        was_checked=True,
        reason="Semantic neighbor under cap.",
    )

    await db.replace_relation_candidates("rel-run-1", [anchor, vector])

    assert await db.get_relation_candidates("rel-run-1") == [anchor, vector]


@pytest.mark.asyncio
async def test_replace_relation_candidates_is_idempotent_per_run(
    db: Database,
) -> None:
    await db.upsert_evidence_unit(_unit())
    await db.insert_memory(_memory("mem-old"))
    await db.insert_memory(_memory("mem-new"))
    await _record_run(db)
    old_candidate = RelationCandidateRecord(
        relation_run_id="rel-run-1",
        evidence_unit_id="eu-1",
        memory_id="mem-old",
        bucket=CandidateBucket.EXACT_SOURCE_ANCHOR,
        bucket_rank=0,
        candidate_rank=0,
        score=None,
        is_mandatory=True,
        bucket_complete=True,
        was_checked=True,
    )
    new_candidate = RelationCandidateRecord(
        relation_run_id="rel-run-1",
        evidence_unit_id="eu-1",
        memory_id="mem-new",
        bucket=CandidateBucket.SAME_DOC_LINEAGE,
        bucket_rank=1,
        candidate_rank=0,
        score=None,
        is_mandatory=True,
        bucket_complete=True,
        was_checked=True,
    )

    await db.replace_relation_candidates("rel-run-1", [old_candidate])
    await db.replace_relation_candidates("rel-run-1", [new_candidate])

    assert await db.get_relation_candidates("rel-run-1") == [new_candidate]


@pytest.mark.asyncio
async def test_relation_outcome_bundle_exact_retry_is_idempotent(
    db: Database,
) -> None:
    unit = _unit()
    await db.upsert_evidence_unit(unit)
    await db.insert_memory(_memory("mem-original"))
    await db.insert_memory(_memory("mem-retry"))
    original_candidate = RelationCandidateRecord(
        relation_run_id="rel-run-1",
        evidence_unit_id="eu-1",
        memory_id="mem-original",
        bucket=CandidateBucket.EXACT_SOURCE_ANCHOR,
        bucket_rank=0,
        candidate_rank=0,
        score=None,
        is_mandatory=True,
        bucket_complete=True,
        was_checked=True,
        reason="Original audited candidate.",
    )
    original_run = RelationRunRecord(
        id="rel-run-1",
        evidence_unit_id="eu-1",
        access_context_hash="ctx-1",
        candidate_count=1,
        mandatory_candidate_count=1,
        checked_candidate_count=1,
        incomplete_mandatory_buckets=(),
        classifier_version="test-v1",
        lifecycle_action=LifecycleAction.CREATE_MEMORY,
        review_case=None,
        status="success",
        result_memory_id="mem-original",
        audit={},
        started_at="2026-06-22T12:00:00+00:00",
        completed_at="2026-06-22T12:00:01+00:00",
    )
    original_bundle = RelationOutcomeBundle(
        evidence_unit=unit,
        relation_run=original_run,
        candidates=(original_candidate,),
        relations=(_relation("mem-original"),),
    )

    await db.record_relation_outcome_bundle(original_bundle)
    await db.record_relation_outcome_bundle(original_bundle)

    assert await db.get_relation_run("rel-run-1") == replace(
        original_run,
        audit={
            **original_run.audit,
            **relation_bundle_snapshot_audit(
                candidates=(original_candidate,),
                relations=(_relation("mem-original"),),
            ),
        },
    )
    assert await db.get_relation_candidates("rel-run-1") == [original_candidate]
    assert [relation.memory_id for relation in await db.get_evidence_relations_by_memory("mem-original")] == [
        "mem-original"
    ]
    assert await db.get_evidence_relations_by_memory("mem-retry") == []


@pytest.mark.asyncio
async def test_relation_outcome_bundle_retry_uses_immutable_relation_snapshot(
    db: Database,
) -> None:
    unit = _unit()
    await db.upsert_evidence_unit(unit)
    await db.insert_memory(_memory("mem-original"))
    await db.insert_memory(_memory("mem-later"))
    original_candidate = RelationCandidateRecord(
        relation_run_id="rel-run-1",
        evidence_unit_id=unit.id,
        memory_id="mem-original",
        bucket=CandidateBucket.EXACT_SOURCE_ANCHOR,
        bucket_rank=0,
        candidate_rank=0,
        score=None,
        is_mandatory=True,
        bucket_complete=True,
        was_checked=True,
        reason="Original audited candidate.",
    )
    original_run = RelationRunRecord(
        id="rel-run-1",
        evidence_unit_id=unit.id,
        access_context_hash="ctx-1",
        candidate_count=1,
        mandatory_candidate_count=1,
        checked_candidate_count=1,
        incomplete_mandatory_buckets=(),
        classifier_version="test-v1",
        lifecycle_action=LifecycleAction.CREATE_MEMORY,
        review_case=None,
        status="success",
        result_memory_id="mem-original",
        audit={},
        started_at="2026-06-22T12:00:00+00:00",
        completed_at="2026-06-22T12:00:01+00:00",
    )
    original_bundle = RelationOutcomeBundle(
        evidence_unit=unit,
        relation_run=original_run,
        candidates=(original_candidate,),
        relations=(_relation("mem-original", run_id=original_run.id),),
    )
    later_run = RelationRunRecord(
        id="rel-run-2",
        evidence_unit_id=unit.id,
        access_context_hash="ctx-2",
        candidate_count=1,
        mandatory_candidate_count=1,
        checked_candidate_count=1,
        incomplete_mandatory_buckets=(),
        classifier_version="test-v1",
        lifecycle_action=LifecycleAction.CREATE_MEMORY,
        review_case=None,
        status="success",
        result_memory_id="mem-later",
        audit={},
        started_at="2026-06-22T12:05:00+00:00",
        completed_at="2026-06-22T12:05:01+00:00",
    )
    later_candidate = RelationCandidateRecord(
        relation_run_id="rel-run-2",
        evidence_unit_id=unit.id,
        memory_id="mem-later",
        bucket=CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS,
        bucket_rank=6,
        candidate_rank=0,
        score=0.8,
        is_mandatory=False,
        bucket_complete=False,
        was_checked=True,
        reason="Later run changed the current projection.",
    )
    later_bundle = RelationOutcomeBundle(
        evidence_unit=unit,
        relation_run=later_run,
        candidates=(later_candidate,),
        relations=(_relation("mem-later", run_id=later_run.id),),
    )

    await db.record_relation_outcome_bundle(original_bundle)
    await db.record_relation_outcome_bundle(later_bundle)
    await db.record_relation_outcome_bundle(original_bundle)

    current_original = await db.get_evidence_relations_by_memory("mem-original")
    current_later = await db.get_evidence_relations_by_memory("mem-later")
    assert current_original == []
    assert len(current_later) == 1
    assert current_later[0].relation_run_id == later_run.id


@pytest.mark.asyncio
async def test_relation_run_snapshot_migration_backfills_legacy_audit(db: Database) -> None:
    unit = _unit()
    await db.upsert_evidence_unit(unit)
    await db.insert_memory(_memory("mem-original"))
    candidate = RelationCandidateRecord(
        relation_run_id="rel-run-legacy",
        evidence_unit_id=unit.id,
        memory_id="mem-original",
        bucket=CandidateBucket.EXACT_SOURCE_ANCHOR,
        bucket_rank=0,
        candidate_rank=0,
        score=None,
        is_mandatory=True,
        bucket_complete=True,
        was_checked=True,
        reason="Legacy candidate snapshot.",
    )
    relation = _relation("mem-original", run_id="rel-run-legacy")
    await db.db.execute(
        """INSERT INTO relation_runs (
            id, evidence_unit_id, access_context_hash, candidate_count,
            mandatory_candidate_count, checked_candidate_count,
            incomplete_mandatory_buckets_json, classifier_version,
            lifecycle_action, review_case, status, result_memory_id,
            audit_json, started_at, completed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "rel-run-legacy",
            unit.id,
            "ctx-1",
            1,
            1,
            1,
            "[]",
            "test-v1",
            LifecycleAction.CREATE_MEMORY.value,
            None,
            "success",
            "mem-original",
            "{}",
            "2026-06-22T12:00:00+00:00",
            "2026-06-22T12:00:01+00:00",
        ),
    )
    await db.db.execute(
        """INSERT INTO relation_candidates (
            relation_run_id, evidence_unit_id, memory_id, bucket,
            bucket_rank, candidate_rank, score, is_mandatory,
            bucket_complete, was_checked, reason
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            candidate.relation_run_id,
            candidate.evidence_unit_id,
            candidate.memory_id,
            candidate.bucket.value,
            candidate.bucket_rank,
            candidate.candidate_rank,
            candidate.score,
            1,
            1,
            1,
            candidate.reason,
        ),
    )
    await db.db.execute(
        """INSERT INTO relation_run_relations (
            relation_run_id, evidence_unit_id, memory_id, relation_type,
            authority_case, is_authoritative_support, source_lineage_id,
            confidence, reason, proposed_memory_content, excerpt,
            classifier_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            relation.relation_run_id,
            relation.evidence_unit_id,
            relation.memory_id,
            relation.relation_type.value,
            relation.authority_case.value,
            1,
            relation.source_lineage_id,
            relation.confidence,
            relation.reason,
            relation.proposed_memory_content,
            relation.excerpt,
            relation.classifier_version,
            relation.created_at,
        ),
    )
    await db.db.execute("DELETE FROM schema_migrations WHERE version = 26")
    await db.db.commit()

    await db._run_migrations()

    expected_audit = relation_bundle_snapshot_audit(candidates=(candidate,), relations=(relation,))
    stored_run = await db.get_relation_run("rel-run-legacy")
    assert stored_run is not None
    assert stored_run.audit["candidate_snapshot_hash"] == expected_audit["candidate_snapshot_hash"]
    assert stored_run.audit["relation_snapshot_hash"] == expected_audit["relation_snapshot_hash"]
    await db.record_relation_outcome_bundle(
        RelationOutcomeBundle(
            evidence_unit=unit,
            relation_run=replace(stored_run, audit={}),
            candidates=(candidate,),
            relations=(relation,),
        )
    )


@pytest.mark.asyncio
async def test_insert_memory_with_relation_retry_is_idempotent_at_store_boundary(db: Database) -> None:
    await db.upsert_document(_document("doc-1"))
    unit = _unit()
    memory = _memory("mem-create-retry")
    memory.created_at = datetime.fromisoformat("2026-06-22T12:00:00+00:00")
    retry_memory = _memory("mem-create-retry")
    retry_memory.created_at = datetime.fromisoformat("2026-06-23T12:00:00+00:00")
    run = RelationRunRecord(
        id="rel-run-create-retry",
        evidence_unit_id=unit.id,
        access_context_hash="ctx-1",
        candidate_count=0,
        mandatory_candidate_count=0,
        checked_candidate_count=0,
        incomplete_mandatory_buckets=(),
        classifier_version="test-v1",
        lifecycle_action=LifecycleAction.CREATE_MEMORY,
        review_case=None,
        status="success",
        result_memory_id=memory.id,
        audit={},
        started_at="2026-06-22T12:00:00+00:00",
        completed_at="2026-06-22T12:00:01+00:00",
    )
    bundle = RelationOutcomeBundle(
        evidence_unit=unit,
        relation_run=run,
        candidates=(),
        relations=(_relation(memory.id, run_id=run.id),),
    )

    await db.insert_memory_with_source_and_relation(
        memory,
        doc_id="doc-1",
        source_type="confluence",
        relation_outcome=bundle,
    )
    await db.insert_memory_with_source_and_relation(
        retry_memory,
        doc_id="doc-1",
        source_type="confluence",
        relation_outcome=bundle,
    )

    stored = await db.get_memory(memory.id)
    assert stored is not None
    assert stored.created_at == memory.created_at
    async with db.db.execute("SELECT COUNT(*) FROM memories WHERE id = ?", (memory.id,)) as cursor:
        assert (await cursor.fetchone())[0] == 1
    async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (memory.id,)) as cursor:
        assert (await cursor.fetchone())[0] == 1
    async with db.db.execute("SELECT COUNT(*) FROM relation_runs WHERE id = ?", (run.id,)) as cursor:
        assert (await cursor.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_supersede_memory_with_relation_retry_is_idempotent_at_store_boundary(db: Database) -> None:
    await db.upsert_document(_document("doc-1"))
    old = _memory("mem-old-retry")
    new = _memory("mem-new-retry")
    new.created_at = datetime.fromisoformat("2026-06-22T12:00:00+00:00")
    retry_new = _memory("mem-new-retry")
    retry_new.created_at = datetime.fromisoformat("2026-06-23T12:00:00+00:00")
    unit = _unit()
    run = RelationRunRecord(
        id="rel-run-supersede-retry",
        evidence_unit_id=unit.id,
        access_context_hash="ctx-1",
        candidate_count=1,
        mandatory_candidate_count=1,
        checked_candidate_count=1,
        incomplete_mandatory_buckets=(),
        classifier_version="test-v1",
        lifecycle_action=LifecycleAction.SUPERSEDE_MEMORY,
        review_case=None,
        status="success",
        result_memory_id=new.id,
        audit={},
        started_at="2026-06-22T12:00:00+00:00",
        completed_at="2026-06-22T12:00:01+00:00",
    )
    candidate = RelationCandidateRecord(
        relation_run_id=run.id,
        evidence_unit_id=unit.id,
        memory_id=old.id,
        bucket=CandidateBucket.SAME_DOC_LINEAGE,
        bucket_rank=1,
        candidate_rank=0,
        score=None,
        is_mandatory=True,
        bucket_complete=True,
        was_checked=True,
    )
    bundle = RelationOutcomeBundle(
        evidence_unit=unit,
        relation_run=run,
        candidates=(candidate,),
        relations=(_relation(new.id, run_id=run.id),),
    )
    await db.insert_memory(old)

    await db.supersede_memory_with_source_and_relation(
        old.id,
        new,
        replacement_kind="revision",
        doc_id="doc-1",
        source_type="confluence",
        relation_outcome=bundle,
    )
    await db.supersede_memory_with_source_and_relation(
        old.id,
        retry_new,
        replacement_kind="revision",
        doc_id="doc-1",
        source_type="confluence",
        relation_outcome=bundle,
    )

    stored_old = await db.get_memory(old.id)
    stored_new = await db.get_memory(new.id)
    assert stored_old is not None
    assert stored_old.status == "superseded"
    assert stored_old.superseded_by == new.id
    assert stored_new is not None
    assert stored_new.created_at == new.created_at
    async with db.db.execute("SELECT COUNT(*) FROM memories_fts WHERE memory_id = ?", (new.id,)) as cursor:
        assert (await cursor.fetchone())[0] == 1
    async with db.db.execute("SELECT COUNT(*) FROM relation_runs WHERE id = ?", (run.id,)) as cursor:
        assert (await cursor.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_relation_outcome_bundle_rejects_run_id_collision(
    db: Database,
) -> None:
    unit = _unit()
    await db.upsert_evidence_unit(unit)
    await db.insert_memory(_memory("mem-original"))
    await db.insert_memory(_memory("mem-retry"))
    original_candidate = RelationCandidateRecord(
        relation_run_id="rel-run-1",
        evidence_unit_id="eu-1",
        memory_id="mem-original",
        bucket=CandidateBucket.EXACT_SOURCE_ANCHOR,
        bucket_rank=0,
        candidate_rank=0,
        score=None,
        is_mandatory=True,
        bucket_complete=True,
        was_checked=True,
        reason="Original audited candidate.",
    )
    retry_candidate = replace(
        original_candidate,
        memory_id="mem-retry",
        bucket=CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS,
        bucket_rank=6,
        reason="A different candidate universe must not share the same run id.",
    )
    original_run = RelationRunRecord(
        id="rel-run-1",
        evidence_unit_id="eu-1",
        access_context_hash="ctx-1",
        candidate_count=1,
        mandatory_candidate_count=1,
        checked_candidate_count=1,
        incomplete_mandatory_buckets=(),
        classifier_version="test-v1",
        lifecycle_action=LifecycleAction.CREATE_MEMORY,
        review_case=None,
        status="success",
        result_memory_id="mem-original",
        audit={},
        started_at="2026-06-22T12:00:00+00:00",
        completed_at="2026-06-22T12:00:01+00:00",
    )
    retry_run = replace(
        original_run,
        candidate_count=7,
        mandatory_candidate_count=7,
        checked_candidate_count=7,
        result_memory_id="mem-retry",
        audit={},
        completed_at="2026-06-22T12:05:00+00:00",
    )

    await db.record_relation_outcome_bundle(
        RelationOutcomeBundle(
            evidence_unit=unit,
            relation_run=original_run,
            candidates=(original_candidate,),
            relations=(_relation("mem-original"),),
        )
    )

    with pytest.raises(RuntimeError, match="relation_run_id collision"):
        await db.record_relation_outcome_bundle(
            RelationOutcomeBundle(
                evidence_unit=unit,
                relation_run=retry_run,
                candidates=(retry_candidate,),
                relations=(_relation("mem-retry"),),
            )
        )

    assert await db.get_relation_run("rel-run-1") == replace(
        original_run,
        audit={
            **original_run.audit,
            **relation_bundle_snapshot_audit(
                candidates=(original_candidate,),
                relations=(_relation("mem-original"),),
            ),
        },
    )
    assert await db.get_relation_candidates("rel-run-1") == [original_candidate]
    assert [relation.memory_id for relation in await db.get_evidence_relations_by_memory("mem-original")] == [
        "mem-original"
    ]
    assert await db.get_evidence_relations_by_memory("mem-retry") == []


@pytest.mark.asyncio
async def test_relation_outcome_bundle_rejects_same_run_id_candidate_payload_collision(
    db: Database,
) -> None:
    unit = _unit()
    await db.upsert_evidence_unit(unit)
    await db.insert_memory(_memory("mem-original"))
    original_candidate = RelationCandidateRecord(
        relation_run_id="rel-run-1",
        evidence_unit_id="eu-1",
        memory_id="mem-original",
        bucket=CandidateBucket.EXACT_SOURCE_ANCHOR,
        bucket_rank=0,
        candidate_rank=0,
        score=None,
        is_mandatory=True,
        bucket_complete=True,
        was_checked=True,
        reason="Original audited candidate.",
    )
    retry_candidate = replace(
        original_candidate,
        bucket=CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS,
        bucket_rank=6,
        reason="Same run id cannot point at a different candidate universe.",
    )
    run = RelationRunRecord(
        id="rel-run-1",
        evidence_unit_id="eu-1",
        access_context_hash="ctx-1",
        candidate_count=1,
        mandatory_candidate_count=1,
        checked_candidate_count=1,
        incomplete_mandatory_buckets=(),
        classifier_version="test-v1",
        lifecycle_action=LifecycleAction.CREATE_MEMORY,
        review_case=None,
        status="success",
        result_memory_id="mem-original",
        audit={"batch": "one"},
    )
    relation = _relation("mem-original")

    await db.record_relation_outcome_bundle(
        RelationOutcomeBundle(
            evidence_unit=unit,
            relation_run=run,
            candidates=(original_candidate,),
            relations=(relation,),
        )
    )

    with pytest.raises(RuntimeError, match="relation_run_id collision.*relation_candidates"):
        await db.record_relation_outcome_bundle(
            RelationOutcomeBundle(
                evidence_unit=unit,
                relation_run=run,
                candidates=(retry_candidate,),
                relations=(relation,),
            )
        )


@pytest.mark.asyncio
async def test_relation_outcome_bundle_rejects_same_run_id_audit_payload_collision(
    db: Database,
) -> None:
    unit = _unit()
    run = RelationRunRecord(
        id="rel-run-1",
        evidence_unit_id="eu-1",
        access_context_hash="ctx-1",
        candidate_count=0,
        mandatory_candidate_count=0,
        checked_candidate_count=0,
        incomplete_mandatory_buckets=(),
        classifier_version="test-v1",
        lifecycle_action=LifecycleAction.NONE,
        review_case=None,
        status="success",
        result_memory_id=None,
        audit={"batch": "one"},
        started_at="2026-06-22T12:00:00+00:00",
    )

    await db.record_relation_outcome_bundle(
        RelationOutcomeBundle(evidence_unit=unit, relation_run=run)
    )

    with pytest.raises(RuntimeError, match="relation_run_id collision"):
        await db.record_relation_outcome_bundle(
            RelationOutcomeBundle(
                evidence_unit=unit,
                relation_run=replace(run, audit={"batch": "two"}),
            )
        )

    assert await db.get_relation_run("rel-run-1") == replace(
        run,
        audit={
            **run.audit,
            **relation_bundle_snapshot_audit(candidates=(), relations=()),
        },
    )


@pytest.mark.asyncio
async def test_relation_outcome_bundle_rejects_same_run_id_relation_payload_collision(
    db: Database,
) -> None:
    unit = _unit()
    await db.upsert_evidence_unit(unit)
    await db.insert_memory(_memory("mem-original"))
    candidate = RelationCandidateRecord(
        relation_run_id="rel-run-1",
        evidence_unit_id="eu-1",
        memory_id="mem-original",
        bucket=CandidateBucket.EXACT_SOURCE_ANCHOR,
        bucket_rank=0,
        candidate_rank=0,
        score=None,
        is_mandatory=True,
        bucket_complete=True,
        was_checked=True,
        reason="Original audited candidate.",
    )
    run = RelationRunRecord(
        id="rel-run-1",
        evidence_unit_id="eu-1",
        access_context_hash="ctx-1",
        candidate_count=1,
        mandatory_candidate_count=1,
        checked_candidate_count=1,
        incomplete_mandatory_buckets=(),
        classifier_version="test-v1",
        lifecycle_action=LifecycleAction.CREATE_MEMORY,
        review_case=None,
        status="success",
        result_memory_id="mem-original",
        audit={"batch": "one"},
    )
    relation = _relation("mem-original")

    await db.record_relation_outcome_bundle(
        RelationOutcomeBundle(
            evidence_unit=unit,
            relation_run=run,
            candidates=(candidate,),
            relations=(relation,),
        )
    )

    with pytest.raises(RuntimeError, match="relation_run_id collision.*evidence_relations"):
        await db.record_relation_outcome_bundle(
            RelationOutcomeBundle(
                evidence_unit=unit,
                relation_run=run,
                candidates=(candidate,),
                relations=(replace(relation, confidence=0.42, reason="Different edge payload."),),
            )
        )


@pytest.mark.asyncio
async def test_candidate_memories_by_source_doc_are_active_and_complete(db: Database) -> None:
    active = _memory("mem-active")
    corroborated = _memory("mem-corroborated")
    retired = _memory("mem-retired")
    retired.status = "retired"
    await db.upsert_document(_document("doc-1"))
    await db.insert_memory(active)
    await db.insert_memory(corroborated)
    await db.insert_memory(retired)
    await db.add_memory_source(active.id, "doc-1", "confluence", "active excerpt")
    await db.add_memory_source(
        corroborated.id,
        "doc-1",
        "confluence",
        "corroborated excerpt",
        support_kind="corroborated",
    )
    await db.add_memory_source(retired.id, "doc-1", "confluence", "retired excerpt")

    candidates = await db.get_candidate_memories_by_source_doc(doc_id="doc-1")

    assert candidates == [
        CandidateMemory(
            memory_id=active.id,
            source_id="src-1",
            doc_id="doc-1",
            source_lineage_id="doc-1",
            visibility=active.visibility,
            owner_user_id=active.owner_user_id,
            repo_identifier=active.repo_identifier,
        ),
        CandidateMemory(
            memory_id=corroborated.id,
            source_id="src-1",
            doc_id="doc-1",
            source_lineage_id="doc-1",
            visibility=corroborated.visibility,
            owner_user_id=corroborated.owner_user_id,
            repo_identifier=corroborated.repo_identifier,
        ),
    ]
    extracted_only = await db.get_candidate_memories_by_source_doc(
        doc_id="doc-1",
        support_kind="extracted",
    )
    assert [candidate.memory_id for candidate in extracted_only] == [active.id]


@pytest.mark.asyncio
async def test_candidate_memories_by_source_anchor_use_current_evidence_relations(db: Database) -> None:
    await db.upsert_evidence_unit(_unit())
    memory = _memory("mem-anchor")
    await db.insert_memory(memory)
    await db.replace_evidence_relations(
        "eu-1",
        [
            EvidenceRelationRecord(
                evidence_unit_id="eu-1",
                memory_id=memory.id,
                relation_type=RelationType.SUPPORTS,
                authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
                is_authoritative_support=True,
                source_lineage_id="lineage-1",
                confidence=0.9,
                classifier_version="test-v1",
                relation_run_id="rel-run-1",
                created_at="2026-06-22T12:00:00+00:00",
            )
        ],
    )

    candidates = await db.get_candidate_memories_by_source_anchor(
        source_id="src-1",
        source_anchor="page-1#claim-1",
    )

    assert len(candidates) == 1
    assert candidates[0].memory_id == memory.id
    assert candidates[0].source_id == "src-1"
    assert candidates[0].source_anchor == "page-1#claim-1"
    assert candidates[0].source_lineage_id == "lineage-1"
    assert candidates[0].source_metadata == {"space": "SFPAY"}


@pytest.mark.asyncio
async def test_candidate_memories_by_agent_claim_use_claim_anchor_lineage(db: Database) -> None:
    claim_unit = replace(
        _unit(),
        id="eu-claim",
        source_type="agent_session",
        source_anchor="u:repo:concept:claim",
        source_lineage_id="u:repo:concept:claim",
        source_metadata={"claim_anchor": "u:repo:concept:claim"},
        visibility="private",
        owner_user_id="u",
        repo_identifier="repo",
    )
    await db.upsert_evidence_unit(claim_unit)
    memory = _memory("mem-claim")
    memory.visibility = "private"
    memory.owner_user_id = "u"
    memory.repo_identifier = "repo"
    await db.insert_memory(memory)
    await db.replace_evidence_relations(
        "eu-claim",
        [
            EvidenceRelationRecord(
                evidence_unit_id="eu-claim",
                memory_id=memory.id,
                relation_type=RelationType.SUPPORTS,
                authority_case=AuthorityCase.SAME_AGENT_CLAIM,
                is_authoritative_support=True,
                source_lineage_id="u:repo:concept:claim",
                confidence=0.9,
                classifier_version="test-v1",
                relation_run_id="rel-run-claim",
                created_at="2026-06-22T12:00:00+00:00",
            )
        ],
    )

    candidates = await db.get_candidate_memories_by_agent_claim(
        claim_anchor="u:repo:concept:claim",
    )

    assert len(candidates) == 1
    assert candidates[0].memory_id == memory.id
    assert candidates[0].visibility == "private"
    assert candidates[0].owner_user_id == "u"
    assert candidates[0].repo_identifier == "repo"


@pytest.mark.asyncio
async def test_candidate_memories_by_existing_relation_graph_are_active(db: Database) -> None:
    await db.upsert_evidence_unit(_unit())
    active = _memory("mem-existing-active")
    inactive = _memory("mem-existing-inactive")
    inactive.status = "superseded"
    await db.insert_memory(active)
    await db.insert_memory(inactive)
    await db.replace_evidence_relations(
        "eu-1",
        [
            EvidenceRelationRecord(
                evidence_unit_id="eu-1",
                memory_id=active.id,
                relation_type=RelationType.SUPPORTS,
                authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
                is_authoritative_support=True,
                source_lineage_id="lineage-1",
                confidence=0.9,
                classifier_version="test-v1",
                relation_run_id="rel-run-existing",
                created_at="2026-06-22T12:00:00+00:00",
            ),
            EvidenceRelationRecord(
                evidence_unit_id="eu-1",
                memory_id=inactive.id,
                relation_type=RelationType.SUPPORTS,
                authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
                is_authoritative_support=True,
                source_lineage_id="lineage-1",
                confidence=0.9,
                classifier_version="test-v1",
                relation_run_id="rel-run-existing",
                created_at="2026-06-22T12:00:00+00:00",
            ),
        ],
    )

    candidates = await db.get_candidate_memories_by_existing_relation_graph(
        evidence_unit_id="eu-1",
    )

    assert len(candidates) == 1
    assert candidates[0].memory_id == active.id
    assert candidates[0].source_anchor == "page-1#claim-1"
    assert candidates[0].source_metadata == {"space": "SFPAY"}


@pytest.mark.asyncio
async def test_materialized_evidence_history_survives_current_relation_invalidation(db: Database) -> None:
    await db.upsert_evidence_unit(_unit())
    await db.insert_memory(_memory("mem-materialized"))
    await _record_run(db, action=LifecycleAction.CREATE_MEMORY, result_memory_id="mem-purge-evidence")
    await db.replace_evidence_relations("eu-1", [_relation("mem-materialized")])

    await db.db.execute("DELETE FROM evidence_relations WHERE evidence_unit_id = ?", ("eu-1",))
    await db.db.commit()

    assert await db.has_materialized_evidence_unit("eu-1") is True


@pytest.mark.asyncio
async def test_purge_memory_deletes_related_evidence_unit_content_and_runs(db: Database) -> None:
    await db.upsert_evidence_unit(_unit())
    await db.insert_memory(_memory("mem-purge-evidence"))
    await _record_run(db, action=LifecycleAction.CREATE_MEMORY, result_memory_id="mem-purge-evidence")
    await db.replace_evidence_relations("eu-1", [_relation("mem-purge-evidence")])
    await db.replace_relation_candidates(
        "rel-run-1",
        [
            RelationCandidateRecord(
                relation_run_id="rel-run-1",
                evidence_unit_id="eu-1",
                memory_id="mem-purge-evidence",
                bucket=CandidateBucket.EXACT_SOURCE_ANCHOR,
                bucket_rank=0,
                candidate_rank=0,
                score=1.0,
                is_mandatory=True,
                bucket_complete=True,
                was_checked=True,
                reason="same source anchor",
            )
        ],
    )

    assert await db.purge_memory("mem-purge-evidence") is True

    assert await db.get_evidence_unit("eu-1") is None
    assert await db.get_relation_run("rel-run-1") is None
    assert await db.get_relation_candidates("rel-run-1") == []
    assert await db.get_evidence_relations("eu-1") == []


@pytest.mark.asyncio
async def test_purge_memory_does_not_delete_non_materializing_evidence_unit(db: Database) -> None:
    await db.upsert_evidence_unit(_unit())
    await db.insert_memory(_memory("mem-purge-candidate"))
    await _record_run(db, action=LifecycleAction.CREATE_REVIEW)
    await db.replace_evidence_relations("eu-1", [_relation("mem-purge-candidate")])

    assert await db.purge_memory("mem-purge-candidate") is True

    assert await db.get_evidence_unit("eu-1") is not None
    assert await db.get_relation_run("rel-run-1") is not None
    assert await db.get_evidence_relations("eu-1") == []


@pytest.mark.asyncio
async def test_delete_document_deletes_derived_evidence_graph(db: Database) -> None:
    await db.upsert_document(_document("doc-1"))
    await db.upsert_evidence_unit(_unit())
    await db.insert_memory(_memory("mem-doc-evidence"))
    await db.add_memory_source("mem-doc-evidence", "doc-1", "confluence")
    await _record_run(db, action=LifecycleAction.CREATE_MEMORY)
    await db.replace_evidence_relations("eu-1", [_relation("mem-doc-evidence")])

    await db.delete_document("doc-1")

    assert await db.get_evidence_unit("eu-1") is None
    assert await db.get_relation_run("rel-run-1") is None
    assert await db.get_evidence_relations("eu-1") == []


@pytest.mark.asyncio
async def test_delete_source_cascade_deletes_docless_source_owned_evidence_graph(db: Database) -> None:
    unit = replace(
        _unit(),
        id="eu-docless",
        source_id="src-docless",
        doc_id=None,
        source_anchor="agent-session#claim-1",
        source_lineage_id="agent-session#claim-1",
        source_type="agent_session",
    )
    await db.upsert_evidence_unit(unit)
    await db.insert_memory(_memory("mem-docless-source"))
    run = RelationRunRecord(
        id="rel-run-docless",
        evidence_unit_id=unit.id,
        access_context_hash="ctx-1",
        candidate_count=0,
        mandatory_candidate_count=0,
        checked_candidate_count=0,
        incomplete_mandatory_buckets=(),
        classifier_version="test-v1",
        lifecycle_action=LifecycleAction.CREATE_MEMORY,
        review_case=None,
        status="success",
        result_memory_id="mem-docless-source",
        audit={},
    )
    await db.record_relation_run(run)
    await db.replace_evidence_relations(
        unit.id,
        [
            replace(
                _relation("mem-docless-source", run_id=run.id),
                evidence_unit_id=unit.id,
                source_lineage_id=unit.source_lineage_id,
            )
        ],
    )

    await db.delete_source_cascade("src-docless")

    assert await db.get_evidence_unit(unit.id) is None
    assert await db.get_relation_run(run.id) is None
    assert await db.get_evidence_relations(unit.id) == []


@pytest.mark.asyncio
async def test_remove_memory_source_deletes_linked_evidence_graph(db: Database) -> None:
    await db.upsert_document(_document("doc-1"))
    await db.upsert_evidence_unit(_unit())
    await db.insert_memory(_memory("mem-source-evidence"))
    await db.add_memory_source("mem-source-evidence", "doc-1", "confluence")
    await _record_run(db, action=LifecycleAction.ATTACH_SUPPORT, result_memory_id="mem-source-evidence")
    await db.replace_evidence_relations("eu-1", [_relation("mem-source-evidence")])

    await db.remove_memory_source("mem-source-evidence", "doc-1")

    assert await db.get_evidence_unit("eu-1") is None
    assert await db.get_relation_run("rel-run-1") is None
    assert await db.get_evidence_relations("eu-1") == []


@pytest.mark.asyncio
async def test_mark_pending_review_with_case_is_idempotent_for_same_review_id(db: Database) -> None:
    await db.insert_memory(_memory("mem-incumbent"))
    await db.insert_memory(_memory("mem-review-repeat"))
    review = MemoryReview(
        id="rev-repeat",
        kind="supersede",
        status="pending",
        incumbent_memory_id="mem-incumbent",
        challenger_memory_id="mem-review-repeat",
        reason="conflict",
        created_at=datetime.fromisoformat("2026-06-22T12:00:00+00:00"),
    )

    await db.mark_memory_pending_review_with_case("mem-review-repeat", reason="conflict", review=review)
    await db.mark_memory_pending_review_with_case("mem-review-repeat", reason="conflict", review=review)

    stored = await db.get_memory_review("rev-repeat")
    reviews = await db.list_memory_reviews(status="pending")
    memory = await db.get_memory("mem-review-repeat")
    assert stored is not None
    assert [item.id for item in reviews] == ["rev-repeat"]
    assert memory.status == "pending_review"


@pytest.mark.asyncio
async def test_mark_pending_review_does_not_reopen_resolved_review(db: Database) -> None:
    await db.insert_memory(_memory("mem-incumbent"))
    await db.insert_memory(_memory("mem-review-resolved"))
    original = MemoryReview(
        id="rev-resolved",
        kind="supersede",
        status="pending",
        incumbent_memory_id="mem-incumbent",
        challenger_memory_id="mem-review-resolved",
        reason="original conflict",
        created_at=datetime.fromisoformat("2026-06-22T12:00:00+00:00"),
    )
    await db.mark_memory_pending_review_with_case("mem-review-resolved", reason="conflict", review=original)
    await db.resolve_memory_review(
        "rev-resolved",
        status="approved",
        reviewer="tester",
        review_note="resolved by a person",
    )
    await db.update_memory_status("mem-review-resolved", "active")

    retry = MemoryReview(
        id="rev-resolved",
        kind="supersede",
        status="pending",
        incumbent_memory_id="mem-incumbent",
        challenger_memory_id="mem-review-resolved",
        reason="retry conflict",
        created_at=datetime.fromisoformat("2026-06-23T12:00:00+00:00"),
    )
    await db.mark_memory_pending_review_with_case("mem-review-resolved", reason="retry", review=retry)

    stored = await db.get_memory_review("rev-resolved")
    memory = await db.get_memory("mem-review-resolved")
    assert stored is not None
    assert stored.status == "approved"
    assert stored.reason == "original conflict"
    assert stored.created_at == original.created_at
    assert stored.resolved_at is not None
    assert memory.status == "active"


@pytest.mark.asyncio
async def test_supersede_preserves_evidence_relation_history_for_old_memory(
    db: Database,
) -> None:
    await db.upsert_evidence_unit(_unit())
    old = _memory("mem-old")
    new = _memory("mem-new")
    await db.upsert_document(_document("doc-1"))
    await db.insert_memory(old)
    await db.add_memory_source(old.id, "doc-1", "confluence", "source excerpt")
    await db.replace_evidence_relations(
        "eu-1",
        [
            EvidenceRelationRecord(
                evidence_unit_id="eu-1",
                memory_id=old.id,
                relation_type=RelationType.EQUIVALENT,
                authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
                is_authoritative_support=True,
                source_lineage_id="lineage-1",
                confidence=0.91,
                reason="Same source lineage revised the claim.",
                classifier_version="test-v1",
                relation_run_id="rel-run-1",
                created_at="2026-06-22T12:00:00+00:00",
            )
        ],
    )

    await db.supersede_memory(
        old.id,
        new,
        replacement_reason="same_source_replacement",
        replacement_kind="supersession",
    )

    relations = await db.get_evidence_relations("eu-1")
    assert relations[0].memory_id == old.id
    assert relations[0].relation_type == RelationType.EQUIVALENT
    sources = await db.get_memory_sources(old.id)
    assert [(source.doc_id, source.source_type, source.excerpt) for source in sources] == [
        ("doc-1", "confluence", "source excerpt")
    ]


@pytest.mark.asyncio
async def test_supersede_memory_rejects_self_supersede(db: Database) -> None:
    memory = _memory("mem-self")

    with pytest.raises(ValueError, match="cannot supersede a memory with itself"):
        await db.supersede_memory(
            "mem-self",
            memory,
            replacement_reason="same id",
            replacement_kind="revision",
        )
