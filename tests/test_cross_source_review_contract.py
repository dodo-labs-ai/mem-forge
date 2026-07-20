from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.memory.evidence import (
    AuthorityCase,
    EvidenceContentProvenance,
    EvidenceRelationRecord,
    EvidenceUnit,
    LifecycleAction,
    RelationOutcomeBundle,
    RelationRunRecord,
    RelationType,
    ReviewCase,
)
from memforge.memory.review_contract import (
    CrossSourceReviewMemorySnapshot,
    CrossSourceReviewSupportSnapshot,
    validate_cross_source_review_write,
    validate_pending_review_retry,
)
from memforge.models import Memory, MemoryReview, content_hash
from memforge.storage.database import Database


def _review() -> MemoryReview:
    return MemoryReview(
        id="review-cross-source",
        kind="cross_source_conflict",
        status="pending",
        incumbent_memory_id="mem-incumbent",
        challenger_memory_id="mem-challenger",
        expected_incumbent_updated_at="2026-07-19T00:00:00+00:00",
        expected_challenger_updated_at="2026-07-19T00:01:00+00:00",
    )


def test_pending_review_retry_requires_the_same_immutable_identity() -> None:
    review = _review()

    validate_pending_review_retry(review, review)
    validate_pending_review_retry(
        review,
        replace(
            review,
            expected_incumbent_updated_at=datetime.fromisoformat("2026-07-19T00:00:00+00:00"),
        ),
    )

    with pytest.raises(ValueError, match="retry identity mismatch"):
        validate_pending_review_retry(
            replace(review, reason="different finding"),
            review,
        )


def _bundle() -> RelationOutcomeBundle:
    unit = EvidenceUnit(
        id="evidence-challenger",
        source_id="src-challenger",
        doc_id="doc-challenger",
        doc_revision_id="doc-revision-challenger",
        source_type="confluence",
        source_anchor="page#claim",
        source_lineage_id="unit-challenger",
        project_key="PROJECT-B",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier="repo-b",
        content="Challenger claim",
        excerpt=None,
        evidence_provenance=EvidenceContentProvenance.NO_EXCERPT,
        access_context_hash="access-challenger",
    )
    run = RelationRunRecord(
        id="relation-run-cross-source",
        evidence_unit_id=unit.id,
        access_context_hash="access-challenger",
        candidate_count=1,
        mandatory_candidate_count=1,
        checked_candidate_count=1,
        incomplete_mandatory_buckets=(),
        classifier_version="test-v1",
        lifecycle_action=LifecycleAction.CREATE_REVIEW,
        review_case=ReviewCase.CROSS_SOURCE_CONFLICT,
        status="review_required",
        audit={},
    )
    relation = EvidenceRelationRecord(
        evidence_unit_id=unit.id,
        memory_id="mem-incumbent",
        relation_type=RelationType.CONTRADICTS,
        authority_case=AuthorityCase.CROSS_SOURCE_CONFLICT,
        is_authoritative_support=False,
        source_lineage_id=unit.source_lineage_id,
        confidence=1.0,
        relation_run_id=run.id,
    )
    return RelationOutcomeBundle(
        evidence_unit=unit,
        relations=(relation,),
        relation_run=run,
    )


def _memories() -> tuple[CrossSourceReviewMemorySnapshot, ...]:
    return (
        CrossSourceReviewMemorySnapshot(
            memory_id="mem-incumbent",
            status="active",
            superseded_by=None,
            updated_at="2026-07-19T00:00:00+00:00",
            visibility="workspace",
            owner_user_id=None,
            repo_identifier="repo-a",
            project_key="PROJECT-A",
        ),
        CrossSourceReviewMemorySnapshot(
            memory_id="mem-challenger",
            status="active",
            superseded_by=None,
            updated_at="2026-07-19T00:01:00+00:00",
            visibility="workspace",
            owner_user_id=None,
            repo_identifier="repo-b",
            project_key="PROJECT-B",
        ),
    )


def _support(
    *,
    memory_id: str,
    source_id: str,
    evidence_unit_id: str,
    source_unit_id: str,
    access_context_hash: str,
    current_revision_id: str,
    observation_revision_id: str | None = None,
    visibility: str = "workspace",
    owner_user_id: str | None = None,
    repo_identifier: str | None = None,
    project_key: str = "PROJECT-A",
) -> CrossSourceReviewSupportSnapshot:
    return CrossSourceReviewSupportSnapshot(
        memory_id=memory_id,
        assertion_source_id=source_id,
        assertion_access_context_hash=access_context_hash,
        evidence_unit_id=evidence_unit_id,
        evidence_source_id=source_id,
        evidence_source_lineage_id=source_unit_id,
        evidence_visibility=visibility,
        evidence_owner_user_id=owner_user_id,
        evidence_repo_identifier=repo_identifier,
        evidence_project_key=project_key,
        evidence_access_context_hash=access_context_hash,
        observation_id=f"observation-{memory_id}",
        observation_source_id=source_id,
        observation_revision_id=observation_revision_id or current_revision_id,
        current_observation_revision_id=current_revision_id,
        source_unit_id=source_unit_id,
        source_unit_source_id=source_id,
        source_access_policy=visibility,
        source_owner_user_id=owner_user_id or f"owner-{source_id}",
    )


def _supports() -> tuple[CrossSourceReviewSupportSnapshot, ...]:
    return (
        _support(
            memory_id="mem-incumbent",
            source_id="src-incumbent",
            evidence_unit_id="evidence-incumbent",
            source_unit_id="unit-incumbent",
            access_context_hash="access-incumbent",
            current_revision_id="revision-incumbent",
            repo_identifier="repo-a",
        ),
        _support(
            memory_id="mem-challenger",
            source_id="src-challenger",
            evidence_unit_id="evidence-challenger",
            source_unit_id="unit-challenger",
            access_context_hash="access-challenger",
            current_revision_id="revision-challenger",
            repo_identifier="repo-b",
            project_key="PROJECT-B",
        ),
    )


def test_cross_source_review_contract_accepts_distinct_workspace_lineages() -> None:
    validate_cross_source_review_write(
        _review(),
        _bundle(),
        memories=_memories(),
        supports=_supports(),
    )


def test_cross_source_review_contract_rejects_same_source_pair() -> None:
    supports = (
        _support(
            memory_id="mem-incumbent",
            source_id="src-challenger",
            evidence_unit_id="evidence-incumbent",
            source_unit_id="unit-incumbent",
            access_context_hash="access-incumbent",
            current_revision_id="revision-incumbent",
        ),
        _supports()[1],
    )

    try:
        validate_cross_source_review_write(
            _review(),
            _bundle(),
            memories=_memories(),
            supports=supports,
        )
    except ValueError as exc:
        assert str(exc) == "cross-source review requires distinct source lineages"
    else:
        raise AssertionError("same-source review was accepted")


def test_cross_source_review_contract_rejects_stale_active_support() -> None:
    supports = (
        _support(
            memory_id="mem-incumbent",
            source_id="src-incumbent",
            evidence_unit_id="evidence-incumbent",
            source_unit_id="unit-incumbent",
            access_context_hash="access-incumbent",
            current_revision_id="revision-current",
            observation_revision_id="revision-stale",
        ),
        _supports()[1],
    )

    try:
        validate_cross_source_review_write(
            _review(),
            _bundle(),
            memories=_memories(),
            supports=supports,
        )
    except ValueError as exc:
        assert str(exc) == "cross-source review requires current Support"
    else:
        raise AssertionError("stale active Support was accepted")


def test_cross_source_review_contract_rejects_cross_visibility_pair() -> None:
    private_memories = (
        _memories()[0],
        CrossSourceReviewMemorySnapshot(
            memory_id="mem-challenger",
            status="active",
            superseded_by=None,
            updated_at="2026-07-19T00:01:00+00:00",
            visibility="private",
            owner_user_id="owner-private",
            repo_identifier="repo-b",
            project_key="PROJECT-B",
        ),
    )

    try:
        validate_cross_source_review_write(
            _review(),
            _bundle(),
            memories=private_memories,
            supports=_supports(),
        )
    except ValueError as exc:
        assert str(exc) == "cross-source review access scope is incompatible"
    else:
        raise AssertionError("cross-visibility review was accepted")


def test_cross_source_review_contract_binds_challenger_evidence_and_access_context() -> None:
    supports = (
        _supports()[0],
        _support(
            memory_id="mem-challenger",
            source_id="src-challenger",
            evidence_unit_id="other-evidence",
            source_unit_id="unit-challenger",
            access_context_hash="other-access",
            current_revision_id="revision-challenger",
            project_key="PROJECT-B",
        ),
    )

    try:
        validate_cross_source_review_write(
            _review(),
            _bundle(),
            memories=_memories(),
            supports=supports,
        )
    except ValueError as exc:
        assert str(exc) == "cross-source review evidence is not active challenger Support"
    else:
        raise AssertionError("unbound challenger evidence was accepted")


async def _seed_sqlite_review_graph(
    db: Database,
    *,
    same_source: bool = False,
) -> MemoryReview:
    memory_specs = (
        ("mem-incumbent", "Incumbent claim", "PROJECT-A", "repo-a"),
        ("mem-challenger", "Challenger claim", "PROJECT-B", "repo-b"),
    )
    for memory_id, content, project_key, repo_identifier in memory_specs:
        await db.insert_memory(
            Memory(
                id=memory_id,
                memory_type="fact",
                content=content,
                content_hash=content_hash(content),
                visibility="workspace",
                project_key=project_key,
                repo_identifier=repo_identifier,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )
        )

    for source_id in ("src-incumbent", "src-challenger"):
        await db.upsert_source(
            source_id,
            "confluence",
            source_id,
            "{}",
            "workspace",
            f"owner-{source_id}",
            project_binding={"mode": "fixed", "project_key": "PROJECT-A"},
        )

    incumbent_source_id = "src-challenger" if same_source else "src-incumbent"
    support_specs = (
        (
            "mem-incumbent",
            incumbent_source_id,
            "unit-incumbent",
            "observation-incumbent",
            "revision-incumbent",
            "evidence-incumbent",
            "reference-incumbent",
            "access-incumbent",
            "PROJECT-A",
            "repo-a",
        ),
        (
            "mem-challenger",
            "src-challenger",
            "unit-challenger",
            "observation-challenger",
            "revision-challenger",
            "evidence-challenger",
            "reference-challenger",
            "access-challenger",
            "PROJECT-B",
            "repo-b",
        ),
    )
    now = datetime.now(timezone.utc).isoformat()
    for (
        memory_id,
        source_id,
        source_unit_id,
        observation_id,
        revision_id,
        evidence_unit_id,
        reference_id,
        access_context_hash,
        project_key,
        repo_identifier,
    ) in support_specs:
        await db.db.execute(
            """INSERT INTO source_units (
                   id, source_id, unit_type, provider_key, locator_json,
                   current_revision_id, updated_at
               ) VALUES (?, ?, 'page', ?, '{}', NULL, ?)""",
            (source_unit_id, source_id, source_unit_id, now),
        )
        await db.db.execute(
            """INSERT INTO source_observations (
                   id, source_id, source_unit_id, observation_type, provider_key,
                   locator_json, current_revision_id, updated_at
               ) VALUES (?, ?, ?, 'body', ?, '{}', ?, ?)""",
            (
                observation_id,
                source_id,
                source_unit_id,
                observation_id,
                revision_id,
                now,
            ),
        )
        await db.db.execute(
            """INSERT INTO source_observation_revisions (
                   id, observation_id, semantic_hash, content, metadata_json,
                   observed_at, created_at
               ) VALUES (?, ?, ?, ?, '{}', ?, ?)""",
            (
                revision_id,
                observation_id,
                f"hash-{revision_id}",
                f"content-{revision_id}",
                now,
                now,
            ),
        )
        await db.db.execute(
            """INSERT INTO evidence_units (
                   id, source_id, doc_id, doc_revision_id, source_type,
                   repo_identifier, source_lineage_id, project_key, visibility,
                   owner_user_id, access_context_hash, content,
                   evidence_provenance, created_at, updated_at
               ) VALUES (?, ?, ?, ?, 'confluence', ?, ?, ?, 'workspace',
                         NULL, ?, ?, 'no_excerpt', ?, ?)""",
            (
                evidence_unit_id,
                source_id,
                f"doc-{memory_id}",
                f"doc-revision-{memory_id}",
                repo_identifier,
                source_unit_id,
                project_key,
                access_context_hash,
                f"content-{memory_id}",
                now,
                now,
            ),
        )
        await db.db.execute(
            """INSERT INTO evidence_references (
                   id, evidence_unit_id, role, anchor_kind, observation_id,
                   observation_revision_id, created_at
               ) VALUES (?, ?, 'primary', 'whole_observation', ?, ?, ?)""",
            (
                reference_id,
                evidence_unit_id,
                observation_id,
                revision_id,
                now,
            ),
        )
        await db.db.execute(
            """INSERT INTO memory_support_assertions (
                   id, memory_id, evidence_reference_id, source_id,
                   access_context_hash, active, created_at
               ) VALUES (?, ?, ?, ?, ?, 1, ?)""",
            (
                f"support-{memory_id}",
                memory_id,
                reference_id,
                source_id,
                access_context_hash,
                now,
            ),
        )
    await db.db.commit()

    incumbent = await db.get_memory("mem-incumbent")
    challenger = await db.get_memory("mem-challenger")
    assert incumbent is not None and challenger is not None
    return MemoryReview(
        id="review-cross-source",
        kind="cross_source_conflict",
        status="pending",
        incumbent_memory_id=incumbent.id,
        challenger_memory_id=challenger.id,
        expected_incumbent_updated_at=(incumbent.updated_at.isoformat() if incumbent.updated_at else None),
        expected_challenger_updated_at=(challenger.updated_at.isoformat() if challenger.updated_at else None),
    )


@pytest.mark.asyncio
async def test_sqlite_review_write_enforces_shared_cross_source_contract(
    tmp_path,
) -> None:
    db = Database(str(tmp_path / "review-contract.db"))
    await db.connect()
    try:
        review = await _seed_sqlite_review_graph(db)

        assert (
            await db.record_memory_review_with_relation_outcome(
                review,
                _bundle(),
            )
            == review.id
        )
        stored_review = await db.get_memory_review(review.id)
        assert stored_review is not None
        assert stored_review.status == "pending"
        assert await db.get_relation_run("relation-run-cross-source") is not None
        assert (await db.get_memory("mem-incumbent")).status == "active"
        assert (await db.get_memory("mem-challenger")).status == "active"
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sqlite_review_retry_rejects_identity_drift_before_relation_write(
    tmp_path,
) -> None:
    db = Database(str(tmp_path / "review-retry-identity.db"))
    await db.connect()
    try:
        review = await _seed_sqlite_review_graph(db)
        first_bundle = _bundle()
        await db.record_memory_review_with_relation_outcome(review, first_bundle)
        retry_run_id = "relation-run-review-retry-mismatch"
        retry_bundle = replace(
            first_bundle,
            relation_run=replace(first_bundle.relation_run, id=retry_run_id),
            relations=tuple(replace(relation, relation_run_id=retry_run_id) for relation in first_bundle.relations),
        )

        with pytest.raises(ValueError, match="retry identity mismatch"):
            await db.record_memory_review_with_relation_outcome(
                replace(review, reason="different finding"),
                retry_bundle,
            )

        assert await db.get_relation_run(retry_run_id) is None
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_sqlite_review_write_rolls_back_same_source_finding(tmp_path) -> None:
    db = Database(str(tmp_path / "review-contract-rejected.db"))
    await db.connect()
    try:
        review = await _seed_sqlite_review_graph(db, same_source=True)

        with pytest.raises(
            ValueError,
            match="cross-source review requires distinct source lineages",
        ):
            await db.record_memory_review_with_relation_outcome(
                review,
                _bundle(),
            )

        assert await db.get_memory_review(review.id) is None
        assert await db.get_relation_run("relation-run-cross-source") is None
    finally:
        await db.close()
