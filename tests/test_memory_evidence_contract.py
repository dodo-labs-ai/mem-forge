from __future__ import annotations

from dataclasses import replace

import pytest

from memforge.memory.evidence import (
    AccessContext,
    AuthorityCase,
    CandidateBucket,
    CandidateBucketResult,
    CandidateMemory,
    EvidenceContentProvenance,
    EvidenceUnit,
    LifecycleAction,
    MemoryRelationApplyService,
    RelationDecision,
    RelationType,
    ReviewCase,
    build_candidate_universe,
    build_mandatory_candidate_bucket_results,
    classify_authority_case,
    is_destructive_authority,
    is_mandatory_candidate_bucket,
    relation_run_id_for,
)


def _unit(**overrides) -> EvidenceUnit:
    defaults = dict(
        id="eu-1",
        source_id="src-1",
        doc_id="doc-1",
        doc_revision_id="rev-1",
        source_type="confluence",
        source_anchor="anchor-1",
        source_lineage_id="lineage-1",
        project_key="SFPAY",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
        content="SFPAY validation rejects stale lifecycle group assignments.",
        excerpt="The validator rejects stale lifecycle group assignments.",
        evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
    )
    defaults.update(overrides)
    return EvidenceUnit(**defaults)


def _candidate(**overrides) -> CandidateMemory:
    defaults = dict(
        memory_id="mem-1",
        source_id="src-1",
        doc_id="doc-1",
        source_lineage_id="lineage-1",
        visibility="workspace",
        owner_user_id=None,
        repo_identifier=None,
    )
    defaults.update(overrides)
    return CandidateMemory(**defaults)


def _access(**overrides) -> AccessContext:
    defaults = dict(
        actor_user_id="andrew.sun01@sap.com",
        workspace_ids=("ws-1",),
        role_grants=("workspace_admin",),
        source_subscriptions=("src-1", "src-2"),
        repo_identifier=None,
        operation_type="source_sync",
    )
    defaults.update(overrides)
    return AccessContext(**defaults)


def test_relation_run_id_includes_classifier_action_and_candidate_contract() -> None:
    unit = _unit(access_context_hash="scope-a", extractor_run_id="run-a")

    first = relation_run_id_for(
        prefix="doc",
        unit=unit,
        action=LifecycleAction.SUPERSEDE_MEMORY,
        classifier_version="memory-engine-v1",
        candidate_memory_id="mem-old",
        relation_type=RelationType.REFINES,
        authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
    )
    retry = relation_run_id_for(
        prefix="doc",
        unit=unit,
        action=LifecycleAction.SUPERSEDE_MEMORY,
        classifier_version="memory-engine-v1",
        candidate_memory_id="mem-old",
        relation_type=RelationType.REFINES,
        authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
    )
    changed_classifier = relation_run_id_for(
        prefix="doc",
        unit=unit,
        action=LifecycleAction.SUPERSEDE_MEMORY,
        classifier_version="memory-engine-v2",
        candidate_memory_id="mem-old",
        relation_type=RelationType.REFINES,
        authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
    )
    changed_candidate = relation_run_id_for(
        prefix="doc",
        unit=unit,
        action=LifecycleAction.SUPERSEDE_MEMORY,
        classifier_version="memory-engine-v1",
        candidate_memory_id="mem-other",
        relation_type=RelationType.REFINES,
        authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
    )

    assert first == retry
    assert first != changed_classifier
    assert first != changed_candidate


def test_independent_visible_refines_is_non_destructive_authority() -> None:
    candidate = _candidate(source_id="src-2", doc_id="doc-2", source_lineage_id="lineage-2")

    authority = classify_authority_case(
        _unit(),
        candidate,
        CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS,
        RelationType.REFINES,
        _access(),
    )

    assert authority is AuthorityCase.INDEPENDENT_REFINEMENT


def test_cross_scope_private_candidate_is_blocked_before_semantic_authority() -> None:
    private_candidate = _candidate(
        visibility="private",
        owner_user_id="other-user",
        repo_identifier="github.com/example/repo",
    )

    authority = classify_authority_case(
        _unit(owner_user_id="andrew.sun01@sap.com", repo_identifier="github.com/example/repo"),
        private_candidate,
        CandidateBucket.SAME_AGENT_CLAIM,
        RelationType.EQUIVALENT,
        _access(repo_identifier="github.com/example/repo"),
    )

    assert authority is AuthorityCase.CROSS_SCOPE_BLOCKED


def test_same_private_repo_scope_is_visible_but_not_destructive_authority() -> None:
    candidate = _candidate(
        visibility="private",
        owner_user_id="andrew.sun01@sap.com",
        repo_identifier="github.com/example/repo",
        source_id="src-2",
        doc_id="doc-2",
        source_lineage_id="lineage-2",
    )

    authority = classify_authority_case(
        _unit(
            visibility="private",
            owner_user_id="andrew.sun01@sap.com",
            repo_identifier="github.com/example/repo",
        ),
        candidate,
        CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS,
        RelationType.EQUIVALENT,
        _access(repo_identifier="github.com/example/repo"),
    )

    assert authority is AuthorityCase.SAME_PRIVATE_REPO_SCOPE
    assert not is_destructive_authority(authority)


@pytest.mark.parametrize(
    "bucket",
    [
        CandidateBucket.EXACT_SOURCE_ANCHOR,
        CandidateBucket.SAME_DOC_LINEAGE,
        CandidateBucket.SAME_AGENT_CLAIM,
        CandidateBucket.EXISTING_RELATION_GRAPH,
        CandidateBucket.SAME_MEMORY_SOURCE_AUTHORITY,
    ],
)
def test_mandatory_candidate_buckets_are_declared_as_uncapped_contract(bucket: CandidateBucket) -> None:
    assert is_mandatory_candidate_bucket(bucket)


@pytest.mark.parametrize(
    "bucket",
    [
        CandidateBucket.SHARED_ENTITIES,
        CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS,
        CandidateBucket.LEXICAL_BM25,
        CandidateBucket.SAME_PROJECT,
        CandidateBucket.SOURCE_TITLE_OR_TAG_OVERLAP,
    ],
)
def test_recall_candidate_buckets_are_not_destructive_completeness_authority(bucket: CandidateBucket) -> None:
    assert not is_mandatory_candidate_bucket(bucket)


def test_legacy_limited_evidence_cannot_supersede_memory() -> None:
    service = MemoryRelationApplyService()
    unit = _unit(
        excerpt=None,
        evidence_provenance=EvidenceContentProvenance.LEGACY_LIMITED,
    )
    decision = RelationDecision(
        candidate_memory_id="mem-1",
        relation_type=RelationType.EQUIVALENT,
        authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
        confidence=0.95,
        proposed_memory_content="Updated lifecycle group assignment rule.",
    )

    result = service.derive_lifecycle(unit, [decision])

    assert result.action is LifecycleAction.CREATE_REVIEW
    assert result.review_case is ReviewCase.LEGACY_LIMITED_EVIDENCE


def test_llm_proposed_content_requires_source_provenance() -> None:
    service = MemoryRelationApplyService()
    unit = _unit(excerpt=None, evidence_provenance=EvidenceContentProvenance.NO_EXCERPT)
    decision = RelationDecision(
        candidate_memory_id="mem-1",
        relation_type=RelationType.REFINES,
        authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
        confidence=0.9,
        proposed_memory_content="Refined lifecycle group assignment rule.",
    )

    result = service.derive_lifecycle(unit, [decision])

    assert result.action is LifecycleAction.CREATE_REVIEW
    assert result.review_case is ReviewCase.MISSING_CONTENT_PROVENANCE


def test_destructive_decision_requires_source_excerpt_even_without_proposed_content() -> None:
    service = MemoryRelationApplyService()
    unit = _unit(excerpt=None, evidence_provenance=EvidenceContentProvenance.NO_EXCERPT)
    decision = RelationDecision(
        candidate_memory_id="mem-1",
        relation_type=RelationType.EQUIVALENT,
        authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
        confidence=0.95,
        matched_bucket=CandidateBucket.SAME_DOC_LINEAGE,
        matched_bucket_complete=True,
    )

    result = service.derive_lifecycle(unit, [decision])

    assert result.action is LifecycleAction.CREATE_REVIEW
    assert result.review_case is ReviewCase.MISSING_CONTENT_PROVENANCE


def test_cross_batch_destructive_conflicts_create_review() -> None:
    service = MemoryRelationApplyService()
    decisions = [
        RelationDecision(
            candidate_memory_id="mem-a",
            relation_type=RelationType.EQUIVALENT,
            authority_case=AuthorityCase.SAME_DOCUMENT_REVISION,
            confidence=0.94,
            classifier_batch_key="batch-1",
        ),
        RelationDecision(
            candidate_memory_id="mem-b",
            relation_type=RelationType.EQUIVALENT,
            authority_case=AuthorityCase.SAME_DOCUMENT_REVISION,
            confidence=0.92,
            classifier_batch_key="batch-2",
        ),
    ]

    result = service.derive_lifecycle(_unit(), decisions)

    assert result.action is LifecycleAction.CREATE_REVIEW
    assert result.review_case is ReviewCase.MULTI_DESTRUCTIVE_MATCH


def test_same_batch_multi_target_destructive_conflicts_create_review() -> None:
    service = MemoryRelationApplyService()
    decisions = [
        RelationDecision(
            candidate_memory_id="mem-a",
            relation_type=RelationType.EQUIVALENT,
            authority_case=AuthorityCase.SAME_DOCUMENT_REVISION,
            confidence=0.94,
            classifier_batch_key="batch-1",
        ),
        RelationDecision(
            candidate_memory_id="mem-b",
            relation_type=RelationType.REFINES,
            authority_case=AuthorityCase.SAME_DOCUMENT_REVISION,
            confidence=0.92,
            classifier_batch_key="batch-1",
        ),
    ]

    result = service.derive_lifecycle(_unit(), decisions)

    assert result.action is LifecycleAction.CREATE_REVIEW
    assert result.review_case is ReviewCase.MULTI_DESTRUCTIVE_MATCH


def test_cross_source_conflict_creates_review_not_new_memory() -> None:
    service = MemoryRelationApplyService()
    decision = RelationDecision(
        candidate_memory_id="mem-cross-source",
        relation_type=RelationType.CONTRADICTS,
        authority_case=AuthorityCase.CROSS_SOURCE_CONFLICT,
        confidence=0.91,
        matched_bucket=CandidateBucket.SHARED_ENTITIES,
        matched_bucket_complete=True,
    )

    result = service.derive_lifecycle(_unit(), [decision])

    assert result.action is LifecycleAction.CREATE_REVIEW
    assert result.review_case is None
    assert result.target_memory_id == "mem-cross-source"


def test_evidence_unit_lifetime_uniqueness_blocks_second_created_memory() -> None:
    service = MemoryRelationApplyService()
    unit = _unit()

    first = service.derive_lifecycle(unit, [])
    second = service.derive_lifecycle(replace(unit, access_context_hash="changed-access"), [])

    assert first.action is LifecycleAction.CREATE_MEMORY
    assert second.action is LifecycleAction.CREATE_REVIEW
    assert second.review_case is ReviewCase.EVIDENCE_UNIT_ALREADY_MATERIALIZED
    assert service.created_memory_ids_by_evidence_unit[unit.id] == first.created_memory_id


@pytest.mark.parametrize(
    ("bucket", "complete", "expected_review"),
    [
        (CandidateBucket.EXACT_SOURCE_ANCHOR, True, None),
        (CandidateBucket.SAME_DOC_LINEAGE, False, ReviewCase.MANDATORY_INCOMPLETE),
    ],
)
def test_mandatory_bucket_completeness_controls_destructive_actions(
    bucket: CandidateBucket,
    complete: bool,
    expected_review: ReviewCase | None,
) -> None:
    service = MemoryRelationApplyService()
    decision = RelationDecision(
        candidate_memory_id="mem-1",
        relation_type=RelationType.EQUIVALENT,
        authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
        confidence=0.9,
        matched_bucket=bucket,
        matched_bucket_complete=complete,
    )

    result = service.derive_lifecycle(_unit(), [decision])

    if expected_review is None:
        assert result.action is LifecycleAction.SUPERSEDE_MEMORY
    else:
        assert result.action is LifecycleAction.CREATE_REVIEW
        assert result.review_case is expected_review


def test_non_destructive_incomplete_bucket_does_not_block_destructive_match() -> None:
    service = MemoryRelationApplyService()
    result = service.derive_lifecycle(
        _unit(),
        [
            RelationDecision(
                candidate_memory_id="mem-authoritative",
                relation_type=RelationType.EQUIVALENT,
                authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
                confidence=0.94,
                matched_bucket=CandidateBucket.SAME_DOC_LINEAGE,
                matched_bucket_complete=True,
            ),
            RelationDecision(
                candidate_memory_id="mem-recall-aid",
                relation_type=RelationType.SUPPORTS,
                authority_case=AuthorityCase.INDEPENDENT_SUPPORT,
                confidence=0.52,
                matched_bucket=CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS,
                matched_bucket_complete=False,
            ),
        ],
    )

    assert result.action is LifecycleAction.SUPERSEDE_MEMORY
    assert result.target_memory_id == "mem-authoritative"


def test_candidate_universe_keeps_mandatory_candidates_beyond_recall_cap() -> None:
    mandatory = CandidateBucketResult(
        bucket=CandidateBucket.SAME_DOC_LINEAGE,
        bucket_rank=1,
        complete=True,
        candidates=tuple(
            _candidate(memory_id=f"mem-mandatory-{index}") for index in range(3)
        ),
    )
    recall = CandidateBucketResult(
        bucket=CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS,
        bucket_rank=6,
        complete=False,
        candidates=tuple(_candidate(memory_id=f"mem-recall-{index}") for index in range(4)),
        scores={f"mem-recall-{index}": 0.9 - index / 10 for index in range(4)},
    )

    universe = build_candidate_universe(
        relation_run_id="rel-run-1",
        evidence_unit_id="eu-1",
        bucket_results=(mandatory, recall),
        recall_candidate_cap=2,
    )

    assert [candidate.memory_id for candidate in universe.candidates] == [
        "mem-mandatory-0",
        "mem-mandatory-1",
        "mem-mandatory-2",
        "mem-recall-0",
        "mem-recall-1",
    ]
    assert universe.total_unique_candidates == 7
    assert universe.checked_candidate_count == 5
    assert universe.mandatory_candidate_count == 3


def test_candidate_universe_records_incomplete_mandatory_buckets() -> None:
    universe = build_candidate_universe(
        relation_run_id="rel-run-1",
        evidence_unit_id="eu-1",
        bucket_results=(
            CandidateBucketResult(
                bucket=CandidateBucket.EXACT_SOURCE_ANCHOR,
                bucket_rank=0,
                complete=False,
                candidates=(_candidate(memory_id="mem-anchor"),),
            ),
        ),
    )

    assert universe.incomplete_mandatory_buckets == ("exact_source_anchor",)
    assert universe.candidates[0].bucket_complete is False
    assert universe.candidates[0].is_mandatory is True


def test_candidate_universe_deduplicates_by_first_bucket_rank() -> None:
    universe = build_candidate_universe(
        relation_run_id="rel-run-1",
        evidence_unit_id="eu-1",
        bucket_results=(
            CandidateBucketResult(
                bucket=CandidateBucket.SEMANTIC_VECTOR_NEIGHBORS,
                bucket_rank=6,
                complete=True,
                candidates=(_candidate(memory_id="mem-1"),),
                scores={"mem-1": 0.88},
            ),
            CandidateBucketResult(
                bucket=CandidateBucket.SAME_DOC_LINEAGE,
                bucket_rank=1,
                complete=True,
                candidates=(_candidate(memory_id="mem-1"),),
            ),
        ),
    )

    assert len(universe.candidates) == 1
    assert universe.candidates[0].bucket is CandidateBucket.SAME_DOC_LINEAGE
    assert universe.candidates[0].bucket_rank == 1
    assert universe.candidates[0].score is None


@pytest.mark.asyncio
async def test_mandatory_candidate_bucket_loader_uses_complete_seed_order() -> None:
    class Store:
        async def get_candidate_memories_by_source_anchor(self, *, source_id: str, source_anchor: str):
            assert (source_id, source_anchor) == ("src-1", "anchor-1")
            return [_candidate(memory_id="mem-anchor", source_anchor=source_anchor)]

        async def get_candidate_memories_by_source_doc(self, *, doc_id: str, support_kind: str | None = None):
            assert (doc_id, support_kind) == ("doc-1", None)
            return [_candidate(memory_id="mem-anchor"), _candidate(memory_id="mem-doc")]

        async def get_candidate_memories_by_agent_claim(self, *, claim_anchor: str):
            assert claim_anchor == "u:repo:concept:claim"
            return [_candidate(memory_id="mem-claim", source_anchor=claim_anchor)]

        async def get_candidate_memories_by_existing_relation_graph(self, *, evidence_unit_id: str):
            assert evidence_unit_id == "eu-1"
            return [_candidate(memory_id="mem-existing")]

    buckets = await build_mandatory_candidate_bucket_results(
        store=Store(),
        unit=_unit(
            source_metadata={"claim_anchor": "u:repo:concept:claim"},
        ),
        access_context=_access(),
    )
    universe = build_candidate_universe(
        relation_run_id="rel-run-1",
        evidence_unit_id="eu-1",
        bucket_results=buckets,
    )

    assert [bucket.bucket for bucket in buckets] == [
        CandidateBucket.EXACT_SOURCE_ANCHOR,
        CandidateBucket.SAME_DOC_LINEAGE,
        CandidateBucket.SAME_AGENT_CLAIM,
        CandidateBucket.EXISTING_RELATION_GRAPH,
    ]
    assert all(bucket.complete for bucket in buckets)
    assert [candidate.memory_id for candidate in universe.candidates] == [
        "mem-anchor",
        "mem-doc",
        "mem-claim",
        "mem-existing",
    ]
    assert universe.mandatory_candidate_count == 4


@pytest.mark.asyncio
async def test_mandatory_candidate_bucket_loader_filters_cross_scope_candidates() -> None:
    class Store:
        async def get_candidate_memories_by_source_anchor(self, *, source_id: str, source_anchor: str):
            return [
                _candidate(memory_id="mem-visible"),
                _candidate(memory_id="mem-unsubscribed", source_id="src-other"),
                _candidate(
                    memory_id="mem-private-other",
                    visibility="private",
                    owner_user_id="someone.else@example.com",
                ),
            ]

        async def get_candidate_memories_by_source_doc(self, *, doc_id: str, support_kind: str | None = None):
            return []

        async def get_candidate_memories_by_agent_claim(self, *, claim_anchor: str):
            return []

        async def get_candidate_memories_by_existing_relation_graph(self, *, evidence_unit_id: str):
            return []

    buckets = await build_mandatory_candidate_bucket_results(
        store=Store(),
        unit=_unit(),
        access_context=_access(source_subscriptions=("src-1",)),
    )

    assert [candidate.memory_id for candidate in buckets[0].candidates] == ["mem-visible"]
