from __future__ import annotations

from dataclasses import replace

import pytest

from memforge.memory.evidence import (
    AuthorityCase,
    CandidateMemory,
    EvidenceContentProvenance,
    EvidenceUnit,
    RelationDirection,
)
from memforge.memory.relation_candidate_retrieval import (
    CrossDocumentCandidateSelection,
    RetrievedRelationCandidate,
)
from memforge.memory.relation_classifier import (
    MemoryPairClassification,
    MemoryPairClassificationError,
    MemoryPairClassificationPlan,
    MemoryPairDecision,
    MemoryRelationType,
)
from memforge.memory.relation_discovery import RelationDiscovery, RelationDiscoveryBudget
from memforge.memory.lifecycle_planner import lifecycle_access_context_hash
from memforge.memory.relation_discovery_contract import (
    PreclassifiedRelationDecision,
    RelationDiscoveryRequest,
    RelationDiscoveryWork,
    RelationDiscoveryWorkStatus,
)
from memforge.models import Memory, MemoryStatus, content_hash
from memforge.storage.adapters.protocols import (
    ActiveMemorySupportState,
    active_support_rows_hash,
)


def _memory(memory_id: str, content: str) -> Memory:
    return Memory(
        id=memory_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        status=MemoryStatus.ACTIVE.value,
    )


class _Classifier:
    def __init__(self) -> None:
        self.classified_pair_ids: tuple[str, ...] = ()

    def plan(self, pairs):
        return MemoryPairClassificationPlan(
            pair_count=len(pairs),
            llm_calls=1 if pairs else 0,
            prompt_chars=10 * len(pairs),
        )

    async def classify(self, pairs):
        self.classified_pair_ids = tuple(pair.candidate.id for pair in pairs)
        return MemoryPairClassification(
            decisions=tuple(
                MemoryPairDecision(
                    pair=pair,
                    relation_type=MemoryRelationType.UNRELATED,
                    direction=RelationDirection.SYMMETRIC,
                    reason="deterministic fixture",
                )
                for pair in pairs
            ),
            llm_calls=1 if pairs else 0,
            prompt_chars=10 * len(pairs),
        )


class _Candidates:
    def __init__(self, candidates: tuple[Memory, ...]) -> None:
        self._candidates = candidates
        self.actor_user_id = None
        self.excluded_source_ids = ()

    async def retrieve(self, **kwargs):
        self.actor_user_id = kwargs["actor_user_id"]
        self.excluded_source_ids = tuple(kwargs["excluded_source_ids"])
        return CrossDocumentCandidateSelection(
            discovery=tuple(
                RetrievedRelationCandidate(
                    memory=CandidateMemory(
                        memory_id=candidate.id,
                        source_id=f"src-{index}",
                        doc_id=f"doc-{index}",
                        source_lineage_id=f"doc-{index}",
                        visibility=candidate.visibility,
                        owner_user_id=candidate.owner_user_id,
                        repo_identifier=candidate.repo_identifier,
                    ),
                    score=1.0 / (index + 1),
                    channels=("lexical_bm25",),
                )
                for index, candidate in enumerate(self._candidates)
            ),
            audit={"candidate_count_kind": "windowed"},
        )

    async def load_selected_memories(self, selection, **_kwargs):
        return selection, {candidate.id: candidate for candidate in self._candidates}

    async def ensure_selection_current(self, *_args, **_kwargs):
        return None


class _Store:
    def __init__(self, challenger: Memory, candidates: tuple[Memory, ...]) -> None:
        self.challenger = challenger
        self.candidates = candidates
        self.leased = False
        self.completed = None
        self.reviews = ()
        self.disabled_lookup_user_id = None
        self.work = RelationDiscoveryWork(
            request=RelationDiscoveryRequest(
                id="work-1",
                memory_id=challenger.id,
                expected_content_hash=challenger.content_hash,
                source_id="src-challenger",
                source_unit_id="unit-1",
                source_unit_revision_id="unit-revision-1",
                doc_id="doc-challenger",
                actor_user_id=None,
            ),
            lifecycle_plan_id="plan-1",
            status=RelationDiscoveryWorkStatus.RUNNING,
            attempts=1,
            lease_owner="worker-1",
            lease_token="token-1",
        )

    async def lease_relation_discovery_work(self, **_kwargs):
        if self.leased:
            return []
        self.leased = True
        return [self.work]

    async def get_memory(self, memory_id):
        return self.challenger if memory_id == self.challenger.id else None

    async def get_current_relation_evidence_unit(self, *_args, **_kwargs):
        return EvidenceUnit(
            id="evidence-1",
            source_id="src-challenger",
            doc_id="doc-challenger",
            doc_revision_id="unit-revision-1",
            source_type="github_repo",
            source_anchor=None,
            source_lineage_id="unit-1",
            project_key=self.challenger.project_key,
            visibility=self.challenger.visibility,
            owner_user_id=self.challenger.owner_user_id,
            repo_identifier=self.challenger.repo_identifier,
            content=self.challenger.content,
            excerpt=None,
            evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
        )

    async def list_disabled_source_ids_for_user(self, user_id):
        self.disabled_lookup_user_id = user_id
        return []

    async def get_memory_entity_ids(self, _memory_id):
        return []

    async def get_active_memory_support_states(self, memory_ids):
        return {
            memory_id: ActiveMemorySupportState(
                reference_ids=(),
                support_set_hash=active_support_rows_hash(()),
                current_reference_ids=(),
                current_support_set_hash=active_support_rows_hash(()),
            )
            for memory_id in memory_ids
        }

    async def complete_relation_discovery_work(self, _work_id, **kwargs):
        self.completed = kwargs["relation_outcome"]
        self.reviews = kwargs["reviews"]

    async def fail_relation_discovery_work(self, *_args, **_kwargs):
        pytest.fail("work should not fail")

    async def obsolete_relation_discovery_work(self, *_args, **_kwargs):
        pytest.fail("work should not become obsolete")


class _FailingStore(_Store):
    def __init__(self, challenger: Memory, candidates: tuple[Memory, ...]) -> None:
        super().__init__(challenger, candidates)
        self.failure = None

    async def fail_relation_discovery_work(self, *_args, **kwargs):
        self.failure = kwargs


class _CompletionFailingStore(_FailingStore):
    def __init__(self, challenger: Memory, candidates: tuple[Memory, ...]) -> None:
        super().__init__(challenger, candidates)
        self.lease_calls = 0

    async def lease_relation_discovery_work(self, **_kwargs):
        self.lease_calls += 1
        return [self.work]

    async def complete_relation_discovery_work(self, *_args, **_kwargs):
        raise ValueError("completion currentness guard rejected the result")


class _UsageReportingFailureClassifier(_Classifier):
    async def classify(self, pairs):
        raise MemoryPairClassificationError(
            "second classifier batch failed",
            pair_count=len(pairs),
            llm_calls=2,
            prompt_chars=321,
        )


class _ConflictClassifier(_Classifier):
    async def classify(self, pairs):
        return MemoryPairClassification(
            decisions=tuple(
                MemoryPairDecision(
                    pair=pair,
                    relation_type=MemoryRelationType.CONTRADICTS,
                    direction=RelationDirection.SYMMETRIC,
                    reason="deterministic conflict",
                )
                for pair in pairs
            ),
            llm_calls=1 if pairs else 0,
            prompt_chars=10 * len(pairs),
        )


@pytest.mark.asyncio
async def test_relation_discovery_finishes_one_selected_ledger_before_slice_budget_stops() -> None:
    challenger = _memory("challenger", "Current claim")
    candidates = tuple(_memory(f"candidate-{index}", f"Candidate {index}") for index in range(3))
    store = _Store(challenger, candidates)

    result = await RelationDiscovery(
        store=store,  # type: ignore[arg-type]
        candidate_retriever=_Candidates(candidates),  # type: ignore[arg-type]
        pair_classifier=_Classifier(),
    ).process_slice(
        worker_id="worker-1",
        budget=RelationDiscoveryBudget(max_candidate_pairs=1, max_llm_calls=1),
    )

    assert result.completed_work == 1
    assert result.checked_candidate_pairs == 3
    assert store.completed is not None
    assert store.completed.relation_run.result_memory_id == challenger.id
    assert {item.memory_id for item in store.completed.candidates} == {
        "candidate-0",
        "candidate-1",
        "candidate-2",
    }


@pytest.mark.asyncio
async def test_relation_discovery_reuses_current_identity_pair_and_only_classifies_new_candidates() -> None:
    challenger = _memory("challenger", "Current claim")
    candidates = (
        _memory("candidate-reused", "Previously classified claim"),
        _memory("candidate-new", "Newly recalled claim"),
    )
    store = _Store(challenger, candidates)
    store.work = replace(
        store.work,
        request=replace(
            store.work.request,
            preclassified_decisions=(
                PreclassifiedRelationDecision(
                    candidate_memory_id=candidates[0].id,
                    expected_candidate_content_hash=candidates[0].content_hash,
                    expected_candidate_support_set_hash=active_support_rows_hash(()),
                    expected_candidate_access_context_hash=lifecycle_access_context_hash(
                        visibility=candidates[0].visibility,
                        owner_user_id=candidates[0].owner_user_id,
                        project_key=candidates[0].project_key,
                        repo_identifier=candidates[0].repo_identifier,
                    ),
                    expected_challenger_access_context_hash=lifecycle_access_context_hash(
                        visibility=challenger.visibility,
                        owner_user_id=challenger.owner_user_id,
                        project_key=challenger.project_key,
                        repo_identifier=challenger.repo_identifier,
                    ),
                    relation_type=MemoryRelationType.UNRELATED,
                    direction=RelationDirection.SYMMETRIC,
                    reason="identity stage already checked this pair",
                    classifier_version="memory-relation-v1",
                ),
            ),
        ),
    )
    classifier = _Classifier()

    result = await RelationDiscovery(
        store=store,  # type: ignore[arg-type]
        candidate_retriever=_Candidates(candidates),  # type: ignore[arg-type]
        pair_classifier=classifier,
    ).process_slice(worker_id="worker-1")

    assert result.checked_candidate_pairs == 2
    assert result.reused_candidate_pairs == 1
    assert result.llm_calls == 1
    assert classifier.classified_pair_ids == ("candidate-new",)
    assert store.completed is not None
    assert store.completed.relation_run.audit["reused_identity_pair_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stale_field",
    ("expected_candidate_support_set_hash", "expected_candidate_access_context_hash"),
)
async def test_relation_discovery_reclassifies_identity_seed_with_stale_context(
    stale_field: str,
) -> None:
    challenger = _memory("challenger", "Current claim")
    candidate = _memory("candidate", "Previously classified claim")
    store = _Store(challenger, (candidate,))
    seed = PreclassifiedRelationDecision(
        candidate_memory_id=candidate.id,
        expected_candidate_content_hash=candidate.content_hash,
        expected_candidate_support_set_hash=active_support_rows_hash(()),
        expected_candidate_access_context_hash=lifecycle_access_context_hash(
            visibility=candidate.visibility,
            owner_user_id=candidate.owner_user_id,
            project_key=candidate.project_key,
            repo_identifier=candidate.repo_identifier,
        ),
        expected_challenger_access_context_hash=lifecycle_access_context_hash(
            visibility=challenger.visibility,
            owner_user_id=challenger.owner_user_id,
            project_key=challenger.project_key,
            repo_identifier=challenger.repo_identifier,
        ),
        relation_type=MemoryRelationType.UNRELATED,
        direction=RelationDirection.SYMMETRIC,
        reason="identity stage already checked this pair",
        classifier_version="memory-relation-v1",
    )
    store.work = replace(
        store.work,
        request=replace(
            store.work.request,
            preclassified_decisions=(replace(seed, **{stale_field: "stale"}),),
        ),
    )
    classifier = _Classifier()

    result = await RelationDiscovery(
        store=store,  # type: ignore[arg-type]
        candidate_retriever=_Candidates((candidate,)),  # type: ignore[arg-type]
        pair_classifier=classifier,
    ).process_slice(worker_id="worker-1")

    assert result.reused_candidate_pairs == 0
    assert classifier.classified_pair_ids == ("candidate",)
    assert store.completed is not None
    assert store.completed.relation_run.audit["reused_identity_pair_count"] == 0


@pytest.mark.asyncio
async def test_failed_classification_usage_counts_against_slice_budget() -> None:
    challenger = _memory("challenger", "Current claim")
    candidates = tuple(_memory(f"candidate-{index}", f"Candidate {index}") for index in range(3))
    store = _FailingStore(challenger, candidates)

    result = await RelationDiscovery(
        store=store,  # type: ignore[arg-type]
        candidate_retriever=_Candidates(candidates),  # type: ignore[arg-type]
        pair_classifier=_UsageReportingFailureClassifier(),
    ).process_slice(
        worker_id="worker-1",
        budget=RelationDiscoveryBudget(max_candidate_pairs=1, max_llm_calls=1),
    )

    assert result.failed_work == 1
    assert result.checked_candidate_pairs == 3
    assert result.llm_calls == 2
    assert result.prompt_chars == 321
    assert store.failure is not None


@pytest.mark.asyncio
async def test_completion_guard_failure_keeps_classification_usage_in_slice_budget() -> None:
    challenger = _memory("challenger", "Current claim")
    candidates = tuple(_memory(f"candidate-{index}", f"Candidate {index}") for index in range(3))
    store = _CompletionFailingStore(challenger, candidates)

    result = await RelationDiscovery(
        store=store,  # type: ignore[arg-type]
        candidate_retriever=_Candidates(candidates),  # type: ignore[arg-type]
        pair_classifier=_Classifier(),
    ).process_slice(
        worker_id="worker-1",
        budget=RelationDiscoveryBudget(max_candidate_pairs=1, max_llm_calls=1),
    )

    assert result.failed_work == 1
    assert result.checked_candidate_pairs == 3
    assert result.llm_calls == 1
    assert result.prompt_chars == 30
    assert store.failure is not None
    assert store.lease_calls == 1


@pytest.mark.asyncio
async def test_private_relation_discovery_without_explicit_actor_builds_valid_cross_source_review() -> None:
    challenger = replace(
        _memory("challenger", "Current claim"),
        visibility="private",
        owner_user_id="user-1",
        repo_identifier="repo-1",
        project_key="PROJECT",
    )
    incumbent = replace(
        _memory("incumbent", "Conflicting claim"),
        visibility="private",
        owner_user_id="user-1",
        repo_identifier="repo-1",
        project_key="PROJECT",
    )
    store = _Store(challenger, (incumbent,))
    candidates = _Candidates((incumbent,))

    result = await RelationDiscovery(
        store=store,  # type: ignore[arg-type]
        candidate_retriever=candidates,  # type: ignore[arg-type]
        pair_classifier=_ConflictClassifier(),
    ).process_slice(
        worker_id="worker-1",
        budget=RelationDiscoveryBudget(max_candidate_pairs=1, max_llm_calls=1),
    )

    assert result.completed_work == 1
    assert result.failed_work == 0
    assert store.disabled_lookup_user_id == "user-1"
    assert candidates.actor_user_id == "user-1"
    assert store.completed is not None
    assert store.completed.relations[0].authority_case is AuthorityCase.CROSS_SOURCE_CONFLICT
    assert len(store.reviews) == 1
    assert store.reviews[0].status == "pending"
