from __future__ import annotations

from dataclasses import replace

import pytest

from memforge.memory.cross_document_relation import (
    CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
    CrossDocumentRelationClassification,
    CrossDocumentRelationJudgment,
    CrossDocumentRelationLabel,
)
from memforge.memory.evidence import (
    CandidateMemory,
    EvidenceContentProvenance,
    EvidenceUnit,
    LifecycleAction,
)
from memforge.memory.relation_candidate_retrieval import (
    CrossDocumentCandidateSelection,
    RetrievedRelationCandidate,
)
from memforge.memory.relation_classifier import MemoryPairClassificationError
from memforge.memory.relation_discovery import RelationDiscovery, RelationDiscoveryBudget
from memforge.memory.relation_discovery_contract import (
    RelationDiscoveryRequest,
    RelationDiscoveryWork,
    RelationDiscoveryWorkState,
    RelationDiscoveryWorkStatus,
)
from memforge.models import DocumentRecord, Memory, MemoryStatus, content_hash
from memforge.storage.adapters.protocols import (
    ActiveMemorySupportState,
    active_support_rows_hash,
)
from tests.relation_evidence_fixture import primary_evidence_unit_fixture, primary_observation_revision_fixture


def _memory(memory_id: str, content: str) -> Memory:
    return Memory(
        id=memory_id,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content),
        status=MemoryStatus.ACTIVE.value,
    )


class _Classifier:
    """Label every pair none unless its candidate has a label."""

    def __init__(self, labels: dict[str, CrossDocumentRelationLabel] | None = None) -> None:
        self.labels = labels or {}
        self.pairs: tuple = ()

    @property
    def classified_pair_ids(self) -> tuple[str, ...]:
        return tuple(pair.candidate.memory_id for pair in self.pairs)

    async def classify(self, pairs):
        self.pairs = pairs
        return CrossDocumentRelationClassification(
            judgments=tuple(
                CrossDocumentRelationJudgment(
                    pair=pair,
                    label=self.labels.get(pair.candidate.memory_id, CrossDocumentRelationLabel.NONE),
                    reason="deterministic fixture",
                )
                for pair in pairs
            ),
            llm_calls=1 if pairs else 0,
            prompt_chars=10 * len(pairs),
        )


class _Candidates:
    def __init__(
        self,
        candidates: tuple[Memory, ...],
        *,
        source_ids: tuple[str, ...] | None = None,
    ) -> None:
        self._candidates = candidates
        self._source_ids = source_ids or tuple(f"src-{index}" for index in range(len(candidates)))
        if len(self._source_ids) != len(candidates):
            raise ValueError("source_ids must align with candidates")
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
                        source_id=self._source_ids[index],
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
        self.completed_run = None
        self.completion_kwargs: dict | None = None
        self.disabled_lookup_user_id = None
        self.lease_kwargs = None
        self.exhausted_selection = None
        # Source time of each Memory's Primary Evidence; the fixture time otherwise.
        self.observed_at: dict[str, str] = {}
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

    async def lease_relation_discovery_work(self, **kwargs):
        self.lease_kwargs = kwargs
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

    async def get_memory_evidence_units(self, memory_id):
        return (primary_evidence_unit_fixture(memory_id),)

    async def get_current_source_observation_revisions(self, source_unit_id):
        memory_id = source_unit_id.removeprefix("unit-")
        revision = primary_observation_revision_fixture(memory_id)
        if memory_id in self.observed_at:
            revision = replace(revision, observed_at=self.observed_at[memory_id])
        return {f"obs-{memory_id}": revision}

    async def get_document(self, doc_id):
        return DocumentRecord(
            doc_id=doc_id,
            source="jira",
            source_url="",
            title=f"Title of {doc_id}",
            space_or_project="",
            author=None,
            last_modified=None,  # type: ignore[arg-type]
            labels=[],
            version="1",
            content_hash="",
            token_count=None,
            raw_content_uri=None,
            raw_content_type=None,
            normalized_content_uri=None,
            pdf_content_uri=None,
            last_synced=None,  # type: ignore[arg-type]
        )

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
        self.completion_kwargs = kwargs
        self.completed_run = kwargs["relation_run"]
        self.completed = kwargs["document_relations"]

    async def fail_relation_discovery_work(self, *_args, **_kwargs):
        pytest.fail("work should not fail")

    async def obsolete_relation_discovery_work(self, *_args, **_kwargs):
        pytest.fail("work should not become obsolete")

    async def count_relation_discovery_work(self, selection):
        self.exhausted_selection = selection
        return 7


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
    def __init__(self, error_code: str | None = "output_invalid") -> None:
        super().__init__()
        self.error_code = error_code

    async def classify(self, pairs):
        raise MemoryPairClassificationError(
            "second classifier batch failed",
            pair_count=len(pairs),
            llm_calls=2,
            prompt_chars=321,
            error_code=self.error_code,
        )


def _discovery(store: _Store, candidates: tuple[Memory, ...], classifier: _Classifier, **kwargs) -> RelationDiscovery:
    return RelationDiscovery(
        store=store,  # type: ignore[arg-type]
        candidate_retriever=_Candidates(candidates, **kwargs),  # type: ignore[arg-type]
        pair_classifier=classifier,
    )


@pytest.mark.asyncio
async def test_relation_discovery_finishes_one_selected_ledger_before_slice_budget_stops() -> None:
    challenger = _memory("challenger", "Current claim")
    candidates = tuple(_memory(f"candidate-{index}", f"Candidate {index}") for index in range(3))
    store = _Store(challenger, candidates)

    result = await _discovery(store, candidates, _Classifier()).process_slice(
        worker_id="worker-1",
        budget=RelationDiscoveryBudget(max_candidate_pairs=1, max_llm_calls=1),
    )

    assert result.completed_work == 1
    assert result.checked_candidate_pairs == 3
    assert store.completed_run is not None
    assert store.completed_run.relation_run.result_memory_id == challenger.id
    assert {item.memory_id for item in store.completed_run.candidates} == {
        "candidate-0",
        "candidate-1",
        "candidate-2",
    }


@pytest.mark.asyncio
async def test_relation_discovery_can_scope_leases_to_one_source() -> None:
    challenger = _memory("challenger", "Current claim")
    store = _Store(challenger, ())

    await _discovery(store, (), _Classifier()).process_slice(
        worker_id="controlled-recovery",
        source_id="src-challenger",
    )

    assert store.lease_kwargs is not None
    assert store.lease_kwargs["source_id"] == "src-challenger"


@pytest.mark.asyncio
async def test_relation_discovery_records_one_relation_per_labeled_pair_and_no_review() -> None:
    challenger = _memory("mem-m", "Current claim")
    candidates = (
        _memory("mem-a", "Unrelated claim"),
        _memory("mem-b", "Same claim elsewhere"),
        _memory("mem-y", "Earlier value"),
        _memory("mem-z", "Opposite claim"),
    )
    store = _Store(challenger, candidates)
    store.observed_at = {"mem-m": "2026-04-02T09:00:00+00:00", "mem-y": "2026-03-01T09:00:00+00:00"}
    classifier = _Classifier(
        {
            "mem-b": CrossDocumentRelationLabel.EQUIVALENT,
            "mem-y": CrossDocumentRelationLabel.UPDATES,
            "mem-z": CrossDocumentRelationLabel.CONTRADICTS,
        }
    )

    result = await _discovery(store, candidates, classifier).process_slice(worker_id="worker-1")

    assert result.completed_work == 1
    assert store.completion_kwargs is not None
    assert set(store.completion_kwargs) == {"worker_id", "lease_token", "relation_run", "document_relations"}
    bundle = store.completed_run
    assert bundle.relations == ()
    outcome = store.completed
    assert outcome.challenger_id == challenger.id
    assert outcome.challenger_content_hash == challenger.content_hash
    assert outcome.judged_content_hashes == {candidate.id: candidate.content_hash for candidate in candidates}
    by_pair = {(record.memory_low_id, record.memory_high_id): record for record in outcome.relations}
    assert set(by_pair) == {("mem-b", "mem-m"), ("mem-m", "mem-y"), ("mem-m", "mem-z")}
    assert by_pair["mem-b", "mem-m"].label is CrossDocumentRelationLabel.EQUIVALENT
    assert by_pair["mem-b", "mem-m"].low_content_hash == candidates[1].content_hash
    assert by_pair["mem-b", "mem-m"].high_content_hash == challenger.content_hash
    assert by_pair["mem-m", "mem-y"].label is CrossDocumentRelationLabel.UPDATES
    assert (by_pair["mem-m", "mem-y"].low_evidence_time, by_pair["mem-m", "mem-y"].high_evidence_time) == (
        "2026-04-02",
        "2026-03-01",
    )
    assert by_pair["mem-m", "mem-z"].label is CrossDocumentRelationLabel.CONTRADICTS
    for record in outcome.relations:
        assert record.classifier_version == CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION
        assert record.relation_run_id == bundle.relation_run.id
        assert record.discovery_work_id == "work-1"
    run = bundle.relation_run
    assert run.lifecycle_action is LifecycleAction.NONE
    assert run.review_case is None
    assert run.status == "checked"
    assert run.classifier_version == CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION
    assert run.audit["labels"] == {
        "mem-a": "none",
        "mem-b": "equivalent",
        "mem-y": "updates",
        "mem-z": "contradicts",
    }


@pytest.mark.asyncio
async def test_relation_discovery_records_updates_as_contradicts_when_evidence_times_do_not_order_the_pair() -> None:
    challenger = _memory("mem-m", "Current claim")
    same_day = _memory("mem-s", "Claim recorded the same day")
    untimed = _memory("mem-u", "Claim without a source time")
    store = _Store(challenger, (same_day, untimed))
    store.observed_at = {"mem-u": ""}

    await _discovery(
        store,
        (same_day, untimed),
        _Classifier({"mem-s": CrossDocumentRelationLabel.UPDATES, "mem-u": CrossDocumentRelationLabel.UPDATES}),
    ).process_slice(worker_id="worker-1")

    assert [record.label for record in store.completed.relations] == [
        CrossDocumentRelationLabel.CONTRADICTS,
        CrossDocumentRelationLabel.CONTRADICTS,
    ]
    assert store.completed_run.relation_run.audit["labels"] == {"mem-s": "updates", "mem-u": "updates"}


@pytest.mark.asyncio
async def test_relation_discovery_judges_other_units_of_the_same_source() -> None:
    challenger = _memory("challenger", "Current claim")
    sibling = _memory("sibling", "Conflicting sibling ticket claim")
    store = _Store(challenger, (sibling,))

    await _discovery(
        store,
        (sibling,),
        _Classifier({"sibling": CrossDocumentRelationLabel.CONTRADICTS}),
        source_ids=("src-challenger",),
    ).process_slice(worker_id="worker-1")

    assert [record.label for record in store.completed.relations] == [
        CrossDocumentRelationLabel.CONTRADICTS
    ]


@pytest.mark.asyncio
async def test_relation_discovery_shows_the_classifier_statements_titles_time_and_evidence() -> None:
    challenger = _memory("challenger", "Current claim")
    candidate = _memory("candidate", "Other claim")
    store = _Store(challenger, (candidate,))
    classifier = _Classifier()

    await _discovery(store, (candidate,), classifier).process_slice(worker_id="worker-1")

    (pair,) = classifier.pairs
    assert pair.challenger.statement == "Current claim"
    assert pair.candidate.document_title == "Title of doc-candidate"
    assert pair.candidate.evidence == ("Evidence for candidate.",)
    assert pair.candidate.evidence_time == "2026-03-25"
    assert pair.candidate.source_type == "jira"


@pytest.mark.asyncio
async def test_relation_discovery_run_id_changes_with_each_operator_rerun() -> None:
    challenger = _memory("challenger", "Current claim")
    candidate = _memory("candidate", "Other claim")
    first = _Store(challenger, (candidate,))
    rerun = _Store(challenger, (candidate,))
    rerun.work = replace(rerun.work, run_generation=1)

    await _discovery(first, (candidate,), _Classifier()).process_slice(worker_id="worker-1")
    await _discovery(rerun, (candidate,), _Classifier()).process_slice(worker_id="worker-1")

    assert first.completed_run.relation_run.id != rerun.completed_run.relation_run.id
    assert rerun.completed_run.relation_run.audit["run_generation"] == 1


@pytest.mark.asyncio
async def test_relation_discovery_reports_exhausted_work_after_attempting_work() -> None:
    challenger = _memory("challenger", "Current claim")
    store = _Store(challenger, ())

    result = await _discovery(store, (), _Classifier()).process_slice(
        worker_id="worker-1",
        budget=RelationDiscoveryBudget(max_attempts=3),
    )

    assert result.exhausted_work == 7
    assert store.exhausted_selection.state is RelationDiscoveryWorkState.EXHAUSTED
    assert store.exhausted_selection.max_attempts == 3


@pytest.mark.asyncio
async def test_idle_relation_discovery_slice_does_not_count_exhausted_work() -> None:
    challenger = _memory("challenger", "Current claim")
    store = _Store(challenger, ())
    store.leased = True

    result = await _discovery(store, (), _Classifier()).process_slice(worker_id="worker-1")

    assert result.attempted_work == 0
    assert result.exhausted_work == 0
    assert store.exhausted_selection is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error_code", "recorded_code"),
    [("output_invalid", "output_invalid"), (None, "MemoryPairClassificationError")],
)
async def test_failed_classification_usage_counts_against_slice_budget(
    error_code: str | None, recorded_code: str
) -> None:
    challenger = _memory("challenger", "Current claim")
    candidates = tuple(_memory(f"candidate-{index}", f"Candidate {index}") for index in range(3))
    store = _FailingStore(challenger, candidates)

    result = await _discovery(store, candidates, _UsageReportingFailureClassifier(error_code)).process_slice(
        worker_id="worker-1",
        budget=RelationDiscoveryBudget(max_candidate_pairs=1, max_llm_calls=1),
    )

    assert result.failed_work == 1
    assert result.checked_candidate_pairs == 3
    assert result.llm_calls == 2
    assert result.prompt_chars == 321
    assert store.failure is not None
    assert store.failure["error_code"] == recorded_code
    assert store.failure["error"].startswith("MemoryPairClassificationError: ")


@pytest.mark.asyncio
async def test_completion_guard_failure_keeps_classification_usage_in_slice_budget() -> None:
    challenger = _memory("challenger", "Current claim")
    candidates = tuple(_memory(f"candidate-{index}", f"Candidate {index}") for index in range(3))
    store = _CompletionFailingStore(challenger, candidates)

    result = await _discovery(store, candidates, _Classifier()).process_slice(
        worker_id="worker-1",
        budget=RelationDiscoveryBudget(max_candidate_pairs=1, max_llm_calls=1),
    )

    assert result.failed_work == 1
    assert result.checked_candidate_pairs == 3
    assert result.llm_calls == 1
    assert result.prompt_chars == 30
    assert store.failure is not None
    assert store.failure["error_code"] == "ValueError"
    assert store.lease_calls == 1


@pytest.mark.asyncio
async def test_private_relation_discovery_without_explicit_actor_runs_as_the_owner() -> None:
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
        pair_classifier=_Classifier({"incumbent": CrossDocumentRelationLabel.CONTRADICTS}),
    ).process_slice(worker_id="worker-1")

    assert result.completed_work == 1
    assert store.disabled_lookup_user_id == "user-1"
    assert candidates.actor_user_id == "user-1"
    assert [record.label for record in store.completed.relations] == [
        CrossDocumentRelationLabel.CONTRADICTS
    ]
