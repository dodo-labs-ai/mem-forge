"""Budgeted post-commit discovery of cross-document Memory relations.

Discovery labels each (challenger, candidate) pair and writes relations only.
It creates no Review and changes neither Memory.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from memforge.memory.cross_document_relation import (
    CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
    CrossDocumentRelationClassification,
    CrossDocumentRelationClassifier,
    CrossDocumentRelationOutcome,
    CrossDocumentRelationPair,
    load_relation_subjects,
)
from memforge.memory.evidence import (
    LifecycleAction,
    RelationOutcomeBundle,
    RelationRunRecord,
    build_candidate_universe,
)
from memforge.memory.relation_candidate_retrieval import (
    CrossDocumentCandidateRetriever,
    CrossDocumentCandidateSelection,
)
from memforge.memory.relation_classifier import MemoryPairClassificationError
from memforge.memory.relation_discovery_contract import (
    RelationDiscoveryWork,
    RelationDiscoveryWorkSelection,
    RelationDiscoveryWorkState,
    resolve_relation_discovery_actor_user_id,
)
from memforge.models import Memory, MemoryStatus
from memforge.storage.adapters.protocols import RelationalStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RelationDiscoveryBudget:
    max_work_items: int = 4
    max_candidate_pairs: int = 512
    max_llm_calls: int = 16
    max_wall_time_seconds: float = 60.0
    lease_seconds: int = 300
    max_attempts: int = 5
    retry_base_seconds: int = 30
    retry_max_seconds: int = 900

    def __post_init__(self) -> None:
        values = (
            self.max_work_items,
            self.max_candidate_pairs,
            self.max_llm_calls,
            self.lease_seconds,
            self.max_attempts,
            self.retry_base_seconds,
            self.retry_max_seconds,
        )
        if any(value < 1 for value in values) or self.max_wall_time_seconds <= 0:
            raise ValueError("relation discovery budget values must be positive")


DEFAULT_RELATION_DISCOVERY_BUDGET = RelationDiscoveryBudget()


@dataclass(frozen=True, slots=True)
class RelationDiscoverySliceResult:
    attempted_work: int = 0
    completed_work: int = 0
    failed_work: int = 0
    obsolete_work: int = 0
    checked_candidate_pairs: int = 0
    llm_calls: int = 0
    prompt_chars: int = 0
    elapsed_ms: int = 0
    # Work that used every attempt, counted across the store after a slice that
    # attempted work.
    exhausted_work: int = 0


class RelationDiscovery:
    """Advance durable relation work without owning Memory lifecycle authority."""

    def __init__(
        self,
        *,
        store: RelationalStore,
        candidate_retriever: CrossDocumentCandidateRetriever,
        pair_classifier: CrossDocumentRelationClassifier,
    ) -> None:
        self._store = store
        self._candidate_retriever = candidate_retriever
        self._pair_classifier = pair_classifier

    async def process_slice(
        self,
        *,
        worker_id: str,
        budget: RelationDiscoveryBudget | None = None,
        source_id: str | None = None,
    ) -> RelationDiscoverySliceResult:
        policy = budget or DEFAULT_RELATION_DISCOVERY_BUDGET
        started = time.perf_counter()
        attempted = completed = failed = obsolete = 0
        checked_pairs = llm_calls = prompt_chars = 0

        while attempted < policy.max_work_items:
            if time.perf_counter() - started >= policy.max_wall_time_seconds:
                break
            if checked_pairs >= policy.max_candidate_pairs or llm_calls >= policy.max_llm_calls:
                break
            leased = await self._store.lease_relation_discovery_work(
                worker_id=worker_id,
                limit=1,
                lease_seconds=policy.lease_seconds,
                max_attempts=policy.max_attempts,
                source_id=source_id,
            )
            if not leased:
                break
            work = leased[0]
            attempted += 1
            try:
                outcome = await self._process_work(work)
                if outcome is None:
                    await self._store.obsolete_relation_discovery_work(
                        work.request.id,
                        worker_id=worker_id,
                        lease_token=_lease_token(work),
                        reason="activated Memory or source evidence is no longer current",
                    )
                    obsolete += 1
                    continue
                relation_run, document_relations, classification = outcome
                checked_pairs += classification.pair_count
                llm_calls += classification.llm_calls
                prompt_chars += classification.prompt_chars
                await self._store.complete_relation_discovery_work(
                    work.request.id,
                    worker_id=worker_id,
                    lease_token=_lease_token(work),
                    relation_run=relation_run,
                    document_relations=document_relations,
                )
                completed += 1
            except Exception as error:
                recorded_error = error
                if isinstance(error, _WorkProcessingError):
                    checked_pairs += error.pair_count
                    llm_calls += error.llm_calls
                    prompt_chars += error.prompt_chars
                    recorded_error = error.cause
                logger.exception("Relation discovery work %s failed", work.request.id)
                exhausted = work.attempts >= policy.max_attempts
                retry_at = None
                if not exhausted:
                    exponent = max(0, work.attempts - 1)
                    delay = min(
                        policy.retry_max_seconds,
                        policy.retry_base_seconds * (2**exponent),
                    )
                    retry_at = datetime.now(timezone.utc) + timedelta(seconds=delay)
                await self._store.fail_relation_discovery_work(
                    work.request.id,
                    worker_id=worker_id,
                    lease_token=_lease_token(work),
                    error=f"{type(recorded_error).__name__}: {recorded_error}",
                    error_code=relation_discovery_error_code(recorded_error),
                    next_attempt_at=(retry_at.isoformat() if retry_at is not None else None),
                    exhausted=exhausted,
                )
                failed += 1

        exhausted_work = (
            await self._store.count_relation_discovery_work(
                RelationDiscoveryWorkSelection(
                    state=RelationDiscoveryWorkState.EXHAUSTED,
                    max_attempts=policy.max_attempts,
                )
            )
            if attempted
            else 0
        )
        return RelationDiscoverySliceResult(
            attempted_work=attempted,
            completed_work=completed,
            failed_work=failed,
            obsolete_work=obsolete,
            checked_candidate_pairs=checked_pairs,
            llm_calls=llm_calls,
            prompt_chars=prompt_chars,
            elapsed_ms=max(0, round((time.perf_counter() - started) * 1000)),
            exhausted_work=exhausted_work,
        )

    async def _process_work(
        self,
        work: RelationDiscoveryWork,
    ) -> tuple[RelationOutcomeBundle, CrossDocumentRelationOutcome, _CompletedClassification] | None:
        request = work.request
        challenger = await self._store.get_memory(request.memory_id)
        if (
            challenger is None
            or challenger.status != MemoryStatus.ACTIVE.value
            or challenger.content_hash != request.expected_content_hash
        ):
            return None
        evidence_unit = await self._store.get_current_relation_evidence_unit(
            challenger.id,
            source_id=request.source_id,
            source_unit_id=request.source_unit_id,
        )
        if evidence_unit is None:
            return None

        actor_user_id = resolve_relation_discovery_actor_user_id(
            visibility=challenger.visibility,
            owner_user_id=challenger.owner_user_id,
            requested_actor_user_id=request.actor_user_id,
        )
        disabled_source_ids = (
            await self._store.list_disabled_source_ids_for_user(actor_user_id) if actor_user_id else []
        )
        entity_ids = request.entity_ids or tuple(await self._store.get_memory_entity_ids(challenger.id))
        selection = await self._candidate_retriever.retrieve(
            challenger=challenger,
            entity_ids=entity_ids,
            doc_id=evidence_unit.doc_id or request.doc_id,
            actor_user_id=actor_user_id,
            source_id=request.source_id,
            excluded_source_ids=disabled_source_ids,
        )
        selection, loaded_by_id = await self._candidate_retriever.load_selected_memories(
            selection,
            challenger=challenger,
            doc_id=evidence_unit.doc_id or request.doc_id,
            source_id=request.source_id,
            excluded_source_ids=disabled_source_ids,
        )
        candidates = tuple(loaded_by_id[memory_id] for memory_id in selection.candidate_ids)
        candidate_support = await self._store.get_active_memory_support_states(
            tuple(candidate.id for candidate in candidates)
        )
        subjects = await load_relation_subjects(self._store, (challenger, *candidates))
        pairs = tuple(
            CrossDocumentRelationPair(challenger=subjects[challenger.id], candidate=subjects[candidate.id])
            for candidate in candidates
        )
        try:
            classification = (
                await self._pair_classifier.classify(pairs)
                if pairs
                else CrossDocumentRelationClassification(judgments=(), llm_calls=0, prompt_chars=0)
            )
        except MemoryPairClassificationError as error:
            raise _WorkProcessingError(
                cause=error,
                pair_count=error.pair_count,
                llm_calls=error.llm_calls,
                prompt_chars=error.prompt_chars,
            ) from error
        try:
            candidate_support_set_hashes = {
                candidate.id: candidate_support[candidate.id].current_support_set_hash
                for candidate in candidates
            }
            await self._candidate_retriever.ensure_selection_current(
                selection,
                challenger=challenger,
                doc_id=evidence_unit.doc_id or request.doc_id,
                source_id=request.source_id,
                excluded_source_ids=disabled_source_ids,
                expected_support_set_hashes=candidate_support_set_hashes,
            )
            relation_run = self._build_relation_run(
                work=work,
                challenger=challenger,
                evidence_unit=evidence_unit,
                selection=selection,
                classification=classification,
                candidate_support_set_hashes=candidate_support_set_hashes,
            )
            document_relations = CrossDocumentRelationOutcome.from_judgments(
                subjects[challenger.id],
                classification.judgments,
                relation_run_id=relation_run.relation_run.id,
                discovery_work_id=work.request.id,
            )
        except Exception as error:
            raise _WorkProcessingError(
                cause=error,
                pair_count=len(pairs),
                llm_calls=classification.llm_calls,
                prompt_chars=classification.prompt_chars,
            ) from error
        return (
            relation_run,
            document_relations,
            _CompletedClassification(
                pair_count=len(pairs),
                llm_calls=classification.llm_calls,
                prompt_chars=classification.prompt_chars,
            ),
        )

    @staticmethod
    def _build_relation_run(
        *,
        work: RelationDiscoveryWork,
        challenger: Memory,
        evidence_unit,
        selection: CrossDocumentCandidateSelection,
        classification: CrossDocumentRelationClassification,
        candidate_support_set_hashes: dict[str, str],
    ) -> RelationOutcomeBundle:
        relation_run_id = _relation_run_id(work, selection)
        universe = build_candidate_universe(
            relation_run_id=relation_run_id,
            evidence_unit_id=evidence_unit.id,
            bucket_results=selection.bucket_results(),
            recall_candidate_cap=max(1, len(selection.discovery)),
        )
        now = datetime.now(timezone.utc).isoformat()
        relation_run = RelationRunRecord(
            id=relation_run_id,
            evidence_unit_id=evidence_unit.id,
            access_context_hash=evidence_unit.access_context_hash,
            candidate_count=len(universe.candidates),
            mandatory_candidate_count=universe.mandatory_candidate_count,
            checked_candidate_count=universe.checked_candidate_count,
            incomplete_mandatory_buckets=universe.incomplete_mandatory_buckets,
            classifier_version=CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
            lifecycle_action=LifecycleAction.NONE,
            review_case=None,
            status="checked",
            result_memory_id=challenger.id,
            audit={
                "source": "relation_discovery",
                "work_id": work.request.id,
                "run_generation": work.run_generation,
                "candidate_count_kind": "windowed",
                "llm_calls": classification.llm_calls,
                "prompt_chars": classification.prompt_chars,
                "labels": {
                    judgment.pair.candidate.memory_id: judgment.label.value
                    for judgment in classification.judgments
                },
                **selection.audit,
                **selection.telemetry,
            },
            started_at=now,
            completed_at=now,
        )
        return RelationOutcomeBundle(
            evidence_unit=evidence_unit,
            relation_run=relation_run,
            candidates=universe.candidates,
            candidate_provenance=tuple(candidate.memory for candidate in selection.discovery),
            expected_candidate_support_set_hashes=candidate_support_set_hashes,
        )


@dataclass(frozen=True, slots=True)
class _CompletedClassification:
    pair_count: int
    llm_calls: int
    prompt_chars: int


@dataclass(frozen=True, slots=True)
class _WorkProcessingError(RuntimeError):
    cause: Exception
    pair_count: int
    llm_calls: int
    prompt_chars: int


def relation_discovery_error_code(error: BaseException) -> str:
    """A stable code for one failure: the classifier's code, else the exception type."""

    return str(getattr(error, "error_code", None) or type(error).__name__)


def _relation_run_id(
    work: RelationDiscoveryWork,
    selection: CrossDocumentCandidateSelection,
) -> str:
    digest = sha256(
        "\x1f".join(
            (
                work.request.id,
                str(work.run_generation),
                work.request.expected_content_hash,
                selection.snapshot_identity,
                CROSS_DOCUMENT_RELATION_CLASSIFIER_VERSION,
            )
        ).encode("utf-8")
    ).hexdigest()[:20]
    return f"relation-run-{digest}"


def _lease_token(work: RelationDiscoveryWork) -> str:
    if not work.lease_token:
        raise ValueError("relation discovery work is not leased")
    return work.lease_token
