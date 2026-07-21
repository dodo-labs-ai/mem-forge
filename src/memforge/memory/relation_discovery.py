"""Budgeted post-commit discovery of non-destructive Memory relations."""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from memforge.memory.evidence import (
    AccessContext,
    AuthorityCase,
    EvidenceRelationRecord,
    LifecycleAction,
    RelationOutcomeBundle,
    RelationRunRecord,
    RelationType,
    ReviewCase,
    build_candidate_universe,
    classify_authority_case,
)
from memforge.memory.relation_candidate_retrieval import (
    CrossDocumentCandidateRetriever,
    CrossDocumentCandidateSelection,
)
from memforge.memory.relation_classifier import (
    MemoryPair,
    MemoryPairClassifier,
    MemoryPairClassificationError,
    MemoryPairDecision,
    MemoryRelationType,
)
from memforge.memory.relation_discovery_contract import (
    RelationDiscoveryWork,
    resolve_relation_discovery_actor_user_id,
)
from memforge.models import (
    Memory,
    MemoryReview,
    MemoryStatus,
    ReviewKind,
    ReviewStatus,
    generate_deterministic_review_id,
)
from memforge.storage.adapters.protocols import RelationalStore


RELATION_DISCOVERY_CLASSIFIER_VERSION = "memory-relation-v1"
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


class RelationDiscovery:
    """Advance durable relation work without owning Memory lifecycle authority."""

    def __init__(
        self,
        *,
        store: RelationalStore,
        candidate_retriever: CrossDocumentCandidateRetriever,
        pair_classifier: MemoryPairClassifier,
    ) -> None:
        self._store = store
        self._candidate_retriever = candidate_retriever
        self._pair_classifier = pair_classifier

    async def process_slice(
        self,
        *,
        worker_id: str,
        budget: RelationDiscoveryBudget | None = None,
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
            )
            if not leased:
                break
            work = leased[0]
            attempted += 1
            try:
                outcome = await self._process_work(
                    work,
                )
                if outcome is None:
                    await self._store.obsolete_relation_discovery_work(
                        work.request.id,
                        worker_id=worker_id,
                        lease_token=_lease_token(work),
                        reason="activated Memory or source evidence is no longer current",
                    )
                    obsolete += 1
                    continue
                relation_outcome, reviews, classification = outcome
                checked_pairs += classification.pair_count
                llm_calls += classification.llm_calls
                prompt_chars += classification.prompt_chars
                await self._store.complete_relation_discovery_work(
                    work.request.id,
                    worker_id=worker_id,
                    lease_token=_lease_token(work),
                    relation_outcome=relation_outcome,
                    reviews=reviews,
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
                    next_attempt_at=(retry_at.isoformat() if retry_at is not None else None),
                    exhausted=exhausted,
                )
                failed += 1

        return RelationDiscoverySliceResult(
            attempted_work=attempted,
            completed_work=completed,
            failed_work=failed,
            obsolete_work=obsolete,
            checked_candidate_pairs=checked_pairs,
            llm_calls=llm_calls,
            prompt_chars=prompt_chars,
            elapsed_ms=max(0, round((time.perf_counter() - started) * 1000)),
        )

    async def _process_work(
        self,
        work: RelationDiscoveryWork,
    ) -> (
        tuple[
            RelationOutcomeBundle,
            tuple[MemoryReview, ...],
            _CompletedClassification,
        ]
        | None
    ):
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
        pairs = tuple(
            MemoryPair(challenger=challenger, candidate=loaded_by_id[memory_id])
            for memory_id in selection.candidate_ids
        )
        try:
            classification = await self._pair_classifier.classify(pairs)
        except MemoryPairClassificationError as error:
            raise _WorkProcessingError(
                cause=error,
                pair_count=error.pair_count,
                llm_calls=error.llm_calls,
                prompt_chars=error.prompt_chars,
            ) from error
        try:
            await self._candidate_retriever.ensure_selection_current(
                selection,
                challenger=challenger,
                doc_id=evidence_unit.doc_id or request.doc_id,
                source_id=request.source_id,
                excluded_source_ids=disabled_source_ids,
            )
            bundle, reviews = await self._build_outcome(
                work=work,
                challenger=challenger,
                actor_user_id=actor_user_id,
                evidence_unit=evidence_unit,
                selection=selection,
                decisions=classification.decisions,
                loaded_by_id=loaded_by_id,
                classification_llm_calls=classification.llm_calls,
                classification_prompt_chars=classification.prompt_chars,
            )
        except Exception as error:
            raise _WorkProcessingError(
                cause=error,
                pair_count=len(pairs),
                llm_calls=classification.llm_calls,
                prompt_chars=classification.prompt_chars,
            ) from error
        return (
            bundle,
            reviews,
            _CompletedClassification(
                pair_count=len(pairs),
                llm_calls=classification.llm_calls,
                prompt_chars=classification.prompt_chars,
            ),
        )

    async def _build_outcome(
        self,
        *,
        work: RelationDiscoveryWork,
        challenger: Memory,
        actor_user_id: str | None,
        evidence_unit,
        selection: CrossDocumentCandidateSelection,
        decisions: tuple[MemoryPairDecision, ...],
        loaded_by_id,
        classification_llm_calls: int,
        classification_prompt_chars: int,
    ) -> tuple[RelationOutcomeBundle, tuple[MemoryReview, ...]]:
        relation_run_id = _relation_run_id(work, selection)
        universe = build_candidate_universe(
            relation_run_id=relation_run_id,
            evidence_unit_id=evidence_unit.id,
            bucket_results=selection.bucket_results(),
            recall_candidate_cap=max(1, len(selection.discovery)),
        )
        candidate_by_id = {candidate.memory.memory_id: candidate.memory for candidate in selection.discovery}
        source_subscriptions = tuple(
            dict.fromkeys(
                source_id
                for source_id in (
                    evidence_unit.source_id,
                    *(candidate.source_id for candidate in candidate_by_id.values()),
                )
                if source_id
            )
        )
        access_context = AccessContext(
            actor_user_id=actor_user_id,
            source_subscriptions=source_subscriptions,
            repo_identifier=challenger.repo_identifier,
            operation_type="relation_discovery",
        )
        relations: list[EvidenceRelationRecord] = []
        cross_source_conflicts: list[tuple[MemoryPairDecision, Memory]] = []
        now = datetime.now(timezone.utc)
        for decision in decisions:
            relation_type = _persisted_relation_type(decision.relation_type)
            if relation_type is None:
                continue
            candidate_row = candidate_by_id[decision.pair.candidate.id]
            authority = classify_authority_case(
                evidence_unit,
                candidate_row,
                universe.candidates[
                    next(
                        index
                        for index, candidate in enumerate(universe.candidates)
                        if candidate.memory_id == decision.pair.candidate.id
                    )
                ].bucket,
                relation_type,
                access_context,
            )
            relations.append(
                EvidenceRelationRecord(
                    evidence_unit_id=evidence_unit.id,
                    memory_id=decision.pair.candidate.id,
                    relation_type=relation_type,
                    direction=decision.direction,
                    authority_case=authority,
                    is_authoritative_support=False,
                    source_lineage_id=evidence_unit.source_lineage_id,
                    confidence=1.0,
                    reason=decision.reason,
                    classifier_version=RELATION_DISCOVERY_CLASSIFIER_VERSION,
                    relation_run_id=relation_run_id,
                    created_at=now.isoformat(),
                )
            )
            if (
                relation_type is RelationType.CONTRADICTS
                and authority is AuthorityCase.CROSS_SOURCE_CONFLICT
                and candidate_row.source_id
                and candidate_row.source_id != evidence_unit.source_id
            ):
                cross_source_conflicts.append((decision, dict(loaded_by_id)[decision.pair.candidate.id]))

        lifecycle_action = LifecycleAction.CREATE_REVIEW if cross_source_conflicts else LifecycleAction.NONE
        review_case = ReviewCase.CROSS_SOURCE_CONFLICT if cross_source_conflicts else None
        relation_run = RelationRunRecord(
            id=relation_run_id,
            evidence_unit_id=evidence_unit.id,
            access_context_hash=evidence_unit.access_context_hash,
            candidate_count=len(universe.candidates),
            mandatory_candidate_count=universe.mandatory_candidate_count,
            checked_candidate_count=universe.checked_candidate_count,
            incomplete_mandatory_buckets=universe.incomplete_mandatory_buckets,
            classifier_version=RELATION_DISCOVERY_CLASSIFIER_VERSION,
            lifecycle_action=lifecycle_action,
            review_case=review_case,
            status="review" if cross_source_conflicts else "checked",
            result_memory_id=challenger.id,
            audit={
                "source": "relation_discovery",
                "work_id": work.request.id,
                "candidate_count_kind": "windowed",
                "llm_calls": classification_llm_calls,
                "prompt_chars": classification_prompt_chars,
                **selection.audit,
                **selection.telemetry,
            },
            started_at=now.isoformat(),
            completed_at=now.isoformat(),
        )
        bundle = RelationOutcomeBundle(
            evidence_unit=evidence_unit,
            relation_run=relation_run,
            candidates=universe.candidates,
            relations=tuple(relations),
            candidate_provenance=tuple(candidate.memory for candidate in selection.discovery),
        )
        reviews = tuple(
            _cross_source_review(
                decision=decision,
                challenger=challenger,
                incumbent=incumbent,
                relation_run=relation_run,
                evidence_unit_id=evidence_unit.id,
                now=now,
            )
            for decision, incumbent in cross_source_conflicts
        )
        return bundle, reviews


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


def _persisted_relation_type(
    relation_type: MemoryRelationType,
) -> RelationType | None:
    if relation_type is MemoryRelationType.UNRELATED:
        return None
    return RelationType(relation_type.value)


def _relation_run_id(
    work: RelationDiscoveryWork,
    selection: CrossDocumentCandidateSelection,
) -> str:
    digest = sha256(
        "\x1f".join(
            (
                work.request.id,
                work.request.expected_content_hash,
                selection.snapshot_identity,
                RELATION_DISCOVERY_CLASSIFIER_VERSION,
            )
        ).encode("utf-8")
    ).hexdigest()[:20]
    return f"relation-run-{digest}"


def _cross_source_review(
    *,
    decision: MemoryPairDecision,
    challenger: Memory,
    incumbent: Memory,
    relation_run: RelationRunRecord,
    evidence_unit_id: str,
    now: datetime,
) -> MemoryReview:
    kind = ReviewKind.CROSS_SOURCE_CONFLICT.value
    return MemoryReview(
        id=generate_deterministic_review_id(
            kind=kind,
            incumbent_memory_id=incumbent.id,
            challenger_memory_id=challenger.id,
            relation_run_id=relation_run.id,
            evidence_unit_id=evidence_unit_id,
            review_case=ReviewCase.CROSS_SOURCE_CONFLICT.value,
        ),
        kind=kind,
        status=ReviewStatus.PENDING.value,
        incumbent_memory_id=incumbent.id,
        challenger_memory_id=challenger.id,
        reason=f"contradicts: {decision.reason}" if decision.reason else "contradicts",
        expected_incumbent_updated_at=(incumbent.updated_at.isoformat() if incumbent.updated_at else None),
        expected_challenger_updated_at=(challenger.updated_at.isoformat() if challenger.updated_at else None),
        created_at=now,
    )


def _lease_token(work: RelationDiscoveryWork) -> str:
    if not work.lease_token:
        raise ValueError("relation discovery work is not leased")
    return work.lease_token
