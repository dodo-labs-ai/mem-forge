"""Memory extraction and projected-lifecycle orchestration.

The engine is the ownership boundary between source projection/extraction and
durable memory state.  A projected lifecycle call derives its reconciliation
scope, access context, staged evidence, lifecycle plan, and outbox work from one
``SourceProjection`` so those records commit atomically.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from time import perf_counter
from typing import TYPE_CHECKING, Any

from memforge.memory.candidate_ledger import (
    CandidateLedgerError,
    CandidateLedgerResult,
    select_unique_memory_candidates,
)
from memforge.memory.entity_resolver import EntityResolver
from memforge.memory.evidence import EvidenceRole
from memforge.memory.identity_resolver import (
    IdentityResolutionRequest,
    IdentityResolver,
)
from memforge.memory.lifecycle_plan import ReconciliationScope
from memforge.memory.lifecycle_planner import (
    NewMemoryDefaults,
    build_lifecycle_plan,
    lifecycle_access_context_hash,
    lifecycle_plan_id,
)
from memforge.memory.quality import classify_memory_candidate
from memforge.memory.relation_candidate_retrieval import CrossDocumentCandidateRetriever
from memforge.memory.relation_classifier import (
    MEMORY_PAIR_CLASSIFIER_VERSION,
    StructuredMemoryPairClassifier,
)
from memforge.memory.relation_discovery_contract import PreclassifiedRelationDecision
from memforge.source_access import memory_visibility_for_document
from memforge.source_projection import ImpactResult, ProjectionCoverage, resolve_anchor_impact
from memforge.storage.adapters.protocols import EntityResolutionScope
from memforge.models import (
    Memory,
    RawMemory,
    ReconcileAction,
    ReconcileOperation,
    content_hash,
    generate_memory_id,
    parse_memory_validity_date,
)

if TYPE_CHECKING:
    from memforge.memory.store import MemoryStore
    from memforge.source_projection import SourceProjection
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)

__all__ = ["MemoryEngine"]


MEMORY_SUPPORT_VALIDATION_PROMPT = """Determine whether the current evidence still supports the exact Memory claim.
Return supported=true only when the claim's truth conditions remain entailed by the current
Primary and every current Required observation. A change in scope, subject, condition,
polarity, or applicability means supported=false.
When supported=true and the previous Primary quote is no longer present verbatim, return
evidence_quote as one exact, non-empty substring copied from the current Primary observation
that directly supports the claim. Never paraphrase evidence_quote.

<case_json>
{case_json}
</case_json>
"""


class MemoryEngine:
    """Turns enrichment and extracted claims into durable memory state.

    Responsibilities:
    - Resolve extracted entities and aliases.
    - Build and apply an atomic lifecycle plan for a source projection.
    - Support direct, non-projected memory ingestion where explicitly requested.
    """

    def __init__(
        self,
        cross_document_candidates: CrossDocumentCandidateRetriever,
        db: Database,
        memory_store: MemoryStore,
        embed_cfg: dict | None = None,
        structured_llm_client: Any = None,
        llm_model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self.cross_document_candidates = cross_document_candidates
        self.db = db
        self.memory_store = memory_store
        self.structured_llm_client = structured_llm_client
        self.llm_model = llm_model
        self.pair_classifier = (
            StructuredMemoryPairClassifier(
                client=structured_llm_client,
                model=llm_model,
            )
            if callable(getattr(structured_llm_client, "classify_memory_relations", None))
            else None
        )
        self.identity_resolver = IdentityResolver(
            memory_store=memory_store,
            pair_classifier=self.pair_classifier,
            llm_model=llm_model,
        )
        # Entity resolver with embedding + LLM capabilities
        self.entity_resolver = EntityResolver(
            store=db,
            embed_cfg=embed_cfg,
            structured_llm_client=structured_llm_client,
            llm_model=llm_model,
        )

    async def _active_projected_incumbents(
        self,
        *,
        doc_id: str,
        source_unit_id: str,
    ) -> tuple[list[Memory], dict[str, tuple[str, ...]]]:
        """Load the complete active ledger by stable Unit, with a legacy fallback.

        A provider-backed rename can change ``doc_id`` without changing the
        Source Unit. Support Assertions are therefore authoritative for the
        projected path; same-document extracted support is included only to
        keep pre-cutover rows visible to the conservative lineage gate.
        """
        unit_support = await self.db.get_source_unit_support_reference_ids(source_unit_id)
        incumbents_by_id = {
            memory.id: memory
            for memory in await self.db.list_active_memories(tuple(sorted(unit_support)))
        }
        for memory in await self.db.get_memories_by_source_doc(
            doc_id,
            support_kind="extracted",
        ):
            if memory.status == "active":
                incumbents_by_id.setdefault(memory.id, memory)
        return [incumbents_by_id[key] for key in sorted(incumbents_by_id)], unit_support

    async def _projected_incumbent_impacts(
        self,
        *,
        projection: SourceProjection,
        incumbent_ids: frozenset[str],
        unit_support: Mapping[str, tuple[str, ...]],
    ) -> dict[str, ImpactResult]:
        """Resolve each incumbent against the current Revision Delta.

        Missing legacy Support, ambiguous mappings, and mixed evidence stay
        UNKNOWN. A single affected reference makes the incumbent AFFECTED;
        only a complete set of disjoint references proves DISJOINT.
        """

        delta = projection.deltas[0]
        ordered_incumbent_ids = tuple(sorted(incumbent_ids))
        evidence_by_memory_id = await self.db.get_active_memory_support_evidence_many(
            ordered_incumbent_ids,
            source_id=projection.source_id,
        )
        resolved: dict[str, ImpactResult] = {}
        for memory_id in ordered_incumbent_ids:
            reference_ids = unit_support.get(memory_id)
            if not reference_ids:
                resolved[memory_id] = ImpactResult.UNKNOWN
                continue
            scoped_reference_ids = frozenset(reference_ids)
            evidence = evidence_by_memory_id.get(memory_id, ())
            impacts = {
                resolve_anchor_impact(item.anchor, delta)
                for item in evidence
                if item.reference_id in scoped_reference_ids
            }
            if ImpactResult.AFFECTED in impacts:
                resolved[memory_id] = ImpactResult.AFFECTED
            elif not impacts or ImpactResult.UNKNOWN in impacts:
                resolved[memory_id] = ImpactResult.UNKNOWN
            else:
                resolved[memory_id] = ImpactResult.DISJOINT
        return resolved

    @staticmethod
    def _partial_projection_protected_incumbents(
        *,
        projection: SourceProjection,
        incumbent_impacts: Mapping[str, ImpactResult],
    ) -> frozenset[str]:
        """Return partial-projection incumbents without affected-anchor proof."""

        if projection.coverage is not ProjectionCoverage.PARTIAL_PROJECTION:
            return frozenset()
        return frozenset(
            memory_id for memory_id, impact in incumbent_impacts.items() if impact is not ImpactResult.AFFECTED
        )

    @staticmethod
    def _enforce_partial_projection_keep(
        operations: tuple[ReconcileOperation, ...],
        protected_memory_ids: frozenset[str],
    ) -> tuple[ReconcileOperation, ...]:
        destructive = {
            ReconcileAction.UPDATE,
            ReconcileAction.SUPERSEDE,
            ReconcileAction.DELETE,
        }
        return tuple(
            ReconcileOperation(
                action=ReconcileAction.NOOP,
                memory_id=operation.memory_id,
                reason="partial projection has no deterministic affected-anchor proof",
            )
            if operation.memory_id in protected_memory_ids and operation.action in destructive
            else operation
            for operation in operations
        )

    async def _rebind_noop_evidence_to_current_revision(
        self,
        *,
        operations: tuple[ReconcileOperation, ...],
        incumbents: dict[str, Memory],
        unit_support: Mapping[str, tuple[str, ...]],
        projection: SourceProjection,
    ) -> tuple[ReconcileOperation, ...]:
        """Carry an exact, still-present claim forward without re-extracting it.

        Incremental extraction intentionally sees only changed ranges. A NOOP
        for an incumbent therefore may not contain a new candidate. If its
        supporting Observation was revised, prove the old exact excerpt still
        exists in that same stable Observation and stage a current-revision
        reference. Missing or ambiguous evidence fails closed.
        """

        current_revisions = {revision.observation_id: revision for revision in projection.observation_revisions}
        rebound: list[ReconcileOperation] = []
        for operation in operations:
            if (
                operation.action is not ReconcileAction.NOOP
                or operation.memory_id is None
                or operation.memory is not None
            ):
                rebound.append(operation)
                continue
            source_support = await self.db.get_active_memory_support_evidence(
                operation.memory_id,
                source_id=projection.source_id,
            )
            scoped_reference_ids = frozenset(unit_support.get(operation.memory_id, ()))
            support = tuple(item for item in source_support if item.reference_id in scoped_reference_ids)
            stale = [
                item
                for item in support
                if item.anchor.observation_id in current_revisions
                and current_revisions[item.anchor.observation_id].id != item.anchor.observation_revision_id
            ]
            if not stale:
                rebound.append(operation)
                continue
            missing_dependencies = [item for item in support if item.anchor.observation_id not in current_revisions]
            if missing_dependencies and projection.coverage.proves_absence:
                raise RuntimeError(f"NOOP incumbent has a removed evidence dependency: {operation.memory_id}")
            primary = [item for item in support if item.role is EvidenceRole.PRIMARY]
            if len(primary) != 1:
                raise RuntimeError(f"NOOP incumbent lacks exactly one PRIMARY dependency: {operation.memory_id}")
            selected = primary[0]
            primary_needs_validation = selected in stale and (
                not selected.excerpt
                or selected.excerpt not in current_revisions[selected.anchor.observation_id].content
            )
            required_observation_ids = sorted(
                {item.anchor.observation_id for item in support if item.role is EvidenceRole.REQUIRED}
            )
            incumbent = incumbents[operation.memory_id]
            stale_required = [item for item in stale if item.role is EvidenceRole.REQUIRED]
            support_validation: dict[str, object] = {}
            current_primary_quote = selected.excerpt or ""
            if primary_needs_validation or stale_required:
                validator = getattr(
                    self.structured_llm_client,
                    "validate_memory_support",
                    None,
                )
                if validator is None:
                    raise RuntimeError(f"revised evidence needs structured semantic validation: {operation.memory_id}")
                current_primary = current_revisions.get(selected.anchor.observation_id)
                if current_primary is None:
                    raise RuntimeError(
                        f"NOOP incumbent current PRIMARY observation is unavailable: {operation.memory_id}"
                    )
                validation = await validator(
                    MEMORY_SUPPORT_VALIDATION_PROMPT.format(
                        case_json=json.dumps(
                            {
                                "memory_claim": incumbent.content,
                                "previous_primary_quote": selected.excerpt,
                                "primary": current_primary.content,
                                "required": [
                                    current_revisions[item.anchor.observation_id].content for item in stale_required
                                ],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                    ),
                    max_tokens=512,
                    model=self.llm_model,
                )
                support_validation = {
                    "method": "structured_classifier",
                    "model": self.llm_model,
                    "supported": bool(validation.supported),
                    "reason": validation.reason,
                    "primary_observation_id": selected.anchor.observation_id,
                    "required_observation_ids": sorted(item.anchor.observation_id for item in stale_required),
                }
                if not validation.supported:
                    rebound.append(
                        ReconcileOperation(
                            action=ReconcileAction.DELETE,
                            memory_id=operation.memory_id,
                            reason=(f"revised REQUIRED evidence no longer validates claim: {validation.reason}"),
                            flag_for_review=True,
                        )
                    )
                    continue
                if primary_needs_validation:
                    current_primary_quote = str(getattr(validation, "evidence_quote", "") or "").strip()
                    if not current_primary_quote or current_primary_quote not in current_primary.content:
                        raise RuntimeError(
                            "NOOP incumbent support validation lacks exact current "
                            "PRIMARY evidence: "
                            f"{operation.memory_id}"
                        )
            rebound.append(
                ReconcileOperation(
                    action=operation.action,
                    memory_id=operation.memory_id,
                    memory=RawMemory(
                        content=incumbent.content,
                        memory_type=incumbent.memory_type,
                        confidence=incumbent.confidence,
                        extraction_context=current_primary_quote,
                        evidence_quote=current_primary_quote,
                        evidence_anchor="revalidated_noop",
                        source_observation_id=selected.anchor.observation_id,
                        required_source_observation_ids=required_observation_ids,
                        support_validation=support_validation,
                    ),
                    reason=operation.reason,
                    flag_for_review=operation.flag_for_review,
                )
            )
        return tuple(rebound)

    async def apply_projected_lifecycle(
        self,
        *,
        projection: SourceProjection,
        doc_id: str,
        raw_memories: list[RawMemory],
        doc_type: str,
        project_key: str | None,
        repo_identifier: str | None,
        document_content: str,
        update_mode: str,
        changed_hunks: str | None,
        update_plan_stats: dict[str, Any] | None,
        source_updated_at: datetime | None,
        user_id: str | None = None,
        expected_source_activity_epoch: int | None = None,
    ) -> dict[str, int]:
        """Reconcile a complete Source Unit ledger and atomically apply one plan."""

        from memforge.pipeline.reconciler import (
            ReconciliationResult,
            reconcile_memories,
        )
        from memforge.pipeline.projection_evidence import build_projected_claim_evidence

        if len(projection.deltas) != 1:
            raise ValueError("projected lifecycle requires exactly one Revision Delta")
        delta = projection.deltas[0]
        scope = ReconciliationScope(
            id=f"scope:{projection.run_id}",
            source_id=projection.source_id,
            source_unit_id=delta.source_unit_id,
            base_unit_revision_id=delta.previous_unit_revision_id,
            target_unit_revision_id=delta.current_unit_revision_id,
        )
        observation_revision_ids = tuple(revision.id for revision in projection.observation_revisions)
        source_type = projection.source_type

        stats = {
            "added": 0,
            "reactivated": 0,
            "corroborated": 0,
            "updated": 0,
            "superseded": 0,
            "deleted": 0,
            "noop": 0,
            "pending_review": 0,
            "skipped": 0,
            "vector_delivery_pending": 0,
            "relation_discovery_enqueued": 0,
        }
        filtered_memories: list[RawMemory] = []
        for raw in raw_memories:
            if self._candidate_can_persist(
                raw,
                stats,
                observation_semantic_class=_observation_semantic_class(
                    projection,
                    raw.source_observation_id,
                ),
            ):
                filtered_memories.append(raw)
        quality_candidate_count = len(filtered_memories)
        candidate_ledger = await self._select_projected_candidates(
            projection=projection,
            doc_id=doc_id,
            candidates=filtered_memories,
        )
        filtered_memories = list(candidate_ledger.candidates)
        stats.update(
            {
                "candidate_ledger_input_count": candidate_ledger.input_count,
                "candidate_ledger_selected_count": len(candidate_ledger.candidates),
                "candidate_ledger_dropped_exact_count": (
                    candidate_ledger.dropped_exact_count
                ),
                "candidate_ledger_dropped_redundant_count": (
                    candidate_ledger.dropped_redundant_count
                ),
                "candidate_ledger_llm_calls": candidate_ledger.structured_llm_calls,
                "candidate_ledger_llm_elapsed_ms": (
                    candidate_ledger.structured_llm_elapsed_ms
                ),
                "candidate_ledger_validation_retries": (
                    candidate_ledger.validation_retries
                ),
                "candidate_ledger_prompt_chars": candidate_ledger.prompt_chars,
            }
        )
        stats["skipped"] += quality_candidate_count - len(filtered_memories)
        reconciliation_started = perf_counter()
        incumbents, unit_support = await self._active_projected_incumbents(
            doc_id=doc_id,
            source_unit_id=scope.source_unit_id,
        )
        incumbent_impacts: dict[str, ImpactResult] = {}
        needs_incumbent_impacts = projection.coverage is ProjectionCoverage.PARTIAL_PROJECTION or (
            not filtered_memories and bool(document_content.strip())
        )
        if needs_incumbent_impacts:
            incumbent_impacts = await self._projected_incumbent_impacts(
                projection=projection,
                incumbent_ids=frozenset(memory.id for memory in incumbents),
                unit_support=unit_support,
            )
        model_incumbent_count = 0
        deterministic_disjoint_keep_count = 0
        model_batch_count = 0
        structured_llm_call_count = 0
        structured_llm_elapsed_ms = 0
        bounded_reconciliation_elapsed_ms = 0
        if not document_content.strip() and not filtered_memories:
            operations = tuple(
                ReconcileOperation(
                    action=ReconcileAction.DELETE,
                    memory_id=memory.id,
                    reason="source observation is explicitly empty",
                )
                for memory in sorted(incumbents, key=lambda item: item.id)
            )
        else:
            deterministic_keep_ids = (
                frozenset(
                    memory_id for memory_id, impact in incumbent_impacts.items() if impact is ImpactResult.DISJOINT
                )
                if not filtered_memories
                else frozenset()
            )
            model_incumbents = [memory for memory in incumbents if memory.id not in deterministic_keep_ids]
            model_incumbent_count = len(model_incumbents)
            deterministic_disjoint_keep_count = len(deterministic_keep_ids)
            if model_incumbents and not self.structured_llm_client:
                raise RuntimeError("complete lifecycle reconciliation requires an LLM client")
            result = await reconcile_memories(
                new_extractions=filtered_memories,
                existing_memories=model_incumbents,
                doc_type=doc_type,
                structured_llm_client=self.structured_llm_client,
                llm_model=self.llm_model,
                updated_document=document_content,
                update_mode=update_mode,
                changed_hunks=changed_hunks,
                update_plan_stats=update_plan_stats,
                include_metadata=True,
            )
            if not isinstance(result, ReconciliationResult):
                raise TypeError("metadata reconciliation must return ReconciliationResult")
            reconciliation_metrics = result.metrics
            model_batch_count = reconciliation_metrics.model_batch_count
            structured_llm_call_count = reconciliation_metrics.structured_llm_calls
            structured_llm_elapsed_ms = reconciliation_metrics.structured_llm_elapsed_ms
            bounded_reconciliation_elapsed_ms = reconciliation_metrics.reconciliation_elapsed_ms
            if result.failure is not None:
                raise RuntimeError(
                    f"complete lifecycle reconciliation failed: {result.failure.error_type}: {result.failure.error}"
                )
            operations = tuple(result.operations) + tuple(
                ReconcileOperation(
                    action=ReconcileAction.NOOP,
                    memory_id=memory_id,
                    reason="Revision Delta proves the incumbent evidence is disjoint",
                )
                for memory_id in sorted(deterministic_keep_ids)
            )
        protected_memory_ids = self._partial_projection_protected_incumbents(
            projection=projection,
            incumbent_impacts=incumbent_impacts,
        )
        operations = self._enforce_partial_projection_keep(
            operations,
            protected_memory_ids,
        )
        logger.info(
            json.dumps(
                {
                    "event": "projected_lifecycle_reconciliation",
                    "source_id": projection.source_id,
                    "source_unit_id": scope.source_unit_id,
                    "reconciliation_new_candidate_count": len(filtered_memories),
                    "reconciliation_incumbent_count": len(incumbents),
                    "reconciliation_model_incumbent_count": model_incumbent_count,
                    "reconciliation_disjoint_keep_count": (deterministic_disjoint_keep_count),
                    "reconciliation_llm_batch_count": model_batch_count,
                    "reconciliation_llm_call_count": structured_llm_call_count,
                    "reconciliation_llm_elapsed_ms": structured_llm_elapsed_ms,
                    "reconciliation_bounded_elapsed_ms": (bounded_reconciliation_elapsed_ms),
                    "reconciliation_total_elapsed_ms": max(
                        0,
                        round((perf_counter() - reconciliation_started) * 1000),
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        for operation in operations:
            if (
                operation.action
                not in {
                    ReconcileAction.ADD,
                    ReconcileAction.UPDATE,
                    ReconcileAction.SUPERSEDE,
                }
                or operation.memory is None
            ):
                continue
            quality = classify_memory_candidate(operation.memory)
            if not quality.keep:
                raise RuntimeError(
                    "complete lifecycle reconciliation produced an unsafe Memory candidate: "
                    f"{quality.skip_reason or 'quality_rejected'}"
                )
        incumbents_by_id = {memory.id: memory for memory in incumbents}
        operations = await self._rebind_noop_evidence_to_current_revision(
            operations=operations,
            incumbents=incumbents_by_id,
            unit_support=unit_support,
            projection=projection,
        )
        gate = await self.db.get_lifecycle_gate(scope.source_id)
        incumbent_support_states = await self.db.get_active_memory_support_states(
            tuple(incumbents_by_id)
        )
        all_support = {
            memory_id: state.reference_ids
            for memory_id, state in incumbent_support_states.items()
        }
        support_hashes = {
            memory_id: state.support_set_hash
            for memory_id, state in incumbent_support_states.items()
        }
        visibility, owner_user_id = await memory_visibility_for_document(self.db, doc_id=doc_id)
        if visibility == "private" and user_id is not None and user_id != owner_user_id:
            raise PermissionError("private projected lifecycle actor does not own the document")
        access_context_hash = lifecycle_access_context_hash(
            visibility=visibility,
            owner_user_id=owner_user_id,
            project_key=project_key,
            repo_identifier=repo_identifier,
        )
        corroboration_targets: dict[str, Memory] = {}
        corroboration_proofs: dict[str, dict[str, object]] = {}
        preclassified_relations: dict[str, tuple[PreclassifiedRelationDecision, ...]] = {}
        identity_claim_hashes: list[str] = []
        identity_requests: list[IdentityResolutionRequest] = []
        operation_memories = tuple(
            operation.memory for operation in operations if operation.memory is not None
        )
        entity_resolution = await self.entity_resolver.resolve_many(
            tuple(
                entity_ref
                for raw_memory in operation_memories
                for entity_ref in raw_memory.entity_refs
            ),
            scope=EntityResolutionScope(access_context_hash=access_context_hash),
            doc_context=document_content[:2000],
        )
        stats.update(
            {
                "entity_resolution_unique_mentions": entity_resolution.metrics.unique_mentions,
                "entity_resolution_exact_hits": entity_resolution.metrics.exact_hits,
                "entity_resolution_alias_hits": entity_resolution.metrics.alias_hits,
                "entity_resolution_embedded_mentions": entity_resolution.metrics.embedded_mentions,
                "entity_resolution_ambiguous_mentions": entity_resolution.metrics.ambiguous_mentions,
                "entity_resolution_embedding_batches": entity_resolution.metrics.embedding_batches,
                "entity_resolution_llm_calls": entity_resolution.metrics.structured_llm_calls,
                "entity_resolution_candidate_count": entity_resolution.metrics.candidate_count,
                "entity_resolution_new_entities": entity_resolution.metrics.new_entities,
                "entity_resolution_elapsed_ms": entity_resolution.metrics.elapsed_ms,
            }
        )
        entity_ids_by_claim_hash = {
            content_hash(raw_memory.content.strip()): tuple(
                dict.fromkeys(
                    entity_id
                    for entity_ref in raw_memory.entity_refs
                    if (entity_id := entity_resolution.entity_id(entity_ref)) is not None
                )
            )
            for raw_memory in operation_memories
        }
        for operation in operations:
            if operation.action is not ReconcileAction.ADD or operation.memory is None:
                continue
            candidate = self._build_memory(
                operation.memory,
                project_key,
                visibility=visibility,
                owner_user_id=owner_user_id,
                repo_identifier=repo_identifier,
            )
            identity_claim_hashes.append(content_hash(operation.memory.content.strip()))
            identity_requests.append(
                IdentityResolutionRequest(
                    challenger=candidate,
                    doc_id=doc_id,
                    entity_ids=entity_ids_by_claim_hash.get(
                        content_hash(operation.memory.content.strip()),
                        (),
                    ),
                    excluded_memory_ids=frozenset(incumbents_by_id),
                )
            )
        identity_resolution = await self.identity_resolver.resolve(tuple(identity_requests))
        identity_resolutions = identity_resolution.resolutions
        stats.update(
            {
                "identity_resolution_pair_count": identity_resolution.metrics.pair_count,
                "identity_resolution_llm_calls": identity_resolution.metrics.llm_calls,
                "identity_resolution_prompt_chars": identity_resolution.metrics.prompt_chars,
                "identity_resolution_elapsed_ms": identity_resolution.metrics.elapsed_ms,
            }
        )
        classified_candidate_ids = tuple(
            dict.fromkeys(
                memory_id
                for resolution in identity_resolutions
                for memory_id in (
                    *(decision.pair.candidate.id for decision in resolution.classified_pairs),
                    *((resolution.target.id,) if resolution.target is not None else ()),
                )
            )
        )
        classified_candidate_support = await self.db.get_active_memory_support_states(
            classified_candidate_ids
        )
        attached_target_ids: list[str] = []
        for claim_hash, resolution in zip(
            identity_claim_hashes,
            identity_resolutions,
            strict=True,
        ):
            target = resolution.target
            equivalence_proof = resolution.equivalence_proof
            preclassified_relations[claim_hash] = tuple(
                PreclassifiedRelationDecision(
                    candidate_memory_id=decision.pair.candidate.id,
                    expected_candidate_content_hash=decision.pair.candidate.content_hash,
                    expected_candidate_support_set_hash=(
                        classified_candidate_support[
                            decision.pair.candidate.id
                        ].current_support_set_hash
                    ),
                    expected_candidate_access_context_hash=lifecycle_access_context_hash(
                        visibility=decision.pair.candidate.visibility,
                        owner_user_id=decision.pair.candidate.owner_user_id,
                        project_key=decision.pair.candidate.project_key,
                        repo_identifier=decision.pair.candidate.repo_identifier,
                    ),
                    expected_challenger_access_context_hash=access_context_hash,
                    relation_type=decision.relation_type,
                    direction=decision.direction,
                    reason=decision.reason,
                    classifier_version=MEMORY_PAIR_CLASSIFIER_VERSION,
                )
                for decision in resolution.classified_pairs
            )
            if target is None or equivalence_proof is None:
                continue
            corroboration_targets[claim_hash] = target
            corroboration_proofs[claim_hash] = dict(equivalence_proof)
            attached_target_ids.append(target.id)
        attached_support_states = {
            memory_id: classified_candidate_support[memory_id]
            for memory_id in attached_target_ids
        }
        for memory_id, state in attached_support_states.items():
            all_support[memory_id] = state.reference_ids
            support_hashes[memory_id] = state.support_set_hash
        evidence_memories = [operation.memory for operation in operations if operation.memory is not None]
        projected_evidence = build_projected_claim_evidence(
            projection=projection,
            raw_memories=evidence_memories,
            doc_id=doc_id,
            source_type=source_type,
            project_key=project_key,
            visibility=visibility,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            access_context_hash=access_context_hash,
            extractor_run_id=projection.run_id,
            observed_at=(source_updated_at.isoformat() if source_updated_at is not None else None),
        )
        plan_id = lifecycle_plan_id(scope)
        plan = build_lifecycle_plan(
            plan_id=plan_id,
            scope=scope,
            gate_state=gate.state,
            operations=operations,
            incumbents=incumbents_by_id,
            source_support_reference_ids=unit_support,
            all_active_support_reference_ids=all_support,
            support_set_hashes=support_hashes,
            observation_revision_ids=observation_revision_ids,
            new_evidence_reference_ids=(),
            evidence_reference_ids_by_claim_hash=(projected_evidence.reference_ids_by_claim_hash),
            corroboration_targets_by_claim_hash=corroboration_targets,
            corroboration_proofs_by_claim_hash=corroboration_proofs,
            defaults=NewMemoryDefaults(
                visibility=visibility,
                owner_user_id=owner_user_id,
                project_key=project_key,
                repo_identifier=repo_identifier,
                doc_id=doc_id,
                source_type=source_type,
                access_context_hash=access_context_hash,
                actor_user_id=user_id,
                entity_ids_by_claim_hash=entity_ids_by_claim_hash,
                preclassified_relations_by_claim_hash=preclassified_relations,
                source_updated_at=(source_updated_at.isoformat() if source_updated_at is not None else None),
            ),
            evidence_units=projected_evidence.units,
            evidence_references=projected_evidence.references,
        )
        await self.db.apply_source_projection_lifecycle(
            projection,
            plan,
            expected_source_activity_epoch=expected_source_activity_epoch,
        )
        delivery = await self.memory_store.attempt_lifecycle_vector_delivery(plan.id)
        stats["vector_delivery_pending"] = int(delivery.pending)

        stats["relation_discovery_enqueued"] = len(plan.relation_discovery_requests)

        for mutation in plan.mutations:
            if mutation.mutation_type.value == "create_memory":
                stats["added"] += 1
            elif mutation.mutation_type.value == "reactivate_memory":
                stats["reactivated"] += 1
            elif mutation.mutation_type.value == "supersede_memory":
                stats["superseded"] += 1
            elif mutation.mutation_type.value == "retire_memory":
                stats["deleted"] += 1
            elif mutation.mutation_type.value == "create_review":
                stats["pending_review"] += 1
        stats["corroborated"] = len(
            {
                mutation.memory_id
                for mutation in plan.mutations
                if mutation.mutation_type.value == "attach_support"
                and mutation.memory_id in {target.id for target in corroboration_targets.values()}
            }
        )
        stats["noop"] = sum(
            decision.disposition.value == "keep" for decision in plan.coverage_proof.incumbent_decisions
        )
        return stats

    async def _select_projected_candidates(
        self,
        *,
        projection: SourceProjection,
        doc_id: str,
        candidates: list[RawMemory],
    ) -> CandidateLedgerResult:
        """Select one complete within-revision candidate ledger before writes."""

        try:
            result = await select_unique_memory_candidates(
                candidates,
                structured_llm_client=self.structured_llm_client,
                llm_model=self.llm_model,
            )
        except CandidateLedgerError as exc:
            await self._record_candidate_ledger_audit(
                projection=projection,
                doc_id=doc_id,
                status="failed",
                reason=exc.error_type,
                payload={
                    "input_count": exc.input_count,
                    "semantic_input_count": exc.semantic_input_count,
                    "selected_count": 0,
                    "structured_llm_calls": exc.structured_llm_calls,
                    "structured_llm_elapsed_ms": exc.structured_llm_elapsed_ms,
                    "validation_retries": exc.validation_retries,
                    "prompt_chars": exc.prompt_chars,
                    "candidate_fingerprints": _candidate_fingerprints(candidates),
                    "fingerprints_truncated": len(candidates) > 200,
                },
                error=str(exc),
            )
            raise RuntimeError(f"candidate ledger failed closed: {exc.error_type}: {exc}") from exc

        if result.semantic_input_count > 1 or result.dropped_exact_count:
            await self._record_candidate_ledger_audit(
                projection=projection,
                doc_id=doc_id,
                status="committed",
                reason="complete_candidate_ledger",
                payload=_candidate_ledger_audit_payload(result),
            )
        return result

    async def _record_candidate_ledger_audit(
        self,
        *,
        projection: SourceProjection,
        doc_id: str,
        status: str,
        reason: str,
        payload: dict[str, Any],
        error: str | None = None,
    ) -> None:
        context = self.memory_store.operation_context(
            run_id=projection.run_id,
            source_id=projection.source_id,
            doc_id=doc_id,
        )
        await self.memory_store.record_audit_event(
            "candidate_ledger_completed" if status == "committed" else "candidate_ledger_failed",
            status,
            context=context,
            doc_id=doc_id,
            source_id=projection.source_id,
            decision="select_unique_candidates",
            reason=reason,
            payload=payload,
            error=error,
        )

    async def apply_projected_tombstone(
        self,
        *,
        projection: SourceProjection,
        doc_id: str,
        reason: str,
        lifecycle_cycle_id: str,
        expected_source_activity_epoch: int | None = None,
    ) -> dict[str, int | bool]:
        """Apply an authoritative Source Unit tombstone without an LLM call.

        Provider absence is already an explicit deterministic fact at this
        boundary. Every active same-document incumbent therefore receives a
        DELETE ledger entry, while the per-source lifecycle gate still decides
        whether that becomes support removal/retirement or a durable review.
        """

        if len(projection.deltas) != 1 or not projection.coverage.proves_absence:
            raise ValueError("projected tombstone requires one absence-proving Revision Delta")
        if not lifecycle_cycle_id.strip():
            raise ValueError("projected tombstone requires lifecycle cycle identity")
        delta = projection.deltas[0]
        scope = ReconciliationScope(
            id=(f"tombstone:{lifecycle_cycle_id}:{delta.source_unit_id}:{delta.current_unit_revision_id or 'removed'}"),
            source_id=projection.source_id,
            source_unit_id=delta.source_unit_id,
            base_unit_revision_id=delta.previous_unit_revision_id,
            target_unit_revision_id=delta.current_unit_revision_id,
        )
        plan_id = lifecycle_plan_id(scope)
        applied_payload = await self.db.get_lifecycle_plan_payload(plan_id)
        if applied_payload is not None:
            stored_scope = applied_payload.get("scope")
            mutations = applied_payload.get("mutations")
            if (
                not isinstance(stored_scope, Mapping)
                or stored_scope.get("id") != scope.id
                or stored_scope.get("source_id") != scope.source_id
                or stored_scope.get("source_unit_id") != scope.source_unit_id
                or stored_scope.get("target_unit_revision_id") != scope.target_unit_revision_id
                or not isinstance(mutations, list)
            ):
                raise ValueError("applied tombstone lifecycle ledger is malformed")
            mutation_types = [mutation.get("mutation_type") for mutation in mutations if isinstance(mutation, Mapping)]
            if len(mutation_types) != len(mutations):
                raise ValueError("applied tombstone lifecycle mutation ledger is malformed")
            await self.memory_store.attempt_lifecycle_vector_delivery(plan_id)
            return await self._projected_tombstone_result(
                doc_id=doc_id,
                mutation_types=mutation_types,
            )
        source_type = projection.source_type
        incumbents, unit_support = await self._active_projected_incumbents(
            doc_id=doc_id,
            source_unit_id=scope.source_unit_id,
        )
        incumbents_by_id = {memory.id: memory for memory in incumbents}
        operations = tuple(
            ReconcileOperation(
                action=ReconcileAction.DELETE,
                memory_id=memory_id,
                reason=reason,
            )
            for memory_id in sorted(incumbents_by_id)
        )
        gate = await self.db.get_lifecycle_gate(scope.source_id)
        support_states = await self.db.get_active_memory_support_states(tuple(incumbents_by_id))
        all_support = {
            memory_id: state.reference_ids for memory_id, state in support_states.items()
        }
        support_hashes = {
            memory_id: state.support_set_hash for memory_id, state in support_states.items()
        }
        visibility, owner_user_id = await memory_visibility_for_document(self.db, doc_id=doc_id)
        plan = build_lifecycle_plan(
            plan_id=plan_id,
            scope=scope,
            gate_state=gate.state,
            operations=operations,
            incumbents=incumbents_by_id,
            source_support_reference_ids=unit_support,
            all_active_support_reference_ids=all_support,
            support_set_hashes=support_hashes,
            observation_revision_ids=(),
            new_evidence_reference_ids=(),
            defaults=NewMemoryDefaults(
                visibility=visibility,
                owner_user_id=owner_user_id,
                project_key=None,
                repo_identifier=None,
                doc_id=doc_id,
                source_type=source_type,
                access_context_hash=lifecycle_access_context_hash(
                    visibility=visibility,
                    owner_user_id=owner_user_id,
                    project_key=None,
                    repo_identifier=None,
                ),
            ),
        )
        await self.db.apply_source_projection_lifecycle(
            projection,
            plan,
            expected_source_activity_epoch=expected_source_activity_epoch,
        )
        await self.memory_store.attempt_lifecycle_vector_delivery(plan.id)
        return await self._projected_tombstone_result(
            doc_id=doc_id,
            mutation_types=tuple(
                mutation.mutation_type.value for mutation in plan.mutations
            ),
        )

    async def _projected_tombstone_result(
        self,
        *,
        doc_id: str,
        mutation_types: Sequence[str],
    ) -> dict[str, int | bool]:
        """Return deletion eligibility from the committed document provenance."""

        pending_review = mutation_types.count("create_review")
        remaining_document_support = await self.db.get_memories_by_source_doc(
            doc_id,
            support_kind=None,
        )
        return {
            "retired": mutation_types.count("retire_memory"),
            "pending_review": pending_review,
            "can_delete_document": (
                pending_review == 0 and not remaining_document_support
            ),
        }

    def _candidate_can_persist(
        self,
        raw: RawMemory,
        stats: dict | None = None,
        *,
        observation_semantic_class: str | None = None,
    ) -> bool:
        """Return whether a raw candidate should be persisted, updating stats when skipped."""
        quality = classify_memory_candidate(
            raw,
            observation_semantic_class=observation_semantic_class,
        )
        if quality.keep:
            return True

        if stats is not None:
            stats["skipped"] = stats.get("skipped", 0) + 1
        logger.info(
            "Skipping memory candidate (%s): %s",
            quality.skip_reason,
            raw.content.strip()[:120],
        )
        return False

    def _build_memory(
        self,
        raw: RawMemory,
        project_key: str | None,
        *,
        visibility: str,
        owner_user_id: str | None,
        repo_identifier: str | None = None,
        memory_id: str | None = None,
    ) -> Memory:
        """Build a Memory object from a RawMemory."""
        return Memory(
            id=memory_id or generate_memory_id(),
            memory_type=raw.memory_type,
            content=raw.content.strip(),
            content_hash=content_hash(raw.content.strip()),
            visibility=visibility,
            owner_user_id=owner_user_id,
            project_key=project_key,
            repo_identifier=repo_identifier,
            entity_refs=raw.entity_refs,
            confidence=raw.confidence,
            corroboration_count=1,
            contradiction_count=0,
            valid_from=parse_memory_validity_date(raw.valid_from),
            valid_until=parse_memory_validity_date(raw.valid_until),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            status="active",
            extraction_context=raw.extraction_context,
        )

def _observation_semantic_class(
    projection: SourceProjection,
    observation_id: str | None,
) -> str | None:
    if observation_id is None:
        return None
    for revision in projection.observation_revisions:
        if revision.observation_id != observation_id:
            continue
        value = revision.metadata.get("semantic_class")
        return str(value) if isinstance(value, str) and value else None
    return None


def _candidate_ledger_audit_payload(result: CandidateLedgerResult) -> dict[str, Any]:
    return {
        "input_count": result.input_count,
        "semantic_input_count": result.semantic_input_count,
        "selected_count": len(result.candidates),
        "dropped_exact_count": result.dropped_exact_count,
        "dropped_redundant_count": result.dropped_redundant_count,
        "structured_llm_calls": result.structured_llm_calls,
        "structured_llm_elapsed_ms": result.structured_llm_elapsed_ms,
        "validation_retries": result.validation_retries,
        "prompt_chars": result.prompt_chars,
        "drops": [
            {
                "candidate_content_hash": content_hash(drop.candidate.content),
                "candidate_source_observation_id": drop.candidate.source_observation_id,
                "canonical_content_hash": content_hash(drop.canonical_candidate.content),
                "canonical_source_observation_id": (drop.canonical_candidate.source_observation_id),
                "method": drop.method,
                "reason": drop.reason[:240],
            }
            for drop in result.drops
        ],
    }


def _candidate_fingerprints(
    candidates: list[RawMemory],
    *,
    limit: int = 200,
) -> list[dict[str, str | None]]:
    return [
        {
            "content_hash": content_hash(candidate.content),
            "source_observation_id": candidate.source_observation_id,
        }
        for candidate in candidates[:limit]
    ]
