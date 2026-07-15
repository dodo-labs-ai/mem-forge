"""Memory extraction and projected-lifecycle orchestration.

The engine is the ownership boundary between source projection/extraction and
durable memory state.  A projected lifecycle call derives its reconciliation
scope, access context, staged evidence, lifecycle plan, and outbox work from one
``SourceProjection`` so those records commit atomically.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from memforge.memory.entity_resolver import EntityResolver, insert_llm_alias, resolve_entity
from memforge.memory.evidence import (
    AuthorityCase,
    EvidenceContentProvenance,
    EvidenceRole,
    EvidenceRelationRecord,
    EvidenceUnit,
    LifecycleAction,
    MemoryRelationApplyService,
    RelationCandidateRecord,
    RelationOutcomeBundle,
    RelationRunRecord,
    RelationType,
    relation_run_id_for,
)
from memforge.memory.lifecycle_plan import ReconciliationScope
from memforge.memory.lifecycle_planner import (
    NewMemoryDefaults,
    build_lifecycle_plan,
    lifecycle_access_context_hash,
    lifecycle_plan_id,
)
from memforge.memory.quality import classify_memory_candidate
from memforge.source_access import memory_visibility_for_document
from memforge.models import (
    Memory,
    RawMemory,
    ReconcileAction,
    ReconcileOperation,
    content_hash,
    generate_memory_id,
)

from memforge.storage.adapters.protocols import RelationalStore, VectorStore

if TYPE_CHECKING:
    from memforge.memory.store import MemoryStore
    from memforge.models import EnrichmentResult
    from memforge.source_projection import SourceProjection
    from memforge.storage.database import Database

logger = logging.getLogger(__name__)

__all__ = ["MemoryEngine"]


MEMORY_EQUIVALENCE_PROMPT = """Decide whether two durable Memory claims have exactly the same truth conditions.
Equivalent wording, language, or abbreviation is allowed. Return equivalent=false if either claim contradicts,
narrows, broadens, conditions, updates, or adds any material fact relative to the other.

<candidate_pair>
{pair_json}
</candidate_pair>
"""

MEMORY_SUPPORT_VALIDATION_PROMPT = """Determine whether the current evidence still supports the exact Memory claim.
Return supported=true only when the claim's truth conditions remain entailed by the current
Primary and every current Required observation. A change in scope, subject, condition,
polarity, or applicability means supported=false.

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
        relational: RelationalStore,
        vector: VectorStore,
        db: Database,
        memory_store: MemoryStore,
        embed_cfg: dict | None = None,
        structured_llm_client: Any = None,
        llm_model: str = "claude-sonnet-4-20250514",
    ) -> None:
        # Held so a later phase can stamp visibility through them without
        # re-plumbing this constructor; the orchestration here reads neither yet.
        self.relational = relational
        self.vector = vector
        self.db = db
        self.memory_store = memory_store
        self.structured_llm_client = structured_llm_client
        self.llm_model = llm_model
        # Entity resolver with embedding + LLM capabilities
        self.entity_resolver = EntityResolver(
            db=db,
            embed_cfg=embed_cfg,
            structured_llm_client=structured_llm_client,
            llm_model=llm_model,
        )

    # -------------------------------------------------------------------
    # Process Call 1: Entity resolution + alias insertion
    # -------------------------------------------------------------------

    async def process_enrichment(
        self,
        doc_id: str,
        enrichment: EnrichmentResult,
        doc_context: str | None = None,
    ) -> list[int]:
        """Process enrichment result: resolve entities and register their aliases.

        For each entity extracted by Call 1, resolves it against the DB
        (exact match → alias lookup → embedding search → create new),
        then registers any aliases the LLM found for it.

        Returns list of resolved entity IDs (passed to Call 2 as context).
        """
        from memforge.pipeline.entity_filter import filter_entities

        # Filter low-confidence or malformed entities
        raw_dicts = [
            {"name": e.name, "type": e.type, "tags": e.tags, "confidence": e.confidence} for e in enrichment.entities
        ]
        filtered_dicts, filter_stats = filter_entities(raw_dicts)
        filtered_names = {d["name"] for d in filtered_dicts}

        logger.info(
            "Entity filter: %d → %d (removed %d noise)",
            len(enrichment.entities),
            len(filtered_dicts),
            len(enrichment.entities) - len(filtered_dicts),
        )

        resolved_ids: list[int] = []

        for raw_entity in enrichment.entities:
            if raw_entity.name not in filtered_names:
                continue

            tags = (
                raw_entity.tags
                if raw_entity.tags
                else ([raw_entity.type] if raw_entity.type and raw_entity.type != "unknown" else [])
            )

            # Resolve: find existing entity or create new
            entity_id = await self.entity_resolver.resolve(
                extracted_name=raw_entity.name,
                db=self.db,
                extracted_tags=tags,
                doc_context=doc_context,
            )
            resolved_ids.append(entity_id)

            # Register aliases the LLM found for this entity
            for alias_name in raw_entity.aliases or []:
                await insert_llm_alias(
                    alias_name=alias_name,
                    canonical_name=raw_entity.name,
                    canonical_id=entity_id,
                    evidence="",
                    db=self.db,
                )

        self.entity_resolver.invalidate_cache()
        return resolved_ids

    # -------------------------------------------------------------------
    # Process Call 2: Build memories, dedup, insert
    # -------------------------------------------------------------------

    async def process_memories(
        self,
        doc_id: str,
        raw_memories: list[RawMemory],
        source_type: str,
        project_key: str | None = None,
        repo_identifier: str | None = None,
        entity_ids: list[int] | None = None,
        *,
        audit_context: Any | None = None,
        user_id: str | None = None,
        source_updated_at: datetime | None,
    ) -> dict:
        """Process extracted memories: build, dedup, and persist.

        Returns stats dict with counts of inserted, corroborated, skipped.
        """
        stats = {"inserted": 0, "corroborated": 0, "skipped": 0}

        for raw in raw_memories:
            if not self._candidate_can_persist(raw, stats):
                continue

            unit = await self._document_evidence_unit(
                doc_id=doc_id,
                raw=raw,
                source_type=source_type,
                project_key=project_key,
                repo_identifier=repo_identifier,
                extractor_run_id=getattr(audit_context, "run_id", None),
            )
            if await self._evidence_unit_has_materialized_memory(unit):
                stats["skipped"] += 1
                continue
            lifecycle = MemoryRelationApplyService().derive_lifecycle(unit, [])
            if lifecycle.action is not LifecycleAction.CREATE_MEMORY or not lifecycle.created_memory_id:
                await self._record_document_relation_outcome(
                    unit=unit,
                    relation_run_id=_document_relation_run_id(
                        unit,
                        LifecycleAction.CREATE_REVIEW,
                        relation_type=RelationType.SUPPORTS,
                    ),
                    lifecycle_action=lifecycle.action,
                    memory_id=None,
                    status="review",
                    review_case=lifecycle.review_case,
                    audit={
                        "source": "memory_engine.process_memories",
                        "review_case": lifecycle.review_case.value if lifecycle.review_case else None,
                    },
                )
                stats["skipped"] += 1
                continue

            # Build memory object
            memory = Memory(
                id=lifecycle.created_memory_id,
                memory_type=raw.memory_type,
                content=raw.content.strip(),
                content_hash=content_hash(raw.content.strip()),
                visibility=unit.visibility,
                owner_user_id=unit.owner_user_id,
                project_key=project_key,
                repo_identifier=repo_identifier,
                entity_refs=raw.entity_refs,
                tags=raw.tags,
                confidence=raw.confidence,
                corroboration_count=1,
                contradiction_count=0,
                valid_from=_parse_date(raw.valid_from),
                valid_until=_parse_date(raw.valid_until),
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
                status="active",
                extraction_context=raw.extraction_context,
            )

            # Resolve entity refs to IDs for linking
            # Filter first — Call 2's LLM sometimes outputs code names as entity_refs
            from memforge.pipeline.entity_filter import filter_entities

            ref_dicts = [{"name": name, "type": "unknown"} for name in raw.entity_refs]
            filtered_refs, _ = filter_entities(ref_dicts)
            filtered_ref_names = {d["name"] for d in filtered_refs}

            memory_entity_ids: list[int] = []
            for entity_name in raw.entity_refs:
                if entity_name not in filtered_ref_names:
                    continue
                try:
                    eid = await resolve_entity(entity_name, db=self.db)
                    memory_entity_ids.append(eid)
                except Exception as e:
                    logger.warning("Failed to resolve entity %r: %s", entity_name, e)

            # Dedup + insert (or corroborate)
            result = await self.memory_store.deduplicate_and_insert(
                memory=memory,
                doc_id=doc_id,
                source_type=source_type,
                entity_ids=memory_entity_ids,
                excerpt=raw.extraction_context,
                source_updated_at=source_updated_at,
                relation_outcome=self._document_relation_outcome_bundle(
                    unit=unit,
                    relation_run_id=_document_relation_run_id(
                        unit,
                        LifecycleAction.CREATE_MEMORY,
                        relation_type=RelationType.SUPPORTS,
                    ),
                    lifecycle_action=lifecycle.action,
                    memory_id=memory.id,
                    status="applied",
                    review_case=lifecycle.review_case,
                    audit={"source": "memory_engine.process_memories"},
                ),
            )

            if result == "inserted":
                stats["inserted"] += 1
                stats.setdefault("_inserted_ids", []).append(memory.id)
            elif result == "corroborated":
                stats["corroborated"] += 1
            else:
                stats["skipped"] += 1

        # Cross-document contradiction detection
        inserted_ids = stats.pop("_inserted_ids", [])
        if inserted_ids and self.structured_llm_client:
            from memforge.pipeline.contradiction_detector import detect_cross_doc_contradictions

            contradiction_stats = await detect_cross_doc_contradictions(
                new_memory_ids=inserted_ids,
                doc_id=doc_id,
                db=self.db,
                memory_store=self.memory_store,
                structured_llm_client=self.structured_llm_client,
                llm_model=self.llm_model,
                audit_context=audit_context,
                actor_user_id=user_id,
            )
            stats["contradictions_found"] = contradiction_stats.get("contradictions", 0)

        logger.info(
            "Memory processing for %s: %d inserted, %d corroborated, %d skipped",
            doc_id,
            stats["inserted"],
            stats["corroborated"],
            stats["skipped"],
        )
        return stats

    # -------------------------------------------------------------------
    # Process with reconciliation (for document UPDATES)
    # -------------------------------------------------------------------

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
        incumbents_by_id: dict[str, Memory] = {}
        for memory_id in sorted(unit_support):
            memory = await self.db.get_memory(memory_id)
            if memory is not None and memory.status == "active":
                incumbents_by_id[memory.id] = memory
        for memory in await self.db.get_memories_by_source_doc(
            doc_id,
            support_kind="extracted",
        ):
            if memory.status == "active":
                incumbents_by_id.setdefault(memory.id, memory)
        return [incumbents_by_id[key] for key in sorted(incumbents_by_id)], unit_support

    async def _rebind_noop_evidence_to_current_revision(
        self,
        *,
        operations: tuple[ReconcileOperation, ...],
        incumbents: dict[str, Memory],
        projection: SourceProjection,
    ) -> tuple[ReconcileOperation, ...]:
        """Carry an exact, still-present claim forward without re-extracting it.

        Incremental extraction intentionally sees only changed ranges. A NOOP
        for an incumbent therefore may not contain a new candidate. If its
        supporting Observation was revised, prove the old exact excerpt still
        exists in that same stable Observation and stage a current-revision
        reference. Missing or ambiguous evidence fails closed.
        """

        current_revisions = {
            revision.observation_id: revision
            for revision in projection.observation_revisions
        }
        rebound: list[ReconcileOperation] = []
        for operation in operations:
            if (
                operation.action is not ReconcileAction.NOOP
                or operation.memory_id is None
                or operation.memory is not None
            ):
                rebound.append(operation)
                continue
            support = await self.db.get_active_memory_support_evidence(
                operation.memory_id,
                source_id=projection.source_id,
            )
            stale = [
                item
                for item in support
                if item.anchor.observation_id in current_revisions
                and current_revisions[item.anchor.observation_id].id
                != item.anchor.observation_revision_id
            ]
            if not stale:
                rebound.append(operation)
                continue
            missing_dependencies = [
                item
                for item in support
                if item.anchor.observation_id not in current_revisions
            ]
            if missing_dependencies and projection.coverage.proves_absence:
                raise RuntimeError(
                    "NOOP incumbent has a removed evidence dependency: "
                    f"{operation.memory_id}"
                )
            primary = [
                item
                for item in support
                if item.role is EvidenceRole.PRIMARY
            ]
            if len(primary) != 1:
                raise RuntimeError(
                    "NOOP incumbent lacks exactly one PRIMARY dependency: "
                    f"{operation.memory_id}"
                )
            selected = primary[0]
            if (
                selected in stale
                and (
                    not selected.excerpt
                    or selected.excerpt
                    not in current_revisions[selected.anchor.observation_id].content
                )
            ):
                raise RuntimeError(
                    "NOOP incumbent lacks exact current-revision PRIMARY evidence: "
                    f"{operation.memory_id}"
                )
            required_observation_ids = sorted(
                {
                    item.anchor.observation_id
                    for item in support
                    if item.role is EvidenceRole.REQUIRED
                }
            )
            incumbent = incumbents[operation.memory_id]
            stale_required = [
                item for item in stale if item.role is EvidenceRole.REQUIRED
            ]
            support_validation: dict[str, object] = {}
            if stale_required:
                validator = getattr(
                    self.structured_llm_client,
                    "validate_memory_support",
                    None,
                )
                if validator is None:
                    raise RuntimeError(
                        "revised REQUIRED evidence needs structured semantic validation: "
                        f"{operation.memory_id}"
                    )
                validation = await validator(
                    MEMORY_SUPPORT_VALIDATION_PROMPT.format(
                        case_json=json.dumps(
                            {
                                "memory_claim": incumbent.content,
                                "primary": current_revisions[
                                    selected.anchor.observation_id
                                ].content,
                                "required": [
                                    current_revisions[item.anchor.observation_id].content
                                    for item in stale_required
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
                    "required_observation_ids": sorted(
                        item.anchor.observation_id for item in stale_required
                    ),
                }
                if not validation.supported:
                    rebound.append(
                        ReconcileOperation(
                            action=ReconcileAction.DELETE,
                            memory_id=operation.memory_id,
                            reason=(
                                "revised REQUIRED evidence no longer validates claim: "
                                f"{validation.reason}"
                            ),
                            flag_for_review=True,
                        )
                    )
                    continue
            rebound.append(
                ReconcileOperation(
                    action=operation.action,
                    memory_id=operation.memory_id,
                    memory=RawMemory(
                        content=incumbent.content,
                        memory_type=incumbent.memory_type,
                        confidence=incumbent.confidence,
                        tags=list(incumbent.tags),
                        extraction_context=selected.excerpt,
                        evidence_quote=selected.excerpt,
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

    async def _claims_semantically_equivalent(
        self,
        candidate: Memory,
        incumbent: Memory,
    ) -> dict[str, object] | None:
        """Return an auditable semantic proof for cross-source identity reuse."""

        if (
            candidate.content_hash == incumbent.content_hash
            and candidate.content.strip() == incumbent.content.strip()
        ):
            return {
                "method": "exact_content",
                "candidate_content_hash": candidate.content_hash,
                "incumbent_content_hash": incumbent.content_hash,
            }
        classifier = getattr(
            self.structured_llm_client,
            "classify_memory_equivalence",
            None,
        )
        if classifier is None:
            return None
        prompt = MEMORY_EQUIVALENCE_PROMPT.format(
            pair_json=json.dumps(
                {
                    "candidate": candidate.content,
                    "incumbent": incumbent.content,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        try:
            result = await classifier(
                prompt,
                max_tokens=512,
                model=self.llm_model,
            )
        except Exception:
            logger.warning(
                "Semantic Memory equivalence classification failed; preserving separate identities",
                exc_info=True,
            )
            return None
        if not result.equivalent:
            return None
        return {
            "method": "structured_classifier",
            "model": self.llm_model or "default",
            "reason": result.reason,
            "candidate_content_hash": candidate.content_hash,
            "incumbent_content_hash": incumbent.content_hash,
        }

    async def apply_projected_lifecycle(
        self,
        *,
        projection: SourceProjection,
        doc_id: str,
        raw_memories: list[RawMemory],
        doc_type: str,
        project_key: str | None,
        repo_identifier: str | None,
        entity_ids: list[int],
        document_content: str,
        update_mode: str,
        changed_hunks: str | None,
        update_plan_stats: dict[str, Any] | None,
        source_updated_at: datetime | None,
        user_id: str | None = None,
    ) -> dict[str, int]:
        """Reconcile a complete Source Unit ledger and atomically apply one plan."""

        from memforge.pipeline.reconciler import reconcile_memories
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
        observation_revision_ids = tuple(
            revision.id for revision in projection.observation_revisions
        )
        source_type = projection.source_type

        stats = {
            "added": 0,
            "corroborated": 0,
            "updated": 0,
            "superseded": 0,
            "deleted": 0,
            "noop": 0,
            "pending_review": 0,
            "skipped": 0,
        }
        filtered_memories: list[RawMemory] = []
        for raw in raw_memories:
            if self._candidate_can_persist(raw, stats):
                filtered_memories.append(raw)

        incumbents, unit_support = await self._active_projected_incumbents(
            doc_id=doc_id,
            source_unit_id=scope.source_unit_id,
        )
        if incumbents and not self.structured_llm_client:
            raise RuntimeError("complete lifecycle reconciliation requires an LLM client")
        result = await reconcile_memories(
            new_extractions=filtered_memories,
            existing_memories=incumbents,
            doc_type=doc_type,
            structured_llm_client=self.structured_llm_client,
            llm_model=self.llm_model,
            updated_document=document_content,
            update_mode=update_mode,
            changed_hunks=changed_hunks,
            update_plan_stats=update_plan_stats,
            include_metadata=True,
        )
        failure = getattr(result, "failure", None)
        if failure is not None:
            raise RuntimeError(
                f"complete lifecycle reconciliation failed: {failure.error_type}: {failure.error}"
            )
        operations = tuple(getattr(result, "operations", result))
        for operation in operations:
            if operation.action not in {
                ReconcileAction.ADD,
                ReconcileAction.UPDATE,
                ReconcileAction.SUPERSEDE,
            } or operation.memory is None:
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
            projection=projection,
        )
        gate = await self.db.get_lifecycle_gate(scope.source_id)
        all_support = {
            memory_id: await self.db.get_active_memory_support_reference_ids(memory_id)
            for memory_id in incumbents_by_id
        }
        support_hashes = {
            memory_id: await self.db.get_memory_support_set_hash(memory_id)
            for memory_id in incumbents_by_id
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
        candidate_finder = getattr(
            self.memory_store,
            "find_access_compatible_equivalence_candidates",
            None,
        )
        if candidate_finder is not None:
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
                targets = await candidate_finder(
                    candidate,
                    excluded_memory_ids=frozenset(incumbents_by_id),
                    excluded_source_id=scope.source_id,
                )
                target = None
                for possible_target in targets:
                    if lifecycle_access_context_hash(
                        visibility=possible_target.visibility,
                        owner_user_id=possible_target.owner_user_id,
                        project_key=possible_target.project_key,
                        repo_identifier=possible_target.repo_identifier,
                    ) != lifecycle_access_context_hash(
                        visibility=candidate.visibility,
                        owner_user_id=candidate.owner_user_id,
                        project_key=candidate.project_key,
                        repo_identifier=candidate.repo_identifier,
                    ):
                        continue
                    equivalence_proof = await self._claims_semantically_equivalent(
                        candidate,
                        possible_target,
                    )
                    if equivalence_proof is not None:
                        target = possible_target
                        break
                if target is None:
                    continue
                claim_hash = content_hash(operation.memory.content.strip())
                corroboration_targets[claim_hash] = target
                corroboration_proofs[claim_hash] = equivalence_proof
                all_support[target.id] = await self.db.get_active_memory_support_reference_ids(
                    target.id
                )
                support_hashes[target.id] = await self.db.get_memory_support_set_hash(target.id)
        evidence_memories = [
            operation.memory
            for operation in operations
            if operation.memory is not None
        ]
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
            observed_at=(
                source_updated_at.isoformat() if source_updated_at is not None else None
            ),
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
            evidence_reference_ids_by_claim_hash=(
                projected_evidence.reference_ids_by_claim_hash
            ),
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
                entity_ids=tuple(entity_ids),
                source_updated_at=(
                    source_updated_at.isoformat() if source_updated_at is not None else None
                ),
            ),
            evidence_units=projected_evidence.units,
            evidence_references=projected_evidence.references,
        )
        await self.db.apply_source_projection_lifecycle(projection, plan)
        await self.memory_store.drain_lifecycle_vector_outbox(plan.id)

        created_memory_ids = [
            mutation.memory_id
            for mutation in plan.mutations
            if mutation.mutation_type.value == "create_memory"
        ]
        if created_memory_ids and self.structured_llm_client:
            from memforge.pipeline.contradiction_detector import detect_cross_doc_contradictions

            contradiction_stats = await detect_cross_doc_contradictions(
                new_memory_ids=created_memory_ids,
                doc_id=doc_id,
                db=self.db,
                memory_store=self.memory_store,
                structured_llm_client=self.structured_llm_client,
                llm_model=self.llm_model,
                actor_user_id=user_id,
            )
            stats["contradictions_found"] = contradiction_stats.get("contradictions", 0)

        for mutation in plan.mutations:
            if mutation.mutation_type.value == "create_memory":
                stats["added"] += 1
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
                and mutation.memory_id
                in {target.id for target in corroboration_targets.values()}
            }
        )
        stats["noop"] = sum(
            decision.disposition.value == "keep"
            for decision in plan.coverage_proof.incumbent_decisions
        )
        return stats

    async def apply_projected_tombstone(
        self,
        *,
        projection: SourceProjection,
        doc_id: str,
        reason: str,
    ) -> dict[str, int | bool]:
        """Apply an authoritative Source Unit tombstone without an LLM call.

        Provider absence is already an explicit deterministic fact at this
        boundary. Every active same-document incumbent therefore receives a
        DELETE ledger entry, while the per-source lifecycle gate still decides
        whether that becomes support removal/retirement or a durable review.
        """

        if len(projection.deltas) != 1 or not projection.coverage.proves_absence:
            raise ValueError("projected tombstone requires one absence-proving Revision Delta")
        delta = projection.deltas[0]
        scope = ReconciliationScope(
            id=f"tombstone:{delta.source_unit_id}:{delta.current_unit_revision_id or 'removed'}",
            source_id=projection.source_id,
            source_unit_id=delta.source_unit_id,
            base_unit_revision_id=delta.previous_unit_revision_id,
            target_unit_revision_id=delta.current_unit_revision_id,
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
        all_support = {
            memory_id: await self.db.get_active_memory_support_reference_ids(memory_id)
            for memory_id in incumbents_by_id
        }
        support_hashes = {
            memory_id: await self.db.get_memory_support_set_hash(memory_id)
            for memory_id in incumbents_by_id
        }
        visibility, owner_user_id = await memory_visibility_for_document(self.db, doc_id=doc_id)
        plan = build_lifecycle_plan(
            plan_id=lifecycle_plan_id(scope),
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
        await self.db.apply_source_projection_lifecycle(projection, plan)
        await self.memory_store.drain_lifecycle_vector_outbox(plan.id)
        pending_review = sum(
            mutation.mutation_type.value == "create_review" for mutation in plan.mutations
        )
        retired = sum(
            mutation.mutation_type.value == "retire_memory" for mutation in plan.mutations
        )
        return {
            "retired": retired,
            "pending_review": pending_review,
            "can_delete_document": pending_review == 0,
        }

    def _candidate_can_persist(self, raw: RawMemory, stats: dict | None = None) -> bool:
        """Return whether a raw candidate should be persisted, updating stats when skipped."""
        quality = classify_memory_candidate(raw)
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
            tags=raw.tags,
            confidence=raw.confidence,
            corroboration_count=1,
            contradiction_count=0,
            valid_from=_parse_date(raw.valid_from),
            valid_until=_parse_date(raw.valid_until),
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            status="active",
            extraction_context=raw.extraction_context,
        )

    async def _resolve_entity_refs(self, entity_refs: list[str]) -> list[int]:
        """Resolve entity names to IDs."""
        ids: list[int] = []
        for name in entity_refs:
            try:
                eid = await resolve_entity(name, db=self.db)
                ids.append(eid)
            except Exception as e:
                logger.warning("Failed to resolve entity %r: %s", name, e)
        return ids

    async def _document_evidence_unit(
        self,
        *,
        doc_id: str,
        raw: RawMemory,
        source_type: str,
        project_key: str | None,
        repo_identifier: str | None,
        extractor_run_id: str | None,
    ) -> EvidenceUnit:
        document = await self.db.get_document(doc_id)
        doc_revision_id = None
        source_id = source_type
        if document is not None:
            doc_revision_id = document.content_hash or document.version
            source_id = document.source or source_type
        visibility, owner_user_id = await memory_visibility_for_document(
            self.db,
            doc_id=doc_id,
        )
        content = raw.content.strip()
        evidence_id = _document_evidence_unit_id(
            source_id=source_id,
            doc_id=doc_id,
            doc_revision_id=doc_revision_id,
            source_type=source_type,
            content=content,
        )
        return EvidenceUnit(
            id=evidence_id,
            source_id=source_id,
            doc_id=doc_id,
            doc_revision_id=doc_revision_id,
            source_type=source_type,
            source_anchor=evidence_id,
            source_lineage_id=doc_id,
            project_key=project_key,
            visibility=visibility,
            owner_user_id=owner_user_id,
            repo_identifier=repo_identifier,
            content=content,
            excerpt=raw.extraction_context,
            evidence_provenance=(
                EvidenceContentProvenance.SOURCE_EXCERPT
                if raw.extraction_context
                else EvidenceContentProvenance.NO_EXCERPT
            ),
            source_metadata={
                "memory_type": raw.memory_type,
                "content_hash": content_hash(content),
            },
            observed_at=datetime.now(timezone.utc).isoformat(),
            extractor_run_id=extractor_run_id,
        )

    async def _evidence_unit_has_materialized_memory(self, unit: EvidenceUnit) -> bool:
        return await self.db.has_materialized_evidence_unit(unit.id)

    async def _record_document_relation_outcome(
        self,
        *,
        unit: EvidenceUnit,
        relation_run_id: str,
        lifecycle_action: LifecycleAction,
        memory_id: str | None,
        status: str,
        review_case,
        audit: dict[str, object],
        candidates: list[RelationCandidateRecord] | None = None,
        incomplete_mandatory_buckets: tuple[str, ...] = (),
        candidate_count: int | None = None,
    ) -> None:
        await self.db.record_relation_outcome_bundle(
            self._document_relation_outcome_bundle(
                unit=unit,
                relation_run_id=relation_run_id,
                lifecycle_action=lifecycle_action,
                memory_id=memory_id,
                status=status,
                review_case=review_case,
                audit=audit,
                candidates=candidates,
                incomplete_mandatory_buckets=incomplete_mandatory_buckets,
                candidate_count=candidate_count,
            )
        )

    def _document_relation_outcome_bundle(
        self,
        *,
        unit: EvidenceUnit,
        relation_run_id: str,
        lifecycle_action: LifecycleAction,
        memory_id: str | None,
        status: str,
        review_case,
        audit: dict[str, object],
        candidates: list[RelationCandidateRecord] | None = None,
        incomplete_mandatory_buckets: tuple[str, ...] = (),
        candidate_count: int | None = None,
    ) -> RelationOutcomeBundle:
        candidates = candidates or []
        run_audit = dict(audit)
        relations: tuple[EvidenceRelationRecord, ...] = ()
        if memory_id is not None:
            relations = (
                EvidenceRelationRecord(
                    evidence_unit_id=unit.id,
                    memory_id=memory_id,
                    relation_type=RelationType.SUPPORTS,
                    authority_case=AuthorityCase.SAME_SOURCE_LINEAGE,
                    is_authoritative_support=True,
                    source_lineage_id=unit.source_lineage_id,
                    confidence=1.0,
                    reason="Memory created from this source evidence unit.",
                    excerpt=unit.excerpt,
                    classifier_version="memory-engine-v1",
                    relation_run_id=relation_run_id,
                    created_at=datetime.now(timezone.utc).isoformat(),
                ),
            )
        return RelationOutcomeBundle(
            evidence_unit=unit,
            relation_run=RelationRunRecord(
                id=relation_run_id,
                evidence_unit_id=unit.id,
                access_context_hash=unit.access_context_hash,
                candidate_count=len(candidates) if candidate_count is None else candidate_count,
                mandatory_candidate_count=sum(1 for candidate in candidates if candidate.is_mandatory),
                checked_candidate_count=sum(1 for candidate in candidates if candidate.was_checked),
                incomplete_mandatory_buckets=incomplete_mandatory_buckets,
                classifier_version="memory-engine-v1",
                lifecycle_action=lifecycle_action,
                review_case=review_case,
                status=status,
                result_memory_id=memory_id,
                audit=run_audit,
            ),
            candidates=tuple(candidates),
            relations=relations,
        )

def _document_evidence_unit_id(
    *,
    source_id: str,
    doc_id: str,
    doc_revision_id: str | None,
    source_type: str,
    content: str,
) -> str:
    digest = sha256(
        "\x1f".join([source_id, doc_id, doc_revision_id or "", source_type, content_hash(content)]).encode("utf-8")
    ).hexdigest()[:16]
    return f"eu-doc-{digest}"


def _document_relation_run_id(
    unit: EvidenceUnit,
    action: str | LifecycleAction,
    *,
    classifier_version: str = "memory-engine-v1",
    candidate_memory_id: str | None = None,
    relation_type: RelationType | None = None,
    authority_case: AuthorityCase | None = None,
) -> str:
    return relation_run_id_for(
        prefix="doc",
        unit=unit,
        action=action,
        classifier_version=classifier_version,
        candidate_memory_id=candidate_memory_id,
        relation_type=relation_type,
        authority_case=authority_case,
    )


def _parse_date(value: str | None) -> date | None:
    """Parse an ISO date or datetime string into a calendar date."""
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except (ValueError, TypeError):
        return None
