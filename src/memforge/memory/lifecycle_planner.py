"""Translate a complete reconciliation ledger into one atomic Lifecycle Plan."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from memforge.memory.evidence import EvidenceReference, EvidenceUnit
from memforge.memory.lifecycle_plan import (
    CoverageProof,
    IncumbentDecision,
    IncumbentDisposition,
    LifecycleGateState,
    LifecycleMutation,
    LifecycleMutationType,
    LifecyclePlan,
    ReconciliationScope,
    StaleGuard,
)
from memforge.memory.relation_discovery_contract import (
    RelationDiscoveryRequest,
    relation_discovery_request_id,
)
from memforge.models import (
    Memory,
    RawMemory,
    ReconcileAction,
    ReconcileOperation,
    content_hash,
    parse_memory_validity_date,
)


@dataclass(frozen=True, slots=True)
class NewMemoryDefaults:
    visibility: str
    owner_user_id: str | None
    project_key: str | None
    repo_identifier: str | None
    doc_id: str
    source_type: str
    access_context_hash: str
    actor_user_id: str | None = None
    entity_ids: tuple[int, ...] = ()
    source_updated_at: str | None = None


def build_lifecycle_plan(
    *,
    plan_id: str,
    scope: ReconciliationScope,
    gate_state: LifecycleGateState,
    operations: Sequence[ReconcileOperation],
    incumbents: Mapping[str, Memory],
    source_support_reference_ids: Mapping[str, tuple[str, ...]],
    all_active_support_reference_ids: Mapping[str, tuple[str, ...]],
    support_set_hashes: Mapping[str, str],
    observation_revision_ids: tuple[str, ...],
    new_evidence_reference_ids: tuple[str, ...],
    evidence_reference_ids_by_claim_hash: Mapping[str, tuple[str, ...]] | None = None,
    corroboration_targets_by_claim_hash: Mapping[str, Memory] | None = None,
    corroboration_proofs_by_claim_hash: Mapping[str, Mapping[str, object]] | None = None,
    defaults: NewMemoryDefaults,
    evidence_units: Sequence[EvidenceUnit] = (),
    evidence_references: Sequence[EvidenceReference] = (),
    incumbent_batch_size: int = 30,
) -> LifecyclePlan:
    """Build a complete plan without performing any storage mutation."""

    incumbent_ids = tuple(sorted(incumbents))
    by_incumbent: dict[str, ReconcileOperation] = {}
    add_operations: list[ReconcileOperation] = []
    for operation in operations:
        if operation.memory_id in incumbents:
            memory_id = operation.memory_id or ""
            if memory_id in by_incumbent:
                raise ValueError(f"duplicate lifecycle operation for incumbent: {memory_id}")
            by_incumbent[memory_id] = operation
        elif operation.action is ReconcileAction.ADD and operation.memory is not None:
            add_operations.append(operation)
        elif operation.action is ReconcileAction.NOOP and operation.memory_id is None:
            continue
        else:
            raise ValueError("reconciliation operation targets an unknown incumbent")

    missing = sorted(set(incumbent_ids).difference(by_incumbent))
    if missing:
        raise ValueError(f"missing lifecycle operation for incumbent: {missing}")

    mutations: list[LifecycleMutation] = []
    decisions: list[IncumbentDecision] = []
    created_ids: set[str] = set()
    relation_discovery_requests: list[RelationDiscoveryRequest] = []

    def request_relation_discovery(memory_id: str, expected_content_hash: str) -> None:
        relation_discovery_requests.append(
            RelationDiscoveryRequest(
                id=relation_discovery_request_id(
                    lifecycle_plan_id=plan_id,
                    memory_id=memory_id,
                    expected_content_hash=expected_content_hash,
                ),
                memory_id=memory_id,
                expected_content_hash=expected_content_hash,
                source_id=scope.source_id,
                source_unit_id=scope.source_unit_id,
                source_unit_revision_id=scope.target_unit_revision_id,
                doc_id=defaults.doc_id,
                actor_user_id=defaults.actor_user_id,
                entity_ids=defaults.entity_ids,
            )
        )

    def references_for(raw: RawMemory) -> tuple[str, ...]:
        if evidence_reference_ids_by_claim_hash is not None:
            references = evidence_reference_ids_by_claim_hash.get(content_hash(raw.content.strip()), ())
            if references:
                return references
        return new_evidence_reference_ids

    def memory_creation_mutations(raw: RawMemory) -> tuple[str, tuple[LifecycleMutation, ...]]:
        memory_id = _new_memory_id(scope.id, raw)
        evidence_reference_ids = references_for(raw)
        if not evidence_reference_ids:
            raise ValueError("new Memory candidate lacks support-granting evidence")
        return (
            memory_id,
            (
                LifecycleMutation(
                    LifecycleMutationType.CREATE_MEMORY,
                    memory_id=memory_id,
                    source_id=scope.source_id,
                    payload={"memory": _memory_payload(raw, defaults)},
                ),
                LifecycleMutation(
                    LifecycleMutationType.ATTACH_SUPPORT,
                    memory_id=memory_id,
                    source_id=scope.source_id,
                    evidence_reference_ids=evidence_reference_ids,
                    payload={
                        "access_context_hash": defaults.access_context_hash,
                        "source_updated_at": defaults.source_updated_at,
                    },
                ),
            ),
        )

    def create_memory(raw: RawMemory) -> str:
        memory_id, creation_mutations = memory_creation_mutations(raw)
        if memory_id not in created_ids:
            mutations.extend(creation_mutations)
            created_ids.add(memory_id)
            request_relation_discovery(memory_id, content_hash(raw.content.strip()))
        return memory_id

    corroboration_targets = corroboration_targets_by_claim_hash or {}
    corroboration_proofs = corroboration_proofs_by_claim_hash or {}
    corroboration_targets_by_id = {target.id: target for target in corroboration_targets.values()}
    attached_target_ids: set[str] = set()
    for operation in add_operations:
        assert operation.memory is not None
        claim_hash = content_hash(operation.memory.content.strip())
        target = corroboration_targets.get(claim_hash)
        if target is None:
            create_memory(operation.memory)
            continue
        evidence_reference_ids = references_for(operation.memory)
        if not evidence_reference_ids:
            raise ValueError("corroborated Memory candidate lacks support-granting evidence")
        reactivation_mutations: tuple[LifecycleMutation, ...] = ()
        if target.status == "retired":
            if target.retirement_reason != "source_rebaseline":
                raise ValueError("only source-rebaseline retirement may be reactivated")
            reactivation_mutations = (
                LifecycleMutation(
                    LifecycleMutationType.REACTIVATE_MEMORY,
                    memory_id=target.id,
                    source_id=scope.source_id,
                    payload={
                        "expected_content_hash": target.content_hash,
                        "reason": "exact claim replayed after source rebaseline",
                    },
                ),
            )
            request_relation_discovery(target.id, target.content_hash)
        mutations.extend(
            (
                *reactivation_mutations,
                LifecycleMutation(
                    LifecycleMutationType.ATTACH_SUPPORT,
                    memory_id=target.id,
                    source_id=scope.source_id,
                    evidence_reference_ids=evidence_reference_ids,
                    payload={
                        "access_context_hash": defaults.access_context_hash,
                        "source_updated_at": defaults.source_updated_at,
                        "equivalence_proof": dict(corroboration_proofs.get(claim_hash, {})),
                    },
                ),
                LifecycleMutation(
                    LifecycleMutationType.REFRESH_MEMORY_INDEX,
                    memory_id=target.id,
                    source_id=scope.source_id,
                ),
            )
        )
        attached_target_ids.add(target.id)

    for memory_id in incumbent_ids:
        operation = by_incumbent[memory_id]
        current_source_support = source_support_reference_ids.get(memory_id, ())
        all_support = all_active_support_reference_ids.get(memory_id, ())
        external_support = set(all_support).difference(current_source_support)

        if operation.action is ReconcileAction.NOOP:
            decisions.append(IncumbentDecision(memory_id, IncumbentDisposition.KEEP, operation.reason or "kept"))
            evidence_reference_ids = references_for(operation.memory) if operation.memory is not None else ()
            if evidence_reference_ids:
                if current_source_support and set(evidence_reference_ids) != set(current_source_support):
                    mutations.append(
                        LifecycleMutation(
                            LifecycleMutationType.REMOVE_SUPPORT,
                            memory_id=memory_id,
                            source_id=scope.source_id,
                            evidence_reference_ids=current_source_support,
                        )
                    )
                mutations.append(
                    LifecycleMutation(
                        LifecycleMutationType.ATTACH_SUPPORT,
                        memory_id=memory_id,
                        source_id=scope.source_id,
                        evidence_reference_ids=evidence_reference_ids,
                        payload={
                            "access_context_hash": defaults.access_context_hash,
                            "source_updated_at": defaults.source_updated_at,
                            "support_validation": dict(operation.memory.support_validation),
                        },
                    )
                )
            continue

        if operation.action is ReconcileAction.DELETE:
            if not current_source_support:
                raise ValueError(f"destructive incumbent lacks current-scope support: {memory_id}")
            if gate_state is LifecycleGateState.GATED or operation.flag_for_review:
                proposed_mutations = [
                    LifecycleMutation(
                        LifecycleMutationType.REMOVE_SUPPORT,
                        memory_id=memory_id,
                        source_id=scope.source_id,
                        evidence_reference_ids=current_source_support,
                        payload={"document_id": defaults.doc_id},
                    )
                ]
                if not external_support:
                    proposed_mutations.append(
                        LifecycleMutation(
                            LifecycleMutationType.RETIRE_MEMORY,
                            memory_id=memory_id,
                            source_id=scope.source_id,
                            payload={"reason": operation.reason or "support removed"},
                        )
                    )
                else:
                    proposed_mutations.append(
                        LifecycleMutation(
                            LifecycleMutationType.REFRESH_MEMORY_INDEX,
                            memory_id=memory_id,
                            source_id=scope.source_id,
                        )
                    )
                decisions.append(IncumbentDecision(memory_id, IncumbentDisposition.REVIEW, operation.reason or "gate"))
                mutations.append(
                    _review_mutation(
                        scope,
                        operation,
                        memory_id,
                        proposed_mutations=tuple(proposed_mutations),
                        disposition=IncumbentDisposition.REMOVE_SUPPORT,
                    )
                )
                continue
            decisions.append(
                IncumbentDecision(
                    memory_id,
                    IncumbentDisposition.REMOVE_SUPPORT,
                    operation.reason or "source evidence removed",
                )
            )
            mutations.append(
                LifecycleMutation(
                    LifecycleMutationType.REMOVE_SUPPORT,
                    memory_id=memory_id,
                    source_id=scope.source_id,
                    evidence_reference_ids=current_source_support,
                    payload={"document_id": defaults.doc_id},
                )
            )
            if not external_support:
                mutations.append(
                    LifecycleMutation(
                        LifecycleMutationType.RETIRE_MEMORY,
                        memory_id=memory_id,
                        source_id=scope.source_id,
                        payload={"reason": operation.reason or "support removed"},
                    )
                )
            else:
                mutations.append(
                    LifecycleMutation(
                        LifecycleMutationType.REFRESH_MEMORY_INDEX,
                        memory_id=memory_id,
                        source_id=scope.source_id,
                    )
                )
            continue

        if operation.action in {ReconcileAction.UPDATE, ReconcileAction.SUPERSEDE}:
            if operation.memory is None:
                raise ValueError("replacement operation requires a new Memory candidate")
            if not current_source_support:
                raise ValueError(f"replacement incumbent lacks current-scope support: {memory_id}")
            if gate_state is LifecycleGateState.GATED or external_support or operation.flag_for_review:
                replacement_id, creation_mutations = memory_creation_mutations(operation.memory)
                proposed_mutations = [*creation_mutations]
                if current_source_support:
                    proposed_mutations.append(
                        LifecycleMutation(
                            LifecycleMutationType.REMOVE_SUPPORT,
                            memory_id=memory_id,
                            source_id=scope.source_id,
                            evidence_reference_ids=current_source_support,
                            payload={"document_id": defaults.doc_id},
                        )
                    )
                if not external_support:
                    proposed_mutations.append(
                        LifecycleMutation(
                            LifecycleMutationType.SUPERSEDE_MEMORY,
                            memory_id=memory_id,
                            source_id=scope.source_id,
                            replacement_memory_id=replacement_id,
                            payload={
                                "reason": operation.reason or "authoritative replacement",
                                "replacement_kind": (
                                    "revision" if operation.action is ReconcileAction.UPDATE else "supersession"
                                ),
                            },
                        )
                    )
                else:
                    proposed_mutations.append(
                        LifecycleMutation(
                            LifecycleMutationType.REFRESH_MEMORY_INDEX,
                            memory_id=memory_id,
                            source_id=scope.source_id,
                        )
                    )
                decisions.append(
                    IncumbentDecision(memory_id, IncumbentDisposition.REVIEW, operation.reason or "review")
                )
                mutations.append(
                    _review_mutation(
                        scope,
                        operation,
                        memory_id,
                        proposed_mutations=tuple(proposed_mutations),
                        disposition=(
                            IncumbentDisposition.REMOVE_SUPPORT if external_support else IncumbentDisposition.SUPERSEDE
                        ),
                        replacement_memory_id=replacement_id,
                        relation_discovery_seed={
                            "memory_id": replacement_id,
                            "expected_content_hash": content_hash(operation.memory.content.strip()),
                            "source_id": scope.source_id,
                            "source_unit_id": scope.source_unit_id,
                            "source_unit_revision_id": scope.target_unit_revision_id,
                            "doc_id": defaults.doc_id,
                            "actor_user_id": defaults.actor_user_id,
                            "entity_ids": list(defaults.entity_ids),
                        },
                    )
                )
                continue
            replacement_id = create_memory(operation.memory)
            decisions.append(
                IncumbentDecision(
                    memory_id,
                    IncumbentDisposition.SUPERSEDE,
                    operation.reason or "authoritative replacement",
                    replacement_memory_id=replacement_id,
                )
            )
            mutations.extend(
                (
                    LifecycleMutation(
                        LifecycleMutationType.REMOVE_SUPPORT,
                        memory_id=memory_id,
                        source_id=scope.source_id,
                        evidence_reference_ids=current_source_support,
                        payload={"document_id": defaults.doc_id},
                    ),
                    LifecycleMutation(
                        LifecycleMutationType.SUPERSEDE_MEMORY,
                        memory_id=memory_id,
                        source_id=scope.source_id,
                        replacement_memory_id=replacement_id,
                        payload={
                            "reason": operation.reason or "authoritative replacement",
                            "replacement_kind": (
                                "revision" if operation.action is ReconcileAction.UPDATE else "supersession"
                            ),
                        },
                    ),
                )
            )
            continue
        raise ValueError(f"unsupported reconcile action: {operation.action.value}")

    batch_count = max(1, math.ceil(len(incumbent_ids) / max(1, incumbent_batch_size)))
    batch_ids = tuple(f"{scope.id}:batch:{index}" for index in range(batch_count))
    plan = LifecyclePlan(
        id=plan_id,
        scope=scope,
        gate_state=gate_state,
        coverage_proof=CoverageProof(
            mandatory_incumbent_ids=incumbent_ids,
            incumbent_decisions=tuple(decisions),
            batch_ids=batch_ids,
            completed_batch_ids=batch_ids,
        ),
        stale_guard=StaleGuard(
            observation_revision_ids=observation_revision_ids,
            support_set_hashes={
                memory_id: support_set_hashes[memory_id] for memory_id in (*incumbent_ids, *sorted(attached_target_ids))
            },
            memory_versions={
                memory_id: _memory_version(incumbents.get(memory_id) or corroboration_targets_by_id[memory_id])
                for memory_id in (*incumbent_ids, *sorted(attached_target_ids))
            },
        ),
        mutations=tuple(mutations),
        evidence_units=tuple(evidence_units),
        evidence_references=tuple(evidence_references),
        relation_discovery_requests=tuple(relation_discovery_requests),
    )
    plan.validate()
    return plan


def lifecycle_access_context_hash(
    *,
    visibility: str,
    owner_user_id: str | None,
    project_key: str | None,
    repo_identifier: str | None,
) -> str:
    """Return the canonical access identity shared by every lifecycle caller.

    ``project_key`` is intentionally not part of the identity: projects tune
    relevance and ownership routing, but do not create visibility boundaries.
    The parameter remains explicit so callers cannot accidentally invent a
    second access-hash contract.
    """

    del project_key

    return hashlib.sha256(
        "\x1f".join(
            (
                visibility,
                owner_user_id or "",
                repo_identifier or "",
            )
        ).encode("utf-8")
    ).hexdigest()


def lifecycle_plan_id(scope: ReconciliationScope) -> str:
    """Return the deterministic plan identity for one reconciliation scope."""

    digest = hashlib.sha256(
        "\x1f".join(
            (
                scope.id,
                scope.source_id,
                scope.source_unit_id,
                scope.target_unit_revision_id or "",
            )
        ).encode("utf-8")
    ).hexdigest()[:20]
    return f"lplan-{digest}"


def _review_mutation(
    scope: ReconciliationScope,
    operation: ReconcileOperation,
    memory_id: str,
    *,
    proposed_mutations: tuple[LifecycleMutation, ...],
    disposition: IncumbentDisposition,
    replacement_memory_id: str | None = None,
    relation_discovery_seed: Mapping[str, object] | None = None,
) -> LifecycleMutation:
    staged = {"action": operation.action.value}
    if operation.memory is not None:
        staged["candidate"] = {
            "content": operation.memory.content,
            "memory_type": operation.memory.memory_type,
            "confidence": operation.memory.confidence,
            "tags": operation.memory.tags,
        }
    staged["proposed_disposition"] = disposition.value
    staged["replacement_memory_id"] = replacement_memory_id
    staged["proposed_mutations"] = [_serialize_mutation(item) for item in proposed_mutations]
    if relation_discovery_seed is not None:
        staged["relation_discovery_seed"] = dict(relation_discovery_seed)
    return LifecycleMutation(
        LifecycleMutationType.CREATE_REVIEW,
        memory_id=memory_id,
        source_id=scope.source_id,
        payload={
            "review_id": _stable_id("review", scope.id, memory_id, operation.action.value),
            "reason": operation.reason or "lifecycle review required",
            "staged_evidence": staged,
        },
    )


def _serialize_mutation(mutation: LifecycleMutation) -> dict[str, object]:
    return {
        "mutation_type": mutation.mutation_type.value,
        "memory_id": mutation.memory_id,
        "source_id": mutation.source_id,
        "evidence_reference_ids": list(mutation.evidence_reference_ids),
        "replacement_memory_id": mutation.replacement_memory_id,
        "payload": dict(mutation.payload),
    }


def _memory_payload(raw: RawMemory, defaults: NewMemoryDefaults) -> dict[str, object]:
    content = raw.content.strip()
    valid_from = parse_memory_validity_date(raw.valid_from)
    valid_until = parse_memory_validity_date(raw.valid_until)
    return {
        "content": content,
        "content_hash": content_hash(content),
        "memory_type": raw.memory_type,
        "confidence": raw.confidence,
        "tags": list(raw.tags),
        "visibility": defaults.visibility,
        "owner_user_id": defaults.owner_user_id,
        "project_key": defaults.project_key,
        "repo_identifier": defaults.repo_identifier,
        "extraction_context": raw.extraction_context,
        "valid_from": valid_from.isoformat() if valid_from is not None else None,
        "valid_until": valid_until.isoformat() if valid_until is not None else None,
        "entity_refs": list(raw.entity_refs),
        "entity_ids": list(defaults.entity_ids),
        "document_source": {
            "doc_id": defaults.doc_id,
            "source_type": defaults.source_type,
            "excerpt": raw.extraction_context,
            "source_updated_at": defaults.source_updated_at,
        },
    }


def _new_memory_id(scope_id: str, raw: RawMemory) -> str:
    return _stable_id("mem", scope_id, raw.memory_type, raw.content.strip())


def _memory_version(memory: Memory) -> str:
    updated_at = memory.updated_at.isoformat() if memory.updated_at is not None else ""
    return _stable_id(
        "memory-version",
        memory.status,
        memory.content_hash,
        updated_at,
    )


def _stable_id(prefix: str, *values: object) -> str:
    digest = hashlib.sha256("\x1f".join(str(value) for value in values).encode("utf-8")).hexdigest()[:16]
    return f"{prefix}-{digest}"
