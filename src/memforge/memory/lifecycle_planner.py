"""Translate a complete reconciliation ledger into one atomic Lifecycle Plan."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from memforge.memory.evidence import (
    EvidenceReference,
    EvidenceUnit,
)
from memforge.memory.lifecycle_plan import (
    CoverageProof,
    IncumbentAuthority,
    IncumbentAuthorityGrant,
    IncumbentDecision,
    IncumbentDisposition,
    LifecycleMutation,
    LifecycleMutationType,
    LifecyclePlan,
    ReconciliationScope,
    StaleGuard,
    lifecycle_stable_id,
)
from memforge.memory.relation_discovery_contract import (
    RelationDiscoveryRequest,
    relation_discovery_request_id,
    resolve_relation_discovery_actor_user_id,
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
    entity_ids_by_claim_hash: Mapping[str, tuple[int, ...]] | None = None
    source_updated_at: str | None = None


def build_lifecycle_plan(
    *,
    plan_id: str,
    scope: ReconciliationScope,
    operations: Sequence[ReconcileOperation],
    incumbents: Mapping[str, Memory],
    source_support_unit_ids: Mapping[str, tuple[str, ...]],
    all_active_support_unit_ids: Mapping[str, tuple[str, ...]],
    support_set_hashes: Mapping[str, str],
    observation_revision_ids: tuple[str, ...],
    new_evidence_unit_ids: tuple[str, ...] = (),
    evidence_unit_ids_by_claim_hash: Mapping[str, tuple[str, ...]] | None = None,
    defaults: NewMemoryDefaults,
    evidence_units: Sequence[EvidenceUnit] = (),
    evidence_references: Sequence[EvidenceReference] = (),
    incumbent_batch_size: int = 30,
    incumbent_authority_grants: Mapping[str, IncumbentAuthorityGrant] | None = None,
) -> LifecyclePlan:
    """Build a complete plan without performing any storage mutation."""

    incumbent_ids = tuple(sorted(incumbents))
    authority_grants = dict(incumbent_authority_grants or {})
    unknown_authority_incumbents = set(authority_grants).difference(incumbents)
    if unknown_authority_incumbents:
        raise ValueError(
            "lifecycle authority targets an unknown incumbent: "
            f"{sorted(unknown_authority_incumbents)}"
        )

    def incumbent_authority(memory_id: str) -> IncumbentAuthority:
        grant = authority_grants.get(memory_id)
        return grant.authority if grant is not None else IncumbentAuthority.CURRENT_SOURCE_SUPPORT

    def incumbent_authority_actor(memory_id: str) -> str | None:
        grant = authority_grants.get(memory_id)
        return grant.actor_id if grant is not None else None
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
    relation_discovery_actor_user_id = resolve_relation_discovery_actor_user_id(
        visibility=defaults.visibility,
        owner_user_id=defaults.owner_user_id,
        requested_actor_user_id=defaults.actor_user_id,
    )

    def entity_ids_for(raw: RawMemory) -> tuple[int, ...]:
        if defaults.entity_ids_by_claim_hash is None:
            return ()
        return defaults.entity_ids_by_claim_hash.get(content_hash(raw.content.strip()), ())

    def request_relation_discovery(
        memory_id: str,
        expected_content_hash: str,
        entity_ids: tuple[int, ...] = (),
    ) -> None:
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
                actor_user_id=relation_discovery_actor_user_id,
                entity_ids=entity_ids,
            )
        )

    def support_ids_for(raw: RawMemory) -> tuple[str, ...]:
        return _claim_support_ids(raw, evidence_unit_ids_by_claim_hash, new_evidence_unit_ids)

    def create_memory(raw: RawMemory) -> str:
        memory_id = _new_memory_id(scope.id, raw)
        if memory_id in created_ids:
            return memory_id
        support_ids = support_ids_for(raw)
        if not support_ids:
            raise ValueError("new Memory candidate lacks support-granting evidence")
        mutations.extend(
            (
                LifecycleMutation(
                    LifecycleMutationType.CREATE_MEMORY,
                    memory_id=memory_id,
                    source_id=scope.source_id,
                    payload={"memory": _memory_payload(raw, defaults, entity_ids=entity_ids_for(raw))},
                ),
                LifecycleMutation(
                    LifecycleMutationType.ATTACH_SUPPORT,
                    memory_id=memory_id,
                    source_id=scope.source_id,
                    evidence_unit_ids=support_ids,
                    payload={
                        "access_context_hash": defaults.access_context_hash,
                        "source_updated_at": defaults.source_updated_at,
                    },
                ),
            )
        )
        created_ids.add(memory_id)
        request_relation_discovery(
            memory_id,
            content_hash(raw.content.strip()),
            entity_ids_for(raw),
        )
        return memory_id

    for operation in add_operations:
        assert operation.memory is not None
        create_memory(operation.memory)

    for memory_id in incumbent_ids:
        operation = by_incumbent[memory_id]
        current_source_support = source_support_unit_ids.get(memory_id, ())
        all_support = all_active_support_unit_ids.get(memory_id, ())
        external_support = set(all_support).difference(current_source_support)

        if operation.support_revalidation_skipped and (
            operation.action is not ReconcileAction.NOOP or operation.memory is not None
        ):
            raise ValueError("skipped Support revalidation requires a bare NOOP")
        if operation.action is ReconcileAction.NOOP:
            decisions.append(IncumbentDecision(
                memory_id, IncumbentDisposition.KEEP, operation.reason or "kept",
                skipped_support_ids=tuple(current_source_support) if operation.support_revalidation_skipped else (),
            ))
            support_ids = support_ids_for(operation.memory) if operation.memory is not None else ()
            if support_ids:
                if _rebind_removes_support(current_source_support, support_ids):
                    mutations.append(
                        LifecycleMutation(
                            LifecycleMutationType.REMOVE_SUPPORT,
                            memory_id=memory_id,
                            source_id=scope.source_id,
                            evidence_unit_ids=current_source_support,
                        )
                    )
                mutations.append(
                    LifecycleMutation(
                        LifecycleMutationType.ATTACH_SUPPORT,
                        memory_id=memory_id,
                        source_id=scope.source_id,
                        evidence_unit_ids=support_ids,
                        payload={
                            "access_context_hash": defaults.access_context_hash,
                            "source_updated_at": defaults.source_updated_at,
                            "support_validation": dict(operation.memory.support_validation),
                        },
                    )
                )
            continue

        if operation.action is ReconcileAction.DELETE:
            evidence_free_action = memory_id in authority_grants
            if not current_source_support and not evidence_free_action:
                raise ValueError(f"destructive incumbent lacks current-scope support: {memory_id}")
            decisions.append(
                IncumbentDecision(
                    memory_id,
                    IncumbentDisposition.REMOVE_SUPPORT,
                    operation.reason or "source evidence removed",
                    authority=incumbent_authority(memory_id),
                    authority_actor_id=incumbent_authority_actor(memory_id),
                )
            )
            if current_source_support:
                mutations.append(
                    LifecycleMutation(
                        LifecycleMutationType.REMOVE_SUPPORT,
                        memory_id=memory_id,
                        source_id=scope.source_id,
                        evidence_unit_ids=current_source_support,
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
            grant = authority_grants.get(memory_id)
            if grant is not None and grant.authority is IncumbentAuthority.MAINTENANCE_OPERATOR:
                raise ValueError(
                    f"maintenance operator cannot replace incumbent: {memory_id}"
                )
            if not current_source_support:
                raise ValueError(f"replacement incumbent lacks current-scope support: {memory_id}")
            replacement_id = create_memory(operation.memory)
            if external_support:
                # Another Source Unit still supports the old Memory: this Unit revises
                # only its own Support, the Candidate becomes its own Memory, and
                # relation discovery relates the pair.
                decisions.append(
                    IncumbentDecision(
                        memory_id,
                        IncumbentDisposition.REMOVE_SUPPORT,
                        operation.reason or "revised by this source; other sources still support it",
                        authority=incumbent_authority(memory_id),
                        authority_actor_id=incumbent_authority_actor(memory_id),
                    )
                )
                if current_source_support:
                    mutations.append(
                        LifecycleMutation(
                            LifecycleMutationType.REMOVE_SUPPORT,
                            memory_id=memory_id,
                            source_id=scope.source_id,
                            evidence_unit_ids=current_source_support,
                            payload={"document_id": defaults.doc_id},
                        )
                    )
                mutations.append(
                    LifecycleMutation(
                        LifecycleMutationType.REFRESH_MEMORY_INDEX,
                        memory_id=memory_id,
                        source_id=scope.source_id,
                    )
                )
                continue
            decisions.append(
                IncumbentDecision(
                    memory_id,
                    IncumbentDisposition.SUPERSEDE,
                    operation.reason or "authoritative replacement",
                    replacement_memory_id=replacement_id,
                    authority=incumbent_authority(memory_id),
                    authority_actor_id=incumbent_authority_actor(memory_id),
                )
            )
            if current_source_support:
                mutations.append(
                    LifecycleMutation(
                        LifecycleMutationType.REMOVE_SUPPORT,
                        memory_id=memory_id,
                        source_id=scope.source_id,
                        evidence_unit_ids=current_source_support,
                        payload={"document_id": defaults.doc_id},
                    )
                )
            mutations.append(
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
            continue
        raise ValueError(f"unsupported reconcile action: {operation.action.value}")

    batch_count = max(1, math.ceil(len(incumbent_ids) / max(1, incumbent_batch_size)))
    batch_ids = tuple(f"{scope.id}:batch:{index}" for index in range(batch_count))
    plan = LifecyclePlan(
        id=plan_id,
        scope=scope,
        coverage_proof=CoverageProof(
            mandatory_incumbent_ids=incumbent_ids,
            incumbent_decisions=tuple(decisions),
            batch_ids=batch_ids,
            completed_batch_ids=batch_ids,
        ),
        stale_guard=StaleGuard(
            observation_revision_ids=observation_revision_ids,
            support_set_hashes={
                memory_id: support_set_hashes[memory_id] for memory_id in incumbent_ids
            },
            memory_versions={
                memory_id: lifecycle_memory_version(incumbents[memory_id]) for memory_id in incumbent_ids
            },
        ),
        mutations=tuple(mutations),
        evidence_units=tuple(evidence_units),
        evidence_references=tuple(evidence_references),
        relation_discovery_requests=tuple(relation_discovery_requests),
    )
    plan.validate()
    return plan


def _claim_support_ids(
    raw: RawMemory,
    evidence_unit_ids_by_claim_hash: Mapping[str, tuple[str, ...]] | None,
    fallback: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """The Evidence Units this round built for a claim."""
    if evidence_unit_ids_by_claim_hash is not None:
        units = evidence_unit_ids_by_claim_hash.get(content_hash(raw.content.strip()), ())
        if units:
            return units
    return fallback


def _rebind_removes_support(current_source_support: Sequence[str], support_ids: Sequence[str]) -> bool:
    """A kept old Memory's rebind replaces this source's current Support with other Evidence Units."""
    return bool(support_ids) and bool(current_source_support) and set(support_ids) != set(current_source_support)


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


def _memory_payload(
    raw: RawMemory,
    defaults: NewMemoryDefaults,
    *,
    entity_ids: tuple[int, ...],
) -> dict[str, object]:
    content = raw.content.strip()
    canonical_excerpt = raw.evidence_quote
    valid_from = parse_memory_validity_date(raw.valid_from)
    valid_until = parse_memory_validity_date(raw.valid_until)
    return {
        "content": content,
        "content_hash": content_hash(content),
        "memory_type": raw.memory_type,
        "visibility": defaults.visibility,
        "owner_user_id": defaults.owner_user_id,
        "project_key": defaults.project_key,
        "repo_identifier": defaults.repo_identifier,
        "valid_from": valid_from.isoformat() if valid_from is not None else None,
        "valid_until": valid_until.isoformat() if valid_until is not None else None,
        "entity_refs": list(raw.entity_refs),
        "entity_ids": list(entity_ids),
        "document_source": {
            "doc_id": defaults.doc_id,
            "source_type": defaults.source_type,
            "excerpt": canonical_excerpt,
            "source_updated_at": defaults.source_updated_at,
        },
    }


def _new_memory_id(scope_id: str, raw: RawMemory) -> str:
    return lifecycle_stable_id("mem", scope_id, raw.memory_type, raw.content.strip())


def lifecycle_memory_version(memory: Memory) -> str:
    """Return the adapter-neutral stale-guard identity for one Memory."""

    updated_at = memory.updated_at.isoformat() if memory.updated_at is not None else ""
    return lifecycle_stable_id(
        "memory-version",
        memory.status,
        memory.content_hash,
        updated_at,
    )

