"""Current-revision recovery for legacy-limited Support.

This module never repairs historical lineage.  It either reconstructs an exact
stored part set or asks the configured structured LLM to select current,
application-owned Fragment references, then stages ordinary v2 ATTACH_SUPPORT
mutations for the existing Memory.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Mapping, Sequence

from memforge.llm.structured import LegacySupportRevalidationResponse
from memforge.memory.evidence import (
    EvidencePartKind,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    ResolvedEvidencePart,
    ResolvedEvidenceSelection,
    SupportScopeVersion,
    evidence_part_set_digest,
    evidence_unit_id_v2,
)
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
from memforge.models import Memory, RawMemory
from memforge.pipeline.evidence_fragments import COMPILER_CONTRACT_VERSION
from memforge.pipeline.projection_context import (
    ProjectionExtractionBatch,
    observation_is_inference_eligible,
)
from memforge.pipeline.projection_evidence import build_projected_claim_evidence
from memforge.pipeline.projection_fragments import (
    FragmentSelectionError,
    ProjectionFragmentCatalog,
    compile_projection_fragment_catalog,
)
from memforge.source_projection import AnchorKind, SourceAnchor, SourceProjection


LEGACY_SUPPORT_REVALIDATION_BATCH_SIZE = 20


class LegacySupportRecoveryDisposition(str, Enum):
    MECHANICALLY_RECOVERABLE = "mechanically_recoverable"
    SUPPORTED = "supported"
    NOT_SUPPORTED = "not_supported"
    INCONCLUSIVE = "inconclusive"
    ALREADY_SUPPORTED = "already_supported"
    HISTORICAL_ONLY = "historical_only"


@dataclass(frozen=True, slots=True)
class LegacySupportRecoveryCandidate:
    memory: Memory
    memory_version: str
    support_set_hash: str
    source_id: str
    source_type: str
    source_unit_id: str
    doc_id: str
    access_context_hash: str
    projection: SourceProjection | None
    legacy_evidence_unit_ids: tuple[str, ...]
    legacy_references: tuple[EvidenceReference, ...]
    reason_codes: tuple[str, ...]
    active_v2_unit_ids: tuple[str, ...] = ()
    inactive_v2_unit_ids: tuple[str, ...] = ()
    memory_source_current: bool = True
    legacy_support_active: bool = True

    def __post_init__(self) -> None:
        if not all(
            (
                self.memory.id,
                self.memory_version,
                self.support_set_hash,
                self.source_id,
                self.source_type,
                self.source_unit_id,
                self.doc_id,
                self.access_context_hash,
            )
        ):
            raise ValueError("legacy Support recovery candidate identity is incomplete")
        if self.projection is None:
            return
        unit_revision = self.projection.source_unit_revisions
        units = self.projection.source_units
        if len(units) != 1 or units[0].id != self.source_unit_id:
            raise ValueError("legacy Support recovery requires one current Source Unit")
        if len(unit_revision) != 1 or unit_revision[0].source_unit_id != self.source_unit_id:
            raise ValueError("legacy Support recovery requires one current Unit revision")


@dataclass(frozen=True, slots=True)
class LegacySupportRecoveryDecision:
    candidate: LegacySupportRecoveryCandidate
    disposition: LegacySupportRecoveryDisposition
    reason: str
    selection: ResolvedEvidenceSelection | None = None
    catalog_digest: str | None = None
    primary_ref: str | None = None
    required_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        supporting = self.disposition in {
            LegacySupportRecoveryDisposition.MECHANICALLY_RECOVERABLE,
            LegacySupportRecoveryDisposition.SUPPORTED,
        }
        if supporting != (self.selection is not None):
            raise ValueError("supporting recovery decisions require one resolved selection")
        if self.primary_ref is None and self.required_refs:
            raise ValueError("required Fragment refs require a Primary ref")


@dataclass(frozen=True, slots=True)
class LegacySupportRecoveryReportEntry:
    memory_id: str
    memory_version: str
    support_set_hash: str
    source_unit_id: str
    target_unit_revision_id: str
    doc_id: str
    access_context_hash: str
    legacy_evidence_unit_ids: tuple[str, ...]
    disposition: LegacySupportRecoveryDisposition
    reason: str
    catalog_digest: str | None = None
    primary_ref: str | None = None
    required_refs: tuple[str, ...] = ()

    def to_payload(self) -> Mapping[str, object]:
        return {
            "memory_id": self.memory_id,
            "memory_version": self.memory_version,
            "support_set_hash": self.support_set_hash,
            "source_unit_id": self.source_unit_id,
            "target_unit_revision_id": self.target_unit_revision_id,
            "doc_id": self.doc_id,
            "access_context_hash": self.access_context_hash,
            "legacy_evidence_unit_ids": list(self.legacy_evidence_unit_ids),
            "disposition": self.disposition.value,
            "reason": self.reason,
            "catalog_digest": self.catalog_digest,
            "primary_ref": self.primary_ref,
            "required_refs": list(self.required_refs),
        }

    def identity_payload(self) -> Mapping[str, object]:
        """Exclude explanatory model prose from exact apply identity."""

        payload = dict(self.to_payload())
        payload.pop("reason")
        return payload


@dataclass(frozen=True, slots=True)
class LegacySupportRecoveryReport:
    id: str
    source_id: str
    llm_model: str | None
    entries: tuple[LegacySupportRecoveryReportEntry, ...]
    created_at: str

    @property
    def ready_count(self) -> int:
        return sum(
            item.disposition
            in {
                LegacySupportRecoveryDisposition.MECHANICALLY_RECOVERABLE,
                LegacySupportRecoveryDisposition.SUPPORTED,
            }
            for item in self.entries
        )

    @property
    def legacy_group_count(self) -> int:
        return sum(len(item.legacy_evidence_unit_ids) for item in self.entries)

    @property
    def ready_group_count(self) -> int:
        return sum(
            len(item.legacy_evidence_unit_ids)
            for item in self.entries
            if item.disposition
            in {
                LegacySupportRecoveryDisposition.MECHANICALLY_RECOVERABLE,
                LegacySupportRecoveryDisposition.SUPPORTED,
            }
        )

    def to_payload(self) -> Mapping[str, object]:
        return {
            "kind": "legacy_support_recovery",
            "source_id": self.source_id,
            "llm_model": self.llm_model,
            "entries": [item.to_payload() for item in self.entries],
        }


@dataclass(frozen=True, slots=True)
class PreparedLegacySupportRecovery:
    report: LegacySupportRecoveryReport
    decisions: tuple[LegacySupportRecoveryDecision, ...]
    plans: tuple[LifecyclePlan, ...]


def compile_legacy_support_revalidation_catalog(
    candidate: LegacySupportRecoveryCandidate,
) -> ProjectionFragmentCatalog:
    """Compile all current inference-eligible text in one Source Unit."""

    projection = candidate.projection
    if projection is None:
        raise ValueError("current Source Unit revision is unavailable")
    revisions = {item.observation_id: item for item in projection.observation_revisions}
    primary_ids = tuple(
        observation.id
        for observation in projection.observations
        if observation.id in revisions
        and observation.observation_type != "binary_artifact"
        and observation_is_inference_eligible(
            observation.observation_type,
            revisions[observation.id].metadata,
        )
    )
    if not primary_ids:
        raise ValueError("current Source Unit has no inference-eligible text Evidence")
    batch = ProjectionExtractionBatch(
        id=(
            "legacy-support-revalidation-"
            + hashlib.sha256(
                "\x1f".join(
                    (
                        candidate.source_id,
                        candidate.source_unit_id,
                        projection.source_unit_revisions[0].id,
                        candidate.access_context_hash,
                    )
                ).encode("utf-8")
            ).hexdigest()[:20]
        ),
        source_unit_id=candidate.source_unit_id,
        primary_image_bytes=0,
        primary_observation_ids=primary_ids,
        primary_content_by_observation_id=tuple(
            (observation_id, revisions[observation_id].content) for observation_id in primary_ids
        ),
        context_observation_ids=(),
        context_observation_ids_by_primary=tuple((observation_id, ()) for observation_id in primary_ids),
        primary_markdown="\n\n".join(revisions[observation_id].content for observation_id in primary_ids),
        context_markdown="",
        primary_authority_spans=tuple(
            (observation_id, 0, revisions[observation_id].content)
            for observation_id in primary_ids
            if revisions[observation_id].content
        ),
    )
    return compile_projection_fragment_catalog(
        projection,
        batch,
        access_context_hash=candidate.access_context_hash,
    )


def legacy_support_revalidation_prompt(
    catalog: ProjectionFragmentCatalog,
    candidates: Sequence[LegacySupportRecoveryCandidate],
) -> str:
    """Return one bounded prompt containing judgments but no durable authority."""

    return (
        "You are revalidating existing MemForge Memories against the current authoritative "
        "Source Unit revision. For every request_position, decide supported, not_supported, "
        "or inconclusive. supported means the complete claim is entailed by the selected "
        "current Evidence. Select exactly one primary_ref and only indispensable required_refs "
        "from the authorized catalog. Do not rewrite the claim, infer missing history, use "
        "outside knowledge, or return Evidence text. not_supported means the current catalog "
        "does not support the claim. inconclusive means the available current Evidence is "
        "ambiguous or insufficient to decide. Return exactly one decision for every position.\n\n"
        + "AUTHORIZED_FRAGMENT_CATALOG\n"
        + json.dumps(catalog.model_payload(), ensure_ascii=False, separators=(",", ":"))
        + "\n\nMEMORIES\n"
        + json.dumps(
            [
                {
                    "request_position": index,
                    "memory_type": candidate.memory.memory_type,
                    "claim": candidate.memory.content,
                }
                for index, candidate in enumerate(candidates)
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def resolve_legacy_support_revalidation_response(
    *,
    catalog: ProjectionFragmentCatalog,
    candidates: Sequence[LegacySupportRecoveryCandidate],
    response: LegacySupportRevalidationResponse,
) -> tuple[LegacySupportRecoveryDecision, ...]:
    """Bind a complete model ledger to current application-owned Fragments."""

    by_position = {item.request_position: item for item in response.decisions}
    expected = set(range(len(candidates)))
    if len(by_position) != len(response.decisions) or set(by_position) != expected:
        raise ValueError("legacy Support revalidation response coverage is incomplete")
    output: list[LegacySupportRecoveryDecision] = []
    for position, candidate in enumerate(candidates):
        item = by_position[position]
        if item.decision == "supported":
            try:
                selection = catalog.resolve_selection(
                    primary_ref=item.primary_ref or "",
                    required_refs=item.required_refs,
                )
            except FragmentSelectionError as exc:
                raise ValueError(f"legacy Support revalidation selected invalid Evidence: {exc.code.value}") from exc
            output.append(
                LegacySupportRecoveryDecision(
                    candidate=candidate,
                    disposition=LegacySupportRecoveryDisposition.SUPPORTED,
                    reason=item.reason,
                    selection=selection,
                    catalog_digest=catalog.digest,
                    primary_ref=item.primary_ref,
                    required_refs=tuple(item.required_refs),
                )
            )
            continue
        output.append(
            LegacySupportRecoveryDecision(
                candidate=candidate,
                disposition=(
                    LegacySupportRecoveryDisposition.NOT_SUPPORTED
                    if item.decision == "not_supported"
                    else LegacySupportRecoveryDisposition.INCONCLUSIVE
                ),
                reason=item.reason,
            )
        )
    return tuple(output)


def resolve_mechanical_legacy_support(
    candidate: LegacySupportRecoveryCandidate,
) -> LegacySupportRecoveryDecision:
    """Rebuild exact mixed text/Artifact parts using each Reference profile."""

    projection = candidate.projection
    if projection is None:
        raise ValueError("current Source Unit revision is unavailable")
    revisions = {item.id: item for item in projection.observation_revisions}
    current_revision = projection.source_unit_revisions[0]
    if not candidate.legacy_references:
        raise ValueError("mechanical recovery requires legacy References")
    parts: list[ResolvedEvidencePart] = []
    for reference in candidate.legacy_references:
        if reference.role not in {EvidenceRole.PRIMARY, EvidenceRole.REQUIRED}:
            continue
        revision = revisions.get(reference.anchor.observation_revision_id)
        if revision is None or revision.id not in current_revision.observation_revision_ids:
            raise ValueError("legacy Reference is not current in the Source Unit revision")
        profile = revision.evidence_profile
        if profile is not None and profile.name == "binary-artifact":
            artifact = revision.metadata.get("source_artifact")
            if not isinstance(artifact, Mapping):
                raise ValueError("legacy Artifact lacks authoritative metadata")
            digest = str(artifact.get("sha256") or "").lower()
            if len(digest) != 64 or any(value not in "0123456789abcdef" for value in digest):
                raise ValueError("legacy Artifact lacks a valid digest")
            parts.append(
                ResolvedEvidencePart(
                    role=reference.role,
                    kind=EvidencePartKind.ARTIFACT,
                    anchor=SourceAnchor(
                        kind=AnchorKind.WHOLE_OBSERVATION,
                        observation_id=revision.observation_id,
                        observation_revision_id=revision.id,
                    ),
                    raw_content_sha256=digest,
                    presentation_sha256=hashlib.sha256(b"").hexdigest(),
                    artifact_metadata=dict(artifact),
                )
            )
            continue
        if profile is None:
            raise ValueError("legacy text Reference has no representation profile")
        if reference.anchor.kind is AnchorKind.WHOLE_OBSERVATION:
            start, end = 0, len(revision.content)
        elif reference.anchor.kind is AnchorKind.REVISION_RANGE:
            start, end = reference.anchor.range_start, reference.anchor.range_end
        else:
            raise ValueError("legacy stable Fragment cannot be mechanically recovered")
        if start is None or end is None or not 0 <= start < end <= len(revision.content):
            raise ValueError("legacy text Reference range is invalid")
        raw = revision.content[start:end]
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        parts.append(
            ResolvedEvidencePart(
                role=reference.role,
                kind=EvidencePartKind.TEXT,
                anchor=SourceAnchor(
                    kind=AnchorKind.REVISION_RANGE,
                    observation_id=revision.observation_id,
                    observation_revision_id=revision.id,
                    range_start=start,
                    range_end=end,
                ),
                raw_content_sha256=digest,
                presentation_sha256=digest,
                excerpt=raw,
            )
        )
    selection = ResolvedEvidenceSelection(
        source_id=candidate.source_id,
        source_unit_id=candidate.source_unit_id,
        target_unit_revision_id=current_revision.id,
        access_context_hash=candidate.access_context_hash,
        catalog_digest=(
            "legacy-mechanical-"
            + hashlib.sha256(
                "\x1f".join(
                    sorted(
                        f"{part.role.value}:{part.anchor.observation_revision_id}:{part.raw_content_sha256}"
                        for part in parts
                    )
                ).encode("utf-8")
            ).hexdigest()
        ),
        compiler_contract_version=COMPILER_CONTRACT_VERSION,
        parts=tuple(parts),
    )
    return LegacySupportRecoveryDecision(
        candidate=candidate,
        disposition=LegacySupportRecoveryDisposition.MECHANICALLY_RECOVERABLE,
        reason="legacy parts reconstructed from current immutable revisions",
        selection=selection,
        catalog_digest=selection.catalog_digest,
    )


async def prepare_legacy_support_recovery(
    db,
    *,
    source_id: str,
    structured_llm_client,
    llm_model: str | None = None,
) -> PreparedLegacySupportRecovery:
    """Produce and persist one exact report without applying Support."""

    candidates = await db.list_legacy_support_recovery_candidates(source_id)
    decisions: list[LegacySupportRecoveryDecision] = []
    pending_by_scope: dict[
        tuple[str, str, str],
        list[LegacySupportRecoveryCandidate],
    ] = {}
    for candidate in candidates:
        if not candidate.legacy_support_active:
            decisions.append(
                LegacySupportRecoveryDecision(
                    candidate=candidate,
                    disposition=LegacySupportRecoveryDisposition.HISTORICAL_ONLY,
                    reason="inactive_legacy_support",
                )
            )
            continue
        if candidate.memory.status != "active":
            decisions.append(
                LegacySupportRecoveryDecision(
                    candidate=candidate,
                    disposition=LegacySupportRecoveryDisposition.HISTORICAL_ONLY,
                    reason=f"terminal_memory:{candidate.memory.status}",
                )
            )
            continue
        if not candidate.memory_source_current:
            decisions.append(
                LegacySupportRecoveryDecision(
                    candidate=candidate,
                    disposition=LegacySupportRecoveryDisposition.INCONCLUSIVE,
                    reason="memory_source_projection_incomplete",
                )
            )
            continue
        if candidate.active_v2_unit_ids:
            decisions.append(
                LegacySupportRecoveryDecision(
                    candidate=candidate,
                    disposition=LegacySupportRecoveryDisposition.ALREADY_SUPPORTED,
                    reason="Memory already has active v2 Support",
                )
            )
            continue
        if candidate.projection is None:
            decisions.append(
                LegacySupportRecoveryDecision(
                    candidate=candidate,
                    disposition=LegacySupportRecoveryDisposition.INCONCLUSIVE,
                    reason="current_source_unit_revision_unavailable",
                )
            )
            continue
        if (
            "part_unresolvable" in candidate.reason_codes
            and "unit_revision_lineage_invalid" not in candidate.reason_codes
        ):
            try:
                decisions.append(resolve_mechanical_legacy_support(candidate))
                continue
            except ValueError:
                # Exact current revalidation remains safe; only the historical
                # mechanical conversion is unavailable.
                pass
        key = (
            candidate.source_unit_id,
            candidate.projection.source_unit_revisions[0].id,
            candidate.access_context_hash,
        )
        pending_by_scope.setdefault(key, []).append(candidate)

    if pending_by_scope and structured_llm_client is None:
        for pending in pending_by_scope.values():
            decisions.extend(
                LegacySupportRecoveryDecision(
                    candidate=candidate,
                    disposition=LegacySupportRecoveryDisposition.INCONCLUSIVE,
                    reason="structured_llm_client_unavailable",
                )
                for candidate in pending
            )
    else:
        llm_failure_reason: str | None = None
        for pending in pending_by_scope.values():
            if llm_failure_reason is not None:
                decisions.extend(
                    LegacySupportRecoveryDecision(
                        candidate=candidate,
                        disposition=LegacySupportRecoveryDisposition.INCONCLUSIVE,
                        reason=llm_failure_reason,
                    )
                    for candidate in pending
                )
                continue
            try:
                catalog = compile_legacy_support_revalidation_catalog(pending[0])
            except ValueError as exc:
                decisions.extend(
                    LegacySupportRecoveryDecision(
                        candidate=candidate,
                        disposition=LegacySupportRecoveryDisposition.INCONCLUSIVE,
                        reason=f"catalog_unavailable:{exc}",
                    )
                    for candidate in pending
                )
                continue
            if not catalog.usable:
                reason = "catalog_unusable:" + ",".join(sorted({error.code.value for error in catalog.errors}))
                decisions.extend(
                    LegacySupportRecoveryDecision(
                        candidate=candidate,
                        disposition=LegacySupportRecoveryDisposition.INCONCLUSIVE,
                        reason=reason,
                        catalog_digest=catalog.digest,
                    )
                    for candidate in pending
                )
                continue
            for offset in range(0, len(pending), LEGACY_SUPPORT_REVALIDATION_BATCH_SIZE):
                batch = pending[offset : offset + LEGACY_SUPPORT_REVALIDATION_BATCH_SIZE]
                try:
                    response = await structured_llm_client.revalidate_legacy_support(
                        legacy_support_revalidation_prompt(catalog, batch),
                        max_tokens=8192,
                        model=llm_model,
                    )
                    resolved = resolve_legacy_support_revalidation_response(
                        catalog=catalog,
                        candidates=batch,
                        response=response,
                    )
                except Exception as exc:
                    llm_failure_reason = f"llm_revalidation_failed:{type(exc).__name__}"
                    decisions.extend(
                        LegacySupportRecoveryDecision(
                            candidate=candidate,
                            disposition=LegacySupportRecoveryDisposition.INCONCLUSIVE,
                            reason=llm_failure_reason,
                            catalog_digest=catalog.digest,
                        )
                        for candidate in pending[offset:]
                    )
                    break
                decisions.extend(resolved)

    safe_decisions: list[LegacySupportRecoveryDecision] = []
    for item in decisions:
        if item.selection is None:
            safe_decisions.append(item)
            continue
        target_unit_id = _recovery_evidence_unit_id(item)
        if target_unit_id in item.candidate.inactive_v2_unit_ids:
            safe_decisions.append(
                replace(
                    item,
                    disposition=LegacySupportRecoveryDisposition.INCONCLUSIVE,
                    reason="previously_removed_v2_support",
                    selection=None,
                    catalog_digest=None,
                    primary_ref=None,
                    required_refs=(),
                )
            )
            continue
        safe_decisions.append(item)

    ordered = tuple(
        sorted(
            safe_decisions,
            key=lambda item: (
                item.candidate.source_unit_id,
                item.candidate.access_context_hash,
                item.candidate.memory.id,
            ),
        )
    )
    entries = tuple(
        LegacySupportRecoveryReportEntry(
            memory_id=item.candidate.memory.id,
            memory_version=item.candidate.memory_version,
            support_set_hash=item.candidate.support_set_hash,
            source_unit_id=item.candidate.source_unit_id,
            target_unit_revision_id=(
                item.candidate.projection.source_unit_revisions[0].id if item.candidate.projection is not None else ""
            ),
            doc_id=item.candidate.doc_id,
            access_context_hash=item.candidate.access_context_hash,
            legacy_evidence_unit_ids=item.candidate.legacy_evidence_unit_ids,
            disposition=item.disposition,
            reason=item.reason,
            catalog_digest=item.catalog_digest,
            primary_ref=item.primary_ref,
            required_refs=item.required_refs,
        )
        for item in ordered
    )
    identity = {
        "kind": "legacy_support_recovery",
        "source_id": source_id,
        "llm_model": llm_model,
        "entries": [entry.identity_payload() for entry in entries],
    }
    report_id = (
        "legacy-support-recovery-"
        + hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:24]
    )
    report = LegacySupportRecoveryReport(
        id=report_id,
        source_id=source_id,
        llm_model=llm_model,
        entries=entries,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    grouped_decisions: dict[tuple[str, str, str], list[LegacySupportRecoveryDecision]] = {}
    for item in ordered:
        if item.candidate.projection is None:
            continue
        key = (
            item.candidate.source_unit_id,
            item.candidate.projection.source_unit_revisions[0].id,
            item.candidate.access_context_hash,
        )
        grouped_decisions.setdefault(key, []).append(item)
    gate = await db.get_lifecycle_gate(source_id)
    plans = tuple(
        plan
        for group in grouped_decisions.values()
        if (
            plan := build_legacy_support_recovery_plan(
                decisions=group,
                gate_state=gate.state,
                report_id=report.id,
            )
        )
        is not None
    )
    await db.persist_legacy_support_recovery_report(report)
    return PreparedLegacySupportRecovery(
        report=report,
        decisions=ordered,
        plans=plans,
    )


def _recovery_evidence_unit_id(
    decision: LegacySupportRecoveryDecision,
) -> str:
    selection = decision.selection
    if selection is None:
        raise ValueError("recovery Evidence Unit identity requires resolved selection")
    references = tuple(
        EvidenceReference(
            role=part.role,
            anchor=part.anchor,
            kind=part.kind,
            raw_content_sha256=part.raw_content_sha256,
            presentation_sha256=part.presentation_sha256,
            excerpt=part.excerpt,
            artifact_metadata=dict(part.artifact_metadata),
        )
        for part in selection.parts
    )
    return evidence_unit_id_v2(
        source_unit_id=selection.source_unit_id,
        claim_content=decision.candidate.memory.content,
        part_set_digest=evidence_part_set_digest(references),
        access_context_hash=selection.access_context_hash,
    )


async def apply_prepared_legacy_support_recovery(
    db,
    prepared: PreparedLegacySupportRecovery,
    *,
    expected_report_id: str,
) -> int:
    """Apply only an exact, freshly reproduced recovery report."""

    if prepared.report.id != expected_report_id:
        raise ValueError("legacy Support recovery report is stale")
    applied = 0
    for plan in prepared.plans:
        await db.apply_lifecycle_plan(plan)
        applied += sum(item.mutation_type is LifecycleMutationType.ATTACH_SUPPORT for item in plan.mutations)
    return applied


def build_legacy_support_recovery_plan(
    *,
    decisions: Sequence[LegacySupportRecoveryDecision],
    gate_state: LifecycleGateState,
    report_id: str,
) -> LifecyclePlan | None:
    """Stage only non-destructive v2 ATTACH_SUPPORT mutations."""

    if not decisions:
        return None
    first = decisions[0].candidate
    if first.projection is None:
        raise ValueError("legacy Support recovery Plan requires current Source projection")
    scope_key = (
        first.source_id,
        first.source_unit_id,
        first.access_context_hash,
        first.projection.source_unit_revisions[0].id,
    )
    if any(
        (
            item.candidate.source_id,
            item.candidate.source_unit_id,
            item.candidate.access_context_hash,
            item.candidate.projection.source_unit_revisions[0].id if item.candidate.projection is not None else "",
        )
        != scope_key
        for item in decisions
    ):
        raise ValueError("legacy Support recovery Plan requires one current Unit/access scope")
    supporting = [item for item in decisions if item.selection is not None]
    if not supporting:
        return None
    units: dict[str, EvidenceUnit] = {}
    references: dict[str, EvidenceReference] = {}
    mutations: list[LifecycleMutation] = []
    for item in supporting:
        candidate = item.candidate
        if candidate.projection is None:
            raise ValueError("supporting recovery decision lacks current Source projection")
        raw = RawMemory(
            content=candidate.memory.content,
            memory_type=candidate.memory.memory_type,
            confidence=candidate.memory.confidence,
            entity_refs=list(candidate.memory.entity_refs),
            resolved_evidence_selection=item.selection,
            support_validation={
                "legacy_support_recovery": True,
                "recovery_report_id": report_id,
                "recovery_disposition": item.disposition.value,
            },
        )
        evidence = build_projected_claim_evidence(
            projection=candidate.projection,
            raw_memories=(raw,),
            doc_id=candidate.doc_id,
            source_type=candidate.source_type,
            project_key=candidate.memory.project_key,
            visibility=candidate.memory.visibility,
            owner_user_id=candidate.memory.owner_user_id,
            repo_identifier=candidate.memory.repo_identifier,
            access_context_hash=candidate.access_context_hash,
            extractor_run_id=report_id,
            support_scope_version=SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
        )
        unit_ids = tuple(evidence.evidence_unit_ids_by_claim_hash.values())
        flattened_unit_ids = tuple(value for values in unit_ids for value in values)
        if len(flattened_unit_ids) != 1:
            raise ValueError("legacy Support recovery must materialize one Evidence Unit")
        for unit in evidence.units:
            units.setdefault(unit.id, unit)
        for reference in evidence.references:
            references.setdefault(str(reference.id), reference)
        mutations.extend(
            (
                LifecycleMutation(
                    LifecycleMutationType.ATTACH_SUPPORT,
                    memory_id=candidate.memory.id,
                    source_id=candidate.source_id,
                    evidence_unit_ids=flattened_unit_ids,
                    payload={
                        "access_context_hash": candidate.access_context_hash,
                        "support_validation": dict(raw.support_validation),
                    },
                ),
                LifecycleMutation(
                    LifecycleMutationType.REFRESH_MEMORY_INDEX,
                    memory_id=candidate.memory.id,
                    source_id=candidate.source_id,
                ),
            )
        )
    digest = hashlib.sha256(
        "\x1f".join(
            (
                report_id,
                *sorted(units),
                *sorted(item.candidate.memory.id for item in decisions),
            )
        ).encode("utf-8")
    ).hexdigest()[:20]
    target_revision_id = first.projection.source_unit_revisions[0].id
    return LifecyclePlan(
        id=f"legacy-support-recovery-{digest}",
        scope=ReconciliationScope(
            id=f"legacy-support-recovery-scope-{digest}",
            source_id=first.source_id,
            source_unit_id=first.source_unit_id,
            base_unit_revision_id=target_revision_id,
            target_unit_revision_id=target_revision_id,
        ),
        gate_state=gate_state,
        coverage_proof=CoverageProof(
            mandatory_incumbent_ids=tuple(item.candidate.memory.id for item in decisions),
            incumbent_decisions=tuple(
                IncumbentDecision(
                    memory_id=item.candidate.memory.id,
                    disposition=IncumbentDisposition.KEEP,
                    reason=(item.reason or item.disposition.value),
                )
                for item in decisions
            ),
            batch_ids=(report_id,),
            completed_batch_ids=(report_id,),
        ),
        stale_guard=StaleGuard(
            observation_revision_ids=tuple(first.projection.source_unit_revisions[0].observation_revision_ids),
            support_set_hashes={item.candidate.memory.id: item.candidate.support_set_hash for item in decisions},
            support_scope_version=SupportScopeVersion.EVIDENCE_UNIT_SET_V2,
            memory_versions={item.candidate.memory.id: item.candidate.memory_version for item in decisions},
        ),
        mutations=tuple(mutations),
        evidence_units=tuple(units.values()),
        evidence_references=tuple(references.values()),
    )
