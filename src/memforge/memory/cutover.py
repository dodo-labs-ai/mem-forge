"""Conservative legacy-lineage backfill for per-source lifecycle cutover."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass
from collections.abc import Awaitable, Callable
from typing import Any

from memforge.memory.evidence import (
    EvidenceContentProvenance,
    EvidenceReference,
    EvidenceRole,
    EvidenceUnit,
    MemorySupportAssertion,
)
from memforge.memory.lifecycle_plan import (
    CutoverFindingReason,
    CutoverFindingStatus,
    LegacyMemoryProvenance,
    LifecycleCutoverFinding,
    LifecycleBackfillJob,
    LifecycleBackfillJobStatus,
)
from memforge.source_projection import AnchorKind, SourceAnchor, SourceObservationRevision


@dataclass(frozen=True, slots=True)
class CutoverBackfillResult:
    source_id: str
    scanned_memories: int
    mapped_memories: int
    finding_count: int
    gate_enabled: bool


async def run_source_lifecycle_backfill_job(
    db: Any,
    source_id: str,
    *,
    job_id: str | None = None,
) -> LifecycleBackfillJob:
    """Run backfill through a durable operator-visible job state machine."""

    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id=job_id or f"lifecycle-backfill-{uuid.uuid4().hex}",
            source_id=source_id,
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    if job.status is LifecycleBackfillJobStatus.COMPLETED:
        return job
    if job.status is not LifecycleBackfillJobStatus.QUEUED:
        raise ValueError(f"lifecycle backfill job is already {job.status.value}")
    await db.start_lifecycle_backfill_job(job.id)
    try:
        result = await run_source_lifecycle_backfill(db, source_id)
        return await db.complete_lifecycle_backfill_job(
            job.id,
            scanned_memories=result.scanned_memories,
            mapped_memories=result.mapped_memories,
            finding_count=result.finding_count,
        )
    except Exception as exc:
        await db.fail_lifecycle_backfill_job(job.id, error=str(exc))
        raise


async def run_source_lifecycle_recovery_job(
    db: Any,
    source_id: str,
    *,
    job_id: str,
    reextract_documents: Callable[[frozenset[str]], Awaitable[None]],
) -> LifecycleBackfillJob:
    """Audit, selectively re-extract identifiable documents, then re-audit.

    Similarity never establishes lineage.  The recovery callback receives only
    document identifiers already present in durable source provenance.  A
    successful callback is not enough to close a finding: the second audit
    must persist and validate Source Unit/Observation support first.
    """

    job = await db.create_lifecycle_backfill_job(
        LifecycleBackfillJob(
            id=job_id,
            source_id=source_id,
            status=LifecycleBackfillJobStatus.QUEUED,
        )
    )
    if job.status is LifecycleBackfillJobStatus.COMPLETED:
        return job
    if job.status is not LifecycleBackfillJobStatus.QUEUED:
        raise ValueError(f"lifecycle backfill job is already {job.status.value}")
    await db.start_lifecycle_backfill_job(job.id)
    try:
        result = await run_source_lifecycle_backfill(db, source_id)
        if result.finding_count:
            target_document_ids = await _identifiable_finding_document_ids(db, source_id)
            if target_document_ids:
                await reextract_documents(target_document_ids)
                result = await run_source_lifecycle_backfill(db, source_id)
        return await db.complete_lifecycle_backfill_job(
            job.id,
            scanned_memories=result.scanned_memories,
            mapped_memories=result.mapped_memories,
            finding_count=result.finding_count,
        )
    except Exception as exc:
        await db.fail_lifecycle_backfill_job(job.id, error=str(exc))
        raise


async def run_source_lifecycle_backfill(db: Any, source_id: str) -> CutoverBackfillResult:
    """Map legacy Memory provenance to persisted Source Observation lineage.

    The service never uses similarity as lineage evidence. It accepts an exact
    document-to-unit locator plus either an exact excerpt match or a unit with a
    single current observation. Ambiguous or unavailable mappings create a
    durable finding and keep the source destructive lifecycle gate closed.
    """

    candidates = await db.list_legacy_memory_provenance(source_id)
    by_memory: dict[str, list[LegacyMemoryProvenance]] = defaultdict(list)
    for candidate in candidates:
        by_memory[candidate.memory_id].append(candidate)

    mapped = 0
    findings = 0
    for memory_id, provenance_rows in sorted(by_memory.items()):
        finding_id = _stable_id("finding", source_id, memory_id)
        mapped_lineage: tuple[str, str] | None = None
        attempts: list[dict[str, object]] = []
        terminal_reason = CutoverFindingReason.OBSERVATION_NOT_FOUND

        for provenance in provenance_rows:
            source_unit = await db.find_source_unit_by_document_id(source_id, provenance.doc_id)
            if source_unit is None:
                attempts.append({"doc_id": provenance.doc_id, "result": "source_unit_not_found"})
                terminal_reason = CutoverFindingReason.MISSING_SOURCE_PROVENANCE
                continue
            revisions = await db.get_current_source_observation_revisions(source_unit.id)
            selected = _select_observation_revision(provenance, revisions)
            if selected is None:
                attempts.append(
                    {
                        "doc_id": provenance.doc_id,
                        "source_unit_id": source_unit.id,
                        "result": "ambiguous_observation" if len(revisions) > 1 else "observation_not_found",
                        "candidate_observation_ids": sorted(revisions),
                    }
                )
                terminal_reason = (
                    CutoverFindingReason.AMBIGUOUS_OBSERVATION
                    if len(revisions) > 1
                    else CutoverFindingReason.OBSERVATION_NOT_FOUND
                )
                continue

            observation_id, revision = selected
            evidence_unit_id = _stable_id("eu-backfill", source_id, memory_id, revision.id)
            access_hash = _access_context_hash(provenance)
            unit = EvidenceUnit(
                id=evidence_unit_id,
                source_id=source_id,
                doc_id=provenance.doc_id,
                doc_revision_id=revision.id,
                source_type=provenance.source_type,
                source_anchor=observation_id,
                source_lineage_id=source_unit.id,
                project_key=provenance.project_key,
                visibility=provenance.visibility,
                owner_user_id=provenance.owner_user_id,
                repo_identifier=provenance.repo_identifier,
                content=revision.content,
                excerpt=provenance.excerpt,
                evidence_provenance=EvidenceContentProvenance.SOURCE_EXCERPT,
                source_metadata={"backfill": True, "source_unit_id": source_unit.id},
                access_context_hash=access_hash,
            )
            await db.upsert_evidence_unit(unit)
            references = await db.record_evidence_references(
                unit.id,
                (
                    EvidenceReference(
                        role=EvidenceRole.PRIMARY,
                        anchor=SourceAnchor(
                            kind=AnchorKind.WHOLE_OBSERVATION,
                            observation_id=observation_id,
                            observation_revision_id=revision.id,
                        ),
                        evidence_unit_id=unit.id,
                    ),
                ),
            )
            reference = references[0]
            await db.upsert_memory_support_assertion(
                MemorySupportAssertion(
                    id=_stable_id("support", memory_id, reference.id),
                    memory_id=memory_id,
                    evidence_reference_id=reference.id or "",
                    source_id=source_id,
                    access_context_hash=access_hash,
                )
            )
            mapped_lineage = (observation_id, source_unit.id)
            attempts.append(
                {
                    "doc_id": provenance.doc_id,
                    "source_unit_id": source_unit.id,
                    "observation_id": observation_id,
                    "result": "mapped",
                }
            )
            break

        if mapped_lineage is not None:
            mapped += 1
            existing = await db.get_lifecycle_cutover_finding(finding_id)
            if existing is not None and existing.status is CutoverFindingStatus.OPEN:
                await db.resolve_lifecycle_cutover_finding(
                    finding_id,
                    observation_id=mapped_lineage[0],
                    source_unit_id=mapped_lineage[1],
                )
            continue

        findings += 1
        await db.upsert_lifecycle_cutover_finding(
            LifecycleCutoverFinding(
                id=finding_id,
                source_id=source_id,
                memory_id=memory_id,
                reason=terminal_reason,
                status=CutoverFindingStatus.OPEN,
                available_provenance={
                    "documents": [
                        {
                            "doc_id": row.doc_id,
                            "source_type": row.source_type,
                            "excerpt": row.excerpt,
                        }
                        for row in provenance_rows
                    ]
                },
                mapping_attempt={"strategy": "exact_document_locator_then_excerpt", "attempts": attempts},
            )
        )

    gate_enabled = False
    if findings == 0:
        await db.enable_lifecycle_gate(source_id)
        gate_enabled = True
    else:
        await db.gate_destructive_lifecycle(
            source_id,
            reason=f"{findings} open lifecycle cutover finding(s)",
        )
    return CutoverBackfillResult(
        source_id=source_id,
        scanned_memories=len(by_memory),
        mapped_memories=mapped,
        finding_count=findings,
        gate_enabled=gate_enabled,
    )


def _select_observation_revision(
    provenance: LegacyMemoryProvenance,
    revisions: dict[str, SourceObservationRevision] | Any,
) -> tuple[str, SourceObservationRevision] | None:
    if not revisions:
        return None
    excerpt = (provenance.excerpt or "").strip()
    if excerpt:
        exact = [
            (observation_id, revision)
            for observation_id, revision in revisions.items()
            if excerpt in revision.content
        ]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            return None
    if len(revisions) == 1:
        return next(iter(revisions.items()))
    return None


async def _identifiable_finding_document_ids(
    db: Any,
    source_id: str,
) -> frozenset[str]:
    findings = await db.list_lifecycle_cutover_findings(
        source_id,
        status=CutoverFindingStatus.OPEN,
    )
    document_ids: set[str] = set()
    for finding in findings:
        documents = finding.available_provenance.get("documents", [])
        if not isinstance(documents, list):
            continue
        for document in documents:
            if not isinstance(document, dict):
                continue
            doc_id = document.get("doc_id")
            if isinstance(doc_id, str) and doc_id.strip():
                document_ids.add(doc_id.strip())
    return frozenset(document_ids)


def _access_context_hash(provenance: LegacyMemoryProvenance) -> str:
    payload = json.dumps(
        {
            "visibility": provenance.visibility,
            "owner_user_id": provenance.owner_user_id,
            "project_key": provenance.project_key,
            "repo_identifier": provenance.repo_identifier,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _stable_id(prefix: str, *values: object) -> str:
    digest = hashlib.sha256("\x1f".join(str(value) for value in values).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-{digest}"
