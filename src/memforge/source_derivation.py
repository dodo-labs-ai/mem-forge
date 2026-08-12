"""Durable staging contracts for Source Unit Memory derivation."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Literal, Protocol

from memforge.models import DocumentRecord, MemoryExtractionResult, RawMemory
from memforge.pipeline.bounded_work import collect_bounded
from memforge.pipeline.extraction_contract import (
    PROJECTION_EXTRACTION_CONTRACT_VERSION,
)
from memforge.pipeline.document_units import (
    ExtractionContext,
    ExtractionContextPacker,
    UnitizationPolicy,
    unitize_markdown,
)
from memforge.pipeline.projection_context import (
    ProjectionExtractionBatch,
    observation_is_inference_eligible,
    plan_projection_extraction_batches,
)
from memforge.source_artifacts import SourceArtifactSummary
from memforge.source_projection import (
    SourceProjection,
    source_projection_to_payload,
)


SOURCE_DERIVATION_CONTRACT_VERSION = PROJECTION_EXTRACTION_CONTRACT_VERSION

SOURCE_DERIVATION_PENDING = "pending"
SOURCE_DERIVATION_RETRYABLE_FAILURE = "retryable_failure"
SOURCE_DERIVATION_COMPLETED = "completed"
SOURCE_DERIVATION_APPLIED = "applied"
SOURCE_DERIVATION_SUPERSEDED = "superseded"

SOURCE_DERIVATION_BATCH_PENDING = "pending"
SOURCE_DERIVATION_BATCH_COMPLETED = "completed"
SOURCE_DERIVATION_BATCH_RETRYABLE_FAILURE = "retryable_failure"

_SAFE_DERIVATION_DIAGNOSTIC_RE = re.compile(r"^[A-Za-z0-9_.\[\]$-]+$")
_MAX_SAFE_DERIVATION_ERROR_FIELDS = 32
_EVIDENCE_BLOCK_FALLBACK_SAMPLE_LIMIT = 16


@dataclass(frozen=True, slots=True)
class SourceDerivationBatchManifest:
    batch_id: str
    input_payload_hash: str
    primary_observation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceDerivationManifest:
    id: str
    source_id: str
    source_unit_id: str
    base_unit_revision_id: str | None
    target_unit_revision_id: str
    projection_payload: Mapping[str, object]
    projection_payload_hash: str
    projection_identity_hash: str
    context_payload: Mapping[str, object]
    context_payload_hash: str
    context_identity_hash: str
    extraction_contract_version: str
    batches: tuple[SourceDerivationBatchManifest, ...]


@dataclass(frozen=True, slots=True)
class SourceDerivationBatchRecord:
    batch_id: str
    input_payload_hash: str
    primary_observation_ids: tuple[str, ...]
    status: str
    output_payload_hash: str | None
    error_type: str | None
    error_code: str | None
    error_fields: tuple[tuple[str, str], ...]
    attempt_count: int


@dataclass(frozen=True, slots=True)
class SourceDerivationAttempt:
    id: str
    source_id: str
    source_unit_id: str
    base_unit_revision_id: str | None
    target_unit_revision_id: str
    projection: SourceProjection
    projection_payload_hash: str
    projection_identity_hash: str
    context: SourceUnitDerivationContext
    context_payload_hash: str
    context_identity_hash: str
    extraction_contract_version: str
    status: str
    batches: tuple[SourceDerivationBatchRecord, ...]
    created_at: str
    updated_at: str
    completed_at: str | None


@dataclass(frozen=True, slots=True)
class SourceUnitDerivationContext:
    document: DocumentRecord
    doc_type: str
    project_key: str | None
    repo_identifier: str | None
    document_content: str
    update_mode: str
    changed_hunks: str | None
    update_plan_stats: Mapping[str, object] | None
    source_updated_at: str | None
    user_id: str | None
    source_activity_epoch: int | None
    current_changed_ranges: tuple[tuple[int, int], ...] = ()
    reprocess_all_current_observations: bool = False
    work_strategy: Literal["auto", "structural"] = "auto"


@dataclass(frozen=True, slots=True)
class DiffGuidedExtractionBatch:
    """One changed-range work item for a single textual Source Observation."""

    id: str
    source_unit_id: str
    primary_observation_ids: tuple[str, ...]
    changed_hunks: str
    updated_document: str
    kind: Literal["diff_guided"] = "diff_guided"


@dataclass(frozen=True, slots=True)
class StructuralExtractionBatch:
    """One deterministic Markdown unit in full-document extraction."""

    id: str
    source_unit_id: str
    primary_observation_ids: tuple[str, ...]
    context: ExtractionContext
    kind: Literal["structural_unit"] = "structural_unit"


SourceDerivationBatch = ProjectionExtractionBatch | DiffGuidedExtractionBatch | StructuralExtractionBatch


class SourceDerivationStore(Protocol):
    async def stage_source_derivation(
        self,
        manifest: SourceDerivationManifest,
    ) -> SourceDerivationAttempt: ...

    async def get_completed_source_derivation_batch_results(
        self,
        *,
        derivation_id: str,
    ) -> dict[str, MemoryExtractionResult]: ...

    async def record_source_derivation_batch_result(
        self,
        *,
        derivation_id: str,
        batch_id: str,
        result: MemoryExtractionResult,
    ) -> SourceDerivationAttempt: ...

    async def supersede_source_derivation(
        self,
        derivation_id: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class SourceUnitDerivationRequest:
    projection: SourceProjection
    context: SourceUnitDerivationContext
    extract_batch: Callable[
        [SourceDerivationBatch],
        Awaitable[MemoryExtractionResult],
    ]
    max_concurrent: int


@dataclass(frozen=True, slots=True)
class SourceUnitDerivationResult:
    derivation: SourceDerivationAttempt
    extraction: MemoryExtractionResult
    reused_batch_count: int
    executed_batch_count: int


class SourceUnitDeriver:
    """Own durable, idempotent derivation for one immutable Source Unit target."""

    def __init__(
        self,
        store: SourceDerivationStore,
        *,
        plan_work: Callable[
            [SourceProjection, SourceUnitDerivationContext],
            tuple[SourceDerivationBatch, ...],
        ]
        | None = None,
    ) -> None:
        self._store = store
        self._plan_work = plan_work or plan_source_derivation_work

    async def derive(
        self,
        request: SourceUnitDerivationRequest,
    ) -> SourceUnitDerivationResult:
        batches = self._plan_work(request.projection, request.context)
        manifest = source_derivation_manifest(
            request.projection,
            batches,
            context=request.context,
        )
        derivation = await self._store.stage_source_derivation(manifest)
        completed_results = await self._store.get_completed_source_derivation_batch_results(
            derivation_id=derivation.id,
        )
        pending_batches = tuple(batch for batch in batches if batch.id not in completed_results)

        async def extract_and_persist(
            batch: SourceDerivationBatch,
        ) -> MemoryExtractionResult:
            try:
                result = await request.extract_batch(batch)
            except Exception as exc:
                if not isinstance(batch, DiffGuidedExtractionBatch):
                    raise
                result = MemoryExtractionResult(
                    error_type="diff_guided_extraction_error",
                    metadata={"safe_error_code": type(exc).__name__},
                )
            for sample in result.metadata.get(
                "evidence_block_fallback_samples",
                [],
            ):
                if isinstance(sample, dict):
                    sample["source_derivation_batch_id"] = batch.id
            await self._store.record_source_derivation_batch_result(
                derivation_id=derivation.id,
                batch_id=batch.id,
                result=result,
            )
            return result

        pending_results = await collect_bounded(
            pending_batches,
            extract_and_persist,
            max_concurrent=request.max_concurrent,
        )
        derivation = await self._store.stage_source_derivation(manifest)
        new_results_by_batch_id = {
            batch.id: result
            for batch, result in zip(
                pending_batches,
                pending_results,
                strict=True,
            )
        }
        ordered_results = tuple(
            completed_results.get(batch.id) or new_results_by_batch_id[batch.id] for batch in batches
        )
        extraction = _assemble_derivation_results(
            projection=request.projection,
            results=ordered_results,
        )
        extraction.metadata = {
            **extraction.metadata,
            "derivation_work_kinds": sorted({_derivation_batch_kind(batch) for batch in batches}),
            "reused_derivation_batch_count": len(completed_results),
            "executed_derivation_batch_count": len(pending_batches),
        }
        if (
            request.context.work_strategy == "auto"
            and len(batches) == 1
            and isinstance(batches[0], DiffGuidedExtractionBatch)
            and extraction.error_type
        ):
            fallback = await self.derive(
                replace(
                    request,
                    context=replace(
                        request.context,
                        work_strategy="structural",
                    ),
                )
            )
            fallback.extraction.metadata = {
                **fallback.extraction.metadata,
                "diff_guided_fallback": True,
                "failed_diff_derivation_count": 1,
            }
            if not fallback.extraction.error_type:
                await self._store.supersede_source_derivation(derivation.id)
            return fallback
        structural_batches = tuple(batch for batch in batches if isinstance(batch, StructuralExtractionBatch))
        if structural_batches:
            units = tuple(batch.context.unit for batch in structural_batches)
            extraction.metadata.update(
                {
                    "unitized": True,
                    "unit_count": len(units),
                    "failed_unit_count": extraction.metadata.get(
                        "failed_batch_count",
                        0,
                    ),
                    "segmentation_version": units[0].segmentation_version,
                    "partition_strategy": "recursive_fit_first",
                    "max_unit_input_tokens": (UnitizationPolicy().max_unit_input_tokens),
                }
            )
        return SourceUnitDerivationResult(
            derivation=derivation,
            extraction=extraction,
            reused_batch_count=len(completed_results),
            executed_batch_count=len(pending_batches),
        )


def plan_source_derivation_work(
    projection: SourceProjection,
    context: SourceUnitDerivationContext,
) -> tuple[SourceDerivationBatch, ...]:
    """Plan provider-neutral extraction work for one immutable Source Unit."""

    projection_batches = plan_projection_extraction_batches(
        projection,
        primary_observation_ids=(
            tuple(observation.id for observation in projection.observations)
            if context.reprocess_all_current_observations
            else None
        ),
    )
    if context.reprocess_all_current_observations:
        return projection_batches
    if len(projection.observations) != 1 or len(projection.observation_revisions) != 1:
        return projection_batches

    observation = projection.observations[0]
    revision = projection.observation_revisions[0]
    if (
        revision.observation_id != observation.id
        or observation.observation_type == "binary_artifact"
        or not observation_is_inference_eligible(
            observation.observation_type,
            revision.metadata,
        )
    ):
        return projection_batches

    changed_observation_ids = {
        anchor.observation_id for delta in projection.deltas for anchor in delta.changed_anchors
    } | {observation_id for delta in projection.deltas for observation_id in delta.added_observation_ids}
    if changed_observation_ids != {observation.id}:
        return projection_batches

    if (
        context.work_strategy == "structural"
        or context.update_mode != "diff_guided"
        or not context.changed_hunks
    ):
        policy = UnitizationPolicy()
        units = unitize_markdown(
            context.document_content,
            doc_id=context.document.doc_id,
            policy=policy,
        )
        packer = ExtractionContextPacker(units)
        target_revision_id = projection.source_unit_revisions[0].id
        return tuple(
            StructuralExtractionBatch(
                id=(
                    "ubatch-"
                    + hashlib.sha256(
                        (
                            f"{SOURCE_DERIVATION_CONTRACT_VERSION}\x1f"
                            f"{target_revision_id}\x1fstructural_unit\x1f"
                            f"{unit.unit_id}\x1f{unit.content_fingerprint}"
                        ).encode("utf-8")
                    ).hexdigest()[:16]
                ),
                source_unit_id=projection.source_units[0].id,
                primary_observation_ids=(observation.id,),
                context=packer.pack(
                    document_title=context.document.title,
                    document_url=context.document.source_url,
                    source_type=projection.source_type,
                    unit=unit,
                    entities=[],
                ),
            )
            for unit in units
        )

    target_revision_id = projection.source_unit_revisions[0].id
    digest = hashlib.sha256(
        (
            f"{SOURCE_DERIVATION_CONTRACT_VERSION}\x1f"
            f"{target_revision_id}\x1fdiff_guided\x1f"
            f"{hashlib.sha256(context.changed_hunks.encode('utf-8')).hexdigest()}"
        ).encode("utf-8")
    ).hexdigest()[:16]
    return (
        DiffGuidedExtractionBatch(
            id=f"dbatch-{digest}",
            source_unit_id=projection.source_units[0].id,
            primary_observation_ids=(observation.id,),
            changed_hunks=context.changed_hunks,
            updated_document=context.document_content,
        ),
    )


def source_derivation_manifest(
    projection: SourceProjection,
    batches: tuple[SourceDerivationBatch, ...],
    *,
    context: SourceUnitDerivationContext,
    extraction_contract_version: str = SOURCE_DERIVATION_CONTRACT_VERSION,
) -> SourceDerivationManifest:
    """Build one stable manifest for an immutable target Source Unit revision."""

    if len(projection.deltas) != 1 or len(projection.source_unit_revisions) != 1:
        raise ValueError("Source derivation requires exactly one Source Unit revision")
    delta = projection.deltas[0]
    target_revision_id = delta.current_unit_revision_id
    if not target_revision_id:
        raise ValueError("Source derivation requires a target Source Unit revision")
    if projection.source_unit_revisions[0].id != target_revision_id:
        raise ValueError("Source derivation target does not match projected Source Unit revision")
    projection_payload = source_projection_to_payload(projection)
    projection_payload_json = _canonical_json(projection_payload)
    projection_payload_hash = hashlib.sha256(projection_payload_json.encode("utf-8")).hexdigest()
    projection_identity_hash = source_derivation_projection_identity_hash(projection)
    context_payload = source_unit_derivation_context_to_payload(context)
    context_payload_hash = hashlib.sha256(_canonical_json(context_payload).encode("utf-8")).hexdigest()
    context_identity_hash = source_derivation_context_identity_hash(context)
    manifest_batches = tuple(
        SourceDerivationBatchManifest(
            batch_id=batch.id,
            input_payload_hash=_batch_input_payload_hash(
                target_revision_id=target_revision_id,
                extraction_contract_version=extraction_contract_version,
                batch=batch,
            ),
            primary_observation_ids=batch.primary_observation_ids,
        )
        for batch in batches
    )
    identity_payload = {
        "source_id": projection.source_id,
        "source_unit_id": delta.source_unit_id,
        "base_unit_revision_id": delta.previous_unit_revision_id,
        "target_unit_revision_id": target_revision_id,
        "projection_identity_hash": projection_identity_hash,
        "context_identity_hash": context_identity_hash,
        "extraction_contract_version": extraction_contract_version,
        "batch_input_hashes": [item.input_payload_hash for item in manifest_batches],
    }
    derivation_id = "sdrv-" + hashlib.sha256(_canonical_json(identity_payload).encode("utf-8")).hexdigest()[:32]
    return SourceDerivationManifest(
        id=derivation_id,
        source_id=projection.source_id,
        source_unit_id=delta.source_unit_id,
        base_unit_revision_id=delta.previous_unit_revision_id,
        target_unit_revision_id=target_revision_id,
        projection_payload=projection_payload,
        projection_payload_hash=projection_payload_hash,
        projection_identity_hash=projection_identity_hash,
        context_payload=context_payload,
        context_payload_hash=context_payload_hash,
        context_identity_hash=context_identity_hash,
        extraction_contract_version=extraction_contract_version,
        batches=manifest_batches,
    )


def source_unit_derivation_context_to_payload(
    context: SourceUnitDerivationContext,
) -> dict[str, object]:
    return {
        "document": _document_record_payload(context.document),
        "doc_type": context.doc_type,
        "project_key": context.project_key,
        "repo_identifier": context.repo_identifier,
        "document_content": context.document_content,
        "update_mode": context.update_mode,
        "changed_hunks": context.changed_hunks,
        "update_plan_stats": (dict(context.update_plan_stats) if context.update_plan_stats is not None else None),
        "source_updated_at": context.source_updated_at,
        "user_id": context.user_id,
        "source_activity_epoch": context.source_activity_epoch,
        "current_changed_ranges": [[start, end] for start, end in context.current_changed_ranges],
        "reprocess_all_current_observations": (
            context.reprocess_all_current_observations
        ),
        "work_strategy": context.work_strategy,
    }


def source_derivation_projection_identity_hash(
    projection: SourceProjection,
) -> str:
    """Hash stable source truth while excluding operational observation fields."""

    payload = source_projection_to_payload(projection)
    return hashlib.sha256(_canonical_json(_projection_identity_payload(payload)).encode("utf-8")).hexdigest()


def source_derivation_context_identity_hash(
    context: SourceUnitDerivationContext,
) -> str:
    """Hash the stable Document and lifecycle inputs authorized for reuse."""

    payload = source_unit_derivation_context_to_payload(context)
    return hashlib.sha256(_canonical_json(_derivation_context_identity_payload(payload)).encode("utf-8")).hexdigest()


def source_derivation_document_identity_hash(
    document: DocumentRecord,
) -> str:
    """Hash the stable portion of a staged Document snapshot."""

    stable_payload = _document_record_payload(document)
    for field in ("last_synced", "created_at", "updated_at"):
        stable_payload.pop(field, None)
    return hashlib.sha256(_canonical_json(stable_payload).encode("utf-8")).hexdigest()


def _document_record_payload(
    document: DocumentRecord,
) -> dict[str, object]:
    return {
        "doc_id": document.doc_id,
        "source": document.source,
        "source_url": document.source_url,
        "title": document.title,
        "space_or_project": document.space_or_project,
        "author": document.author,
        "last_modified": document.last_modified.isoformat(),
        "labels": list(document.labels),
        "version": document.version,
        "content_hash": document.content_hash,
        "token_count": document.token_count,
        "raw_content_uri": document.raw_content_uri,
        "raw_content_type": document.raw_content_type,
        "normalized_content_uri": document.normalized_content_uri,
        "pdf_content_uri": document.pdf_content_uri,
        "last_synced": document.last_synced.isoformat(),
        "client": document.client,
        "created_at": (document.created_at.isoformat() if document.created_at is not None else None),
        "updated_at": (document.updated_at.isoformat() if document.updated_at is not None else None),
    }


def source_unit_derivation_context_from_payload(
    payload: Mapping[str, object],
) -> SourceUnitDerivationContext:
    raw_document = payload.get("document")
    if not isinstance(raw_document, Mapping):
        raise ValueError("Source derivation context has no Document snapshot")
    last_modified = datetime.fromisoformat(str(raw_document["last_modified"]))
    return SourceUnitDerivationContext(
        document=DocumentRecord(
            doc_id=str(raw_document["doc_id"]),
            source=str(raw_document["source"]),
            source_url=str(raw_document.get("source_url") or ""),
            title=str(raw_document.get("title") or ""),
            space_or_project=str(raw_document.get("space_or_project") or ""),
            author=_optional_string(raw_document.get("author")),
            last_modified=last_modified,
            labels=_string_list(raw_document.get("labels")),
            version=str(raw_document.get("version") or ""),
            content_hash=str(raw_document["content_hash"]),
            token_count=(int(raw_document["token_count"]) if raw_document.get("token_count") is not None else None),
            raw_content_uri=_optional_string(raw_document.get("raw_content_uri")),
            raw_content_type=_optional_string(raw_document.get("raw_content_type")),
            normalized_content_uri=_optional_string(raw_document.get("normalized_content_uri")),
            pdf_content_uri=_optional_string(raw_document.get("pdf_content_uri")),
            last_synced=datetime.fromisoformat(str(raw_document["last_synced"])),
            client=_optional_string(raw_document.get("client")),
            created_at=(
                datetime.fromisoformat(str(raw_document["created_at"]))
                if raw_document.get("created_at") is not None
                else None
            ),
            updated_at=(
                datetime.fromisoformat(str(raw_document["updated_at"]))
                if raw_document.get("updated_at") is not None
                else None
            ),
        ),
        doc_type=str(payload["doc_type"]),
        project_key=_optional_string(payload.get("project_key")),
        repo_identifier=_optional_string(payload.get("repo_identifier")),
        document_content=str(payload.get("document_content") or ""),
        update_mode=str(payload.get("update_mode") or "full_document"),
        changed_hunks=_optional_string(payload.get("changed_hunks")),
        update_plan_stats=(
            dict(payload["update_plan_stats"]) if isinstance(payload.get("update_plan_stats"), Mapping) else None
        ),
        source_updated_at=_optional_string(payload.get("source_updated_at")),
        user_id=_optional_string(payload.get("user_id")),
        source_activity_epoch=(
            int(payload["source_activity_epoch"]) if payload.get("source_activity_epoch") is not None else None
        ),
        current_changed_ranges=tuple(
            (int(value[0]), int(value[1]))
            for value in payload.get("current_changed_ranges", [])
            if isinstance(value, list) and len(value) == 2
        ),
        reprocess_all_current_observations=bool(
            payload.get("reprocess_all_current_observations", False)
        ),
        work_strategy=(
            "structural"
            if payload.get("work_strategy") == "structural"
            else "auto"
        ),
    )


def memory_extraction_output_payload(
    result: MemoryExtractionResult,
) -> dict[str, object]:
    """Serialize only validated derivation output needed for deterministic reuse."""

    if result.error_type:
        raise ValueError("failed derivation batches have no reusable output payload")
    return {
        "memories": [_raw_memory_payload(memory) for memory in result.memories],
        "artifact_summaries": [
            {
                "source_observation_id": summary.source_observation_id,
                "summary": summary.summary,
            }
            for summary in result.artifact_summaries
        ],
        "evidence_telemetry": _safe_evidence_telemetry(result.metadata),
    }


def memory_extraction_result_from_output_payload(
    payload: Mapping[str, object],
) -> MemoryExtractionResult:
    """Restore one previously validated derivation batch result."""

    raw_memories = payload.get("memories")
    raw_summaries = payload.get("artifact_summaries")
    if not isinstance(raw_memories, list) or not isinstance(raw_summaries, list):
        raise ValueError("Source derivation output payload is invalid")
    memories = []
    for value in raw_memories:
        if not isinstance(value, Mapping):
            raise ValueError("Source derivation Memory payload is invalid")
        memories.append(
            RawMemory(
                content=str(value.get("content") or ""),
                memory_type=str(value.get("memory_type") or ""),
                confidence=float(value.get("confidence", 0.7)),
                entity_refs=_string_list(value.get("entity_refs")),
                valid_from=_optional_string(value.get("valid_from")),
                valid_until=_optional_string(value.get("valid_until")),
                extraction_context=_optional_string(value.get("extraction_context")),
                evidence_quote=_optional_string(value.get("evidence_quote")),
                evidence_resolved_from_block=(
                    value.get("evidence_resolved_from_block") is True
                ),
                evidence_range_start=_optional_int(value.get("evidence_range_start")),
                evidence_range_end=_optional_int(value.get("evidence_range_end")),
                evidence_anchor=_optional_string(value.get("evidence_anchor")),
                source_observation_id=_optional_string(value.get("source_observation_id")),
                required_source_observation_ids=_string_list(value.get("required_source_observation_ids")),
                support_validation=(
                    dict(value.get("support_validation") or {})
                    if isinstance(value.get("support_validation"), Mapping)
                    else {}
                ),
            )
        )
    summaries = []
    for value in raw_summaries:
        if not isinstance(value, Mapping):
            raise ValueError("Source derivation Artifact summary payload is invalid")
        summaries.append(
            SourceArtifactSummary(
                source_observation_id=str(value.get("source_observation_id") or ""),
                summary=str(value.get("summary") or ""),
            )
        )
    return MemoryExtractionResult(
        memories=memories,
        artifact_summaries=tuple(summaries),
        metadata=_safe_evidence_telemetry(payload.get("evidence_telemetry")),
    )


def output_payload_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def safe_derivation_error(result: MemoryExtractionResult) -> tuple[str, str | None, tuple[tuple[str, str], ...]]:
    """Return content-free error identity suitable for durable telemetry."""

    error_type = _safe_derivation_diagnostic(
        result.error_type,
        fallback="unknown_error",
    )
    raw_code = result.metadata.get("safe_error_code")
    error_code = _safe_derivation_diagnostic(raw_code)
    raw_fields = result.metadata.get("safe_validation_fields")
    fields: list[tuple[str, str]] = []
    if isinstance(raw_fields, list):
        for value in raw_fields:
            if len(fields) >= _MAX_SAFE_DERIVATION_ERROR_FIELDS:
                break
            if not isinstance(value, Mapping):
                continue
            location = _safe_derivation_diagnostic(
                value.get("location"),
            )
            kind = _safe_derivation_diagnostic(value.get("type"))
            if location and kind:
                fields.append((location, kind))
    return error_type, error_code, tuple(fields)


def _safe_derivation_diagnostic(
    value: object,
    *,
    fallback: str | None = None,
) -> str | None:
    if not isinstance(value, str):
        return fallback
    normalized = value.strip()
    if not normalized or len(normalized) > 255 or _SAFE_DERIVATION_DIAGNOSTIC_RE.fullmatch(normalized) is None:
        return fallback
    return normalized


def _batch_input_payload_hash(
    *,
    target_revision_id: str,
    extraction_contract_version: str,
    batch: SourceDerivationBatch,
) -> str:
    if isinstance(batch, DiffGuidedExtractionBatch):
        payload = {
            "target_unit_revision_id": target_revision_id,
            "extraction_contract_version": extraction_contract_version,
            "batch_id": batch.id,
            "work_kind": batch.kind,
            "primary_observation_ids": list(batch.primary_observation_ids),
            "changed_hunks_sha256": hashlib.sha256(batch.changed_hunks.encode("utf-8")).hexdigest(),
            "updated_document_sha256": hashlib.sha256(batch.updated_document.encode("utf-8")).hexdigest(),
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    if isinstance(batch, StructuralExtractionBatch):
        unit = batch.context.unit
        payload = {
            "target_unit_revision_id": target_revision_id,
            "extraction_contract_version": extraction_contract_version,
            "batch_id": batch.id,
            "work_kind": batch.kind,
            "primary_observation_ids": list(batch.primary_observation_ids),
            "unit_id": unit.unit_id,
            "unit_content_sha256": hashlib.sha256(unit.unit_markdown.encode("utf-8")).hexdigest(),
            "heading_path": list(unit.heading_path),
            "segmentation_version": unit.segmentation_version,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    payload = {
        "target_unit_revision_id": target_revision_id,
        "extraction_contract_version": extraction_contract_version,
        "batch_id": batch.id,
        "primary_observation_ids": list(batch.primary_observation_ids),
        "primary_content_sha256": hashlib.sha256(batch.primary_markdown.encode("utf-8")).hexdigest(),
        "context_observation_ids": list(batch.context_observation_ids),
        "context_content_sha256": hashlib.sha256(batch.context_markdown.encode("utf-8")).hexdigest(),
        "primary_image_bytes": batch.primary_image_bytes,
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _derivation_batch_kind(batch: SourceDerivationBatch) -> str:
    if isinstance(batch, DiffGuidedExtractionBatch):
        return batch.kind
    if isinstance(batch, StructuralExtractionBatch):
        return batch.kind
    return "projection_batch"


def _raw_memory_payload(memory: RawMemory) -> dict[str, object]:
    return {
        "content": memory.content,
        "memory_type": memory.memory_type,
        "confidence": memory.confidence,
        "entity_refs": list(memory.entity_refs),
        "valid_from": memory.valid_from,
        "valid_until": memory.valid_until,
        "extraction_context": memory.extraction_context,
        "evidence_quote": memory.evidence_quote,
        # evidence_block_id is deliberately absent: batch-local addresses are
        # resolved to exact source excerpts before durable output is staged.
        "evidence_resolved_from_block": memory.evidence_resolved_from_block,
        "evidence_range_start": memory.evidence_range_start,
        "evidence_range_end": memory.evidence_range_end,
        "evidence_anchor": memory.evidence_anchor,
        "source_observation_id": memory.source_observation_id,
        "required_source_observation_ids": list(memory.required_source_observation_ids),
        "support_validation": dict(memory.support_validation),
    }


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None else None


def _optional_int(value: object) -> int | None:
    return int(value) if value is not None else None


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _projection_identity_payload(
    projection_payload: Mapping[str, object],
) -> dict[str, object]:
    """Remove observation-event fields from an immutable revision identity."""

    identity = json.loads(_canonical_json(projection_payload))
    identity["run_id"] = "<source-derivation>"
    identity["checkpoint"] = {}
    for collection_name in (
        "observation_revisions",
        "source_unit_revisions",
    ):
        values = identity.get(collection_name)
        if isinstance(values, list):
            for value in values:
                if isinstance(value, dict):
                    value["observed_at"] = None
                    metadata = value.get("metadata")
                    if isinstance(metadata, dict):
                        source_artifact = metadata.get("source_artifact")
                        if isinstance(source_artifact, dict):
                            source_artifact.pop("summary", None)
    return identity


def _derivation_context_identity_payload(
    context_payload: Mapping[str, object],
) -> dict[str, object]:
    document = context_payload.get("document")
    if not isinstance(document, Mapping):
        raise ValueError("Source derivation context has no Document snapshot")
    # work_strategy is an operational fallback choice. Batch input hashes
    # distinguish its manifest without changing the lifecycle context identity.
    return {
        "doc_id": document.get("doc_id"),
        "source": document.get("source"),
        "content_hash": document.get("content_hash"),
        "doc_type": context_payload.get("doc_type"),
        "project_key": context_payload.get("project_key"),
        "repo_identifier": context_payload.get("repo_identifier"),
        "document_content_sha256": hashlib.sha256(
            str(context_payload.get("document_content") or "").encode("utf-8")
        ).hexdigest(),
        "update_mode": context_payload.get("update_mode"),
        "changed_hunks_sha256": (
            hashlib.sha256(str(context_payload["changed_hunks"]).encode("utf-8")).hexdigest()
            if context_payload.get("changed_hunks") is not None
            else None
        ),
        "update_plan_stats": context_payload.get("update_plan_stats"),
        "source_updated_at": context_payload.get("source_updated_at"),
        "user_id": context_payload.get("user_id"),
        "source_activity_epoch": context_payload.get("source_activity_epoch"),
        "current_changed_ranges": context_payload.get("current_changed_ranges"),
        "reprocess_all_current_observations": context_payload.get(
            "reprocess_all_current_observations",
            False,
        ),
    }


def _assemble_derivation_results(
    *,
    projection: SourceProjection,
    results: tuple[MemoryExtractionResult, ...],
) -> MemoryExtractionResult:
    metrics = _aggregate_extraction_metrics(results)
    revisions_by_observation_id = {revision.observation_id: revision for revision in projection.observation_revisions}
    protected_observation_ids = tuple(
        observation.id
        for observation in projection.observations
        if observation.observation_type == "binary_artifact"
        and observation.id in revisions_by_observation_id
        and not observation_is_inference_eligible(
            observation.observation_type,
            revisions_by_observation_id[observation.id].metadata,
        )
    )
    memories: list[RawMemory] = []
    failures: list[MemoryExtractionResult] = []
    summaries_by_observation_id: dict[str, SourceArtifactSummary] = {}
    invalid_summary_count = 0
    projected_image_ids = {
        revision.observation_id
        for revision in projection.observation_revisions
        if isinstance(revision.metadata.get("source_artifact"), Mapping)
        and str(revision.metadata["source_artifact"].get("media_type") or "").startswith("image/")
    }
    for result in results:
        if result.error_type:
            failures.append(result)
            continue
        memories.extend(result.memories)
        for summary in result.artifact_summaries:
            observation_id = summary.source_observation_id
            if observation_id not in projected_image_ids or observation_id in summaries_by_observation_id:
                invalid_summary_count += 1
                continue
            summaries_by_observation_id[observation_id] = summary
    if failures:
        first = failures[0]
        return MemoryExtractionResult(
            protected_source_observation_ids=protected_observation_ids,
            error_type="source_derivation_work_failure",
            error=first.error or first.error_type,
            metadata={
                **metrics,
                "batch_count": len(results),
                "failed_batch_count": len(failures),
                "extracted_count_before_failure": len(memories),
                "discarded_invalid_artifact_summary_count": (invalid_summary_count),
            },
        )
    return MemoryExtractionResult(
        memories=memories,
        artifact_summaries=tuple(summaries_by_observation_id.values()),
        protected_source_observation_ids=protected_observation_ids,
        metadata={
            **metrics,
            "batch_count": len(results),
            "failed_batch_count": 0,
            "discarded_invalid_artifact_summary_count": invalid_summary_count,
        },
    )


def _aggregate_extraction_metrics(
    results: tuple[MemoryExtractionResult, ...],
) -> dict[str, object]:
    return aggregate_extraction_metrics(results)


def aggregate_extraction_metrics(
    results: tuple[MemoryExtractionResult, ...] | list[MemoryExtractionResult],
) -> dict[str, object]:
    """Aggregate bounded cost and evidence-quality signals across one fan-out."""

    keys = (
        "structured_llm_calls",
        "prompt_chars",
        "structured_llm_elapsed_ms",
        "extraction_queue_wait_ms",
        "input_binary_bytes",
        "multimodal_calls",
        "artifact_summary_count",
        "discarded_orphan_artifact_summary_count",
        "discarded_invalid_artifact_summary_count",
    )
    aggregated = {key: sum(int((result.metadata or {}).get(key, 0) or 0) for result in results) for key in keys}
    aggregated["max_active_multimodal"] = max(
        (
            int(
                (result.metadata or {}).get(
                    "max_active_multimodal",
                    0,
                )
                or 0
            )
            for result in results
        ),
        default=0,
    )
    refinement_counts: dict[str, int] = {}
    fallback_samples: list[dict[str, object]] = []
    fallback_sample_truncated_count = 0
    for result in results:
        telemetry = _safe_evidence_telemetry(result.metadata)
        for refinement, count in telemetry.get(
            "evidence_refinement_counts",
            {},
        ).items():
            refinement_counts[refinement] = refinement_counts.get(refinement, 0) + count
        for sample in telemetry.get("evidence_block_fallback_samples", []):
            if len(fallback_samples) < _EVIDENCE_BLOCK_FALLBACK_SAMPLE_LIMIT:
                fallback_samples.append(sample)
            else:
                fallback_sample_truncated_count += 1
        fallback_sample_truncated_count += int(
            telemetry.get(
                "evidence_block_fallback_sample_truncated_count",
                0,
            )
        )
    aggregated["evidence_refinement_counts"] = refinement_counts
    aggregated["evidence_block_fallback_samples"] = fallback_samples
    aggregated["evidence_block_fallback_sample_truncated_count"] = (
        fallback_sample_truncated_count
    )
    aggregated["invalid_evidence_block_count"] = sum(
        int((result.metadata or {}).get("invalid_evidence_block_count", 0) or 0)
        for result in results
    )
    return aggregated


def _safe_evidence_telemetry(value: object) -> dict[str, object]:
    """Keep only bounded, content-free evidence diagnostics across replay."""

    if not isinstance(value, Mapping):
        return {}
    raw_counts = value.get("evidence_refinement_counts")
    counts = (
        {
            str(name): max(0, int(count))
            for name, count in raw_counts.items()
            if isinstance(name, str) and isinstance(count, int)
        }
        if isinstance(raw_counts, Mapping)
        else {}
    )
    raw_samples = value.get("evidence_block_fallback_samples")
    samples: list[dict[str, object]] = []
    if isinstance(raw_samples, list):
        for sample in raw_samples[:_EVIDENCE_BLOCK_FALLBACK_SAMPLE_LIMIT]:
            if not isinstance(sample, Mapping):
                continue
            samples.append(
                {
                    key: sample.get(key)
                    for key in (
                        "candidate_content_sha256",
                        "source_derivation_batch_id",
                        "source_observation_id",
                        "source_observation_revision_id",
                        "evidence_range_start",
                        "evidence_range_end",
                        "block_text_sha256",
                        "block_chars",
                        "submitted_quote_sha256",
                        "submitted_quote_chars",
                        "extraction_model",
                        "prompt_sha256",
                    )
                }
            )
    return {
        "evidence_refinement_counts": counts,
        "evidence_block_fallback_samples": samples,
        "evidence_block_fallback_sample_truncated_count": max(
            0,
            int(
                value.get(
                    "evidence_block_fallback_sample_truncated_count",
                    0,
                )
                or 0
            ),
        ),
        "invalid_evidence_block_count": max(
            0,
            int(value.get("invalid_evidence_block_count", 0) or 0),
        ),
    }
