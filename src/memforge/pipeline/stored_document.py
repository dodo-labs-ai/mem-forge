"""Stored input of one Source Unit, for reprocessing it at its current revision.

Reprocessing never contacts the provider. It reads the Document row, the raw
content the last sync stored, and the Artifacts the committed Unit revision
cites, and hands them to the same projection path an ordinary sync uses. The
Artifacts come back with the Unit so a complete snapshot does not mistake
them for removals. The descriptor a Gene attaches to a discovered item is not
stored, so adapters read the Unit from the stored payload; a stored input that
no longer places the Unit where its committed revision does cannot stand for
that revision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from memforge.models import ContentItem, DocumentRecord, RawContent
from memforge.source_artifacts import StoredSourceArtifact
from memforge.source_projection import SourceProjection

if TYPE_CHECKING:
    from memforge.storage.adapters.protocols import RelationalStore
    from memforge.storage.document_store import DocumentStore

ARTIFACT_OBSERVATION_TYPE = "binary_artifact"
ARTIFACT_PROVIDER_KEY_PREFIX = "artifact:"


class StoredDocumentUnavailableReason(str, Enum):
    DOCUMENT_MISSING = "stored_document_missing"
    SOURCE_UNIT_MISSING = "stored_source_unit_missing"
    RAW_CONTENT_MISSING = "stored_raw_content_missing"
    CONTENT_EMPTY = "stored_content_empty"
    ARTIFACT_MISSING = "stored_artifact_missing"
    # The stored input does not place the Unit where its committed revision does.
    INPUT_INCOMPLETE = "stored_input_incomplete"


class StoredDocumentUnavailable(RuntimeError):
    """The stored input of one Unit cannot be reprocessed; other Units continue."""

    # Retrying reads the same stored input again.
    retryable = False

    def __init__(self, reason: StoredDocumentUnavailableReason, document_id: str) -> None:
        super().__init__(f"{reason.value}: {document_id}")
        self.reason = reason
        self.document_id = document_id


@dataclass(frozen=True, slots=True)
class StoredSourceDocument:
    document: DocumentRecord
    item: ContentItem
    raw: RawContent
    artifacts: tuple[StoredSourceArtifact, ...]
    # The Unit at its current revision.
    committed: SourceProjection


async def load_stored_source_document(
    db: RelationalStore,
    document_store: DocumentStore,
    *,
    source_id: str,
    document_id: str,
) -> StoredSourceDocument:
    """Read one Document's stored input and the Artifacts of its committed Unit revision."""

    def unavailable(reason: StoredDocumentUnavailableReason) -> StoredDocumentUnavailable:
        return StoredDocumentUnavailable(reason, document_id)

    document = await db.get_document(document_id)
    if document is None or document.source != source_id:
        raise unavailable(StoredDocumentUnavailableReason.DOCUMENT_MISSING)
    unit = await db.find_source_unit_by_document_id(source_id, document_id, current_only=True)
    committed = await db.get_current_source_unit_projection(unit.id) if unit is not None else None
    if committed is None:
        raise unavailable(StoredDocumentUnavailableReason.SOURCE_UNIT_MISSING)
    content_type = document.raw_content_type or "application/octet-stream"
    if not _stored(document_store, document.raw_content_uri, content_type):
        raise unavailable(StoredDocumentUnavailableReason.RAW_CONTENT_MISSING)
    body = document_store.read_artifact(str(document.raw_content_uri))
    if not body.strip():
        raise unavailable(StoredDocumentUnavailableReason.CONTENT_EMPTY)
    artifacts = _committed_artifacts(committed)
    if artifacts is None or not all(
        _stored(document_store, artifact.uri, artifact.media_type) for artifact in artifacts
    ):
        raise unavailable(StoredDocumentUnavailableReason.ARTIFACT_MISSING)
    item = ContentItem(
        item_id=document.doc_id,
        title=document.title,
        source_url=document.source_url,
        last_modified=document.last_modified,
        content_type=content_type,
        space_or_project=document.space_or_project,
        version=document.version,
        author=document.author,
        labels=list(document.labels),
    )
    return StoredSourceDocument(
        document=document,
        item=item,
        raw=RawContent(item=item, body=body, content_type=content_type),
        artifacts=artifacts,
        committed=committed,
    )


def _stored(document_store: DocumentStore, uri: str | None, media_type: str) -> bool:
    try:
        return bool(uri) and document_store.get_artifact(uri, media_type) is not None
    except FileNotFoundError:
        return False


def _committed_artifacts(committed: SourceProjection) -> tuple[StoredSourceArtifact, ...] | None:
    """The committed Artifacts, or None when one no longer has its parent Observation."""
    observations = {observation.id: observation for observation in committed.observations}
    artifacts = []
    for revision in committed.observation_revisions:
        observation = observations[revision.observation_id]
        if observation.observation_type != ARTIFACT_OBSERVATION_TYPE:
            continue
        metadata = revision.metadata["source_artifact"]
        parent = observations.get(str(metadata["parent_observation_id"]))
        if parent is None:
            return None
        artifacts.append(
            StoredSourceArtifact(
                id=str(metadata["artifact_id"]),
                provider_key=observation.provider_key.removeprefix(ARTIFACT_PROVIDER_KEY_PREFIX),
                parent_observation_type=parent.observation_type,
                parent_provider_key=parent.provider_key,
                provider_revision=str(metadata["provider_revision"]),
                filename=str(metadata["filename"]),
                media_type=str(metadata["media_type"]),
                size_bytes=int(metadata["size_bytes"]),
                sha256=str(metadata["sha256"]),
                uri=str(metadata["uri"]),
                inference_eligible=bool(metadata["inference_eligible"]),
                inference_ineligible_reason=metadata.get("inference_ineligible_reason"),
                locator=dict(observation.locator),
            )
        )
    return tuple(artifacts)


async def reprocess_preview(
    db: RelationalStore,
    document_store: DocumentStore,
    *,
    source_id: str,
    document_ids: tuple[str, ...],
) -> dict[str, object]:
    """Report what reprocessing these Documents would read, without writing anything.

    Counts come from each Unit's committed revision. ``estimated_model_calls``
    is an upper bound: one extraction item per ReadingGroup that holds Primary
    content (the runner packs items into fewer requests), one Candidate
    admission, one Relation request when the Unit has Supports, and one
    whole-Unit Support reading per Support (Supports can share a request).
    """

    from memforge.pipeline.revision_assessment import RevisionAssessmentContext

    units: list[dict[str, object]] = []
    for document_id in sorted(set(document_ids)):
        try:
            stored = await load_stored_source_document(
                db, document_store, source_id=source_id, document_id=document_id,
            )
        except StoredDocumentUnavailable as exc:
            units.append({"document_id": document_id, "available": False, "reason": exc.reason.value})
            continue
        unit_revision = stored.committed.source_unit_revisions[0]
        context = RevisionAssessmentContext(projection=stored.committed, base=None, access_context_hash="")
        groups = context.reading_groups(context.full_fragments)
        extraction_items = sum(any(fragment.primary_eligible for fragment in group) for group in groups)
        support_units = await db.get_source_unit_support_unit_ids(unit_revision.source_unit_id)
        active_ids = {memory.id for memory in await db.list_active_memories(tuple(sorted(support_units)))}
        support_readings = sum(len(support_units[memory_id]) for memory_id in active_ids)
        units.append(
            {
                "document_id": document_id,
                "available": True,
                "source_unit_id": unit_revision.source_unit_id,
                "unit_revision_id": unit_revision.id,
                "artifact_count": len(stored.artifacts),
                "active_memory_count": len(active_ids),
                "support_reading_count": support_readings,
                "reading_group_count": len(groups),
                "extraction_item_count": extraction_items,
                "reading_chars": sum(len(fragment.presentation_text) for fragment in context.full_fragments),
                "estimated_model_calls": extraction_items + 1 + int(support_readings > 0) + support_readings,
            }
        )
    available = [unit for unit in units if unit["available"]]
    return {
        "source_id": source_id,
        "units": units,
        "available_unit_count": len(available),
        "unavailable_unit_count": len(units) - len(available),
        "estimated_model_calls": sum(int(unit["estimated_model_calls"]) for unit in available),
    }
