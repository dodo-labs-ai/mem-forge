"""Input of one Source Unit for reprocessing it with the current adapter and compiler.

A reprocess reads the provider's current state when the Source can ask its
provider for one Document by id: the Gene rediscovers the Document and the
ordinary fetch reads it. A Document the provider no longer returns is reported
for that Document and nothing is inferred removed; removal belongs to a sync
whose discovery proves it.

A Source whose content has no provider to ask (a local agent collected it, or a
user uploaded it) reprocesses its stored input: the Document row with the item
metadata its Gene discovered, the raw content the last sync stored, and the
Artifacts the committed Unit revision cites. The Artifacts come back with the
Unit so a complete snapshot does not mistake them for removals. The stored raw
content is never older than the committed revision, because a sync that commits
a new revision stores the raw content it projected. When a sync stored newer
raw content whose revision never committed, reprocessing projects and commits
that content, as the next ordinary sync would. A Document stored before its
item metadata was kept may not place the Unit where its committed revision
does; such a stored input cannot stand for that revision until an ordinary sync
stores the Document again.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from memforge.models import ContentItem, DocumentRecord, RawContent
from memforge.source_artifacts import (
    SOURCE_ARTIFACT_OBSERVATION_TYPE,
    StoredSourceArtifact,
    source_artifact_revision_from_metadata,
    stored_source_artifact_from_observation,
)
from memforge.source_projection import SourceProjection

if TYPE_CHECKING:
    from memforge.genes.base import Gene
    from memforge.storage.adapters.protocols import RelationalStore
    from memforge.storage.document_store import DocumentStore

# Model requests a reprocessed Unit needs besides extraction and Support
# reading, when nothing has to be split: one Candidate admission request, and
# one Relation request when the Unit has active Supports.
_ADMISSION_REQUESTS_PER_UNIT = 1
_RELATION_REQUESTS_PER_SUPPORTED_UNIT = 1


class StoredDocumentUnavailableReason(str, Enum):
    DOCUMENT_MISSING = "stored_document_missing"
    SOURCE_UNIT_MISSING = "stored_source_unit_missing"
    RAW_CONTENT_MISSING = "stored_raw_content_missing"
    CONTENT_EMPTY = "stored_content_empty"
    ARTIFACT_MISSING = "stored_artifact_missing"
    # The committed Artifact metadata cannot be read as an Artifact revision.
    ARTIFACT_INVALID = "stored_artifact_invalid"
    # The stored input does not place the Unit where its committed revision does.
    INPUT_INCOMPLETE = "stored_input_incomplete"
    # The committed revision's current Observations cannot be authorized for extraction.
    EXTRACTION_UNPLANNABLE = "stored_extraction_unplannable"
    # The provider no longer returns the Document under its id.
    PROVIDER_DOCUMENT_MISSING = "provider_document_missing"


class StoredDocumentUnavailable(RuntimeError):
    """One Unit cannot be reprocessed; other Units continue."""

    # Retrying reads the same stored Document again.
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


async def _committed_source_document(
    db: RelationalStore,
    *,
    source_id: str,
    document_id: str,
) -> tuple[DocumentRecord, SourceProjection]:
    """Read one Document of the Source and its Unit at the current revision."""

    document = await db.get_document(document_id)
    if document is None or document.source != source_id:
        raise StoredDocumentUnavailable(StoredDocumentUnavailableReason.DOCUMENT_MISSING, document_id)
    unit = await db.find_source_unit_by_document_id(source_id, document_id, current_only=True)
    committed = await db.get_current_source_unit_projection(unit.id) if unit is not None else None
    if committed is None:
        raise StoredDocumentUnavailable(StoredDocumentUnavailableReason.SOURCE_UNIT_MISSING, document_id)
    return document, committed


def _stored_content_item(document: DocumentRecord) -> ContentItem:
    return ContentItem(
        item_id=document.doc_id,
        title=document.title,
        source_url=document.source_url,
        last_modified=document.last_modified,
        content_type=document.raw_content_type or "application/octet-stream",
        space_or_project=document.space_or_project,
        version=document.version,
        author=document.author,
        labels=list(document.labels),
        extra=dict(document.item_extra),
    )


async def rediscover_source_document(
    db: RelationalStore,
    gene: Gene,
    *,
    source_id: str,
    document_id: str,
) -> ContentItem:
    """Ask the provider for the current item of one Document whose Unit is current."""

    document, _ = await _committed_source_document(db, source_id=source_id, document_id=document_id)
    current = await gene.rediscover(_stored_content_item(document))
    if current is None or current.item_id != document_id:
        raise StoredDocumentUnavailable(StoredDocumentUnavailableReason.PROVIDER_DOCUMENT_MISSING, document_id)
    return current


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

    document, committed = await _committed_source_document(db, source_id=source_id, document_id=document_id)
    item = _stored_content_item(document)
    if not _stored(document_store, document.raw_content_uri, item.content_type):
        raise unavailable(StoredDocumentUnavailableReason.RAW_CONTENT_MISSING)
    body = document_store.read_artifact(str(document.raw_content_uri))
    if not body.strip():
        raise unavailable(StoredDocumentUnavailableReason.CONTENT_EMPTY)
    artifacts = _committed_artifacts(committed, unavailable)
    if not all(_stored(document_store, artifact.uri, artifact.media_type) for artifact in artifacts):
        raise unavailable(StoredDocumentUnavailableReason.ARTIFACT_MISSING)
    return StoredSourceDocument(
        document=document,
        item=item,
        raw=RawContent(item=item, body=body, content_type=item.content_type),
        artifacts=artifacts,
        committed=committed,
    )


def _stored(document_store: DocumentStore, uri: str | None, media_type: str) -> bool:
    try:
        return bool(uri) and document_store.get_artifact(uri, media_type) is not None
    except FileNotFoundError:
        return False


def _committed_artifacts(committed: SourceProjection, unavailable) -> tuple[StoredSourceArtifact, ...]:
    """The projection input of every Artifact the committed revision cites."""

    unit_revision = committed.source_unit_revisions[0]
    observations = {observation.id: observation for observation in committed.observations}
    artifacts = []
    for revision in committed.observation_revisions:
        observation = observations[revision.observation_id]
        if observation.observation_type != SOURCE_ARTIFACT_OBSERVATION_TYPE:
            continue
        artifact = source_artifact_revision_from_metadata(
            observation_id=observation.id,
            observation_revision_id=revision.id,
            source_id=committed.source_id,
            source_unit_id=unit_revision.source_unit_id,
            metadata=revision.metadata,
        )
        if artifact is None:
            raise unavailable(StoredDocumentUnavailableReason.ARTIFACT_INVALID)
        parent = observations.get(artifact.parent_observation_id)
        if parent is None:
            raise unavailable(StoredDocumentUnavailableReason.ARTIFACT_MISSING)
        artifacts.append(
            stored_source_artifact_from_observation(
                revision=artifact,
                observation_provider_key=observation.provider_key,
                locator=observation.locator,
                parent_observation_type=parent.observation_type,
                parent_provider_key=parent.provider_key,
            )
        )
    return tuple(artifacts)


@dataclass(frozen=True, slots=True)
class ReprocessUnitPreview:
    """What reprocessing one Document would read; an unavailable Unit has only its reason."""

    document_id: str
    available: bool
    reason: str | None = None
    source_unit_id: str | None = None
    unit_revision_id: str | None = None
    artifact_count: int | None = None
    active_memory_count: int | None = None
    support_reading_count: int | None = None
    reading_group_count: int | None = None
    extraction_item_count: int | None = None
    reading_chars: int | None = None
    estimated_model_calls: int | None = None


@dataclass(frozen=True, slots=True)
class ReprocessPreview:
    source_id: str
    units: tuple[ReprocessUnitPreview, ...]
    available_unit_count: int
    unavailable_unit_count: int
    estimated_model_calls: int


async def reprocess_preview(
    db: RelationalStore,
    document_store: DocumentStore,
    *,
    source_id: str,
    document_ids: tuple[str, ...],
    rediscovers: bool,
) -> ReprocessPreview:
    """Report what reprocessing these Documents would read, without writing anything.

    A Source that ``rediscovers`` its Documents reads them from the provider,
    so a Document needs no stored input; the preview does not contact the
    provider. Counts come from each Unit's committed revision. ``estimated_model_calls``
    is an estimate for approval, not a bound: one extraction item per
    ReadingGroup that holds authorized Primary content (the runner packs items
    into fewer requests), one Candidate admission request, one Relation request
    when the Unit has Supports, and one whole-Unit reading per Support
    (Supports can share a request). It leaves out requests split for capacity,
    Support readings that take several steps, selector corrections, entity
    resolution and cross-document relation classification.
    """

    from memforge.pipeline.memory_extractor import ExtractionReading
    from memforge.pipeline.projection_context import (
        ProjectionEvidencePlanningFailure,
        whole_revision_extraction_authority,
    )
    from memforge.pipeline.revision_assessment import RevisionAssessmentContext

    units: list[ReprocessUnitPreview] = []
    for document_id in sorted(set(document_ids)):
        try:
            if rediscovers:
                _, committed = await _committed_source_document(db, source_id=source_id, document_id=document_id)
                artifact_count = sum(
                    observation.observation_type == SOURCE_ARTIFACT_OBSERVATION_TYPE
                    for observation in committed.observations
                )
            else:
                stored = await load_stored_source_document(
                    db, document_store, source_id=source_id, document_id=document_id,
                )
                committed = stored.committed
                artifact_count = len(stored.artifacts)
            authority = whole_revision_extraction_authority(committed)
            if isinstance(authority, ProjectionEvidencePlanningFailure):
                raise StoredDocumentUnavailable(StoredDocumentUnavailableReason.EXTRACTION_UNPLANNABLE, document_id)
        except StoredDocumentUnavailable as exc:
            units.append(ReprocessUnitPreview(document_id=document_id, available=False, reason=exc.reason.value))
            continue
        unit_revision = committed.source_unit_revisions[0]
        context = RevisionAssessmentContext(projection=committed, base=None, access_context_hash=source_id)
        extraction = ExtractionReading.of_authority(
            context, authority, source_type=committed.source_type, doc_type=committed.source_type,
        )
        support_units = await db.get_source_unit_support_unit_ids(unit_revision.source_unit_id)
        active_ids = {memory.id for memory in await db.list_active_memories(tuple(sorted(support_units)))}
        support_readings = sum(len(support_units[memory_id]) for memory_id in active_ids)
        relation_requests = _RELATION_REQUESTS_PER_SUPPORTED_UNIT if support_readings else 0
        units.append(
            ReprocessUnitPreview(
                document_id=document_id,
                available=True,
                source_unit_id=unit_revision.source_unit_id,
                unit_revision_id=unit_revision.id,
                artifact_count=artifact_count,
                active_memory_count=len(active_ids),
                support_reading_count=support_readings,
                reading_group_count=len(context.reading_groups(context.full_fragments)),
                extraction_item_count=len(extraction.items),
                reading_chars=sum(len(fragment.presentation_text) for fragment in context.full_fragments),
                estimated_model_calls=(
                    len(extraction.items) + _ADMISSION_REQUESTS_PER_UNIT + relation_requests + support_readings
                ),
            )
        )
    available = [unit for unit in units if unit.available]
    return ReprocessPreview(
        source_id=source_id,
        units=tuple(units),
        available_unit_count=len(available),
        unavailable_unit_count=len(units) - len(available),
        estimated_model_calls=sum(unit.estimated_model_calls or 0 for unit in available),
    )
