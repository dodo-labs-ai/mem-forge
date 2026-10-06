"""Confluence-owned immutable page representation declarations."""

from __future__ import annotations

from memforge.source_adapters.contracts import CanonicalRecordField, CanonicalRecordSchema
from memforge.source_adapters.confluence_storage import STORAGE_FORMAT
from memforge.source_projection import EvidenceCoordinateSpace, EvidenceRepresentationProfile


PAGE_SCHEMA = CanonicalRecordSchema(
    name="confluence-page-storage", version=1,
    fields=(
        CanonicalRecordField("/title", contextual=True),
        CanonicalRecordField("/body", text_format=STORAGE_FORMAT),
    ),
)
CANONICAL_RECORD_SCHEMAS = {(PAGE_SCHEMA.name, PAGE_SCHEMA.version): PAGE_SCHEMA}
# The pre-profile writer persisted normalized Markdown. NULL-profile backfill
# may attest that historical contract, never the newly introduced native schema.
LEGACY_OBSERVATION_PROFILES = {
    ("confluence", "page_body"): EvidenceRepresentationProfile(
        "markdown-structural", 1, EvidenceCoordinateSpace.UNICODE_SCALAR,
    ),
}


def page_record(title: str, storage: object) -> dict[str, str]:
    """The actual fetched storage value; never reconstruct from normalized text."""
    if not isinstance(storage, str):
        raise ValueError("Confluence page projection requires its fetched storage value")
    return {
        "title": title, "body": storage,
        "representation": f"{PAGE_SCHEMA.name}:{PAGE_SCHEMA.version}",
    }

def project_native(*, source_id, item, native, normalized):
    """Build provider identity, immutable Observations, relations and coverage."""
    from memforge.source_projection import ProjectionCoverage, SourceRelationType
    from memforge.source_time import SOURCE_UPDATED_AT_KEY
    from memforge.source_adapters.contracts import _ObservationInput, _NativeProjection, _unit_title, _canonical_json

    body_time = normalized.source_semantics.get(SOURCE_UPDATED_AT_KEY)
    page_id = str(item.extra.get("page_id") or item.item_id.removeprefix("confluence-"))
    parent_id = str(item.extra.get("parent_page_id") or "")
    relations = ()
    if parent_id:
        relations = (
            (
                SourceRelationType.CONTAINED_BY,
                "$unit",
                f"confluence_page:{parent_id}",
                f"{page_id}:parent",
                {},
            ),
        )

    semantic_value = page_record(item.title, native)
    # The declared schema is part of immutable content identity. This creates
    # a new revision when replacing the former lossy Markdown representation,
    # even if the fetched native body has the same semantic hash as before.
    semantic_content = _canonical_json(semantic_value)
    return _NativeProjection(
        unit_type="confluence_page",
        provider_key=page_id,
        observations=(
            _ObservationInput(
                "page_body",
                f"{page_id}:body",
                semantic_content,
                semantic_value,
                {},
                body_time,
            ),
        ),
        relations=relations,
        coverage=ProjectionCoverage.COMPLETE_SNAPSHOT,
        locator={
            "page_id": page_id,
            "space_key": item.extra.get("space_key") or item.space_or_project,
            "parent_page_id": parent_id or None,
            "url": item.source_url,
        },
        title=_unit_title(
            "Confluence page",
            ("Space", item.extra.get("space_key") or item.space_or_project),
            ("Title", item.title),
        ),
    )
