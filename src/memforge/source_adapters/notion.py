"""How a Notion page projects into a Source Unit and its Observations.

Every page, database row or not, is one ``notion_page`` Unit with a
``page_body`` Observation: the page's body as CommonMark. A row also has a
``properties`` Observation, one canonical record of its property values as
text, so a row's properties are read once and never repeated in its body::

    {"properties": [{"name": "Status", "type": "status", "value": "Done"},
                    {"name": "Owner", "type": "people", "value": "Ada Lovelace"}],
     "representation": "notion-row-properties:1"}

Both are read from the versioned envelope :meth:`NotionGene.fetch
<memforge.genes.notion_gene.NotionGene.fetch>` stores, so the bytes of one
fetch alone decide the projection, its coverage included.
"""

from __future__ import annotations

from memforge.source_adapters.contracts import (
    MARKDOWN_STRUCTURAL_PROFILE,
    CanonicalRecordField,
    CanonicalRecordSchema,
    canonical_record_profile,
)

PROPERTIES_SCHEMA = CanonicalRecordSchema(
    name="notion-row-properties",
    version=1,
    fields=(
        CanonicalRecordField("/properties/*/name", contextual=True),
        CanonicalRecordField("/properties/*/value", nested_profile="plain-text"),
    ),
)
CANONICAL_RECORD_SCHEMAS = {(PROPERTIES_SCHEMA.name, PROPERTIES_SCHEMA.version): PROPERTIES_SCHEMA}
CURRENT_OBSERVATION_PROFILES = {
    "page_body": MARKDOWN_STRUCTURAL_PROFILE,
    "properties": canonical_record_profile(PROPERTIES_SCHEMA.name, PROPERTIES_SCHEMA.version),
}


def properties_record(properties: list[dict[str, str]]) -> dict[str, object]:
    """A row's rendered properties as one ``notion-row-properties`` record."""
    return {
        "properties": [{"name": entry["name"], "type": entry["type"], "value": entry["value"]} for entry in properties],
        "representation": PROPERTIES_SCHEMA.qualified_name,
    }


def project_native(*, source_id, item, native, normalized):
    """Build a page's identity, Observations, parent relation and coverage from its envelope.

    The page is ``contained_by`` its nearest page (``notion_page``), or, for
    a row, by its database (``notion_database``), as the walk recorded it.
    The page is a partial projection when the envelope says content is still
    missing (``truncated``).
    """
    from memforge.genes.notion_gene import notion_envelope
    from memforge.genes.notion_markdown import notion_markdown
    from memforge.source_adapters.contracts import (
        _canonical_json,
        _NativeProjection,
        _ObservationInput,
        _unit_title,
        _UnitEndpoint,
    )
    from memforge.source_projection import ProjectionCoverage, SourceRelationType
    from memforge.source_time import SOURCE_UPDATED_AT_KEY

    envelope = notion_envelope(native, item.item_id)
    body_time = normalized.source_semantics.get(SOURCE_UPDATED_AT_KEY)
    page_id = str(item.extra.get("page_id") or item.item_id.removeprefix("notion-"))
    body = notion_markdown(envelope["markdown"])
    observations = [_ObservationInput("page_body", f"{page_id}:body", body, body, {"page_id": page_id}, body_time)]
    if "properties" in envelope:
        record = properties_record(envelope["properties"])
        observations.append(
            _ObservationInput(
                "properties", f"{page_id}:properties", _canonical_json(record), record, {"page_id": page_id}, body_time
            )
        )

    parent_id = str(item.extra.get("parent_page_id") or "")
    parent_kind = item.extra.get("parent_kind")
    relations = ()
    if parent_id and parent_kind in {"page", "database"}:
        relations = (
            (
                SourceRelationType.CONTAINED_BY,
                "$unit",
                _UnitEndpoint(f"notion_{parent_kind}", parent_id),
                f"{page_id}:parent",
                {},
            ),
        )
    return _NativeProjection(
        unit_type="notion_page",
        provider_key=page_id,
        observations=tuple(observations),
        relations=relations,
        coverage=(
            ProjectionCoverage.PARTIAL_PROJECTION if envelope["truncated"] else ProjectionCoverage.COMPLETE_SNAPSHOT
        ),
        locator={"page_id": page_id, "url": item.source_url},
        title=_unit_title("Notion page", ("Title", item.title)),
    )
