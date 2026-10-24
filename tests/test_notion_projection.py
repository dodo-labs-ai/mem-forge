"""How a Notion page projects: a body Observation, a row's properties, its parent and its coverage."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from memforge.models import ContentItem, RawContent
from memforge.pipeline.source_projection_adapters import _stable_id, project_source_item
from memforge.source_projection import ProjectionCoverage, SourceRelationType
from memforge.source_types import source_type_descriptor
from memforge.genes.notion_gene import NotionGene

SOURCE_ID = "src-notion"
PAGE = "1f3c2a4b5d6e4f708a9b0c1d2e3f4a5b"
PARENT = "2a3b4c5d6e7f40818293a4b5c6d7e8f9"
DATABASE = "9c0d1e2f3a4b4c5d8e6f7a8b9c0d1e2f"
EDITED = datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc)


def _envelope(markdown: str, *, properties: list[dict] | None = None, truncated: bool = False) -> dict:
    envelope: dict = {"notion_raw": 1, "markdown": markdown}
    if properties is not None:
        envelope["properties"] = properties
    return {**envelope, "truncated": truncated, "unknown_not_found": 0}


async def _project(envelope: dict, *, parent: tuple[str, str] = (PARENT, "page")):
    item = ContentItem(
        item_id=f"notion-{PAGE}",
        title="Payroll runbook",
        source_url=f"https://app.notion.com/p/{PAGE}",
        last_modified=EDITED,
        version=EDITED.isoformat(),
        extra={"page_id": PAGE, "parent_page_id": parent[0], "parent_kind": parent[1]},
    )
    raw = RawContent(item=item, body=json.dumps(envelope).encode(), content_type="application/json")
    normalized = await NotionGene({"token": "x", "root_pages": [PAGE]}, SOURCE_ID).normalize(raw)
    return project_source_item(
        source_id=SOURCE_ID, source_type="notion", run_id="run-1", item=item, raw=raw, normalized=normalized
    )


def _observations(projection) -> dict[str, tuple[str, str]]:
    """Each Observation's provider key and content, by type."""
    contents = {revision.observation_id: revision.content for revision in projection.observation_revisions}
    return {
        observation.observation_type: (observation.provider_key, contents[observation.id])
        for observation in projection.observations
    }


@pytest.mark.asyncio
async def test_a_page_projects_a_page_body_observation() -> None:
    projection = await _project(_envelope("<callout>Run payroll on the **last Friday**.</callout>"))

    (unit,) = projection.source_units
    assert (unit.unit_type, unit.provider_key) == ("notion_page", PAGE)
    observations = _observations(projection)
    assert set(observations) == {"page_body"}
    key, body = observations["page_body"]
    assert key == f"{PAGE}:body"
    assert "Run payroll on the **last Friday**." in body
    assert "<callout>" not in body
    assert projection.unit_title.kind == "Notion page"
    assert projection.unit_title.fields == (("Title", "Payroll runbook"),)


@pytest.mark.asyncio
async def test_a_row_also_projects_properties() -> None:
    properties = [
        {"name": "Status", "type": "status", "value": "Done"},
        {"name": "Owner", "type": "people", "value": "Ada Lovelace"},
    ]
    projection = await _project(_envelope("", properties=properties), parent=(DATABASE, "database"))

    observations = _observations(projection)
    assert set(observations) == {"page_body", "properties"}
    assert observations["page_body"] == (f"{PAGE}:body", "")
    key, content = observations["properties"]
    assert key == f"{PAGE}:properties"
    assert json.loads(content) == {"properties": properties, "representation": "notion-row-properties:1"}
    profile = source_type_descriptor("notion").evidence_profiles["properties"]
    assert (profile.name, profile.schema_name, profile.schema_version) == (
        "canonical-record", "notion-row-properties", 1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("parent", "unit_type"), [((PARENT, "page"), "notion_page"), ((DATABASE, "database"), "notion_database")])
async def test_contained_by_points_at_the_parent_page_or_database(parent, unit_type) -> None:
    projection = await _project(_envelope("Body"), parent=parent)

    (unit,) = projection.source_units
    (relation,) = projection.relations
    assert relation.relation_type is SourceRelationType.CONTAINED_BY
    assert relation.from_id == unit.id
    # The parent need not be a Unit of this Source: a root's parent lies outside the roots.
    assert relation.to_id == _stable_id("unit", SOURCE_ID, unit_type, parent[0])


@pytest.mark.asyncio
async def test_a_root_under_the_workspace_has_no_parent_relation() -> None:
    projection = await _project(_envelope("Body"), parent=("", ""))

    assert projection.relations == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("truncated", "coverage"),
    [(True, ProjectionCoverage.PARTIAL_PROJECTION), (False, ProjectionCoverage.COMPLETE_SNAPSHOT)],
)
async def test_a_truncated_page_projects_partial_coverage(truncated, coverage) -> None:
    projection = await _project(_envelope("Body", truncated=truncated))

    assert projection.coverage is coverage


@pytest.mark.asyncio
async def test_the_locator_holds_no_signed_url() -> None:
    envelope = _envelope("![Diagram](notion-file:/space/file/diagram.png)")

    projection = await _project(envelope)

    (unit,) = projection.source_units
    assert unit.locator["page_id"] == PAGE
    assert unit.locator["url"] == f"https://app.notion.com/p/{PAGE}"
    stored = json.dumps([unit.locator, *(observation.locator for observation in projection.observations)])
    assert "X-Amz-" not in stored and "amazonaws" not in stored


@pytest.mark.asyncio
async def test_an_envelope_of_another_version_fails_the_projection() -> None:
    with pytest.raises(ValueError, match="version 1 envelope"):
        await _project({**_envelope("Body"), "notion_raw": 2})
