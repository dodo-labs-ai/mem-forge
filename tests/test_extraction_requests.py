"""Claim Extraction reads whole ReadingGroups that hold authorized Primary, through the runner."""

import json
import re
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.llm.structured import (
    INPUT_CAPACITY_EXCEEDED,
    LiteLlmStructuredClient,
    ProjectionFragmentMemoryExtractionResponse,
    StructuredLlmConfig,
    StructuredLlmError,
)
from memforge.pipeline.extraction_requests import ExtractionCapacityExceeded, plan_extraction_requests
from memforge.pipeline.memory_extractor import MemoryExtractor
from memforge.pipeline.projection_context import ExtractionAuthority, plan_projection_evidence_work
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.source_projection_adapters import project_source_item
from memforge.source_representation import UNIT_TITLE_OBSERVATION_TYPE
from tests.test_projection_context import _committed_snapshot, _confluence_projection, _jira_projection, _requests
from tests.test_projection_fragments import _projection
from tests.test_revision_work import Client

# The design's acceptance length for a reading that is never truncated or grouped by character count.
LONG_READING_CHARS = 20_000


def bounded_output(extractor, catalog):
    """The output a planned request is sent with: requested, then bounded by the route."""
    budget = extractor.structured_llm_client.request_budget(extractor.model)
    return budget.output_reserve(extractor.fragment_output_tokens(catalog))


def plan(projection, authority, *, extractor, base=None):
    context = RevisionAssessmentContext(projection=projection, base=base, access_context_hash="scope")
    return plan_extraction_requests(
        context, authority, extractor=extractor, source_type="confluence", doc_type="document",
    )


def whole(projection):
    authority = plan_projection_evidence_work(projection, reprocess_all_current_observations=False)
    assert isinstance(authority, ExtractionAuthority)
    return authority


def body_id(projection):
    return next(item.id for item in projection.observations if item.observation_type != UNIT_TITLE_OBSERVATION_TYPE)


def primary(requests):
    return [fragment for request in requests for fragment in request.catalog.fragments if fragment.primary_eligible]


def texts(request):
    return [fragment.presentation_text.strip() for fragment in request.catalog.fragments]


@pytest.mark.parametrize(("representation", "rows"), [("markdown", 1_200), ("html", 1_200), ("html", 3_000)])
def test_large_complete_table_reaches_actual_request_budget(representation, rows):
    if representation == "markdown":
        table = "| Rule | Sandbox | Small Box |\n| --- | --- | --- |\n" + "\n".join(
            f"| Approval {row} | Yes | No |" for row in range(rows))
    else:
        table = "<table><tr><th>Rule</th><th>Sandbox</th><th>Small Box</th></tr>" + "".join(
            f"<tr><td>Approval {row}</td><td>Yes</td><td>No</td></tr>" for row in range(rows)) + "</table>"
    projection = _confluence_projection("Before the table.\n\n" + table + "\n\nAfter the table.")

    for window in (100_000, 8_000):
        client = LiteLlmStructuredClient(StructuredLlmConfig(
            model="bedrock/anthropic.claude-sonnet-4-6", base_url=None, api_key=None, timeout_s=1,
            max_input_tokens=window, context_window_tokens=window, max_output_tokens=1024))
        extractor = MemoryExtractor(model=client.config.model, structured_llm_client=client)
        if window == 8_000:
            # One ReadingGroup that alone exceeds the route is a typed limitation, never truncated.
            with pytest.raises(ExtractionCapacityExceeded) as error:
                plan(projection, whole(projection), extractor=extractor)
            assert error.value.reason_code == "input_capacity_exceeded" and not error.value.retryable
            continue
        requests = plan(projection, whole(projection), extractor=extractor)
        tables = [f for f in primary(requests) if "table" in f.fragment_type]
        assert [f.presentation_text for f in tables] == [table]
        context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")
        for request in requests:
            prompt = MemoryExtractor.projection_fragment_prompt(
                request.catalog, source_type="confluence", doc_type="document", revision_context=context,
            )
            assert client.request_fits(prompt, response_format=ProjectionFragmentMemoryExtractionResponse,
                                       max_tokens=bounded_output(extractor, request.catalog))


def test_first_import_streams_every_reading_group_through_complete_requests():
    body = "# US payroll\n\n" + "\n\n".join(f"Rule {i}: " + "This is exact source content. " * 80 for i in range(75))
    assert len(body) > LONG_READING_CHARS
    projection = _projection(primary_content=body, context_content="Country: US.\n")
    client = Client(limit=12000)
    extractor = MemoryExtractor(model="fixture", max_tokens=8192, structured_llm_client=client)
    requests = plan(projection, whole(projection), extractor=extractor)

    assert len(requests) > 2
    anchors = [fragment.anchor for fragment in primary(requests)]
    assert len(anchors) == len(set(anchors))
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")
    assert set(anchors) == {fragment.anchor for fragment in context.full_fragments}
    for request in requests:
        # The heading scopes every paragraph under it; outside its own request it is Required-only.
        heading = next(f for f in request.catalog.fragments if f.presentation_text.strip() == "# US payroll")
        if heading.primary_eligible:
            assert request is requests[0]
    assert [request.id for request in plan(projection, whole(projection), extractor=extractor)] == [
        request.id for request in requests
    ]


def test_update_reads_only_changed_structures_with_their_reading_group_context():
    sections = (
        "# Approvals\n\nIntro for approvals.\n\nOld unrelated details.\n\nTwo reviewers approve releases.\n\n"
        "# Rollout\n\nRollout starts {day}.\n"
    )
    initial = _confluence_projection(sections.format(day="Monday"))
    target = _updated_confluence(initial, sections.format(day="Tuesday"))
    [request] = _requests(target, base=initial, committed=_committed_snapshot(initial))
    read = texts(request)

    assert [f.presentation_text.strip() for f in request.catalog.fragments if f.primary_eligible] == [
        "Rollout starts Tuesday."
    ]
    # Its heading is context; an unchanged section elsewhere is not read.
    assert "# Rollout" in read
    assert "Old unrelated details." not in read
    assert "Two reviewers approve releases." not in read
    assert not re.search("input_mode|removed_historical|additional_context", json.dumps(read))


def test_a_changed_list_item_reads_its_whole_list_but_authorizes_only_the_changed_item():
    initial = _confluence_projection("Approvals need:\n\n- one reviewer\n- a ticket\n- a rollback plan\n")
    target = _updated_confluence(initial, "Approvals need:\n\n- two reviewers\n- a ticket\n- a rollback plan\n")
    [request] = _requests(target, base=initial, committed=_committed_snapshot(initial))
    read = " ".join(texts(request))

    assert all(item in read for item in ("two reviewers", "a ticket", "a rollback plan", "Approvals need:"))
    assert [f.presentation_text for f in request.catalog.fragments if f.primary_eligible] == ["- two reviewers"]


def test_reading_context_and_the_unit_title_are_never_primary():
    projection = _jira_projection(2)
    extractor = MemoryExtractor(model="fixture", max_tokens=8192, structured_llm_client=Client(limit=40000))
    comment = next(item.id for item in projection.observations if item.observation_type == "comment")
    [request] = plan(projection, ExtractionAuthority({comment: None}), extractor=extractor)
    by_observation = {}
    for fragment in request.catalog.fragments:
        by_observation.setdefault(fragment.anchor.observation_id, []).append(fragment.primary_eligible)
    title = next(item.id for item in projection.observations if item.observation_type == UNIT_TITLE_OBSERVATION_TYPE)
    core = next(item.id for item in projection.observations if item.observation_type == "issue_core")

    # The comment reads with the Unit Title and the core it follows; only the comment is Primary.
    assert set(by_observation) == {comment, title, core}
    assert all(by_observation[comment]) and not any(by_observation[title]) and not any(by_observation[core])


def test_reading_context_longer_than_20000_characters_is_read_whole():
    long_description = "Payroll context sentence. " * 2_000
    assert len(long_description) > LONG_READING_CHARS
    projection = _jira_projection(1)
    core = next(item for item in projection.observation_revisions if '"description"' in item.content)
    body = json.loads(core.content)
    body["description"] = long_description
    content = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    projection = replace(projection, observation_revisions=tuple(
        replace(item, content=content) if item.id == core.id else item for item in projection.observation_revisions
    ))
    comment = next(item.id for item in projection.observations if item.observation_type == "comment")
    extractor = MemoryExtractor(model="fixture", max_tokens=8192, structured_llm_client=Client(limit=200_000))
    [request] = plan(projection, ExtractionAuthority({comment: None}), extractor=extractor)

    assert long_description.strip() in "\n".join(texts(request))


def test_small_real_window_packs_whole_tables_without_fixed_output_reservation(monkeypatch):
    monkeypatch.setattr("memforge.llm.structured.litellm.get_model_info",
                        lambda _: {"max_input_tokens": 1000000, "max_output_tokens": 64000})
    client = LiteLlmStructuredClient(StructuredLlmConfig(
        model="openai/fixture", base_url=None, api_key=None, timeout_s=1,
        max_input_tokens=8000, context_window_tokens=12000, max_output_tokens=1024,
    ))
    tables = ["| Field | Sandbox | Small Box |\n| --- | --- | --- |\n" + "\n".join(
        f"| Rule {table}-{row} | Yes | No |" for row in range(70)) for table in range(5)]
    projection = _projection(primary_content="\n\n".join(tables), context_content="Country: US.")
    extractor = MemoryExtractor(model=client.config.model, structured_llm_client=client)
    requests = plan(projection, ExtractionAuthority({body_id(projection): None}), extractor=extractor)

    assert len(requests) > 1
    seen = [f.presentation_text for f in primary(requests) if f.fragment_type == "markdown-table"]
    assert all("Sandbox" in text and "Small Box" in text for text in seen)
    assert sorted(t.strip() for t in seen) == sorted(tables)


@pytest.mark.asyncio
async def test_a_multi_item_request_that_times_out_is_halved_and_every_item_completes():
    projection = _projection(
        primary_content="Rule one applies.\n\nRule two applies.\n\nRule three applies.\n\nRule four applies.\n",
        context_content="Country: US.\n",
    )
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")

    class TimingOutClient(Client):
        def __init__(self):
            super().__init__(limit=40000)
            self.extraction_prompts = []

        async def extract_projection_fragment_memories(self, prompt, **kwargs):
            self.extraction_prompts.append(prompt)
            payload = json.loads(prompt.split("<evidence_fragment_catalog", 1)[1].split(">", 1)[1].split(
                "\n</evidence_fragment_catalog>", 1)[0])
            rows = payload["primary_candidates"]
            if len(rows) > 1:
                raise StructuredLlmError("timed out", terminal_category="deadline_exceeded", error_code="timeout")
            ref, text = rows[0][0], rows[0][1]
            return ProjectionFragmentMemoryExtractionResponse.model_validate({"memories": [
                {"content": text, "memory_type": "fact", "primary_ref": ref, "required_refs": []},
            ]})

    client = TimingOutClient()
    extractor = MemoryExtractor(model="fixture", max_tokens=8192, structured_llm_client=client)
    [request] = plan(projection, ExtractionAuthority({body_id(projection): None}), extractor=extractor)
    result = await extractor.extract_projection_fragment_memories(
        request.catalog, source_type="confluence", revision_context=context,
    )

    assert result.error_type is None
    assert sorted(memory.content for memory in result.memories) == [
        "Rule four applies.", "Rule one applies.", "Rule three applies.", "Rule two applies.",
    ]
    assert result.metadata["extraction_request_count"] == 4
    assert len(client.extraction_prompts) > 4


@pytest.mark.asyncio
async def test_a_single_reading_group_beyond_capacity_fails_the_extraction():
    projection = _projection(primary_content="Rule one applies.\n", context_content="Country: US.\n")
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")

    class OverflowingClient(Client):
        async def extract_projection_fragment_memories(self, prompt, **kwargs):
            raise StructuredLlmError("too large", terminal_category="provider_error", error_code=INPUT_CAPACITY_EXCEEDED)

    extractor = MemoryExtractor(model="fixture", max_tokens=8192, structured_llm_client=OverflowingClient(limit=40000))
    [request] = plan(projection, ExtractionAuthority({body_id(projection): None}), extractor=extractor)
    result = await extractor.extract_projection_fragment_memories(
        request.catalog, source_type="confluence", revision_context=context,
    )

    assert result.error_type == "input_capacity_exceeded"
    assert result.memories == []


def _updated_confluence(initial, body):
    """The next revision of ``initial``'s page with this body."""
    item = ContentItem(
        item_id="confluence-42",
        title="Large design",
        source_url="https://confluence.example.test/pages/42",
        last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
        version="8",
        extra={"page_id": "42", "space_key": "ENG"},
    )
    return project_source_item(
        source_id="src-c",
        source_type="confluence",
        run_id="run-c-update",
        item=item,
        raw=RawContent(item=item, body=body.encode(), content_type="text/html"),
        normalized=NormalizedContent(item=item, markdown_body=body),
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={revision.observation_id: revision for revision in initial.observation_revisions},
    )
