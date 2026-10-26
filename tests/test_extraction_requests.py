"""Claim Extraction reads whole ReadingGroups that hold authorized Primary, through the runner."""

from tests.evidence_display_fixture import evidence_displays

import json
import re
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from memforge.models import ContentItem, NormalizedContent, RawContent
from memforge.llm.batch_runner import OUTPUT_CAPACITY_EXCEEDED
from memforge.llm.structured import (
    INPUT_CAPACITY_EXCEEDED,
    OUTPUT_TRUNCATED,
    LiteLlmStructuredClient,
    ProjectionFragmentMemoryExtractionResponse,
    StructuredLlmConfig,
    StructuredLlmError,
)
from memforge.pipeline.extraction_requests import plan_extraction_requests
from memforge.pipeline.memory_extractor import ExtractionReading, MemoryExtractor
from memforge.pipeline.projection_context import (
    ExtractionAuthority, plan_projection_evidence_work,
)
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.source_projection_adapters import project_source_item
from tests.test_projection_context import _committed_snapshot, _normalized_document_projection, _jira_projection, _requests
from tests.test_projection_fragments import _projection
from tests.test_revision_work import Client

# The design's acceptance length for a reading that is never truncated or grouped by character count.
LONG_READING_CHARS = 20_000


def extraction_plan(projection, authority, *, extractor, base=None):
    context = RevisionAssessmentContext(projection=projection, base=base, access_context_hash="scope")
    return plan_extraction_requests(
        context, authority, extractor=extractor, source_type=projection.source_type, doc_type="document",
    )


def plan(projection, authority, *, extractor, base=None):
    extraction = extraction_plan(projection, authority, extractor=extractor, base=base)
    assert extraction.skipped_reading_groups == ()
    return extraction.requests


def whole(projection):
    authority = plan_projection_evidence_work(projection, reprocess_all_current_observations=False)
    assert isinstance(authority, ExtractionAuthority)
    return authority


def body_id(projection):
    return projection.observations[0].id


def primary(requests):
    return [fragment for request in requests for fragment in request.catalog.fragments if fragment.primary_eligible]


def texts(request):
    return [fragment.presentation_text.strip() for fragment in request.catalog.fragments]


@pytest.mark.parametrize(("representation", "rows"), [("markdown", 1_200), ("html", 1_200), ("html", 3_000)])
def test_large_complete_table_reaches_actual_request_budget(representation, rows, caplog):
    if representation == "markdown":
        table = "| Rule | Sandbox | Small Box |\n| --- | --- | --- |\n" + "\n".join(
            f"| Approval {row} | Yes | No |" for row in range(rows))
    else:
        table = "<table><tr><th>Rule</th><th>Sandbox</th><th>Small Box</th></tr>" + "".join(
            f"<tr><td>Approval {row}</td><td>Yes</td><td>No</td></tr>" for row in range(rows)) + "</table>"
    projection = _normalized_document_projection("Before the table.\n\n" + table + "\n\nAfter the table.")

    for window in (100_000, 8_000):
        client = LiteLlmStructuredClient(StructuredLlmConfig(
            model="bedrock/anthropic.claude-sonnet-4-6", base_url=None, api_key=None, timeout_s=1,
            max_input_tokens=window, context_window_tokens=window, max_output_tokens=1024))
        extractor = MemoryExtractor(model=client.config.model, structured_llm_client=client)
        if window == 8_000:
            # A whole unfit group is diagnosed; independent groups remain planned.
            with caplog.at_level("WARNING", logger="memforge.pipeline.memory_extractor"):
                extraction = extraction_plan(projection, whole(projection), extractor=extractor)
            assert len(extraction.skipped_reading_groups) == 1
            assert [f.presentation_text.strip() for f in primary(extraction.requests)] == [
                "Before the table.", "After the table.",
            ]
            assert "extraction_reading_group_unread" in caplog.text
            assert f"reason={INPUT_CAPACITY_EXCEEDED}" in caplog.text
            continue
        requests = plan(projection, whole(projection), extractor=extractor)
        tables = [f for f in primary(requests) if "table" in f.fragment_type]
        assert [f.presentation_text for f in tables] == [table]
        context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")
        for request in requests:
            prompt = MemoryExtractor.projection_fragment_prompt(
                request.catalog, source_type=projection.source_type, doc_type="document", revision_context=context,
            )
            assert client.request_fits(prompt, response_format=ProjectionFragmentMemoryExtractionResponse)


def test_first_import_streams_every_reading_group_through_complete_requests():
    body = "# US payroll\n\n" + "\n\n".join(f"Rule {i}: " + "This is exact source content. " * 80 for i in range(75))
    assert len(body) > LONG_READING_CHARS
    projection = _projection(primary_content=body, context_content="Country: US.\n")
    client = Client(limit=12000)
    extractor = MemoryExtractor(model="fixture", structured_llm_client=client)
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")
    reading = ExtractionReading.of_authority(
        context, whole(projection), source_type=projection.source_type, doc_type="document",
    )
    # This fixture must fit every complete group, while requiring multiple
    # requests; its character quota includes the current instruction overhead.
    def single_group_cost(item_id):
        catalog = reading.catalog_for((item_id,))
        prompt = extractor.projection_fragment_prompt(
            catalog, source_type=projection.source_type,
            doc_type="document", revision_context=context,
        )
        return len(prompt)

    client.limit = max(single_group_cost(item_id) for item_id in reading.items) + 256
    requests = plan(projection, whole(projection), extractor=extractor)

    assert len(requests) > 2
    anchors = [fragment.anchor for fragment in primary(requests)]
    assert len(anchors) == len(set(anchors))
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
    initial = _normalized_document_projection(sections.format(day="Monday"))
    target = _updated_normalized_document(initial, sections.format(day="Tuesday"))
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
    initial = _normalized_document_projection("Approvals need:\n\n- one reviewer\n- a ticket\n- a rollback plan\n")
    target = _updated_normalized_document(initial, "Approvals need:\n\n- two reviewers\n- a ticket\n- a rollback plan\n")
    [request] = _requests(target, base=initial, committed=_committed_snapshot(initial))
    read = " ".join(texts(request))

    assert all(item in read for item in ("two reviewers", "a ticket", "a rollback plan", "Approvals need:"))
    assert [f.presentation_text for f in request.catalog.fragments if f.primary_eligible] == ["- two reviewers"]


def test_reading_context_is_never_primary():
    projection = _jira_projection(2)
    extractor = MemoryExtractor(model="fixture", structured_llm_client=Client(limit=40000))
    comment = next(item.id for item in projection.observations if item.observation_type == "comment")
    [request] = plan(projection, ExtractionAuthority({comment: None}), extractor=extractor)
    by_observation = {}
    for fragment in request.catalog.fragments:
        by_observation.setdefault(fragment.anchor.observation_id, []).append(fragment.primary_eligible)
    core = next(item.id for item in projection.observations if item.observation_type == "issue_core")

    # The comment reads with the core it follows; only the comment is Primary.
    assert set(by_observation) == {comment, core}
    assert all(by_observation[comment]) and not any(by_observation[core])


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
    extractor = MemoryExtractor(model="fixture", structured_llm_client=Client(limit=200_000))
    [request] = plan(projection, ExtractionAuthority({comment: None}), extractor=extractor)

    assert long_description.strip() in "\n".join(texts(request))


def test_small_real_window_packs_whole_tables_beside_the_route_output_capacity(monkeypatch):
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
                {"content": text, "memory_type": "fact", "primary_ref": ref, "required_refs": [], "evidence_displays": evidence_displays(ref, [])},
            ]})

    client = TimingOutClient()
    extractor = MemoryExtractor(model="fixture", structured_llm_client=client)
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
@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (
            StructuredLlmError("too large", terminal_category="provider_error", error_code=INPUT_CAPACITY_EXCEEDED),
            INPUT_CAPACITY_EXCEEDED,
        ),
        (
            StructuredLlmError("cut off", terminal_category="invalid_response", error_code=OUTPUT_TRUNCATED),
            OUTPUT_CAPACITY_EXCEEDED,
        ),
    ],
    ids=["input", "output"],
)
async def test_a_reading_group_beyond_the_route_capacity_alone_is_skipped_locally(error, reason, caplog):
    projection = _projection(
        primary_content="Rule one applies.\n\nRule two is far too large.\n", context_content="Country: US.\n",
    )
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")

    class OverflowingClient(Client):
        async def extract_projection_fragment_memories(self, prompt, **kwargs):
            payload = json.loads(prompt.split("<evidence_fragment_catalog", 1)[1].split(">", 1)[1].split(
                "\n</evidence_fragment_catalog>", 1)[0])
            rows = payload["primary_candidates"]
            if any("too large" in row[1] for row in rows):
                raise error
            ref, text = rows[0][0], rows[0][1]
            return ProjectionFragmentMemoryExtractionResponse.model_validate({"memories": [
                {"content": text.strip(), "memory_type": "fact", "primary_ref": ref, "required_refs": [], "evidence_displays": evidence_displays(ref, [])},
            ]})

    extractor = MemoryExtractor(model="fixture", structured_llm_client=OverflowingClient(limit=40000))
    [request] = plan(projection, ExtractionAuthority({body_id(projection): None}), extractor=extractor)
    with caplog.at_level("WARNING", logger="memforge.pipeline.memory_extractor"):
        result = await extractor.extract_projection_fragment_memories(
            request.catalog, source_type="confluence", revision_context=context,
        )

    assert result.error_type is None
    assert [m.content for m in result.memories] == ["Rule one applies."]
    assert result.metadata["skipped_reading_group_count"] == 1
    assert f"reason={reason}" in caplog.text


@pytest.mark.asyncio
async def test_only_the_reading_group_a_halved_request_cannot_bind_signals_its_rejection():
    from memforge.evals.agent_evaluation import QualitySignalCollector, quality_signal_scope

    projection = _projection(
        primary_content="Rule one applies.\n\nRule two confuses the model.\n", context_content="Country: US.\n",
    )
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")

    class UnknownRefClient(Client):
        """Cites a ref outside the catalog for rule two, even after the correction."""

        def __init__(self):
            super().__init__(limit=40000)
            self.extraction_prompts = []

        async def extract_projection_fragment_memories(self, prompt, **kwargs):
            self.extraction_prompts.append(prompt)
            payload = json.loads(prompt.split("<evidence_fragment_catalog", 1)[1].split(">", 1)[1].split(
                "\n</evidence_fragment_catalog>", 1)[0])
            memories = []
            for ref, text, *_rest in payload["primary_candidates"]:
                primary_ref = "p999999" if "confuses" in text else ref
                memories.append({
                    "content": text.strip(), "memory_type": "fact", "primary_ref": primary_ref,
                    "required_refs": [], "evidence_displays": evidence_displays(primary_ref, []),
                })
            return ProjectionFragmentMemoryExtractionResponse.model_validate({"memories": memories})

    client = UnknownRefClient()
    extractor = MemoryExtractor(model="fixture", structured_llm_client=client)
    [request] = plan(projection, ExtractionAuthority({body_id(projection): None}), extractor=extractor)
    collector = QualitySignalCollector()
    with quality_signal_scope(collector):
        result = await extractor.extract_projection_fragment_memories(
            request.catalog, source_type="confluence", revision_context=context,
        )

    assert result.error_type is None
    assert [m.content for m in result.memories] == ["Rule one applies."]
    # Both groups together, its correction, then rule two alone and its correction.
    assert len(client.extraction_prompts) == 5
    assert result.metadata["fragment_selection_rejection_counts"] == {"unknown_ref": 4}
    assert result.metadata["selection_skipped_reading_group_count"] == 1
    assert sorted((s.outcome, s.reason_code) for s in collector.snapshot()) == [
        ("expected", "fragment_selection_resolved"), ("rejected", "unknown_ref"),
    ]


@pytest.mark.asyncio
async def test_a_reading_group_whose_output_stays_invalid_is_skipped_locally():
    projection = _projection(
        primary_content="Rule one applies.\n\nRule two confuses the model.\n", context_content="Country: US.\n",
    )
    context = RevisionAssessmentContext(projection=projection, base=None, access_context_hash="scope")

    class MalformedClient(Client):
        async def extract_projection_fragment_memories(self, prompt, **kwargs):
            payload = json.loads(prompt.split("<evidence_fragment_catalog", 1)[1].split(">", 1)[1].split(
                "\n</evidence_fragment_catalog>", 1)[0])
            rows = payload["primary_candidates"]
            if any("confuses" in row[1] for row in rows):
                raise StructuredLlmError(
                    "ambiguous structured JSON objects", terminal_category="invalid_response", error_code="ValueError",
                )
            ref, text = rows[0][0], rows[0][1]
            return ProjectionFragmentMemoryExtractionResponse.model_validate({"memories": [
                {"content": text.strip(), "memory_type": "fact", "primary_ref": ref, "required_refs": [], "evidence_displays": evidence_displays(ref, [])},
            ]})

    extractor = MemoryExtractor(model="fixture", structured_llm_client=MalformedClient(limit=40000))
    [request] = plan(projection, ExtractionAuthority({body_id(projection): None}), extractor=extractor)
    result = await extractor.extract_projection_fragment_memories(
        request.catalog, source_type="confluence", revision_context=context,
    )

    assert result.error_type is None
    assert [m.content for m in result.memories] == ["Rule one applies."]
    assert result.metadata["skipped_reading_group_count"] == 1


def _updated_normalized_document(initial, body):
    """The next revision of the declared normalized document."""
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
        source_type="validation_normalized_document",
        run_id="run-c-update",
        item=item,
        raw=RawContent(item=item, body=body.encode(), content_type="text/markdown"),
        normalized=NormalizedContent(item=item, markdown_body=body),
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={revision.observation_id: revision for revision in initial.observation_revisions},
    )


# An incremental Teams window: the base revision holds the earlier message; the
# target adds a reply that agrees with it.
EARLIER_MESSAGE = "checkpointer 先不自己实现，等官方支持"
REPLY_MESSAGE = "嗯，如果不严重的话可以等等"


def _teams_reply_window(earlier_message=EARLIER_MESSAGE, reply_message=REPLY_MESSAGE):
    from tests.test_projection_context import _teams_projection

    initial = _teams_projection((earlier_message,))
    target = _teams_projection(
        (earlier_message, reply_message),
        run_id="run-teams-reply",
        prior_unit_revision=initial.source_unit_revisions[0],
        prior_observation_revisions={revision.observation_id: revision for revision in initial.observation_revisions},
    )
    authority = plan_projection_evidence_work(
        target, committed_base_snapshot=_committed_snapshot(initial), reprocess_all_current_observations=False,
    )
    assert isinstance(authority, ExtractionAuthority)
    return initial, target, authority


class _TeamsReplyClient(Client):
    """Selects the earlier message as Primary; with ``corrects``, its correction selects the reply."""

    def __init__(self, *, corrects):
        super().__init__(limit=40000)
        self.corrects = corrects
        self.extraction_prompts = []
        self.refs = {}

    async def extract_projection_fragment_memories(self, prompt, **kwargs):
        self.extraction_prompts.append(prompt)
        earlier, reply = self.refs["earlier"], self.refs["reply"]
        primary, required = (reply, earlier) if self.corrects and "<correction>" in prompt else (earlier, reply)
        return ProjectionFragmentMemoryExtractionResponse.model_validate({"memories": [{
            "content": "checkpointer 先不自己实现，问题不严重时等官方支持。",
            "memory_type": "decision",
            "primary_ref": primary,
            "required_refs": [required],
            "evidence_displays": evidence_displays(primary, [required]),
        }]})


async def _extract_teams_reply(client):
    from memforge.evals.agent_evaluation import QualitySignalCollector, quality_signal_scope

    initial, target, authority = _teams_reply_window()
    extractor = MemoryExtractor(model="fixture", structured_llm_client=client)
    extraction = extraction_plan(target, authority, extractor=extractor, base=initial)
    [request] = extraction.requests
    [earlier] = [f for f in request.catalog.fragments if EARLIER_MESSAGE in f.presentation_text]
    [reply] = [f for f in request.catalog.fragments if REPLY_MESSAGE in f.presentation_text]
    # On the update only the reply may be Primary; the earlier message is Required-only.
    assert reply.primary_eligible and not earlier.primary_eligible
    client.refs = {"earlier": earlier.reference, "reply": reply.reference}
    collector = QualitySignalCollector()
    with quality_signal_scope(collector):
        result = await extractor.extract_projection_fragment_memories(
            request.catalog, source_type="teams", doc_type="teams",
            revision_context=RevisionAssessmentContext(projection=target, base=initial, access_context_hash="scope"),
        )
    return extraction, result, earlier, reply, collector.snapshot()


def test_a_chinese_reply_to_an_english_message_is_the_only_primary_and_the_prompt_says_it_sets_the_language():
    english = "We will not implement the checkpointer ourselves; we wait for official support."
    initial, target, authority = _teams_reply_window(english, REPLY_MESSAGE)
    extractor = MemoryExtractor(model="fixture", structured_llm_client=Client(limit=40000))

    [request] = extraction_plan(target, authority, extractor=extractor, base=initial).requests
    prompt = MemoryExtractor.projection_fragment_prompt(
        request.catalog, source_type="teams", doc_type="teams",
        revision_context=RevisionAssessmentContext(projection=target, base=initial, access_context_hash="scope"),
    )

    [earlier] = [f for f in request.catalog.fragments if english in f.presentation_text]
    [reply] = [f for f in request.catalog.fragments if REPLY_MESSAGE in f.presentation_text]
    # The Chinese reply is the only Primary; the English message it settles can only be Required.
    assert reply.primary_eligible and not earlier.primary_eligible
    assert "THE PRIMARY SETS THE LANGUAGE." in prompt
    assert "even when its Required evidence is English" in prompt
    assert "in the Primary's language" in prompt


@pytest.mark.asyncio
async def test_teams_reply_selected_as_required_only_is_corrected_to_primary_on_the_reply():
    client = _TeamsReplyClient(corrects=True)
    _extraction, result, earlier, reply, signals = await _extract_teams_reply(client)

    assert "A new reply that agrees with, answers or decides on earlier content" in client.extraction_prompts[0]
    assert len(client.extraction_prompts) == 2
    correction = client.extraction_prompts[1].removeprefix(client.extraction_prompts[0])
    assert (
        f"- memories[0].primary_ref: Fragment reference is not eligible for primary: {earlier.reference}. "
        f"Allowed Primary refs: {reply.reference}"
    ) in correction
    assert result.error_type is None
    [memory] = result.memories
    assert memory.source_observation_id == reply.anchor.observation_id
    assert memory.required_source_observation_ids == [earlier.anchor.observation_id]
    assert result.metadata["fragment_selection_rejection_counts"] == {"ineligible_role": 1}
    assert result.metadata["selection_corrected_request_count"] == 1
    assert result.metadata["skipped_reading_group_count"] == 0
    # The correction recovered the rejection, so only the bound selection is signalled.
    assert [(s.outcome, s.reason_code) for s in signals] == [("expected", "fragment_selection_resolved")]


@pytest.mark.asyncio
async def test_teams_reply_still_selected_as_required_only_skips_its_reading_group_and_the_document_commits(caplog):
    from memforge.source_derivation import assemble_source_derivation_results

    client = _TeamsReplyClient(corrects=False)
    with caplog.at_level("WARNING", logger="memforge.pipeline.memory_extractor"):
        extraction, result, _earlier, _reply, signals = await _extract_teams_reply(client)

    assert len(client.extraction_prompts) == 2
    assert result.error_type is None and result.memories == []
    assert "safe_error_code" not in result.metadata
    assert result.metadata["selection_skipped_reading_group_count"] == 1
    assert result.metadata["skipped_reading_group_count"] == 1
    assert "extraction_reading_group_unread" in caplog.text
    # The skipped ReadingGroup signals its final rejection once; the metrics count both attempts.
    assert [(s.outcome, s.reason_code) for s in signals] == [("rejected", "ineligible_role")]

    _initial, target, _authority = _teams_reply_window()
    unit = assemble_source_derivation_results(projection=target, plan=extraction, results=(result,))
    assert unit.error_type is None
    assert unit.metadata["skipped_reading_group_count"] == 1
    assert unit.metadata["fragment_selection_rejection_counts"] == {"ineligible_role": 2}
    assert unit.metadata["selection_skipped_reading_group_count"] == 1
