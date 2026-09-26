"""Plan exact Claim Extraction requests before their immutable derivation manifest is staged."""

import hashlib

from memforge.llm.batch_runner import LlmBatchRunner
from memforge.llm.structured import INPUT_CAPACITY_EXCEEDED
from memforge.pipeline.memory_extractor import ExtractionReading, MemoryExtractor
from memforge.pipeline.projection_context import ExtractionAuthority, ExtractionPlan, ExtractionRequest
from memforge.pipeline.revision_assessment import RevisionAssessmentContext


def plan_extraction_requests(
    context: RevisionAssessmentContext,
    authority: ExtractionAuthority,
    *,
    extractor: MemoryExtractor,
    source_type: str,
    doc_type: str,
) -> ExtractionPlan:
    """Pack every ReadingGroup that holds authorized Primary into the fewest requests that fit.

    An update authorizes only its changed structures, so it reads those
    ReadingGroups; a first import or reprocess reads every ReadingGroup. There is
    no cost comparison with another reading scope and no truncation: a
    ReadingGroup that alone exceeds the route's capacity is skipped with a
    diagnostic, and every other group is planned.
    """

    reading = ExtractionReading.of_authority(context, authority, source_type=source_type, doc_type=doc_type)
    if not reading.items:
        return ExtractionPlan(())
    runner = LlmBatchRunner(extractor.structured_llm_client, model=extractor.model)
    planned = runner.plan_items(
        tuple(reading.items),
        lambda item_ids, _parts: reading.request(
            item_ids, output_tokens=extractor.fragment_output_tokens, fits=runner.fits,
        ),
    )
    source_unit_id = context.projection.source_units[0].id
    requests = []
    for entry in planned.requests:
        request_catalog = reading.catalog_for(entry.item_ids)
        prompt_sha256 = hashlib.sha256(entry.request.prompt.encode("utf-8")).hexdigest()
        identity = hashlib.sha256((request_catalog.digest + prompt_sha256).encode("utf-8")).hexdigest()
        requests.append(
            ExtractionRequest(
                id="request-" + identity,
                source_unit_id=source_unit_id,
                catalog=request_catalog,
                prompt_sha256=prompt_sha256,
            )
        )

    unfit = set(planned.unfit)
    expected = {
        fragment.anchor
        for item_id, group in reading.items.items() if item_id not in unfit
        for fragment in group if fragment.primary_eligible
    }
    actual = [
        fragment.anchor
        for request in requests
        for fragment in request.catalog.fragments
        if fragment.primary_eligible
    ]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("extraction request authority coverage mismatch")
    return ExtractionPlan(tuple(requests), reading.report_skipped(dict.fromkeys(planned.unfit, INPUT_CAPACITY_EXCEEDED)))
