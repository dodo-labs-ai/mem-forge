"""Plan exact Claim Extraction requests before their immutable derivation manifest is staged."""

import hashlib

from memforge.llm.batch_runner import ItemCapacityExceeded, LlmBatchRunner
from memforge.llm.structured import INPUT_CAPACITY_EXCEEDED
from memforge.pipeline.memory_extractor import (
    EXTRACTION_CAPACITY_EXCEEDED_MESSAGE,
    ExtractionReading,
    MemoryExtractor,
)
from memforge.pipeline.projection_context import ExtractionAuthority, ExtractionRequest
from memforge.pipeline.revision_assessment import RevisionAssessmentContext


class ExtractionCapacityExceeded(RuntimeError):
    """One ReadingGroup with its reading context alone exceeds the extraction route's capacity.

    Planning the same input again fails the same way, so the revision stays
    uncommitted until the route or the Unit changes.
    """

    retryable = False
    reason_code = INPUT_CAPACITY_EXCEEDED

    def __init__(self) -> None:
        super().__init__(f"{INPUT_CAPACITY_EXCEEDED}: {EXTRACTION_CAPACITY_EXCEEDED_MESSAGE}")


def plan_extraction_requests(
    context: RevisionAssessmentContext,
    authority: ExtractionAuthority,
    *,
    extractor: MemoryExtractor,
    source_type: str,
    doc_type: str,
) -> tuple[ExtractionRequest, ...]:
    """Pack every ReadingGroup that holds authorized Primary into the fewest requests that fit.

    An update authorizes only its changed structures, so it reads those
    ReadingGroups; a first import or reprocess reads every ReadingGroup. There is
    no cost comparison with another reading scope and no truncation: a
    ReadingGroup that alone exceeds the route's capacity is a typed failure.
    """

    reading = ExtractionReading.of_authority(context, authority, source_type=source_type, doc_type=doc_type)
    if not reading.items:
        return ()
    runner = LlmBatchRunner(extractor.structured_llm_client, model=extractor.model)
    try:
        planned = runner.plan_items(
            tuple(reading.items),
            lambda item_ids, _parts: reading.request(
                item_ids, output_tokens=extractor.fragment_output_tokens, fits=runner.fits,
            ),
        )
    except ItemCapacityExceeded as error:
        raise ExtractionCapacityExceeded() from error
    source_unit_id = context.projection.source_units[0].id
    requests = []
    for entry in planned:
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

    expected = {fragment.anchor for fragment in reading.catalog.fragments if fragment.primary_eligible}
    actual = [
        fragment.anchor
        for request in requests
        for fragment in request.catalog.fragments
        if fragment.primary_eligible
    ]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("extraction request authority coverage mismatch")
    return tuple(requests)
