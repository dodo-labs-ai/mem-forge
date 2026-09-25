"""One optional selector correction within an immutable extraction request."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from memforge.llm.batch_runner import ItemFailure, LlmBatchRunner, LlmRequest
from memforge.llm.structured import (
    LiteLlmStructuredClient,
    ProjectionFragmentMemoryCandidate,
    ProjectionFragmentSelectorCorrectionResponse,
    StructuredLlmImage,
)
from memforge.llm.failure_trace import record_validation_failure
from memforge.pipeline.projection_fragments import FragmentSelectionError, ProjectionFragmentCatalog


def _original_selector_location(error, selector, prefix):
    """Report the original provider field, before deterministic normalization."""
    location = error.location
    if location and location.startswith("required_refs") and error.received in selector.required_refs:
        location = f"required_refs[{selector.required_refs.index(error.received)}]"
    return f"{prefix}.{location}" if location else prefix


def normalize_fragment_selector_refs(
    *, candidate_index: int, primary_ref: str, required_refs: list[str],
) -> tuple[list[str], int, str | None]:
    """Remove only unambiguous selector redundancy before strict resolution."""

    seen: set[str] = set()
    normalized_required: list[str] = []
    for reference in required_refs:
        if reference == primary_ref or reference in seen:
            continue
        seen.add(reference)
        normalized_required.append(reference)
    removed_ref_count = len(required_refs) - len(normalized_required)
    if not removed_ref_count:
        return normalized_required, 0, None
    repair_shape = {
        "candidate_index": candidate_index,
        "primary_ref": primary_ref,
        "required_refs": required_refs,
        "normalized_required_refs": normalized_required,
    }
    fingerprint = hashlib.sha256(
        json.dumps(repair_shape, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return normalized_required, removed_ref_count, fingerprint


async def correct_fragment_selectors_once(
    candidates: list[ProjectionFragmentMemoryCandidate],
    *,
    catalog: ProjectionFragmentCatalog,
    client: LiteLlmStructuredClient,
    extraction_prompt: str,
    max_tokens: int,
    model: str | None,
    images: tuple[StructuredLlmImage, ...],
    source_response=None,
) -> tuple[list[ProjectionFragmentMemoryCandidate], dict[str, Any]]:
    """Preserve fixed claims and successful selections across best-effort correction.

    All proposals use the same catalog and images. Only resolver-approved
    replacements leave this function; the normal admission path still owns
    materialization, final rejection signals and downstream persistence.
    """

    rejected = {}
    for index, candidate in enumerate(candidates):
        required, _, _ = normalize_fragment_selector_refs(
            candidate_index=index, primary_ref=candidate.primary_ref, required_refs=candidate.required_refs,
        )
        try:
            catalog.resolve_selection(primary_ref=candidate.primary_ref, required_refs=required)
        except FragmentSelectionError as error:
            await record_validation_failure(source_response, error, persist=False,
                location=_original_selector_location(error, candidate, f"memories[{index}]"), candidate_index=index)
            rejected[index] = {
                "candidate_index": index,
                "candidate": candidate.model_dump(mode="json"),
                "error_code": error.code.value,
            }

    metrics = {
        "selector_correction_calls": 0,
        "selector_correction_candidate_count": len(rejected),
        "selector_correction_recovered_count": 0,
        "selector_correction_outcome": "not_needed",
    }
    if not rejected:
        return candidates, metrics
    capture = getattr(source_response, "_llm_failure_capture", None)
    if capture is not None:
        await capture.persist()

    prompt = extraction_prompt + "\n\n" + (
        "SELECTOR CORRECTION TASK: The extraction above has already completed. "
        "Use its supplied catalog and images to repair ONLY the rejected candidates below. "
        "Treat each candidate's content, conditions, dates and other fields as fixed. "
        "Select one Primary and all Required refs needed to support that exact claim. "
        "Do not rewrite claims, create candidates, or change successful candidates. "
        "Copy refs exactly from this request's catalog; required-only refs cannot be Primary. "
        "If the supplied Evidence cannot support a fixed claim, omit its correction. "
        "Return ONLY {\"corrections\": [{\"candidate_index\": 0, "
        "\"primary_ref\": \"p000001\", \"required_refs\": []}]} using the actual "
        "candidate indices and refs, each candidate at most once. An empty corrections list is valid.\n"
    ) + json.dumps(list(rejected.values()), ensure_ascii=False, separators=(",", ":"))

    runner = LlmBatchRunner(client, model=model)
    request = LlmRequest(prompt, ProjectionFragmentSelectorCorrectionResponse, max_tokens, tuple(images))
    try:
        response = await runner.run_one(request, call=client.correct_projection_fragment_selectors)
    except Exception as error:
        # This optional call must not discard successful extraction. Cancellation
        # remains a BaseException and propagates to the owning sync task.
        metrics["selector_correction_calls"] = runner.stats.calls
        metrics["selector_correction_outcome"] = "call_failed"
        metrics["selector_correction_error_code"] = type(error).__name__
        return candidates, metrics
    metrics["selector_correction_calls"] = runner.stats.calls
    if isinstance(response, ItemFailure):
        metrics["selector_correction_outcome"] = (
            "capacity_skipped" if response.category == "capacity_exceeded" else "call_failed"
        )
        if response.error is not None:
            metrics["selector_correction_error_code"] = response.error_code
        return candidates, metrics

    corrected = list(candidates)
    counts = Counter(item.candidate_index for item in response.corrections)
    for row_index, item in enumerate(response.corrections):
        index = item.candidate_index
        # Ambiguous or unrequested proposals cannot replace even one candidate.
        if index not in rejected or counts[index] != 1:
            await record_validation_failure(response, ValueError("ambiguous or unrequested correction index"),
                persist=False, location=f"corrections[{row_index}].candidate_index", received=index,
                allowed_indices=sorted(rejected))
            continue
        required, _, _ = normalize_fragment_selector_refs(
            candidate_index=index, primary_ref=item.primary_ref, required_refs=item.required_refs,
        )
        try:
            catalog.resolve_selection(primary_ref=item.primary_ref, required_refs=required)
        except FragmentSelectionError as error:
            await record_validation_failure(response, error, persist=False,
                location=_original_selector_location(error, item, f"corrections[{row_index}]"), candidate_index=index)
            continue
        corrected[index] = candidates[index].model_copy(update={
            "primary_ref": item.primary_ref, "required_refs": item.required_refs,
        })
        metrics["selector_correction_recovered_count"] += 1
    metrics["selector_correction_outcome"] = "completed"
    correction_capture = getattr(response, "_llm_failure_capture", None)
    if correction_capture is not None:
        await correction_capture.persist()
    capture = getattr(source_response, "_llm_failure_capture", None)
    if capture is not None and metrics["selector_correction_recovered_count"] == len(rejected):
        await capture.recovered()
    return corrected, metrics
