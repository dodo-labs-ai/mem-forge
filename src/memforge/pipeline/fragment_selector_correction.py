"""One optional selector correction within an immutable extraction request."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from typing import Any

from memforge.llm.structured import (
    ProjectionFragmentMemoryCandidate,
    ProjectionFragmentSelectorCorrectionResponse,
    SourceSupportStructuredClient,
    StructuredLlmError,
    StructuredLlmImage,
)
from memforge.pipeline.projection_fragments import FragmentSelectionError, ProjectionFragmentCatalog


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
    client: SourceSupportStructuredClient,
    extraction_prompt: str,
    max_tokens: int,
    model: str | None,
    images: tuple[StructuredLlmImage, ...],
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

    try:
        if not client.request_fits(
            prompt, response_format=ProjectionFragmentSelectorCorrectionResponse,
            max_tokens=max_tokens, model=model, images=images,
        ):
            metrics["selector_correction_outcome"] = "capacity_skipped"
            return candidates, metrics
        metrics["selector_correction_calls"] = 1
        response = await client.correct_projection_fragment_selectors(
            prompt, max_tokens=max_tokens, model=model, images=images,
        )
    except Exception as error:
        # This optional call must not discard successful extraction. Cancellation
        # remains a BaseException and propagates to the owning sync task.
        metrics["selector_correction_outcome"] = "call_failed"
        metrics["selector_correction_error_code"] = (
            error.error_code if isinstance(error, StructuredLlmError) else type(error).__name__
        )
        return candidates, metrics

    corrected = list(candidates)
    counts = Counter(item.candidate_index for item in response.corrections)
    for item in response.corrections:
        index = item.candidate_index
        # Ambiguous or unrequested proposals cannot replace even one candidate.
        if index not in rejected or counts[index] != 1:
            continue
        required, _, _ = normalize_fragment_selector_refs(
            candidate_index=index, primary_ref=item.primary_ref, required_refs=item.required_refs,
        )
        try:
            catalog.resolve_selection(primary_ref=item.primary_ref, required_refs=required)
        except FragmentSelectionError:
            continue
        corrected[index] = candidates[index].model_copy(update={
            "primary_ref": item.primary_ref, "required_refs": item.required_refs,
        })
        metrics["selector_correction_recovered_count"] += 1
    metrics["selector_correction_outcome"] = "completed"
    return corrected, metrics
