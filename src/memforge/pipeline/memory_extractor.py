"""Single semantic extraction of claim-sized Memory candidates."""

from __future__ import annotations

import hashlib
import json
import logging
from time import perf_counter

from memforge.config import DEFAULT_MEMORY_EXTRACTION_MAX_TOKENS
from memforge.evals.agent_evaluation import QualitySignal, record_quality_signal
from memforge.llm.batch_runner import ItemFailure, LlmBatchRunner, LlmRequest
from memforge.llm.structured import (
    LiteLlmStructuredClient,
    ProjectionFragmentMemoryExtractionResponse,
    StructuredLlmConfig,
    StructuredLlmError,
    StructuredLlmImage,
)
from memforge.models import MemoryExtractionResult, RawMemory
from memforge.pipeline.extraction_contract import (
    DURABLE_MEMORY_QUALITY_RULES,
    PROJECTION_EXTRACTION_CONTRACT_VERSION,
)
from memforge.pipeline.fragment_selector_correction import (
    correct_fragment_selectors_once,
    normalize_fragment_selector_refs,
)
from memforge.pipeline.projection_fragments import (
    FragmentSelectionError,
    ProjectionFragmentCatalog,
    SupportRevalidationLimitation,
)

logger = logging.getLogger(__name__)

__all__ = ["MemoryExtractor"]

# Requested extraction output: a response envelope plus, per authorized Primary
# Fragment, room for its claims (at least one claim, else about half its text).
_FRAGMENT_OUTPUT_BASE_TOKENS = 512
_FRAGMENT_OUTPUT_MIN_TOKENS = 768
_FRAGMENT_OUTPUT_CHARS_PER_TOKEN = 2


PROJECTION_FRAGMENT_EXTRACTION_PROMPT = """You are extracting durable atomic knowledge from one authorized Source Unit catalog.

<source_type>{source_type}</source_type>
<doc_type>{doc_type}</doc_type>
Catalog rows are [ref, exact source text, optional metadata]. Headings are ordinary selectable Fragments.
Preserve table column/row associations, list order, code indentation and explicit exceptions.
A table ref contains the complete table; read its headers before asserting a cell value.
A figure preserves its image link and caption together. Only a supplied image Artifact
ref proves image contents; a caption or URL alone never proves unseen image details.
Canonical fromString is the previous value; toString is the new value. Select the
field-name/time refs when needed to state the change accurately.
Structural groups describe ancestry, not additional Evidence. When a heading defines claim scope, select its current ref as Required.
Only the following application-owned Evidence Fragments may support a Memory:
<evidence_fragment_catalog digest="{catalog_digest}">
{fragment_catalog}
</evidence_fragment_catalog>
<read_only_context>
{context_observations}
</read_only_context>

Each Memory must contain exactly:
- "content": one self-contained durable claim
- "memory_type": one of "fact", "decision", "convention", "procedure"
- "confidence": 0.0-1.0
- "entity_refs": entity names copied from supporting Fragments
- "valid_from": YYYY-MM-DD or null
- "valid_until": YYYY-MM-DD or null
- "primary_ref": exactly one `pNNNNNN` ref from `primary_candidates` that directly states the claim
- "required_refs": a duplicate-free list of presented `pNNNNNN` or `rNNNNNN` refs without which the claim would be invalid or ambiguous

Do not return Evidence text, quotes, Observation or Revision IDs, offsets, hashes, profile names, catalog digests, Context refs, or lifecycle actions. Split a candidate that would otherwise need multiple independently claim-bearing Primary refs.

""" + DURABLE_MEMORY_QUALITY_RULES + """Context outside the Fragment catalog is read-only. `required_only_candidates` may be selected as Required but never as Primary. If a durable claim is stated only by required_only_candidates, return an empty memories array. Fragment refs are valid only in this catalog. Never invent or transform a ref.

Return ONLY a JSON object with a "memories" array. Use {{"memories": []}} when there are no memories."""


# ---------------------------------------------------------------------------
# MemoryExtractor class
# ---------------------------------------------------------------------------


class MemoryExtractor:
    """Extract current claims without loading historical Memory rows.

    Application runtimes inject ``structured_llm_client`` through their
    RuntimeProvider. The ``api_key`` construction path remains only for
    provider-neutral standalone library use.
    """

    def fragment_output_tokens(self, catalog) -> int:
        """Request output for this request's authorized content, not a full document."""
        requested = _FRAGMENT_OUTPUT_BASE_TOKENS + sum(
            max(_FRAGMENT_OUTPUT_MIN_TOKENS, len(fragment.presentation_text) // _FRAGMENT_OUTPUT_CHARS_PER_TOKEN)
            for fragment in catalog.fragments if fragment.primary_eligible
        )
        return min(self.max_tokens, requested)

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MEMORY_EXTRACTION_MAX_TOKENS,
        request_timeout_s: float = 300.0,
        structured_llm_client=None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.structured_llm_client = structured_llm_client
        if self.structured_llm_client is None and api_key:
            self.structured_llm_client = LiteLlmStructuredClient(
                StructuredLlmConfig(
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    timeout_s=request_timeout_s,
                )
            )

    @staticmethod
    def projection_fragment_prompt(
        catalog,
        *,
        source_type,
        doc_type,
        context_markdown="",
        context_observation_ids=(),
        revision_context=None,
        mode="authorized_work",
    ):
        payload = revision_context.model_payload(catalog) if revision_context is not None else catalog.model_payload()
        context_ids = set(context_observation_ids)
        if context_ids and revision_context is not None:
            required_context_anchors = {
                fragment.anchor
                for fragment in revision_context.full_fragments
                if fragment.anchor.observation_id in context_ids
            }
            represented_context_anchors = {
                fragment.anchor
                for fragment in catalog.fragments
                if fragment.anchor.observation_id in context_ids
            }
            if (
                context_ids
                <= {anchor.observation_id for anchor in required_context_anchors}
                and required_context_anchors <= represented_context_anchors
            ):
                context_markdown = ""
        if revision_context is not None and mode in {"delta", "full"}:
            context_payload = {
                "input_mode": mode,
                **(
                    {"removed_historical": revision_context.delta()[1]}
                    if mode == "delta"
                    else {}
                ),
                **({"additional_context": context_markdown} if context_markdown else {}),
            }
            context_observations = json.dumps(
                context_payload, ensure_ascii=False, separators=(",", ":")
            )
        else:
            context_observations = context_markdown
        return PROJECTION_FRAGMENT_EXTRACTION_PROMPT.format(
            source_type=source_type, doc_type=doc_type, catalog_digest=catalog.digest,
            fragment_catalog=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            context_observations=context_observations,
        )

    async def extract_projection_fragment_memories(
        self,
        catalog: ProjectionFragmentCatalog,
        *,
        source_type: str,
        doc_type: str = "unknown",
        context_markdown: str = "",
        images: tuple[StructuredLlmImage, ...] = (),
        revision_context=None,
        prepared_prompt: str | None = None,
        prepared_input_mode: str | None = None,
        prepared_selection_reason: str | None = None,
        prepared_estimated_cost: dict[str, int] | None = None,
        context_observation_ids: tuple[str, ...] = (),
    ) -> MemoryExtractionResult:
        """Select exact current Evidence within this work's Primary authority."""

        if not self.structured_llm_client:
            return MemoryExtractionResult(
                error_type="llm_client_unavailable",
                error="No LLM client configured for memory extraction",
            )
        if not catalog.usable:
            reason_codes = sorted({error.code.value for error in catalog.errors})
            reason_code = (
                "catalog_too_large"
                if "catalog_too_large" in reason_codes
                else (reason_codes[0] if reason_codes else "catalog_unusable")
            )
            record_quality_signal(
                QualitySignal(
                    event_name="evidence_admission_outcome",
                    outcome="rejected",
                    reason_code=reason_code,
                )
            )
            return MemoryExtractionResult(
                error_type="evidence_catalog_unusable",
                error="Evidence Fragment catalog is not usable",
                metadata={
                    "extraction_contract_version": PROJECTION_EXTRACTION_CONTRACT_VERSION,
                    "catalog_digest": catalog.digest,
                    "catalog_error_codes": reason_codes,
                },
            )
        invoke = getattr(
            self.structured_llm_client,
            "extract_projection_fragment_memories",
            None,
        )
        if not callable(invoke):
            return MemoryExtractionResult(
                error_type="projection_extraction_v9_unavailable",
                error="Structured client does not implement projection-extraction-v9",
            )

        def make_prompt(selected_catalog, mode):
            return self.projection_fragment_prompt(selected_catalog, source_type=source_type, doc_type=doc_type,
                context_markdown=context_markdown, context_observation_ids=context_observation_ids,
                revision_context=revision_context, mode=mode)

        input_mode = prepared_input_mode or "authorized_work"
        input_selection_reason = prepared_selection_reason
        estimated_input_cost = prepared_estimated_cost
        prompt = prepared_prompt or make_prompt(catalog, input_mode)
        if revision_context is not None and prepared_prompt is None:
            from memforge.pipeline.revision_input import (
                ExtractionInputTask,
                InputCandidate,
                InputCost,
                PlannedTransport,
                RevisionInputPlanner,
            )

            runner = LlmBatchRunner(self.structured_llm_client, model=self.model)

            def plan_request(candidate: InputCandidate, *, load_images: bool) -> LlmRequest | None:
                """Bound one whole-catalog request, or None when it cannot fit."""

                def render():
                    request = LlmRequest(
                        make_prompt(candidate.catalog, candidate.mode.value),
                        ProjectionFragmentMemoryExtractionResponse,
                        self.fragment_output_tokens(candidate.catalog),
                    )
                    if not load_images:
                        return request
                    return revision_context.attach_images(request, candidate.catalog, fits=runner.fits)

                return runner.fit(render)

            def request_cost(request: LlmRequest, *, complete: bool = True) -> InputCost:
                return InputCost(
                    input_tokens=self.structured_llm_client.request_tokens(
                        request.prompt, response_format=request.response_format,
                        model=self.model, images=request.images,
                    ),
                    output_tokens=request.max_tokens,
                    request_count=1,
                    image_count=len(request.images),
                    image_bytes=sum(len(image.body) for image in request.images),
                    complete=complete,
                )

            class RequestPolicy:
                @staticmethod
                def lower_bound(candidate: InputCandidate):
                    request = plan_request(candidate, load_images=False)
                    if request is None:
                        return None
                    has_images = any(fragment.kind.value == "artifact" for fragment in candidate.catalog.fragments)
                    return request_cost(request, complete=not has_images)

                @staticmethod
                def materialize(candidate: InputCandidate):
                    request = plan_request(candidate, load_images=True)
                    if request is None:
                        return None
                    return PlannedTransport(request_cost(request), (candidate.catalog, request.prompt, request.images))

            baseline = (
                revision_context.projection.deltas[0].previous_unit_revision_id
                if revision_context.projection.deltas
                else None
            )
            try:
                plan = RevisionInputPlanner.plan(
                    context=revision_context,
                    task=ExtractionInputTask(catalog, named_baseline_revision_id=baseline),
                    request_policy=RequestPolicy(),
                )
            except SupportRevalidationLimitation as error:
                return MemoryExtractionResult(
                    error_type="evidence_catalog_unusable",
                    error=str(error),
                    metadata={"catalog_error_codes": [error.code.value]},
                )
            catalog, prompt, images = plan.transport
            input_mode = plan.mode.value
            input_selection_reason = plan.selection_reason
            estimated_input_cost = plan.estimated_cost.as_payload()
        started = perf_counter()
        metrics = {
            "structured_llm_calls": 1,
            "input_mode": input_mode,
            "input_selection_reason": input_selection_reason,
            "estimated_input_cost": estimated_input_cost,
            "extraction_model": self.model,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
            "image_count": len(images),
            "image_bytes": sum(len(image.body) for image in images),
            "extraction_contract_version": PROJECTION_EXTRACTION_CONTRACT_VERSION,
            "catalog_digest": catalog.digest,
            "catalog_fragment_count": len(catalog.fragments),
        }
        runner = LlmBatchRunner(self.structured_llm_client, model=self.model)
        request = LlmRequest(
            prompt, ProjectionFragmentMemoryExtractionResponse, self.fragment_output_tokens(catalog), tuple(images),
        )
        try:
            response = await runner.run_one(request, call=invoke)
        except Exception as error:
            logger.error("Unexpected projection Fragment extraction error: %s", error)
            return MemoryExtractionResult(
                error_type="unexpected_error",
                error=str(error),
                metadata={
                    **metrics,
                    "structured_llm_elapsed_ms": max(
                        0, round((perf_counter() - started) * 1000)
                    ),
                },
            )
        if isinstance(response, ItemFailure):
            if response.category == "capacity_exceeded":
                return MemoryExtractionResult(
                    error_type="input_capacity_exceeded",
                    error="planned extraction request exceeds configured capability",
                )
            error = response.error
            validation_fields = error.validation_fields if isinstance(error, StructuredLlmError) else ()
            return MemoryExtractionResult(
                error_type="structured_llm_error",
                error=str(error),
                metadata={
                    **metrics,
                    "safe_error_code": response.error_code,
                    "safe_validation_fields": [
                        {"location": location, "type": rule_type}
                        for location, rule_type in validation_fields
                    ],
                    "structured_llm_elapsed_ms": max(
                        0, round((perf_counter() - started) * 1000)
                    ),
                },
            )

        candidates, correction_metrics = await correct_fragment_selectors_once(
            response.memories, catalog=catalog, client=self.structured_llm_client,
            extraction_prompt=prompt, max_tokens=self.fragment_output_tokens(catalog), model=self.model, images=images,
            source_response=response,
        )
        metrics.update(correction_metrics)
        metrics["structured_llm_calls"] += correction_metrics["selector_correction_calls"]

        memories: list[RawMemory] = []
        rejection_counts: dict[str, int] = {}
        selector_normalization_count = 0
        selector_normalization_fingerprints: list[str] = []
        for candidate_index, candidate in enumerate(candidates):
            normalized_required, removed_ref_count, repair_fingerprint = (
                normalize_fragment_selector_refs(
                    candidate_index=candidate_index,
                    primary_ref=candidate.primary_ref,
                    required_refs=candidate.required_refs,
                )
            )
            selector_normalization_count += removed_ref_count
            if repair_fingerprint is not None:
                selector_normalization_fingerprints.append(repair_fingerprint)
            candidate_content_hash = hashlib.sha256(
                candidate.content.encode("utf-8")
            ).hexdigest()
            candidate_hash = catalog.selection_fingerprint(
                candidate_content_hash=candidate_content_hash,
                primary_ref=candidate.primary_ref,
                required_refs=normalized_required,
            )
            try:
                selection = catalog.resolve_selection(
                    primary_ref=candidate.primary_ref,
                    required_refs=normalized_required,
                )
            except FragmentSelectionError as error:
                rejection_counts[error.code.value] = rejection_counts.get(error.code.value, 0) + 1
                record_quality_signal(
                    QualitySignal(
                        event_name="evidence_admission_outcome",
                        outcome="rejected",
                        reason_code=error.code.value,
                        prompt_hash=metrics["prompt_sha256"],
                        candidate_hash=candidate_hash,
                    )
                )
                continue
            record_quality_signal(
                QualitySignal(
                    event_name="evidence_admission_outcome",
                    outcome="degraded" if removed_ref_count else "expected",
                    reason_code=(
                        "fragment_selector_normalized"
                        if removed_ref_count
                        else "fragment_selection_resolved"
                    ),
                    prompt_hash=metrics["prompt_sha256"],
                    candidate_hash=candidate_hash,
                )
            )
            memories.append(
                RawMemory(
                    content=candidate.content,
                    memory_type=candidate.memory_type,
                    confidence=candidate.confidence,
                    entity_refs=list(candidate.entity_refs),
                    valid_from=candidate.valid_from,
                    valid_until=candidate.valid_until,
                    evidence_anchor="projection_fragment_catalog",
                    source_observation_id=selection.parts[0].anchor.observation_id,
                    required_source_observation_ids=[
                        part.anchor.observation_id
                        for part in selection.parts
                        if part.role.value == "required"
                    ],
                    resolved_evidence_selection=selection,
                )
            )
        return MemoryExtractionResult(
            memories=memories,
            metadata={
                **metrics,
                "resolved_fragment_selection_count": len(memories),
                "rejected_fragment_selection_count": (
                    len(response.memories) - len(memories)
                ),
                "fragment_selection_rejection_counts": rejection_counts,
                **(
                    {
                        "selector_normalized_candidate_count": len(
                            selector_normalization_fingerprints
                        ),
                        "selector_normalization_count": (
                            selector_normalization_count
                        ),
                        "selector_normalization_fingerprints": (
                            selector_normalization_fingerprints
                        ),
                    }
                    if selector_normalization_fingerprints
                    else {}
                ),
                "structured_llm_elapsed_ms": max(
                    0, round((perf_counter() - started) * 1000)
                ),
            },
        )
