"""Single semantic extraction of claim-sized Memory candidates."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import replace
from time import perf_counter

from memforge.config import DEFAULT_MEMORY_EXTRACTION_MAX_TOKENS
from memforge.evals.agent_evaluation import QualitySignal, record_quality_signal
from memforge.llm.batch_runner import ItemFailure, ItemTask, LlmBatchRunner, LlmRequest
from memforge.llm.structured import (
    INPUT_CAPACITY_EXCEEDED,
    LiteLlmStructuredClient,
    ProjectionFragmentMemoryExtractionResponse,
    StructuredLlmConfig,
    StructuredLlmError,
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
)
from memforge.pipeline.projection_context import ExtractionAuthority
from memforge.pipeline.revision_assessment import RevisionAssessmentContext, reading_group_label

logger = logging.getLogger(__name__)

__all__ = ["ExtractionReading", "MemoryExtractor"]

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
A row whose format is unit-identity names the Source Unit this catalog belongs to, such as its key, type or title.
It states no claim itself: select it as Required when a claim names or depends on that Unit.
Only the following application-owned Evidence Fragments may support a Memory:
<evidence_fragment_catalog digest="{catalog_digest}">
{fragment_catalog}
</evidence_fragment_catalog>

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

""" + DURABLE_MEMORY_QUALITY_RULES + """`required_only_candidates` may be selected as Required but never as Primary. If a durable claim is stated only by required_only_candidates, return an empty memories array. Fragment refs are valid only in this catalog. Never invent or transform a ref.

Return ONLY a JSON object with a "memories" array. Use {{"memories": []}} when there are no memories."""


class ExtractionReading:
    """One extraction catalog read as items: each ReadingGroup that holds authorized Primary.

    The catalog holds exactly its items and the context every reading of them
    adds. An item is keyed by the reference of its first Fragment. A request for
    some of the items reads them with their context, which is never Primary.
    """

    def __init__(
        self,
        catalog: ProjectionFragmentCatalog,
        context: RevisionAssessmentContext,
        *,
        source_type: str,
        doc_type: str,
    ) -> None:
        self.catalog = catalog
        self.context = context
        self.source_type = source_type
        self.doc_type = doc_type
        self.items = {
            group[0].reference: group
            for group in context.reading_groups(catalog.fragments)
            if any(fragment.primary_eligible for fragment in group)
        }

    @classmethod
    def of_authority(
        cls, context: RevisionAssessmentContext, authority: ExtractionAuthority, *, source_type: str, doc_type: str,
    ) -> ExtractionReading:
        """Read every ReadingGroup that holds authorized Primary, and nothing else, of the current revision."""
        authorized = tuple(
            replace(fragment, primary_eligible=fragment.primary_eligible and authority.authorizes(fragment))
            for fragment in context.full_fragments
        )
        return cls(
            _read_catalog(context, authorized, context.reading_groups(authorized)),
            context,
            source_type=source_type,
            doc_type=doc_type,
        )

    def report_skipped(self, reasons: Mapping[str, str]) -> tuple[str, ...]:
        """Report the items that cannot be read even alone, each with its reason; return their labels in reading order."""
        source_unit_id = self.context.projection.source_units[0].id
        skipped = tuple((reading_group_label(group), reasons[item_id]) for item_id, group in self.items.items() if item_id in reasons)
        for label, reason in skipped:
            logger.warning(
                "extraction_reading_group_skipped source_unit_id=%s reading_group=%s reason=%s",
                source_unit_id, label, reason,
            )
        return tuple(label for label, _reason in skipped)

    def catalog_for(self, item_ids) -> ProjectionFragmentCatalog:
        """The catalog of one request that reads these items."""
        if set(item_ids) == set(self.items):
            return self.catalog
        return _read_catalog(self.context, self.catalog.fragments, tuple(self.items[item_id] for item_id in item_ids))

    def request(self, item_ids, *, output_tokens, fits) -> LlmRequest:
        """The one request that reads these items, with the Artifact images its catalog cites."""
        return self.request_for(self.catalog_for(item_ids), output_tokens=output_tokens, fits=fits)

    def request_for(self, catalog: ProjectionFragmentCatalog, *, output_tokens, fits) -> LlmRequest:
        """The request that reads one catalog of these items."""
        request = LlmRequest(
            MemoryExtractor.projection_fragment_prompt(
                catalog, source_type=self.source_type, doc_type=self.doc_type, revision_context=self.context,
            ),
            ProjectionFragmentMemoryExtractionResponse,
            output_tokens(catalog),
        )
        return self.context.attach_images(request, catalog, fits=fits)


def _read_catalog(context: RevisionAssessmentContext, fragments, groups) -> ProjectionFragmentCatalog:
    """The groups that hold Primary, with their reading context demoted to Required-only."""
    own = {
        fragment.anchor
        for group in groups
        if any(fragment.primary_eligible for fragment in group)
        for fragment in group
    }
    read = own | context.reading_context(tuple(f for f in fragments if f.anchor in own))
    return context.catalog(
        tuple(
            fragment if fragment.anchor in own else replace(fragment, primary_eligible=False)
            for fragment in fragments
            if fragment.anchor in read
        )
    )


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
    def projection_fragment_prompt(catalog, *, source_type, doc_type, revision_context) -> str:
        payload = revision_context.model_payload(catalog)
        return PROJECTION_FRAGMENT_EXTRACTION_PROMPT.format(
            source_type=source_type, doc_type=doc_type, catalog_digest=catalog.digest,
            fragment_catalog=json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        )

    async def extract_projection_fragment_memories(
        self,
        catalog: ProjectionFragmentCatalog,
        *,
        source_type: str,
        doc_type: str = "unknown",
        revision_context: RevisionAssessmentContext,
    ) -> MemoryExtractionResult:
        """Select exact current Evidence within this work's Primary authority.

        Each ReadingGroup that holds authorized Primary is one runner item, so a
        request that times out, exceeds capacity or keeps returning invalid
        output is halved and resent. A ReadingGroup that alone exceeds the
        route's capacity, or whose output alone stays invalid, is skipped with a
        diagnostic, and the other groups' Candidates are kept.
        """

        if not self.structured_llm_client:
            return MemoryExtractionResult(
                error_type="llm_client_unavailable",
                error="No LLM client configured for memory extraction",
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
        reading = ExtractionReading(catalog, revision_context, source_type=source_type, doc_type=doc_type)
        runner = LlmBatchRunner(self.structured_llm_client, model=self.model)

        # The request last rendered for each set of items is the one sent for it.
        rendered: dict[tuple[str, ...], tuple[LlmRequest, ProjectionFragmentCatalog]] = {}

        def render(item_ids, _parts) -> LlmRequest:
            request_catalog = reading.catalog_for(item_ids)
            request = reading.request_for(request_catalog, output_tokens=self.fragment_output_tokens, fits=runner.fits)
            rendered[tuple(item_ids)] = (request, request_catalog)
            return request

        started = perf_counter()
        metrics: dict[str, object] = {
            "extraction_model": self.model,
            "extraction_contract_version": PROJECTION_EXTRACTION_CONTRACT_VERSION,
            "catalog_digest": catalog.digest,
            "catalog_fragment_count": len(catalog.fragments),
            "reading_group_count": len(reading.items),
        }

        def elapsed() -> dict[str, object]:
            return {
                "structured_llm_calls": runner.stats.calls,
                "prompt_chars": runner.stats.prompt_chars,
                "structured_llm_elapsed_ms": max(0, round((perf_counter() - started) * 1000)),
            }

        try:
            outcomes = await runner.run_items(ItemTask(
                item_ids=tuple(reading.items),
                render=render,
                # Every item of a request shares that request's response.
                decode=lambda response, item_ids, _parts: ((item_id, (item_ids, response)) for item_id in item_ids),
                call=invoke,
            ))
        except Exception as error:
            logger.error("Unexpected projection Fragment extraction error: %s", error)
            return MemoryExtractionResult(
                error_type="unexpected_error", error=str(error), metadata={**metrics, **elapsed()},
            )
        # An item that cannot be read even alone is skipped; a transient failure fails the work.
        failures = {item_id: outcome for item_id, outcome in outcomes.items() if isinstance(outcome, ItemFailure)}
        failure = next((outcome for outcome in failures.values() if not outcome.unjudgeable), None)
        if failure is not None:
            error = failure.error
            validation_fields = error.validation_fields if isinstance(error, StructuredLlmError) else ()
            return MemoryExtractionResult(
                error_type="structured_llm_error",
                error=str(error),
                metadata={
                    **metrics,
                    **elapsed(),
                    "safe_error_code": failure.error_code,
                    "safe_validation_fields": [
                        {"location": location, "type": rule_type}
                        for location, rule_type in validation_fields
                    ],
                },
            )
        skipped = reading.report_skipped({
            item_id: INPUT_CAPACITY_EXCEEDED if outcome.category == "capacity_exceeded" else outcome.category
            for item_id, outcome in failures.items()
        })

        responses = dict(chunks[0] for item_id, chunks in outcomes.items() if item_id not in failures)
        memories: list[RawMemory] = []
        resolution = _SelectionResolution()
        image_count = image_bytes = 0
        for item_ids, response in responses.items():
            request, request_catalog = rendered[tuple(item_ids)]
            image_count += len(request.images)
            image_bytes += sum(len(image.body) for image in request.images)
            candidates, correction_metrics = await correct_fragment_selectors_once(
                response.memories, catalog=request_catalog, client=self.structured_llm_client,
                extraction_prompt=request.prompt, max_tokens=request.max_tokens, model=self.model,
                images=request.images, source_response=response,
            )
            resolution.add_correction(correction_metrics)
            memories.extend(resolution.resolve(
                candidates, catalog=request_catalog,
                prompt_hash=hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
                returned=len(response.memories),
            ))
        return MemoryExtractionResult(
            memories=memories,
            metadata={
                **metrics,
                **elapsed(),
                "structured_llm_calls": runner.stats.calls + resolution.correction["selector_correction_calls"],
                "extraction_request_count": len(responses),
                "skipped_reading_group_count": len(skipped),
                "image_count": image_count,
                "image_bytes": image_bytes,
                **resolution.metrics(),
            },
        )


class _SelectionResolution:
    """Resolve each extracted candidate's selectors into exact Evidence, across one operation's requests."""

    def __init__(self) -> None:
        self.returned = 0
        self.resolved = 0
        self.rejection_counts: dict[str, int] = {}
        self.normalization_count = 0
        self.normalization_fingerprints: list[str] = []
        self.correction: dict[str, int] = {
            "selector_correction_calls": 0,
            "selector_correction_candidate_count": 0,
            "selector_correction_recovered_count": 0,
        }
        self.correction_outcomes: list[str] = []

    def add_correction(self, metrics: dict) -> None:
        for key in self.correction:
            self.correction[key] += int(metrics.get(key, 0) or 0)
        self.correction_outcomes.append(str(metrics["selector_correction_outcome"]))

    def resolve(self, candidates, *, catalog, prompt_hash: str, returned: int) -> list[RawMemory]:
        self.returned += returned
        memories = []
        for candidate_index, candidate in enumerate(candidates):
            normalized_required, removed_ref_count, repair_fingerprint = normalize_fragment_selector_refs(
                candidate_index=candidate_index,
                primary_ref=candidate.primary_ref,
                required_refs=candidate.required_refs,
            )
            self.normalization_count += removed_ref_count
            if repair_fingerprint is not None:
                self.normalization_fingerprints.append(repair_fingerprint)
            candidate_hash = catalog.selection_fingerprint(
                candidate_content_hash=hashlib.sha256(candidate.content.encode("utf-8")).hexdigest(),
                primary_ref=candidate.primary_ref,
                required_refs=normalized_required,
            )
            try:
                selection = catalog.resolve_selection(
                    primary_ref=candidate.primary_ref,
                    required_refs=normalized_required,
                )
            except FragmentSelectionError as error:
                self.rejection_counts[error.code.value] = self.rejection_counts.get(error.code.value, 0) + 1
                record_quality_signal(
                    QualitySignal(
                        event_name="evidence_admission_outcome",
                        outcome="rejected",
                        reason_code=error.code.value,
                        prompt_hash=prompt_hash,
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
                    prompt_hash=prompt_hash,
                    candidate_hash=candidate_hash,
                )
            )
            self.resolved += 1
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
        return memories

    def metrics(self) -> dict[str, object]:
        return {
            **self.correction,
            "selector_correction_outcome": next(
                (outcome for outcome in self.correction_outcomes if outcome != "not_needed"), "not_needed",
            ),
            "resolved_fragment_selection_count": self.resolved,
            "rejected_fragment_selection_count": self.returned - self.resolved,
            "fragment_selection_rejection_counts": self.rejection_counts,
            **(
                {
                    "selector_normalized_candidate_count": len(self.normalization_fingerprints),
                    "selector_normalization_count": self.normalization_count,
                    "selector_normalization_fingerprints": self.normalization_fingerprints,
                }
                if self.normalization_fingerprints
                else {}
            ),
        }
