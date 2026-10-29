"""Single semantic extraction of claim-sized Memory candidates."""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import replace
from time import perf_counter

from memforge.evals.agent_evaluation import QualitySignal, record_quality_signal
from memforge.llm.batch_runner import ItemFailure, ItemTask, LlmBatchRunner, LlmRequest
from memforge.llm.structured import (
    LiteLlmStructuredClient,
    ProjectionFragmentMemoryExtractionResponse,
    StructuredLlmConfig,
    StructuredLlmError,
)
from memforge.models import MemoryExtractionResult, RawMemory
from memforge.pipeline.extraction_contract import (
    DURABLE_MEMORY_QUALITY_RULES,
    EVIDENCE_DISPLAY_RULES,
    MEMORY_CLAIM_EVIDENCE_RULES,
    PROJECTION_EXTRACTION_CONTRACT_VERSION,
)

from memforge.pipeline.projection_fragments import (
    FragmentSelectionError,
    FragmentSelectionErrorCode,
    ProjectionFragmentCatalog,
)
from memforge.pipeline.projection_context import ExtractionAuthority
from memforge.pipeline.revision_assessment import RevisionAssessmentContext, reading_group_label
from memforge.pipeline.unit_title import unit_title_block

logger = logging.getLogger(__name__)

__all__ = ["ExtractionReading", "MemoryExtractor"]


PROJECTION_FRAGMENT_EXTRACTION_PROMPT = """You are extracting durable atomic knowledge from one authorized Source Unit catalog.

<source_type>{source_type}</source_type>
<doc_type>{doc_type}</doc_type>
{unit_title}Catalog rows are [ref, faithful source view, optional metadata]. Headings are ordinary selectable Fragments.
Preserve table column/row associations, list order, code indentation and explicit exceptions.
Each ref covers its supplied selection, which may be one row or a larger structure.
Read the supplied labels and structural context before interpreting a value;
do not assume that one ref covers the entire document or table.
A figure preserves its image link and caption together. Only a supplied image Artifact
ref proves image contents; a caption or URL alone never proves unseen image details.
Use the adapter's format_interpretation to read source-specific field meanings.
Structural groups describe ancestry, not additional Evidence. Authored scope must
remain in the claim; select its ref only when it adds relevant evidence.
Only the following application-owned Evidence Fragments may support a Memory:
<evidence_fragment_catalog digest="{catalog_digest}">
{fragment_catalog}
</evidence_fragment_catalog>

Use the supplied response schema. Within each Memory choose primary_ref, write
final content and metadata, then select required_refs in the same response.

Do not return Observation or Revision IDs, offsets, hashes, profile names, catalog digests, Context refs, or lifecycle actions. Split a candidate that would otherwise need multiple independently claim-bearing Primary refs. One ref that states several independently changeable assertions also yields several Memories; they may share that primary_ref.

""" + MEMORY_CLAIM_EVIDENCE_RULES + DURABLE_MEMORY_QUALITY_RULES + EVIDENCE_DISPLAY_RULES + """`required_only_candidates` may be selected as Required but never as Primary. Do not originate a Memory whose central assertion is stated only by required_only_candidates; a primary_candidates reply that agrees with, answers or decides on such content states its own stance and is that Memory's Primary. Preserve other useful claims with eligible Primary evidence. Fragment refs are valid only in this catalog. Never invent or transform a ref.

Return ONLY a JSON object with a "memories" array. Use {{"memories": []}} when there are no memories."""

# The one correction of a response whose selections this catalog cannot bind;
# ``{error}`` names each rejected selection and the refs it may use instead.
_SELECTION_CORRECTION_BLOCK = (
    "\n\n<correction>\n"
    "The previous response selected Evidence this catalog cannot bind:\n"
    "{error}\n"
    "Return the complete response again, with every Memory, including those that were valid. "
    "Choose each primary_ref from primary_candidates; a required_only_candidates ref may be "
    "selected only in required_refs. When a new reply agrees with, answers or decides on earlier "
    "content, select the reply as Primary and the earlier content as Required. Omit a Memory "
    "whose central assertion has no eligible Primary. Use only the refs supplied above.\n"
    "</correction>"
)


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

    def report_unread(self, reasons: Mapping[str, str]) -> tuple[str, ...]:
        """Report the items that cannot be read even alone, each with its reason; return their labels in reading order."""
        source_unit_id = self.context.projection.source_units[0].id
        unread = tuple((reading_group_label(group), reasons[item_id]) for item_id, group in self.items.items() if item_id in reasons)
        for label, reason in unread:
            logger.warning(
                "extraction_reading_group_unread source_unit_id=%s reading_group=%s reason=%s",
                source_unit_id, label, reason,
            )
        return tuple(label for label, _reason in unread)

    def catalog_for(self, item_ids) -> ProjectionFragmentCatalog:
        """The catalog of one request that reads these items."""
        if set(item_ids) == set(self.items):
            return self.catalog
        return _read_catalog(self.context, self.catalog.fragments, tuple(self.items[item_id] for item_id in item_ids))

    def request(self, item_ids, *, fits) -> LlmRequest:
        """The one request that reads these items, with the Artifact images its catalog cites."""
        return self.request_for(self.catalog_for(item_ids), fits=fits)

    def request_for(self, catalog: ProjectionFragmentCatalog, *, fits) -> LlmRequest:
        """The request that reads one catalog of these items."""
        request = LlmRequest(
            MemoryExtractor.projection_fragment_prompt(
                catalog, source_type=self.source_type, doc_type=self.doc_type, revision_context=self.context,
            ),
            ProjectionFragmentMemoryExtractionResponse,
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

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        base_url: str | None = None,
        api_key: str | None = None,
        request_timeout_s: float = 300.0,
        structured_llm_client=None,
    ) -> None:
        self.model = model
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
            unit_title=unit_title_block(revision_context.projection.unit_title),
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
        output is halved and resent. A response binds whole: one candidate whose
        selection the request's catalog cannot bind rejects it, and the runner
        corrects it once, naming each rejected selection and the refs it may
        use. A ReadingGroup that alone exceeds the route's capacity, or whose
        output alone stays invalid or unbound, is diagnosed and skipped. An
        execution error fails the work without returning a successful subset.
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
                error_type="projection_extraction_unavailable",
                error=f"Structured client does not implement {PROJECTION_EXTRACTION_CONTRACT_VERSION}",
            )
        reading = ExtractionReading(catalog, revision_context, source_type=source_type, doc_type=doc_type)
        runner = LlmBatchRunner(self.structured_llm_client, model=self.model)

        # The request last rendered for each set of items is the one sent for it.
        rendered: dict[tuple[str, ...], tuple[LlmRequest, ProjectionFragmentCatalog]] = {}

        def render(item_ids, _parts) -> LlmRequest:
            request_catalog = reading.catalog_for(item_ids)
            request = reading.request_for(request_catalog, fits=runner.fits)
            rendered[tuple(item_ids)] = (request, request_catalog)
            return request

        selections = _SelectionResolution()

        def decode(response, item_ids, _parts):
            request, request_catalog = rendered[tuple(item_ids)]
            memories = selections.bind(
                response.memories, catalog=request_catalog, request_key=tuple(item_ids),
                prompt_hash=hashlib.sha256(request.prompt.encode("utf-8")).hexdigest(),
            )
            # Every item of a request shares that request's bound Memories.
            return tuple((item_id, (tuple(item_ids), (request, memories))) for item_id in item_ids)

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
                decode=decode,
                call=invoke,
                correction=_SELECTION_CORRECTION_BLOCK,
            ))
        except Exception as error:
            logger.error("Unexpected projection Fragment extraction error: %s", error)
            return MemoryExtractionResult(
                error_type="unexpected_error", error=str(error), metadata={**metrics, **elapsed()},
            )
        if set(outcomes) != set(reading.items):
            return MemoryExtractionResult(
                error_type="projection_extraction_incomplete",
                error="The authorized ReadingGroups were not completely accounted for.",
                metadata={**metrics, **elapsed(), "safe_error_code": "EXTRACTION_WORK_INCOMPLETE"},
            )
        failures = {item_id: outcome for item_id, outcome in outcomes.items() if isinstance(outcome, ItemFailure)}
        failure = next((outcome for outcome in failures.values() if not outcome.unjudgeable), None)
        if failure is not None:
            error = failure.error
            validation_fields = error.validation_fields if isinstance(error, StructuredLlmError) else ()
            return MemoryExtractionResult(
                error_type="structured_llm_error",
                error=str(error) if error is not None else "An authorized ReadingGroup could not be processed.",
                metadata={
                    **metrics,
                    **elapsed(),
                    "safe_error_code": failure.error_code,
                    "failed_reading_group_count": len(failures),
                    "safe_validation_fields": [
                        {"location": location, "type": rule_type}
                        for location, rule_type in validation_fields
                    ],
                },
            )
        skipped = reading.report_unread({item_id: outcome.error_code for item_id, outcome in failures.items()})
        unresolved = {item_id for item_id, outcome in failures.items() if isinstance(outcome.error, UnresolvedSelection)}
        selections.signal_unresolved(unresolved)
        bound = dict(chunks[0] for chunks in outcomes.values() if not isinstance(chunks, ItemFailure))
        memories = [memory for _request, request_memories in bound.values() for memory in request_memories]
        requests = [request for request, _memories in bound.values()]
        return MemoryExtractionResult(memories=memories, metadata={
            **metrics,
            **elapsed(),
            "extraction_request_count": len(bound),
            "skipped_reading_group_count": len(skipped),
            "image_count": sum(len(request.images) for request in requests),
            "image_bytes": sum(len(image.body) for request in requests for image in request.images),
            **selections.metrics(skipped=len(unresolved)),
        })


class UnresolvedSelection(ValueError):
    """A response with candidates whose selections the request's catalog cannot bind.

    Its message names each rejected selection by its response location, with
    the refs that location may use, for the response's one correction.
    """

    def __init__(self, rejections: tuple[tuple[int, FragmentSelectionError], ...]) -> None:
        super().__init__("\n".join(_rejection_line(index, error) for index, error in rejections))
        self.rejections = rejections


def _rejection_line(index: int, error: FragmentSelectionError) -> str:
    location = f"memories[{index}]" + (f".{error.location}" if error.location else "")
    line = f"- {location}: {error}"
    if error.allowed_refs is not None:
        role = "Primary refs" if error.code is FragmentSelectionErrorCode.INELIGIBLE_ROLE else "refs"
        line += f". Allowed {role}: {', '.join(error.allowed_refs)}"
    return line


class _SelectionResolution:
    """Bind each response's candidates to exact Evidence, across one operation's requests.

    A response binds whole or not at all. Every rejected candidate is counted when
    it is read, in every response including corrections and halved requests. Bound
    candidates are signalled once their response is accepted; a rejection is
    signalled only when its ReadingGroup is skipped, because a rejection the
    correction or a halved request recovers loses nothing.
    """

    def __init__(self) -> None:
        self.resolved = 0
        self.rejection_counts: dict[str, int] = {}
        self.corrected_requests = 0
        # The rejections of each request's latest response that did not bind.
        self._unresolved: dict[tuple[str, ...], list[QualitySignal]] = {}
        self._rejected_requests: set[tuple[str, ...]] = set()

    def bind(self, candidates, *, catalog, prompt_hash: str, request_key: tuple[str, ...]) -> list[RawMemory]:
        """The response's Memories; raises ``UnresolvedSelection`` when any candidate cannot be bound."""

        memories: list[RawMemory] = []
        accepted: list[QualitySignal] = []
        rejected: list[QualitySignal] = []
        rejections: list[tuple[int, FragmentSelectionError]] = []
        for candidate_index, candidate in enumerate(candidates):
            candidate_hash = catalog.selection_fingerprint(
                candidate_content_hash=hashlib.sha256(candidate.content.encode("utf-8")).hexdigest(),
                primary_ref=candidate.primary_ref,
                required_refs=candidate.required_refs,
            )
            try:
                selection = catalog.resolve_selection(
                    primary_ref=candidate.primary_ref,
                    required_refs=candidate.required_refs,
                    display_text_by_ref={display.ref: display.text for display in candidate.evidence_displays},
                )
            except FragmentSelectionError as error:
                rejections.append((candidate_index, error))
                self.rejection_counts[error.code.value] = self.rejection_counts.get(error.code.value, 0) + 1
                rejected.append(
                    QualitySignal(
                        event_name="evidence_admission_outcome",
                        outcome="rejected",
                        reason_code=error.code.value,
                        prompt_hash=prompt_hash,
                        candidate_hash=candidate_hash,
                    )
                )
                continue
            accepted.append(
                QualitySignal(
                    event_name="evidence_admission_outcome",
                    outcome="expected",
                    reason_code="fragment_selection_resolved",
                    prompt_hash=prompt_hash,
                    candidate_hash=candidate_hash,
                )
            )
            memories.append(
                RawMemory(
                    content=candidate.content,
                    memory_type=candidate.memory_type,
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
        if rejections:
            self._rejected_requests.add(request_key)
            self._unresolved[request_key] = rejected
            raise UnresolvedSelection(tuple(rejections))
        self._unresolved.pop(request_key, None)
        for signal in accepted:
            record_quality_signal(signal)
        self.resolved += len(memories)
        if request_key in self._rejected_requests:
            self.corrected_requests += 1
        return memories

    def signal_unresolved(self, skipped: set[str]) -> None:
        """Signal the final rejections of each skipped ReadingGroup.

        A ReadingGroup is skipped only after the request that holds it alone
        stays unbound, so its rejections are that request's latest response's.
        """

        for request_key, signals in self._unresolved.items():
            if len(request_key) == 1 and request_key[0] in skipped:
                for signal in signals:
                    record_quality_signal(signal)

    def metrics(self, *, skipped: int) -> dict[str, object]:
        """Per-reason rejections, and how the rejected requests ended: corrected, or skipped ReadingGroups."""

        return {
            "resolved_fragment_selection_count": self.resolved,
            "rejected_fragment_selection_count": sum(self.rejection_counts.values()),
            "fragment_selection_rejection_counts": dict(sorted(self.rejection_counts.items())),
            "selection_corrected_request_count": self.corrected_requests,
            "selection_skipped_reading_group_count": skipped,
        }
