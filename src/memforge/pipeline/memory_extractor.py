"""Single semantic extraction of claim-sized Memory candidates."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Awaitable, Callable
from time import perf_counter
from typing import Any
from memforge.config import DEFAULT_MEMORY_EXTRACTION_MAX_TOKENS
from memforge.llm.structured import (
    LiteLlmStructuredClient,
    StructuredLlmConfig,
    StructuredLlmError,
    StructuredLlmImage,
)
from memforge.models import MemoryExtractionResult, RawMemory
from memforge.pipeline.document_units import ExtractionContext
from memforge.pipeline.document_update import DEFAULT_MAX_DIFF_CHARS
from memforge.pipeline.evidence_catalog import (
    EvidenceAuthoritySpan,
    EvidenceCatalog,
    EvidenceResolution,
)
from memforge.pipeline.extraction_contract import DURABLE_MEMORY_QUALITY_RULES
from memforge.pipeline.projection_context import ProjectionExtractionBatch
from memforge.source_artifacts import (
    MAX_SOURCE_ARTIFACT_SUMMARY_CHARS,
    SourceArtifactSummary,
)

logger = logging.getLogger(__name__)

__all__ = ["MemoryExtractor"]

_EVIDENCE_BLOCK_FALLBACK_SAMPLE_LIMIT = 16

# ---------------------------------------------------------------------------
# Caps and bands shared by the extraction prompts and runtime truncation. Both
# the prompt prose and the .format() arguments reference these constants so the
# LLM sees the same limits the code enforces.
# ---------------------------------------------------------------------------

EXTRACTION_QUOTE_MAX_CHARS = 4_000
DOC_CONTENT_CHAR_CAP = 100_000
CHANGED_HUNK_CHAR_CAP = DEFAULT_MAX_DIFF_CHARS
UPDATED_DOC_CHAR_CAP = 100_000
DOCUMENT_OUTLINE_CHAR_CAP = 8_000
GLOSSARY_APPENDIX_CHAR_CAP = 2_000
UNIT_MARKDOWN_CHAR_CAP = 80_000
PROJECTION_CONTEXT_CHAR_CAP = 20_000

# ---------------------------------------------------------------------------
# Memory extraction prompt (Call 2)
# ---------------------------------------------------------------------------

MEMORY_EXTRACTION_PROMPT = """You are extracting atomic knowledge from a document for a team memory system.

<source_type>{source_type}</source_type>
<doc_type>{doc_type}</doc_type>
Only the following Evidence Blocks may support a Memory:
<evidence_catalog>
{evidence_catalog}
</evidence_catalog>

Extract durable atomic knowledge units justified by the document. Returning an empty "memories" array IS the correct answer when the document contains no durable team memory; do not invent memories to fill output. Each memory must be a JSON object with:
- "content": self-contained factual sentence (understandable without the source document)
- "memory_type": one of "fact", "decision", "convention", "procedure"
- "confidence": 0.0-1.0 (use high confidence only when the source directly states durable domain knowledge)
- "entity_refs": list of key entity mentions copied from the owned evidence
- "valid_from": YYYY-MM-DD calendar date if time-bound, null otherwise
- "valid_until": YYYY-MM-DD calendar date if time-bound, null otherwise
- "evidence_block_id": exactly one Evidence Block ID copied from <evidence_catalog>
- "evidence_quote": optional claim-local text copied from that block (guidance maximum {quote_max} chars). It helps produce a narrower excerpt but is not required.

""" + DURABLE_MEMORY_QUALITY_RULES + """Standard rules:
- Each memory must be SELF-CONTAINED (understandable without the source document).
- Extraction does not decide novelty against historical Memory rows. Emit each
  durable claim once with exact current evidence; lifecycle reconciliation owns
  identity, support, and replacement decisions after extraction.
- Prefer specifics ("PostgreSQL 15" not "a database").
- For tickets: extract the decision/outcome, not the discussion.
- For runbooks: each distinct step is a separate procedural memory.
- For design docs: extract decisions, dependencies, constraints.
- For agent_session sources: keep only durable, reusable project knowledge from the submitted summary — confirmed decisions, conventions, procedures, and architectural rules that stay true beyond this session AND are not visible by reading the current code. Record the durable OUTCOME and the WHY of a change as a single fact; do NOT emit before/after/verified play-by-play, prior or superseded code states, or step-by-step narration of one edit. Do NOT create memories about the memory system, the agent's own tooling or context injection, or session mechanics, and never include internal memory ids (for example "memories are loaded at SessionStart" or "mem-1a2b3c"). Skip one-off run output and smoke-test/verification results (for example "the command printed 6"); a passing check is evidence, not durable knowledge unless it states a lasting behavior. Skip receipt/session metadata, validation commands, runtime notes, service start/stop state, local paths, and working-tree state. When the project being worked on IS a memory or tooling system, treat its symbol names, ID strings, and column names as code-recoverable per rule 1; only emit memories that state a rule about how the system must behave (e.g., "push-based source types must not be user-configurable in the dialog") rather than what the code currently does.
- For discussions: extract DECISIONS and CONVENTIONS that reached consensus — skip unresolved opinions, tentative suggestions, and questions without answers.
- For chat sources: skip transient status updates, review-in-progress notes, and temporary caveats. Focus on decisions, persistent facts, action items, and repeatable procedures.
- Do not extract document metadata as memories: author names, last modified dates, document status, revision-history rows, reviewer lists, and link list rows belong to provenance/source metadata.
- Do not infer relationships from reference/link-only evidence. If a source only provides a link or label, skip it or preserve the weaker relationship exactly as stated.
- Preserve conditional language. If the source says "if", "provided", "as long as", "would", or "should", keep that condition in the memory. Do not turn open questions into decisions.
- Do NOT extract: formatting details, boilerplate, table-of-contents entries.
- Do NOT extract: passwords, credentials, tokens, API keys, or any secret/authentication information.

Return ONLY a JSON object with a "memories" array. Use {{"memories": []}} when there are no memories."""


MEMORY_CHANGE_EXTRACTION_PROMPT = """You are extracting memory changes from an updated document for a team memory system.

<source_type>{source_type}</source_type>
<doc_type>{doc_type}</doc_type>
<changed_hunks>
{changed_hunks}
</changed_hunks>
Only the following blocks from inserted or replaced current text may support a Memory:
<changed_evidence_catalog>
{changed_evidence_catalog}
</changed_evidence_catalog>
<updated_document>
{updated_document}
</updated_document>

The changed hunks show what changed between the previous and updated normalized document.
Use the full updated document only as context and for validating exact quotes.

Returning an empty "memories" array IS the correct answer when the changes do not introduce, refine, replace, or remove durable team knowledge; do not invent memories to fill output.
For changed durable knowledge, return JSON objects with:
- "content": self-contained factual sentence (understandable without the source document)
- "memory_type": one of "fact", "decision", "convention", "procedure"
- "confidence": 0.0-1.0
- "entity_refs": list of key entity mentions copied from the changed evidence
- "valid_from": YYYY-MM-DD calendar date if time-bound, null otherwise
- "valid_until": YYYY-MM-DD calendar date if time-bound, null otherwise
- "evidence_block_id": exactly one Evidence Block ID copied from <changed_evidence_catalog>
- "evidence_quote": optional claim-local text copied from that block (guidance maximum {quote_max} chars). It helps produce a narrower excerpt but is not required.

""" + DURABLE_MEMORY_QUALITY_RULES + """Standard rules:
- Focus ONLY on durable memory changes caused by <changed_hunks>.
- Use <updated_document> only to understand context and copy exact quotes; do not extract unaffected facts elsewhere in it.
- Extract the current durable claims changed by <changed_hunks>; lifecycle
  reconciliation, not this prompt, compares them with existing Memory rows.
- If <changed_hunks> only removes old durable knowledge without stating replacement current knowledge, return an empty "memories" array; reconciliation will decide whether to retire the old memory.
- Do not create memories about the edit itself, such as "was removed", "no longer mentioned", "the document changed", or "previously".
- For agent_session sources: keep only durable, reusable project knowledge (confirmed decisions, conventions, procedures, architectural rules) that is NOT visible by reading the current code. Record a change's durable outcome and the WHY as a single fact, not before/after/verified play-by-play. Do not create memories about the memory system, the agent's tooling or context injection, or session mechanics, and never include internal memory ids. Skip one-off run output and smoke-test results, receipt/session metadata, runtime notes, local paths, and working-tree state.
- Treat normalized source headers and platform/provenance fields as operational metadata: workflow status, assignee/owner routing, sprint/milestone, rank/order, labels/tags, timestamps, participants, reactions, edit time, author/reviewer rows, revision history, link-list rows, and formatting.
- Return an empty "memories" array for operational metadata-only changes unless the changed text explicitly states durable team knowledge, such as a decision, constraint, convention, procedure, product behavior, architectural fact, or long-lived ownership/responsibility rule.
- Preserve conditional language. Do not turn open questions, suggestions, or unresolved discussion into decisions.
- Do NOT extract table-of-contents entries, boilerplate, passwords, credentials, tokens, or API keys.

Return ONLY a JSON object with a "memories" array. Use {{"memories": []}} when there are no memory changes."""


UNIT_MEMORY_EXTRACTION_PROMPT = """You are extracting atomic knowledge from one deterministic document unit.

<source_type>{source_type}</source_type>
<doc_type>{doc_type}</doc_type>
<document_title>{document_title}</document_title>
<document_url>{document_url}</document_url>
<heading_path>{heading_path}</heading_path>
The following context is read-only. Use it only to resolve scope, acronyms, and references.
Do not extract facts that appear only in this context.
<document_outline>
{document_outline}
</document_outline>
<glossary_appendix>
{glossary_appendix}
</glossary_appendix>

Extract memories only from the Evidence Blocks derived from this owned unit:
<evidence_catalog>
{evidence_catalog}
</evidence_catalog>

Each memory must be a JSON object with:
- "content": self-contained factual sentence
- "memory_type": one of "fact", "decision", "convention", "procedure"
- "confidence": 0.0-1.0
- "entity_refs": list of key entity mentions copied from <evidence_catalog>
- "valid_from": YYYY-MM-DD calendar date if time-bound, null otherwise
- "valid_until": YYYY-MM-DD calendar date if time-bound, null otherwise
- "evidence_block_id": exactly one Evidence Block ID copied from <evidence_catalog>
- "evidence_quote": optional claim-local text copied from that block (guidance maximum {quote_max} chars). It helps produce a narrower excerpt but is not required.

""" + DURABLE_MEMORY_QUALITY_RULES + """Standard rules:
- Extract only durable team knowledge grounded in <evidence_catalog>.
- Extraction emits each current durable claim once with exact evidence.
  Reconciliation owns historical identity and support decisions.
- Do not extract document outline, glossary, title, URL, or source metadata as memories.
- For agent_session sources, extract only durable project decisions, conventions, procedures, and architectural rules that are NOT visible by reading the current code. Skip receipt/session metadata, validation commands/results, runtime notes, service start/stop state, local paths, working-tree state, and facts about the agent session itself.
- Do not extract passwords, credentials, tokens, API keys, or secrets.
- Preserve conditional language.

Return ONLY a JSON object with a "memories" array. Use {{"memories": []}} when there are no memories."""


PROJECTION_BATCH_EXTRACTION_PROMPT = """You are extracting durable atomic knowledge from changed source observations.

<source_type>{source_type}</source_type>
<doc_type>{doc_type}</doc_type>
Only PRIMARY observations grant extraction authority:
<primary_observations>
{primary_evidence_catalog}
</primary_observations>

The following observations are CONTEXT only. Use them to resolve references and chronology, but do not extract a claim stated only here:
<context_observations>
{context_observations}
</context_observations>

""" + DURABLE_MEMORY_QUALITY_RULES + """Return durable, self-contained facts, decisions, conventions, or procedures grounded in PRIMARY Evidence Blocks. Each textual item must include exactly one `evidence_block_id` copied from <primary_observations>. It may also include an optional claim-local `evidence_quote` copied from that block; the quote improves excerpt precision but never replaces the Block ID. The selected block determines `source_observation_id`; do not invent or repeat that identity for textual evidence. Never use a CONTEXT observation as primary Evidence. If the claim would become invalid or ambiguous without specific CONTEXT observations, include their exact Observation IDs in `required_source_observation_ids`; otherwise return an empty list. Do not mark merely helpful reading context as required.

For a PRIMARY `binary_artifact` observation with separately supplied image
evidence, inspect the image itself. A claim grounded in that image must set
`source_observation_id` to that exact Artifact Observation, and must leave
`evidence_quote` empty because binary evidence has no
text quote. Do not infer image contents from the filename, upload event, parent
text, or metadata. Do not emit a claim for an Artifact image that was not
supplied in this request.

Also return one `artifact_summaries` item for every image supplied in this
request. Each item must copy the exact `source_observation_id` printed before
that image and provide one concise selection summary of its visible purpose or
content (maximum {artifact_summary_max} characters). Do not include customer names, case numbers,
person IDs, credentials, or other unnecessary unique identifiers. A summary is
only a hint for deciding whether to fetch an Artifact; it is not Evidence and
must not replace inspection of the Artifact. Return an empty
`artifact_summaries` array when no images were supplied.

Extraction emits each current durable claim once with exact evidence from a
PRIMARY observation. Reconciliation owns historical identity and support.

Prefer an empty memories array over weak or transient claims. Do not emit records that only say an item was created, updated, uploaded, attached, assigned, labeled, ranked, moved, reprioritized, or passed through a routine workflow status. Do not emit revision history, source metadata, routing fields, questions, or secrets. An attachment-upload event is provenance, not authority about the attachment's contents; only separately supplied attachment-content evidence may support a claim. Preserve durable resolution rationale, settled outcomes, and conditions.

Return ONLY a JSON object with a "memories" array."""


# ---------------------------------------------------------------------------
# MemoryExtractor class
# ---------------------------------------------------------------------------


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

    async def extract_memories(
        self,
        content: str,
        source_type: str = "unknown",
        doc_type: str = "unknown",
    ) -> MemoryExtractionResult:
        """Extract atomic memories from current document evidence."""
        if not self.structured_llm_client:
            logger.warning("No LLM client — skipping memory extraction")
            return MemoryExtractionResult(
                error_type="llm_client_unavailable",
                error="No LLM client configured for memory extraction",
            )

        catalog = EvidenceCatalog.from_text(content[:DOC_CONTENT_CHAR_CAP])
        prompt = MEMORY_EXTRACTION_PROMPT.format(
            source_type=source_type,
            doc_type=doc_type,
            evidence_catalog=catalog.render(),
            quote_max=EXTRACTION_QUOTE_MAX_CHARS,
        )

        result = await self._extract_with_schema(
            prompt,
            label="memory extraction",
            invoke=self.structured_llm_client.extract_memories,
        )
        return self._resolve_text_result(result, catalog=catalog, evidence_anchor="document")

    async def extract_memory_changes(
        self,
        *,
        changed_hunks: str,
        updated_document: str,
        current_changed_ranges: tuple[tuple[int, int], ...] = (),
        source_type: str = "unknown",
        doc_type: str = "unknown",
    ) -> MemoryExtractionResult:
        """Extract only durable memory changes from a document update."""
        if not self.structured_llm_client:
            logger.warning("No LLM client — skipping memory change extraction")
            return MemoryExtractionResult(
                error_type="llm_client_unavailable",
                error="No LLM client configured for memory change extraction",
            )

        # The document prefix is bounded read-only context. Selectable changed
        # Evidence must still cover authoritative ranges anywhere in the full
        # current revision.
        context_document = updated_document[:UPDATED_DOC_CHAR_CAP]
        spans = tuple(
            EvidenceAuthoritySpan(
                text=updated_document[start:end],
                source_start=start,
            )
            for start, end in current_changed_ranges
            if 0 <= start < end <= len(updated_document)
        )
        catalog = EvidenceCatalog.from_spans(spans)
        prompt = MEMORY_CHANGE_EXTRACTION_PROMPT.format(
            source_type=source_type,
            doc_type=doc_type,
            changed_hunks=changed_hunks[:CHANGED_HUNK_CHAR_CAP],
            changed_evidence_catalog=catalog.render(),
            updated_document=context_document,
            quote_max=EXTRACTION_QUOTE_MAX_CHARS,
        )

        result = await self._extract_with_schema(
            prompt,
            label="memory change extraction",
            invoke=self.structured_llm_client.extract_memories,
        )
        return self._resolve_text_result(result, catalog=catalog, evidence_anchor="changed_range")

    @staticmethod
    def _resolve_text_result(
        result: MemoryExtractionResult,
        *,
        catalog: EvidenceCatalog,
        evidence_anchor: str,
    ) -> MemoryExtractionResult:
        if result.error_type:
            return result
        localized_memories = []
        refinement_counts: dict[str, int] = {}
        fallback_samples: list[dict[str, object]] = []
        fallback_sample_truncated_count = 0
        for memory in result.memories:
            resolved = catalog.resolve(memory)
            if resolved is None:
                continue
            resolved.memory.evidence_anchor = evidence_anchor
            if evidence_anchor == "unit":
                # Unit coordinates are not Observation-revision coordinates.
                resolved.memory.evidence_range_start = None
                resolved.memory.evidence_range_end = None
            localized_memories.append(resolved.memory)
            refinement_counts[resolved.refinement] = refinement_counts.get(resolved.refinement, 0) + 1
            if resolved.refinement == "block_fallback":
                if len(fallback_samples) < _EVIDENCE_BLOCK_FALLBACK_SAMPLE_LIMIT:
                    fallback_samples.append(
                        _block_fallback_sample(
                            original=memory,
                            resolved=resolved,
                            extraction_metadata=result.metadata,
                        )
                    )
                else:
                    fallback_sample_truncated_count += 1
        return MemoryExtractionResult(
            memories=localized_memories,
            artifact_summaries=result.artifact_summaries,
            metadata={
                **result.metadata,
                "evidence_refinement_counts": refinement_counts,
                "evidence_block_fallback_samples": fallback_samples,
                "evidence_block_fallback_sample_truncated_count": (
                    fallback_sample_truncated_count
                ),
                "invalid_evidence_block_count": len(result.memories) - len(localized_memories),
            },
        )

    async def extract_unit_memories(
        self,
        context: ExtractionContext,
        *,
        doc_type: str = "unknown",
    ) -> MemoryExtractionResult:
        """Extract memories from one deterministic unit and enforce unit evidence."""
        if not self.structured_llm_client:
            logger.warning("No LLM client — skipping unit memory extraction")
            return MemoryExtractionResult(
                error_type="llm_client_unavailable",
                error="No LLM client configured for memory extraction",
            )

        catalog = EvidenceCatalog.from_text(
            context.unit.unit_markdown[:UNIT_MARKDOWN_CHAR_CAP]
        )
        prompt = UNIT_MEMORY_EXTRACTION_PROMPT.format(
            source_type=context.source_type,
            doc_type=doc_type,
            document_title=context.document_title,
            document_url=context.document_url,
            heading_path=" > ".join(context.unit.heading_path),
            document_outline=context.document_outline[:DOCUMENT_OUTLINE_CHAR_CAP],
            glossary_appendix=context.glossary_appendix[:GLOSSARY_APPENDIX_CHAR_CAP],
            evidence_catalog=catalog.render(),
            quote_max=EXTRACTION_QUOTE_MAX_CHARS,
        )
        result = await self._extract_with_schema(
            prompt,
            label="unit memory extraction",
            invoke=self.structured_llm_client.extract_memories,
        )
        if result.error_type:
            return result

        # Compatibility for in-flight v5 responses. New v6 prompts do not ask
        # the model to emit an evidence_anchor.
        result.memories = [
            memory
            for memory in result.memories
            if memory.evidence_block_id
            or memory.evidence_anchor in {None, "unit"}
        ]

        return self._resolve_text_result(result, catalog=catalog, evidence_anchor="unit")

    async def extract_projection_batch_memories(
        self,
        batch: ProjectionExtractionBatch,
        *,
        source_type: str,
        doc_type: str = "unknown",
        images: tuple[StructuredLlmImage, ...] = (),
    ) -> MemoryExtractionResult:
        """Extract from Primary observations while treating neighbors as context."""

        if not self.structured_llm_client:
            return MemoryExtractionResult(
                error_type="llm_client_unavailable",
                error="No LLM client configured for memory extraction",
            )
        catalog = EvidenceCatalog.from_spans(
            tuple(
                EvidenceAuthoritySpan(
                    text=content,
                    observation_id=observation_id,
                    source_start=source_start,
                )
                for observation_id, source_start, content in (
                    batch.primary_authority_spans
                    or tuple(
                        (observation_id, 0, content)
                        for observation_id, content in batch.primary_content_by_observation_id
                    )
                )
                if content
            )
        )
        prompt = PROJECTION_BATCH_EXTRACTION_PROMPT.format(
            source_type=source_type,
            doc_type=doc_type,
            primary_evidence_catalog=catalog.render(),
            context_observations=batch.context_markdown[:PROJECTION_CONTEXT_CHAR_CAP],
            artifact_summary_max=MAX_SOURCE_ARTIFACT_SUMMARY_CHARS,
        )
        result = await self._extract_with_schema(
            prompt,
            label="projection batch extraction",
            invoke=self.structured_llm_client.extract_projection_memories,
            images=images,
        )
        if result.error_type:
            return result
        kept = []
        refinement_counts: dict[str, int] = {}
        fallback_samples: list[dict[str, object]] = []
        fallback_sample_truncated_count = 0
        context_by_primary = dict(batch.context_observation_ids_by_primary)
        visual_observation_ids = {
            image.source_observation_id
            for image in images
            if image.source_observation_id in batch.primary_observation_ids
        }
        for memory in result.memories:
            explicit_observation_id = memory.source_observation_id
            if explicit_observation_id in visual_observation_ids:
                quote = memory.evidence_quote or ""
                if quote.strip():
                    continue
                memory.evidence_quote = None
                memory.evidence_anchor = "source_artifact"
                memory.extraction_context = None
                required_ids = tuple(dict.fromkeys(memory.required_source_observation_ids))
                allowed_context_ids = context_by_primary.get(explicit_observation_id, ())
                if (
                    explicit_observation_id in required_ids
                    or any(item not in allowed_context_ids for item in required_ids)
                ):
                    continue
                memory.required_source_observation_ids = list(required_ids)
                kept.append(memory)
                continue
            resolved = catalog.resolve(memory)
            if resolved is None or resolved.block.observation_id is None:
                continue
            localized_memory = resolved.memory
            source_observation_id = resolved.block.observation_id
            localized_memory.evidence_anchor = "projection_batch"
            localized_memory.source_observation_id = source_observation_id
            required_ids = tuple(
                dict.fromkeys(localized_memory.required_source_observation_ids)
            )
            allowed_context_ids = context_by_primary.get(source_observation_id, ())
            if (
                source_observation_id in required_ids
                or any(item not in allowed_context_ids for item in required_ids)
            ):
                continue
            localized_memory.required_source_observation_ids = list(required_ids)
            refinement_counts[resolved.refinement] = (
                refinement_counts.get(resolved.refinement, 0) + 1
            )
            if resolved.refinement == "block_fallback":
                if len(fallback_samples) < _EVIDENCE_BLOCK_FALLBACK_SAMPLE_LIMIT:
                    fallback_samples.append(
                        _block_fallback_sample(
                            original=memory,
                            resolved=resolved,
                            extraction_metadata=result.metadata,
                        )
                    )
                else:
                    fallback_sample_truncated_count += 1
            kept.append(localized_memory)
        return MemoryExtractionResult(
            memories=kept,
            artifact_summaries=result.artifact_summaries,
            metadata={
                **result.metadata,
                "evidence_refinement_counts": refinement_counts,
                "evidence_block_fallback_samples": fallback_samples,
                "evidence_block_fallback_sample_truncated_count": (
                    fallback_sample_truncated_count
                ),
                "invalid_evidence_block_count": (
                    len(result.memories) - len(kept)
                ),
            },
        )

    async def _extract_with_schema(
        self,
        prompt: str,
        *,
        label: str,
        invoke: Callable[..., Awaitable[Any]],
        images: tuple[StructuredLlmImage, ...] = (),
    ) -> MemoryExtractionResult:
        started = perf_counter()
        metrics = {
            "structured_llm_calls": 1,
            "extraction_model": self.model,
            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "prompt_chars": len(prompt),
            "image_count": len(images),
            "image_bytes": sum(len(image.body) for image in images),
        }
        try:
            call_kwargs = {
                "max_tokens": self.max_tokens,
                "model": self.model,
            }
            if images:
                call_kwargs["images"] = images
            response = await invoke(prompt, **call_kwargs)
            supplied_image_ids = tuple(image.source_observation_id for image in images)
            discarded_orphan_summary_count = 0
            discarded_invalid_summary_count = 0
            if not supplied_image_ids:
                discarded_orphan_summary_count = len(
                    response.artifact_summaries
                )
                returned_summaries = ()
            else:
                supplied_image_id_set = set(supplied_image_ids)
                valid_summaries: dict[str, SourceArtifactSummary] = {}
                for item in response.artifact_summaries:
                    observation_id = item.source_observation_id
                    summary_text = item.summary
                    if (
                        not isinstance(observation_id, str)
                        or not isinstance(summary_text, str)
                        or not observation_id
                        or not summary_text
                        or len(summary_text)
                        > MAX_SOURCE_ARTIFACT_SUMMARY_CHARS
                        or observation_id not in supplied_image_id_set
                        or observation_id in valid_summaries
                    ):
                        discarded_invalid_summary_count += 1
                        continue
                    valid_summaries[observation_id] = (
                        SourceArtifactSummary(
                            source_observation_id=observation_id,
                            summary=summary_text,
                        )
                    )
                returned_summaries = tuple(valid_summaries.values())
            memories = [
                RawMemory(
                    content=memory.content,
                    memory_type=memory.memory_type,
                    confidence=memory.confidence,
                    entity_refs=memory.entity_refs,
                    valid_from=memory.valid_from,
                    valid_until=memory.valid_until,
                    extraction_context=memory.extraction_context,
                    evidence_quote=memory.evidence_quote,
                    evidence_block_id=memory.evidence_block_id,
                    evidence_anchor=memory.evidence_anchor,
                    source_observation_id=memory.source_observation_id,
                    required_source_observation_ids=list(memory.required_source_observation_ids),
                )
                for memory in response.memories
            ]
            logger.info("Extracted %d memories from document", len(memories))
            return MemoryExtractionResult(
                memories=memories,
                artifact_summaries=returned_summaries,
                metadata={
                    **metrics,
                    "artifact_summary_count": len(returned_summaries),
                    "discarded_orphan_artifact_summary_count": (
                        discarded_orphan_summary_count
                    ),
                    "discarded_invalid_artifact_summary_count": (
                        discarded_invalid_summary_count
                    ),
                    "structured_llm_elapsed_ms": max(
                        0, round((perf_counter() - started) * 1000)
                    ),
                },
            )
        except StructuredLlmError as e:
            logger.warning("Structured %s failed: %s", label, e)
            return MemoryExtractionResult(
                error_type="structured_llm_error",
                error=str(e),
                metadata={
                    **metrics,
                    "safe_error_code": e.error_code,
                    "safe_validation_fields": [
                        {"location": location, "type": rule_type}
                        for location, rule_type in e.validation_fields
                    ],
                    "structured_llm_elapsed_ms": max(
                        0, round((perf_counter() - started) * 1000)
                    ),
                },
            )
        except Exception as e:
            logger.error("Unexpected %s error: %s", label, e)
            return MemoryExtractionResult(
                error_type="unexpected_error",
                error=str(e),
                metadata={
                    **metrics,
                    "structured_llm_elapsed_ms": max(
                        0, round((perf_counter() - started) * 1000)
                    ),
                },
            )


def _block_fallback_sample(
    *,
    original: RawMemory,
    resolved: EvidenceResolution,
    extraction_metadata: dict[str, object],
) -> dict[str, object]:
    """Return content-free evidence telemetry for one whole-Block fallback."""

    submitted_quote = original.evidence_quote or ""
    return {
        "candidate_content_sha256": hashlib.sha256(
            original.content.encode("utf-8")
        ).hexdigest(),
        "source_observation_id": resolved.block.observation_id,
        "evidence_range_start": resolved.memory.evidence_range_start,
        "evidence_range_end": resolved.memory.evidence_range_end,
        "block_text_sha256": hashlib.sha256(
            resolved.block.text.encode("utf-8")
        ).hexdigest(),
        "block_chars": len(resolved.block.text),
        "submitted_quote_sha256": (
            hashlib.sha256(submitted_quote.encode("utf-8")).hexdigest()
            if submitted_quote
            else None
        ),
        "submitted_quote_chars": len(submitted_quote),
        "extraction_model": extraction_metadata.get("extraction_model"),
        "prompt_sha256": extraction_metadata.get("prompt_sha256"),
    }
