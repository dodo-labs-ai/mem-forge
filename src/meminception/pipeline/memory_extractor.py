"""Call 2: Memory Extraction.

Extracts atomic knowledge units (memories) from a document.
Receives Call 1 output (entities, doc_type) + existing memories as context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from meminception.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig, StructuredLlmError
from meminception.models import MemoryExtractionResult, RawMemory
from meminception.pipeline.document_units import ExtractionContext

if TYPE_CHECKING:
    from meminception.models import Memory

logger = logging.getLogger(__name__)

__all__ = ["MemoryExtractor"]

# ---------------------------------------------------------------------------
# Memory extraction prompt (Call 2)
# ---------------------------------------------------------------------------

MEMORY_EXTRACTION_PROMPT = """You are extracting atomic knowledge from a document for a team memory system.

<source_type>{source_type}</source_type>
<doc_type>{doc_type}</doc_type>
<entities_found>{entities_found}</entities_found>
<existing_memories_for_these_entities>
{existing_memories}
</existing_memories_for_these_entities>
<document>
{content}
</document>

Extract all durable atomic knowledge units justified by the document. Return an empty "memories" array if the document contains no durable team memory. Each memory must be a JSON object with:
- "content": self-contained factual sentence (understandable without the source document)
- "memory_type": one of "fact", "decision", "convention", "procedure"
- "confidence": 0.0-1.0 (use high confidence only when the source directly states durable domain knowledge)
- "entity_refs": list of key entity names (use the canonical names from <entities_found>)
- "tags": 2-5 lowercase topic tags
- "valid_from": ISO date if time-bound, null otherwise
- "valid_until": ISO date if time-bound, null otherwise
- "extraction_context": exact quote from the document this was extracted from (max 200 chars). For chat/message sources, include the sender name and timestamp prefix (e.g. "**Alice** (10:05): the actual message content")

Rules:
- Each memory must be SELF-CONTAINED (understandable without the source document)
- Do NOT re-extract facts listed in <existing_memories_for_these_entities>
- Focus on NEW or UPDATED information
- Use entity names from <entities_found>, not your own variations
- Prefer specifics ("PostgreSQL 15" not "a database")
- Emit each durable claim once, in a single canonical phrasing. Do not output multiple reworded variants of the same fact, decision, or procedure
- For tickets: extract the decision/outcome, not the discussion
- For runbooks: each distinct step is a separate procedural memory
- For design docs: extract decisions, dependencies, constraints
- For agent_session sources: keep only durable, reusable project knowledge from the submitted summary (confirmed decisions, conventions, procedures, and verified implementation facts that stay true beyond this session). Record the durable OUTCOME of a change as a single fact. Do NOT emit before/after/verified play-by-play, prior or superseded code states, or step-by-step narration of one edit. Do NOT create memories about the memory system, the agent's own tooling or context injection, or session mechanics, and never include internal memory ids (for example "memories are loaded at SessionStart" or "mem-1a2b3c"). Skip one-off run output and smoke-test/verification results (for example "the command printed 6"); a passing check is evidence, not durable knowledge unless it states a lasting behavior. Skip receipt/session metadata, validation commands, runtime notes, service start/stop state, local paths, and working-tree state
- For discussions: extract DECISIONS and CONVENTIONS that reached consensus — skip unresolved opinions, tentative suggestions, and questions without answers
- For chat sources: skip transient status updates, review-in-progress notes, and temporary caveats. Focus on decisions, persistent facts, and action items
- Do not extract document metadata as memories: author names, last modified dates, document status, revision-history rows, reviewer lists, and link list rows belong to provenance/source metadata
- Do not infer relationships from reference/link-only evidence. If a source only provides a link or label, skip it or preserve the weaker relationship exactly as stated
- Preserve conditional language. If the source says "if", "provided", "as long as", "would", or "should", keep that condition in the memory. Do not turn open questions into decisions
- Do NOT extract: formatting details, boilerplate, table-of-contents entries
- Do NOT extract: passwords, credentials, tokens, API keys, or any secret/authentication information

Return ONLY a JSON object with a "memories" array. Use {{"memories": []}} when there are no memories."""


MEMORY_CHANGE_EXTRACTION_PROMPT = """You are extracting memory changes from an updated document for a team memory system.

<source_type>{source_type}</source_type>
<doc_type>{doc_type}</doc_type>
<entities_found>{entities_found}</entities_found>
<existing_memories_for_this_document>
{existing_memories}
</existing_memories_for_this_document>
<changed_hunks>
{changed_hunks}
</changed_hunks>
<updated_document>
{updated_document}
</updated_document>

The changed hunks show what changed between the previous and updated normalized document.
Use the full updated document only as context and for validating exact quotes.

Return an empty "memories" array if the changes do not introduce, refine, replace, or remove durable team knowledge.
For changed durable knowledge, return JSON objects with:
- "content": self-contained factual sentence (understandable without the source document)
- "memory_type": one of "fact", "decision", "convention", "procedure"
- "confidence": 0.0-1.0
- "entity_refs": list of key entity names (use the canonical names from <entities_found>)
- "tags": 2-5 lowercase topic tags
- "valid_from": ISO date if time-bound, null otherwise
- "valid_until": ISO date if time-bound, null otherwise
- "extraction_context": exact quote from the updated document this was extracted from (max 200 chars). For chat/message sources, include the sender name and timestamp prefix from the updated document.

Rules:
- Focus ONLY on durable memory changes caused by <changed_hunks>
- Use <updated_document> only to understand context and copy exact quotes; do not extract unaffected facts elsewhere in it
- Do NOT re-extract facts already covered by <existing_memories_for_this_document> unless <changed_hunks> materially changes the current durable claim
- If <changed_hunks> only removes old durable knowledge without stating replacement current knowledge, return an empty "memories" array; reconciliation will decide whether to retire the old memory
- Do not create memories about the edit itself, such as "was removed", "no longer mentioned", "the document changed", or "previously"
- For agent_session sources: keep only durable, reusable project knowledge (confirmed decisions, conventions, procedures, verified implementation facts). Record a change's durable outcome as a single fact, not before/after/verified play-by-play. Do not create memories about the memory system, the agent's tooling or context injection, or session mechanics, and never include internal memory ids. Skip one-off run output and smoke-test results, receipt/session metadata, runtime notes, local paths, and working-tree state
- Emit each durable claim once, in a single canonical phrasing; do not output reworded duplicates of the same fact, decision, or procedure
- Treat normalized source headers and platform/provenance fields as operational metadata: workflow status, assignee/owner routing, sprint/milestone, rank/order, labels/tags, timestamps, participants, reactions, edit time, author/reviewer rows, revision history, link-list rows, and formatting
- Return an empty "memories" array for operational metadata-only changes unless the changed text explicitly states durable team knowledge, such as a decision, constraint, convention, procedure, product behavior, architectural fact, or long-lived ownership/responsibility rule
- Preserve conditional language. Do not turn open questions, suggestions, or unresolved discussion into decisions
- Do NOT extract table-of-contents entries, boilerplate, passwords, credentials, tokens, or API keys

Return ONLY a JSON object with a "memories" array. Use {{"memories": []}} when there are no memory changes."""


UNIT_MEMORY_EXTRACTION_PROMPT = """You are extracting atomic knowledge from one deterministic document unit.

<source_type>{source_type}</source_type>
<doc_type>{doc_type}</doc_type>
<document_title>{document_title}</document_title>
<document_url>{document_url}</document_url>
<heading_path>{heading_path}</heading_path>
<entities_found>{entities_found}</entities_found>
<existing_memories_for_these_entities>
{existing_memories}
</existing_memories_for_these_entities>

The following context is read-only. Use it only to resolve scope, acronyms, and references.
Do not extract facts that appear only in this context.
<document_outline>
{document_outline}
</document_outline>
<glossary_appendix>
{glossary_appendix}
</glossary_appendix>

Extract memories only from this owned unit:
<unit_markdown>
{unit_markdown}
</unit_markdown>

Each memory must be a JSON object with:
- "content": self-contained factual sentence
- "memory_type": one of "fact", "decision", "convention", "procedure"
- "confidence": 0.0-1.0
- "entity_refs": list of key entity names from <entities_found>
- "tags": 2-5 lowercase topic tags
- "valid_from": ISO date if time-bound, null otherwise
- "valid_until": ISO date if time-bound, null otherwise
- "extraction_context": exact quote from <unit_markdown> (max 200 chars)
- "evidence_quote": exact quote copied from <unit_markdown>
- "evidence_anchor": "unit"

Rules:
- Extract only durable team knowledge grounded in <unit_markdown>
- Do not extract document outline, glossary, title, URL, or source metadata as memories
- For agent_session sources, extract only durable project decisions, conventions, procedures, and verified implementation facts from the submitted summary. Skip receipt/session metadata, validation commands/results, runtime notes, service start/stop state, local paths, working-tree state, and facts about the agent session itself
- Do not extract passwords, credentials, tokens, API keys, or secrets
- Preserve conditional language

Return ONLY a JSON object with a "memories" array. Use {{"memories": []}} when there are no memories."""


# ---------------------------------------------------------------------------
# MemoryExtractor class
# ---------------------------------------------------------------------------


class MemoryExtractor:
    """Memory extraction via LLM (Call 2 of the two-call extraction pipeline).

    Receives enrichment output (entities, doc_type) and existing memories as context,
    so it can focus on NEW or UPDATED information and use canonical entity names.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-20250514",
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 64000,
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
        entities: list[str] | None = None,
        existing_memories: list[Memory] | None = None,
    ) -> MemoryExtractionResult:
        """Run Call 2: extract atomic memories from document content.

        Args:
            content: Normalized markdown content of the document.
            source_type: Type of the source gene (confluence, jira, etc.)
            doc_type: Document type from Call 1 (design-doc, runbook, etc.)
            entities: Canonical entity names found by Call 1.
            existing_memories: Memories already in DB for these entities
                (so the LLM can skip re-extracting known facts).

        Returns:
            MemoryExtractionResult with list of RawMemory candidates.
        """
        if not self.structured_llm_client:
            logger.warning("No LLM client — skipping memory extraction")
            return MemoryExtractionResult(
                error_type="llm_client_unavailable",
                error="No LLM client configured for memory extraction",
            )

        # Format entities list
        entities_str = ", ".join(entities) if entities else "(none found)"

        # Format existing memories
        if existing_memories:
            existing_str = "\n".join(f"- [{m.memory_type}] {m.content}" for m in existing_memories[:30])
        else:
            existing_str = "(no existing memories for these entities)"

        prompt = MEMORY_EXTRACTION_PROMPT.format(
            source_type=source_type,
            doc_type=doc_type,
            entities_found=entities_str,
            existing_memories=existing_str,
            content=content[:100_000],
        )

        return await self._extract_with_schema(prompt, label="memory extraction")

    async def extract_memory_changes(
        self,
        *,
        changed_hunks: str,
        updated_document: str,
        source_type: str = "unknown",
        doc_type: str = "unknown",
        entities: list[str] | None = None,
        existing_memories: list[Memory] | None = None,
    ) -> MemoryExtractionResult:
        """Extract only durable memory changes from a document update."""
        if not self.structured_llm_client:
            logger.warning("No LLM client — skipping memory change extraction")
            return MemoryExtractionResult(
                error_type="llm_client_unavailable",
                error="No LLM client configured for memory change extraction",
            )

        entities_str = ", ".join(entities) if entities else "(none found)"
        if existing_memories:
            existing_str = "\n".join(f"- [{m.id}] [{m.memory_type}] {m.content}" for m in existing_memories[:50])
        else:
            existing_str = "(no existing memories for this document)"

        prompt = MEMORY_CHANGE_EXTRACTION_PROMPT.format(
            source_type=source_type,
            doc_type=doc_type,
            entities_found=entities_str,
            existing_memories=existing_str,
            changed_hunks=changed_hunks[:40_000],
            updated_document=updated_document[:100_000],
        )

        return await self._extract_with_schema(prompt, label="memory change extraction")

    async def extract_unit_memories(
        self,
        context: ExtractionContext,
        *,
        doc_type: str = "unknown",
        existing_memories: list[Memory] | None = None,
    ) -> MemoryExtractionResult:
        """Extract memories from one deterministic unit and enforce unit evidence."""
        if not self.structured_llm_client:
            logger.warning("No LLM client — skipping unit memory extraction")
            return MemoryExtractionResult(
                error_type="llm_client_unavailable",
                error="No LLM client configured for memory extraction",
            )

        entities_str = ", ".join(context.entities) if context.entities else "(none found)"
        if existing_memories:
            existing_str = "\n".join(f"- [{m.memory_type}] {m.content}" for m in existing_memories[:30])
        else:
            existing_str = "(no existing memories for these entities)"

        prompt = UNIT_MEMORY_EXTRACTION_PROMPT.format(
            source_type=context.source_type,
            doc_type=doc_type,
            document_title=context.document_title,
            document_url=context.document_url,
            heading_path=" > ".join(context.unit.heading_path),
            entities_found=entities_str,
            existing_memories=existing_str,
            document_outline=context.document_outline[:8_000],
            glossary_appendix=context.glossary_appendix[:2_000],
            unit_markdown=context.unit.unit_markdown[:80_000],
        )
        result = await self._extract_with_schema(prompt, label="unit memory extraction")
        if result.error_type:
            return result

        kept: list[RawMemory] = []
        for memory in result.memories:
            if memory.evidence_anchor != "unit":
                continue
            evidence_quote = memory.evidence_quote or memory.extraction_context or ""
            memory.evidence_quote = evidence_quote
            memory.evidence_anchor = "unit"
            memory.extraction_context = evidence_quote[:200]
            kept.append(memory)
        return MemoryExtractionResult(memories=kept)

    async def _extract_with_schema(self, prompt: str, *, label: str) -> MemoryExtractionResult:
        try:
            response = await self.structured_llm_client.extract_memories(
                prompt,
                max_tokens=self.max_tokens,
                model=self.model,
            )
            memories = [
                RawMemory(
                    content=memory.content,
                    memory_type=memory.memory_type,
                    confidence=memory.confidence,
                    entity_refs=memory.entity_refs,
                    tags=memory.tags,
                    valid_from=memory.valid_from,
                    valid_until=memory.valid_until,
                    extraction_context=memory.extraction_context,
                    evidence_quote=memory.evidence_quote,
                    evidence_anchor=memory.evidence_anchor,
                )
                for memory in response.memories
            ]
            logger.info("Extracted %d memories from document", len(memories))
            return MemoryExtractionResult(memories=memories)
        except StructuredLlmError as e:
            logger.warning("Structured %s failed: %s", label, e)
            return MemoryExtractionResult(error_type="structured_llm_error", error=str(e))
        except Exception as e:
            logger.error("Unexpected %s error: %s", label, e)
            return MemoryExtractionResult(error_type="unexpected_error", error=str(e))
