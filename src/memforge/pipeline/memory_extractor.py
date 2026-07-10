"""Call 2: Memory Extraction.

Extracts atomic knowledge units (memories) from a document.
Receives Call 1 output (entities, doc_type) + existing memories as context.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from memforge.config import DEFAULT_ENRICHMENT_MAX_TOKENS
from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig, StructuredLlmError
from memforge.models import MemoryExtractionResult, RawMemory
from memforge.pipeline.document_units import ExtractionContext

if TYPE_CHECKING:
    from memforge.models import Memory

logger = logging.getLogger(__name__)

__all__ = ["MemoryExtractor"]

# ---------------------------------------------------------------------------
# Caps and bands shared by the extraction prompts and runtime truncation. Both
# the prompt prose and the .format() arguments reference these constants so the
# LLM sees the same limits the code enforces.
# ---------------------------------------------------------------------------

EXTRACTION_QUOTE_MAX_CHARS = 200
EXTRACTION_TAG_MIN = 2
EXTRACTION_TAG_MAX = 5
DOC_CONTENT_CHAR_CAP = 100_000
CHANGED_HUNK_CHAR_CAP = 40_000
UPDATED_DOC_CHAR_CAP = 100_000
EXISTING_MEMORIES_WINDOW = 30
EXISTING_MEMORIES_WINDOW_CHANGE = 50
DOCUMENT_OUTLINE_CHAR_CAP = 8_000
GLOSSARY_APPENDIX_CHAR_CAP = 2_000
UNIT_MARKDOWN_CHAR_CAP = 80_000

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

Extract durable atomic knowledge units justified by the document. Returning an empty "memories" array IS the correct answer when the document contains no durable team memory; do not invent memories to fill output. Each memory must be a JSON object with:
- "content": self-contained factual sentence (understandable without the source document)
- "memory_type": one of "fact", "decision", "convention", "procedure"
- "confidence": 0.0-1.0 (use high confidence only when the source directly states durable domain knowledge)
- "entity_refs": list of key entity names (use the canonical names from <entities_found>)
- "tags": {tag_min}-{tag_max} lowercase topic tags
- "valid_from": YYYY-MM-DD calendar date if time-bound, null otherwise
- "valid_until": YYYY-MM-DD calendar date if time-bound, null otherwise
- "extraction_context": exact quote from the document this was extracted from (max {quote_max} chars). For chat/message sources, include the sender name and timestamp prefix (e.g. "**Alice** (10:05): the actual message content")

Top rules (apply these first; reject candidates that fail any of them):

0. PREFER EMPTY. Returning {{"memories": []}} is the default. The bar for emitting a memory is high: it must teach a future developer something they would otherwise miss, six months from now, after the code has been refactored. Sessions full of routine work, mechanical edits, debugging detours, version-control bookkeeping, conversational exchanges, and meta-discussion of the work itself produce ZERO memories — that is the expected outcome, not a failure.

1. CODE-RECOVERABLE FACTS ARE NOT MEMORIES. Reject any candidate a developer could verify by reading the current code, schema, types, configuration, or running `grep` / `git log -p` in under a minute. Specifically, do not emit memories that restate function or method names, class names, type signatures, prop names, parameter lists, ID or constant string values, file paths, schema column names, migration numbers, framework configuration values, or "X passes Y to Z" wiring sentences. Keep a candidate only when it states a constraint, reason, rule, or invariant that survives a future refactor and is NOT visible in any single file.

2. ONE CLAIM, ONE MEMORY. If the document restates the same underlying claim more than once (e.g., "X must be populated for icons" and "without X, icons fall back to dots"), pick the single most general phrasing and emit one memory. Do not emit reworded duplicates of the same fact, decision, or procedure.

3. FOLD REJECTED ALTERNATIVES INTO THE CHOSEN DECISION. When a discussion records that path A was picked over B and C, emit ONE decision memory of the form "picked A over B and C because <reason>". Do not emit B and C as their own "rejected" memories.

4. FUTURE USEFULNESS CHECK. Before emitting any memory, ask: "Will a developer six months from now act better because this memory exists, after the code has been refactored?" If the answer is no — for example, the claim is true only because of how the code is currently written, or the claim self-resolves within days (a not-yet-validated risk, a temporary caveat) — skip it.

5. NO META-MEMORIES. Do NOT emit memories about the act of working: how a commit was structured, how a diff was split, that a procedure was followed, that a tool was used, that a test failure was pre-existing, that work was in progress at session end, that a rule from a project guidance file was respected. The session log, git history, and the guidance file itself already record all of this. Memory is about the project's domain, not the meta-process of editing it.

Standard rules:
- Each memory must be SELF-CONTAINED (understandable without the source document).
- Do NOT re-extract facts already listed in <existing_memories_for_these_entities>; focus on NEW or UPDATED information.
- Use entity names from <entities_found>, not your own variations.
- Prefer specifics ("PostgreSQL 15" not "a database").
- For tickets: extract the decision/outcome, not the discussion.
- For runbooks: each distinct step is a separate procedural memory.
- For design docs: extract decisions, dependencies, constraints.
- For agent_session sources: keep only durable, reusable project knowledge from the submitted summary — confirmed decisions, conventions, procedures, and architectural rules that stay true beyond this session AND are not visible by reading the current code. Record the durable OUTCOME and the WHY of a change as a single fact; do NOT emit before/after/verified play-by-play, prior or superseded code states, or step-by-step narration of one edit. Do NOT create memories about the memory system, the agent's own tooling or context injection, or session mechanics, and never include internal memory ids (for example "memories are loaded at SessionStart" or "mem-1a2b3c"). Skip one-off run output and smoke-test/verification results (for example "the command printed 6"); a passing check is evidence, not durable knowledge unless it states a lasting behavior. Skip receipt/session metadata, validation commands, runtime notes, service start/stop state, local paths, and working-tree state. When the project being worked on IS a memory or tooling system, treat its symbol names, ID strings, and column names as code-recoverable per rule 1; only emit memories that state a rule about how the system must behave (e.g., "push-based source types must not be user-configurable in the dialog") rather than what the code currently does.
- For discussions: extract DECISIONS and CONVENTIONS that reached consensus — skip unresolved opinions, tentative suggestions, and questions without answers.
- For chat sources: skip transient status updates, review-in-progress notes, and temporary caveats. Focus on decisions, persistent facts, and action items.
- Do not extract document metadata as memories: author names, last modified dates, document status, revision-history rows, reviewer lists, and link list rows belong to provenance/source metadata.
- Do not infer relationships from reference/link-only evidence. If a source only provides a link or label, skip it or preserve the weaker relationship exactly as stated.
- Preserve conditional language. If the source says "if", "provided", "as long as", "would", or "should", keep that condition in the memory. Do not turn open questions into decisions.
- Preserve the source language of the durable claim in memory.content. If the source evidence is primarily Chinese, write the memory in Chinese. Do not translate memories to English unless the source itself is English or mixed-language phrasing is needed for exact technical identifiers.
- Do NOT extract: formatting details, boilerplate, table-of-contents entries.
- Do NOT extract: passwords, credentials, tokens, API keys, or any secret/authentication information.

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

Returning an empty "memories" array IS the correct answer when the changes do not introduce, refine, replace, or remove durable team knowledge; do not invent memories to fill output.
For changed durable knowledge, return JSON objects with:
- "content": self-contained factual sentence (understandable without the source document)
- "memory_type": one of "fact", "decision", "convention", "procedure"
- "confidence": 0.0-1.0
- "entity_refs": list of key entity names (use the canonical names from <entities_found>)
- "tags": {tag_min}-{tag_max} lowercase topic tags
- "valid_from": YYYY-MM-DD calendar date if time-bound, null otherwise
- "valid_until": YYYY-MM-DD calendar date if time-bound, null otherwise
- "extraction_context": exact quote from the updated document this was extracted from (max {quote_max} chars). For chat/message sources, include the sender name and timestamp prefix from the updated document.

Top rules (apply these first; reject candidates that fail any of them):

0. PREFER EMPTY. Returning {{"memories": []}} is the default. The bar for emitting a memory is high. Routine refactors, formatting changes, dependency bumps, mechanical renames, and metadata-only edits produce ZERO memories — that is the expected outcome.

1. CODE-RECOVERABLE FACTS ARE NOT MEMORIES. Reject any candidate a developer could verify by reading the current code, schema, types, configuration, or running `grep` / `git log -p` in under a minute (function/class names, type signatures, prop names, ID/constant values, file paths, schema columns, migration numbers, "X passes Y to Z" wiring). Keep a candidate only when it states a constraint, reason, rule, or invariant that survives a future refactor.

2. ONE CLAIM, ONE MEMORY. Pick the single most general phrasing and emit one memory; do not output reworded duplicates of the same claim.

3. FOLD REJECTED ALTERNATIVES INTO THE CHOSEN DECISION. Emit one "picked A over B and C because <reason>" decision memory rather than separate "rejected B" / "rejected C" memories.

4. FUTURE USEFULNESS CHECK. Skip claims that will be obvious after the next refactor, or that self-resolve within days (a not-yet-validated risk, a temporary caveat).

5. NO META-MEMORIES. Do not emit memories about the editing process itself: commit structure, diff splitting, that a guidance-file rule was followed, that a test failure was pre-existing, that work was in progress. Memory is about the project's domain, not the meta-process.

Standard rules:
- Focus ONLY on durable memory changes caused by <changed_hunks>.
- Use <updated_document> only to understand context and copy exact quotes; do not extract unaffected facts elsewhere in it.
- Do NOT re-extract facts already covered by <existing_memories_for_this_document> unless <changed_hunks> materially changes the current durable claim.
- If <changed_hunks> only removes old durable knowledge without stating replacement current knowledge, return an empty "memories" array; reconciliation will decide whether to retire the old memory.
- Do not create memories about the edit itself, such as "was removed", "no longer mentioned", "the document changed", or "previously".
- For agent_session sources: keep only durable, reusable project knowledge (confirmed decisions, conventions, procedures, architectural rules) that is NOT visible by reading the current code. Record a change's durable outcome and the WHY as a single fact, not before/after/verified play-by-play. Do not create memories about the memory system, the agent's tooling or context injection, or session mechanics, and never include internal memory ids. Skip one-off run output and smoke-test results, receipt/session metadata, runtime notes, local paths, and working-tree state.
- Treat normalized source headers and platform/provenance fields as operational metadata: workflow status, assignee/owner routing, sprint/milestone, rank/order, labels/tags, timestamps, participants, reactions, edit time, author/reviewer rows, revision history, link-list rows, and formatting.
- Return an empty "memories" array for operational metadata-only changes unless the changed text explicitly states durable team knowledge, such as a decision, constraint, convention, procedure, product behavior, architectural fact, or long-lived ownership/responsibility rule.
- Preserve conditional language. Do not turn open questions, suggestions, or unresolved discussion into decisions.
- Preserve the source language of the durable claim in memory.content. If the source evidence is primarily Chinese, write the memory in Chinese. Do not translate memories to English unless the source itself is English or mixed-language phrasing is needed for exact technical identifiers.
- Do NOT extract table-of-contents entries, boilerplate, passwords, credentials, tokens, or API keys.

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
- "tags": {tag_min}-{tag_max} lowercase topic tags
- "valid_from": YYYY-MM-DD calendar date if time-bound, null otherwise
- "valid_until": YYYY-MM-DD calendar date if time-bound, null otherwise
- "extraction_context": exact quote from <unit_markdown> (max {quote_max} chars)
- "evidence_quote": exact quote copied from <unit_markdown>
- "evidence_anchor": "unit"

Top rules (apply these first; reject candidates that fail any of them):

0. PREFER EMPTY. Returning {{"memories": []}} is the default. Most units do not contain durable team knowledge worth keeping; emit zero rather than weak memories.

1. CODE-RECOVERABLE FACTS ARE NOT MEMORIES. Reject any candidate a developer could verify by reading the current code, schema, types, configuration, or running `grep` / `git log -p` in under a minute. Keep a candidate only when it states a constraint, reason, rule, or invariant that survives a future refactor.

2. ONE CLAIM, ONE MEMORY. Pick the most general phrasing for each underlying claim; do not output reworded duplicates.

3. FOLD REJECTED ALTERNATIVES INTO THE CHOSEN DECISION. Emit one "picked A over B because <reason>" decision rather than separate "rejected" memories.

4. FUTURE USEFULNESS CHECK. Skip claims that self-resolve within days or that will be obvious after the next refactor.

5. NO META-MEMORIES. Do not emit memories about the editing process itself: commit structure, diff splitting, that a guidance-file rule was followed, that a test failure was pre-existing, that work was in progress.

Standard rules:
- Extract only durable team knowledge grounded in <unit_markdown>.
- Do not extract document outline, glossary, title, URL, or source metadata as memories.
- For agent_session sources, extract only durable project decisions, conventions, procedures, and architectural rules that are NOT visible by reading the current code. Skip receipt/session metadata, validation commands/results, runtime notes, service start/stop state, local paths, working-tree state, and facts about the agent session itself.
- Do not extract passwords, credentials, tokens, API keys, or secrets.
- Preserve conditional language.
- Preserve the source language of the durable claim in memory.content. If the source evidence is primarily Chinese, write the memory in Chinese. Do not translate memories to English unless the source itself is English or mixed-language phrasing is needed for exact technical identifiers.

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
        max_tokens: int = DEFAULT_ENRICHMENT_MAX_TOKENS,
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
            existing_str = "\n".join(
                f"- [{m.memory_type}] {m.content}"
                for m in existing_memories[:EXISTING_MEMORIES_WINDOW]
            )
        else:
            existing_str = "(no existing memories for these entities)"

        prompt = MEMORY_EXTRACTION_PROMPT.format(
            source_type=source_type,
            doc_type=doc_type,
            entities_found=entities_str,
            existing_memories=existing_str,
            content=content[:DOC_CONTENT_CHAR_CAP],
            tag_min=EXTRACTION_TAG_MIN,
            tag_max=EXTRACTION_TAG_MAX,
            quote_max=EXTRACTION_QUOTE_MAX_CHARS,
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
            existing_str = "\n".join(
                f"- [{m.id}] [{m.memory_type}] {m.content}"
                for m in existing_memories[:EXISTING_MEMORIES_WINDOW_CHANGE]
            )
        else:
            existing_str = "(no existing memories for this document)"

        prompt = MEMORY_CHANGE_EXTRACTION_PROMPT.format(
            source_type=source_type,
            doc_type=doc_type,
            entities_found=entities_str,
            existing_memories=existing_str,
            changed_hunks=changed_hunks[:CHANGED_HUNK_CHAR_CAP],
            updated_document=updated_document[:UPDATED_DOC_CHAR_CAP],
            tag_min=EXTRACTION_TAG_MIN,
            tag_max=EXTRACTION_TAG_MAX,
            quote_max=EXTRACTION_QUOTE_MAX_CHARS,
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
            existing_str = "\n".join(
                f"- [{m.memory_type}] {m.content}"
                for m in existing_memories[:EXISTING_MEMORIES_WINDOW]
            )
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
            document_outline=context.document_outline[:DOCUMENT_OUTLINE_CHAR_CAP],
            glossary_appendix=context.glossary_appendix[:GLOSSARY_APPENDIX_CHAR_CAP],
            unit_markdown=context.unit.unit_markdown[:UNIT_MARKDOWN_CHAR_CAP],
            tag_min=EXTRACTION_TAG_MIN,
            tag_max=EXTRACTION_TAG_MAX,
            quote_max=EXTRACTION_QUOTE_MAX_CHARS,
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
            memory.extraction_context = evidence_quote[:EXTRACTION_QUOTE_MAX_CHARS]
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
