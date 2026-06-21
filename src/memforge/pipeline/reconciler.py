"""Call 3: Memory Reconciliation.

Compares newly extracted memories against existing memories from the same
source document. Uses an LLM to decide: ADD, UPDATE, SUPERSEDE, DELETE, or NOOP.

Only runs on document UPDATES (when content hash changed). New documents
go through the normal deduplicate_and_insert path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from memforge.llm.structured import StructuredLlmError
from memforge.models import Memory, RawMemory, ReconcileAction, ReconcileOperation

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

__all__ = ["ReconciliationFailure", "ReconciliationResult", "reconcile_memories"]


@dataclass(frozen=True)
class ReconciliationFailure:
    """Failure metadata for a reconciliation call that produced no safe decisions."""

    error_type: str
    error: str


@dataclass(frozen=True)
class ReconciliationResult:
    """Reconciliation call result with operations and optional failure metadata."""

    operations: list[ReconcileOperation]
    failure: ReconciliationFailure | None = None


RECONCILIATION_PROMPT = """You are reconciling team knowledge. A document was updated and new facts
were extracted. Compare them against existing memories from the same document
and the updated document content.

<update_mode>{update_mode}</update_mode>
<diff_stats>
{diff_stats}
</diff_stats>
<changed_hunks>
{changed_hunks}
</changed_hunks>

For each new extraction, decide ONE action:

- ADD: Genuinely new information not covered by any existing memory.
- UPDATE: An existing memory covers the same fact but needs minor refinement.
  Provide the merged current text. The core claim remains true. The old memory
  will be preserved with a timestamp but hidden from default search.
- SUPERSEDE: An existing memory covers the same topic but is now materially wrong.
  The old fact was true before but is no longer. The new fact replaces the old meaning.
  The old memory will be preserved with a timestamp but hidden from search.
- DELETE: An existing memory is demonstrably false or was extracted in error.
- NOOP: The new extraction adds nothing beyond what existing memories capture.

Also audit existing memories from this same document against the updated
document. If an existing memory is no longer supported by the updated document
and no new extraction supersedes it, return a DELETE action with its memory_id.
If an existing memory is still supported, you may omit it.

When update_mode is diff_guided, use changed_hunks as the authority for what
changed. Use the full updated document only to validate support and understand
context. Do not update, supersede, or delete memories from other documents.
In diff_guided mode, DELETE or SUPERSEDE an existing memory only when <changed_hunks>
removes, replaces, or contradicts the supporting text, or when the updated
document clearly shows the relevant section now states a different current fact.
Do not DELETE solely because support is absent from unrelated context or because
the updated document excerpt may be incomplete.

<doc_type>{doc_type}</doc_type>

<updated_document>
{updated_document}
</updated_document>

<new_extractions>
{new_extractions}
</new_extractions>

<existing_memories>
{existing_memories}
</existing_memories>

Rules:
1. Each new extraction gets EXACTLY ONE action.
2. For UPDATE and SUPERSEDE, specify which existing memory ID is affected.
3. For UPDATE, provide the merged current text as "updated_content".
4. If uncertain between UPDATE and SUPERSEDE, prefer SUPERSEDE when the old meaning is materially wrong.
5. If an existing memory has corroboration_count >= 3 and you want to SUPERSEDE it,
   set "flag_for_review": true.
6. For UPDATE updated_content and SUPERSEDE replacement memory content, write the
   canonical current memory, not the edit history. The replacement memory content must state the current durable fact as it should appear in search results.
   Do not write replacement content as edit history such as "no longer marked",
   "was removed", "the document changed", or "previously". Put that rationale in
   the reason field instead. Only mention a historical transition in memory
   content when the updated document itself says the transition is a durable fact.

Return a JSON object with a "decisions" array:
{{
  "decisions": [
    {{"index": 0, "action": "ADD", "reason": "New fact about deployment"}},
    {{"index": 1, "action": "SUPERSEDE", "memory_id": "mem-abc123",
      "reason": "Database migrated from v14 to v16", "flag_for_review": false}},
    {{"index": 2, "action": "UPDATE", "memory_id": "mem-def456",
      "updated_content": "Service uses OAuth 2.0 with PKCE and supports Google and GitHub.",
      "reason": "Added GitHub as identity provider"}},
    {{"index": 3, "action": "NOOP", "memory_id": "mem-ghi789",
      "reason": "Already captured"}},
    {{"action": "DELETE", "memory_id": "mem-old999",
      "reason": "The updated document no longer supports this memory"}}
  ]
}}

Return ONLY the JSON object."""


async def reconcile_memories(
    new_extractions: list[RawMemory],
    existing_memories: list[Memory],
    doc_type: str,
    structured_llm_client,
    llm_model: str = "claude-sonnet-4-20250514",
    updated_document: str | None = None,
    update_mode: str = "full_document",
    changed_hunks: str | None = None,
    update_plan_stats: dict | None = None,
    include_metadata: bool = False,
) -> list[ReconcileOperation] | ReconciliationResult:
    """Call 3: LLM reconciliation of new extractions against existing memories.

    Returns a list of ReconcileOperations, one per new extraction.
    Falls back to ADD for all if the LLM call fails.
    """
    if not new_extractions and not existing_memories:
        return _return_result([], include_metadata=include_metadata)

    # If no existing memories, everything is ADD (skip LLM call)
    if not existing_memories:
        return _return_result(
            [ReconcileOperation(action=ReconcileAction.ADD, memory=raw) for raw in new_extractions],
            include_metadata=include_metadata,
        )

    # Format for the prompt
    new_json = json.dumps(
        [
            {
                "index": i,
                "content": raw.content,
                "memory_type": raw.memory_type,
                "confidence": raw.confidence,
                "entity_refs": raw.entity_refs,
            }
            for i, raw in enumerate(new_extractions)
        ],
        indent=2,
    )

    existing_json = json.dumps(
        [
            {
                "id": mem.id,
                "content": mem.content,
                "memory_type": mem.memory_type,
                "confidence": mem.confidence,
                "corroboration_count": mem.corroboration_count,
            }
            for mem in existing_memories
        ],
        indent=2,
    )

    prompt = RECONCILIATION_PROMPT.format(
        update_mode=update_mode,
        diff_stats=json.dumps(update_plan_stats or {}, indent=2),
        changed_hunks=(changed_hunks or "")[:40_000],
        doc_type=doc_type,
        updated_document=(updated_document or "")[:100_000],
        new_extractions=new_json,
        existing_memories=existing_json,
    )

    try:
        response = await structured_llm_client.reconcile_memories(
            prompt,
            max_tokens=4096,
            model=llm_model,
        )
        decisions = [decision.model_dump() for decision in response.decisions]

        return _return_result(
            _parse_decisions(decisions, new_extractions, existing_memories),
            include_metadata=include_metadata,
        )

    except (StructuredLlmError, KeyError) as e:
        logger.warning("Structured reconciliation failed: %s — skipping reconciliation mutations", e)
        operations = [] if existing_memories else _fallback_add_all(new_extractions)
        return _return_result(
            operations,
            failure=ReconciliationFailure(error_type="structured_llm_error", error=str(e)),
            include_metadata=include_metadata,
        )
    except Exception as e:
        logger.error("Reconciliation LLM call failed: %s — skipping reconciliation mutations", e)
        operations = [] if existing_memories else _fallback_add_all(new_extractions)
        return _return_result(
            operations,
            failure=ReconciliationFailure(error_type="unexpected_error", error=str(e)),
            include_metadata=include_metadata,
        )


def _return_result(
    operations: list[ReconcileOperation],
    *,
    failure: ReconciliationFailure | None = None,
    include_metadata: bool,
) -> list[ReconcileOperation] | ReconciliationResult:
    if include_metadata:
        return ReconciliationResult(operations=operations, failure=failure)
    return operations


def _parse_decisions(
    decisions: list[dict],
    new_extractions: list[RawMemory],
    existing_memories: list[Memory],
) -> list[ReconcileOperation]:
    """Parse LLM decisions into ReconcileOperations."""
    ops: list[ReconcileOperation] = []
    existing_ids = {mem.id for mem in existing_memories}

    # Track which indices we've seen
    seen_indices: set[int] = set()

    for dec in decisions:
        idx = dec.get("index", -1)
        if not isinstance(idx, int) or idx < 0 or idx >= len(new_extractions) or idx in seen_indices:
            memory_id = dec.get("memory_id")
            if memory_id not in existing_ids:
                continue
            action_str = dec.get("action", "").upper()
            try:
                action = ReconcileAction(action_str)
            except ValueError:
                continue
            if action not in (ReconcileAction.DELETE, ReconcileAction.NOOP):
                continue
            ops.append(
                ReconcileOperation(
                    action=action,
                    memory_id=memory_id,
                    reason=dec.get("reason", ""),
                    flag_for_review=bool(dec.get("flag_for_review")),
                )
            )
            continue
        seen_indices.add(idx)

        action_str = dec.get("action", "ADD").upper()
        try:
            action = ReconcileAction(action_str)
        except ValueError:
            action = ReconcileAction.ADD

        raw = new_extractions[idx]
        memory_id = dec.get("memory_id")
        reason = dec.get("reason", "")

        if action == ReconcileAction.UPDATE and memory_id and dec.get("updated_content"):
            # UPDATE: create a modified version of the raw memory with merged content
            updated_raw = RawMemory(
                content=dec["updated_content"],
                memory_type=raw.memory_type,
                confidence=raw.confidence,
                entity_refs=raw.entity_refs,
                tags=raw.tags,
                valid_from=raw.valid_from,
                valid_until=raw.valid_until,
                extraction_context=raw.extraction_context,
            )
            ops.append(
                ReconcileOperation(
                    action=action,
                    memory_id=memory_id,
                    memory=updated_raw,
                    reason=reason,
                    flag_for_review=bool(dec.get("flag_for_review")),
                )
            )
        elif action in (ReconcileAction.SUPERSEDE, ReconcileAction.DELETE) and memory_id:
            ops.append(
                ReconcileOperation(
                    action=action,
                    memory_id=memory_id,
                    memory=raw,
                    reason=reason,
                    flag_for_review=bool(dec.get("flag_for_review")),
                )
            )
        elif action == ReconcileAction.NOOP:
            ops.append(
                ReconcileOperation(
                    action=action,
                    memory_id=memory_id,
                    reason=reason,
                    flag_for_review=bool(dec.get("flag_for_review")),
                )
            )
        else:
            # ADD or fallback
            ops.append(
                ReconcileOperation(
                    action=ReconcileAction.ADD,
                    memory=raw,
                    reason=reason,
                    flag_for_review=bool(dec.get("flag_for_review")),
                )
            )

    # Any new extractions not covered by decisions → ADD
    for i, raw in enumerate(new_extractions):
        if i not in seen_indices:
            ops.append(
                ReconcileOperation(
                    action=ReconcileAction.ADD,
                    memory=raw,
                    reason="Not covered by reconciliation",
                )
            )

    return ops


def _fallback_add_all(new_extractions: list[RawMemory]) -> list[ReconcileOperation]:
    """Fallback: treat everything as ADD (same as no reconciliation)."""
    return [ReconcileOperation(action=ReconcileAction.ADD, memory=raw) for raw in new_extractions]
