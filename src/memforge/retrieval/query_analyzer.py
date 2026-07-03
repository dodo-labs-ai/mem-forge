"""Query analysis: entity detection for retrieval routing.

The agent explicitly passes memory_types, entities, and date filters via the
MCP/API tool schemas. This analyzer only detects entity mentions from the query
text.

Entity detection is two-tier:
  1. Regex scan against known entity names + aliases (< 5ms, deterministic).
  2. LLM fallback via Haiku when regex finds nothing (~ 200ms, semantic).
The LLM tier catches informal names, abbreviations, and paraphrases that
the alias table hasn't seen yet.
"""

from __future__ import annotations

import asyncio
import re
import logging
from dataclasses import dataclass, field
from typing import Any

from memforge.llm.structured import StructuredLlmError
from memforge.models import canonicalize_entity_name
from memforge.storage.adapters.protocols import EntityLinkCandidate

logger = logging.getLogger(__name__)

__all__ = ["QueryAnalysis", "analyze_query"]


@dataclass
class QueryAnalysis:
    """Result of rule-based query analysis."""

    # Entity detection
    detected_entities: list[str] = field(default_factory=list)
    detected_entity_ids: list[int] = field(default_factory=list)

    # Strategies to activate
    use_vector: bool = True      # always on
    use_bm25: bool = True        # always on
    use_graph: bool = False      # only when entities detected

    # Query-time linker diagnostics. Compact-only candidates may be present
    # here without activating graph retrieval.
    entity_linking: list[EntityLinkCandidate] = field(default_factory=list)
    entity_linking_channels: tuple[str, ...] = ()
    unmatched_explicit_entities: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Entity detection — Tier 1: regex
# ---------------------------------------------------------------------------

def _detect_entities_regex(
    query: str,
    known_entities: dict[str, int],
) -> tuple[list[str], list[int]]:
    """Detect entity mentions in the query by matching against known entity names.

    The query is canonicalized (hyphens/underscores → spaces) to match the
    canonicalized entity names stored in the database.

    Args:
        query: The search query text.
        known_entities: Mapping of entity_name → entity_id (canonical names + aliases).

    Returns:
        (detected_names, detected_ids) — both in matched order, deduplicated by entity_id.
    """
    q = canonicalize_entity_name(query)
    detected_names: list[str] = []
    detected_ids: list[int] = []
    matched_ranges: list[tuple[int, int]] = []

    # Sort by length descending to match longer names first
    # ("payment gateway" before "payment")
    for name, entity_id in sorted(known_entities.items(), key=lambda x: len(x[0]), reverse=True):
        if not name:
            continue

        # Skip if this entity ID was already detected (via a longer name or alias)
        if entity_id in detected_ids:
            continue

        pattern = re.escape(name)
        match = re.search(rf"(?:^|[^a-zA-Z0-9]){pattern}(?:[^a-zA-Z0-9]|$)", q)
        if match:
            # Adjust span to exclude the boundary characters
            start = match.start()
            end = match.end()
            if q[start:start + 1] != name[0]:
                start += 1  # skip leading boundary char
            if end > 0 and q[end - 1:end] != name[-1]:
                end -= 1  # skip trailing boundary char

            overlaps = any(
                not (end <= ms or start >= me)
                for ms, me in matched_ranges
            )
            if overlaps:
                continue

            detected_names.append(name)
            detected_ids.append(entity_id)
            matched_ranges.append((start, end))

    return detected_names, detected_ids


# ---------------------------------------------------------------------------
# Entity detection — Tier 2: LLM fallback
# ---------------------------------------------------------------------------

async def _detect_entities_llm(
    query: str,
    known_entities: dict[str, int],
    model: str,
    timeout_s: float,
    structured_llm_client: Any = None,
) -> tuple[list[str], list[int]]:
    """Detect entities via LLM when regex matching finds nothing.

    Sends the full entity list + query to a fast model (Haiku) and asks
    which entities the query references. Handles semantic references,
    abbreviations, and informal names that the alias table hasn't seen.

    Returns (detected_names, detected_ids), or empty lists on failure.
    """
    if structured_llm_client is None:
        return [], []

    # Build compact entity list: group aliases under canonical names
    id_to_canonical: dict[int, str] = {}
    id_to_aliases: dict[int, list[str]] = {}
    for name, eid in known_entities.items():
        if not name:
            continue
        if eid not in id_to_canonical:
            id_to_canonical[eid] = name
            id_to_aliases[eid] = []
        elif name != id_to_canonical[eid]:
            id_to_aliases[eid].append(name)

    lines: list[str] = []
    for eid in sorted(id_to_canonical):
        canonical = id_to_canonical[eid]
        aliases = id_to_aliases[eid]
        if aliases:
            lines.append(f"{eid}:{canonical} [{','.join(aliases)}]")
        else:
            lines.append(f"{eid}:{canonical}")
    entity_list = "\n".join(lines)

    if not entity_list:
        return [], []

    prompt = (
        "You are an entity detector for a team knowledge base. "
        "Given a search query, identify which entities from the list are "
        "mentioned or referenced — directly, by abbreviation, or semantically.\n\n"
        "Rules:\n"
        "- Return ONLY entity IDs the query is actually about.\n"
        "- Match direct mentions, aliases, abbreviations, AND semantic references.\n"
        "- If no entities match, return an empty entity_ids array.\n"
        "- Return at most 5 entities. Prefer fewer, more precise matches.\n\n"
        'Respond with ONLY a JSON object like {"entity_ids": [1, 2]}. No explanation.\n\n'
        f"<entities>\n{entity_list}\n</entities>\n\n"
        f"<query>{query}</query>"
    )

    try:
        response = await asyncio.wait_for(
            structured_llm_client.detect_query_entities(
                prompt,
                max_tokens=64,
                model=model,
            ),
            timeout=timeout_s,
        )

        entity_ids = response.entity_ids

        # Validate IDs against known entities and resolve names
        valid_ids: set[int] = set(known_entities.values())
        id_to_name: dict[int, str] = {}
        for name, eid in known_entities.items():
            if eid not in id_to_name:
                id_to_name[eid] = name

        detected_names: list[str] = []
        detected_ids: list[int] = []
        for eid in entity_ids:
            if isinstance(eid, int) and eid in valid_ids and eid not in detected_ids:
                detected_names.append(id_to_name.get(eid, str(eid)))
                detected_ids.append(eid)

        if detected_ids:
            logger.info(
                "LLM entity detection found %d entities: %s",
                len(detected_ids),
                detected_names,
            )

        return detected_names, detected_ids

    except asyncio.TimeoutError:
        logger.warning("LLM entity detection timed out after %.1fs", timeout_s)
        return [], []
    except StructuredLlmError as e:
        logger.warning("Structured entity detection failed: %s", e)
        return [], []
    except Exception:
        logger.warning("LLM entity detection failed, skipping", exc_info=True)
        return [], []


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

async def analyze_query(
    query: str,
    known_entities: dict[str, int] | None = None,
    entity_model: str = "claude-haiku-4-5-20251001",
    entity_timeout_s: float = 1.0,
    structured_llm_client: Any = None,
) -> QueryAnalysis:
    """Analyze a search query to determine retrieval strategies.

    Entity mentions activate graph traversal (regex first, LLM fallback).

    All queries run vector + BM25 in parallel. Date filtering is explicit and
    handled outside this analyzer.

    Args:
        query: The search query text.
        known_entities: Mapping of canonical_name → entity_id from the database.
            If None, entity detection is skipped.
        entity_model: Model to use for LLM entity detection fallback.
        entity_timeout_s: Hard timeout for the LLM call.
        structured_llm_client: structured LiteLLM client for the LLM fallback.
            If None, the LLM tier is skipped silently.

    Returns:
        QueryAnalysis with strategy flags and detected signals.
    """
    analysis = QueryAnalysis()

    # Entity mentions — regex first, LLM fallback
    if known_entities:
        names, ids = _detect_entities_regex(query, known_entities)
        if not names:
            names, ids = await _detect_entities_llm(
                query, known_entities, entity_model, entity_timeout_s, structured_llm_client,
            )
        if names:
            analysis.detected_entities = names
            analysis.detected_entity_ids = ids
            analysis.use_graph = True

    return analysis
