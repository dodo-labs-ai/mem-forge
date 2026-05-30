"""Query analysis: temporal detection + entity detection.

Two detections — no 7-type classifier. The agent explicitly passes memory_types
and entities via the MCP tool schema; the analyzer detects temporal intent and
entity mentions from the query text.

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
from datetime import datetime, timedelta, timezone
from typing import Any

from meminception.llm.structured import StructuredLlmError
from meminception.models import canonicalize_entity_name

logger = logging.getLogger(__name__)

__all__ = ["QueryAnalysis", "analyze_query"]


@dataclass
class QueryAnalysis:
    """Result of rule-based query analysis."""

    # Temporal detection
    is_temporal: bool = False
    temporal_start: datetime | None = None
    temporal_end: datetime | None = None

    # Entity detection
    detected_entities: list[str] = field(default_factory=list)
    detected_entity_ids: list[int] = field(default_factory=list)

    # Strategies to activate
    use_vector: bool = True      # always on
    use_bm25: bool = True        # always on
    use_graph: bool = False      # only when entities detected
    use_temporal: bool = False   # only when temporal intent detected


# ---------------------------------------------------------------------------
# Temporal patterns
# ---------------------------------------------------------------------------

_TEMPORAL_KEYWORDS = [
    "recently", "recent", "latest", "newest", "new",
    "changed", "updated", "modified",
    "today", "yesterday",
    "this week", "last week", "past week",
    "this month", "last month", "past month",
    "this quarter", "last quarter",
]

_TEMPORAL_REGEX = re.compile(
    r"(?:last|past)\s+(\d+)\s+(day|week|month|hour)s?",
    re.IGNORECASE,
)

_SINCE_REGEX = re.compile(
    r"since\s+(\d{4}-\d{2}-\d{2})",
    re.IGNORECASE,
)


def _detect_temporal(query: str) -> tuple[bool, datetime | None, datetime | None]:
    """Detect temporal intent and resolve to a date range.

    Returns (is_temporal, start_date, end_date).
    """
    q = query.lower().strip()
    now = datetime.now(timezone.utc)

    # Check "since YYYY-MM-DD"
    since_match = _SINCE_REGEX.search(q)
    if since_match:
        try:
            start = datetime.fromisoformat(since_match.group(1)).replace(tzinfo=timezone.utc)
            return True, start, None
        except ValueError:
            pass

    # Check "last/past N days/weeks/months"
    range_match = _TEMPORAL_REGEX.search(q)
    if range_match:
        amount = int(range_match.group(1))
        unit = range_match.group(2).lower()
        if unit == "hour":
            delta = timedelta(hours=amount)
        elif unit == "day":
            delta = timedelta(days=amount)
        elif unit == "week":
            delta = timedelta(weeks=amount)
        elif unit == "month":
            delta = timedelta(days=amount * 30)
        else:
            delta = timedelta(days=amount)
        return True, now - delta, None

    # Check keyword triggers
    for keyword in _TEMPORAL_KEYWORDS:
        if keyword in q:
            # Default temporal range: 7 days for "recently", "changed", etc.
            if keyword in ("today",):
                return True, now.replace(hour=0, minute=0, second=0, microsecond=0), None
            elif keyword in ("yesterday",):
                yesterday = now - timedelta(days=1)
                return True, yesterday.replace(hour=0, minute=0, second=0, microsecond=0), now.replace(hour=0, minute=0, second=0, microsecond=0)
            elif keyword in ("this week",):
                start = now - timedelta(days=now.weekday())
                return True, start.replace(hour=0, minute=0, second=0, microsecond=0), None
            elif keyword in ("last week", "past week"):
                start = now - timedelta(days=now.weekday() + 7)
                end = now - timedelta(days=now.weekday())
                return True, start.replace(hour=0, minute=0, second=0, microsecond=0), end.replace(hour=0, minute=0, second=0, microsecond=0)
            elif keyword in ("this month",):
                return True, now.replace(day=1, hour=0, minute=0, second=0, microsecond=0), None
            elif keyword in ("last month", "past month"):
                first_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                prev_month = first_of_month - timedelta(days=1)
                return True, prev_month.replace(day=1), first_of_month
            else:
                # Generic "recently" / "changed" / "updated" → last 7 days
                return True, now - timedelta(days=7), None

    return False, None, None


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

    Two detections:
    1. Temporal intent → adds SQL date filter
    2. Entity mentions → activates graph traversal (regex first, LLM fallback)

    All queries run vector + BM25 in parallel. Graph traversal and temporal
    filtering are additive when detected.

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

    # Detection 1: Temporal intent
    is_temporal, start, end = _detect_temporal(query)
    if is_temporal:
        analysis.is_temporal = True
        analysis.temporal_start = start
        analysis.temporal_end = end
        analysis.use_temporal = True

    # Detection 2: Entity mentions — regex first, LLM fallback
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
