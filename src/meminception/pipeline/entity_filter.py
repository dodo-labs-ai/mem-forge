"""Entity quality gate — minimal sanity filter for LLM-extracted entities.

This runs AFTER LLM extraction and BEFORE entity resolution. It applies only
rules with near-zero false positive risk. The LLM prompt + confidence score
handles all nuanced filtering (code patterns, generic terms, domain judgment).

Design principle: The prompt teaches what good entities look like via few-shot
examples. This filter is just a sanity check — not the primary quality gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["filter_entities", "EntityFilterStats"]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_ENTITIES_PER_DOC = 15
MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 50
MIN_ENTITY_CONFIDENCE = 0.7


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class EntityFilterStats:
    total_input: int = 0
    total_output: int = 0
    removed_too_short: int = 0
    removed_too_long: int = 0
    removed_low_confidence: int = 0
    removed_cap: int = 0


# ---------------------------------------------------------------------------
# Main filter
# ---------------------------------------------------------------------------

def filter_entities(
    entities: list[dict],
    *,
    max_entities: int = MAX_ENTITIES_PER_DOC,
) -> tuple[list[dict], EntityFilterStats]:
    """Filter entities with minimal sanity rules.

    Only applies rules with near-zero false positive risk:
    - Confidence threshold (LLM's own uncertainty signal)
    - Length limits (basic sanity)
    - Hard cap per document (safety valve)

    Everything else — code patterns, generic terms, naming conventions —
    is handled by the enrichment prompt via few-shot examples + confidence.

    Returns (filtered_list, stats).
    """
    stats = EntityFilterStats(total_input=len(entities))
    result = []

    for entity in entities:
        # Support both dict and dataclass
        if isinstance(entity, dict):
            name = entity.get("name", "").strip()
        else:
            name = getattr(entity, "canonical_name", getattr(entity, "name", "")).strip()

        if not name:
            continue

        # --- Confidence check ---
        if isinstance(entity, dict):
            confidence = entity.get("confidence")
            if confidence is not None and float(confidence) < MIN_ENTITY_CONFIDENCE:
                stats.removed_low_confidence += 1
                continue

        # --- Length checks ---
        if len(name) < MIN_NAME_LENGTH:
            stats.removed_too_short += 1
            continue
        if len(name) > MAX_NAME_LENGTH:
            stats.removed_too_long += 1
            continue

        result.append(entity)

    # --- Hard cap ---
    if len(result) > max_entities:
        stats.removed_cap = len(result) - max_entities
        result = result[:max_entities]

    stats.total_output = len(result)
    if stats.total_input != stats.total_output:
        logger.info(
            "Entity filter: %d → %d (-%d confidence, -%d length, -%d cap)",
            stats.total_input, stats.total_output,
            stats.removed_low_confidence,
            stats.removed_too_short + stats.removed_too_long,
            stats.removed_cap,
        )
    return result, stats
