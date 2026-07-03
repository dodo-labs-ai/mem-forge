"""Query analysis data returned by the search engine."""

from __future__ import annotations

from dataclasses import dataclass, field

from memforge.storage.adapters.protocols import EntityLinkCandidate

__all__ = ["QueryAnalysis"]


@dataclass
class QueryAnalysis:
    """Search routing and deterministic entity-linking diagnostics."""

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
