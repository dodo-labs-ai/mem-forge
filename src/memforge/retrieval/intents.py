"""Language-neutral ranked retrieval intent contract and resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, cast

RankedRetrievalIntent = Literal["general_hybrid", "known_item", "relationship"]
IntentSource = Literal["requested", "deterministic", "default", "fallback"]
FallbackReason = Literal["no_traversable_entity"]

RANKED_RETRIEVAL_INTENTS: tuple[RankedRetrievalIntent, ...] = (
    "general_hybrid",
    "known_item",
    "relationship",
)

_FUSION_WEIGHTS: dict[RankedRetrievalIntent, dict[str, float]] = {
    "general_hybrid": {
        "vector": 0.45,
        "bm25_content": 0.30,
        "metadata_lexical": 0.15,
        "graph": 0.10,
    },
    "known_item": {
        "vector": 0.15,
        "bm25_content": 0.25,
        "metadata_lexical": 0.50,
        "graph": 0.10,
    },
    "relationship": {
        "vector": 0.25,
        "bm25_content": 0.25,
        "metadata_lexical": 0.20,
        "graph": 0.30,
    },
}


@dataclass(frozen=True)
class RetrievalIntentResolution:
    """One validated request hint and the ranked intent used by the service."""

    requested_intent: RankedRetrievalIntent | None
    resolved_intent: RankedRetrievalIntent
    intent_source: IntentSource
    fallback_reason: FallbackReason | None = None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def validate_requested_intent(value: str | None) -> RankedRetrievalIntent | None:
    """Reject unknown hints before a caller reaches retrieval adapters."""

    if value is None:
        return None
    if value not in RANKED_RETRIEVAL_INTENTS:
        raise ValueError(f"unsupported retrieval intent: {value}")
    return cast(RankedRetrievalIntent, value)


def resolve_retrieval_intent(
    requested_intent: str | None,
    *,
    has_external_id: bool,
    has_code_symbol: bool,
    has_strong_metadata_identity: bool,
    has_traversable_entity: bool,
) -> RetrievalIntentResolution:
    """Resolve intent without language detection, translation, or an LLM call."""

    requested = validate_requested_intent(requested_intent)
    if requested == "relationship" and not has_traversable_entity:
        return RetrievalIntentResolution(
            requested_intent=requested,
            resolved_intent="general_hybrid",
            intent_source="fallback",
            fallback_reason="no_traversable_entity",
        )
    if requested is not None:
        return RetrievalIntentResolution(
            requested_intent=requested,
            resolved_intent=requested,
            intent_source="requested",
        )
    if has_external_id or has_code_symbol or has_strong_metadata_identity:
        return RetrievalIntentResolution(
            requested_intent=None,
            resolved_intent="known_item",
            intent_source="deterministic",
        )
    return RetrievalIntentResolution(
        requested_intent=None,
        resolved_intent="general_hybrid",
        intent_source="default",
    )


def fusion_weights(intent: RankedRetrievalIntent) -> dict[str, float]:
    """Return the fixed weighted-RRF policy for one resolved intent."""

    return dict(_FUSION_WEIGHTS[intent])


__all__ = [
    "FallbackReason",
    "RANKED_RETRIEVAL_INTENTS",
    "RankedRetrievalIntent",
    "RetrievalIntentResolution",
    "fusion_weights",
    "resolve_retrieval_intent",
    "validate_requested_intent",
]
