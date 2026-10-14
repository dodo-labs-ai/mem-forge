"""Language-neutral ranked retrieval intent contract and resolution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, cast

from memforge.retrieval.query_plan import IdentifierKind, QueryIdentity

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


# Identifier kinds that name one item wherever they appear in a query.  A
# camel-case word, a quoted phrase, or a slash or dotted term may equally be a
# product name, prose or a phrase inside a question ("MemForge", "and/or",
# "Node.js"), so those name an item only when the query is nothing else.
_SELF_IDENTIFYING_KINDS: frozenset[IdentifierKind] = frozenset(
    {"url", "uuid", "path", "external_id", "dotted_name", "snake_case"}
)


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


def names_one_item(identity: QueryIdentity) -> bool:
    """Whether the query's identifier syntax alone names one item.

    True for a self-identifying identifier anywhere in the query, or for a
    query made only of identifiers.
    """

    if identity.kinds & _SELF_IDENTIFYING_KINDS:
        return True
    return bool(identity.identifiers) and not identity.has_free_text


def resolve_retrieval_intent(
    requested_intent: str | None,
    *,
    identity: QueryIdentity,
    has_strong_metadata_identity: bool,
    has_traversable_entity: bool,
) -> RetrievalIntentResolution:
    """Resolve intent without language detection, translation, or an LLM call.

    An omitted intent is ``known_item`` when the query's identifiers name one
    item or a metadata field equals the whole query, and ``general_hybrid``
    otherwise.
    """

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
    if names_one_item(identity) or has_strong_metadata_identity:
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
    "names_one_item",
    "resolve_retrieval_intent",
    "validate_requested_intent",
]
