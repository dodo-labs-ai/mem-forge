from __future__ import annotations

import pytest


@pytest.mark.parametrize("requested", ["general_hybrid", "known_item"])
def test_requested_ranked_intent_is_honored(requested: str) -> None:
    from memforge.retrieval.intents import resolve_retrieval_intent

    resolution = resolve_retrieval_intent(
        requested,
        has_external_id=False,
        has_code_symbol=False,
        has_strong_metadata_identity=False,
        has_traversable_entity=False,
    )

    assert resolution.to_dict() == {
        "requested_intent": requested,
        "resolved_intent": requested,
        "intent_source": "requested",
        "fallback_reason": None,
    }


def test_relationship_intent_falls_back_without_a_traversable_entity() -> None:
    from memforge.retrieval.intents import resolve_retrieval_intent

    resolution = resolve_retrieval_intent(
        "relationship",
        has_external_id=False,
        has_code_symbol=False,
        has_strong_metadata_identity=False,
        has_traversable_entity=False,
    )

    assert resolution.to_dict() == {
        "requested_intent": "relationship",
        "resolved_intent": "general_hybrid",
        "intent_source": "fallback",
        "fallback_reason": "no_traversable_entity",
    }


def test_relationship_intent_is_honored_with_a_traversable_entity() -> None:
    from memforge.retrieval.intents import resolve_retrieval_intent

    resolution = resolve_retrieval_intent(
        "relationship",
        has_external_id=False,
        has_code_symbol=False,
        has_strong_metadata_identity=False,
        has_traversable_entity=True,
    )

    assert resolution.resolved_intent == "relationship"
    assert resolution.intent_source == "requested"
    assert resolution.fallback_reason is None


@pytest.mark.parametrize(
    ("evidence", "resolved"),
    [
        ({"has_external_id": True}, "known_item"),
        ({"has_code_symbol": True}, "known_item"),
        ({"has_strong_metadata_identity": True}, "known_item"),
        ({}, "general_hybrid"),
    ],
)
def test_omitted_intent_uses_only_deterministic_identity_evidence(evidence, resolved) -> None:
    from memforge.retrieval.intents import resolve_retrieval_intent

    resolution = resolve_retrieval_intent(
        None,
        has_external_id=evidence.get("has_external_id", False),
        has_code_symbol=evidence.get("has_code_symbol", False),
        has_strong_metadata_identity=evidence.get("has_strong_metadata_identity", False),
        has_traversable_entity=False,
    )

    assert resolution.resolved_intent == resolved
    assert resolution.intent_source == (
        "deterministic" if resolved == "known_item" else "default"
    )


def test_unknown_intent_is_rejected() -> None:
    from memforge.retrieval.intents import resolve_retrieval_intent

    with pytest.raises(ValueError, match="unsupported retrieval intent"):
        resolve_retrieval_intent(
            "semantic_lookup",
            has_external_id=False,
            has_code_symbol=False,
            has_strong_metadata_identity=False,
            has_traversable_entity=False,
        )


def test_each_ranked_intent_has_one_fixed_normalized_fusion_policy() -> None:
    from memforge.retrieval.intents import RANKED_RETRIEVAL_INTENTS, fusion_weights

    policies = {
        intent: fusion_weights(intent)
        for intent in RANKED_RETRIEVAL_INTENTS
    }

    assert policies == {
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
    assert all(sum(weights.values()) == pytest.approx(1.0) for weights in policies.values())
