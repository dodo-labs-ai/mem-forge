from __future__ import annotations

import inspect

from memforge.storage.adapters.protocols import (
    DEFAULT_ENTITY_LINK_LIMIT,
    EntityLinkCandidate,
    EntityLinkResult,
    RelationalStore,
)


def test_entity_link_candidate_carries_channel_evidence() -> None:
    candidate = EntityLinkCandidate(
        entity_id=42,
        canonical_name="payroll control center",
        matched_alias="payroll control center",
        channel="alias_exact",
        contributing_channels=("alias_exact",),
        score=0.95,
        matched_text="payroll control center",
        activates_graph=True,
    )

    assert candidate.entity_id == 42
    assert candidate.channel == "alias_exact"
    assert candidate.contributing_channels == ("alias_exact",)
    assert candidate.activates_graph is True


def test_entity_link_result_defaults_empty() -> None:
    result = EntityLinkResult()

    assert result.candidates == ()
    assert result.unmatched_explicit_entities == ()


def test_relational_store_exposes_query_entity_linking_contract() -> None:
    signature = inspect.signature(RelationalStore.link_query_entities)

    assert list(signature.parameters) == [
        "self",
        "query",
        "scope",
        "explicit_entities",
        "source_filter",
        "memory_types",
        "limit",
    ]
    assert signature.parameters["query"].default is inspect.Parameter.empty
    assert signature.parameters["scope"].default is inspect.Parameter.empty
    assert signature.parameters["explicit_entities"].default == ()
    assert signature.parameters["source_filter"].default is None
    assert signature.parameters["memory_types"].default is None
    assert signature.parameters["limit"].default == DEFAULT_ENTITY_LINK_LIMIT
