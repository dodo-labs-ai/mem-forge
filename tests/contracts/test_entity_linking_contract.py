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
        visible_memory_count=3,
        visible_source_count=2,
        specificity=0.5,
    )

    assert candidate.entity_id == 42
    assert candidate.channel == "alias_exact"
    assert candidate.contributing_channels == ("alias_exact",)
    assert candidate.activates_graph is True
    assert candidate.visible_memory_count == 3
    assert candidate.visible_source_count == 2
    assert candidate.specificity == 0.5


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
        "time_range",
        "memory_types",
        "limit",
    ]
    assert signature.parameters["query"].default is inspect.Parameter.empty
    assert signature.parameters["scope"].default is inspect.Parameter.empty
    assert signature.parameters["explicit_entities"].default == ()
    assert signature.parameters["source_filter"].default is None
    assert signature.parameters["time_range"].default is None
    assert signature.parameters["memory_types"].default is None
    assert signature.parameters["limit"].default == DEFAULT_ENTITY_LINK_LIMIT


def test_relational_store_exposes_graph_search_source_contract() -> None:
    signature = inspect.signature(RelationalStore.graph_search)

    assert list(signature.parameters) == [
        "self",
        "entity_ids",
        "scope",
        "memory_types",
        "limit",
        "source_filter",
        "time_range",
    ]
    assert signature.parameters["entity_ids"].default is inspect.Parameter.empty
    assert signature.parameters["scope"].default is inspect.Parameter.empty
    assert signature.parameters["memory_types"].default is inspect.Parameter.empty
    assert signature.parameters["limit"].default is inspect.Parameter.empty
    assert signature.parameters["source_filter"].default is None
    assert signature.parameters["time_range"].default is None
