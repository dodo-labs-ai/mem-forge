from __future__ import annotations

import pytest

from memforge.retrieval.query_plan import QueryIdentity, build_lexical_query_plan

_NO_IDENTIFIERS = QueryIdentity(identifiers=(), has_free_text=True)


@pytest.mark.parametrize("requested", ["general_hybrid", "known_item"])
def test_requested_ranked_intent_is_honored(requested: str) -> None:
    from memforge.retrieval.intents import resolve_retrieval_intent

    resolution = resolve_retrieval_intent(
        requested,
        identity=_NO_IDENTIFIERS,
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
        identity=_NO_IDENTIFIERS,
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
        identity=_NO_IDENTIFIERS,
        has_strong_metadata_identity=False,
        has_traversable_entity=True,
    )

    assert resolution.resolved_intent == "relationship"
    assert resolution.intent_source == "requested"
    assert resolution.fallback_reason is None


@pytest.mark.parametrize(
    ("query", "has_strong_metadata_identity", "resolved"),
    [
        ("PAYR-179397", False, "known_item"),
        ("persist_cutoff_blocker_hints", False, "known_item"),
        ("Create Access Review in Quarterly Payroll", True, "known_item"),
        ("edit a locked rule", False, "general_hybrid"),
    ],
)
def test_omitted_intent_uses_only_deterministic_identity_evidence(
    query,
    has_strong_metadata_identity,
    resolved,
) -> None:
    from memforge.retrieval.intents import resolve_retrieval_intent

    resolution = resolve_retrieval_intent(
        None,
        identity=build_lexical_query_plan(query).identity,
        has_strong_metadata_identity=has_strong_metadata_identity,
        has_traversable_entity=False,
    )

    assert resolution.resolved_intent == resolved
    assert resolution.intent_source == (
        "deterministic" if resolved == "known_item" else "default"
    )


@pytest.mark.parametrize(
    "query",
    [
        "How do I configure workspaces in MemForge?",
        "在 MemForge 里怎么配置工作区？",
        "How does GitHub sync pull requests?",
        "GitHub 的同步是怎么工作的？",
        "What is MemForge?",
        "Which OpenAI model does PostgreSQL search use for Acme Payroll?",
        "how does getUserName work",
        "PersistCutOffBlockerHintsCommand blocker hint",
        'Find the memory titled "Create Access Review in Quarterly Payroll"',
        "Use Node.js for the script",
        "what is pkg/errors",
    ],
)
def test_ambiguous_name_inside_a_question_is_not_a_known_item(query) -> None:
    from memforge.retrieval.intents import names_one_item

    assert not names_one_item(build_lexical_query_plan(query).identity)


@pytest.mark.parametrize(
    "query",
    [
        "PersistCutOffBlockerHintsCommand",
        "MemForge",
        "getUserName",
        "GitHub/GitLab",
        '"Create Access Review in Quarterly Payroll"',
        "“Create Access Review in Quarterly Payroll”",
        "Cargo.toml",
        "Node.js",
        "src/memforge",
        "pkg/errors",
    ],
)
def test_query_made_only_of_names_is_a_known_item(query) -> None:
    from memforge.retrieval.intents import names_one_item

    assert names_one_item(build_lexical_query_plan(query).identity)


@pytest.mark.parametrize(
    "query",
    [
        "PAYR-179397 lifecycle blocker",
        "what changed in #42",
        "123e4567-e89b-12d3-a456-426614174000 owner",
        "why does persist_cutoff_blocker_hints run twice",
        "where is VALIDATION_STATE written",
        "who imports src/memforge/retrieval/search.py",
        "files under a/b/c",
        "what is in docs/architecture.md",
        "why does litellm.aembedding time out",
        "what does memforge.retrieval.search export",
        "PayrollCutoffCommand.resolveWindow return value",
        "where is search.py tested",
        "what does README.md say about setup",
        "docker-compose.yml",
        "what is in docker-compose.yml",
        ".pre-commit-config.yaml",
        "src/App.tsx",
        "why does lib/Foo.java fail",
        "core/Cargo.toml",
        "read ~/notes.md",
        "contents of /etc/hosts",
        "see https://github.com/shno-labs/mem-forge for details",
    ],
)
def test_self_identifying_identifier_anywhere_is_a_known_item(query) -> None:
    from memforge.retrieval.intents import names_one_item

    assert names_one_item(build_lexical_query_plan(query).identity)


@pytest.mark.parametrize(
    "query",
    [
        "should approvals go to managers and/or HR",
        "list payroll exceptions, e.g. late approvals",
        "only one approver, i.e. the manager",
        "TCP/IP retries during sync",
        "upgrade Node.js before the release",
        "does ASP.NET hosting support it",
        "what happened on 10/14/2026",
        "U.S. payroll rules",
        "client/server handshake",
        "use React/Node.js for the app",
    ],
)
def test_slash_and_dot_prose_is_not_a_known_item(query) -> None:
    from memforge.retrieval.intents import names_one_item

    identity = build_lexical_query_plan(query).identity

    assert identity.kinds <= {"slash_term", "dotted_term"}
    assert not names_one_item(identity)


def test_unknown_intent_is_rejected() -> None:
    from memforge.retrieval.intents import resolve_retrieval_intent

    with pytest.raises(ValueError, match="unsupported retrieval intent"):
        resolve_retrieval_intent(
            "semantic_lookup",
            identity=_NO_IDENTIFIERS,
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
