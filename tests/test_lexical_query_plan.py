"""Shared lexical query-plan policy tests."""

from memforge.retrieval.query_plan import build_lexical_query_plan


def test_metadata_query_plan_requires_three_of_five_ordinary_terms() -> None:
    plan = build_lexical_query_plan("process map failed command error")

    assert plan.metadata.ordinary_terms == (
        "process",
        "map",
        "failed",
        "command",
        "error",
    )
    assert plan.metadata.minimum_should_match == 3


def test_metadata_query_plan_keeps_short_queries_all_term() -> None:
    plan = build_lexical_query_plan("process tree")

    assert plan.metadata.ordinary_terms == ("process", "tree")
    assert plan.metadata.minimum_should_match == 2


def test_metadata_query_plan_separates_structured_and_quoted_anchors() -> None:
    plan = build_lexical_query_plan(
        'SFPAY-181363 HandlePeriodInitializationCommand "No process tree found" root cause'
    )

    assert [(anchor.kind, anchor.value) for anchor in plan.metadata.exact_anchors] == [
        ("external_id", "SFPAY-181363"),
        ("code_symbol", "HandlePeriodInitializationCommand"),
        ("quoted_phrase", "No process tree found"),
    ]
    assert plan.metadata.ordinary_terms == ("root", "cause")
    assert plan.metadata.minimum_should_match == 2


def test_metadata_query_plan_removes_fillers_and_softens_work_item_modifiers() -> None:
    plan = build_lexical_query_plan(
        "please help find process map failed command error jira ticket"
    )

    assert plan.metadata.ordinary_terms == (
        "process",
        "map",
        "failed",
        "command",
        "error",
    )
    assert plan.metadata.minimum_should_match == 3


def test_metadata_query_plan_keeps_modifiers_when_they_are_the_only_terms() -> None:
    plan = build_lexical_query_plan("jira task")

    assert plan.metadata.ordinary_terms == ("jira", "task")
    assert plan.metadata.minimum_should_match == 2


def test_query_identity_records_each_identifier_kind_and_surrounding_text() -> None:
    plan = build_lexical_query_plan(
        'SFPAY-181363 "No process tree found" in src/memforge/retrieval/search.py '
        "via litellm.aembedding and persist_cutoff_blocker_hints for MemForge"
    )

    assert [(identifier.kind, identifier.value) for identifier in plan.identity.identifiers] == [
        ("external_id", "SFPAY-181363"),
        ("quoted_phrase", "No process tree found"),
        ("path", "src/memforge/retrieval/search.py"),
        ("dotted_name", "litellm.aembedding"),
        ("snake_case", "persist_cutoff_blocker_hints"),
        ("camel_case", "MemForge"),
    ]
    assert plan.identity.has_free_text


def test_query_identity_has_no_free_text_when_only_identifiers_and_punctuation_remain() -> None:
    plan = build_lexical_query_plan("MemForge, GitHub?")

    assert [identifier.kind for identifier in plan.identity.identifiers] == ["camel_case", "camel_case"]
    assert not plan.identity.has_free_text


def test_query_identity_counts_free_text_in_any_script() -> None:
    plan = build_lexical_query_plan("在 MemForge 里怎么配置工作区？")

    assert [identifier.value for identifier in plan.identity.identifiers] == ["MemForge"]
    assert plan.identity.has_free_text


def test_query_identity_does_not_change_exact_anchors() -> None:
    plan = build_lexical_query_plan(
        "Does OpenAI or PostgreSQL in getUserName reach litellm.aembedding via src/memforge/x.py?"
    )

    assert plan.metadata.exact_anchors == ()
    assert {identifier.kind for identifier in plan.identity.identifiers} == {
        "camel_case",
        "dotted_name",
        "path",
    }


def test_query_identity_separates_paths_and_file_names_from_slash_and_dotted_terms() -> None:
    plan = build_lexical_query_plan(
        "Is docker-compose.yml in src/App.tsx or pkg/errors, e.g. with Node.js and Cargo.toml?"
    )

    assert [(identifier.kind, identifier.value) for identifier in plan.identity.identifiers] == [
        ("dotted_name", "docker-compose.yml"),
        ("path", "src/App.tsx"),
        ("slash_term", "pkg/errors"),
        ("dotted_term", "e.g"),
        ("dotted_term", "Node.js"),
        ("dotted_term", "Cargo.toml"),
    ]
    assert plan.metadata.exact_anchors == ()
