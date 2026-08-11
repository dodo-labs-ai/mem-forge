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
