from memforge.pipeline.document_update import (
    plan_document_update,
    quote_overlaps_current_changes,
)


def test_diff_guided_plan_rejects_evidence_from_unchanged_content() -> None:
    previous = "\n".join(
        (
            "# Shared HANA Database Connections",
            "",
            "| Thread Group | Min | Max |",
            "| payrollTaskExecutor | 5 | 5 |",
            "",
            "![](../../../../../Desktop/old.png)",
        )
    )
    updated = "\n".join(
        (
            "# Shared HANA Database Connections",
            "",
            "| Thread Group | Min | Max |",
            "| payrollTaskExecutor | 5 | 5 |",
            "",
            "Here is an example of running threads:",
            "![](assets/list-of-threads.png)",
        )
    )

    plan = plan_document_update(
        previous_content=previous,
        updated_content=updated,
        data_shape="document",
    )

    assert plan.mode == "diff_guided"
    assert plan.current_changed_ranges
    assert not quote_overlaps_current_changes(
        updated,
        "| payrollTaskExecutor | 5 | 5 |",
        plan.current_changed_ranges,
    )
    assert quote_overlaps_current_changes(
        updated,
        "Here is an example of running threads:",
        plan.current_changed_ranges,
    )


def test_deletion_only_diff_grants_no_current_candidate_authority() -> None:
    previous = "# Policy\n\nA7 is retained."
    updated = "# Policy"

    plan = plan_document_update(
        previous_content=previous,
        updated_content=updated,
        data_shape="document",
    )

    assert plan.mode == "diff_guided"
    assert plan.current_changed_ranges == ()
    assert not quote_overlaps_current_changes(
        updated,
        "# Policy",
        plan.current_changed_ranges,
    )


def test_diff_payload_larger_than_prompt_budget_falls_back_to_full_document() -> None:
    previous = "A" * 25_000
    updated = "B" * 25_000

    plan = plan_document_update(
        previous_content=previous,
        updated_content=updated,
        data_shape="document",
    )

    assert plan.mode == "full_document"
    assert plan.reason == "diff_payload_too_large"
    assert plan.thresholds["max_diff_chars"] == 40_000
