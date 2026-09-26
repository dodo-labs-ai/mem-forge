from memforge.pipeline.document_update import plan_document_update


def test_small_diff_records_only_inserted_or_replaced_ranges() -> None:
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
    [(range_start, range_end)] = plan.current_changed_ranges
    changed_text = updated[range_start:range_end]
    assert "Here is an example of running threads:" in changed_text
    assert "![](assets/list-of-threads.png)" in changed_text
    assert "payrollTaskExecutor" not in changed_text


def test_deletion_only_diff_records_no_current_changed_range() -> None:
    previous = "# Policy\n\nA7 is retained."
    updated = "# Policy"

    plan = plan_document_update(
        previous_content=previous,
        updated_content=updated,
        data_shape="document",
    )

    assert plan.mode == "diff_guided"
    assert plan.current_changed_ranges == ()


def test_diff_payload_larger_than_limit_is_recorded_as_full_document() -> None:
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
