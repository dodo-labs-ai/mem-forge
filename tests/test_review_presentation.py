from memforge.memory.review_presentation import (
    present_lifecycle_review,
    present_memory_review,
)


def test_lifecycle_review_exposes_exactly_two_user_decisions() -> None:
    presentation = present_lifecycle_review(
        staged_evidence={
            "proposed_disposition": "supersede",
            "candidate": {"content": "The service is owned by the new team."},
        },
        reason="audit_keep_vs_candidate_supersede",
    )

    assert [action.key for action in presentation.actions] == [
        "use_latest_state",
        "keep_current_state",
    ]
    assert [action.label for action in presentation.actions] == [
        "Use latest state",
        "Keep current state",
    ]
    assert presentation.actions[0].requires_note is False
    assert presentation.actions[1].requires_note is True
    assert presentation.technical_reason == "audit_keep_vs_candidate_supersede"


def test_support_removal_explains_that_other_support_can_keep_memory_active() -> None:
    presentation = present_lifecycle_review(
        staged_evidence={"proposed_disposition": "remove_support"},
        reason="source_no_longer_supports_claim",
    )

    assert presentation.summary == "The latest source state no longer supports this memory."
    assert "only if no other support remains" in presentation.actions[0].consequence


def test_legacy_internal_reason_is_only_technical_detail() -> None:
    presentation = present_memory_review(
        kind="supersede",
        reason="deterministic_relation_conflict:v7:v8",
    )

    assert "deterministic_relation_conflict" not in presentation.summary
    assert "deterministic_relation_conflict" not in presentation.why_human
    assert presentation.technical_reason == "deterministic_relation_conflict:v7:v8"
