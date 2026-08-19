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
    assert [action.decision for action in presentation.actions] == ["approve", "reject"]
    assert presentation.actions[0].requires_note is False
    assert presentation.actions[1].requires_note is True
    assert presentation.decision_label == "Updated"
    assert presentation.summary == "Use the proposed source state or keep the current memory?"
    assert presentation.current_label == "Current memory"
    assert presentation.proposed_label == "Proposed memory"
    assert presentation.technical_reason == "audit_keep_vs_candidate_supersede"


def test_support_removal_explains_that_other_support_can_keep_memory_active() -> None:
    presentation = present_lifecycle_review(
        staged_evidence={"proposed_disposition": "remove_support"},
        reason="source_no_longer_supports_claim",
    )

    assert presentation.decision_label == "Support removed"
    assert presentation.summary == "Apply this source-support removal?"
    assert presentation.proposed_label == "Source change"
    assert "No replacement memory" in presentation.proposed_empty_text
    assert "only if no other support remains" in presentation.actions[0].consequence


def test_legacy_internal_reason_is_only_technical_detail() -> None:
    presentation = present_memory_review(
        kind="supersede",
        reason="deterministic_relation_conflict:v7:v8",
    )

    assert "deterministic_relation_conflict" not in presentation.summary
    assert "deterministic_relation_conflict" not in presentation.why_human
    assert presentation.technical_reason == "deterministic_relation_conflict:v7:v8"


def test_source_backed_correction_explains_support_override() -> None:
    presentation = present_memory_review(
        kind="supersede",
        reason="user corrected source-backed knowledge",
        source_backed_correction=True,
    )

    assert "source-backed" in presentation.summary
    assert "active Source Support" in presentation.why_human
    assert "preserve its Source evidence" in presentation.actions[0].consequence


def test_cross_source_conflict_is_presented_as_a_non_destructive_finding() -> None:
    presentation = present_memory_review(
        kind="cross_source_conflict",
        reason="conflicting_source_authority",
    )

    assert presentation.decision_label == "Conflict"
    assert presentation.summary == "Do these source-backed memories really conflict?"
    assert presentation.current_label == "Source-backed memory"
    assert presentation.proposed_label == "Other source-backed memory"
    assert [action.key for action in presentation.actions] == [
        "confirm_conflict",
        "not_a_conflict",
    ]
    assert [action.decision for action in presentation.actions] == ["approve", "reject"]
    assert all("keep both" in action.consequence for action in presentation.actions)
