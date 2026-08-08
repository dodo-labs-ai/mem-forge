"""Provider-neutral language for human lifecycle decisions.

This module translates an already-authoritative Review record into a small
decision contract.  It never decides lifecycle authority and never inspects
provider-specific fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping


ReviewActionKey = Literal[
    "use_latest_state",
    "keep_current_state",
    "confirm_conflict",
    "not_a_conflict",
]
ReviewDecision = Literal["approve", "reject"]
ReviewDecisionLabel = Literal["Updated", "Support removed", "Conflict"]


@dataclass(frozen=True, slots=True)
class ReviewActionPresentation:
    key: ReviewActionKey
    decision: ReviewDecision
    label: str
    consequence: str
    requires_note: bool


@dataclass(frozen=True, slots=True)
class ReviewPresentation:
    decision_label: ReviewDecisionLabel
    summary: str
    why_human: str
    current_label: str
    proposed_label: str
    proposed_empty_text: str
    actions: tuple[ReviewActionPresentation, ReviewActionPresentation]
    technical_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def present_memory_review(*, kind: str, reason: str | None) -> ReviewPresentation:
    """Present a workbench Review without exposing its internal action names."""

    if kind == "cross_source_conflict":
        return ReviewPresentation(
            decision_label="Conflict",
            summary="Do these source-backed memories really conflict?",
            why_human=(
                "Both memories have independent source evidence. Confirming or dismissing "
                "the conflict closes this Review without choosing a winning source."
            ),
            current_label="Source-backed memory",
            proposed_label="Other source-backed memory",
            proposed_empty_text="The other source-backed memory snapshot is unavailable.",
            actions=(
                ReviewActionPresentation(
                    key="confirm_conflict",
                    decision="approve",
                    label="Confirm conflict",
                    consequence=(
                        "Record this as a reviewed conflict and keep both source-backed "
                        "memories active."
                    ),
                    requires_note=False,
                ),
                ReviewActionPresentation(
                    key="not_a_conflict",
                    decision="reject",
                    label="Not a conflict",
                    consequence=(
                        "Dismiss this conflict finding and keep both source-backed memories active."
                    ),
                    requires_note=True,
                ),
            ),
            technical_reason=reason,
        )
    return _presentation(
        decision_label="Updated",
        summary="Use the proposed memory or keep the current one?",
        why_human=(
            "The update would change active memory state, so MemForge needs your "
            "decision before applying it."
        ),
        current_label="Current memory",
        proposed_label="Proposed memory",
        proposed_empty_text="The proposed memory snapshot is unavailable.",
        use_latest_consequence=("Use the proposed state going forward and keep the previous state in audit history."),
        technical_reason=reason,
    )


def present_lifecycle_review(
    *,
    staged_evidence: Mapping[str, object],
    reason: str | None,
) -> ReviewPresentation:
    """Present one complete lifecycle proposal without granting it authority."""

    disposition = staged_evidence.get("proposed_disposition")
    candidate = staged_evidence.get("candidate")
    has_candidate = isinstance(candidate, Mapping) and bool(candidate.get("content"))
    if disposition == "supersede" or has_candidate:
        decision_label = "Updated"
        summary = "Use the proposed source state or keep the current memory?"
        use_latest = "Use the proposed state going forward and keep the previous state in audit history."
        proposed_label = "Proposed memory"
        proposed_empty_text = "The proposed memory snapshot is unavailable."
    elif disposition == "remove_support":
        decision_label = "Support removed"
        summary = "Apply this source-support removal?"
        use_latest = "Apply the source update. The current memory will retire only if no other support remains."
        proposed_label = "Source change"
        proposed_empty_text = "No replacement memory; this proposal removes source support."
    else:
        decision_label = "Updated"
        summary = "Apply the proposed source change or keep the current memory?"
        use_latest = "Apply the complete lifecycle proposal shown here."
        proposed_label = "Proposed memory"
        proposed_empty_text = "The proposal does not include a replacement memory."
    return _presentation(
        decision_label=decision_label,
        summary=summary,
        why_human=(
            "Lifecycle checks produced one complete proposal, but applying it would "
            "change active memory state and requires your decision."
        ),
        current_label="Current memory",
        proposed_label=proposed_label,
        proposed_empty_text=proposed_empty_text,
        use_latest_consequence=use_latest,
        technical_reason=reason,
    )


def _presentation(
    *,
    decision_label: ReviewDecisionLabel,
    summary: str,
    why_human: str,
    current_label: str,
    proposed_label: str,
    proposed_empty_text: str,
    use_latest_consequence: str,
    technical_reason: str | None,
) -> ReviewPresentation:
    return ReviewPresentation(
        decision_label=decision_label,
        summary=summary,
        why_human=why_human,
        current_label=current_label,
        proposed_label=proposed_label,
        proposed_empty_text=proposed_empty_text,
        actions=(
            ReviewActionPresentation(
                key="use_latest_state",
                decision="approve",
                label="Use latest state",
                consequence=use_latest_consequence,
                requires_note=False,
            ),
            ReviewActionPresentation(
                key="keep_current_state",
                decision="reject",
                label="Keep current state",
                consequence="Keep the current memory active and discard this proposal.",
                requires_note=True,
            ),
        ),
        technical_reason=technical_reason,
    )
