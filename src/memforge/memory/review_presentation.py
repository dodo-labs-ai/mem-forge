"""Provider-neutral language for human lifecycle decisions.

This module translates an already-authoritative Review record into a small
decision contract.  It never decides lifecycle authority and never inspects
provider-specific fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, Mapping


ReviewActionKey = Literal["use_latest_state", "keep_current_state"]


@dataclass(frozen=True, slots=True)
class ReviewActionPresentation:
    key: ReviewActionKey
    label: str
    consequence: str
    requires_note: bool


@dataclass(frozen=True, slots=True)
class ReviewPresentation:
    summary: str
    why_human: str
    current_label: str
    proposed_label: str
    actions: tuple[ReviewActionPresentation, ReviewActionPresentation]
    technical_reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def present_memory_review(*, kind: str, reason: str | None) -> ReviewPresentation:
    """Present a workbench Review without exposing its internal action names."""

    if kind == "cross_source_conflict":
        summary = "Two current memories describe incompatible states."
        why_human = (
            "Both memories have valid source evidence, but MemForge has no authority "
            "to choose which state should remain active."
        )
    else:
        summary = "A newer candidate may replace the memory currently in use."
        why_human = "The update would change active memory state, so MemForge needs your decision before applying it."
    return _presentation(
        summary=summary,
        why_human=why_human,
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
        summary = "A source update proposes a newer state for this memory."
        use_latest = "Use the proposed state going forward and keep the previous state in audit history."
    elif disposition == "remove_support":
        summary = "The latest source state no longer supports this memory."
        use_latest = "Apply the source update. The current memory will retire only if no other support remains."
    else:
        summary = "A source update proposes changing the current memory state."
        use_latest = "Apply the complete lifecycle proposal shown here."
    return _presentation(
        summary=summary,
        why_human=(
            "Lifecycle checks produced one complete proposal, but applying it would "
            "change active memory state and requires your decision."
        ),
        use_latest_consequence=use_latest,
        technical_reason=reason,
    )


def _presentation(
    *,
    summary: str,
    why_human: str,
    use_latest_consequence: str,
    technical_reason: str | None,
) -> ReviewPresentation:
    return ReviewPresentation(
        summary=summary,
        why_human=why_human,
        current_label="Current state",
        proposed_label="Proposed state",
        actions=(
            ReviewActionPresentation(
                key="use_latest_state",
                label="Use latest state",
                consequence=use_latest_consequence,
                requires_note=False,
            ),
            ReviewActionPresentation(
                key="keep_current_state",
                label="Keep current state",
                consequence="Keep the current memory active and discard this proposal.",
                requires_note=True,
            ),
        ),
        technical_reason=technical_reason,
    )
