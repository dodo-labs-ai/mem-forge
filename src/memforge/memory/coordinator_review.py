"""Identity, staging and recurrence inputs of SupportRelationCoordinator Reviews.

A coordinator Review is a Lifecycle Plan ``CREATE_REVIEW`` in ``lifecycle_reviews``.
Its ID names one conflict: the Source Unit, the old Memory, the proposal and the
normalized claim of the staged Candidate, so every revision that raises the same
conflict names the same Review, and a human decision on one proposal never
silences a different one. The staged Candidate keeps its exact Evidence digests; a
later revision that did not extract it again carries the conflict forward only
while every part of that Evidence is still exactly current.
"""

from __future__ import annotations

from memforge.memory.text_view import TextEvidenceView

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date

from memforge.memory.candidate_admission import normalized_claim
from memforge.memory.evidence import ActiveSupportEvidence, EvidenceRole, RelationDirection
from memforge.memory.lifecycle_plan import LifecycleReview, LifecycleReviewStatus, lifecycle_stable_id
from memforge.memory.relation_classifier import MemoryRelationType
from memforge.models import CoordinatorProposal, Memory, RawMemory, content_hash
from memforge.pipeline.support_reading import EvidenceCorrespondence, correspond_evidence, exact_evidence_selection
from memforge.pipeline.revision_assessment import RevisionAssessmentContext
from memforge.pipeline.support_reading import SupportWorkItem
from memforge.source_projection import AnchorKind, SourceAnchor
from memforge.pipeline.support_relation_coordinator import RelationLedgerEntry

COORDINATOR_REVIEW_ORIGIN = "support_relation_coordinator"


def coordinator_review_id(source_unit_id: str, memory_id: str, proposal: CoordinatorProposal, claim: str) -> str:
    """One conflict's Review ID; it never includes the per-run reconciliation scope."""
    return lifecycle_stable_id(
        "review", source_unit_id, memory_id, proposal.value, content_hash(normalized_claim(claim)),
    )


def is_coordinator_review(review: LifecycleReview, *, source_unit_id: str | None = None) -> bool:
    """Whether SupportRelationCoordinator raised this Review, optionally for one Source Unit."""
    staged = review.staged_evidence
    return staged.get("origin") == COORDINATOR_REVIEW_ORIGIN and (
        source_unit_id is None or staged.get("source_unit_id") == source_unit_id
    )


def staged_candidate(raw: RawMemory) -> dict[str, object]:
    """The Candidate as a Review stages it: its claim and the exact identity of its Evidence."""
    selection = raw.resolved_evidence_selection
    if selection is None:
        raise ValueError("a coordinator Review stages only a Candidate with resolved Evidence")
    return {
        "content": raw.content,
        "memory_type": raw.memory_type,
        "valid_from": raw.valid_from,
        "valid_until": raw.valid_until,
        "entity_refs": list(raw.entity_refs),
        "evidence": [
            {
                "role": part.role.value,
                "observation_id": part.anchor.observation_id,
                "raw_content_sha256": part.raw_content_sha256,
                "presentation_sha256": part.presentation_sha256,
                "observation_revision_id": part.anchor.observation_revision_id,
                "anchor_kind": part.anchor.kind.value,
                "range_start": part.anchor.range_start,
                "range_end": part.anchor.range_end,
                "fragment_id": part.anchor.fragment_id,
                "excerpt": part.excerpt,
                "text_view": part.text_view.payload() if part.text_view else None,
            }
            for part in selection.parts
        ],
    }


@dataclass(frozen=True)
class CarriedConflict:
    """A pending coordinator Review whose staged Candidate this revision did not extract again."""

    memory_id: str
    proposal: CoordinatorProposal
    candidate: RawMemory
    reason: str
    # The equivalent Candidate a rejection binds the old Memory to, while still exactly current.
    rejection_rebind: RawMemory | None = None


@dataclass(frozen=True)
class CarriedReviews:
    conflicts: tuple[CarriedConflict, ...]
    unresolved_review_ids: frozenset[str]


async def carried_conflicts(
    reviews: Sequence[LifecycleReview], context: RevisionAssessmentContext,
    *, evaluator, context_for_review,
) -> CarriedReviews:
    """The pending conflicts among one Source Unit's coordinator Reviews whose staged Candidate is exactly current.

    An update extracts only changed structures, so an unchanged Candidate is not
    extracted again and Relation cannot raise its conflict. Correspondence only
    locates its Evidence. The ordinary fixed-claim revision work then validates
    the whole Candidate against changes since the Review was staged. An
    unjudgeable Candidate preserves its Review without entering the ledger.
    """
    carried = []
    unresolved = set()
    work = []
    staged_claims = {}
    for review in reviews:
        if review.status is not LifecycleReviewStatus.PENDING:
            continue
        staged = review.staged_evidence
        candidate, verifiable = _current_candidate(staged.get("candidate"), context)
        if not verifiable:
            unresolved.add(review.id)
        if candidate is None:
            continue
        rejection, rejection_verifiable = _current_candidate(staged.get("rejection_candidate"), context)
        if staged.get("rejection_candidate") is not None and not rejection_verifiable:
            unresolved.add(review.id)
        conflict = CarriedConflict(
            memory_id=review.incumbent_memory_id,
            proposal=CoordinatorProposal(str(staged.get("proposal"))),
            candidate=candidate,
            reason=review.reason or "",
            rejection_rebind=rejection,
        )
        baseline_context = await context_for_review(review)
        for name, raw in (("candidate", candidate), ("rejection", rejection)):
            if raw is None:
                continue
            work_id = f"review-{review.id}-{name}"
            fixed = Memory(
                id=work_id, content=raw.content, content_hash=content_hash(raw.content),
                memory_type=raw.memory_type, entity_refs=list(raw.entity_refs),
                valid_from=date.fromisoformat(raw.valid_from) if raw.valid_from else None,
                valid_until=date.fromisoformat(raw.valid_until) if raw.valid_until else None,
            )
            parts = tuple(ActiveSupportEvidence(
                memory_id=work_id, source_id=context.projection.source_id,
                reference_id=f"{work_id}-{index}", evidence_unit_id=work_id,
                role=part.role, anchor=part.anchor, excerpt=part.excerpt,
                raw_content_sha256=part.raw_content_sha256, presentation_sha256=part.presentation_sha256,
                text_view=part.text_view,
            ) for index, part in enumerate(raw.resolved_evidence_selection.parts))
            work.append(SupportWorkItem(work_id, fixed, parts, baseline_context))
        staged_claims[review.id] = conflict
    assessed = await evaluator.assess_many(work) if work else {}
    for review_id, conflict in staged_claims.items():
        candidate = assessed[f"review-{review_id}-candidate"]
        if candidate.supported is None:
            unresolved.add(review_id)
            continue
        if not candidate.supported:
            continue
        rejection = None
        if conflict.rejection_rebind is not None:
            outcome = assessed[f"review-{review_id}-rejection"]
            if outcome.supported is None:
                unresolved.add(review_id)
            elif outcome.supported:
                rejection = replace(outcome.memory, entity_refs=conflict.rejection_rebind.entity_refs)
        carried.append(replace(
            conflict, candidate=replace(candidate.memory, entity_refs=conflict.candidate.entity_refs),
            rejection_rebind=rejection,
        ))
    return CarriedReviews(tuple(carried), frozenset(unresolved))


@dataclass(frozen=True)
class CarriedLedger:
    """This revision's Candidates and Relation edges with the carried conflicts added."""

    candidates: tuple[RawMemory, ...]
    entries: tuple[RelationLedgerEntry, ...]
    # The carried pairs: a pending Review already holds their conflict, so none is re-checked.
    carried_pairs: frozenset[tuple[int, str]]


def carry_into_ledger(
    candidates: Sequence[RawMemory],
    entries: Sequence[RelationLedgerEntry],
    conflicts: Sequence[CarriedConflict],
) -> CarriedLedger:
    """Add each carried conflict's Candidate and edge, as Relation raised them when the Review was created.

    A Candidate extracted again this revision keeps only the edges Relation gives it now.
    """
    extracted = {normalized_claim(raw.content) for raw in candidates}
    merged = list(candidates)
    ledger = list(entries)
    pairs = {(entry.candidate_index, entry.incumbent_id) for entry in entries}
    indices: dict[str, int] = {}
    carried: set[tuple[int, str]] = set()

    def carry(memory_id: str, raw: RawMemory, relation_type: MemoryRelationType, reason: str) -> None:
        claim = normalized_claim(raw.content)
        if claim in extracted:
            return
        if claim not in indices:
            indices[claim] = len(merged)
            merged.append(raw)
        pair = (indices[claim], memory_id)
        if pair in pairs:
            return
        pairs.add(pair)
        carried.add(pair)
        ledger.append(RelationLedgerEntry(
            candidate_index=pair[0], incumbent_id=memory_id, relation_type=relation_type,
            direction=RelationDirection.SYMMETRIC, reason=reason,
        ))

    for conflict in conflicts:
        carry(
            conflict.memory_id, conflict.candidate,
            MemoryRelationType.CONTRADICTS if conflict.proposal is CoordinatorProposal.SUPERSEDE
            else MemoryRelationType.EQUIVALENT,
            conflict.reason,
        )
        if conflict.rejection_rebind is not None:
            carry(conflict.memory_id, conflict.rejection_rebind, MemoryRelationType.EQUIVALENT, conflict.reason)
    return CarriedLedger(tuple(merged), tuple(ledger), frozenset(carried))


def _current_candidate(value: object, context: RevisionAssessmentContext) -> tuple[RawMemory | None, bool]:
    if not isinstance(value, Mapping):
        return None, False
    if any(not isinstance(value.get(name), str) or not value[name].strip() for name in ("content", "memory_type")):
        return None, False
    try:
        for name in ("valid_from", "valid_until"):
            if value.get(name) is not None:
                date.fromisoformat(value[name])
    except (TypeError, ValueError):
        return None, False
    evidence = value.get("evidence")
    if not isinstance(evidence, Sequence) or not evidence:
        return None, False
    # Legacy digest-only staging cannot prove the old occurrence. Keep its
    # existing Review, but never manufacture a pin from a current match.
    if any(not isinstance(part, Mapping) or not part.get("observation_revision_id") or not part.get("anchor_kind") for part in evidence):
        return None, False
    try:
        parts = [ActiveSupportEvidence(
            memory_id="staged", source_id=context.projection.source_id,
            reference_id=f"staged-{index}", evidence_unit_id="staged",
            role=EvidenceRole(str(part["role"])),
            anchor=SourceAnchor(
                kind=AnchorKind(str(part["anchor_kind"])),
                observation_id=str(part["observation_id"]),
                observation_revision_id=str(part["observation_revision_id"]),
                range_start=part.get("range_start"), range_end=part.get("range_end"),
                fragment_id=part.get("fragment_id"),
            ),
            excerpt=part.get("excerpt"), raw_content_sha256=part.get("raw_content_sha256"),
            presentation_sha256=part.get("presentation_sha256"),
            text_view=TextEvidenceView.from_payload(part.get("text_view")),
        ) for index, part in enumerate(evidence)]
    except (ValueError, TypeError, KeyError):
        return None, False
    if any(part.anchor.observation_revision_id not in context.revisions for part in parts):
        return None, False
    if any(part.role not in {EvidenceRole.PRIMARY, EvidenceRole.REQUIRED} for part in parts):
        return None, False
    correspondences = correspond_evidence(context, parts)
    if any(item.status in {EvidenceCorrespondence.UNKNOWN, EvidenceCorrespondence.AMBIGUOUS} for item in correspondences):
        return None, False
    if any(
        (target := context.current.get(part.anchor.observation_id)) is not None
        and context.revisions[part.anchor.observation_revision_id].evidence_profile != target.evidence_profile
        for part in parts
    ):
        return None, False
    selection = exact_evidence_selection(context, parts)
    if selection is None:
        return None, True
    primary = next(part for part in selection.parts if part.role is EvidenceRole.PRIMARY)
    return RawMemory(
        content=str(value["content"]),
        memory_type=str(value["memory_type"]),
        entity_refs=[str(ref) for ref in value.get("entity_refs") or ()],
        valid_from=_optional_text(value.get("valid_from")),
        valid_until=_optional_text(value.get("valid_until")),
        evidence_quote=primary.excerpt,
        extraction_context=primary.excerpt or "",
        source_observation_id=primary.anchor.observation_id,
        required_source_observation_ids=list(dict.fromkeys(
            part.anchor.observation_id for part in selection.parts if part.role is EvidenceRole.REQUIRED
        )),
        resolved_evidence_selection=selection,
    ), True


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) else None
