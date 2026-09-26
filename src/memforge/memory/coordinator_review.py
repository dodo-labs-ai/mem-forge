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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from memforge.memory.candidate_admission import normalized_claim
from memforge.memory.evidence import EvidenceRole, RelationDirection
from memforge.memory.lifecycle_plan import LifecycleReview, LifecycleReviewStatus, lifecycle_stable_id
from memforge.memory.relation_classifier import MemoryRelationType
from memforge.models import CoordinatorProposal, RawMemory, content_hash
from memforge.pipeline.projection_fragments import ProjectionFragmentCatalog
from memforge.pipeline.support_reading import exact_evidence_selection
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
        "confidence": raw.confidence,
        "valid_from": raw.valid_from,
        "valid_until": raw.valid_until,
        "entity_refs": list(raw.entity_refs),
        "evidence": [
            {
                "role": part.role.value,
                "observation_id": part.anchor.observation_id,
                "raw_content_sha256": part.raw_content_sha256,
                "presentation_sha256": part.presentation_sha256,
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


def carried_conflicts(
    reviews: Sequence[LifecycleReview], catalog: ProjectionFragmentCatalog,
) -> tuple[CarriedConflict, ...]:
    """The pending conflicts among one Source Unit's coordinator Reviews whose staged Candidate is exactly current.

    An update extracts only changed structures, so an unchanged Candidate is not
    extracted again and Relation cannot raise its conflict. Its exact Evidence
    proves the Candidate still stands, so the conflict enters the coordinator
    again. A Candidate whose Evidence changed was extracted again if it still
    holds; otherwise its conflict is gone.
    """
    carried = []
    for review in reviews:
        if review.status is not LifecycleReviewStatus.PENDING:
            continue
        staged = review.staged_evidence
        candidate = _current_candidate(staged.get("candidate"), catalog)
        if candidate is None:
            continue
        carried.append(CarriedConflict(
            memory_id=review.incumbent_memory_id,
            proposal=CoordinatorProposal(str(staged.get("proposal"))),
            candidate=candidate,
            reason=review.reason or "",
            rejection_rebind=_current_candidate(staged.get("rejection_candidate"), catalog),
        ))
    return tuple(carried)


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


def _current_candidate(value: object, catalog: ProjectionFragmentCatalog) -> RawMemory | None:
    if not isinstance(value, Mapping):
        return None
    evidence = value.get("evidence")
    if not isinstance(evidence, Sequence) or not evidence:
        return None
    selection = exact_evidence_selection(catalog, [
        (
            EvidenceRole(str(part["role"])),
            (str(part["observation_id"]), str(part["raw_content_sha256"]), str(part["presentation_sha256"])),
        )
        for part in evidence
        if isinstance(part, Mapping)
    ])
    if selection is None:
        return None
    primary = next(part for part in selection.parts if part.role is EvidenceRole.PRIMARY)
    return RawMemory(
        content=str(value["content"]),
        memory_type=str(value["memory_type"]),
        confidence=float(value["confidence"]),
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
    )


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) else None
