"""Automatic DestructiveValidation of one Source Unit revision's destructive decisions.

Before a DELETE, SUPERSEDE or UPDATE reaches the Lifecycle Plan, the program
verifies the facts that decision rests on. It runs no model and reads nothing
again: Support Assessment owns the single ordered read, and this step only
checks what that read and the Relation line recorded.

1. Coverage and manifest: the old Memory has a Support result for every Support
   it has in this Unit, and none is ``UNRESOLVED``. An ``UNKNOWN`` Evidence part
   under partial coverage is always ``UNRESOLVED(partial_coverage)``, and a part
   is ``REMOVED`` only under an explicit tombstone or coverage that proves absence.
2. Receipt: a DELETE or SUPERSEDE rests on Supports that each read the whole
   reading order of the current revision and recorded its completion receipt.
3. Relation: a SUPERSEDE or UPDATE rests on a Relation line that gave every
   admitted Candidate its completion row over the complete old-Memory catalog.

The remaining checks of the contract live where their facts are: the Plan's
stale guard rejects a commit whose Support sets, Memory versions or Observation
revisions moved since these results were read, and the planner lets only the
last Active Support's removal retire a Memory.

A decision that fails is kept: the old Memory stays with its Support and its
validation baseline, and a replacing Candidate is consumed without an ADD, like
a local unresolved relationship. Decisions held for Review are proposals and
are validated when approved, by their stale guard.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from memforge.models import ReconcileAction, ReconcileOperation
from memforge.pipeline.support_relation_coordinator import MemorySupport

__all__ = ["DestructiveValidation", "KeptReason", "validate_destructive_operations"]

_DESTRUCTIVE = frozenset({ReconcileAction.DELETE, ReconcileAction.SUPERSEDE, ReconcileAction.UPDATE})
_RETIRES_SUPPORT = frozenset({ReconcileAction.DELETE, ReconcileAction.SUPERSEDE})
_REPLACES_MEMORY = frozenset({ReconcileAction.SUPERSEDE, ReconcileAction.UPDATE})


class KeptReason(str, Enum):
    """Why DestructiveValidation kept an old Memory."""

    SUPPORT_UNRESOLVED = "support_unresolved"
    READ_INCOMPLETE = "read_incomplete"
    RELATION_INCOMPLETE = "relation_incomplete"


@dataclass(frozen=True)
class DestructiveValidation:
    operations: tuple[ReconcileOperation, ...]
    # Old Memory id -> why its destructive decision was kept.
    kept: Mapping[str, KeptReason]


def validate_destructive_operations(
    operations: Sequence[ReconcileOperation],
    *,
    supports: Mapping[str, MemorySupport],
    relation_complete: bool,
) -> DestructiveValidation:
    """Keep every destructive decision whose Support or Relation facts are incomplete."""
    validated: list[ReconcileOperation] = []
    kept: dict[str, KeptReason] = {}
    for operation in operations:
        memory_id = operation.memory_id
        if operation.action not in _DESTRUCTIVE or memory_id is None or operation.flag_for_review:
            validated.append(operation)
            continue
        reason = _failed_check(operation, supports.get(memory_id), relation_complete)
        if reason is None:
            validated.append(operation)
            continue
        kept[memory_id] = reason
        validated.append(ReconcileOperation(
            action=ReconcileAction.NOOP,
            memory_id=memory_id,
            reason=f"DestructiveValidation kept the old Memory: {reason.value}",
            support_revalidation_skipped=True,
        ))
    return DestructiveValidation(tuple(validated), kept)


def _failed_check(
    operation: ReconcileOperation, support: MemorySupport | None, relation_complete: bool,
) -> KeptReason | None:
    # A Memory-level result is unresolved exactly when one of its assessments is.
    if support is None or not support.assessments or support.result.unresolved:
        return KeptReason.SUPPORT_UNRESOLVED
    if operation.action in _RETIRES_SUPPORT and not all(
        assessment.complete_read for assessment in support.assessments
    ):
        return KeptReason.READ_INCOMPLETE
    if operation.action in _REPLACES_MEMORY and not relation_complete:
        return KeptReason.RELATION_INCOMPLETE
    return None
