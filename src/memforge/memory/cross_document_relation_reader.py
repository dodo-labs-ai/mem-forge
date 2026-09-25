"""Read Cross-Document Relations for a caller.

Storage returns the relations that are current for the caller, each with the
label it was recorded with and the Evidence time of both Memories it was
decided on. An ``updates`` pair is ordered by those times. Search, Memory
detail and the relation view all read relations here, so every surface shows
the same label, order and dates.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

from memforge.memory.cross_document_relation import (
    RELATION_READ_ORDER,
    CrossDocumentRelationDismissal,
    CrossDocumentRelationLabel,
    CurrentCrossDocumentRelation,
    RelationDismissalConflict,
)
from memforge.models import Memory, MemoryRelationContext, MemorySourceRef, RelatedMemory
from memforge.storage.adapters.context import AccessScope


class RelationReadStore(Protocol):
    async def list_cross_document_relations(
        self,
        memory_ids: Sequence[str],
        scope: AccessScope,
    ) -> Mapping[str, tuple[CurrentCrossDocumentRelation, ...]]: ...

    async def list_memories_by_ids(self, memory_ids: Sequence[str]) -> list[Memory]: ...

    async def get_memory_source_refs_many(
        self,
        memory_ids: Sequence[str],
        scope: AccessScope,
    ) -> Mapping[str, tuple[MemorySourceRef, ...]]: ...


class RelationViewStore(RelationReadStore, Protocol):
    async def list_current_cross_document_relations(
        self,
        scope: AccessScope,
        *,
        labels: Collection[CrossDocumentRelationLabel] | None = None,
    ) -> tuple[CurrentCrossDocumentRelation, ...]: ...


class RelationDismissalStore(RelationReadStore, Protocol):
    async def record_cross_document_relation_dismissal(
        self,
        *,
        memory_id: str,
        counterpart_memory_id: str,
        label: CrossDocumentRelationLabel,
        expected_content_hash: str,
        counterpart_expected_content_hash: str,
        actor: str,
        note: str | None = None,
    ) -> CrossDocumentRelationDismissal: ...


class RelationRole(str, Enum):
    """A Memory's side of a relation."""

    NEWER = "newer"
    OLDER = "older"
    PEER = "peer"


def relation_role(relation: CurrentCrossDocumentRelation, memory_id: str) -> RelationRole:
    newer_memory_id = relation.newer_memory_id
    if newer_memory_id is None:
        return RelationRole.PEER
    return RelationRole.NEWER if memory_id == newer_memory_id else RelationRole.OLDER


@dataclass(frozen=True, slots=True)
class RelationGraph:
    """The current relations of a set of Memories."""

    relations: Mapping[str, tuple[CurrentCrossDocumentRelation, ...]]

    def relations_of(self, memory_id: str) -> tuple[CurrentCrossDocumentRelation, ...]:
        return self.relations.get(memory_id, ())


async def load_relation_graph(
    store: RelationReadStore,
    memory_ids: Sequence[str],
    scope: AccessScope,
) -> RelationGraph:
    return RelationGraph(relations=await store.list_cross_document_relations(memory_ids, scope))


def order_by_relations(memory_ids: Sequence[str], graph: RelationGraph) -> list[str]:
    """Apply relations to one ranked window before it is paged.

    A Memory equivalent to a higher-ranked kept Memory is left out; the kept
    one names it through its relations. Within the window, the newer Memory
    of an ``updates`` pair is placed directly ahead of the older one.
    """

    kept: list[str] = []
    kept_ids: set[str] = set()
    for memory_id in memory_ids:
        if any(
            relation.label is CrossDocumentRelationLabel.EQUIVALENT
            and relation.counterpart_of(memory_id) in kept_ids
            for relation in graph.relations_of(memory_id)
        ):
            continue
        kept.append(memory_id)
        kept_ids.add(memory_id)
    rank = {memory_id: index for index, memory_id in enumerate(kept)}
    newer_first = {
        memory_id: sorted(
            (
                relation.newer_memory_id
                for relation in graph.relations_of(memory_id)
                if relation.newer_memory_id is not None
                and relation.newer_memory_id != memory_id
                and relation.newer_memory_id in rank
            ),
            key=rank.__getitem__,
        )
        for memory_id in kept
    }
    # Each relation keeps the Evidence times it was decided on, so relations
    # decided at different times can form a newer-than cycle. A Memory already
    # being placed is not waited for again, so each Memory is placed once.
    ordered: list[str] = []
    placed: set[str] = set()
    for memory_id in kept:
        if memory_id in placed:
            continue
        entered = {memory_id}
        stack = [(memory_id, iter(newer_first[memory_id]))]
        while stack:
            current, newer = stack[-1]
            waiting = next((newer_id for newer_id in newer if newer_id not in placed and newer_id not in entered), None)
            if waiting is None:
                stack.pop()
                placed.add(current)
                ordered.append(current)
                continue
            entered.add(waiting)
            stack.append((waiting, iter(newer_first[waiting])))
    return ordered


async def related_memories(
    store: RelationReadStore,
    memory_ids: Sequence[str],
    scope: AccessScope,
) -> Mapping[str, RelatedMemory]:
    """The caller's view of each Memory, without a date; each relation dates its own Memories."""

    ids = tuple(dict.fromkeys(memory_ids))
    if not ids:
        return {}
    memories = {memory.id: memory for memory in await store.list_memories_by_ids(ids)}
    sources = await store.get_memory_source_refs_many(ids, scope)
    return {
        memory_id: RelatedMemory(
            memory_id=memory_id,
            summary=memory.content,
            content_hash=memory.content_hash,
            sources=sources.get(memory_id, ()),
        )
        for memory_id, memory in memories.items()
    }


def _dated(memory: RelatedMemory, relation: CurrentCrossDocumentRelation) -> RelatedMemory:
    return replace(memory, evidence_time=relation.evidence_time_of(memory.memory_id))


async def relation_contexts(
    store: RelationReadStore,
    graph: RelationGraph,
    memory_ids: Sequence[str],
    scope: AccessScope,
) -> Mapping[str, tuple[MemoryRelationContext, ...]]:
    counterparts = await related_memories(
        store,
        [
            relation.counterpart_of(memory_id)
            for memory_id in memory_ids
            for relation in graph.relations_of(memory_id)
        ],
        scope,
    )
    return {
        memory_id: tuple(
            MemoryRelationContext(
                label=relation.label.value,
                role=relation_role(relation, memory_id).value,
                counterpart=_dated(counterparts[relation.counterpart_of(memory_id)], relation),
                reason=relation.reason,
                decided_by=relation.decided_by.value,
            )
            for relation in graph.relations_of(memory_id)
            if relation.counterpart_of(memory_id) in counterparts
        )
        for memory_id in memory_ids
    }


async def read_memory_relations(
    store: RelationReadStore,
    memory_ids: Sequence[str],
    scope: AccessScope,
) -> Mapping[str, tuple[MemoryRelationContext, ...]]:
    graph = await load_relation_graph(store, memory_ids, scope)
    return await relation_contexts(store, graph, memory_ids, scope)


def relation_notice(contexts: Sequence[MemoryRelationContext]) -> str | None:
    """One sentence per relation a reader must weigh; equivalent and newer need none."""

    notes: list[str] = []
    for context in contexts:
        if context.label == CrossDocumentRelationLabel.CONTRADICTS.value:
            notes.append(f"Conflicts with {_describe(context.counterpart)}.")
        elif context.label == CrossDocumentRelationLabel.UPDATES.value and context.role == RelationRole.OLDER.value:
            notes.append(f"Updated by the newer {_describe(context.counterpart)}.")
    return " ".join(notes) or None


def _describe(memory: RelatedMemory) -> str:
    sources = ", ".join(source.name or source.source_type for source in memory.sources)
    text = f"Memory {memory.memory_id}"
    if sources:
        text += f" from {sources}"
    if memory.evidence_time:
        text += f" (recorded {memory.evidence_time})"
    return text


@dataclass(frozen=True, slots=True)
class DismissedRelationView:
    """The dismissals in force for one pair, read from one of its Memories so they can be undone.

    ``labels`` are the dismissed labels; the newest dismissal gives the actor,
    time and note.
    """

    counterpart: RelatedMemory
    labels: tuple[str, ...]
    dismissed_by: str
    dismissed_at: str
    note: str | None


async def dismissed_relation_views(
    store: RelationReadStore,
    memory_id: str,
    dismissals: Sequence[CrossDocumentRelationDismissal],
    scope: AccessScope,
) -> tuple[DismissedRelationView, ...]:
    """Group one Memory's dismissals in force by pair, newest dismissal first."""

    if not dismissals:
        return ()
    by_counterpart: dict[str, list[CrossDocumentRelationDismissal]] = {}
    for dismissal in sorted(dismissals, key=lambda item: (item.dismissed_at, item.id), reverse=True):
        by_counterpart.setdefault(dismissal.counterpart_of(memory_id), []).append(dismissal)
    counterparts = await related_memories(store, tuple(by_counterpart), scope)
    views: list[DismissedRelationView] = []
    for counterpart_id, pair_dismissals in by_counterpart.items():
        if counterpart_id not in counterparts:
            continue
        dismissed_labels = {item.label for item in pair_dismissals}
        newest = pair_dismissals[0]
        views.append(
            DismissedRelationView(
                counterpart=counterparts[counterpart_id],
                labels=tuple(label.value for label in RELATION_READ_ORDER if label in dismissed_labels),
                dismissed_by=newest.dismissed_by,
                dismissed_at=newest.dismissed_at,
                note=newest.note,
            )
        )
    return tuple(views)


@dataclass(frozen=True, slots=True)
class RelationPairView:
    """One current relation with both Memories, for the relation view."""

    label: str
    newer_memory_id: str | None
    reason: str
    decided_by: str
    decided_at: str
    memories: tuple[RelatedMemory, RelatedMemory]


async def list_relation_pairs(
    store: RelationViewStore,
    scope: AccessScope,
    *,
    label: CrossDocumentRelationLabel | None,
    limit: int,
    offset: int,
) -> tuple[tuple[RelationPairView, ...], int]:
    """One page of the relations current for the caller, and their total."""

    relations = await store.list_current_cross_document_relations(
        scope,
        labels=(label,) if label is not None else None,
    )
    page = relations[offset : offset + limit]
    memories = await related_memories(
        store,
        [memory_id for relation in page for memory_id in (relation.memory_low_id, relation.memory_high_id)],
        scope,
    )
    views = tuple(
        RelationPairView(
            label=relation.label.value,
            newer_memory_id=relation.newer_memory_id,
            reason=relation.reason,
            decided_by=relation.decided_by.value,
            decided_at=relation.decided_at,
            memories=(
                _dated(memories[relation.memory_low_id], relation),
                _dated(memories[relation.memory_high_id], relation),
            ),
        )
        for relation in page
        if relation.memory_low_id in memories and relation.memory_high_id in memories
    )
    return views, len(relations)


async def dismiss_relation(
    store: RelationDismissalStore,
    *,
    memory_id: str,
    counterpart_memory_id: str,
    label: CrossDocumentRelationLabel,
    expected_content_hash: str,
    counterpart_expected_content_hash: str,
    actor: str,
    scope: AccessScope,
    note: str | None = None,
) -> CrossDocumentRelationDismissal:
    """Dismiss the relation the caller sees between two Memories.

    ``label`` is the label the caller saw. Raises LookupError when the caller
    sees no such relation and RelationDismissalConflict when it now carries
    another label or either content changed.
    """

    graph = await load_relation_graph(store, (memory_id,), scope)
    shown = next(
        (
            relation
            for relation in graph.relations_of(memory_id)
            if relation.counterpart_of(memory_id) == counterpart_memory_id
        ),
        None,
    )
    if shown is None:
        raise LookupError("relation not found")
    if shown.label is not label:
        raise RelationDismissalConflict("relation_label_changed")
    return await store.record_cross_document_relation_dismissal(
        memory_id=memory_id,
        counterpart_memory_id=counterpart_memory_id,
        label=label,
        expected_content_hash=expected_content_hash,
        counterpart_expected_content_hash=counterpart_expected_content_hash,
        actor=actor,
        note=note,
    )
