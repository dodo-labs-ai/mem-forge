"""Read Cross-Document Relations for a caller.

Storage returns the relations that are current for the caller. This module
decides how each one reads: an ``updates`` pair is ordered by the newest source
revision time of each Memory's current Support, and reads as ``contradicts``
when those times do not order it. Search, Memory detail and the relation view
all read relations here, so every surface shows the same label and order.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
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
from memforge.storage.adapters.protocols import ActiveMemorySupportState, parse_source_revision_time


class RelationReadStore(Protocol):
    async def list_cross_document_relations(
        self,
        memory_ids: Sequence[str],
        scope: AccessScope,
    ) -> Mapping[str, tuple[CurrentCrossDocumentRelation, ...]]: ...

    async def get_active_memory_support_states(
        self,
        memory_ids: Sequence[str],
    ) -> Mapping[str, ActiveMemorySupportState]: ...

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


@dataclass(frozen=True, slots=True)
class ReadableRelation:
    """A current relation with the label a reader sees."""

    relation: CurrentCrossDocumentRelation
    label: CrossDocumentRelationLabel
    # Set only when ``label`` is updates.
    newer_memory_id: str | None = None

    def role_of(self, memory_id: str) -> RelationRole:
        if self.newer_memory_id is None:
            return RelationRole.PEER
        return RelationRole.NEWER if memory_id == self.newer_memory_id else RelationRole.OLDER


def relation_reading(
    label: CrossDocumentRelationLabel,
    memory_low_id: str,
    memory_high_id: str,
    revision_at: Mapping[str, str | None],
) -> tuple[CrossDocumentRelationLabel, str | None]:
    """How a stored label reads, and the newer Memory when it reads as ``updates``."""

    if label is not CrossDocumentRelationLabel.UPDATES:
        return label, None
    low_time = parse_source_revision_time(revision_at.get(memory_low_id))
    high_time = parse_source_revision_time(revision_at.get(memory_high_id))
    if low_time is None or high_time is None or low_time == high_time:
        return CrossDocumentRelationLabel.CONTRADICTS, None
    return CrossDocumentRelationLabel.UPDATES, memory_low_id if low_time > high_time else memory_high_id


def readable_relation(
    relation: CurrentCrossDocumentRelation,
    revision_at: Mapping[str, str | None],
) -> ReadableRelation:
    label, newer_memory_id = relation_reading(
        relation.label,
        relation.memory_low_id,
        relation.memory_high_id,
        revision_at,
    )
    return ReadableRelation(relation=relation, label=label, newer_memory_id=newer_memory_id)


async def source_revision_times(
    store: RelationReadStore,
    memory_ids: Sequence[str],
) -> dict[str, str | None]:
    """The newest source revision time of each Memory's current Support."""

    ids = tuple(dict.fromkeys(memory_ids))
    if not ids:
        return {}
    support = await store.get_active_memory_support_states(ids)
    return {memory_id: state.latest_source_revision_at for memory_id, state in support.items()}


@dataclass(frozen=True, slots=True)
class RelationGraph:
    """The current relations of a set of Memories, as they read."""

    relations: Mapping[str, tuple[ReadableRelation, ...]]
    revision_at: Mapping[str, str | None]

    def relations_of(self, memory_id: str) -> tuple[ReadableRelation, ...]:
        return self.relations.get(memory_id, ())


async def load_relation_graph(
    store: RelationReadStore,
    memory_ids: Sequence[str],
    scope: AccessScope,
) -> RelationGraph:
    stored = await store.list_cross_document_relations(memory_ids, scope)
    involved = tuple(
        dict.fromkeys(
            memory_id
            for relations in stored.values()
            for relation in relations
            for memory_id in (relation.memory_low_id, relation.memory_high_id)
        )
    )
    revision_at = await source_revision_times(store, involved)
    return RelationGraph(
        relations={
            memory_id: tuple(readable_relation(relation, revision_at) for relation in relations)
            for memory_id, relations in stored.items()
        },
        revision_at=revision_at,
    )


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
            item.label is CrossDocumentRelationLabel.EQUIVALENT
            and item.relation.counterpart_of(memory_id) in kept_ids
            for item in graph.relations_of(memory_id)
        ):
            continue
        kept.append(memory_id)
        kept_ids.add(memory_id)
    rank = {memory_id: index for index, memory_id in enumerate(kept)}
    newer_first = {
        memory_id: sorted(
            (
                item.newer_memory_id
                for item in graph.relations_of(memory_id)
                if item.newer_memory_id is not None
                and item.newer_memory_id != memory_id
                and item.newer_memory_id in rank
            ),
            key=rank.__getitem__,
        )
        for memory_id in kept
    }
    # Source revision times strictly order every updates pair, so the
    # newer-than graph has no cycle and each Memory follows all newer ones.
    ordered: list[str] = []
    placed: set[str] = set()
    for memory_id in kept:
        pending = [memory_id]
        while pending:
            current = pending[-1]
            if current in placed:
                pending.pop()
                continue
            waiting = [newer_id for newer_id in newer_first[current] if newer_id not in placed]
            if waiting:
                pending.extend(reversed(waiting))
                continue
            pending.pop()
            placed.add(current)
            ordered.append(current)
    return ordered


async def related_memories(
    store: RelationReadStore,
    memory_ids: Sequence[str],
    scope: AccessScope,
    revision_at: Mapping[str, str | None],
) -> Mapping[str, RelatedMemory]:
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
            revision_at=revision_at.get(memory_id),
        )
        for memory_id, memory in memories.items()
    }


async def relation_contexts(
    store: RelationReadStore,
    graph: RelationGraph,
    memory_ids: Sequence[str],
    scope: AccessScope,
) -> Mapping[str, tuple[MemoryRelationContext, ...]]:
    counterparts = await related_memories(
        store,
        [
            item.relation.counterpart_of(memory_id)
            for memory_id in memory_ids
            for item in graph.relations_of(memory_id)
        ],
        scope,
        graph.revision_at,
    )
    return {
        memory_id: tuple(
            MemoryRelationContext(
                label=item.label.value,
                role=item.role_of(memory_id).value,
                counterpart=counterparts[item.relation.counterpart_of(memory_id)],
                reason=item.relation.reason,
                decided_by=item.relation.decided_by.value,
            )
            for item in graph.relations_of(memory_id)
            if item.relation.counterpart_of(memory_id) in counterparts
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
    revised = parse_source_revision_time(memory.revision_at)
    if revised is not None:
        text += f" (revised {revised.date().isoformat()})"
    return text


@dataclass(frozen=True, slots=True)
class DismissedRelationView:
    """The dismissals in force for one pair, read from one of its Memories so they can be undone.

    ``labels`` are the dismissed labels as they would read now; the newest
    dismissal gives the actor, time and note.
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
    revision_at = await source_revision_times(store, (memory_id, *by_counterpart))
    counterparts = await related_memories(store, tuple(by_counterpart), scope, revision_at)
    views: list[DismissedRelationView] = []
    for counterpart_id, pair_dismissals in by_counterpart.items():
        if counterpart_id not in counterparts:
            continue
        read_labels = {
            relation_reading(item.label, item.memory_low_id, item.memory_high_id, revision_at)[0]
            for item in pair_dismissals
        }
        newest = pair_dismissals[0]
        views.append(
            DismissedRelationView(
                counterpart=counterparts[counterpart_id],
                labels=tuple(label.value for label in RELATION_READ_ORDER if label in read_labels),
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


# The stored labels that can read as each label.
_STORED_LABELS_READ_AS = {
    CrossDocumentRelationLabel.EQUIVALENT: (CrossDocumentRelationLabel.EQUIVALENT,),
    CrossDocumentRelationLabel.UPDATES: (CrossDocumentRelationLabel.UPDATES,),
    CrossDocumentRelationLabel.CONTRADICTS: (
        CrossDocumentRelationLabel.CONTRADICTS,
        CrossDocumentRelationLabel.UPDATES,
    ),
}


async def list_relation_pairs(
    store: RelationViewStore,
    scope: AccessScope,
    *,
    label: CrossDocumentRelationLabel | None,
    limit: int,
    offset: int,
) -> tuple[tuple[RelationPairView, ...], int]:
    """One page of the relations current for the caller, and their total.

    ``label`` filters by the label a reader sees, so an ``updates`` pair that
    source revision times do not order is listed with the conflicts.
    """

    relations = await store.list_current_cross_document_relations(
        scope,
        labels=_STORED_LABELS_READ_AS[label] if label is not None else None,
    )
    revision_at = await source_revision_times(
        store,
        [
            memory_id
            for relation in relations
            if relation.label is CrossDocumentRelationLabel.UPDATES
            for memory_id in (relation.memory_low_id, relation.memory_high_id)
        ],
    )
    readable = [readable_relation(relation, revision_at) for relation in relations]
    matching = [item for item in readable if label is None or item.label is label]
    page = matching[offset : offset + limit]
    page_ids = [
        memory_id
        for item in page
        for memory_id in (item.relation.memory_low_id, item.relation.memory_high_id)
    ]
    memories = await related_memories(store, page_ids, scope, await source_revision_times(store, page_ids))
    views = tuple(
        RelationPairView(
            label=item.label.value,
            newer_memory_id=item.newer_memory_id,
            reason=item.relation.reason,
            decided_by=item.relation.decided_by.value,
            decided_at=item.relation.decided_at,
            memories=(memories[item.relation.memory_low_id], memories[item.relation.memory_high_id]),
        )
        for item in page
        if item.relation.memory_low_id in memories and item.relation.memory_high_id in memories
    )
    return views, len(matching)


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

    ``label`` is the label the caller saw. The dismissal records the stored
    label, so it keeps hiding the relation however its ``updates`` order reads.
    Raises LookupError when the caller sees no such relation and
    RelationDismissalConflict when it now reads differently or either content
    changed.
    """

    graph = await load_relation_graph(store, (memory_id,), scope)
    shown = next(
        (
            item
            for item in graph.relations_of(memory_id)
            if item.relation.counterpart_of(memory_id) == counterpart_memory_id
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
        label=shown.relation.label,
        expected_content_hash=expected_content_hash,
        counterpart_expected_content_hash=counterpart_expected_content_hash,
        actor=actor,
        note=note,
    )
