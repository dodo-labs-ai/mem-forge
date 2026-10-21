"""Deciding the members of every Unit of one conversation, per message.

The conversation layer decides each Unit's whole membership and projects the
Unit as a complete snapshot of it. A committed member keeps its place unless
its own evidence removes it:

- **proven absent**: the attempt did not return it although a proven
  interval covers its creation time
  (:class:`~memforge.conversations.coverage.ConversationCoverage`). A member
  stored without its own creation time is never proven absent
  (:mod:`~memforge.conversations.undated`);
- **moved**: it is a window member and the root of a thread Unit that held
  it before this run. A root joins its thread Unit in one run and leaves its
  window in the next, once the thread holding it is current, so it is never
  in neither Unit.

A returned message replaces its member record; a member the attempt neither
returned nor proved absent is carried with its stored record, so its
revision stays the same. A message no Unit holds is new: a reply joins its
thread, a root its own thread, and any other top-level message a window under
the Window Policy. A thread Unit that lacks its root takes the root as soon
as the root is returned or held by a window. A Unit left without members is
removed, as absent when any member was proven absent in this run and
otherwise as emptied.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field

from memforge.conversations.coverage import ConversationCoverage
from memforge.conversations.port import ConversationLabel, ConversationRef, ThreadingModel
from memforge.conversations.records import MessageIdentity
from memforge.conversations.undated import is_undated
from memforge.conversations.units import (
    Member,
    Span,
    StoredUnit,
    UnitKind,
    UnitRef,
    thread_unit_key,
    window_unit_key,
)
from memforge.conversations.windows import GAP_WINDOW_V1, WindowPolicy, assign_to_windows
from memforge.source_projection import UnitRemovalReason


@dataclass(frozen=True, slots=True)
class ReturnedMessage:
    """A message this attempt returned, with its record and the thread it replies in."""

    member: Member
    thread_root_id: str | None = None


@dataclass(frozen=True, slots=True)
class ConversationState:
    """What the Source Projection holds of one conversation that this attempt may touch.

    ``units`` are the Units read with their complete membership: every Unit
    that holds a returned message, the thread of every returned reply or
    returned root, every window that holds a thread's root with that thread,
    and every Unit in reach of a proven interval or of a new message.
    ``threads`` names every current thread Unit of the conversation by its
    root; ``rooted_threads`` are the roots their own thread Unit held before
    this run.
    """

    conversation_id: str
    units: tuple[StoredUnit, ...]
    threads: Mapping[str, UnitRef] = field(default_factory=dict)
    rooted_threads: frozenset[str] = frozenset()


@dataclass(frozen=True, slots=True)
class PlannedUnit:
    """One Unit to project with its decided membership, in creation order.

    ``stored`` is the Unit as committed, ``None`` for a Unit this run opens.
    """

    kind: UnitKind
    provider_key: str
    conversation: ConversationLabel
    members: tuple[Member, ...]
    thread_root_id: str | None = None
    stored: StoredUnit | None = None

    @property
    def span(self) -> Span:
        span = Span.of(self.members)
        if span is None:
            raise ValueError("a planned Unit has members")
        return span


@dataclass(frozen=True, slots=True)
class PlannedRemoval:
    unit: UnitRef
    reason: UnitRemovalReason


@dataclass(frozen=True, slots=True)
class ConversationPlan:
    units: tuple[PlannedUnit, ...]
    removals: tuple[PlannedRemoval, ...]


def plan_conversation(
    *,
    source_id: str,
    conversation: ConversationRef,
    coverage: ConversationCoverage,
    returned: Iterable[ReturnedMessage],
    state: ConversationState,
    policy: WindowPolicy = GAP_WINDOW_V1,
    seen_without_record: Collection[MessageIdentity] = (),
) -> ConversationPlan:
    """Decide the membership of every Unit of one conversation this attempt touches.

    ``seen_without_record`` are messages the attempt returned whose record it
    does not hold; each is carried as stored and never proven absent.
    """

    conversation_id = conversation.conversation_id
    if state.conversation_id != conversation_id or any(
        unit.ref.conversation_id != conversation_id for unit in state.units
    ):
        raise ValueError("conversation state holds a Unit of another conversation")
    returned_by_identity = {message.member.identity: message for message in returned}
    threaded = conversation.threading is not ThreadingModel.FLAT

    stored = {unit.provider_key: unit for unit in state.units}
    members: dict[str, dict[MessageIdentity, Member]] = {
        key: {member.identity: member for member in unit.members} for key, unit in stored.items()
    }
    holders: dict[MessageIdentity, list[str]] = {}
    for key, unit in stored.items():
        for member in unit.members:
            holders.setdefault(member.identity, []).append(key)
    thread_keys = {root_id: ref.provider_key for root_id, ref in state.threads.items()}
    thread_keys.update({unit.thread_root_id: key for key, unit in stored.items() if unit.thread_root_id})

    # A returned message replaces its record wherever it is held.
    for identity, message in returned_by_identity.items():
        for key in holders.get(identity, ()):
            members[key][identity] = message.member

    opened: dict[str, tuple[UnitKind, str | None]] = {}

    def thread_for(root_id: str) -> str:
        """The thread Unit of a root, opened if none exists, holding its root once the root is known."""

        key = thread_keys.get(root_id)
        if key is None:
            key = thread_unit_key(source_id=source_id, conversation_id=conversation_id, thread_root_id=root_id)
            thread_keys[root_id] = key
            opened[key] = (UnitKind.THREAD, root_id)
            members[key] = {}
        elif key not in members:
            raise ValueError(f"the thread of root {root_id} was not read for this plan")
        root = MessageIdentity(conversation_id, root_id)
        if root not in members[key]:
            if root in returned_by_identity:
                members[key][root] = returned_by_identity[root].member
            elif holders.get(root):
                held = members[holders[root][0]][root]
                if (
                    root in seen_without_record
                    or is_undated(held.record)
                    or not coverage.proves_absent(held.created_at, is_reply=False)
                ):
                    members[key][root] = held
        return key

    replied_roots = {
        message.thread_root_id for message in returned_by_identity.values() if threaded and message.thread_root_id
    }
    to_window: list[Member] = []
    for identity, message in returned_by_identity.items():
        if identity in holders:
            continue
        if threaded and message.thread_root_id is not None:
            members[thread_for(message.thread_root_id)][identity] = message.member
        elif (
            conversation.threading is ThreadingModel.POST_REPLY
            or identity.message_id in replied_roots
            or identity.message_id in thread_keys
        ):
            thread_for(identity.message_id)
        else:
            to_window.append(message.member)
    # A held root gaining its first replies, or held while its thread lacks it, joins its thread.
    for root_id in sorted(replied_roots | {unit.thread_root_id for unit in stored.values() if unit.thread_root_id}):
        thread_for(root_id)

    proven_absent: set[str] = set()
    for key, unit in stored.items():
        for identity, member in list(members[key].items()):
            if identity in returned_by_identity or identity in seen_without_record or is_undated(member.record):
                continue
            is_reply = unit.kind is UnitKind.THREAD and identity.message_id != unit.thread_root_id
            if coverage.proves_absent(member.created_at, is_reply=is_reply):
                del members[key][identity]
                proven_absent.add(key)
        if unit.kind is UnitKind.WINDOW:
            for identity in [identity for identity in members[key] if identity.message_id in state.rooted_threads]:
                del members[key][identity]

    window_spans = {
        key: span
        for key, unit in stored.items()
        if unit.kind is UnitKind.WINDOW and (span := Span.of(list(members[key].values()))) is not None
    }
    assignment = assign_to_windows(
        window_spans,
        ((member.identity, member.created_at) for member in to_window),
        open_window=lambda identity: window_unit_key(
            source_id=source_id, conversation_id=conversation_id, anchor_message_id=identity.message_id
        ),
        policy=policy,
    )
    new_members = {member.identity: member for member in to_window}
    for identity, key in assignment.window_of.items():
        if key not in members:
            opened[key] = (UnitKind.WINDOW, None)
            members[key] = {}
        members[key][identity] = new_members[identity]

    planned: list[PlannedUnit] = []
    removals: list[PlannedRemoval] = []
    for key in sorted(members):
        decided = tuple(sorted(members[key].values(), key=lambda member: (member.created_at, member.identity.message_id)))
        unit = stored.get(key)
        if unit is None:
            kind, root_id = opened[key]
            if decided:
                planned.append(PlannedUnit(kind, key, conversation.label, decided, thread_root_id=root_id))
            continue
        if not decided:
            reason = UnitRemovalReason.ABSENT_FROM_SOURCE if key in proven_absent else UnitRemovalReason.EMPTIED
            removals.append(PlannedRemoval(unit=unit.ref, reason=reason))
            continue
        if _content(decided) != _content(unit.members):
            planned.append(
                PlannedUnit(
                    unit.kind, key, conversation.label, decided, thread_root_id=unit.thread_root_id, stored=unit
                )
            )
    return ConversationPlan(units=tuple(planned), removals=tuple(removals))


def _content(members: Iterable[Member]) -> Mapping[MessageIdentity, object]:
    return {member.identity: member.content_identity for member in members}
