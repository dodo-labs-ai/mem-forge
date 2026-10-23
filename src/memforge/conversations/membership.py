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

A plan decides from the Units the layer read: :func:`units_to_read` names
the ones it needs by what the attempt returned, and a plan over a state
that lacks one of them is refused before anything is decided.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field

from memforge.conversations.coverage import ConversationCoverage
from memforge.conversations.port import ConversationLabel, ConversationRef, ThreadingModel
from memforge.conversations.records import MessageIdentity
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
    :func:`units_to_read` names for the attempt's messages, and every Unit in
    reach of a proven interval or of a new message. ``holders`` are the
    current Units the store names as holding each returned message, each root
    a returned message replies to and each thread root, by message id.
    ``threads`` names every current thread Unit of the conversation by its
    root; ``rooted_threads`` are the roots their own thread Unit held before
    this run (:func:`rooted_threads`).
    """

    conversation_id: str
    units: tuple[StoredUnit, ...]
    holders: Mapping[str, tuple[UnitRef, ...]]
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


def rooted_threads(*, threads: Mapping[str, UnitRef], holders: Mapping[str, tuple[UnitRef, ...]]) -> frozenset[str]:
    """The roots, of ``threads``, that their own thread Unit holds."""

    return frozenset(root_id for root_id, thread in threads.items() if thread in holders.get(root_id, ()))


def units_to_read(
    *,
    threads: Mapping[str, UnitRef],
    holders: Mapping[str, tuple[UnitRef, ...]],
    message_ids: Collection[str],
    replied_roots: Collection[str],
) -> tuple[UnitRef, ...]:
    """The Units a plan needs read with their members, named by what the attempt returned.

    ``message_ids`` are the messages the attempt returned, ``replied_roots``
    the roots they reply to, ``threads`` every current thread Unit by its root
    and ``holders`` the current Units holding each of those messages, roots and
    thread roots. A plan needs:

    - every Unit holding a returned message or a replied root, where a
      returned record replaces the held one and a held root is lent to its
      thread;
    - the thread Unit of every returned message or replied root that roots
      one, which takes its new replies;
    - every window holding a thread's root, which gives the root up once the
      thread holds it;
    - a thread Unit that lacks its root while a window holds it, which takes
      the root from that window.

    The layer reads exactly these Units over the manifest's messages, and
    adds every Unit whose recorded span meets a proven interval or the range
    a new message may join under the Window Policy, which a proven absence or
    a window assignment can change. A plan sees a message returned without
    its record by its identity only, not the root it replies to, so it checks
    the part of this set it can see.
    """

    wanted: dict[str, UnitRef] = {}
    for message_id in {*message_ids, *replied_roots}:
        wanted.update((unit.provider_key, unit) for unit in holders.get(message_id, ()))
        if message_id in threads:
            wanted[threads[message_id].provider_key] = threads[message_id]
    rooted = rooted_threads(threads=threads, holders=holders)
    for root_id, thread in threads.items():
        windows = [unit for unit in holders.get(root_id, ()) if unit.kind is UnitKind.WINDOW]
        wanted.update((unit.provider_key, unit) for unit in windows)
        if windows and root_id not in rooted:
            wanted[thread.provider_key] = thread
    return tuple(wanted.values())


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
    does not hold; each is carried as stored and never proven absent. The
    state must hold every Unit :func:`units_to_read` names for the returned
    messages; a plan over a state that lacks one is refused before anything
    is decided.
    """

    conversation_id = conversation.conversation_id
    if state.conversation_id != conversation_id or any(
        unit.ref.conversation_id != conversation_id for unit in state.units
    ):
        raise ValueError("conversation state holds a Unit of another conversation")
    returned_by_identity = {message.member.identity: message for message in returned}
    _require_read(state, returned_by_identity.values(), seen_without_record)

    working = _Membership(
        source_id=source_id,
        conversation=conversation,
        coverage=coverage,
        returned=returned_by_identity,
        state=state,
        seen_without_record=seen_without_record,
    )
    working.replace_returned()
    to_window = working.place_new()
    working.remove_proven_absent()
    working.withdraw_moved_roots()
    working.assign_windows(to_window, policy)
    return working.decide()


def _require_read(
    state: ConversationState,
    returned: Iterable[ReturnedMessage],
    seen_without_record: Collection[MessageIdentity],
) -> None:
    """Refuse a state that lacks a Unit the read set names for these messages."""

    returned = tuple(returned)
    needed = units_to_read(
        threads=state.threads,
        holders=state.holders,
        message_ids={
            *(message.member.identity.message_id for message in returned),
            *(identity.message_id for identity in seen_without_record),
        },
        replied_roots={message.thread_root_id for message in returned if message.thread_root_id is not None},
    )
    read = {unit.provider_key for unit in state.units}
    missing = sorted({unit.provider_key for unit in needed} - read)
    if missing:
        raise ValueError("; ".join(f"Unit {key} was not read for this plan" for key in missing))


class _Membership:
    """The members of every Unit one plan touches, decided one step at a time.

    ``members`` starts as the complete membership of every Unit read, by
    provider key; each step updates it, and :meth:`decide` turns it into the
    plan.
    """

    def __init__(
        self,
        *,
        source_id: str,
        conversation: ConversationRef,
        coverage: ConversationCoverage,
        returned: Mapping[MessageIdentity, ReturnedMessage],
        state: ConversationState,
        seen_without_record: Collection[MessageIdentity],
    ) -> None:
        self._source_id = source_id
        self._conversation = conversation
        self._coverage = coverage
        self._returned = returned
        self._seen_without_record = seen_without_record
        self._rooted_threads = state.rooted_threads
        self._threaded = conversation.threading is not ThreadingModel.FLAT
        self._stored = {unit.provider_key: unit for unit in state.units}
        self._members: dict[str, dict[MessageIdentity, Member]] = {
            key: {member.identity: member for member in unit.members} for key, unit in self._stored.items()
        }
        self._held_in: dict[MessageIdentity, list[str]] = {}
        for key, unit in self._stored.items():
            for member in unit.members:
                self._held_in.setdefault(member.identity, []).append(key)
        self._thread_keys = {root_id: ref.provider_key for root_id, ref in state.threads.items()}
        self._thread_keys.update(
            {unit.thread_root_id: key for key, unit in self._stored.items() if unit.thread_root_id}
        )
        self._opened: dict[str, tuple[UnitKind, str | None]] = {}
        self._proven_absent_from: set[str] = set()

    @property
    def _conversation_id(self) -> str:
        return self._conversation.conversation_id

    def replace_returned(self) -> None:
        """A returned message replaces its record wherever it is held."""

        for identity, message in self._returned.items():
            for key in self._held_in.get(identity, ()):
                self._members[key][identity] = message.member

    def place_new(self) -> list[Member]:
        """Place every returned message no Unit holds, and seat each root its thread lacks.

        A reply joins its thread, a root opens or keeps its own thread, and
        any other top-level message is returned for a window. A held root
        gaining its first replies, or held while its thread lacks it, joins
        its thread.
        """

        replied_roots = {
            message.thread_root_id
            for message in self._returned.values()
            if self._threaded and message.thread_root_id
        }
        to_window: list[Member] = []
        for identity, message in self._returned.items():
            if identity in self._held_in:
                continue
            if self._threaded and message.thread_root_id is not None:
                self._members[self.seat_root(message.thread_root_id)][identity] = message.member
            elif (
                self._conversation.threading is ThreadingModel.POST_REPLY
                or identity.message_id in replied_roots
                or identity.message_id in self._thread_keys
            ):
                self.seat_root(identity.message_id)
            else:
                to_window.append(message.member)
        stored_roots = {unit.thread_root_id for unit in self._stored.values() if unit.thread_root_id}
        for root_id in sorted(replied_roots | stored_roots):
            self.seat_root(root_id)
        return to_window

    def seat_root(self, root_id: str) -> str:
        """The thread Unit of a root, opened if none exists, holding its root once the root is known.

        A thread takes its root when the attempt returned it, or from a Unit
        holding it unless the root is proven absent.
        """

        key = self._thread_keys.get(root_id)
        if key is None:
            key = thread_unit_key(source_id=self._source_id, conversation_id=self._conversation_id, thread_root_id=root_id)
            self._thread_keys[root_id] = key
            self._opened[key] = (UnitKind.THREAD, root_id)
            self._members[key] = {}
        root = MessageIdentity(self._conversation_id, root_id)
        if root not in self._members[key]:
            if root in self._returned:
                self._members[key][root] = self._returned[root].member
            elif self._held_in.get(root):
                held = self._members[self._held_in[root][0]][root]
                if not self._proven_absent(root, held, is_reply=False):
                    self._members[key][root] = held
        return key

    def remove_proven_absent(self) -> None:
        """Remove every committed member a proven interval shows the provider no longer holds."""

        for key, unit in self._stored.items():
            for identity, member in list(self._members[key].items()):
                is_reply = unit.kind is UnitKind.THREAD and identity.message_id != unit.thread_root_id
                if self._proven_absent(identity, member, is_reply=is_reply):
                    del self._members[key][identity]
                    self._proven_absent_from.add(key)

    def withdraw_moved_roots(self) -> None:
        """Withdraw from its window every root whose own thread Unit held it before this run."""

        for key, unit in self._stored.items():
            if unit.kind is UnitKind.WINDOW:
                moved = [identity for identity in self._members[key] if identity.message_id in self._rooted_threads]
                for identity in moved:
                    del self._members[key][identity]

    def assign_windows(self, to_window: Iterable[Member], policy: WindowPolicy) -> None:
        """Place new top-level messages in windows under the Window Policy, opening windows as it says."""

        to_window = tuple(to_window)
        window_spans = {
            key: span
            for key, unit in self._stored.items()
            if unit.kind is UnitKind.WINDOW and (span := Span.of(list(self._members[key].values()))) is not None
        }
        assignment = assign_to_windows(
            window_spans,
            ((member.identity, member.created_at) for member in to_window),
            open_window=lambda identity: window_unit_key(
                source_id=self._source_id, conversation_id=self._conversation_id, anchor_message_id=identity.message_id
            ),
            policy=policy,
        )
        new_members = {member.identity: member for member in to_window}
        for identity, key in assignment.window_of.items():
            if key not in self._members:
                self._opened[key] = (UnitKind.WINDOW, None)
                self._members[key] = {}
            self._members[key][identity] = new_members[identity]

    def decide(self) -> ConversationPlan:
        """Project every Unit whose membership changed or that this run opens, and remove every Unit left empty.

        An empty Unit is removed as absent when a member was proven absent
        from it in this run, otherwise as emptied.
        """

        planned: list[PlannedUnit] = []
        removals: list[PlannedRemoval] = []
        label = self._conversation.label
        for key in sorted(self._members):
            decided = tuple(
                sorted(self._members[key].values(), key=lambda member: (member.created_at, member.identity.message_id))
            )
            unit = self._stored.get(key)
            if unit is None:
                kind, root_id = self._opened[key]
                if decided:
                    planned.append(PlannedUnit(kind, key, label, decided, thread_root_id=root_id))
                continue
            if not decided:
                reason = (
                    UnitRemovalReason.ABSENT_FROM_SOURCE if key in self._proven_absent_from else UnitRemovalReason.EMPTIED
                )
                removals.append(PlannedRemoval(unit=unit.ref, reason=reason))
                continue
            if _content(decided) != _content(unit.members):
                planned.append(PlannedUnit(unit.kind, key, label, decided, thread_root_id=unit.thread_root_id, stored=unit))
        return ConversationPlan(units=tuple(planned), removals=tuple(removals))

    def _proven_absent(self, identity: MessageIdentity, member: Member, *, is_reply: bool) -> bool:
        """Whether a committed member is proven absent: not returned, dated, and inside a proven interval."""

        if identity in self._returned or identity in self._seen_without_record or member.undated:
            return False
        return self._coverage.proves_absent(member.created_at, is_reply=is_reply)


def _content(members: Iterable[Member]) -> Mapping[MessageIdentity, object]:
    return {member.identity: member.content_identity for member in members}
