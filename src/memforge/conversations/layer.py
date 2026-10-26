"""The conversation layer: every conversation rule, on the server, for every provider.

It builds each collection request from the current Source Projection and
decides from the returned Collection Manifest. It never calls a provider, and
an executor never reads the projection.

Per conversation recorded in the manifest:

- **readable**: decide the membership of every Unit the attempt touches
  (:func:`~memforge.conversations.membership.plan_conversation`), then apply
  Rolling Retention with this attempt's evidence;
- **gone**: remove every Unit as absent from the source;
- **no access**: keep every Unit, except under Rolling Retention, and report
  the conversation for the user to restore access or deselect it.

A conversation the manifest does not record has no status in this attempt and
is left as it is. Conversations the user deselected are removed on that
intent alone.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from memforge.conversations.coverage import JUST_AFTER, ProvenInterval, conversation_coverage, proven_interval
from memforge.conversations.manifest import (
    CollectionManifest,
    CollectionRequest,
    ConversationEvidence,
    ManifestMessage,
    PollEvidence,
    RequestedConversation,
)
from memforge.conversations.membership import (
    ConversationState,
    PlannedRemoval,
    PlannedUnit,
    ReturnedMessage,
    plan_conversation,
    rooted_threads,
    units_to_read,
)
from memforge.conversations.port import ConversationAccess, ConversationRef, Interval, PollOutcome
from memforge.conversations.reader import ConversationUnitReader
from memforge.conversations.records import MessageIdentity, record_revision
from memforge.conversations.time_scope import MAX_INITIAL_HISTORY_DAYS
from memforge.conversations.units import Member, MemberArtifact, Span
from memforge.conversations.windows import GAP_WINDOW_V1, WindowPolicy
from memforge.source_projection import UnitRemovalReason


@dataclass(frozen=True, slots=True)
class ConversationSelection:
    """What the Source's configuration says about collection.

    Initial History is between one day and
    :data:`~memforge.conversations.time_scope.MAX_INITIAL_HISTORY_DAYS`, and
    no longer than a finite Rolling Retention. ``regular_ceiling`` bounds a
    regular poll; ``None`` polls the whole of Initial History on every run.
    """

    conversation_ids: tuple[str, ...]
    initial_history_days: int
    rolling_retention_days: int | None = None
    regular_ceiling: timedelta | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.initial_history_days <= MAX_INITIAL_HISTORY_DAYS:
            raise ValueError(f"initial history must be between 1 and {MAX_INITIAL_HISTORY_DAYS} days")
        if self.rolling_retention_days and self.initial_history_days > self.rolling_retention_days:
            raise ValueError("initial history cannot exceed rolling retention")


@dataclass(frozen=True, slots=True)
class SourcePlan:
    """What one collection attempt changes: Units to project, Units to remove, and conversations to report."""

    units: tuple[PlannedUnit, ...]
    removals: tuple[PlannedRemoval, ...]
    unreadable: tuple[ConversationEvidence, ...]


def retention_cutoff(attempt_started_at: datetime, rolling_retention_days: int | None) -> datetime | None:
    """The Rolling Retention cutoff of one collection attempt, computed once from server time."""

    if not rolling_retention_days:
        return None
    if rolling_retention_days < 0:
        raise ValueError("rolling retention days cannot be negative")
    return attempt_started_at - timedelta(days=rolling_retention_days)


class ConversationLayer:
    def __init__(self, reader: ConversationUnitReader, *, policy: WindowPolicy = GAP_WINDOW_V1) -> None:
        self._reader = reader
        self._policy = policy

    @staticmethod
    def collection_request(
        *,
        selection: ConversationSelection,
        now: datetime,
        previous: CollectionManifest | None = None,
    ) -> CollectionRequest:
        """The conversations and intervals one attempt collects.

        The regular poll starts ``min(initial history, regular ceiling)`` ago,
        never before the retention cutoff, and has no end. At most two
        backfills follow it, over what the previous attempt left unread after
        the history floor, ``max(now − initial history, retention cutoff)``
        (:func:`_left_unread`): one for the lowest unread range, and one
        spanning every other, so a request holds at most three intervals per
        conversation. Each conversation carries the label the previous attempt
        recorded for it.

        The previous attempt is the manifest the Source received last, which
        may be a failed run or a dry run; a range only a dry run read counts
        as read.
        """

        initial_history = timedelta(days=selection.initial_history_days)
        reach = initial_history
        if selection.regular_ceiling is not None:
            reach = min(reach, selection.regular_ceiling)
        cutoff = retention_cutoff(now, selection.rolling_retention_days)
        floor = now - initial_history if cutoff is None else max(now - initial_history, cutoff)
        regular_start = now - reach if cutoff is None else max(now - reach, cutoff)
        conversations = []
        for conversation_id in selection.conversation_ids:
            evidence = previous.conversation(conversation_id) if previous is not None else None
            unread = (
                _left_unread(evidence, floor=floor, regular_start=regular_start, received_at=previous.received_at)
                if previous is not None and evidence is not None
                else []
            )
            intervals = [Interval(start=regular_start)]
            if unread:
                intervals.append(Interval(start=unread[0][0], end=unread[0][1]))
            if len(unread) > 1:
                intervals.append(Interval(start=unread[1][0], end=unread[-1][1]))
            conversations.append(
                RequestedConversation(
                    conversation_id=conversation_id,
                    intervals=tuple(intervals),
                    last_known=evidence.label if evidence is not None else None,
                )
            )
        return CollectionRequest(conversations=tuple(conversations))

    async def plan(
        self,
        *,
        manifest: CollectionManifest,
        bodies: Mapping[MessageIdentity, Mapping[str, object]],
        selection: ConversationSelection,
        attempt_started_at: datetime,
        deselected_conversation_ids: Collection[str] = (),
        uploaded_artifacts: Mapping[MessageIdentity, tuple[MemberArtifact, ...]] | None = None,
    ) -> SourcePlan:
        """Decide everything one collection attempt changes, from its manifest alone.

        ``bodies`` and ``uploaded_artifacts`` are what the attempt collected
        that the Source did not hold; the rest comes from the current
        Units holding each message.
        """

        source_id = manifest.source_id
        cutoff = retention_cutoff(attempt_started_at, selection.rolling_retention_days)
        selected = set(selection.conversation_ids)
        units: list[PlannedUnit] = []
        removals: list[PlannedRemoval] = []
        unreadable: list[ConversationEvidence] = []
        for evidence in manifest.conversations:
            conversation_id = evidence.conversation_id
            if conversation_id not in selected:
                raise ValueError(f"manifest records conversation {conversation_id} outside the selection")
            access = evidence.status.access
            if access is ConversationAccess.GONE:
                removals.extend(
                    PlannedRemoval(unit=unit, reason=UnitRemovalReason.ABSENT_FROM_SOURCE)
                    for unit in await self._reader.units_of(source_id, conversation_id)
                )
                continue
            planned: Sequence[PlannedUnit] = ()
            decided: Sequence[PlannedRemoval] = ()
            if access is ConversationAccess.NO_ACCESS:
                unreadable.append(evidence)
            else:
                planned, decided = await self._plan_readable(manifest, evidence, bodies, uploaded_artifacts or {})
            if cutoff is not None:
                planned, decided = await self._retain(source_id, conversation_id, cutoff, planned, decided)
            units.extend(planned)
            removals.extend(decided)
        for conversation_id in sorted(set(deselected_conversation_ids)):
            if conversation_id in selected:
                raise ValueError(f"conversation {conversation_id} is both selected and deselected")
            removals.extend(
                PlannedRemoval(unit=unit, reason=UnitRemovalReason.REMOVED_FROM_SCOPE)
                for unit in await self._reader.units_of(source_id, conversation_id)
            )
        return SourcePlan(units=tuple(units), removals=_once(removals), unreadable=tuple(unreadable))

    async def _plan_readable(
        self,
        manifest: CollectionManifest,
        evidence: ConversationEvidence,
        bodies: Mapping[MessageIdentity, Mapping[str, object]],
        uploaded_artifacts: Mapping[MessageIdentity, tuple[MemberArtifact, ...]],
    ) -> tuple[tuple[PlannedUnit, ...], tuple[PlannedRemoval, ...]]:
        if evidence.ref is None:
            raise ValueError(f"readable conversation {evidence.conversation_id} has no reference")
        coverage = conversation_coverage(evidence.ref, evidence.polls, received_at=manifest.received_at)
        messages = manifest.messages_of(evidence.conversation_id)
        state = await self._state(manifest.source_id, evidence.conversation_id, coverage.intervals, messages)
        held = {member.identity: member for unit in state.units for member in unit.members}
        returned = {message.identity: _returned(message, bodies, uploaded_artifacts, held) for message in messages}
        plan = plan_conversation(
            source_id=manifest.source_id,
            conversation=evidence.ref,
            coverage=coverage,
            returned=tuple(message for message in returned.values() if message is not None),
            state=state,
            policy=self._policy,
            seen_without_record=frozenset(identity for identity, message in returned.items() if message is None),
        )
        return plan.units, plan.removals

    async def _state(
        self,
        source_id: str,
        conversation_id: str,
        intervals: Sequence[ProvenInterval],
        messages: Sequence[ManifestMessage],
    ) -> ConversationState:
        """Read the Units of one conversation this attempt may change, with their members.

        It reads exactly the Units
        :func:`~memforge.conversations.membership.units_to_read` names for the
        manifest's messages, then every Unit in this attempt's reach
        (:meth:`_reach`).
        """

        threads = {
            unit.thread_root_id: unit
            for unit in await self._reader.thread_units(source_id, conversation_id)
            if unit.thread_root_id is not None
        }
        message_ids = {message.identity.message_id for message in messages}
        replied_roots = {message.thread_root_id for message in messages if message.thread_root_id is not None}
        holders = await self._reader.holders(source_id, conversation_id, message_ids | replied_roots | set(threads))
        wanted = units_to_read(threads=threads, holders=holders, message_ids=message_ids, replied_roots=replied_roots)
        units = {unit.provider_key: unit for unit in await self._reader.read(wanted)}
        for start, end in self._reach(intervals, messages):
            for unit in await self._reader.units_spanning(source_id, conversation_id, start, end):
                units.setdefault(unit.provider_key, unit)
        return ConversationState(
            conversation_id=conversation_id,
            units=tuple(units.values()),
            holders=holders,
            threads=threads,
            rooted_threads=rooted_threads(threads=threads, holders=holders),
        )

    def _reach(
        self, intervals: Sequence[ProvenInterval], messages: Sequence[ManifestMessage]
    ) -> list[tuple[datetime, datetime]]:
        """The ranges whose Units this attempt may change: proven intervals, and windows new messages may join."""

        reach = [(interval.start, interval.end) for interval in intervals if not interval.empty]
        if messages:
            times = [message.created_at for message in messages]
            reach.append((min(times) - self._policy.max_span, max(times) + self._policy.max_span))
        return reach

    async def _retain(
        self,
        source_id: str,
        conversation_id: str,
        cutoff: datetime,
        planned: Sequence[PlannedUnit],
        decided: Sequence[PlannedRemoval],
    ) -> tuple[list[PlannedUnit], list[PlannedRemoval]]:
        """Apply Rolling Retention to one conversation: a Unit whose latest member is older than the cutoff goes.

        The decision reads each Unit's members, not its recorded span.
        """

        removed = {removal.unit.provider_key for removal in decided}
        kept: list[PlannedUnit] = []
        removals = list(decided)
        for unit in planned:
            if unit.span.latest >= cutoff:
                kept.append(unit)
            elif unit.stored is not None:
                removals.append(PlannedRemoval(unit=unit.stored.ref, reason=UnitRemovalReason.OUTSIDE_ROLLING_RETENTION))
                removed.add(unit.provider_key)
        decided_keys = removed | {unit.provider_key for unit in planned}
        for unit in await self._reader.units_ending_before(source_id, conversation_id, cutoff):
            span = Span.of(unit.members)
            if unit.provider_key not in decided_keys and (span is None or span.latest < cutoff):
                removals.append(PlannedRemoval(unit=unit.ref, reason=UnitRemovalReason.OUTSIDE_ROLLING_RETENTION))
        return kept, removals


def _returned(
    message: ManifestMessage,
    bodies: Mapping[MessageIdentity, Mapping[str, object]],
    uploaded_artifacts: Mapping[MessageIdentity, tuple[MemberArtifact, ...]],
    held: Mapping[MessageIdentity, Member],
) -> ReturnedMessage | None:
    """A manifest message with its body and artifacts: collected in this attempt, or already held.

    ``None`` when neither holds its record: intake waived the body of a
    message a Unit held at this revision, and an earlier run has changed that
    Unit since. The next attempt asks for the body again. An artifact neither
    uploaded nor held is left out; the next attempt names it again until the
    device uploads it.
    """

    stored = held.get(message.identity)
    record = bodies.get(message.identity)
    if record is None and stored is not None and stored.revision == message.revision:
        record = stored.record
    if record is None:
        return None
    if record_revision(record) != message.revision:
        raise ValueError(f"the body of message {message.identity} does not match its manifest revision")
    available = {
        (artifact.provider_key, artifact.provider_revision): artifact
        for artifact in (*(stored.artifacts if stored is not None else ()), *uploaded_artifacts.get(message.identity, ()))
    }
    return ReturnedMessage(
        member=Member(
            identity=message.identity,
            created_at=message.created_at,
            record=record,
            revision=message.revision,
            artifacts=tuple(
                available[(artifact.provider_key, artifact.provider_revision)]
                for artifact in message.artifacts
                if (artifact.provider_key, artifact.provider_revision) in available
            ),
        ),
        thread_root_id=message.thread_root_id,
    )


_Range = tuple[datetime, datetime]
"""A half-open creation-time range ``[start, end)``."""


def _left_unread(
    evidence: ConversationEvidence,
    *,
    floor: datetime,
    regular_start: datetime,
    received_at: datetime,
) -> list[_Range]:
    """The ranges after the history floor the previous attempt left unread, sorted and disjoint.

    A poll that reached its start or ran out of history read
    ``[requested start, proven end)``: nothing it did not return lies below
    its last page. A poll that stopped early read its proven interval
    (:func:`~memforge.conversations.coverage.proven_interval`), so a range
    the visibility margin leaves unproven stays unread. The regular poll is
    the one whose requested interval has no end.

    Let ``L`` be the later of ``floor`` and the earliest requested start of
    the previous polls, ``R`` the regular poll's proven end, and ``T`` the
    highest end of the previous backfill requests, or ``L`` when there were
    none. Unread are:

    - below ``T``, what a backfill requested and no poll read;
    - from ``T`` up to ``R``, what the regular poll requested and no poll read;
    - ``[regular_start, previous regular requested start)`` when this
      attempt's regular start is the earlier, because Initial History was
      widened: no previous poll requested it;
    - the instant a stopped regular poll is complete from, when it lies in
      ``[L, R)``. Its proven interval starts just after that instant, and
      requesting it keeps the next ``T`` at that poll's read start, so a chat
      whose regular poll keeps stopping continues from there instead of
      re-reading from the floor. A stop at or after ``R``, inside the
      visibility margin, proved nothing: ``[T, R)`` is unread whole, and the
      instant lies in the next regular poll's range.

    Above ``R`` lies the next regular poll's range. A conversation the
    previous attempt could not read leaves nothing to backfill; the regular
    poll restarts the chain. The previous attempt may be a failed run or a dry
    run; a range only a dry run read counts as read.
    """

    conversation = evidence.ref
    if conversation is None or not evidence.polls:
        return []
    regular = next((poll for poll in evidence.polls if poll.requested.end is None), None)
    if regular is None:
        raise ValueError(f"the previous attempt has no regular poll of conversation {evidence.conversation_id}")
    backfills = [poll.requested for poll in evidence.polls if poll.requested.end is not None]
    read = _union(_read(poll, conversation, received_at) for poll in evidence.polls)
    lowest = max(floor, min(poll.requested.start for poll in evidence.polls))
    end = proven_interval(regular, conversation, received_at=received_at).end
    top = max((backfill.end for backfill in backfills if backfill.end is not None), default=lowest)
    requested: list[_Range] = [
        (max(backfill.start, lowest), backfill.end) for backfill in backfills if backfill.end is not None
    ]
    requested.append((max(top, lowest, regular.requested.start), end))
    if regular_start < regular.requested.start:
        requested.append((regular_start, regular.requested.start))
    unread = _without(_union(requested), read)
    complete = regular.earliest_complete_page_time
    if regular.outcome is PollOutcome.STOPPED_EARLY and complete is not None and lowest <= complete < end:
        unread = _union([*unread, (complete, complete + JUST_AFTER)])
    return unread


def _read(poll: PollEvidence, conversation: ConversationRef, received_at: datetime) -> _Range:
    """The range one poll read: its proven interval, from its requested start unless it stopped early."""

    proven = proven_interval(poll, conversation, received_at=received_at)
    if poll.outcome is PollOutcome.STOPPED_EARLY:
        return proven.start, proven.end
    return poll.requested.start, proven.end


def _union(ranges: Iterable[_Range]) -> list[_Range]:
    """The non-empty ranges merged where they overlap or touch, sorted."""

    merged: list[_Range] = []
    for start, end in sorted(item for item in ranges if item[0] < item[1]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _without(ranges: Sequence[_Range], removed: Sequence[_Range]) -> list[_Range]:
    """``ranges`` minus ``removed``; both sorted and disjoint."""

    result: list[_Range] = []
    for start, end in ranges:
        for cut_start, cut_end in removed:
            if cut_end <= start or cut_start >= end:
                continue
            if cut_start > start:
                result.append((start, cut_start))
            start = max(start, cut_end)
            if start >= end:
                break
        if start < end:
            result.append((start, end))
    return result


def _once(removals: Sequence[PlannedRemoval]) -> tuple[PlannedRemoval, ...]:
    seen: set[str] = set()
    result = []
    for removal in removals:
        if removal.unit.provider_key not in seen:
            seen.add(removal.unit.provider_key)
            result.append(removal)
    return tuple(result)
