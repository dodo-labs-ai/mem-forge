"""The conversation provider port: what a conversation connector implements.

A conversation connector calls its provider and answers in the neutral terms
below. It maps each message to the canonical record its schema declares and
translates the provider's answers into conversation statuses and poll
evidence. It holds no window, coverage, absence or retention logic; the
conversation layer owns those, on the server, for every provider alike.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from memforge.source_adapters.contracts import CanonicalRecordSchema


class ThreadingModel(str, Enum):
    """How a provider threads the messages of one conversation."""

    # Every top-level post is a thread from the start and never enters a window.
    POST_REPLY = "post_reply"
    # Top-level messages without replies go into windows; a message with
    # replies roots a thread.
    INLINE = "inline"
    # No threads: every message goes into a window.
    FLAT = "flat"


class ConversationAccess(str, Enum):
    """Whether the Source can read one selected conversation."""

    READABLE = "readable"
    # The provider confirms the conversation no longer exists, by an answer
    # that cannot also mean lost access.
    GONE = "gone"
    # The provider refuses to show the conversation, or answers in a way that
    # may mean either deletion or lost access.
    NO_ACCESS = "no_access"


class PollOutcome(str, Enum):
    """How a poll of one conversation ended, read newest first.

    A page is complete from the time the provider holds every message up to
    the page's end: what the provider certifies for the page, or the time of
    its oldest message when the provider orders pages by creation time. A
    provider that may place an older message on a newer page, such as Teams
    placing a deleted message where it was deleted, takes that time from its
    own certificate, never from the times of the messages it returned.

    A poll that returned a message names the time its read is complete from,
    whatever its outcome; only a poll that returned nothing may name none.
    """

    # The read was complete from before the requested start, so the provider
    # showed history at least that far back; the complete time lies before
    # that start.
    REACHED_START = "reached_start"
    # Paging read every page the provider holds before the requested start was
    # reached: the provider has nothing older to show, which may be hidden
    # history as well as the conversation's start. The complete time is the
    # oldest page's, at or after the requested start.
    HISTORY_EXHAUSTED = "history_exhausted"
    # Paging stopped before either, for example on a time budget. What was
    # read contiguously from the newest message still counts, from the time
    # the last fully read page is complete from, which lies inside the
    # requested interval.
    STOPPED_EARLY = "stopped_early"


@dataclass(frozen=True, slots=True)
class ConversationLabel:
    """How one conversation is named and shown; every Unit of it keeps the label.

    ``title_kind`` and ``title_fields`` are the provider's own labels for the
    Unit Title, such as the conversation kind and the team it belongs to; the
    conversation layer adds the conversation name and the Unit's start.
    ``message_link`` links to one message, with ``{message_id}`` in place of
    its id.
    """

    conversation_id: str
    kind: str
    name: str | None
    title_kind: str = "Conversation"
    title_fields: tuple[tuple[str, str], ...] = ()
    message_link: str | None = None


@dataclass(frozen=True, slots=True)
class ConversationRef:
    """One selected conversation: its label and how its provider serves it."""

    label: ConversationLabel
    threading: ThreadingModel
    poll_includes_replies: bool
    # The start of visible history the provider certifies, if it can.
    visible_from_certified: datetime | None = None

    @property
    def conversation_id(self) -> str:
        return self.label.conversation_id


@dataclass(frozen=True, slots=True)
class ConversationStatus:
    """The neutral status of one conversation, with the provider's answer as evidence."""

    access: ConversationAccess
    provider_answer: str


@dataclass(frozen=True, slots=True)
class Interval:
    """A half-open creation-time interval ``[start, end)``.

    ``end`` is ``None`` when the executor polls up to the newest message; the
    conversation layer fixes the end of what that poll proves (see
    :mod:`memforge.conversations.coverage`).
    """

    start: datetime
    end: datetime | None = None

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or (self.end is not None and self.end.tzinfo is None):
            raise ValueError("interval times must be offset-aware")
        if self.end is not None and self.end < self.start:
            raise ValueError("interval ends before it starts")


@dataclass(frozen=True, slots=True, order=True)
class MessageArtifact:
    """A binary a message carries, such as an inline image: its provider identity and revision."""

    provider_key: str
    provider_revision: str


@dataclass(frozen=True, slots=True)
class ArtifactBody:
    """The bytes of one message artifact as the provider serves them."""

    content: bytes
    media_type: str
    filename: str


@dataclass(frozen=True, slots=True)
class ProviderMessage:
    """One message as the provider returned it, mapped to its canonical record.

    ``thread_root_id`` names the root of the thread a reply belongs to, as the
    provider states it, and is ``None`` for a top-level message.
    """

    message_id: str
    created_at: datetime
    record: Mapping[str, object]
    thread_root_id: str | None = None
    artifacts: tuple[MessageArtifact, ...] = ()

    def __post_init__(self) -> None:
        if not self.message_id:
            raise ValueError("a provider message needs its id")
        if self.created_at.tzinfo is None:
            raise ValueError("message creation time must be offset-aware")
        if self.thread_root_id == self.message_id:
            raise ValueError("a thread root is not a reply to itself")


@dataclass(frozen=True, slots=True)
class ConversationPoll:
    """The messages of one conversation in one requested interval, read newest first."""

    messages: tuple[ProviderMessage, ...]
    outcome: PollOutcome
    # The time the read is complete from (see PollOutcome): earlier than the
    # requested start when paging reached it; the oldest page's, at or after
    # the requested start, when history ran out; when paging stopped early,
    # the time the last fully read page is complete from, inside the requested
    # interval. A poll that returned a message always names it.
    earliest_complete_page_time: datetime | None = None
    stop_reason: str = ""


@dataclass(frozen=True, slots=True)
class ConversationListing:
    """The conversations the provider lists for the Source's credential, and the status of each.

    A listing may hold only some conversations, such as the most recent ones;
    a selected conversation it leaves out is read by id.
    """

    conversations: tuple[ConversationRef, ...]
    statuses: Mapping[str, ConversationStatus] = field(default_factory=dict)


class ConversationUnavailable(RuntimeError):
    """The provider refused one poll of a conversation.

    It carries the conversation's status; the executor records that status
    and nothing the conversation returned in this attempt.
    """

    def __init__(self, status: ConversationStatus) -> None:
        super().__init__(status.provider_answer)
        self.status = status


REQUIRED_PROVIDER_MEMBERS = ("record_schema", "list_conversations", "get_conversation", "poll", "read_artifact")
"""The members every conversation provider class defines; startup refuses a provider that lacks one."""

THREAD_PROVIDER_MEMBERS = ("fetch_thread",)
"""The members a provider whose polls can omit replies also defines.

Such a provider sets the class attribute ``polls_omit_replies = True``. Teams'
polls return every reply with its post, so Teams leaves it ``False`` and
defines none of these.
"""


@runtime_checkable
class ConversationProvider(Protocol):
    """What one conversation connector implements.

    ``record_schema`` is the canonical record schema of the records this
    provider returns. ``poll`` raises :class:`ConversationUnavailable` when
    the provider refuses the conversation; any other error fails the
    collection attempt, which then removes nothing.

    A source type names its provider class and the factory that opens it in
    its descriptor (``conversation_provider``), and startup checks the class
    for :data:`REQUIRED_PROVIDER_MEMBERS`.
    """

    @property
    def record_schema(self) -> CanonicalRecordSchema: ...

    async def list_conversations(self) -> ConversationListing: ...

    async def get_conversation(self, conversation_id: str) -> tuple[ConversationStatus, ConversationRef | None]:
        """A selected conversation the listing did not include, read by id; its reference when readable."""
        ...

    async def poll(self, conversation: ConversationRef, interval: Interval) -> ConversationPoll: ...

    async def read_artifact(
        self, conversation: ConversationRef, message: ProviderMessage, artifact: MessageArtifact
    ) -> ArtifactBody: ...
