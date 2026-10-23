"""Fake providers for the conversation provider contract suite.

``tests/contracts/test_conversation_provider_contract.py`` runs every
registered conversation source type against the module here named after it:
``teams.py`` fakes the Teams Chat Service for the ``teams`` type. The suite
runs the real provider and its real client; only the service behind them is
fake. A registered conversation type without a module fails the suite, so a
new provider is checked from the moment it registers.

Adding a conversation source type
---------------------------------

Add ``conversation_fakes/<source_type>.py`` defining ``FAKES``, an instance of
a :class:`ConversationProviderFakes` subclass. Fake the provider's service at
its transport, never the provider's own methods, so the checks exercise the
provider's paging, certificates, error translation and record mapping. Model
the service exactly as it answers, in its own order, cursor and precision,
never more leniently: the suite then shows what a poll's evidence proves
against the messages the service really holds.

The suite describes what the service holds as a :class:`Scenario`: the
messages of one conversation as ground truth, and how the service pages them.
The fake serves that scenario the way its service would, deriving each page
and the certificate it carries from the ground truth.

What each fake must answer
--------------------------

``conversation_id``
    The id of a conversation the provider can read; every scenario holds the
    messages of this conversation.
``time_precision``
    The step the service records message times in, such as one millisecond
    for Teams. Every scenario time is a whole multiple of it.
``refused_answers``
    Names of answers that may mean lost access, such as Teams' HTTP 403 and
    404: a poll answered so raises ``ConversationUnavailable`` with
    ``no_access``, and nothing answered so reports ``gone``.
``gone_answers``
    Names of answers that cannot mean lost access, so that reading the
    conversation by id answered so reports ``gone``. Teams has none: its
    service never tells a deleted conversation from a hidden one.
``failing_answers``
    Names of every other answer the service may give, such as an expired
    session, a server error or a malformed page. A poll answered so raises
    an error that is not ``ConversationUnavailable``.
``connect(scenario, answer=None)``
    The real provider over a fake service holding ``scenario``. With an
    ``answer`` named in one of the lists above, the service gives that answer
    to every request. The connection closes the provider's client when it
    ends.
``FakeConversationConnection.stop_after(pages)``
    Makes every later poll of the connection stop early once it has read
    ``pages`` pages, the way the provider stops on its own (for Teams, the
    poll's time budget runs out on the client's injected clock).
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from memforge.conversations.port import ConversationProvider
from memforge.genes.family import CollectionFamilyKind
from memforge.source_types import registered_source_types

__all__ = [
    "NOW",
    "ConversationProviderFakes",
    "FakeConversationConnection",
    "MissingConversationFakes",
    "Scenario",
    "ScenarioMessage",
    "conversation_source_types",
    "fakes_for",
    "fakes_modules",
]

# Every scenario message is created before NOW, and the suite starts each
# poll at NOW, so a poll open to the newest message reads them all.
NOW = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ScenarioMessage:
    """One message the service holds.

    ``reply_to`` names the message a reply answers, and is ``None`` for a
    top-level message. ``deleted_at`` is when the message was deleted; a
    service that keeps a deleted message in place of its content, as Teams
    does, places it at that time.
    """

    message_id: str
    created_at: datetime
    reply_to: str | None = None
    deleted_at: datetime | None = None

    @property
    def live(self) -> bool:
        return self.deleted_at is None


@dataclass(frozen=True)
class Scenario:
    """What the service holds for the fake's conversation, and how it pages it.

    ``page_size`` is how many messages the service puts on one page, newest
    first. With ``ends_on_empty_page`` the page holding the oldest message is
    not certified as the oldest one, and the request after it answers an
    empty page.
    """

    messages: tuple[ScenarioMessage, ...]
    page_size: int = 200
    ends_on_empty_page: bool = False

    def created_between(self, start: datetime, end: datetime | None) -> set[str]:
        """The ids of the messages, live or deleted, created in ``[start, end)``."""

        return {
            message.message_id
            for message in self.messages
            if start <= message.created_at and (end is None or message.created_at < end)
        }


@dataclass
class FakeConversationConnection:
    """A provider connected to a fake service, and the way to make its polls stop early."""

    provider: ConversationProvider
    stop_after: Callable[[int], None] = field(repr=False)


class ConversationProviderFakes:
    """What the contract suite needs from one conversation source type; see the package docstring."""

    conversation_id: str
    time_precision: timedelta
    refused_answers: tuple[str, ...] = ()
    gone_answers: tuple[str, ...] = ()
    failing_answers: tuple[str, ...] = ()

    def connect(
        self, scenario: Scenario, *, answer: str | None = None
    ) -> AbstractAsyncContextManager[FakeConversationConnection]:
        raise NotImplementedError(f"{type(self).__name__} does not fake a provider")


class MissingConversationFakes(AssertionError):
    """A registered conversation source type has no module in this package."""


def conversation_source_types() -> list[str]:
    """Every registered conversation source type, which the contract suite is parameterized by."""

    return sorted(
        descriptor.source_type
        for descriptor in registered_source_types()
        if descriptor.family is CollectionFamilyKind.CONVERSATION
    )


def fakes_modules() -> set[str]:
    """The source types this package has a fakes module for."""

    return {module.name for module in pkgutil.iter_modules(__path__) if not module.name.startswith("_")}


def fakes_for(source_type: str) -> ConversationProviderFakes:
    """The fakes of one source type; a registered type without a module fails here."""

    try:
        module = importlib.import_module(f"{__name__}.{source_type}")
    except ModuleNotFoundError as exc:
        if exc.name != f"{__name__}.{source_type}":
            raise
        raise MissingConversationFakes(
            f"conversation source type {source_type!r} has no fakes: add "
            f"tests/contracts/conversation_fakes/{source_type}.py defining FAKES (see the package docstring)"
        ) from exc
    return module.FAKES
