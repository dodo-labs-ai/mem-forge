"""Fake providers for the document connector contract suite.

``tests/contracts/test_document_connector_contract.py`` runs every registered
document source type against the module here named after it: ``jira.py``
fakes Jira for the ``jira`` type. The suite runs the real Gene; only the
provider behind it is fake. A registered document type without a module fails
the suite, so a new connector is checked from the moment it registers.

Adding a document source type
-----------------------------

Add ``document_fakes/<source_type>.py`` defining ``FAKES``, an instance of a
:class:`DocumentConnectorFakes` subclass. Fake the provider at its transport
(an ``httpx.MockTransport``, or the client class the Gene builds in
``authenticate``), never the Gene's own methods, so the checks exercise the
Gene's paging, validation and error handling. Give every item a time either
before or after :data:`CHANGED_SINCE`, so incremental discovery yields some
items and skips others.

What each fake must answer
--------------------------

``listing_configs``
    Named configurations ``lists_scope`` answers ``True`` for and the server
    lists. ``walks_scope`` says whether discovery visits the whole scope
    whatever ``since`` is, so that ``list_scope`` must reuse that walk instead
    of asking the provider again.
``non_listing_configs``
    Named configurations ``lists_scope`` answers ``False`` for.
``device_configs``
    Named configurations collected on a device, whose listing is the snapshot
    manifest; ``manifest_entries`` gives the entries such a device submits.
``connect(config, listing)``
    The Gene for one configuration, authenticated against a fake provider that
    holds a few items in the scope and at least one outside it. With
    :attr:`ProviderListing.TRUNCATED` the provider ends or truncates the
    listing early while claiming more; with :attr:`ProviderListing.FAILED` it
    answers the listing with a server error. The connection records every
    request the provider receives.
``reading_by_id(status)``
    Required when ``listing_kinds`` include ``query``: a Gene and stored items
    whose read by identifier answers HTTP ``status``.
``rediscover_candidates(config)``
    Required for a configuration ``rediscovers_documents`` answers ``True``
    for: stored items to rediscover against ``connect(config)``'s provider,
    one for every item a full discovery yields and some for items outside the
    scope or gone from the provider.

A type whose ``listing_kinds`` is empty has only ``non_listing_configs``.
"""

from __future__ import annotations

import importlib
import pkgutil
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from memforge.genes.document_gene import DocumentGene
from memforge.genes.family import CollectionFamilyKind
from memforge.models import ContentItem
from memforge.source_types import registered_source_types

__all__ = [
    "CHANGED_SINCE",
    "RECENTLY",
    "LONG_AGO",
    "DocumentConnectorFakes",
    "FakeConnection",
    "ListingConfig",
    "MissingDocumentFakes",
    "ProviderListing",
    "document_source_types",
    "fakes_for",
    "fakes_modules",
]

# Incremental discovery in the suite reads from CHANGED_SINCE. Each fake item
# changed either LONG_AGO, before it, or RECENTLY, after it.
CHANGED_SINCE = datetime(2026, 3, 1, tzinfo=timezone.utc)
LONG_AGO = datetime(2026, 1, 1, tzinfo=timezone.utc)
RECENTLY = datetime(2026, 6, 1, tzinfo=timezone.utc)


class ProviderListing(Enum):
    """How a fake provider answers the requests that list the scope."""

    COMPLETE = "complete"
    TRUNCATED = "truncated"
    FAILED = "failed"


@dataclass(frozen=True)
class ListingConfig:
    """A configuration the server lists, and whether its discovery already walks the whole scope."""

    config: Mapping[str, object]
    walks_scope: bool


@dataclass
class FakeConnection:
    """A Gene connected to a fake provider, and every request the provider received, in order."""

    gene: DocumentGene
    requests: list[str] = field(default_factory=list)


class DocumentConnectorFakes:
    """What the contract suite needs from one document source type; see the package docstring."""

    listing_configs: Mapping[str, ListingConfig] = {}
    non_listing_configs: Mapping[str, Mapping[str, object]] = {}
    device_configs: Mapping[str, Mapping[str, object]] = {}

    def connect(
        self,
        config: Mapping[str, object],
        listing: ProviderListing = ProviderListing.COMPLETE,
    ) -> AbstractAsyncContextManager[FakeConnection]:
        raise NotImplementedError(f"{type(self).__name__} does not fake a provider")

    def manifest_entries(self, config: Mapping[str, object]) -> list[dict]:
        raise NotImplementedError(f"{type(self).__name__} does not fake a device manifest")

    def reading_by_id(self, status: int) -> AbstractAsyncContextManager[tuple[DocumentGene, Sequence[ContentItem]]]:
        raise NotImplementedError(f"{type(self).__name__} does not fake reads by identifier")

    def rediscover_candidates(self, config: Mapping[str, object]) -> list[ContentItem]:
        raise NotImplementedError(f"{type(self).__name__} does not fake rediscovery")


class MissingDocumentFakes(AssertionError):
    """A registered document source type has no module in this package."""


def document_source_types() -> list[str]:
    """Every registered document source type, which the contract suite is parameterized by."""

    return sorted(
        descriptor.source_type
        for descriptor in registered_source_types()
        if descriptor.family is CollectionFamilyKind.DOCUMENT
    )


def fakes_modules() -> set[str]:
    """The source types this package has a fakes module for."""

    return {module.name for module in pkgutil.iter_modules(__path__) if not module.name.startswith("_")}


def fakes_for(source_type: str) -> DocumentConnectorFakes:
    """The fakes of one source type; a registered type without a module fails here."""

    try:
        module = importlib.import_module(f"{__name__}.{source_type}")
    except ModuleNotFoundError as exc:
        if exc.name != f"{__name__}.{source_type}":
            raise
        raise MissingDocumentFakes(
            f"document source type {source_type!r} has no fakes: add tests/contracts/document_fakes/{source_type}.py "
            "defining FAKES (see the package docstring)"
        ) from exc
    return module.FAKES

