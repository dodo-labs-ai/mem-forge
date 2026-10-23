"""The behaviour every registered document connector owes its listing contract.

Startup validation checks what a class declares; this suite checks what the
Gene does, against a fake of its provider. It is parameterized by the
registered document source types, so a new connector joins it by registering
and fails it until ``tests/contracts/document_fakes`` has a module for it.
"""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

from memforge.genes import create_gene
from memforge.genes.document_gene import DocumentGene
from memforge.local_agent.source_contract import (
    local_agent_sync_operation,
    snapshot_document_ids,
    source_with_sync_inputs,
)
from memforge.models import AbsenceAnswer, ScopeListingKind
from memforge.source_types import source_type_descriptor
from tests.contracts.document_fakes import (
    CHANGED_SINCE,
    DocumentConnectorFakes,
    MissingDocumentFakes,
    ProviderListing,
    document_source_types,
    fakes_for,
    fakes_modules,
)
from tests.gene_fixture import FakeGene
from tests.source_type_fixture import register_test_source_type

DOCUMENT_TYPES = document_source_types()

# What reading one item by identifier answers for each provider status;
# every other status raises.
STATUS_ANSWERS = {
    httpx.codes.NOT_FOUND: AbsenceAnswer.ABSENT,
    httpx.codes.GONE: AbsenceAnswer.ABSENT,
    httpx.codes.FORBIDDEN: AbsenceAnswer.HIDDEN,
    httpx.codes.OK: AbsenceAnswer.EXISTS,
}
STATUSES_THAT_RAISE = (
    httpx.codes.UNAUTHORIZED,
    httpx.codes.TOO_MANY_REQUESTS,
    httpx.codes.INTERNAL_SERVER_ERROR,
    httpx.codes.SERVICE_UNAVAILABLE,
)
# Statuses that ask the client to wait: it retries them for any method, then
# raises once its attempts run out.
STATUSES_RETRIED = (httpx.codes.TOO_MANY_REQUESTS, httpx.codes.SERVICE_UNAVAILABLE)


def _gene_class(source_type: str) -> type[DocumentGene]:
    descriptor = source_type_descriptor(source_type)
    assert descriptor is not None
    return descriptor.gene_class()


def _fakes_or_none(source_type: str) -> DocumentConnectorFakes | None:
    try:
        return fakes_for(source_type)
    except MissingDocumentFakes:
        return None


def _cases(configs_of) -> list:
    """One case per source type with fakes and per configuration ``configs_of(source_type, fakes)`` names."""
    cases = []
    for source_type in DOCUMENT_TYPES:
        fakes = _fakes_or_none(source_type)
        if fakes is None:
            continue
        for name in configs_of(source_type, fakes):
            cases.append(pytest.param(source_type, name, id=f"{source_type}-{name}"))
    return cases


LISTING_CASES = _cases(lambda _source_type, fakes: fakes.listing_configs)
WALKING_CASES = _cases(
    lambda _source_type, fakes: [name for name, listing in fakes.listing_configs.items() if listing.walks_scope]
)
DEVICE_CASES = _cases(lambda _source_type, fakes: fakes.device_configs)
REDISCOVER_CASES = _cases(
    lambda source_type, fakes: [
        name
        for name, listing in fakes.listing_configs.items()
        if _gene_class(source_type).rediscovers_documents(listing.config)
    ]
)
QUERY_TYPES = [
    source_type
    for source_type in DOCUMENT_TYPES
    if ScopeListingKind.QUERY in _gene_class(source_type).listing_kinds
]


@pytest.fixture(autouse=True)
def own_origin_limiters(monkeypatch):
    """Give each test its own Atlassian origin limiters; the fakes' stub sleep skips their gaps."""
    monkeypatch.setattr("memforge.genes.atlassian_auth._ATLASSIAN_REQUEST_LIMITERS", {})


# ---------------------------------------------------------------------------
# The suite covers every registered document type
# ---------------------------------------------------------------------------


def test_the_suite_covers_exactly_the_registered_document_types() -> None:
    assert DOCUMENT_TYPES == document_source_types()
    assert fakes_modules() == set(DOCUMENT_TYPES)


def test_a_registered_type_without_fakes_fails_the_suite(monkeypatch) -> None:
    register_test_source_type(monkeypatch, "fakeless_documents", FakeGene)

    assert "fakeless_documents" in document_source_types()
    with pytest.raises(MissingDocumentFakes, match="tests/contracts/document_fakes/fakeless_documents.py"):
        fakes_for("fakeless_documents")


@pytest.mark.parametrize("source_type", DOCUMENT_TYPES)
def test_every_registered_document_type_has_fakes(source_type) -> None:
    assert isinstance(fakes_for(source_type), DocumentConnectorFakes)


# ---------------------------------------------------------------------------
# lists_scope
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("source_type", DOCUMENT_TYPES)
def test_lists_scope_matches_each_fake_configuration(source_type) -> None:
    gene_cls = _gene_class(source_type)
    fakes = fakes_for(source_type)
    if not gene_cls.listing_kinds:
        # A connector none of whose configurations can list never answers True.
        assert not fakes.listing_configs and not fakes.device_configs
    assert fakes.listing_configs or fakes.non_listing_configs or fakes.device_configs
    for name, listing in fakes.listing_configs.items():
        assert gene_cls.lists_scope(listing.config) is True, name
        assert local_agent_sync_operation(source_type, listing.config) is None, name
    for name, config in fakes.non_listing_configs.items():
        assert gene_cls.lists_scope(config) is False, name
    for name, config in fakes.device_configs.items():
        assert gene_cls.lists_scope(config) is True, name
        assert local_agent_sync_operation(source_type, config) is not None, name


# ---------------------------------------------------------------------------
# list_scope on the server
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("source_type", "config_name"), LISTING_CASES)
async def test_a_listing_is_one_of_the_declared_kinds(source_type, config_name) -> None:
    fakes = fakes_for(source_type)
    async with fakes.connect(fakes.listing_configs[config_name].config) as connection:
        listing = await connection.gene.list_scope()

    assert listing.kind in type(connection.gene).listing_kinds
    assert listing.doc_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(("source_type", "config_name"), LISTING_CASES)
async def test_a_standalone_listing_equals_the_listing_after_discovery(source_type, config_name) -> None:
    fakes = fakes_for(source_type)
    config = fakes.listing_configs[config_name].config
    async with fakes.connect(config) as connection:
        changed = {item.item_id async for item in connection.gene.discover(CHANGED_SINCE)}
        after_discovery = await connection.gene.list_scope()
    async with fakes.connect(config) as connection:
        everything = {item.item_id async for item in connection.gene.discover(None)}
    async with fakes.connect(config) as connection:
        standalone = await connection.gene.list_scope()

    assert standalone == after_discovery
    # Whatever ``since`` discovery used, the listing names every item it can yield.
    assert changed <= everything <= standalone.doc_ids


@pytest.mark.asyncio
@pytest.mark.parametrize(("source_type", "config_name"), WALKING_CASES)
async def test_a_listing_reuses_the_discovery_walk(source_type, config_name) -> None:
    fakes = fakes_for(source_type)
    async with fakes.connect(fakes.listing_configs[config_name].config) as connection:
        _ = [item async for item in connection.gene.discover(CHANGED_SINCE)]
        walked = len(connection.requests)
        await connection.gene.list_scope()

        assert connection.requests[walked:] == []


@pytest.mark.asyncio
@pytest.mark.parametrize(("source_type", "config_name"), LISTING_CASES)
async def test_a_truncated_listing_raises(source_type, config_name) -> None:
    fakes = fakes_for(source_type)
    async with fakes.connect(fakes.listing_configs[config_name].config, ProviderListing.TRUNCATED) as connection:
        with pytest.raises(Exception):  # noqa: B017 - any error is a refusal to list
            await connection.gene.list_scope()


@pytest.mark.asyncio
@pytest.mark.parametrize(("source_type", "config_name"), LISTING_CASES)
async def test_a_failed_listing_raises(source_type, config_name) -> None:
    fakes = fakes_for(source_type)
    async with fakes.connect(fakes.listing_configs[config_name].config, ProviderListing.FAILED) as connection:
        with pytest.raises(Exception):  # noqa: B017 - any error is a refusal to list
            await connection.gene.list_scope()


# ---------------------------------------------------------------------------
# Collected on a device: the snapshot manifest is the listing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("source_type", "config_name"), DEVICE_CASES)
async def test_a_device_collected_configuration_lists_through_its_manifest(source_type, config_name) -> None:
    fakes = fakes_for(source_type)
    config = fakes.device_configs[config_name]
    gene_cls = _gene_class(source_type)
    entries = fakes.manifest_entries(config)
    inputs = [
        SimpleNamespace(
            metadata={"manifest_entry": entry},
            raw_uri=f"artifact://{entry['doc_id']}",
            raw_sha256=f"input-{generation}",
            input_generation=generation,
        )
        for generation, entry in enumerate(entries)
    ]
    projected = source_with_sync_inputs(
        {"type": source_type, "config": dict(config)}, inputs, input_snapshot_supplied=True
    )
    gene = create_gene(source_type, projected["config"], "src-contract")
    await gene.authenticate()

    # The run lists through the manifest with the Gene's one kind, never through list_scope.
    assert len(gene_cls.listing_kinds) == 1
    with pytest.raises(Exception, match="snapshot manifest"):
        await gene.list_scope()
    # A run bound to the snapshot discovers exactly what the manifest lists.
    discovered = {item.item_id async for item in gene.discover(None)}
    assert discovered == snapshot_document_ids(inputs)
    assert discovered
    # The device read the provider; the server holds no client to release.
    assert getattr(gene, "_client", None) is None
    await gene.close()


# ---------------------------------------------------------------------------
# confirm_absent
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [*STATUS_ANSWERS, *STATUSES_THAT_RAISE], ids=str)
@pytest.mark.parametrize("source_type", QUERY_TYPES)
async def test_confirm_absent_maps_each_provider_status(source_type, status) -> None:
    async with fakes_for(source_type).reading_by_id(status) as (connection, items):
        assert items
        if status in STATUS_ANSWERS:
            answers = await connection.gene.confirm_absent(items)
            assert answers == {item.item_id: STATUS_ANSWERS[status] for item in items}
        else:
            with pytest.raises(Exception):  # noqa: B017 - any error leaves the run's absence unproven
                await connection.gene.confirm_absent(items)


@pytest.mark.asyncio
@pytest.mark.parametrize("status", STATUSES_RETRIED, ids=str)
@pytest.mark.parametrize("source_type", QUERY_TYPES)
async def test_confirm_absent_retries_a_provider_that_asks_to_wait_then_raises(source_type, status) -> None:
    async with fakes_for(source_type).reading_by_id(status) as (connection, items):
        with pytest.raises(Exception):  # noqa: B017 - any error leaves the run's absence unproven
            await connection.gene.confirm_absent(items)

    # The read was sent again after a wait, and the retries ended.
    read = connection.requests[-1]
    assert connection.requests.count(read) > 1
    assert connection.waits


# ---------------------------------------------------------------------------
# rediscover
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("source_type", "config_name"), REDISCOVER_CASES)
async def test_rediscover_keeps_the_predicates_of_discover(source_type, config_name) -> None:
    fakes = fakes_for(source_type)
    config = fakes.listing_configs[config_name].config
    async with fakes.connect(config) as connection:
        discovered = {item.item_id async for item in connection.gene.discover(None)}
    async with fakes.connect(config) as connection:
        candidates = fakes.rediscover_candidates(config)
        rediscovered = [await connection.gene.rediscover(candidate) for candidate in candidates]

    found = [item.item_id for item in rediscovered if item is not None]
    # Every item a full discovery yields is found again, and nothing else is.
    assert sorted(found) == sorted(discovered)
    assert len(candidates) > len(found)


# ---------------------------------------------------------------------------
# close
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(("source_type", "config_name"), LISTING_CASES)
async def test_close_releases_every_client_the_gene_opened(source_type, config_name) -> None:
    fakes = fakes_for(source_type)
    async with fakes.connect(fakes.listing_configs[config_name].config) as connection:
        await connection.gene.list_scope()
        assert connection.clients
        assert not any(client.is_closed for client in connection.clients)

        await connection.gene.close()

        assert all(client.is_closed for client in connection.clients)
        # Closing again does nothing.
        await connection.gene.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("source_type", DOCUMENT_TYPES)
async def test_close_before_authenticate_does_nothing(source_type) -> None:
    fakes = fakes_for(source_type)
    configs = [
        *(listing.config for listing in fakes.listing_configs.values()),
        *fakes.non_listing_configs.values(),
        *fakes.device_configs.values(),
    ]
    for config in configs:
        gene = create_gene(source_type, dict(config), "src-contract")
        await gene.close()
        await gene.close()
