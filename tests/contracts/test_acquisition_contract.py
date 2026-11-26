"""Each document type's acquisition returns what its declared acquisition contract says (ADR 0058).

A device acquires with the MemForge version it has installed, and the server
interprets with its own; the two agree while their ``acquisition_contract``
for the type is equal. This suite keeps, per type and contract, a digest of
what the device's acquisition hands the server for each configuration its
fake provider serves: the manifest, every acquired input and every artifact
body. A change to acquisition that alters it fails here until the type
declares a new contract, whose digests are then added to
``fixtures/acquisition_contracts.json``; so a contract cannot change by
accident.

Set ``MEMFORGE_PRINT_ACQUISITION_CONTRACTS=1`` to print the current digests.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from weakref import WeakValueDictionary

import pytest

from memforge.local_agent.acquisition_daemon import acquire_document_source
from memforge.local_agent.document_acquisition import (
    AcquiredInputPackage,
    AcquisitionAnswers,
    AcquisitionManifest,
    AcquisitionRequests,
)
from memforge.models import ContentItem
from memforge.source_artifacts import RawSourceArtifact
from memforge.source_types import source_type_descriptor
from tests.contracts.document_fakes import document_source_types, fakes_for

GOLDENS = Path(__file__).parent.parent / "fixtures" / "acquisition_contracts.json"


def _acquisition_configs(source_type: str) -> dict[str, Mapping[str, object]]:
    """The configurations the type's fake provider serves to its own Gene."""

    fakes = fakes_for(source_type)
    return {
        **{name: listing.config for name, listing in fakes.listing_configs.items()},
        **dict(fakes.non_listing_configs),
    }


CASES = [
    pytest.param(source_type, name, id=f"{source_type}-{name}")
    for source_type in document_source_types()
    for name in _acquisition_configs(source_type)
]


def _contract_key(source_type: str) -> str:
    descriptor = source_type_descriptor(source_type)
    assert descriptor is not None and descriptor.acquisition_contract is not None
    return f"{source_type}:{descriptor.acquisition_contract}"


ACQUIRED_AT = datetime(2026, 7, 1, 12, 0, tzinfo=timezone.utc)


class _StoppedClock(datetime):
    """``datetime`` whose ``now`` is :data:`ACQUIRED_AT`, so a Gene that stamps discovery time acquires alike."""

    @classmethod
    def now(cls, tz=None):  # type: ignore[override]
        return ACQUIRED_AT if tz is None else ACQUIRED_AT.astimezone(tz)


@pytest.fixture(autouse=True)
def own_origin_limiters(monkeypatch):
    monkeypatch.setattr("memforge.genes.atlassian_auth._ATLASSIAN_REQUEST_LIMITERS", {})
    monkeypatch.setattr("memforge.genes.toolkit.pacing._SHARED_LIMITERS", WeakValueDictionary())


@pytest.fixture(autouse=True)
def stopped_gene_clocks(monkeypatch):
    """Stop the clock each document Gene's module reads."""

    for source_type in document_source_types():
        module = sys.modules[source_type_descriptor(source_type).gene_class().__module__]
        if getattr(module, "datetime", None) is datetime:
            monkeypatch.setattr(module, "datetime", _StoppedClock)


class _CapturingTransport:
    """Keeps what one attempt hands the server, in the order it does."""

    def __init__(self) -> None:
        self.handed: dict[str, list] = {"manifest": [], "inputs": [], "artifacts": []}

    async def submit_manifest(self, manifest: AcquisitionManifest) -> AcquisitionRequests:
        self.handed["manifest"].append(manifest.to_payload())
        return AcquisitionRequests()

    async def upload_artifact(self, item: ContentItem, artifact: RawSourceArtifact, body: bytes) -> str:
        digest = hashlib.sha256(body).hexdigest()
        self.handed["artifacts"].append([item.item_id, artifact.provider_key, digest])
        return digest

    async def upload_input(self, package: AcquiredInputPackage) -> None:
        self.handed["inputs"].append(package.to_payload())

    async def stored_extras(self, item_ids) -> dict:
        return {}

    async def submit_answers(self, answers: AcquisitionAnswers) -> None:
        raise AssertionError("a run without requests has nothing to answer")


async def _no_wait(_seconds: float) -> None:
    return None


async def acquired_digest(source_type: str, config_name: str) -> str:
    """The digest of everything a full device acquisition of one configuration hands the server.

    It includes the PDF material of a type that exports PDFs.
    """

    config = _acquisition_configs(source_type)[config_name]
    transport = _CapturingTransport()
    async with fakes_for(source_type).connect(config) as connection:
        gene = connection.gene

        async def authenticated() -> None:
            # The fake's connection is already authenticated and closes itself.
            return None

        gene.authenticate = authenticated  # type: ignore[method-assign]
        gene.close = authenticated  # type: ignore[method-assign]
        await acquire_document_source(
            {"config": dict(config), "since": None},
            open_gene=lambda _config: gene,
            transport=transport,
            exports_pdf=source_type_descriptor(source_type).exports_pdf,
            sleep=_no_wait,
        )
    encoded = json.dumps(transport.handed, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def test_every_acquiring_type_has_digests_for_its_declared_contract() -> None:
    goldens = json.loads(GOLDENS.read_text())
    acquiring = {source_type for source_type in document_source_types() if _acquisition_configs(source_type)}

    assert sorted(goldens) == sorted(_contract_key(source_type) for source_type in acquiring)
    for source_type in acquiring:
        assert sorted(goldens[_contract_key(source_type)]) == sorted(_acquisition_configs(source_type))


@pytest.mark.asyncio
@pytest.mark.parametrize(("source_type", "config_name"), CASES)
async def test_acquisition_returns_what_its_declared_contract_recorded(source_type, config_name) -> None:
    digest = await acquired_digest(source_type, config_name)
    if os.environ.get("MEMFORGE_PRINT_ACQUISITION_CONTRACTS"):
        print(f"ACQUISITION {_contract_key(source_type)} {config_name} {digest}")
    goldens = json.loads(GOLDENS.read_text())

    assert digest == goldens.get(_contract_key(source_type), {}).get(config_name), (
        f"{source_type}'s acquisition changed what it hands the server for {config_name!r}; "
        "declare a new acquisition_contract and record its digests"
    )


@pytest.mark.asyncio
async def test_a_changed_acquisition_without_a_new_contract_fails(monkeypatch) -> None:
    source_type, config_name = CASES[0].values
    recorded = json.loads(GOLDENS.read_text())[_contract_key(source_type)][config_name]
    gene_cls = source_type_descriptor(source_type).gene_class()
    fetch = gene_cls.fetch

    async def fetch_with_another_field(self, item):
        raw = await fetch(self, item)
        item.extra = {**dict(item.extra or {}), "read_by_a_newer_acquisition": True}
        return raw

    monkeypatch.setattr(gene_cls, "fetch", fetch_with_another_field)

    assert await acquired_digest(source_type, config_name) != recorded
