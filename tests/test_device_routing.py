"""Which device takes a document Source's run, and what a deploy that changes a contract does to it (ADR 0058).

A ``document_acquisition`` job goes only to a device of the Source's
execution owner whose report names the Source and whose acquisition contract
for the type equals the server's. A device behind or ahead of the server, or
one without the Source's credentials, leaves the job queued without counting
an attempt. A submission made under a contract the server no longer has is
refused, and the job is queued again without counting the attempt.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import memforge.server.admin_api as admin_api
from memforge import source_types
from memforge.local_agent.device_report import declared_acquisition_contracts
from memforge.local_agent.document_acquisition import AcquiredListing, AcquisitionManifest
from memforge.local_agent.source_contract import ACQUISITION_CONTRACT_CHANGED, LOCAL_AGENT_JOB_MAX_ATTEMPTS
from memforge.models import ScopeListingKind
from memforge.pipeline.device_acquisition import enqueue_document_acquisition_job
from memforge.storage.adapters.context import LOCAL_DEV_USER_ID
from memforge.storage.document_store import LocalDocumentStore
from tests.device_report_fixture import device_report
from tests.test_device_acquisition import SOURCE_ID, WORKSPACE, WikiGene, _connect, _own, wiki  # noqa: F401
from tests.test_local_adapter_api import _config

NOW = datetime(2026, 11, 26, 9, 0, tzinfo=timezone.utc)


def _declare_wiki_contract(monkeypatch, contract: int) -> None:
    """Deploy a server whose ``wiki`` type declares ``contract``."""

    descriptors = dict(source_types._DESCRIPTORS)
    descriptors["wiki"] = replace(descriptors["wiki"], acquisition_contract=contract)
    monkeypatch.setattr(source_types, "_DESCRIPTORS", descriptors)


@pytest.fixture
def owned(tmp_path, wiki):  # noqa: F811
    database = _connect(tmp_path / "device.db")
    _own(database)
    store = LocalDocumentStore(str(tmp_path / "docs"))
    source = asyncio.run(database.get_source(SOURCE_ID))
    job_id, created = asyncio.run(
        enqueue_document_acquisition_job(database, store, source, workspace_id=WORKSPACE, trigger="schedule")
    )
    assert created
    try:
        yield database, job_id
    finally:
        asyncio.run(database.close())


def _lease(database, report, *, now: datetime = NOW) -> list[dict]:
    return asyncio.run(
        database.lease_local_agent_jobs(
            user_id=LOCAL_DEV_USER_ID,
            device=report,
            acquisition_contracts=declared_acquisition_contracts(),
            limit=5,
            lease_seconds=60,
            now=now,
        )
    )


@pytest.mark.parametrize(
    "report",
    [
        pytest.param(device_report(), id="without-the-sources-credentials"),
        pytest.param(device_report([SOURCE_ID], acquisition_contracts={"wiki": 2}), id="ahead-of-the-server"),
        pytest.param(device_report([SOURCE_ID], acquisition_contracts={"wiki": 0}), id="behind-the-server"),
        pytest.param(device_report([SOURCE_ID], acquisition_contracts={}), id="without-the-type"),
    ],
)
def test_a_device_that_cannot_acquire_the_source_leaves_its_job_waiting_uncounted(owned, report) -> None:
    database, job_id = owned

    assert _lease(database, report) == []

    job = asyncio.run(database.get_local_agent_job(job_id))
    assert (job["status"], job["attempt_count"], job["last_error"]) == ("queued", 0, None)


def test_a_device_that_names_the_source_under_the_servers_contract_leases_its_job(owned) -> None:
    database, job_id = owned
    assert _lease(database, device_report(device_id="dev-desktop")) == []

    [leased] = _lease(database, device_report([SOURCE_ID]))

    assert leased["job_id"] == job_id
    assert (leased["attempt_count"], leased["acquisition_contract"]) == (1, 1)
    # Either device's lease kept its report.
    devices = asyncio.run(database.list_local_agent_devices(LOCAL_DEV_USER_ID))
    assert sorted(device.report.device_id for device in devices) == ["dev-desktop", "dev-laptop"]


def test_another_users_device_never_leases_the_owners_job(owned) -> None:
    database, job_id = owned

    leased = asyncio.run(
        database.lease_local_agent_jobs(
            user_id="someone-else",
            device=device_report([SOURCE_ID]),
            acquisition_contracts=declared_acquisition_contracts(),
            limit=5,
            lease_seconds=60,
            now=NOW,
        )
    )

    assert leased == []


def test_a_submission_under_a_contract_the_server_no_longer_has_is_requeued_without_a_failure(
    tmp_path, owned, monkeypatch
) -> None:
    database, job_id = owned
    [leased] = _lease(database, device_report([SOURCE_ID]), now=datetime.now(timezone.utc))
    manifest = AcquisitionManifest(listing=AcquiredListing(kind=ScopeListingKind.QUERY, doc_ids=frozenset()))
    body = {"local_agent_job_id": job_id, "local_agent_attempt_count": 1, "manifest": manifest.to_payload()}
    app = admin_api.create_admin_app(db=database, config=_config(tmp_path))

    with TestClient(app) as client:
        # A deploy changes the type's contract while the device acquires.
        _declare_wiki_contract(monkeypatch, 2)
        refused = client.post(f"/api/v1/sources/{SOURCE_ID}/acquisition/manifest", json=body)
        requeued = asyncio.run(database.get_local_agent_job(job_id))
        behind = _lease(database, device_report([SOURCE_ID], acquisition_contracts={"wiki": 1}))
        [updated] = _lease(database, device_report([SOURCE_ID]), now=datetime.now(timezone.utc))
        accepted = client.post(
            f"/api/v1/sources/{SOURCE_ID}/acquisition/manifest",
            json={**body, "local_agent_attempt_count": 2},
        )

    assert leased["acquisition_contract"] == 1
    assert (refused.status_code, refused.json()["detail"]) == (409, ACQUISITION_CONTRACT_CHANGED)
    assert (requeued["status"], requeued["attempt_count"], requeued["uncounted_attempt_count"]) == ("queued", 1, 1)
    assert (requeued["last_error"], requeued["next_attempt_at"]) == (None, None)
    assert behind == []
    assert (updated["attempt_count"], updated["acquisition_contract"]) == (2, 2)
    assert accepted.status_code == 200, accepted.text


def test_an_attempt_requeued_under_a_changed_contract_does_not_spend_the_retry_budget(owned) -> None:
    database, job_id = owned
    report = device_report([SOURCE_ID])
    leased_at = NOW
    [leased] = _lease(database, report, now=leased_at)
    assert asyncio.run(database.requeue_local_agent_job_uncounted(job_id=job_id, attempt_count=1))
    for attempt in range(2, LOCAL_AGENT_JOB_MAX_ATTEMPTS + 1):
        [leased] = _lease(database, report, now=leased_at)
        assert leased["attempt_count"] == attempt
        assert asyncio.run(
            database.complete_local_agent_job(
                job_id=job_id,
                user_id=LOCAL_DEV_USER_ID,
                attempt_count=attempt,
                status="failed",
                result={"retryable": True},
                error="the provider is unavailable",
                retryable=True,
                now=leased_at,
            )
        )
        leased_at += timedelta(days=1)

    # Four counted failures of five: the job is queued once more.
    job = asyncio.run(database.get_local_agent_job(job_id))
    assert (job["status"], job["attempt_count"]) == ("queued", LOCAL_AGENT_JOB_MAX_ATTEMPTS)


def test_the_daemon_ends_an_attempt_the_server_refused_under_a_changed_contract() -> None:
    import memforge.main as main
    from memforge.local_agent.runner import CloudJobLeaseLost

    with pytest.raises(CloudJobLeaseLost, match=ACQUISITION_CONTRACT_CHANGED):
        main._raise_if_local_agent_lease_not_current(
            {"error": "MemForge API request failed", "status_code": 409, "detail": ACQUISITION_CONTRACT_CHANGED}
        )
