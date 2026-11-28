"""Which device takes a document Source's run, and what a deploy that changes a contract does to it (ADR 0058).

A ``document_acquisition`` job goes only to the Source's execution device,
while its report names the Source ready and its acquisition contract for the
type equals the server's. Another device of the owner never takes it, even
one that could. A device behind or ahead of the server, or one not ready for
the Source, leaves the job queued without counting an attempt. The leased
attempt is the leasing device's: only it extends and completes it. A
submission made under a contract the server no longer has is refused, and
the job is queued again without counting the attempt.
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
from memforge.local_agent.device_report import DeviceSourceReadiness, NotReadyReason
from tests.device_report_fixture import DEVICE_ID, device_report
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
        enqueue_document_acquisition_job(database, database, store, source, workspace_id=WORKSPACE, trigger="schedule")
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
        pytest.param(device_report(), id="not-naming-the-source-yet"),
        pytest.param(
            device_report(not_ready={SOURCE_ID: DeviceSourceReadiness(reasons=(NotReadyReason(kind="credential"),))}),
            id="without-the-sources-credentials",
        ),
        pytest.param(
            device_report(
                not_ready={
                    SOURCE_ID: DeviceSourceReadiness(
                        reasons=(NotReadyReason(kind="location", field="root", path="/Users/alex/notes"),)
                    )
                }
            ),
            id="without-its-folder",
        ),
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


def test_another_device_of_the_owner_never_leases_the_job_even_when_it_could(owned) -> None:
    database, job_id = owned

    assert _lease(database, device_report([SOURCE_ID], device_id="dev-desktop")) == []

    [leased] = _lease(database, device_report([SOURCE_ID]))
    assert leased["job_id"] == job_id


def test_only_the_device_that_leased_an_attempt_extends_and_completes_it(owned) -> None:
    database, job_id = owned
    [leased] = _lease(database, device_report([SOURCE_ID]))

    def heartbeat(report) -> bool:
        return asyncio.run(
            database.heartbeat_local_agent_job(
                job_id=job_id, user_id=LOCAL_DEV_USER_ID, attempt_count=1, lease_seconds=60, device=report, now=NOW
            )
        )

    def complete(device_id: str) -> bool:
        return asyncio.run(
            database.complete_local_agent_job(
                job_id=job_id,
                user_id=LOCAL_DEV_USER_ID,
                device_id=device_id,
                attempt_count=1,
                status="succeeded",
                result={},
                error=None,
                now=NOW,
            )
        )

    assert leased["lease_device_id"] == DEVICE_ID
    assert heartbeat(device_report([SOURCE_ID], device_id="dev-desktop")) is False
    assert complete("dev-desktop") is False
    assert heartbeat(device_report([SOURCE_ID])) is True
    assert complete(DEVICE_ID) is True


def test_a_setup_job_that_names_a_device_goes_to_that_device_alone(owned) -> None:
    database, _job_id = owned
    asyncio.run(
        database.insert_local_agent_job(
            workspace_id="default",
            job_id="laj-pick",
            source_id="",
            source_type="local_markdown",
            operation="local_markdown_pick_root",
            payload={},
            created_by_user_id=LOCAL_DEV_USER_ID,
            execution_owner_user_id=LOCAL_DEV_USER_ID,
            device_id="dev-desktop",
        )
    )

    laptop = _lease(database, device_report())
    [desktop] = _lease(database, device_report(device_id="dev-desktop"))

    assert "laj-pick" not in [job["job_id"] for job in laptop]
    assert desktop["job_id"] == "laj-pick"


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
                device_id="dev-laptop",
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


def test_the_folder_picker_opens_on_the_execution_device_or_the_device_chosen_in_where_it_runs(tmp_path) -> None:
    from tests.device_report_fixture import connect_device
    from tests.test_device_acquisition import _connect as connect_database
    from tests.test_source_authority_api import _app

    headers = {"x-test-user": "alex", "x-test-workspace-role": "member"}

    database = connect_database(tmp_path / "picker.db")
    asyncio.run(
        database.upsert_source(
            id="src-notes",
            type="local_markdown",
            name="Notes",
            config_json='{"root": "/Users/alex/notes"}',
            access_policy="workspace",
            owner_user_id="alex",
            execution_owner_user_id="alex",
            executor="device",
            execution_device_id=DEVICE_ID,
        )
    )
    connect_device(database, "alex")
    connect_device(database, "alex", device_report(device_id="dev-desktop"))
    app = _app(tmp_path, database)

    def pick(**request) -> dict:
        response = client.post(
            "/api/v1/local-agent/jobs",
            headers=headers,
            json={"source_type": "local_markdown", "operation": "local_markdown_pick_root", "payload": {}, **request},
        )
        assert response.status_code == 201, response.text
        return asyncio.run(database.get_local_agent_job(response.json()["job_id"]))

    try:
        with TestClient(app) as client:
            on_execution_device = pick(source_id="src-notes")
            on_the_chosen_device = pick(source_id="src-notes", device_id="dev-desktop")
            for_a_new_source = pick(device_id="dev-desktop")
            not_theirs = client.post(
                "/api/v1/local-agent/jobs",
                headers=headers,
                json={
                    "source_type": "local_markdown",
                    "operation": "local_markdown_pick_root",
                    "payload": {},
                    "device_id": "dev-of-someone-else",
                },
            )
            unchosen = client.post(
                "/api/v1/local-agent/jobs",
                headers=headers,
                json={"source_type": "local_markdown", "operation": "local_markdown_pick_root", "payload": {}},
            )
            # Another member picks a folder only on their own device, chosen in Where it runs.
            by_another_member = client.post(
                "/api/v1/local-agent/jobs",
                headers={"x-test-user": "blake", "x-test-workspace-role": "member"},
                json={
                    "source_type": "local_markdown",
                    "operation": "local_markdown_pick_root",
                    "payload": {},
                    "source_id": "src-notes",
                },
            )
    finally:
        asyncio.run(database.close())

    assert on_execution_device["device_id"] == DEVICE_ID
    # Moving the Source picks its folder on the device it moves to.
    assert on_the_chosen_device["device_id"] == "dev-desktop"
    assert for_a_new_source["device_id"] == "dev-desktop"
    assert not_theirs.status_code == unchosen.status_code == 400
    assert by_another_member.status_code == 403, by_another_member.text
    assert by_another_member.json()["detail"]["code"] == "setup_job_execution_owner_forbidden"
