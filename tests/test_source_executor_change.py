"""Changing where a document Source runs (ADR 0058, design §10.8).

The executor is a Source property outside its scope. Changing it opens no
scope transition and keeps what the Source holds; it cancels the Source's
pending run and queued sync jobs and waits for one that runs. Moving to the
device deletes the server's copies of the Source's secrets and makes the
changer the execution owner and one of their devices its execution device;
moving to another device does the same, and the previous device's next lease
answer no longer names the Source; moving to the server requires the
secrets.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

import memforge.server.admin_api as admin_api
from memforge.config import AppConfig
from memforge.local_agent.source_contract import DOCUMENT_ACQUISITION_OPERATION
from memforge.models import SyncState
from memforge.source_activity import SOURCE_SYNC_CANCELLED_BY_SOURCE_CHANGE
from memforge.source_item_failures import SourceItemFailure
from memforge.source_types import source_type_descriptor
from memforge.storage.database import Database
from tests.device_report_fixture import DEVICE_ID, connect_device, device_report

PRINCIPAL_HEADER = "X-Test-Principal"
CREATOR = "owner-a"
CHANGER = "member-b"
TEST_SOURCE_KEY = "VV4JjZLLr2BcgRnhV90gCnxzchn43M900VQy3dXJI30="
CONFLUENCE = {
    "base_url": "https://wiki.example.com",
    "sync_mode": "space",
    "spaces": ["ACME"],
    "pat": "server-held-token",
}
CURSOR = datetime(2026, 11, 20, 9, 0, tzinfo=timezone.utc)
DESKTOP = "dev-desktop"


def _principal(request: Request) -> str:
    return request.headers.get(PRINCIPAL_HEADER, CREATOR)


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)


@pytest.fixture
def database(tmp_path):
    database = Database(str(tmp_path / "sources.db"))
    asyncio.run(database.connect())
    for user in (CREATOR, CHANGER):
        connect_device(database, user)
        connect_device(database, user, device_report(device_id=DESKTOP, name="Desktop"))
    try:
        yield database
    finally:
        asyncio.run(database.close())


@pytest.fixture
def client(tmp_path, database):
    config = AppConfig(base_dir=tmp_path / "memforge")
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    app = admin_api.create_admin_app(
        db=database,
        config=config,
        principal_resolver=_principal,
        workspace_role_resolver=lambda request: "workspace_admin",
    )
    with TestClient(app) as test_client:
        yield test_client


def _create(client, config: dict | None = None, source_type: str = "confluence") -> str:
    created = client.post(
        "/api/v1/sources",
        json={"type": source_type, "name": "Team wiki", "config": config or CONFLUENCE, "access_policy": "workspace"},
    )
    assert created.status_code == 200, created.text
    return created.json()["id"]


def _change(
    client,
    source_id: str,
    executor: str,
    *,
    by: str = CHANGER,
    secrets: dict | None = None,
    device: str | None = DEVICE_ID,
):
    body: dict = {"executor": executor}
    if executor == "device":
        body["execution_device_id"] = device
    if secrets is not None:
        body["secrets"] = secrets
    return client.put(f"/api/v1/sources/{source_id}/executor", json=body, headers={PRINCIPAL_HEADER: by})


def _queue_job(database: Database, source_id: str, job_id: str) -> None:
    asyncio.run(
        database.enqueue_local_agent_job(
            job_id=job_id,
            source_id=source_id,
            source_type="confluence",
            operation=DOCUMENT_ACQUISITION_OPERATION,
            payload={"source_id": source_id},
            created_by_user_id=CREATOR,
            execution_owner_user_id=CREATOR,
        )
    )


def _job_status(database: Database, job_id: str) -> tuple[str, str | None]:
    async def read():
        async with database.db.execute(
            "SELECT status, last_error FROM local_agent_jobs WHERE job_id = ?", (job_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return str(row["status"]), row["last_error"]

    return asyncio.run(read())


def _holdings(database: Database, source_id: str) -> tuple:
    """What a Source holds that an executor change keeps."""

    async def read():
        transition = await database.get_open_projection_scope_transition(source_id)
        state = await database.get_sync_state(source_id)
        return (
            sorted(doc.doc_id for doc in await database.list_documents(source=source_id, limit=100)),
            [failure.document_id for failure in await database.list_source_item_failures(source_id=source_id)],
            state.last_sync_at if state is not None else None,
            transition.id if transition is not None else None,
        )

    return asyncio.run(read())


def _hold_something(database: Database, source_id: str) -> None:
    async def write():
        await database.upsert_sync_state(
            SyncState(source=source_id, last_sync_at=CURSOR, last_sync_status="success", docs_processed=1)
        )
        await database.upsert_source_item_failure(
            SourceItemFailure(
                source_id=source_id,
                document_id="confluence-7",
                stage="fetch",
                error_class="builtins.RuntimeError",
                message="the wiki could not serve page 7",
                retryable=True,
                release="test",
                failures_in_release=1,
                first_failed_at=CURSOR,
                last_failed_at=CURSOR,
                title="Page 7",
            )
        )

    asyncio.run(write())


def test_a_document_type_runs_on_both_executors_unless_its_items_exist_only_on_the_device() -> None:
    assert source_type_descriptor("confluence").executors == {"server", "device"}
    assert source_type_descriptor("notion").executors == {"server", "device"}
    assert source_type_descriptor("github_pages").executors == {"server", "device"}
    assert source_type_descriptor("local_markdown").executors == {"device"}


def test_moving_to_the_device_deletes_the_servers_secrets_and_makes_the_changer_the_owner(client, database):
    source_id = _create(client)
    assert asyncio.run(database.get_source(source_id))["config"]["pat_encrypted"]

    moved = _change(client, source_id, "device")

    assert moved.status_code == 200, moved.text
    assert moved.json() == {
        "id": source_id,
        "executor": "device",
        "execution_owner_user_id": CHANGER,
        "execution_device_id": DEVICE_ID,
        "cancelled_job_ids": [],
    }
    source = asyncio.run(database.get_source(source_id))
    assert source["executor"] == "device"
    assert source["execution_owner_user_id"] == CHANGER
    assert source["execution_device_id"] == DEVICE_ID
    assert not any(key.startswith("pat") for key in source["config"])
    assert {key: source["config"][key] for key in ("base_url", "sync_mode", "spaces")} == {
        "base_url": "https://wiki.example.com",
        "sync_mode": "space",
        "spaces": ["ACME"],
    }
    assert [s["id"] for s in asyncio.run(database.list_device_sources(CHANGER))] == [source_id]


def test_the_change_opens_no_transition_and_keeps_what_the_source_holds(client, database):
    source_id = _create(client)
    _hold_something(database, source_id)
    widened = client.put(
        f"/api/v1/sources/{source_id}", json={"config": {**CONFLUENCE, "spaces": ["ACME", "OPS"]}}
    )
    assert widened.status_code == 200, widened.text
    before = _holdings(database, source_id)
    assert before[1:] == (["confluence-7"], CURSOR, before[3]) and before[3] is not None

    assert _change(client, source_id, "device").status_code == 200
    assert _holdings(database, source_id) == before
    assert _change(client, source_id, "server", secrets={"pat": "new-server-token"}).status_code == 200
    assert _holdings(database, source_id) == before


def test_moving_to_the_server_requires_the_secrets_the_form_requires(client, database):
    source_id = _create(client)
    assert _change(client, source_id, "device").status_code == 200

    missing = _change(client, source_id, "server")
    assert missing.status_code == 400
    assert missing.json()["detail"]["code"] == "config_invalid"
    assert [field["key"] for field in missing.json()["detail"]["fields"]] == ["pat"]
    assert asyncio.run(database.get_source(source_id))["executor"] == "device"

    moved = _change(client, source_id, "server", secrets={"pat": "new-server-token"})

    assert moved.status_code == 200, moved.text
    source = asyncio.run(database.get_source(source_id))
    assert source["executor"] == "server"
    assert source["config"]["pat_encrypted"] and source["config"]["pat_configured"] is True
    assert "new-server-token" not in json.dumps(source["config"])
    assert source["execution_device_id"] is None
    assert asyncio.run(database.list_device_sources(CHANGER)) == []


@pytest.mark.parametrize(
    ("executor", "secrets", "message"),
    [
        ("device", {"pat": "a-token"}, "Secrets are sent only when a source moves to the server."),
        ("server", {"pat": "a-token"}, "Secrets are sent only when a source moves to the server."),
    ],
    ids=["to-device", "staying-on-the-server"],
)
def test_secrets_are_refused_unless_the_source_moves_to_the_server(client, database, executor, secrets, message):
    source_id = _create(client)

    refused = _change(client, source_id, executor, secrets=secrets)

    assert (refused.status_code, refused.json()["detail"]) == (
        400,
        {"code": "executor_change_invalid", "message": message},
    )
    assert asyncio.run(database.get_source(source_id))["executor"] == "server"


def test_a_field_that_is_not_a_secret_is_refused(client, database):
    source_id = _create(client)
    assert _change(client, source_id, "device").status_code == 200

    refused = _change(client, source_id, "server", secrets={"pat": "a-token", "spaces": "OPS"})

    assert refused.status_code == 400
    assert refused.json()["detail"]["message"] == "Not a secret of a confluence source: spaces."


def test_a_device_source_saves_its_configuration_without_secrets(client, database):
    source_id = _create(client)
    assert _change(client, source_id, "device").status_code == 200
    configuration = {key: value for key, value in CONFLUENCE.items() if key != "pat"}

    saved = client.put(
        f"/api/v1/sources/{source_id}",
        json={"config": {**configuration, "exclude_labels": ["draft"]}},
        headers={PRINCIPAL_HEADER: CHANGER},
    )
    with_secret = client.put(
        f"/api/v1/sources/{source_id}", json={"config": CONFLUENCE}, headers={PRINCIPAL_HEADER: CHANGER}
    )

    assert saved.status_code == 200, saved.text
    assert with_secret.status_code == 400
    assert [field["key"] for field in with_secret.json()["detail"]["fields"]] == ["pat"]
    assert not any(key.startswith("pat") for key in asyncio.run(database.get_source(source_id))["config"])


def test_the_change_cancels_the_sources_queued_jobs(client, database):
    source_id = _create(client)
    _queue_job(database, source_id, "job-queued")

    moved = _change(client, source_id, "device")

    assert moved.status_code == 200, moved.text
    assert moved.json()["cancelled_job_ids"] == ["job-queued"]
    assert _job_status(database, "job-queued") == ("failed", SOURCE_SYNC_CANCELLED_BY_SOURCE_CHANGE)


def test_the_change_waits_while_a_device_holds_the_acquisition_job(client, database):
    source_id = _create(client)
    assert _change(client, source_id, "device", by=CREATOR).status_code == 200
    _queue_job(database, source_id, "job-leased")
    leased = asyncio.run(
        database.lease_local_agent_jobs(
            user_id=CREATOR,
            device=device_report([source_id]),
            acquisition_contracts={"confluence": source_type_descriptor("confluence").acquisition_contract},
            limit=1,
            lease_seconds=60,
        )
    )
    assert [job["job_id"] for job in leased] == ["job-leased"]

    refused = _change(client, source_id, "server", secrets={"pat": "a-token"})

    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "source_sync_running"
    assert refused.json()["detail"]["local_agent_job_id"] == "job-leased"
    assert asyncio.run(database.get_source(source_id))["executor"] == "device"
    assert _job_status(database, "job-leased") == ("leased", None)


def test_the_change_waits_while_a_sync_runs_and_cancels_a_pending_one(client, database):
    source_id = _create(client)
    pending = asyncio.run(database.enqueue_source_sync_run(source_id=source_id))

    assert _change(client, source_id, "device").status_code == 200
    assert asyncio.run(database.get_source_sync_run(pending.run_id)).status == "failed"

    running = asyncio.run(database.enqueue_source_sync_run(source_id=source_id))
    leased = asyncio.run(database.lease_next_source_sync_run(worker_id="worker-a", lease_seconds=60))
    assert leased is not None and leased.run_id == running.run_id

    refused = _change(client, source_id, "server", secrets={"pat": "a-token"})

    assert refused.status_code == 409
    assert refused.json()["detail"]["code"] == "source_sync_running"
    assert refused.json()["detail"]["run_id"] == running.run_id
    assert asyncio.run(database.get_source(source_id))["executor"] == "device"


def test_an_executor_the_type_does_not_declare_is_refused(client, database, monkeypatch):
    from dataclasses import replace

    from memforge import source_types

    source_id = _create(client)
    monkeypatch.setitem(
        source_types._DESCRIPTORS,
        "confluence",
        replace(source_type_descriptor("confluence"), executors=frozenset({"server"})),
    )

    refused = _change(client, source_id, "device")

    assert refused.status_code == 400
    assert refused.json()["detail"] == {
        "code": "executor_change_invalid",
        "message": "A confluence source runs on: server.",
    }


def _executed_by(client, user: str, device_id: str) -> list[str]:
    """The Sources the lease answer tells ``device_id`` of ``user`` that it executes."""

    answer = client.post(
        "/api/cloud/local-agent/jobs/lease",
        headers={PRINCIPAL_HEADER: user},
        json={"limit": 1, "device": device_report(device_id=device_id).to_payload()},
    )
    assert answer.status_code == 200, answer.text
    return [source["id"] for source in answer.json()["sources"]]


def test_moving_to_another_device_is_a_change_and_the_old_device_no_longer_runs_the_source(client, database):
    source_id = _create(client)
    assert _change(client, source_id, "device").status_code == 200
    assert _executed_by(client, CHANGER, DEVICE_ID) == [source_id]
    _queue_job(database, source_id, "job-queued")

    moved = _change(client, source_id, "device", device=DESKTOP)

    assert moved.status_code == 200, moved.text
    assert moved.json()["execution_device_id"] == DESKTOP
    assert moved.json()["cancelled_job_ids"] == ["job-queued"]
    assert _executed_by(client, CHANGER, DEVICE_ID) == []
    assert _executed_by(client, CHANGER, DESKTOP) == [source_id]


def test_choosing_their_own_device_makes_another_member_the_execution_owner(client, database):
    source_id = _create(client)
    assert _change(client, source_id, "device").status_code == 200

    taken = _change(client, source_id, "device", by=CREATOR)

    assert taken.status_code == 200, taken.text
    assert (taken.json()["execution_owner_user_id"], taken.json()["execution_device_id"]) == (CREATOR, DEVICE_ID)
    assert _executed_by(client, CHANGER, DEVICE_ID) == []
    assert _executed_by(client, CREATOR, DEVICE_ID) == [source_id]


def test_a_device_that_is_not_the_choosers_own_is_refused(client, database):
    source_id = _create(client)

    unknown = _change(client, source_id, "device", device="dev-someone-elses")
    unchosen = _change(client, source_id, "device", device=None)

    assert unknown.status_code == unchosen.status_code == 400
    assert unknown.json()["detail"]["code"] == unchosen.json()["detail"]["code"] == "execution_device_invalid"
    assert asyncio.run(database.get_source(source_id))["executor"] == "server"


def test_the_same_device_for_the_same_owner_changes_nothing(client, database):
    source_id = _create(client)
    assert _change(client, source_id, "device").status_code == 200
    _queue_job(database, source_id, "job-queued")

    unchanged = _change(client, source_id, "device")

    assert unchanged.status_code == 200
    assert unchanged.json()["cancelled_job_ids"] == []
    assert _job_status(database, "job-queued") == ("queued", None)


def test_removing_a_device_leaves_its_sources_without_an_execution_device(client, database):
    source_id = _create(client)
    assert _change(client, source_id, "device").status_code == 200
    _queue_job(database, source_id, "job-queued")

    removed = client.delete(f"/api/cloud/local-agent/devices/{DEVICE_ID}", headers={PRINCIPAL_HEADER: CHANGER})
    again = client.delete(f"/api/cloud/local-agent/devices/{DEVICE_ID}", headers={PRINCIPAL_HEADER: CHANGER})
    synced = client.post(f"/api/v1/sources/{source_id}/sync", headers={PRINCIPAL_HEADER: CHANGER})

    assert removed.json() == {"device_id": DEVICE_ID, "source_ids": [source_id]}
    assert again.status_code == 404
    assert asyncio.run(database.get_source(source_id))["execution_device_id"] is None
    assert _job_status(database, "job-queued") == ("failed", SOURCE_SYNC_CANCELLED_BY_SOURCE_CHANGE)
    assert (synced.status_code, synced.json()["detail"]["code"]) == (409, "execution_device_required")


def test_the_same_executor_changes_nothing(client, database):
    source_id = _create(client)
    _queue_job(database, source_id, "job-queued")

    unchanged = _change(client, source_id, "server")

    assert unchanged.status_code == 200
    assert unchanged.json()["cancelled_job_ids"] == []
    assert _job_status(database, "job-queued") == ("queued", None)
    assert asyncio.run(database.get_source(source_id))["config"]["pat_encrypted"]


def test_only_a_source_manager_changes_the_executor(tmp_path, database):
    config = AppConfig(base_dir=tmp_path / "memforge-members")
    config.sync.scheduler_enabled = False
    config.sync.worker_enabled = False
    app = admin_api.create_admin_app(
        db=database,
        config=config,
        principal_resolver=_principal,
        workspace_role_resolver=lambda request: "member",
    )
    with TestClient(app) as members:
        source_id = _create(members)
        refused = _change(members, source_id, "device", by=CHANGER)

    assert refused.status_code == 403
    assert asyncio.run(database.get_source(source_id))["executor"] == "server"
