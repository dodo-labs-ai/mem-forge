"""A query manifest asks its device to confirm what it left out, and processing waits for the answers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

import memforge.server.admin_api as admin_api
from memforge.local_agent.document_identity import build_jira_doc_id
from memforge.local_agent.snapshot_manifest import (
    SnapshotConfirmation,
    SnapshotConfirmationAnswer,
    SnapshotManifestItem,
    snapshot_manifest_sha256,
)
from memforge.local_agent.source_contract import LOCAL_PACKAGE_CONTRACT_VERSION
from memforge.models import ContentItem
from memforge.source_item_failures import SourceItemFailure
from memforge.source_projection import SourceUnit
from memforge.storage.database import Database
from tests.test_local_adapter_api import (
    LeaseAwareTestClient,
    _config,
    _connect_database,
    _create_github_repo_source,
    _create_jira_source,
)

JOB = "job-confirm"
EMPTY_DIGEST = snapshot_manifest_sha256([])


class _Lease:
    """The leased job's state, as the job broker validates it for each request."""

    def __init__(self) -> None:
        self.current = True

    async def __call__(self, *args, **kwargs) -> bool:
        return self.current


class _HeldUnits:
    """The current Units of the Source, by Document id, in place of a sync that committed them."""

    def __init__(self, database: Database, units: dict[str, str]) -> None:
        self.units = dict(units)
        database.list_indexed_doc_ids = self.list_indexed_doc_ids
        database.find_source_unit_by_document_id = self.find_source_unit_by_document_id

    async def list_indexed_doc_ids(self, source_id: str) -> set[str]:
        return set(self.units)

    async def find_source_unit_by_document_id(self, source_id, document_id, *, current_only=False):
        provider_key = self.units.get(document_id)
        if provider_key is None:
            return None
        return SourceUnit(id=f"unit-{provider_key}", source_id=source_id, unit_type="jira_issue", provider_key=provider_key)


def _doc(source_id: str, key: str) -> str:
    return build_jira_doc_id(source_id=source_id, issue_key=key)


async def _fail(database: Database, source_id: str, doc_id: str, issue_key: str) -> None:
    failed_at = datetime(2026, 10, 23, tzinfo=timezone.utc)
    await database.upsert_source_item_failure(
        SourceItemFailure(
            source_id=source_id,
            document_id=doc_id,
            stage="normalize",
            error_class="ValueError",
            message="issue unreadable",
            retryable=True,
            release="test",
            failures_in_release=1,
            first_failed_at=failed_at,
            last_failed_at=failed_at,
            item=ContentItem(
                item_id=doc_id,
                title=issue_key,
                source_url=f"https://jira.example.test/browse/{issue_key}",
                last_modified=failed_at,
                extra={"issue_key": issue_key},
            ),
        )
    )


def _manifest(client, source_id: str, *, job: str = JOB, items=(), version=LOCAL_PACKAGE_CONTRACT_VERSION):
    body = {
        "items": [{"doc_id": doc_id, "revision": "r1", "change_kind": "upsert"} for doc_id in items],
        "coverage": "complete_snapshot",
        "sync_snapshot_id": f"{job}:attempt:1",
        "local_agent_job_id": job,
        "local_agent_attempt_count": 1,
    }
    if version is not None:
        body["package_contract_version"] = version
    return client.post(f"/api/v1/sources/{source_id}/adapter/manifest", json=body)


def _confirmations(client, source_id: str, *, job: str = JOB, attempt: int = 1, digest=EMPTY_DIGEST, **body):
    return client.post(
        f"/api/v1/sources/{source_id}/adapter/manifest/confirmations",
        json={
            "sync_snapshot_id": f"{job}:attempt:{attempt}",
            "local_agent_job_id": job,
            "local_agent_attempt_count": attempt,
            "manifest_sha256": digest,
            **body,
        },
    )


def _process(client, source_id: str, *, job: str = JOB):
    return client.post(
        f"/api/v1/sources/{source_id}/process",
        json={"sync_snapshot_id": f"{job}:attempt:1", "local_agent_job_id": job, "local_agent_attempt_count": 1},
    )


def _answers(confirm: list[dict[str, str]], *answers: str) -> list[dict[str, str]]:
    return [{**entry, "answer": answer} for entry, answer in zip(confirm, answers)]


@pytest.fixture
def jira(tmp_path):
    database = _connect_database(tmp_path)
    lease = _Lease()
    app = admin_api.create_admin_app(db=database, config=_config(tmp_path), local_agent_lease_validator=lease)
    try:
        with LeaseAwareTestClient(app) as client:
            source_id = _create_jira_source(client)["id"]
            yield client, database, source_id, lease
    finally:
        asyncio.run(database.close())


def test_a_query_manifest_asks_its_device_about_every_unlisted_unit_and_failed_issue(jira):
    client, database, source_id, _lease = jira
    listed, gone, failed_unit, failed_only = (_doc(source_id, key) for key in ("PAY-1", "PAY-2", "PAY-3", "PAY-5"))
    held = _HeldUnits(database, {listed: "10001", gone: "10002", failed_unit: "10003"})
    for doc_id, key in ((listed, "PAY-1"), (failed_unit, "PAY-3"), (failed_only, "PAY-5")):
        asyncio.run(_fail(database, source_id, doc_id, key))

    planned = _manifest(client, source_id, items=[listed])
    # A replay answers with the set the manifest was issued, even when the Source changed since.
    held.units[_doc(source_id, "PAY-9")] = "10009"
    replayed = _manifest(client, source_id, items=[listed])

    assert planned.status_code == 200, planned.text
    assert planned.json()["manifest_sha256"] == snapshot_manifest_sha256([SnapshotManifestItem(listed, "r1")])
    # A Unit by its provider key; a failed issue that never became a Unit by its stored issue key.
    expected = sorted(
        [
            {"doc_id": gone, "provider_key": "10002"},
            {"doc_id": failed_unit, "provider_key": "10003"},
            {"doc_id": failed_only, "provider_key": "PAY-5"},
        ],
        key=lambda entry: entry["doc_id"],
    )
    assert planned.json()["confirm"] == expected
    assert replayed.status_code == 200, replayed.text
    assert replayed.json()["confirm"] == expected


def test_processing_waits_for_the_answers_bound_to_the_manifest_and_its_attempt(jira):
    client, database, source_id, lease = jira
    _HeldUnits(database, {_doc(source_id, "PAY-2"): "10002", _doc(source_id, "PAY-3"): "10003"})
    confirm = _manifest(client, source_id).json()["confirm"]
    absent_exists = _answers(confirm, "absent", "exists")

    awaiting = _process(client, source_id)
    other_digest = _confirmations(client, source_id, digest="b" * 64, answers=absent_exists)
    other_attempt = _confirmations(client, source_id, attempt=2, answers=absent_exists)
    unissued = _confirmations(
        client,
        source_id,
        answers=[*absent_exists, {"doc_id": _doc(source_id, "PAY-7"), "provider_key": "10007", "answer": "absent"}],
    )
    incomplete = _confirmations(client, source_id, answers=absent_exists[:1])
    lease.current = False
    stale = _confirmations(client, source_id, answers=absent_exists)
    still_awaiting = asyncio.run(
        database.list_source_sync_snapshot_confirmations(
            source_id=source_id, workspace_id="local", snapshot_id=f"{JOB}:attempt:1"
        )
    )
    lease.current = True
    answered = _confirmations(client, source_id, answers=absent_exists)
    identical = _confirmations(client, source_id, answers=absent_exists)
    differing = _confirmations(client, source_id, answers=_answers(confirm, "absent", "hidden"))
    processed = _process(client, source_id)

    assert awaiting.status_code == 409
    assert awaiting.json()["detail"] == "source_snapshot_confirmations_pending"
    assert (other_digest.status_code, other_digest.json()["detail"]) == (
        409,
        "snapshot confirmations answer another manifest digest",
    )
    assert other_attempt.status_code == 409
    assert (unissued.status_code, unissued.json()["detail"]) == (
        409,
        f"snapshot confirmation was not issued: {_doc(source_id, 'PAY-7')}",
    )
    assert incomplete.status_code == 409
    assert (stale.status_code, stale.json()["detail"]) == (409, "local_agent_lease_not_current")
    assert all(item.answer is None for item in still_awaiting)
    assert answered.status_code == 200, answered.text
    assert answered.json()["answered_count"] == 2
    assert identical.status_code == 200, identical.text
    assert (differing.status_code, differing.json()["detail"]) == (
        409,
        "snapshot confirmations were already answered differently",
    )
    assert processed.status_code == 202, processed.text
    stored = asyncio.run(
        database.list_source_sync_snapshot_confirmations(
            source_id=source_id, workspace_id="local", snapshot_id=f"{JOB}:attempt:1"
        )
    )
    assert [item.answer for item in stored] == [SnapshotConfirmationAnswer.ABSENT, SnapshotConfirmationAnswer.EXISTS]


def test_a_device_that_could_not_confirm_sends_its_cause_and_processing_proceeds(jira):
    client, database, source_id, _lease = jira
    _HeldUnits(database, {_doc(source_id, "PAY-2"): "10002", _doc(source_id, "PAY-3"): "10003"})
    confirm = _manifest(client, source_id).json()["confirm"]

    failed = _confirmations(client, source_id, failure_cause="HTTP 401 from Jira")
    with_answers = _confirmations(
        client, source_id, failure_cause="HTTP 401 from Jira", answers=_answers(confirm, "absent", "exists")
    )
    processed = _process(client, source_id)

    assert failed.status_code == 200, failed.text
    assert failed.json()["failed"] is True
    assert with_answers.status_code == 409
    assert processed.status_code == 202, processed.text
    stored = asyncio.run(
        database.list_source_sync_snapshot_confirmations(
            source_id=source_id, workspace_id="local", snapshot_id=f"{JOB}:attempt:1"
        )
    )
    assert stored == tuple(
        SnapshotConfirmation(entry["doc_id"], entry["provider_key"], SnapshotConfirmationAnswer.FAILED, "HTTP 401 from Jira")
        for entry in confirm
    )


@pytest.mark.parametrize("version", [None, 2])
def test_an_agent_that_does_not_confirm_is_asked_nothing_and_not_held(jira, version):
    client, database, source_id, _lease = jira
    _HeldUnits(database, {_doc(source_id, "PAY-2"): "10002"})

    planned = _manifest(client, source_id, version=version)
    processed = _process(client, source_id)

    assert planned.status_code == 200, planned.text
    assert planned.json()["confirm"] == []
    assert processed.status_code == 202, processed.text


def test_a_confirm_set_larger_than_a_manifest_is_not_issued(jira, monkeypatch):
    client, database, source_id, _lease = jira
    monkeypatch.setattr(admin_api, "LOCAL_SOURCE_MANIFEST_MAX_ITEMS", 1)
    _HeldUnits(database, {_doc(source_id, "PAY-2"): "10002", _doc(source_id, "PAY-3"): "10003"})

    planned = _manifest(client, source_id)
    processed = _process(client, source_id)

    # The run then records absence_confirmation_missing and removes nothing.
    assert planned.json()["confirm"] == []
    assert processed.status_code == 202, processed.text


def test_an_existence_manifest_asks_nothing(tmp_path):
    database = _connect_database(tmp_path)
    app = admin_api.create_admin_app(db=database, config=_config(tmp_path), local_agent_lease_validator=_Lease())
    try:
        with LeaseAwareTestClient(app) as client:
            source_id = _create_github_repo_source(client)["id"]
            _HeldUnits(database, {"github-doc-gone": "repo:docs/gone.md"})
            planned = _manifest(client, source_id)
        assert planned.status_code == 200, planned.text
        assert planned.json()["confirm"] == []
    finally:
        asyncio.run(database.close())
