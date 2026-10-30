"""The device executor asks the server what to collect, uploads only what the server lacks, and hands the attempt over."""

from __future__ import annotations

from datetime import timedelta

import pytest

from memforge.main import _sync_progress_snapshot
from memforge import main
from memforge.conversations.layer import ConversationLayer, ConversationSelection
from memforge.conversations.manifest import request_to_payload
from memforge.local_agent import conversation_executor
from memforge.local_agent.conversation_executor import (
    ConversationUploadError,
    DeviceCollectionResult,
    RecordSchemaMismatch,
    run_device_collection,
)
from memforge.local_agent.runner import CloudJobLeaseLost
from memforge.source_adapters.teams import MESSAGE_SCHEMA
from tests.test_conversation_source_sync import CHAT, NOW, FakeTeams


class Server:
    """The server side of the upload: it asks for a body only when it does not hold the message."""

    def __init__(
        self, *, held: set[str] = frozenset(), refuse: str | None = None, reads: str | None = MESSAGE_SCHEMA.qualified_name
    ) -> None:
        self.held = held
        self.refuse = refuse
        self.reads = reads
        self.calls: list[tuple[str, dict]] = []
        self.manifest: dict | None = None
        self.bodies: list[dict] = []

    def conversation_collection_request(self, **attempt):
        self.calls.append(("request", attempt))
        selection = ConversationSelection(conversation_ids=(CHAT,), initial_history_days=30)
        request = ConversationLayer.collection_request(selection=selection, now=NOW)
        answer = {"collection_attempt_id": "job-1:attempt:1", "request": request_to_payload(request)}
        if self.reads is not None:
            answer["record_schema"] = self.reads
        return answer

    def submit_conversation_manifest(self, *, source_id, manifest, **attempt):
        self.calls.append(("manifest", attempt))
        if self.refuse == "manifest":
            return {"error": "Conflict", "status_code": 409, "detail": "local_agent_lease_not_current"}
        self.manifest = manifest
        return {
            "missing": [
                {
                    "conversation_id": message["conversation_id"],
                    "message_id": message["message_id"],
                    "body": message["message_id"] not in self.held,
                    "artifacts": message["artifacts"],
                }
                for message in manifest["messages"]
                if message["message_id"] not in self.held or message["artifacts"]
            ]
        }

    def submit_conversation_bodies(self, *, source_id, bodies, **attempt):
        self.calls.append(("bodies", {"count": len(bodies), "ids": [body["message_id"] for body in bodies]}))
        self.bodies.extend(bodies)
        return {"received": len(bodies)}

    def plan_conversation_collection(self, *, source_id, **attempt):
        self.calls.append(("plan", attempt))
        return {"collection_attempt_id": "job-1:attempt:1", "units": []}

    def start_source_processing(self, *, source_id, sync_snapshot_id, **attempt):
        self.calls.append(("process", {"sync_snapshot_id": sync_snapshot_id}))
        if self.refuse == "process":
            return {"error": "Conflict", "status_code": 409, "detail": "conversation_bodies_incomplete"}
        return {"run_id": "run-1"}

    def push_source_artifact(self, *, content, **descriptor):
        self.calls.append(("artifact", descriptor))
        return {"input_sha256": "a" * 64}


def _teams(count: int) -> FakeTeams:
    teams = FakeTeams()
    for index in range(count):
        teams.post(CHAT, f"m{index:02d}", NOW - timedelta(minutes=10 + index), f"message {index}")
    return teams


async def _run(teams: FakeTeams, server: Server, *, dry_run: bool = False, report=None) -> DeviceCollectionResult:
    return await run_device_collection(
        teams,
        server,
        source_id="src-teams",
        local_agent_job_id="job-1",
        local_agent_attempt_count=1,
        config_revision="cfg",
        dry_run=dry_run,
        **({"report": report} if report is not None else {}),
    )


async def test_the_manifest_goes_first_and_only_missing_bodies_follow():
    server = Server(held={"m00", "m01"})

    result = await _run(_teams(4), server)

    assert [step for step, _ in server.calls] == ["request", "manifest", "bodies", "process"]
    assert server.calls[2][1]["ids"] == ["m02", "m03"]
    assert server.calls[3][1] == {"sync_snapshot_id": "job-1:attempt:1"}
    assert "received_at" not in server.manifest
    assert result == DeviceCollectionResult(
        collection_attempt_id="job-1:attempt:1",
        conversations=1,
        messages=4,
        bodies_uploaded=2,
        processing={"run_id": "run-1"},
    )


async def test_every_progress_report_is_a_valid_sync_progress_snapshot():
    snapshots = []

    def report(phase, completed, total):
        snapshots.append(_sync_progress_snapshot(phase=phase, completed=completed, total=total, unit="message"))

    await _run(_teams(3), Server(), report=report)

    phases = [snapshot["phase"] for snapshot in snapshots]
    assert phases[0] == "fetching"
    assert set(phases[1:]) == {"uploading"}


async def test_a_dry_run_asks_for_the_plan_instead_of_processing():
    server = Server()

    result = await _run(_teams(1), server, dry_run=True)

    assert [step for step, _ in server.calls] == ["request", "manifest", "bodies", "plan"]
    assert result.plan == {"collection_attempt_id": "job-1:attempt:1", "units": []}
    assert result.processing is None


async def test_processing_the_server_refuses_fails_the_attempt():
    with pytest.raises(ConversationUploadError, match="processing start failed"):
        await _run(_teams(1), Server(refuse="process"))


async def test_bodies_upload_in_batches(monkeypatch):
    monkeypatch.setattr(conversation_executor, "BODY_BATCH_SIZE", 2)
    server = Server()

    await _run(_teams(5), server)

    assert [details["count"] for step, details in server.calls if step == "bodies"] == [2, 2, 1]


@pytest.mark.parametrize(
    ("reads", "reason"),
    [
        (f"teams-message:{MESSAGE_SCHEMA.version + 1}", "^local_agent_upgrade_required: .*`memforge daemon restart`$"),
        (f"teams-message:{MESSAGE_SCHEMA.version - 1}", "^server_upgrade_required: .*the server reads the earlier teams-message:"),
        ("conversation-message:1", f"collects messages as {MESSAGE_SCHEMA.qualified_name} and the server reads "
                                   "conversation-message:1$"),
    ],
    ids=["server newer", "server older", "other schema"],
)
async def test_a_request_in_another_record_schema_fails_before_polling(reads, reason):
    teams = _teams(2)
    server = Server(reads=reads)

    async def never_polled(*args, **kwargs):
        raise AssertionError("polled a conversation the server cannot read")

    teams.poll = never_polled

    with pytest.raises(RecordSchemaMismatch, match=reason):
        await _run(teams, server)
    assert [step for step, _ in server.calls] == ["request"]


async def test_a_server_that_names_no_record_schema_leaves_the_check_to_manifest_intake():
    server = Server(reads=None)

    await _run(_teams(1), server)

    assert [step for step, _ in server.calls] == ["request", "manifest", "bodies", "process"]


async def test_a_refused_upload_stops_the_attempt():
    with pytest.raises(ConversationUploadError, match="manifest upload failed"):
        await _run(_teams(1), Server(refuse="manifest"))


def test_a_record_schema_mismatch_fails_the_job_without_a_retry(monkeypatch):
    async def mismatched(job, payload, source_id, client, report_progress):
        raise RecordSchemaMismatch("local_agent_upgrade_required: this local daemon collects messages as teams-message:3")

    monkeypatch.setattr(main, "_collect_conversation_on_device", mismatched)

    class Client:
        target = main.build_target(origin="http://localhost:8000", edition="oss")

    result = main._run_cloud_teams_sync_job(
        {"job_id": "job-1", "attempt_count": 1, "operation": "teams_sync", "source_id": "src-teams", "payload": {}},
        Client(),
    )

    assert result["retryable"] is False
    assert result["error"].startswith("local_agent_upgrade_required: ")


def test_a_lease_the_server_no_longer_honours_ends_the_job(monkeypatch):
    async def refused(job, payload, source_id, client, report_progress):
        raise ConversationUploadError(
            "manifest upload", {"error": "Conflict", "status_code": 409, "detail": "local_agent_lease_not_current"}
        )

    monkeypatch.setattr(main, "_collect_conversation_on_device", refused)

    class Client:
        target = main.build_target(origin="http://localhost:8000", edition="oss")

    with pytest.raises(CloudJobLeaseLost):
        main._run_cloud_teams_sync_job(
            {"job_id": "job-1", "attempt_count": 1, "operation": "teams_sync", "source_id": "src-teams", "payload": {}},
            Client(),
        )


async def test_missing_images_are_uploaded_before_the_bodies_that_name_them():
    teams = FakeTeams()
    teams.post(CHAT, "m1", NOW - timedelta(minutes=5), "diagram", images=("hosted-1", "hosted-2"))
    teams.unavailable_images.add("hosted-2")
    server = Server()

    result = await _run(teams, server)

    assert [step for step, _ in server.calls] == ["request", "manifest", "artifact", "bodies", "process"]
    assert server.calls[2][1]["provider_key"] == "m1/hostedContents/hosted-1"
    assert server.calls[2][1]["parent_provider_key"] == "m1"
    assert server.calls[2][1]["source_unit_key"] == CHAT
    assert server.bodies[0]["artifacts"] == [
        {"provider_key": "m1/hostedContents/hosted-1", "provider_revision": "m1:hosted-1", "input_sha256": "a" * 64}
    ]
    assert (result.artifacts_uploaded, result.artifacts_unavailable) == (1, 1)
