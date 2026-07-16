from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.memory.lifecycle_plan import (
    CutoverFindingReason,
    CutoverFindingStatus,
    LifecycleBackfillJob,
    LifecycleBackfillJobStatus,
    LifecycleCutoverFinding,
    LifecycleGate,
    LifecycleGateState,
)
from memforge.models import DocumentRecord, SyncState
from memforge.local_adapter import build_teams_doc_id
from memforge.local_agent.teams_ledger import build_teams_window_id
from memforge.local_agent.source_contract import (
    local_agent_semantic_input_sha256,
    local_agent_source_config_revision,
)
from memforge.pipeline.sync import SourceSyncMode
from memforge.runtime import DefaultRuntimeProvider
from memforge.server.admin_api import create_admin_app
from memforge.storage.database import Database
from memforge.storage.document_store import LocalDocumentStore


def _config(tmp_path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "memforge")
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    cfg.sync.worker_enabled = False
    return cfg


def _teams_doc_id() -> str:
    return build_teams_doc_id(source_id="src-teams", window_id=_teams_window_id())


def _teams_window_id() -> str:
    return build_teams_window_id(
        source_id="src-teams",
        conversation_id="19:conversation-a@example.test",
        root_or_anchor_message_id="message-a",
        window_type="time_block",
    )


def _canonical_payload_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def _teams_document(now: datetime) -> DocumentRecord:
    return DocumentRecord(
        doc_id=_teams_doc_id(),
        source="src-teams",
        source_url="https://teams.example.test/conversation-a",
        title="Conversation A",
        space_or_project="SFPAY",
        author=None,
        last_modified=now,
        labels=[],
        version="1",
        content_hash="hash-window-a",
        token_count=10,
        raw_content_uri=None,
        raw_content_type=None,
        normalized_content_uri=None,
        pdf_content_uri=None,
        last_synced=now,
    )


async def _setup_completed_teams_local_replay(
    database: Database,
    *,
    document_store: LocalDocumentStore,
    snapshot_id: str | None,
    include_current_input: bool = True,
    include_current_document: bool = True,
    include_retired_input: bool = False,
    include_newer_noncurrent_version: bool = False,
) -> dict[str, str]:
    await database.connect()
    await database.upsert_source(
        id="src-teams",
        type="teams",
        name="Teams",
        config_json='{"conversation_ids":["19:conversation-a@example.test"]}',
        access_policy="workspace",
        owner_user_id="user-a",
    )
    now = datetime.now(timezone.utc)
    if include_current_document:
        await database.upsert_document(
            _teams_document(now),
            require_configured_source=True,
        )
    artifacts: dict[str, str] = {}
    if include_retired_input:
        retired_uri = document_store.store_raw(
            "src-teams",
            "Retired Conversation Window",
            json.dumps(
                {
                    "package_kind": "teams_window_document",
                    "doc_id": "teams-window-retired",
                    "version": "1",
                }
            ).encode(),
            "application/json",
            extension=".local-package.json",
        )
        await database.create_source_sync_input(
            source_id="src-teams",
            raw_uri=retired_uri,
            raw_sha256="sha-retired-window",
            raw_content_type="application/json",
            metadata={
                "manifest_entry": {
                    "doc_id": "teams-window-retired",
                    "title": "Retired Conversation Window",
                    "version": "1",
                }
            },
        )
        artifacts["retired"] = retired_uri
    if include_current_input:
        raw_payload = {
            "conversation_id": "19:conversation-a@example.test",
            "window_id": _teams_window_id(),
            "messages": [
                {
                    "id": "message-a",
                    "content": "Current decision",
                    "time": "2026-07-16T09:00:00+00:00",
                }
            ]
        }
        payload_hash = _canonical_payload_hash(raw_payload)
        package_body = json.dumps(
            {
                "package_kind": "teams_window_document",
                "doc_id": _teams_doc_id(),
                "version": "1",
                "revision_hash": "1",
                "raw_hash": payload_hash,
                "semantic_hash": payload_hash,
                "conversation_id": "19:conversation-a@example.test",
                "window_id": _teams_window_id(),
                "root_message_id": "message-a",
                "window_type": "time_block",
                "raw_payload": raw_payload,
            }
        ).encode()
        current_uri = document_store.store_raw(
            "src-teams",
            "Conversation A",
            package_body,
            "application/json",
            extension=".local-package.json",
        )
        await database.create_source_sync_input(
            source_id="src-teams",
            raw_uri=current_uri,
            raw_sha256=local_agent_semantic_input_sha256(
                _teams_doc_id(),
                payload_hash,
            ),
            raw_content_type="application/json",
            sync_snapshot_id=snapshot_id,
            metadata={
                "manifest_entry": {
                    "doc_id": _teams_doc_id(),
                    "title": "Conversation A",
                    "version": "1",
                },
                "package_sha256": hashlib.sha256(package_body).hexdigest(),
            },
        )
        artifacts["current"] = current_uri
    if include_newer_noncurrent_version:
        newer_payload = {
            "conversation_id": "19:conversation-a@example.test",
            "window_id": _teams_window_id(),
            "messages": [
                {
                    "id": "message-a",
                    "content": "Temporary newer decision",
                    "time": "2026-07-16T10:00:00+00:00",
                }
            ],
        }
        newer_hash = _canonical_payload_hash(newer_payload)
        newer_body = json.dumps(
            {
                "package_kind": "teams_window_document",
                "doc_id": _teams_doc_id(),
                "version": "2",
                "revision_hash": "2",
                "raw_hash": newer_hash,
                "semantic_hash": newer_hash,
                "conversation_id": "19:conversation-a@example.test",
                "window_id": _teams_window_id(),
                "root_message_id": "message-a",
                "window_type": "time_block",
                "raw_payload": newer_payload,
            }
        ).encode()
        newer_uri = document_store.store_raw(
            "src-teams",
            "Temporary Newer Conversation A",
            newer_body,
            "application/json",
            extension=".local-package.json",
        )
        await database.create_source_sync_input(
            source_id="src-teams",
            raw_uri=newer_uri,
            raw_sha256=local_agent_semantic_input_sha256(_teams_doc_id(), newer_hash),
            raw_content_type="application/json",
            metadata={
                "manifest_entry": {
                    "doc_id": _teams_doc_id(),
                    "title": "Temporary Newer Conversation A",
                    "version": "2",
                    "package_uri": newer_uri,
                    "input_sha256": local_agent_semantic_input_sha256(_teams_doc_id(), newer_hash),
                },
                "package_sha256": hashlib.sha256(newer_body).hexdigest(),
            },
        )
        artifacts["newer_noncurrent"] = newer_uri
    source = await database.get_source("src-teams")
    assert source is not None
    run = await database.enqueue_source_sync_run(
        source_id="src-teams",
        trigger="local_agent",
        input_snapshot_id=snapshot_id,
        source_config_revision=local_agent_source_config_revision(source),
    )
    leased = await database.lease_next_source_sync_run(
        worker_id="worker-a",
        lease_seconds=60,
        now=now,
    )
    assert leased is not None
    completed = await database.complete_source_sync_run(
        run.run_id,
        worker_id="worker-a",
        lease_attempt_count=leased.lease_attempt_count,
        final_state=SyncState(
            source="src-teams",
            last_sync_at=now,
            last_sync_status="success",
        ),
        completed_at=now,
    )
    assert completed is True
    return artifacts


def test_source_list_route_uses_storage_neutral_admin_reader(tmp_path):
    class FakeSourceReader:
        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def claim_due_scheduled_sources(
            self,
            *,
            now: datetime | None = None,
            limit: int = 50,
            exclude_source_ids: set[str] | None = None,
        ) -> list[dict]:
            return []

        async def list_sources(self) -> list[dict]:
            return [
                {
                    "id": "src-neutral",
                    "type": "confluence",
                    "name": "Neutral Source",
                    "config": {"base_url": "https://wiki.example.test", "pat": "secret"},
                    "status": "active",
                    "access_policy": "private",
                    "access_state": "active",
                    "owner_user_id": "dev",
                    "last_sync": None,
                    "doc_count": 99,
                    "created_at": "2026-06-13T00:00:00+00:00",
                    "updated_at": "2026-06-13T00:00:00+00:00",
                    "sync_schedule": {
                        "enabled": True,
                        "interval_minutes": 60,
                        "next_run_at": "2026-06-13T01:00:00+00:00",
                        "updated_at": "2026-06-13T00:00:00+00:00",
                    },
                }
            ]

        async def count_source_memories(
            self,
            source_id: str,
            *,
            include_private: bool = False,
            owner_user_id: str | None = None,
        ) -> int:
            assert source_id == "src-neutral"
            assert include_private is True
            assert owner_user_id == "dev"
            return 7

        async def count_documents(self, source: str | None = None) -> int:
            assert source == "src-neutral"
            return 3

        async def get_sync_history(self, source: str | None = None, limit: int = 20) -> list[dict]:
            assert source == "src-neutral"
            assert limit == 1
            return [
                {
                    "status": "partial",
                    "started_at": "2026-06-13T00:00:01+00:00",
                    "finished_at": "2026-06-13T00:00:10+00:00",
                    "docs_processed": 3,
                    "docs_updated": 2,
                    "docs_failed": 1,
                    "memories_extracted": 4,
                    "error_message": "one failed",
                    "failed_docs": [{"doc_id": "doc-1", "error": "boom"}],
                }
            ]

        async def get_latest_source_sync_run(
            self,
            *,
            source_id: str,
            workspace_id: str = "default",
        ):
            assert source_id == "src-neutral"
            assert workspace_id == "default"
            return None

        async def get_active_source_access_transition(self, source_id: str):
            assert source_id == "src-neutral"
            return None

        async def is_source_enabled_for_user(self, source_id: str, user_id: str) -> bool:
            assert source_id == "src-neutral"
            return True

        async def is_source_pinned_for_user(self, source_id: str, user_id: str) -> bool:
            assert source_id == "src-neutral"
            return False

        async def set_source_subscription(self, source_id: str, user_id: str, enabled: bool) -> None:
            raise AssertionError("not used by source list")

        async def set_source_sync_schedule(
            self,
            source_id: str,
            *,
            enabled: bool,
            interval_minutes: int,
            next_run_at: datetime | None = None,
        ) -> None:
            raise AssertionError("not used by source list")

    app = create_admin_app(db=FakeSourceReader(), config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/sources")

    assert response.status_code == 200
    source = response.json()["data"][0]
    assert source["id"] == "src-neutral"
    assert source["config"] == {
        "base_url": "https://wiki.example.test",
        "pat_configured": True,
    }
    assert source["doc_count"] == 3
    assert source["memory_count"] == 7
    assert source["pinned_for_me"] is False
    assert source["client"] is None
    assert source["access_policy"] == "private"
    assert source["owner_user_id"] == "dev"
    assert source["sync"] == {
        "status": "partial",
        "started_at": "2026-06-13T00:00:01+00:00",
        "finished_at": "2026-06-13T00:00:10+00:00",
        "docs_processed": 3,
        "docs_updated": 2,
        "docs_failed": 1,
        "memories_extracted": 4,
        "error_message": "one failed",
        "failed_docs": [{"doc_id": "doc-1", "error": "boom"}],
        "progress": {
            "schema_version": 1,
            "phase": "processing",
            "progress": {"completed": 3, "unit": "page"},
            "counts": {"changed": 2, "failed": 1, "memories_created": 4},
        },
    }
    assert source["sync_schedule"] == {
        "enabled": True,
        "interval_minutes": 60,
        "next_run_at": "2026-06-13T01:00:00+00:00",
        "updated_at": "2026-06-13T00:00:00+00:00",
    }


def test_source_projects_route_uses_storage_neutral_admin_reader(tmp_path):
    class FakeSourceReader:
        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def claim_due_scheduled_sources(
            self,
            *,
            now: datetime | None = None,
            limit: int = 50,
            exclude_source_ids: set[str] | None = None,
        ) -> list[dict]:
            return []

        async def get_source(self, source_id: str) -> dict | None:
            assert source_id == "src-neutral"
            return {
                "id": source_id,
                "type": "confluence",
                "name": "Neutral Source",
                "access_policy": "private",
                "access_state": "active",
                "owner_user_id": "dev",
            }

        async def list_source_projects(
            self,
            source_id: str,
            *,
            include_private: bool = False,
            owner_user_id: str | None = None,
        ) -> list[dict]:
            assert source_id == "src-neutral"
            assert include_private is True
            assert owner_user_id == "dev"
            return [
                {
                    "project": "PAY",
                    "document_count": 3,
                    "memory_count": 7,
                    "last_observed_at": "2026-06-13T00:00:00+00:00",
                }
            ]

        async def set_source_sync_schedule(
            self,
            source_id: str,
            *,
            enabled: bool,
            interval_minutes: int,
            next_run_at: datetime | None = None,
        ) -> None:
            raise AssertionError("not used by source projects")

    app = create_admin_app(db=FakeSourceReader(), config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/sources/src-neutral/projects")

    assert response.status_code == 200
    assert response.json() == {
        "source_id": "src-neutral",
        "projects": [
            {
                "project": "PAY",
                "document_count": 3,
                "memory_count": 7,
                "last_observed_at": "2026-06-13T00:00:00+00:00",
            }
        ],
    }


def test_source_schedule_routes_use_storage_neutral_store(tmp_path):
    class FakeSourceReader:
        def __init__(self) -> None:
            self.updated: tuple[str, bool, int] | None = None

        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def claim_due_scheduled_sources(
            self,
            *,
            now: datetime | None = None,
            limit: int = 50,
            exclude_source_ids: set[str] | None = None,
        ) -> list[dict]:
            return []

        async def get_source(self, source_id: str) -> dict | None:
            assert source_id == "src-neutral"
            return {
                "id": source_id,
                "type": "confluence",
                "name": "Neutral Source",
                "created_by_user_id": "user-a",
                "access_policy": "private",
                "access_state": "active",
                "owner_user_id": "user-a",
                "sync_schedule": {
                    "enabled": False,
                    "interval_minutes": 1440,
                    "next_run_at": None,
                    "updated_at": None,
                },
            }

        async def set_source_sync_schedule(
            self,
            source_id: str,
            *,
            enabled: bool,
            interval_minutes: int,
            next_run_at: datetime | None = None,
        ) -> None:
            self.updated = (source_id, enabled, interval_minutes)

    reader = FakeSourceReader()
    app = create_admin_app(
        db=reader,
        config=_config(tmp_path),
        principal_resolver=lambda request: "user-a",
    )

    with TestClient(app) as client:
        get_response = client.get("/api/sources/src-neutral/schedule")
        put_response = client.put(
            "/api/sources/src-neutral/schedule",
            json={"enabled": True, "interval_minutes": 60},
        )

    assert get_response.status_code == 200, get_response.text
    assert get_response.json() == {
        "enabled": False,
        "interval_minutes": 1440,
        "next_run_at": None,
        "updated_at": None,
    }
    assert put_response.status_code == 200, put_response.text
    assert reader.updated == ("src-neutral", True, 60)


def test_source_memory_lifecycle_routes_expose_durable_operator_axes(tmp_path, monkeypatch):
    class FakeLifecycleStore:
        def __init__(self) -> None:
            self.jobs: dict[str, LifecycleBackfillJob] = {}
            self.gate = LifecycleGate(
                source_id="src-neutral",
                state=LifecycleGateState.GATED,
                reason="audit pending",
            )

        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def claim_due_scheduled_sources(self, **kwargs) -> list[dict]:
            return []

        async def get_source(self, source_id: str) -> dict | None:
            assert source_id == "src-neutral"
            return {
                "id": source_id,
                "type": "confluence",
                "name": "Neutral Source",
                "created_by_user_id": "user-a",
                "access_policy": "private",
                "access_state": "active",
                "owner_user_id": "user-a",
            }

        async def get_lifecycle_gate(self, source_id: str) -> LifecycleGate:
            assert source_id == "src-neutral"
            return self.gate

        async def list_lifecycle_cutover_findings(self, source_id: str, **kwargs) -> list:
            assert source_id == "src-neutral"
            return []

        async def list_lifecycle_backfill_jobs(self, source_id: str, **kwargs) -> list[LifecycleBackfillJob]:
            assert source_id == "src-neutral"
            return list(self.jobs.values())

        async def list_lifecycle_reviews(self, source_id: str, **kwargs) -> list:
            assert source_id == "src-neutral"
            return []

        async def list_lifecycle_vector_tasks(self, **kwargs) -> list:
            assert kwargs.get("source_id") == "src-neutral"
            return []

        async def list_projection_scope_transitions(self, source_id: str, **kwargs) -> list:
            assert source_id == "src-neutral"
            return []

        async def create_lifecycle_backfill_job(self, job: LifecycleBackfillJob) -> LifecycleBackfillJob:
            return self.jobs.setdefault(job.id, job)

        async def start_lifecycle_backfill_job(self, job_id: str) -> LifecycleBackfillJob:
            job = replace(self.jobs[job_id], status=LifecycleBackfillJobStatus.RUNNING)
            self.jobs[job_id] = job
            return job

        async def complete_lifecycle_backfill_job(self, job_id: str, **counts) -> LifecycleBackfillJob:
            job = replace(
                self.jobs[job_id],
                status=LifecycleBackfillJobStatus.COMPLETED,
                **counts,
            )
            self.jobs[job_id] = job
            return job

        async def fail_lifecycle_backfill_job(self, job_id: str, *, error: str) -> LifecycleBackfillJob:
            job = replace(
                self.jobs[job_id],
                status=LifecycleBackfillJobStatus.FAILED,
                error=error,
            )
            self.jobs[job_id] = job
            return job

        async def list_legacy_memory_provenance(self, source_id: str) -> list:
            assert source_id == "src-neutral"
            return []

        async def get_latest_source_sync_run(
            self,
            *,
            source_id: str,
            workspace_id: str = "default",
        ):
            assert source_id == "src-neutral"
            assert workspace_id == "default"
            return None

        async def enable_lifecycle_gate(self, source_id: str) -> LifecycleGate:
            self.gate = LifecycleGate(source_id=source_id, state=LifecycleGateState.ENABLED)
            return self.gate

    class FakeRuntimeProvider:
        def __init__(self) -> None:
            self.reprocessed_document_ids: frozenset[str] | None = None
            self.execution_mode: SourceSyncMode | None = None

        async def run_source_sync(self, **kwargs):
            self.reprocessed_document_ids = kwargs["reprocess_doc_ids"]
            self.execution_mode = kwargs["execution_mode"]
            return SimpleNamespace(
                last_sync_status="partial",
                error_message="1 document could not be synced",
                failed_docs=[
                    SimpleNamespace(
                        error="requested document was not returned by provider discovery",
                    )
                ],
            )

    async def fake_recovery_job(
        db,
        source_id: str,
        *,
        job_id: str,
        reconstruct_documents,
        repair_projections,
    ):
        assert source_id == "src-neutral"
        assert callable(reconstruct_documents)
        await db.start_lifecycle_backfill_job(job_id)
        await repair_projections(frozenset({"doc-1"}))
        await db.enable_lifecycle_gate(source_id)
        return await db.complete_lifecycle_backfill_job(
            job_id,
            scanned_memories=1,
            mapped_memories=1,
            finding_count=0,
        )

    monkeypatch.setattr(
        "memforge.memory.cutover.run_source_lifecycle_recovery_job",
        fake_recovery_job,
    )

    store = FakeLifecycleStore()
    runtime_provider = FakeRuntimeProvider()
    app = create_admin_app(
        db=store,
        config=_config(tmp_path),
        runtime_provider=runtime_provider,
        principal_resolver=lambda request: "user-a",
    )

    with TestClient(app) as client:
        queued = client.post("/api/sources/src-neutral/memory-lifecycle/backfill")
        status = client.get("/api/sources/src-neutral/memory-lifecycle")

    assert queued.status_code == 202, queued.text
    assert status.status_code == 200, status.text
    payload = status.json()
    assert payload["gate"]["state"] == "enabled"
    assert payload["jobs"][0]["status"] == "completed"
    assert payload["jobs"][0]["finding_count"] == 0
    assert payload["findings"] == []
    assert payload["reviews"] == []
    assert payload["vector_outbox"] == []
    assert runtime_provider.reprocessed_document_ids == frozenset({"doc-1"})
    assert runtime_provider.execution_mode is SourceSyncMode.PROJECTION_REPAIR


def test_source_rebaseline_endpoint_is_confirmation_bound_and_completes_replay(tmp_path):
    class SuccessfulReplayProvider(DefaultRuntimeProvider):
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def run_source_sync(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(last_sync_status="success", error_message=None)

    database = Database(str(tmp_path / "rebaseline-api.db"))

    async def setup() -> None:
        await database.connect()
        await database.upsert_source(
            id="src-replayable",
            type="confluence",
            name="Replayable",
            config_json="{}",
            access_policy="workspace",
            owner_user_id="user-a",
        )
        await database.upsert_source(
            id="src-agent",
            type="agent_session",
            name="Managed Agent Session",
            config_json="{}",
            access_policy="workspace",
            owner_user_id="user-a",
        )

    asyncio.run(setup())
    provider = SuccessfulReplayProvider()
    app = create_admin_app(
        db=database,
        config=_config(tmp_path),
        runtime_provider=provider,
        principal_resolver=lambda request: "user-a",
    )
    try:
        with TestClient(app) as client:
            mismatch = client.post(
                "/api/sources/src-replayable/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-other"},
            )
            agent = client.post(
                "/api/sources/src-agent/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-agent"},
            )
            accepted = client.post(
                "/api/sources/src-replayable/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-replayable"},
            )

        assert mismatch.status_code == 409, mismatch.text
        assert agent.status_code == 409, agent.text
        assert accepted.status_code == 202, accepted.text
        assert accepted.json()["operation"] == "source_rebaseline"

        async def inspect():
            return (
                await database.list_lifecycle_backfill_jobs("src-replayable"),
                await database.get_lifecycle_gate("src-replayable"),
            )

        jobs, gate = asyncio.run(inspect())
        assert len(jobs) == 1
        assert jobs[0].status is LifecycleBackfillJobStatus.COMPLETED
        assert jobs[0].finding_count == 0
        assert gate.state is LifecycleGateState.ENABLED
        assert len(provider.calls) == 2
        assert provider.calls[0]["execution_mode"] is SourceSyncMode.REBASELINE_PREFLIGHT
        assert provider.calls[0]["lifecycle_job_id"] == jobs[0].id
        assert provider.calls[1]["force_full_sync"] is True
        assert provider.calls[1]["execution_mode"] is SourceSyncMode.NORMAL
        assert provider.calls[1]["lifecycle_job_id"] == jobs[0].id
    finally:
        asyncio.run(database.close())


def test_server_source_rebaseline_preflight_failure_never_resets_lifecycle(
    tmp_path,
    monkeypatch,
):
    class FailedPreflightProvider(DefaultRuntimeProvider):
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def run_source_sync(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                last_sync_status="failed",
                error_message="provider unavailable",
            )

    class ResetGuard:
        def __init__(self) -> None:
            self.reset_calls = 0

        async def rebaseline_source_lifecycle(self, source_id: str):
            del source_id
            self.reset_calls += 1

    database = Database(str(tmp_path / "server-preflight-failure.db"))

    async def setup() -> None:
        await database.connect()
        await database.upsert_source(
            id="src-confluence",
            type="confluence",
            name="Confluence",
            config_json="{}",
            access_policy="workspace",
            owner_user_id="user-a",
        )

    asyncio.run(setup())
    provider = FailedPreflightProvider()
    reset_guard = ResetGuard()

    async def build_reset_guard(*args, **kwargs):
        del args, kwargs
        return reset_guard

    monkeypatch.setattr(
        "memforge.server.admin_api._build_memory_store",
        build_reset_guard,
    )
    app = create_admin_app(
        db=database,
        config=_config(tmp_path),
        runtime_provider=provider,
        principal_resolver=lambda request: "user-a",
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-confluence/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-confluence"},
            )

        assert response.status_code == 202, response.text
        assert len(provider.calls) == 1
        assert provider.calls[0]["execution_mode"] is SourceSyncMode.REBASELINE_PREFLIGHT
        assert reset_guard.reset_calls == 0
        jobs = asyncio.run(database.list_lifecycle_backfill_jobs("src-confluence"))
        assert len(jobs) == 1
        assert jobs[0].status is LifecycleBackfillJobStatus.FAILED
        assert "provider unavailable" in str(jobs[0].error)
    finally:
        asyncio.run(database.close())


def test_source_rebaseline_uses_post_fence_source_snapshot(tmp_path, monkeypatch):
    class CapturingProvider(DefaultRuntimeProvider):
        def __init__(self) -> None:
            self.sources: list[dict] = []

        async def run_source_sync(self, **kwargs):
            self.sources.append(kwargs["source"])
            return SimpleNamespace(last_sync_status="success", error_message=None)

    database = Database(str(tmp_path / "post-fence-source.db"))

    async def setup() -> None:
        await database.connect()
        await database.upsert_source(
            id="src-confluence",
            type="confluence",
            name="Current source",
            config_json='{"base_url":"https://current.example"}',
            access_policy="workspace",
            owner_user_id="user-a",
        )

    asyncio.run(setup())
    original_get_source = database.get_source
    reads = 0

    async def stale_then_current(source_id: str):
        nonlocal reads
        reads += 1
        current = await original_get_source(source_id)
        if reads == 1 and current is not None:
            return {
                **current,
                "name": "Stale source",
                "config": {"base_url": "https://stale.example"},
            }
        return current

    monkeypatch.setattr(database, "get_source", stale_then_current)
    provider = CapturingProvider()
    app = create_admin_app(
        db=database,
        config=_config(tmp_path),
        runtime_provider=provider,
        principal_resolver=lambda request: "user-a",
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-confluence/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-confluence"},
            )

        assert response.status_code == 202, response.text
        assert len(provider.sources) == 2
        assert all(
            source["config"]["base_url"] == "https://current.example"
            for source in provider.sources
        )
    finally:
        asyncio.run(database.close())


def test_source_rebaseline_returns_conflict_while_source_sync_is_active(tmp_path):
    database = Database(str(tmp_path / "rebaseline-active-sync.db"))

    async def setup() -> None:
        await database.connect()
        await database.upsert_source(
            id="src-confluence",
            type="confluence",
            name="Confluence",
            config_json="{}",
            access_policy="workspace",
            owner_user_id="user-a",
        )
        await database.enqueue_source_sync_run(
            source_id="src-confluence",
            trigger="manual",
        )

    asyncio.run(setup())
    app = create_admin_app(
        db=database,
        config=_config(tmp_path),
        principal_resolver=lambda request: "user-a",
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-confluence/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-confluence"},
            )

        assert response.status_code == 409, response.text
        assert "source sync run already active" in response.json()["detail"]
        assert asyncio.run(
            database.list_lifecycle_backfill_jobs("src-confluence")
        ) == []
    finally:
        asyncio.run(database.close())


def test_source_rebaseline_unexpected_preflight_error_fails_job_and_releases_fence(
    tmp_path,
    monkeypatch,
):
    database = Database(str(tmp_path / "rebaseline-preflight-exception.db"))
    cfg = _config(tmp_path)
    document_store = LocalDocumentStore(cfg.storage.docs_path)

    async def setup() -> None:
        await _setup_completed_teams_local_replay(
            database,
            snapshot_id=None,
            document_store=document_store,
        )

    asyncio.run(setup())

    async def fail_input_read(**kwargs):
        del kwargs
        raise RuntimeError("input store unavailable")

    monkeypatch.setattr(database, "list_source_sync_inputs", fail_input_read)
    app = create_admin_app(
        db=database,
        config=cfg,
        principal_resolver=lambda request: "user-a",
        document_store=document_store,
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-teams/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-teams"},
            )

        assert response.status_code == 500, response.text
        assert response.json()["detail"] == "source_rebaseline_preflight_failed"
        jobs = asyncio.run(database.list_lifecycle_backfill_jobs("src-teams"))
        assert len(jobs) == 1
        assert jobs[0].status is LifecycleBackfillJobStatus.FAILED
        assert jobs[0].error == "input store unavailable"
    finally:
        asyncio.run(database.close())


def test_local_agent_lifecycle_backfill_repairs_from_durable_current_corpus(
    tmp_path,
    monkeypatch,
):
    class CurrentCorpusReplayProvider(DefaultRuntimeProvider):
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def run_source_sync(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(last_sync_status="success", error_message=None)

    async def fake_recovery_job(
        db,
        source_id: str,
        *,
        job_id: str,
        reconstruct_documents,
        repair_projections,
    ):
        assert source_id == "src-teams"
        assert callable(reconstruct_documents)
        await db.start_lifecycle_backfill_job(job_id)
        await repair_projections(frozenset({_teams_doc_id()}))
        await db.enable_lifecycle_gate(source_id)
        return await db.complete_lifecycle_backfill_job(
            job_id,
            scanned_memories=1,
            mapped_memories=1,
            finding_count=0,
        )

    monkeypatch.setattr(
        "memforge.memory.cutover.run_source_lifecycle_recovery_job",
        fake_recovery_job,
    )
    database = Database(str(tmp_path / "local-agent-backfill-replay.db"))

    cfg = _config(tmp_path)
    document_store = LocalDocumentStore(cfg.storage.docs_path)

    async def setup() -> dict[str, str]:
        return await _setup_completed_teams_local_replay(
            database,
            snapshot_id=None,
            document_store=document_store,
        )

    artifacts = asyncio.run(setup())
    provider = CurrentCorpusReplayProvider()
    app = create_admin_app(
        db=database,
        config=cfg,
        runtime_provider=provider,
        principal_resolver=lambda request: "user-a",
        document_store=document_store,
    )
    try:
        with TestClient(app) as client:
            response = client.post("/api/sources/src-teams/memory-lifecycle/backfill")

        assert response.status_code == 202, response.text
        assert len(provider.calls) == 1
        assert provider.calls[0]["execution_mode"] is SourceSyncMode.PROJECTION_REPAIR
        jobs = asyncio.run(database.list_lifecycle_backfill_jobs("src-teams"))
        assert provider.calls[0]["lifecycle_job_id"] == jobs[0].id
        [manifest] = provider.calls[0]["source"]["config"][
            "local_agent_package_manifest"
        ]
        assert {
            key: manifest[key]
            for key in ("doc_id", "title", "version", "package_uri")
        } == {
            "doc_id": _teams_doc_id(),
            "title": "Conversation A",
            "version": "1",
            "package_uri": artifacts["current"],
        }
        assert manifest["input_sha256"]
        assert manifest["package_sha256"]
    finally:
        asyncio.run(database.close())


def test_local_agent_source_rebaseline_fails_before_reset_without_successful_replay(tmp_path):
    class UnexpectedReplayProvider(DefaultRuntimeProvider):
        async def run_source_sync(self, **kwargs):
            raise AssertionError("local-agent rebaseline must not run without a successful replay")

    database = Database(str(tmp_path / "local-agent-rebaseline-preflight.db"))

    async def setup() -> None:
        await database.connect()
        await database.upsert_source(
            id="src-teams",
            type="teams",
            name="Teams",
            config_json='{"conversation_ids":["conversation-a"]}',
            access_policy="workspace",
            owner_user_id="user-a",
        )

    asyncio.run(setup())
    app = create_admin_app(
        db=database,
        config=_config(tmp_path),
        runtime_provider=UnexpectedReplayProvider(),
        principal_resolver=lambda request: "user-a",
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-teams/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-teams"},
            )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "source_lifecycle_successful_local_replay_required"
        jobs = asyncio.run(database.list_lifecycle_backfill_jobs("src-teams"))
        assert len(jobs) == 1
        assert jobs[0].status is LifecycleBackfillJobStatus.FAILED
        assert jobs[0].error == "source_lifecycle_successful_local_replay_required"
    finally:
        asyncio.run(database.close())


def test_local_agent_source_rebaseline_fails_before_reset_when_snapshot_inputs_are_unavailable(tmp_path):
    class UnexpectedReplayProvider(DefaultRuntimeProvider):
        async def run_source_sync(self, **kwargs):
            raise AssertionError("local-agent rebaseline must not run with an empty manifest")

    database = Database(str(tmp_path / "local-agent-rebaseline-missing-inputs.db"))
    cfg = _config(tmp_path)
    document_store = LocalDocumentStore(cfg.storage.docs_path)

    async def setup() -> dict[str, str]:
        return await _setup_completed_teams_local_replay(
            database,
            snapshot_id="snapshot-without-inputs",
            document_store=document_store,
            include_current_input=False,
        )

    asyncio.run(setup())
    app = create_admin_app(
        db=database,
        config=cfg,
        runtime_provider=UnexpectedReplayProvider(),
        principal_resolver=lambda request: "user-a",
        document_store=document_store,
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-teams/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-teams"},
            )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == (
            "source_lifecycle_local_replay_inputs_unavailable"
        )
        jobs = asyncio.run(database.list_lifecycle_backfill_jobs("src-teams"))
        assert len(jobs) == 1
        assert jobs[0].status is LifecycleBackfillJobStatus.FAILED
        assert jobs[0].error == "source_lifecycle_local_replay_inputs_unavailable"
    finally:
        asyncio.run(database.close())


def test_teams_source_rebaseline_replays_current_corpus_across_attempt_bound_inputs(tmp_path):
    class SnapshotReplayProvider(DefaultRuntimeProvider):
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def run_source_sync(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(last_sync_status="success", error_message=None)

    database = Database(str(tmp_path / "local-agent-rebaseline-replay.db"))
    cfg = _config(tmp_path)
    document_store = LocalDocumentStore(cfg.storage.docs_path)

    async def setup() -> dict[str, str]:
        return await _setup_completed_teams_local_replay(
            database,
            snapshot_id="snapshot-teams",
            document_store=document_store,
            include_retired_input=True,
        )

    artifacts = asyncio.run(setup())
    provider = SnapshotReplayProvider()
    app = create_admin_app(
        db=database,
        config=cfg,
        runtime_provider=provider,
        principal_resolver=lambda request: "user-a",
        document_store=document_store,
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-teams/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-teams"},
            )

        assert response.status_code == 202, response.text
        assert len(provider.calls) == 2
        assert provider.calls[0]["execution_mode"] is SourceSyncMode.REBASELINE_PREFLIGHT
        assert provider.calls[1]["authoritative_snapshot"] is True
        [manifest] = provider.calls[1]["source"]["config"][
            "local_agent_package_manifest"
        ]
        assert {
            key: manifest[key]
            for key in ("doc_id", "title", "version", "package_uri")
        } == {
            "doc_id": _teams_doc_id(),
            "title": "Conversation A",
            "version": "1",
            "package_uri": artifacts["current"],
        }
        assert manifest["input_sha256"]
        assert manifest["package_sha256"]
    finally:
        asyncio.run(database.close())


def test_incremental_local_agent_source_rebaseline_replays_exact_current_corpus(tmp_path):
    class CurrentCorpusReplayProvider(DefaultRuntimeProvider):
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def run_source_sync(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(last_sync_status="success", error_message=None)

    database = Database(str(tmp_path / "incremental-local-agent-rebaseline.db"))
    cfg = _config(tmp_path)
    document_store = LocalDocumentStore(cfg.storage.docs_path)

    async def setup() -> dict[str, str]:
        return await _setup_completed_teams_local_replay(
            database,
            snapshot_id=None,
            document_store=document_store,
            include_retired_input=True,
        )

    artifacts = asyncio.run(setup())
    provider = CurrentCorpusReplayProvider()
    app = create_admin_app(
        db=database,
        config=cfg,
        runtime_provider=provider,
        principal_resolver=lambda request: "user-a",
        document_store=document_store,
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-teams/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-teams"},
            )

        assert response.status_code == 202, response.text
        assert len(provider.calls) == 2
        assert provider.calls[0]["execution_mode"] is SourceSyncMode.REBASELINE_PREFLIGHT
        assert provider.calls[1]["authoritative_snapshot"] is True
        [manifest] = provider.calls[1]["source"]["config"][
            "local_agent_package_manifest"
        ]
        assert {
            key: manifest[key]
            for key in ("doc_id", "title", "version", "package_uri")
        } == {
            "doc_id": _teams_doc_id(),
            "title": "Conversation A",
            "version": "1",
            "package_uri": artifacts["current"],
        }
        assert manifest["input_sha256"]
        assert manifest["package_sha256"]
    finally:
        asyncio.run(database.close())


def test_teams_rebaseline_selects_indexed_version_after_semantic_reversion(tmp_path):
    class ReplayProvider(DefaultRuntimeProvider):
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def run_source_sync(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(last_sync_status="success", error_message=None)

    database = Database(str(tmp_path / "reverted-version-rebaseline.db"))
    cfg = _config(tmp_path)
    document_store = LocalDocumentStore(cfg.storage.docs_path)

    async def setup() -> dict[str, str]:
        return await _setup_completed_teams_local_replay(
            database,
            snapshot_id=None,
            document_store=document_store,
            include_newer_noncurrent_version=True,
        )

    artifacts = asyncio.run(setup())
    provider = ReplayProvider()
    app = create_admin_app(
        db=database,
        config=cfg,
        runtime_provider=provider,
        principal_resolver=lambda request: "user-a",
        document_store=document_store,
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-teams/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-teams"},
            )

        assert response.status_code == 202, response.text
        [manifest] = provider.calls[1]["source"]["config"]["local_agent_package_manifest"]
        assert manifest["version"] == "1"
        assert manifest["package_uri"] == artifacts["current"]
        assert manifest["package_uri"] != artifacts["newer_noncurrent"]
    finally:
        asyncio.run(database.close())


def test_incremental_local_rebaseline_ignores_input_uploaded_after_success(tmp_path):
    class ReplayProvider(DefaultRuntimeProvider):
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def run_source_sync(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(last_sync_status="success", error_message=None)

    database = Database(str(tmp_path / "incremental-watermark-rebaseline.db"))
    cfg = _config(tmp_path)
    document_store = LocalDocumentStore(cfg.storage.docs_path)

    async def setup() -> dict[str, str]:
        artifacts = await _setup_completed_teams_local_replay(
            database,
            snapshot_id=None,
            document_store=document_store,
        )
        unconsumed_uri = document_store.store_raw(
            "src-teams",
            "Unconsumed Conversation A",
            json.dumps(
                {
                    "package_kind": "teams_window_document",
                    "doc_id": _teams_doc_id(),
                    "version": "2",
                }
            ).encode(),
            "application/json",
            extension=".local-package.json",
        )
        await database.create_source_sync_input(
            source_id="src-teams",
            raw_uri=unconsumed_uri,
            raw_sha256="sha-unconsumed-window-a",
            raw_content_type="application/json",
            metadata={
                "manifest_entry": {
                    "doc_id": _teams_doc_id(),
                    "title": "Unconsumed Conversation A",
                    "version": "2",
                }
            },
        )
        artifacts["unconsumed"] = unconsumed_uri
        return artifacts

    artifacts = asyncio.run(setup())
    provider = ReplayProvider()
    app = create_admin_app(
        db=database,
        config=cfg,
        runtime_provider=provider,
        principal_resolver=lambda request: "user-a",
        document_store=document_store,
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-teams/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-teams"},
            )

        assert response.status_code == 202, response.text
        assert provider.calls[0]["execution_mode"] is SourceSyncMode.REBASELINE_PREFLIGHT
        manifest = provider.calls[1]["source"]["config"]["local_agent_package_manifest"]
        assert manifest[0]["package_uri"] == artifacts["current"]
        assert manifest[0]["package_uri"] != artifacts["unconsumed"]
    finally:
        asyncio.run(database.close())


def test_local_rebaseline_rejects_unreadable_or_mismatched_artifact_before_reset(tmp_path):
    class UnexpectedReplayProvider(DefaultRuntimeProvider):
        async def run_source_sync(self, **kwargs):
            raise AssertionError("invalid replay artifact must fail before reset")

    database = Database(str(tmp_path / "invalid-artifact-rebaseline.db"))
    cfg = _config(tmp_path)
    document_store = LocalDocumentStore(cfg.storage.docs_path)

    async def setup() -> dict[str, str]:
        return await _setup_completed_teams_local_replay(
            database,
            snapshot_id=None,
            document_store=document_store,
        )

    artifacts = asyncio.run(setup())
    with open(artifacts["current"], "w", encoding="utf-8") as handle:
        json.dump(
            {
                "package_kind": "teams_window_document",
                "doc_id": _teams_doc_id(),
                "version": "wrong-version",
            },
            handle,
        )
    app = create_admin_app(
        db=database,
        config=cfg,
        runtime_provider=UnexpectedReplayProvider(),
        principal_resolver=lambda request: "user-a",
        document_store=document_store,
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-teams/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-teams"},
            )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "source_lifecycle_local_replay_artifact_invalid"
    finally:
        asyncio.run(database.close())


def test_local_rebaseline_rejects_legacy_input_without_package_attestation(tmp_path):
    class UnexpectedReplayProvider(DefaultRuntimeProvider):
        async def run_source_sync(self, **kwargs):
            raise AssertionError("unattested replay must fail before reset")

    database = Database(str(tmp_path / "unattested-artifact-rebaseline.db"))
    cfg = _config(tmp_path)
    document_store = LocalDocumentStore(cfg.storage.docs_path)

    async def setup() -> None:
        await _setup_completed_teams_local_replay(
            database,
            snapshot_id=None,
            document_store=document_store,
        )
        async with database.db.execute(
            "SELECT input_id, metadata_json FROM source_sync_inputs "
            "WHERE source_id = 'src-teams'"
        ) as cursor:
            row = await cursor.fetchone()
        metadata = json.loads(row["metadata_json"])
        metadata.pop("package_sha256", None)
        await database.db.execute(
            "UPDATE source_sync_inputs SET metadata_json = ? WHERE input_id = ?",
            (json.dumps(metadata, sort_keys=True), row["input_id"]),
        )
        await database.db.commit()

    asyncio.run(setup())
    app = create_admin_app(
        db=database,
        config=cfg,
        runtime_provider=UnexpectedReplayProvider(),
        principal_resolver=lambda request: "user-a",
        document_store=document_store,
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-teams/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-teams"},
            )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == (
            "source_lifecycle_local_replay_attestation_required"
        )
        [job] = asyncio.run(
            database.list_lifecycle_backfill_jobs("src-teams")
        )
        assert job.status is LifecycleBackfillJobStatus.FAILED
        assert job.error == "source_lifecycle_local_replay_attestation_required"
    finally:
        asyncio.run(database.close())


def test_local_rebaseline_rejects_source_config_changed_after_success(tmp_path):
    database = Database(str(tmp_path / "changed-config-rebaseline.db"))
    cfg = _config(tmp_path)
    document_store = LocalDocumentStore(cfg.storage.docs_path)

    async def setup() -> None:
        await _setup_completed_teams_local_replay(
            database,
            snapshot_id=None,
            document_store=document_store,
        )
        await database.upsert_source(
            id="src-teams",
            type="teams",
            name="Teams",
            config_json='{"conversation_ids":["conversation-b"]}',
            access_policy="workspace",
            owner_user_id="user-a",
        )

    asyncio.run(setup())
    app = create_admin_app(
        db=database,
        config=cfg,
        principal_resolver=lambda request: "user-a",
        document_store=document_store,
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-teams/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-teams"},
            )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "source_lifecycle_local_replay_config_changed"
    finally:
        asyncio.run(database.close())


def test_teams_rebaseline_rejects_unattested_empty_attempt(tmp_path):
    class EmptyReplayProvider(DefaultRuntimeProvider):
        def __init__(self) -> None:
            self.calls: list[dict] = []

        async def run_source_sync(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(last_sync_status="success", error_message=None)

    database = Database(str(tmp_path / "empty-snapshot-rebaseline.db"))
    cfg = _config(tmp_path)
    document_store = LocalDocumentStore(cfg.storage.docs_path)

    async def setup() -> None:
        await _setup_completed_teams_local_replay(
            database,
            snapshot_id="snapshot-empty",
            document_store=document_store,
            include_current_input=False,
            include_current_document=False,
        )

    asyncio.run(setup())
    provider = EmptyReplayProvider()
    app = create_admin_app(
        db=database,
        config=cfg,
        runtime_provider=provider,
        principal_resolver=lambda request: "user-a",
        document_store=document_store,
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-teams/memory-lifecycle/rebaseline",
                json={"confirm_source_id": "src-teams"},
            )

        assert response.status_code == 409, response.text
        assert response.json()["detail"] == "source_lifecycle_local_replay_inputs_unavailable"
        assert provider.calls == []
    finally:
        asyncio.run(database.close())


def test_source_lifecycle_finding_repair_returns_exact_lineage_and_gate_state(
    tmp_path,
    monkeypatch,
):
    class FakeRepairStore:
        def __init__(self) -> None:
            self.jobs: list[tuple[str, str]] = []

        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def get_source(self, source_id: str):
            assert source_id == "src-neutral"
            return {
                "id": source_id,
                "type": "teams",
                "status": "active",
                "access_policy": "workspace",
                "access_state": "active",
                "owner_user_id": "user-a",
            }

        async def create_lifecycle_backfill_job(self, job):
            self.jobs.append((job.id, "queued"))
            return job

        async def start_lifecycle_backfill_job(self, job_id: str):
            self.jobs.append((job_id, "running"))

        async def complete_lifecycle_backfill_job(self, job_id: str, **kwargs):
            assert kwargs == {
                "scanned_memories": 1,
                "mapped_memories": 1,
                "finding_count": 0,
            }
            self.jobs.append((job_id, "completed"))

        async def fail_lifecycle_backfill_job(self, job_id: str, *, error: str):
            self.jobs.append((job_id, f"failed:{error}"))

    async def fake_repair(
        db,
        *,
        source_id: str,
        finding_id: str,
        observation_id: str,
        evidence_quote: str | None,
        operator_id: str | None,
    ):
        assert isinstance(db, FakeRepairStore)
        assert (source_id, finding_id, observation_id) == (
            "src-neutral",
            "finding-1",
            "observation-2",
        )
        assert evidence_quote == "exact source quote"
        assert operator_id == "user-a"
        return LifecycleCutoverFinding(
            id=finding_id,
            source_id=source_id,
            memory_id="memory-1",
            reason=CutoverFindingReason.AMBIGUOUS_OBSERVATION,
            status=CutoverFindingStatus.RESOLVED,
            available_provenance={},
            mapping_attempt={},
            observation_id=observation_id,
            source_unit_id="unit-1",
        )

    async def fake_backfill(db, source_id: str):
        assert isinstance(db, FakeRepairStore)
        assert source_id == "src-neutral"
        return SimpleNamespace(
            gate_enabled=True,
            finding_count=0,
            scanned_memories=1,
            mapped_memories=1,
        )

    monkeypatch.setattr("memforge.memory.cutover.repair_lifecycle_cutover_finding", fake_repair)
    monkeypatch.setattr("memforge.memory.cutover.run_source_lifecycle_backfill", fake_backfill)
    app = create_admin_app(
        db=FakeRepairStore(),
        config=_config(tmp_path),
        principal_resolver=lambda request: "user-a",
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/sources/src-neutral/memory-lifecycle/findings/finding-1/repair",
            json={
                "observation_id": "observation-2",
                "evidence_quote": "exact source quote",
            },
        )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "source_id": "src-neutral",
        "finding_id": "finding-1",
        "status": "resolved",
        "observation_id": "observation-2",
        "source_unit_id": "unit-1",
        "gate_enabled": True,
        "remaining_findings": 0,
    }


def test_source_lifecycle_finding_repair_conflicts_with_active_source_sync(tmp_path):
    database = Database(str(tmp_path / "finding-repair-active-sync.db"))

    async def setup() -> None:
        await database.connect()
        await database.upsert_source(
            id="src-teams",
            type="teams",
            name="Teams",
            config_json="{}",
            access_policy="workspace",
            owner_user_id="user-a",
        )
        await database.enqueue_source_sync_run(
            source_id="src-teams",
            trigger="manual",
        )

    asyncio.run(setup())
    app = create_admin_app(
        db=database,
        config=_config(tmp_path),
        principal_resolver=lambda request: "user-a",
    )
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/sources/src-teams/memory-lifecycle/findings/finding-1/repair",
                json={"observation_id": "observation-1"},
            )

        assert response.status_code == 409, response.text
        assert "source sync run already active" in response.json()["detail"]
        assert asyncio.run(
            database.list_lifecycle_backfill_jobs("src-teams")
        ) == []
    finally:
        asyncio.run(database.close())
