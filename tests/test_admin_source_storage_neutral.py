from __future__ import annotations

from dataclasses import replace
from datetime import datetime
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
from memforge.pipeline.sync import SourceSyncMode
from memforge.server.admin_api import create_admin_app


def _config(tmp_path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "memforge")
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    cfg.sync.worker_enabled = False
    return cfg


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


def test_source_lifecycle_finding_repair_returns_exact_lineage_and_gate_state(
    tmp_path,
    monkeypatch,
):
    class FakeRepairStore:
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
        return SimpleNamespace(gate_enabled=True, finding_count=0)

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
