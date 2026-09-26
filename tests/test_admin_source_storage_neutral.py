from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.evals.agent_evaluation import QualitySignal, bind_quality_signals, evaluate_runtime_events
from memforge.memory.lifecycle_plan import (
    LifecycleGate,
    LifecycleGateState,
)
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
                    "run_id": "run-neutral",
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
            assert workspace_id == "local"
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
        response = client.get("/api/v1/sources")

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
        "run_id": "run-neutral",
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


def test_source_list_projects_a_retried_sync_as_a_fresh_attempt(tmp_path):
    now = datetime.now(timezone.utc)
    active_retry = SimpleNamespace(
        run_id="run-retry",
        created_at=now,
        status="running",
        trigger="scheduled",
        force_full_sync=False,
        started_at=now,
        completed_at=None,
        next_attempt_at=None,
        recovery_count=1,
        error_message=None,
        progress=None,
        progress_revision=7,
        progress_updated_at=None,
        lease_expires_at=now.replace(year=now.year + 1),
    )

    class RetrySourceReader:
        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def claim_due_scheduled_sources(self, **_: object) -> list[dict]:
            return []

        async def list_sources(self) -> list[dict]:
            return [
                {
                    "id": "src-retry",
                    "type": "confluence",
                    "name": "Retry Source",
                    "config": {},
                    "status": "active",
                    "access_policy": "workspace",
                    "access_state": "active",
                    "owner_user_id": "dev",
                    "created_at": now.isoformat(),
                    "sync_schedule": {
                        "enabled": False,
                        "interval_minutes": 1440,
                        "next_run_at": None,
                        "updated_at": None,
                    },
                }
            ]

        async def count_source_memories(self, *_: object, **__: object) -> int:
            return 0

        async def count_documents(self, *_: object, **__: object) -> int:
            return 0

        async def get_latest_source_sync_run(self, **_: object):
            return active_retry

        async def get_sync_history(self, **_: object) -> list[dict]:
            raise AssertionError("active durable retry must win over history")

        async def get_active_source_access_transition(self, source_id: str):
            assert source_id == "src-retry"
            return None

        async def is_source_enabled_for_user(
            self,
            source_id: str,
            user_id: str,
        ) -> bool:
            return True

        async def is_source_pinned_for_user(
            self,
            source_id: str,
            user_id: str,
        ) -> bool:
            return False

    app = create_admin_app(
        db=RetrySourceReader(),
        config=_config(tmp_path),
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/sources")

    assert response.status_code == 200
    sync = response.json()["data"][0]["sync"]
    assert sync["run_id"] == "run-retry"
    assert sync["status"] == "running"
    assert sync["error_message"] is None
    assert sync["progress"] is None
    assert sync["progress_revision"] == 7
    assert sync["progress_updated_at"] is None
    assert sync["recovery_count"] == 1


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
        response = client.get("/api/v1/sources/src-neutral/projects")

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


def test_source_agent_evaluation_route_is_bounded_and_storage_neutral(tmp_path):
    now = datetime.now(timezone.utc)
    [event] = bind_quality_signals(
        (QualitySignal("evidence_admission_outcome", "rejected", "unknown_evidence_block_id"),),
        source_id="src-neutral",
        source_type="confluence",
        doc_id="doc-1",
        source_unit_id="unit-1",
        target_unit_revision_id="revision-1",
        projection_run_id="projection-1",
        derivation_id="derivation-1",
        batch_id="batch-1",
        batch_attempt=1,
        extraction_contract_version="projection-extraction-v8",
        occurred_at=now,
    )
    assessments = list(evaluate_runtime_events((event,)))

    class FakeSourceReader:
        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def claim_due_scheduled_sources(self, **_kwargs) -> list[dict]:
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

        async def list_agent_runtime_events(self, query):
            assert query.source_id == "src-neutral"
            assert query.requesting_user_id == "dev"
            assert query.include_private is True
            assert query.newest_first is True
            assert query.limit == 1000
            return [event]

        async def list_agent_assessments(self, query):
            assert query.source_id == "src-neutral"
            assert query.requesting_user_id == "dev"
            assert query.include_private is True
            assert query.newest_first is True
            assert query.limit == 1000
            return assessments

    app = create_admin_app(db=FakeSourceReader(), config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/v1/sources/src-neutral/agent-evaluation?days=30")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["summary"] == {
        "total_assessments": 1,
        "label_counts": {"fail": 1},
        "criterion_counts": {"evidence_reference_validity": 1},
        "status_counts": {"completed": 1},
        "runtime_event_count": 1,
        "eligible_assessment_count": 1,
        "missing_assessment_count": 0,
        "action_issue_group_count": 1,
        "review_issue_group_count": 0,
        "truncated": False,
    }
    assert payload["assessments"][0]["target_event_id"] == event.event_id


def test_source_agent_evaluation_route_groups_actionable_cases_and_preserves_semantic_coverage(
    tmp_path,
):
    now = datetime.now(timezone.utc)
    events = bind_quality_signals(
        (
            QualitySignal(
                "evidence_admission_outcome",
                "rejected",
                "legacy_quote_unresolved",
            ),
            QualitySignal(
                "evidence_localization_outcome",
                "degraded",
                "whole_block_fallback",
                observation_id="observation-1",
                observation_revision_id="observation-revision-1",
                range_start=0,
                range_end=12,
            ),
        ),
        source_id="src-neutral",
        source_type="teams",
        doc_id="doc-1",
        source_unit_id="unit-1",
        target_unit_revision_id="revision-1",
        projection_run_id="projection-1",
        derivation_id="derivation-1",
        batch_id="batch-1",
        batch_attempt=1,
        extraction_contract_version="projection-extraction-v8",
        occurred_at=now,
    )
    current_assessments = evaluate_runtime_events(events)
    prior_schema_assessments = [
        replace(
            assessment,
            assessment_id=f"aas-prior-v3-{index}",
            schema_version="agent-assessment-v3",
        )
        for index, assessment in enumerate(current_assessments)
    ]

    class FakeSourceReader:
        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def claim_due_scheduled_sources(self, **_kwargs) -> list[dict]:
            return []

        async def get_source(self, source_id: str) -> dict | None:
            assert source_id == "src-neutral"
            return {
                "id": source_id,
                "type": "teams",
                "name": "Neutral Source",
                "access_policy": "private",
                "access_state": "active",
                "owner_user_id": "dev",
            }

        async def list_agent_runtime_events(self, query):
            assert query.source_id == "src-neutral"
            return list(events)

        async def list_agent_assessments(self, query):
            assert query.source_id == "src-neutral"
            return prior_schema_assessments

    app = create_admin_app(db=FakeSourceReader(), config=_config(tmp_path))

    with TestClient(app) as client:
        response = client.get("/api/v1/sources/src-neutral/agent-evaluation?days=1")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["summary"]["missing_assessment_count"] == 0
    assert payload["coverage"] == {
        "policy": "semantic_evaluator_v1",
        "eligible_occurrences": 2,
        "assessed_occurrences": 2,
        "pending_occurrences": 0,
        "coverage_rate": 1.0,
        "oldest_pending_at": None,
        "evaluator_failure_occurrences": 0,
    }
    assert [group["label"] for group in payload["issue_groups"]] == [
        "fail",
        "needs_review",
    ]
    fail_group, review_group = payload["issue_groups"]
    assert fail_group["criterion"] == "evidence_reference_validity"
    assert fail_group["reason_code"] == "legacy_quote_unresolved"
    assert fail_group["occurrence_count"] == 1
    assert fail_group["criterion_occurrence_count"] == 1
    assert fail_group["criterion_rate"] == 1.0
    assert fail_group["representative_cases"][0]["event_id"] == events[0].event_id
    assert fail_group["representative_cases"][0]["doc_id"] == "doc-1"
    assert fail_group["representative_cases"][0]["source_unit_id"] == "unit-1"
    assert review_group["criterion"] == "evidence_localization"
    assert review_group["representative_cases"][0]["observation_id"] == "observation-1"


def test_workspace_agent_evaluation_route_aggregates_only_discoverable_sources(tmp_path):
    now = datetime.now(timezone.utc)

    def event_for(
        source_id: str,
        source_type: str,
        signal: QualitySignal,
    ):
        return bind_quality_signals(
            (signal,),
            source_id=source_id,
            source_type=source_type,
            doc_id=f"doc-{source_id}",
            source_unit_id=f"unit-{source_id}",
            target_unit_revision_id=f"revision-{source_id}",
            projection_run_id=f"projection-{source_id}",
            derivation_id=f"derivation-{source_id}",
            batch_id=f"batch-{source_id}",
            batch_attempt=1,
            extraction_contract_version="projection-extraction-v8",
            occurred_at=now,
        )[0]

    workspace_failure = event_for(
        "src-workspace-failure",
        "teams",
        QualitySignal(
            "evidence_admission_outcome",
            "rejected",
            "legacy_quote_unresolved",
        ),
    )
    private_review = event_for(
        "src-private-review",
        "github_repo",
        QualitySignal(
            "evidence_localization_outcome",
            "degraded",
            "whole_block_fallback",
            observation_id="observation-private",
            observation_revision_id="observation-revision-private",
            range_start=0,
            range_end=10,
        ),
    )
    workspace_healthy = event_for(
        "src-workspace-healthy",
        "jira",
        QualitySignal(
            "structured_output_outcome",
            "expected",
            "schema_conformant",
        ),
    )
    hidden_private = event_for(
        "src-hidden-private",
        "teams",
        QualitySignal(
            "evidence_admission_outcome",
            "rejected",
            "missing_evidence_reference",
        ),
    )
    all_events = [
        workspace_failure,
        private_review,
        workspace_healthy,
        hidden_private,
    ]
    all_assessments = list(evaluate_runtime_events(tuple(all_events)))
    sources = [
        {
            "id": "src-workspace-failure",
            "name": "Workspace Teams",
            "type": "teams",
            "status": "active",
            "owner_user_id": "owner-a",
            "access_policy": "workspace",
            "access_state": "active",
        },
        {
            "id": "src-private-review",
            "name": "My Repository",
            "type": "github_repo",
            "status": "active",
            "owner_user_id": "dev",
            "access_policy": "private",
            "access_state": "active",
        },
        {
            "id": "src-workspace-healthy",
            "name": "Healthy Jira",
            "type": "jira",
            "status": "active",
            "owner_user_id": "owner-a",
            "access_policy": "workspace",
            "access_state": "active",
        },
        {
            "id": "src-hidden-private",
            "name": "Someone Else Private",
            "type": "teams",
            "status": "active",
            "owner_user_id": "someone-else",
            "access_policy": "private",
            "access_state": "active",
        },
    ]

    class FakeWorkspaceEvaluationReader:
        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def claim_due_scheduled_sources(self, **_kwargs) -> list[dict]:
            return []

        async def list_sources(self):
            return sources

        async def list_agent_runtime_events(self, query):
            assert query.source_id is None
            assert query.requesting_user_id == "dev"
            assert query.include_private is True
            assert query.newest_first is True
            assert query.limit == 1000
            # Deliberately return a hidden Source event. The route must still
            # fail closed against the discoverable Source cohort.
            return all_events

        async def list_agent_assessments(self, query):
            assert query.source_id is None
            assert query.requesting_user_id == "dev"
            assert query.include_private is True
            assert query.newest_first is True
            assert query.limit == 1000
            return all_assessments

    app = create_admin_app(
        db=FakeWorkspaceEvaluationReader(),
        config=_config(tmp_path),
    )

    with TestClient(app) as client:
        response = client.get("/api/v1/agent-evaluations/online-overview?days=1")
        hidden_response = client.get(
            "/api/v1/agent-evaluations/online-overview"
            "?days=1&source_id=src-hidden-private"
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["scope"] == {
        "kind": "workspace",
        "source_id": None,
        "source_type": None,
    }
    assert payload["summary"]["source_count"] == 3
    assert payload["summary"]["affected_source_count"] == 2
    assert payload["coverage"]["eligible_occurrences"] == 3
    assert payload["coverage"]["assessed_occurrences"] == 3
    assert [source["source_id"] for source in payload["sources"]] == [
        "src-workspace-failure",
        "src-private-review",
        "src-workspace-healthy",
    ]
    assert [source["evaluation_status"] for source in payload["sources"]] == [
        "attention",
        "review",
        "healthy",
    ]
    assert {group["label"] for group in payload["issue_groups"]} == {
        "fail",
        "needs_review",
    }
    assert {
        group["label"]: (
            group["affected_source_ids"],
            group["affected_source_count"],
            group["source_types"],
        )
        for group in payload["issue_groups"]
    } == {
        "fail": (["src-workspace-failure"], 1, ["teams"]),
        "needs_review": (["src-private-review"], 1, ["github_repo"]),
    }
    assert all(
        "src-hidden-private" not in group["affected_source_ids"]
        for group in payload["issue_groups"]
    )
    assert hidden_response.status_code == 404


def test_workspace_agent_evaluation_route_filters_source_type_without_new_evaluator(
    tmp_path,
):
    now = datetime.now(timezone.utc)
    teams_event = bind_quality_signals(
        (QualitySignal("structured_output_outcome", "expected", "schema_conformant"),),
        source_id="src-teams",
        source_type="teams",
        doc_id="doc-teams",
        source_unit_id="unit-teams",
        target_unit_revision_id="revision-teams",
        projection_run_id="projection-teams",
        derivation_id="derivation-teams",
        batch_id="batch-teams",
        batch_attempt=1,
        extraction_contract_version="projection-extraction-v8",
        occurred_at=now,
    )[0]
    github_event = replace(
        teams_event,
        event_id="are-github",
        source_id="src-github",
        source_type="github_repo",
        doc_id="doc-github",
        source_unit_id="unit-github",
        target_unit_revision_id="revision-github",
        projection_run_id="projection-github",
    )
    events = [teams_event, github_event]
    assessments = list(evaluate_runtime_events(tuple(events)))

    class FakeWorkspaceEvaluationReader:
        async def get_schedule_config(self) -> dict:
            return {"enabled": False}

        async def claim_due_scheduled_sources(self, **_kwargs) -> list[dict]:
            return []

        async def list_sources(self):
            return [
                {
                    "id": "src-teams",
                    "name": "Teams",
                    "type": "teams",
                    "status": "active",
                    "owner_user_id": "dev",
                    "access_policy": "workspace",
                    "access_state": "active",
                },
                {
                    "id": "src-github",
                    "name": "GitHub",
                    "type": "github_repo",
                    "status": "active",
                    "owner_user_id": "dev",
                    "access_policy": "workspace",
                    "access_state": "active",
                },
            ]

        async def list_agent_runtime_events(self, _query):
            return events

        async def list_agent_assessments(self, _query):
            return assessments

    app = create_admin_app(
        db=FakeWorkspaceEvaluationReader(),
        config=_config(tmp_path),
    )

    with TestClient(app) as client:
        response = client.get(
            "/api/v1/agent-evaluations/online-overview"
            "?days=1&source_type=teams"
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["scope"]["source_type"] == "teams"
    assert payload["available_source_types"] == ["github_repo", "teams"]
    assert [source["source_id"] for source in payload["sources"]] == ["src-teams"]
    assert payload["coverage"]["eligible_occurrences"] == 1


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
        get_response = client.get("/api/v1/sources/src-neutral/schedule")
        put_response = client.put(
            "/api/v1/sources/src-neutral/schedule",
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


def test_source_memory_lifecycle_route_exposes_durable_operator_axes(tmp_path):
    class FakeLifecycleStore:
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
            return LifecycleGate(source_id=source_id, state=LifecycleGateState.ENABLED)

        async def list_lifecycle_reviews(self, source_id: str | None = None, **kwargs) -> list:
            assert source_id == "src-neutral"
            return []

        async def list_lifecycle_vector_tasks(self, **kwargs) -> list:
            assert kwargs.get("source_id") == "src-neutral"
            return []

        async def list_projection_scope_transitions(self, source_id: str, **kwargs) -> list:
            assert source_id == "src-neutral"
            return []

    app = create_admin_app(
        db=FakeLifecycleStore(),
        config=_config(tmp_path),
        principal_resolver=lambda request: "user-a",
    )

    with TestClient(app) as client:
        status = client.get("/api/v1/sources/src-neutral/memory-lifecycle")

    assert status.status_code == 200, status.text
    assert status.json() == {
        "source_id": "src-neutral",
        "scope_transitions": [],
        "gate": {
            "state": "enabled",
            "reason": None,
            "audited_at": None,
            "enabled_at": None,
        },
        "reviews": [],
        "vector_outbox": [],
    }


