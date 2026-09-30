"""The typed response contract of the route behind the Evaluation page.

The route declares a response model, so the model decides which keys reach the
wire. This test builds the view from real runtime events and assessments, the
same objects every storage adapter returns, and checks that the response keeps
every key the view builder produces.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.evals.agent_evaluation import (
    QualitySignal,
    bind_quality_signals,
    build_workspace_online_evaluation_view,
    evaluate_runtime_events,
)
from memforge.server.admin_api import ONLINE_EVALUATION_ROW_LIMIT, create_admin_app

SOURCES = [
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
        "id": "src-repo",
        "name": "Repository",
        "type": "github_repo",
        "status": "active",
        "owner_user_id": "dev",
        "access_policy": "workspace",
        "access_state": "active",
    },
]


def _config(tmp_path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "memforge")
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    cfg.sync.worker_enabled = False
    return cfg


def _event(source_id: str, source_type: str, signal: QualitySignal):
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
        occurred_at=datetime.now(timezone.utc),
    )[0]


def test_online_overview_response_keeps_every_key_the_view_produces(tmp_path):
    events = [
        _event(
            "src-teams",
            "teams",
            QualitySignal("evidence_admission_outcome", "rejected", "legacy_quote_unresolved"),
        ),
        _event(
            "src-repo",
            "github_repo",
            QualitySignal(
                "evidence_localization_outcome",
                "degraded",
                "whole_block_fallback",
                observation_id="observation-1",
                observation_revision_id="observation-revision-1",
                range_start=0,
                range_end=10,
            ),
        ),
    ]
    assessments = list(evaluate_runtime_events(tuple(events)))
    view = build_workspace_online_evaluation_view(SOURCES, events, assessments)

    class FakeEvaluationReader:
        async def claim_due_scheduled_sources(self, **_kwargs) -> list[dict]:
            return []

        async def list_sources(self):
            return SOURCES

        async def list_agent_runtime_events(self, _query):
            return events

        async def list_agent_assessments(self, _query):
            return assessments

    app = create_admin_app(db=FakeEvaluationReader(), config=_config(tmp_path))
    with TestClient(app) as client:
        response = client.get("/api/v1/agent-evaluations/online-overview?days=1")

    assert response.status_code == 200, response.text
    payload = response.json()
    assert set(payload["window"]) == {"from", "to", "days"}
    assert set(payload["summary"]) == set(view["summary"]) | {"truncated", "row_limit"}
    assert payload["summary"]["truncated"] is False
    assert payload["summary"]["row_limit"] == ONLINE_EVALUATION_ROW_LIMIT
    assert set(payload["coverage"]) == set(view["coverage"])

    expected_group = view["issue_groups"][0]
    assert len(payload["issue_groups"]) == 2
    assert set(payload["issue_groups"][0]) == set(expected_group)
    assert set(payload["issue_groups"][0]["representative_cases"][0]) == set(
        expected_group["representative_cases"][0]
    )
    assert set(payload["sources"][0]) == set(view["sources"][0])
    assert set(payload["assessments"][0]) == set(view["assessments"][0])
    assert payload["runtime_events"][0] == view["runtime_events"][0]
