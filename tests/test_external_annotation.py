from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from memforge.evals.external_annotation import LangfuseAnnotationAdapter
from memforge.evals.offline_evaluation import (
    ExternalAnnotationTask,
    ExternalAnnotationTaskState,
)


class _AnnotationQueues:
    def __init__(self) -> None:
        self.item = SimpleNamespace(status="COMPLETED", completed_at=datetime.now(UTC))

    def get_queue(self, queue_id):
        assert queue_id == "queue-1"
        return SimpleNamespace(score_config_ids=["config-1"])

    def get_queue_item(self, queue_id, item_id):
        assert (queue_id, item_id) == ("queue-1", "item-1")
        return self.item


class _ScoreConfigs:
    def __init__(self) -> None:
        self.description = "Evaluate semantic intent."

    def get_by_id(self, config_id):
        assert config_id == "config-1"
        return SimpleNamespace(
            id=config_id,
            name="MemForge semantic intent",
            project_id="project-1",
            data_type="CATEGORICAL",
            is_archived=False,
            categories=[
                SimpleNamespace(label="pass", value=1),
                SimpleNamespace(label="needs_review", value=0.5),
                SimpleNamespace(label="fail", value=0),
            ],
            description=self.description,
        )


class _Scores:
    def __init__(self, score) -> None:
        self.score = score
        self.cursors: list[str | None] = []

    def get_many_v3(self, **kwargs):
        self.cursors.append(kwargs["cursor"])
        if kwargs["cursor"] is None:
            return SimpleNamespace(data=[], meta=SimpleNamespace(cursor="next"))
        return SimpleNamespace(data=[self.score], meta=SimpleNamespace(cursor=None))


def _adapter_and_task():
    subject = SimpleNamespace(
        kind="observation",
        id="observation-1",
        trace_id="1" * 32,
        model_dump=lambda mode: {
            "kind": "observation",
            "id": "observation-1",
            "traceId": "1" * 32,
        },
    )
    score = SimpleNamespace(
        id="score-1",
        source="ANNOTATION",
        project_id="project-1",
        author_user_id="user-1",
        queue_id="queue-1",
        config_id="config-1",
        subject=subject,
        value="pass",
        updated_at=datetime(2026, 8, 18, 12, tzinfo=UTC),
    )
    api = SimpleNamespace(
        annotation_queues=_AnnotationQueues(),
        score_configs=_ScoreConfigs(),
        scores_v3=_Scores(score),
    )
    adapter = LangfuseAnnotationAdapter(SimpleNamespace(api=api))
    binding = adapter.validate_binding(
        queue_id="queue-1",
        reviewer_id="user-1",
        score_config_id="config-1",
    )
    task = ExternalAnnotationTask(
        task_id="aet-1",
        result_id="aeres-1",
        content_policy_id="aep-1",
        criterion="semantic_intent",
        rubric_version="rubric-v1",
        reviewer_id="langfuse:project-1:user-1",
        provider="langfuse",
        provider_project_ref="project-1",
        provider_reviewer_id="user-1",
        queue_id="queue-1",
        score_config_id="config-1",
        score_config_fingerprint=binding.score_config_fingerprint,
        trace_id="1" * 32,
        protected_payload_hash="a" * 64,
        state=ExternalAnnotationTaskState.QUEUED,
        created_at="2026-08-18T10:00:00+00:00",
        updated_at="2026-08-18T10:01:00+00:00",
        observation_id="observation-1",
        queue_item_id="item-1",
    )
    return adapter, task, api


def test_completed_annotation_uses_cursor_and_exact_score_provenance() -> None:
    adapter, task, api = _adapter_and_task()

    imported = adapter.read_completed_annotation(task)

    assert imported.label == "pass"
    assert imported.score_id == "score-1"
    assert api.scores_v3.cursors == [None, "next"]


def test_completed_annotation_rejects_config_drift() -> None:
    adapter, task, api = _adapter_and_task()
    api.score_configs.description = "Changed rubric."

    with pytest.raises(ValueError, match="config drifted"):
        adapter.read_completed_annotation(task)


def test_completed_annotation_rejects_wrong_reviewer_provenance() -> None:
    adapter, task, api = _adapter_and_task()
    api.scores_v3.score.author_user_id = "user-2"

    with pytest.raises(ValueError, match="provenance"):
        adapter.read_completed_annotation(replace(task))
