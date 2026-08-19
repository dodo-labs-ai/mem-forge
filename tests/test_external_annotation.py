from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from memforge.evals.external_annotation import LangfuseAnnotationAdapter
from memforge.evals.offline_evaluation import (
    AgentEvaluationAnnotationTask,
    AgentEvaluationCaseKind,
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
    def __init__(self, labels=("pass", "needs_review", "fail")) -> None:
        self.description = "Evaluate semantic intent."
        self.labels = labels

    def get_by_id(self, config_id):
        assert config_id == "config-1"
        return SimpleNamespace(
            id=config_id,
            name="MemForge semantic intent",
            project_id="project-1",
            data_type="CATEGORICAL",
            is_archived=False,
            categories=[SimpleNamespace(label=label, value=index) for index, label in enumerate(self.labels)],
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


def _adapter_and_task(*, labels=("pass", "needs_review", "fail"), score_value="pass"):
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
        value=score_value,
        updated_at=datetime(2026, 8, 18, 12, tzinfo=UTC),
    )
    api = SimpleNamespace(
        annotation_queues=_AnnotationQueues(),
        score_configs=_ScoreConfigs(labels),
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


def test_completed_annotation_maps_readable_label_to_canonical_value() -> None:
    adapter, task, _api = _adapter_and_task(
        labels=("Correct", "Unsure", "Incorrect"),
        score_value="Correct",
    )

    imported = adapter.read_completed_annotation(task)

    assert imported.label == "pass"


def test_annotation_subject_starts_with_business_readable_review() -> None:
    calls = []
    subject = SimpleNamespace(id="observation-1")
    client = SimpleNamespace(
        start_observation=lambda **kwargs: calls.append(kwargs) or subject,
    )
    adapter = LangfuseAnnotationAdapter(client)
    task = _adapter_and_task()[1]
    protected = AgentEvaluationAnnotationTask(
        result_id="aeres-1",
        case_id="aec-1",
        case_kind=AgentEvaluationCaseKind.SOURCE_UNIT_RECONCILIATION,
        content_policy_id="aep-1",
        case_manifest={
            "doc_type": "jira",
            "updated_document": None,
            "new_extractions": [],
            "incumbents": [{"id": "mem-1", "content": "The tax package is expected in Q1/2027."}],
        },
        candidate_output={
            "operations": [
                {
                    "action": "DELETE",
                    "memory_id": "mem-1",
                    "memory": None,
                    "flag_for_review": False,
                }
            ]
        },
    )

    created = adapter.start_subject(task, protected)

    assert created is subject
    payload = calls[0]
    assert payload["input"]["review"]["question"] == (
        "Are the proposed Memory changes correct for this source evidence?"
    )
    assert payload["input"]["review"]["source"]["state"] == (
        "No current source content is present in this evaluation case."
    )
    assert "does not apply these changes" in payload["input"]["review"]["important"]
    assert payload["output"]["candidate"] == {
        "summary": "Delete 1 existing Memory",
        "proposed_changes": [
            {
                "action": "Delete an existing Memory",
                "current_memory": "The tax package is expected in Q1/2027.",
            }
        ],
    }
    assert payload["input"]["debug_details"]["case_manifest"] == dict(protected.case_manifest)
    assert payload["metadata"]["presentation_version"] == "1"


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
