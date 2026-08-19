"""Durable, provider-neutral human annotation exchange with Langfuse."""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Mapping

from memforge.evals.offline_evaluation import (
    AgentEvaluationCaseKind,
    AgentEvaluationAnnotationTask,
    ExternalAnnotationTask,
    ExternalAnnotationTaskState,
    OfflineAgentEvaluation,
    OfflineEvaluationStore,
)


_ANNOTATION_PRESENTATION_VERSION = "1"
_ANNOTATION_LABELS = {
    frozenset({"fail", "needs_review", "pass"}): {
        "fail": "fail",
        "needs_review": "needs_review",
        "pass": "pass",
    },
    frozenset({"Correct", "Incorrect", "Unsure"}): {
        "Correct": "pass",
        "Incorrect": "fail",
        "Unsure": "needs_review",
    },
}


@dataclass(frozen=True, slots=True)
class LangfuseAnnotationBinding:
    project_ref: str
    queue_id: str
    reviewer_id: str
    score_config_id: str
    score_config_fingerprint: str
    provider_to_canonical_labels: tuple[tuple[str, str], ...] = (
        ("fail", "fail"),
        ("needs_review", "needs_review"),
        ("pass", "pass"),
    )


@dataclass(frozen=True, slots=True)
class ImportedAnnotation:
    score_id: str
    score_updated_at: str
    score_fingerprint: str
    label: str


class LangfuseAnnotationAdapter:
    """Small wrapper over the GA SDK plus Queue and Scores-v3 APIs."""

    def __init__(self, client: Any) -> None:
        self._client = client

    @classmethod
    def from_env(cls) -> LangfuseAnnotationAdapter:
        if os.environ.get("MEMFORGE_LANGFUSE_ANNOTATION_ENABLED", "").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            raise RuntimeError("Langfuse annotation exchange is disabled")
        required = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_BASE_URL")
        if any(not os.environ.get(name, "").strip() for name in required):
            raise RuntimeError("Langfuse annotation exchange configuration is incomplete")
        langfuse = importlib.import_module("langfuse")
        trace_sdk = importlib.import_module("opentelemetry.sdk.trace")
        span_filter = importlib.import_module("langfuse.span_filter")
        client = langfuse.Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            base_url=os.environ["LANGFUSE_BASE_URL"],
            tracer_provider=trace_sdk.TracerProvider(),
            should_export_span=span_filter.is_langfuse_span,
            sample_rate=1.0,
        )
        return cls(client)

    def validate_binding(
        self,
        *,
        queue_id: str,
        reviewer_id: str,
        score_config_id: str,
    ) -> LangfuseAnnotationBinding:
        queue = self._client.api.annotation_queues.get_queue(queue_id)
        if score_config_id not in queue.score_config_ids:
            raise ValueError("annotation score config is not attached to the queue")
        config = self._client.api.score_configs.get_by_id(score_config_id)
        categories = sorted(
            ({"label": item.label, "value": item.value} for item in (config.categories or [])),
            key=lambda item: item["label"],
        )
        if str(config.data_type) not in {"CATEGORICAL", "ScoreConfigDataType.CATEGORICAL"}:
            raise ValueError("annotation score config must be categorical")
        if config.is_archived:
            raise ValueError("annotation score config is archived")
        labels = frozenset(str(item["label"]) for item in categories)
        if labels not in _ANNOTATION_LABELS:
            raise ValueError("annotation score config labels do not match the rubric contract")
        fingerprint = _hash(
            {
                "id": config.id,
                "name": config.name,
                "data_type": "CATEGORICAL",
                "categories": categories,
                "description": config.description,
                "is_archived": config.is_archived,
            }
        )
        return LangfuseAnnotationBinding(
            project_ref=config.project_id,
            queue_id=queue_id,
            reviewer_id=reviewer_id,
            score_config_id=score_config_id,
            score_config_fingerprint=fingerprint,
            provider_to_canonical_labels=tuple(sorted(_ANNOTATION_LABELS[labels].items())),
        )

    def start_subject(
        self,
        task: ExternalAnnotationTask,
        protected: AgentEvaluationAnnotationTask,
    ) -> Any:
        presentation = _annotation_presentation(protected, criterion=task.criterion)
        return self._client.start_observation(
            name="memforge.agent.human_annotation",
            as_type="span",
            trace_context={"trace_id": task.trace_id},
            input=presentation["input"],
            output=presentation["output"],
            metadata={
                "task_id": task.task_id,
                "result_id": task.result_id,
                "case_id": protected.case_id,
                "criterion": task.criterion,
                "rubric_version": task.rubric_version,
                "content_policy_id": task.content_policy_id,
                "presentation_version": _ANNOTATION_PRESENTATION_VERSION,
            },
            version=task.rubric_version,
        )

    def finish_subject(self, subject: Any) -> None:
        subject.end()
        self._client.flush()

    def shutdown(self) -> None:
        self._client.shutdown()

    def subject_exists(self, task: ExternalAnnotationTask) -> bool:
        try:
            trace = self._client.api.trace.get(task.trace_id)
        except Exception as exc:
            if _status_code(exc) == 404:
                return False
            raise
        return any(item.id == task.observation_id for item in trace.observations)

    def find_queue_items(self, task: ExternalAnnotationTask) -> list[Any]:
        matches: list[Any] = []
        page = 1
        while True:
            response = self._client.api.annotation_queues.list_queue_items(
                task.queue_id,
                page=page,
                limit=100,
            )
            matches.extend(
                item
                for item in response.data
                if str(item.object_type).split(".")[-1] == "OBSERVATION" and item.object_id == task.observation_id
            )
            if page >= response.meta.total_pages:
                return matches
            page += 1

    def create_queue_item(self, task: ExternalAnnotationTask) -> Any:
        types = importlib.import_module("langfuse.api.annotation_queues.types")
        return self._client.api.annotation_queues.create_queue_item(
            task.queue_id,
            object_id=str(task.observation_id),
            object_type=types.AnnotationQueueObjectType.OBSERVATION,
        )

    def read_completed_annotation(self, task: ExternalAnnotationTask) -> ImportedAnnotation:
        item = self._client.api.annotation_queues.get_queue_item(
            task.queue_id,
            str(task.queue_item_id),
        )
        if str(item.status).split(".")[-1] != "COMPLETED" or item.completed_at is None:
            raise ValueError("annotation queue item is not completed")
        binding = self.validate_binding(
            queue_id=task.queue_id,
            reviewer_id=task.provider_reviewer_id,
            score_config_id=task.score_config_id,
        )
        if (
            binding.project_ref != task.provider_project_ref
            or binding.score_config_fingerprint != task.score_config_fingerprint
        ):
            raise ValueError("annotation score config drifted")
        scores: dict[str, Any] = {}
        cursor: str | None = None
        while True:
            response = self._client.api.scores_v3.get_many_v3(
                limit=100,
                cursor=cursor,
                fields="details,subject,annotation",
                source="ANNOTATION",
                queue_id=task.queue_id,
                author_user_id=task.provider_reviewer_id,
                config_id=task.score_config_id,
                trace_id=task.trace_id,
                observation_id=task.observation_id,
            )
            for score in response.data:
                scores[score.id] = score
            cursor = response.meta.cursor
            if not cursor:
                break
        if len(scores) != 1:
            raise ValueError("annotation import requires exactly one matching score")
        score = next(iter(scores.values()))
        subject = score.subject
        if (
            str(score.source).split(".")[-1] != "ANNOTATION"
            or score.project_id != task.provider_project_ref
            or score.author_user_id != task.provider_reviewer_id
            or score.queue_id != task.queue_id
            or score.config_id != task.score_config_id
            or subject is None
            or getattr(subject, "kind", None) != "observation"
            or subject.id != task.observation_id
            or subject.trace_id != task.trace_id
        ):
            raise ValueError("annotation score provenance does not match the task")
        provider_label = str(score.value)
        label = dict(binding.provider_to_canonical_labels).get(provider_label)
        if label is None:
            raise ValueError("annotation score label is outside the rubric contract")
        snapshot = {
            "id": score.id,
            "updated_at": score.updated_at.isoformat(),
            "value": provider_label,
            "author_user_id": score.author_user_id,
            "queue_id": score.queue_id,
            "config_id": score.config_id,
            "subject": subject.model_dump(mode="json"),
        }
        return ImportedAnnotation(
            score_id=score.id,
            score_updated_at=score.updated_at.isoformat(),
            score_fingerprint=_hash(snapshot),
            label=label,
        )


def _annotation_presentation(
    protected: AgentEvaluationAnnotationTask,
    *,
    criterion: str,
) -> dict[str, dict[str, object]]:
    if protected.case_kind is AgentEvaluationCaseKind.SOURCE_UNIT_RECONCILIATION:
        review, candidate = _reconciliation_presentation(protected)
    else:
        review = {
            "title": "Review candidate Memory extraction",
            "question": f"Does this candidate satisfy the {criterion.replace('_', ' ')} criterion?",
            "important": "Submitting this review records a label only. It does not change production Memory.",
        }
        candidate = {"summary": "Review the candidate output against the source evidence."}
    review["how_to_score"] = {
        "Correct": "The candidate is correct for the source evidence.",
        "Incorrect": "The candidate contains a material error or unsafe change.",
        "Unsure": "The available evidence is insufficient to decide.",
    }
    return {
        "input": {
            "review": review,
            "debug_details": {"case_manifest": dict(protected.case_manifest)},
        },
        "output": {
            "candidate": candidate,
            "debug_details": {"candidate_output": dict(protected.candidate_output)},
        },
    }


def _reconciliation_presentation(
    protected: AgentEvaluationAnnotationTask,
) -> tuple[dict[str, object], dict[str, object]]:
    manifest = protected.case_manifest
    candidate_output = protected.candidate_output
    incumbents = {
        str(item.get("id")): item for item in _mapping_items(manifest.get("incumbents")) if item.get("id") is not None
    }
    operations = _mapping_items(candidate_output.get("operations"))
    proposed_changes = [_operation_presentation(operation, incumbents=incumbents) for operation in operations]
    updated_document = manifest.get("updated_document")
    source: dict[str, object] = {
        "type": str(manifest.get("doc_type") or "source"),
        "state": (
            "No current source content is present in this evaluation case."
            if updated_document is None
            else "Current source content is present in this evaluation case."
        ),
        "existing_memory_count": len(incumbents),
        "new_extraction_count": len(_mapping_items(manifest.get("new_extractions"))),
    }
    if updated_document is not None:
        source["current_content"] = updated_document
    action_counts: dict[str, int] = {}
    for operation in operations:
        action = str(operation.get("action") or "UNKNOWN")
        action_counts[action] = action_counts.get(action, 0) + 1
    summary = (
        "; ".join(_action_count_summary(action, count) for action, count in sorted(action_counts.items()))
        or "No Memory changes proposed"
    )
    return (
        {
            "title": "Review proposed Memory changes",
            "question": "Are the proposed Memory changes correct for this source evidence?",
            "source": source,
            "important": "Submitting this review records a label only. It does not apply these changes to production Memory.",
        },
        {
            "summary": summary,
            "proposed_changes": proposed_changes,
        },
    )


def _operation_presentation(
    operation: Mapping[str, object],
    *,
    incumbents: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    action = str(operation.get("action") or "UNKNOWN")
    memory_id = str(operation.get("memory_id") or "")
    existing = incumbents.get(memory_id)
    proposed = operation.get("memory")
    result: dict[str, object] = {"action": _action_label(action)}
    if existing is not None and existing.get("content") is not None:
        result["current_memory"] = existing["content"]
    if isinstance(proposed, Mapping) and proposed.get("content") is not None:
        result["proposed_memory"] = proposed["content"]
    if operation.get("flag_for_review") is True:
        result["requires_additional_review"] = True
    return result


def _action_label(action: str) -> str:
    return {
        "ADD": "Add a new Memory",
        "UPDATE": "Update an existing Memory",
        "SUPERSEDE": "Replace an existing Memory",
        "DELETE": "Delete an existing Memory",
        "NOOP": "Keep an existing Memory unchanged",
    }.get(action, "Review an unknown Memory action")


def _action_count_summary(action: str, count: int) -> str:
    memory = "Memory" if count == 1 else "Memories"
    return {
        "ADD": f"Add {count} new {memory}",
        "UPDATE": f"Update {count} existing {memory}",
        "SUPERSEDE": f"Replace {count} existing {memory}",
        "DELETE": f"Delete {count} existing {memory}",
        "NOOP": f"Keep {count} existing {memory} unchanged",
    }.get(action, f"Review {count} unknown Memory actions")


def _mapping_items(value: object) -> list[Mapping[str, object]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


class ExternalAnnotationExchange:
    """Own bridge state while keeping protected content out of that state."""

    def __init__(
        self,
        store: OfflineEvaluationStore,
        evaluation: OfflineAgentEvaluation,
        adapter: LangfuseAnnotationAdapter,
        *,
        lease_seconds: int = 120,
    ) -> None:
        self._store = store
        self._evaluation = evaluation
        self._adapter = adapter
        self._lease_seconds = lease_seconds

    async def export(
        self,
        *,
        result_id: str,
        content_policy_id: str,
        criterion: str,
        rubric_version: str,
        actor_user_id: str,
        provider_reviewer_id: str,
        queue_id: str,
        score_config_id: str,
        lease_owner: str,
    ) -> ExternalAnnotationTask:
        binding = await asyncio.to_thread(
            self._adapter.validate_binding,
            queue_id=queue_id,
            reviewer_id=provider_reviewer_id,
            score_config_id=score_config_id,
        )
        reviewer_id = f"langfuse:{binding.project_ref}:{binding.reviewer_id}"
        task, protected = await self._evaluation.prepare_langfuse_annotation_task(
            result_id=result_id,
            content_policy_id=content_policy_id,
            criterion=criterion,
            rubric_version=rubric_version,
            reviewer_id=reviewer_id,
            actor_user_id=actor_user_id,
            provider_project_ref=binding.project_ref,
            provider_reviewer_id=binding.reviewer_id,
            queue_id=binding.queue_id,
            score_config_id=binding.score_config_id,
            score_config_fingerprint=binding.score_config_fingerprint,
        )
        if task.state is ExternalAnnotationTaskState.IMPORTED:
            return task
        claimed = await self._claim(task.task_id, lease_owner)
        if claimed is None:
            raise RuntimeError("external annotation task is already leased")
        task = claimed
        token = str(task.lease_token)
        try:
            if task.state is ExternalAnnotationTaskState.PREPARED:
                subject = await asyncio.to_thread(
                    self._adapter.start_subject,
                    task,
                    protected,
                )
                task = replace(
                    task,
                    state=ExternalAnnotationTaskState.SUBJECT_PREPARED,
                    observation_id=subject.id,
                    updated_at=_now(),
                )
                await self._require_update(task, token)
                await asyncio.to_thread(self._adapter.finish_subject, subject)
                task = replace(
                    task,
                    state=ExternalAnnotationTaskState.SUBJECT_READY,
                    updated_at=_now(),
                )
                await self._require_update(task, token)
            elif task.state is ExternalAnnotationTaskState.SUBJECT_PREPARED:
                if not await asyncio.to_thread(self._adapter.subject_exists, task):
                    raise RuntimeError("external annotation subject delivery is ambiguous")
                task = replace(
                    task,
                    state=ExternalAnnotationTaskState.SUBJECT_READY,
                    updated_at=_now(),
                )
                await self._require_update(task, token)
            if task.state is ExternalAnnotationTaskState.SUBJECT_READY:
                matches = await asyncio.to_thread(self._adapter.find_queue_items, task)
                if not matches:
                    try:
                        item = await asyncio.to_thread(
                            self._adapter.create_queue_item,
                            task,
                        )
                    except Exception:
                        matches = await asyncio.to_thread(
                            self._adapter.find_queue_items,
                            task,
                        )
                        if len(matches) != 1:
                            raise
                        item = matches[0]
                elif len(matches) == 1:
                    item = matches[0]
                else:
                    conflict = replace(
                        task,
                        state=ExternalAnnotationTaskState.CONFLICT,
                        error_code="duplicate_queue_item",
                        lease_owner=None,
                        lease_token=None,
                        lease_expires_at=None,
                        updated_at=_now(),
                    )
                    await self._require_update(conflict, token)
                    return conflict
                task = replace(
                    task,
                    state=ExternalAnnotationTaskState.QUEUED,
                    queue_item_id=item.id,
                    lease_owner=None,
                    lease_token=None,
                    lease_expires_at=None,
                    updated_at=_now(),
                )
                await self._require_update(task, token)
            return task
        except Exception:
            await self._release(task, token)
            raise

    async def import_completed(
        self,
        *,
        task_id: str,
        submitted_by: str,
        lease_owner: str,
    ) -> ExternalAnnotationTask:
        claimed = await self._claim(task_id, lease_owner)
        if claimed is None:
            raise RuntimeError("external annotation task is already leased")
        if claimed.state is ExternalAnnotationTaskState.IMPORTED:
            return claimed
        token = str(claimed.lease_token)
        try:
            if claimed.state is not ExternalAnnotationTaskState.QUEUED:
                raise ValueError("external annotation task is not queued")
            imported = await asyncio.to_thread(
                self._adapter.read_completed_annotation,
                claimed,
            )
            assessment = await self._evaluation.build_human_annotation(
                result_id=claimed.result_id,
                content_policy_id=claimed.content_policy_id,
                criterion=claimed.criterion,
                label=imported.label,  # type: ignore[arg-type]
                reason_code=f"langfuse_score:{imported.score_id}",
                rubric_version=claimed.rubric_version,
                reviewer_id=claimed.reviewer_id,
                submitted_by=submitted_by,
                external_provider="langfuse",
                external_annotation_id=imported.score_id,
            )
            terminal = replace(
                claimed,
                state=ExternalAnnotationTaskState.IMPORTED,
                provider_score_id=imported.score_id,
                provider_score_updated_at=imported.score_updated_at,
                provider_score_fingerprint=imported.score_fingerprint,
                assessment_id=assessment.assessment_id,
                lease_owner=None,
                lease_token=None,
                lease_expires_at=None,
                updated_at=_now(),
            )
            ok = await self._store.record_external_annotation_import(
                terminal,
                assessment,
                expected_lease_token=token,
            )
            if not ok:
                raise RuntimeError("external annotation import lease is stale")
            return terminal
        except Exception:
            await self._release(claimed, token)
            raise

    async def _claim(
        self,
        task_id: str,
        owner: str,
    ) -> ExternalAnnotationTask | None:
        now = datetime.now(UTC)
        return await self._store.claim_external_annotation_task(
            task_id=task_id,
            lease_owner=owner,
            now=now.isoformat(),
            lease_expires_at=(now + timedelta(seconds=self._lease_seconds)).isoformat(),
        )

    async def _require_update(
        self,
        task: ExternalAnnotationTask,
        token: str,
    ) -> None:
        if not await self._store.update_external_annotation_task(
            task,
            expected_lease_token=token,
        ):
            raise RuntimeError("external annotation task lease is stale")

    async def _release(self, task: ExternalAnnotationTask, token: str) -> None:
        released = replace(
            task,
            lease_owner=None,
            lease_token=None,
            lease_expires_at=None,
            updated_at=_now(),
        )
        await self._store.update_external_annotation_task(
            released,
            expected_lease_token=token,
        )


def _hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _status_code(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None) or getattr(exc, "status", None)


def _now() -> str:
    return datetime.now(UTC).isoformat()
