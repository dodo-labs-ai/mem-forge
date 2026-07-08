"""Local agent daemon runner."""

from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any, Callable

from memforge.local_agent.state import LocalAgentStateStore
from memforge.local_agent.tasks import (
    LocalAgentHandlers,
    LocalAgentTask,
    discover_jira_auth_tasks,
    discover_profile_tasks,
    run_local_agent_task,
)

FAILED_TASK_RETRY_SECONDS = 300


class LocalAgentRunner:
    def __init__(
        self,
        *,
        adapter_config: dict[str, Any],
        adapter_config_provider: Callable[[], dict[str, Any]] | None = None,
        state_store: LocalAgentStateStore,
        handlers: LocalAgentHandlers,
        jira_origins_provider: Callable[[], dict[str, Any]] | None = None,
        cloud_jobs_provider: Callable[[], dict[str, Any]] | None = None,
        cloud_job_completer: Callable[[str, int, str, dict[str, Any], str | None], dict[str, Any]] | None = None,
        default_sync_interval_seconds: int = 3600,
        jira_interval_seconds: int = 1800,
    ) -> None:
        self.adapter_config = adapter_config
        self.adapter_config_provider = adapter_config_provider
        self.state_store = state_store
        self.handlers = handlers
        self.jira_origins_provider = jira_origins_provider
        self.cloud_jobs_provider = cloud_jobs_provider
        self.cloud_job_completer = cloud_job_completer
        self.default_sync_interval_seconds = _positive_seconds(default_sync_interval_seconds, default=3600)
        self.jira_interval_seconds = _positive_seconds(jira_interval_seconds, default=1800)
        self._cached_jira_tasks: list[LocalAgentTask] = []
        self._last_jira_discovery_at: datetime | None = None

    def discover_tasks(self, *, include_jira: bool) -> list[LocalAgentTask]:
        tasks = discover_profile_tasks(
            self._adapter_config(),
            default_interval_seconds=self.default_sync_interval_seconds,
        )
        if include_jira:
            jira_tasks, _ = self._discover_jira_tasks(datetime.now(timezone.utc), only_due=False)
            tasks.extend(jira_tasks)
        return tasks

    def run_once(
        self,
        *,
        now: datetime | None = None,
        include_jira: bool = True,
        only_due: bool = False,
    ) -> dict[str, Any]:
        now = now or datetime.now(timezone.utc)
        tasks = self.discover_tasks(include_jira=False)
        results: list[dict[str, Any]] = []
        if include_jira:
            jira_tasks, discovery_result = self._discover_jira_tasks(now, only_due=only_due)
            tasks.extend(jira_tasks)
            if discovery_result is not None:
                self.state_store.record_result(discovery_result["task_id"], discovery_result)
                results.append(discovery_result)
        state = self.state_store.load()
        for task in tasks:
            if only_due and not _task_due(task, state, now):
                results.append(_skipped_result(task, now, reason="not_due"))
                continue
            result = self._run_task(task, now, _previous_state_result(state, task.task_id))
            self.state_store.record_result(task.task_id, result)
            results.append(result)
        cloud_jobs, cloud_discovery_result = self._lease_cloud_jobs(now)
        if cloud_discovery_result is not None:
            self.state_store.record_result(cloud_discovery_result["task_id"], cloud_discovery_result)
            results.append(cloud_discovery_result)
        for job in cloud_jobs:
            result = self._run_cloud_job(job, now)
            self.state_store.record_result(result["task_id"], result)
            results.append(result)
        return {
            "status": "ok",
            "counts": {
                "total": len(results),
                "success": sum(1 for item in results if item["status"] == "success"),
                "failed": sum(1 for item in results if item["status"] == "failed"),
                "skipped": sum(1 for item in results if item["status"] == "skipped"),
            },
            "results": results,
        }

    def run_forever(
        self,
        *,
        include_jira: bool = True,
        poll_interval_seconds: int = 60,
        stop_after_iterations: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] | None = None,
    ) -> None:
        iterations = 0
        while True:
            try:
                self.run_once(include_jira=include_jira, only_due=True)
            except Exception as exc:
                if log is not None:
                    log(f"Local agent daemon iteration failed: {exc}")
                try:
                    result = _runner_failed_result(datetime.now(timezone.utc), exc)
                    self.state_store.record_result(result["task_id"], result)
                except Exception as state_exc:
                    if log is not None:
                        log(f"Local agent daemon could not record runner failure: {state_exc}")
            iterations += 1
            if stop_after_iterations is not None and iterations >= stop_after_iterations:
                return
            sleep(max(int(poll_interval_seconds), 1))

    def _run_task(
        self,
        task: LocalAgentTask,
        now: datetime,
        previous_result: dict[str, Any] | None,
    ) -> dict[str, Any]:
        started_at = now.isoformat()
        try:
            payload = run_local_agent_task(task, self.handlers, previous_result=previous_result)
            error = _payload_error(task, payload)
            if error:
                return {
                    "task_id": task.task_id,
                    "kind": task.kind,
                    "profile_name": task.profile_name,
                    "origin": task.origin,
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error": error,
                    "payload": payload,
                }
            return {
                "task_id": task.task_id,
                "kind": task.kind,
                "profile_name": task.profile_name,
                "origin": task.origin,
                "status": "success",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
            }
        except Exception as exc:  # daemon keeps other tasks alive
            return {
                "task_id": task.task_id,
                "kind": task.kind,
                "profile_name": task.profile_name,
                "origin": task.origin,
                "status": "failed",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": str(exc),
                "error_type": type(exc).__name__,
            }

    def _run_cloud_job(self, job: dict[str, Any], now: datetime) -> dict[str, Any]:
        job_id = str(job.get("job_id") or "").strip()
        task_id = f"cloud-job:{job_id or 'unknown'}"
        started_at = now.isoformat()
        if not job_id:
            return _cloud_job_failed_result(task_id, started_at, "cloud job is missing job_id")
        attempt_count = _job_attempt_count(job)
        if attempt_count is None:
            return _cloud_job_failed_result(task_id, started_at, "cloud job is missing attempt_count")
        if self.handlers.run_cloud_job is None:
            return _cloud_job_failed_result(task_id, started_at, "cloud job handler is not configured")
        try:
            payload = self.handlers.run_cloud_job(job)
            error = str(payload.get("error") or "").strip()
            status = "failed" if error else "success"
            completion_status = "failed" if error else "succeeded"
            completion_error = self._complete_cloud_job(job_id, attempt_count, completion_status, payload, error or None)
            if completion_error:
                return {
                    "task_id": task_id,
                    "kind": "cloud_job",
                    "profile_name": None,
                    "origin": job.get("source_id"),
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "error": completion_error,
                    "error_type": "CloudJobCompletionError",
                    "payload": payload,
                }
            result = {
                "task_id": task_id,
                "kind": "cloud_job",
                "profile_name": None,
                "origin": job.get("source_id"),
                "status": status,
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "payload": payload,
            }
            if error:
                result["error"] = error
            return result
        except Exception as exc:  # cloud jobs must not stop the daemon loop
            error = str(exc)
            completion_error = self._complete_cloud_job(job_id, attempt_count, "failed", {}, error)
            if completion_error:
                error = f"{error}; completion failed: {completion_error}"
            return {
                "task_id": task_id,
                "kind": "cloud_job",
                "profile_name": None,
                "origin": job.get("source_id"),
                "status": "failed",
                "started_at": started_at,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "error": error,
                "error_type": type(exc).__name__,
            }

    def _lease_cloud_jobs(self, now: datetime) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        if self.cloud_jobs_provider is None:
            return [], None
        try:
            response = self.cloud_jobs_provider()
        except Exception as exc:  # cloud polling must not block local profile work
            started_at = now.isoformat()
            return [], _cloud_job_failed_result(
                "cloud-jobs:lease",
                started_at,
                str(exc),
                error_type=type(exc).__name__,
            )
        if isinstance(response, dict) and response.get("error"):
            return [], _cloud_job_failed_result(
                "cloud-jobs:lease",
                now.isoformat(),
                _api_error_message(response),
                error_type="CloudJobLeaseError",
            )
        jobs = response.get("jobs") if isinstance(response, dict) else None
        if not isinstance(jobs, list):
            return [], None
        return [job for job in jobs if isinstance(job, dict)], None

    def _complete_cloud_job(
        self,
        job_id: str,
        attempt_count: int,
        status: str,
        result: dict[str, Any],
        error: str | None,
    ) -> str | None:
        if self.cloud_job_completer is not None:
            try:
                response = self.cloud_job_completer(job_id, attempt_count, status, result, error)
                if isinstance(response, dict) and response.get("error"):
                    return _api_error_message(response)
            except Exception as exc:  # completion failures should not abort later jobs
                return str(exc)
        return None

    def _discover_jira_tasks(
        self,
        now: datetime,
        *,
        only_due: bool,
    ) -> tuple[list[LocalAgentTask], dict[str, Any] | None]:
        if self.jira_origins_provider is None:
            return [], None
        if (
            only_due
            and self._last_jira_discovery_at is not None
            and (now - self._last_jira_discovery_at).total_seconds() < self.jira_interval_seconds
        ):
            return list(self._cached_jira_tasks), None
        try:
            tasks = discover_jira_auth_tasks(
                self.jira_origins_provider(),
                default_interval_seconds=self.jira_interval_seconds,
            )
            self._cached_jira_tasks = list(tasks)
            self._last_jira_discovery_at = now
            return tasks, None
        except Exception as exc:
            if self._cached_jira_tasks:
                self._last_jira_discovery_at = now
            return list(self._cached_jira_tasks), _discovery_failed_result(now, exc)

    def _adapter_config(self) -> dict[str, Any]:
        if self.adapter_config_provider is not None:
            return self.adapter_config_provider()
        return self.adapter_config


def _payload_error(task: LocalAgentTask, payload: dict[str, Any]) -> str | None:
    error = payload.get("error")
    if error:
        return str(error)
    if task.kind == "jira_auth":
        ok = payload.get("ok")
        if ok is False:
            action = str(payload.get("action") or "unknown").strip() or "unknown"
            return f"Jira browser-session refresh returned {action}"
        action = str(payload.get("action") or "").strip()
        if action and action not in {"uploaded", "unchanged"}:
            return f"Jira browser-session refresh returned {action}"
    return None


def _previous_state_result(state: dict[str, Any], task_id: str) -> dict[str, Any] | None:
    tasks = state.get("tasks") if isinstance(state.get("tasks"), dict) else {}
    entry = tasks.get(task_id) if isinstance(tasks, dict) else None
    if not isinstance(entry, dict):
        return None
    result = entry.get("last_result")
    return result if isinstance(result, dict) else None


def _discovery_failed_result(now: datetime, exc: Exception) -> dict[str, Any]:
    return {
        "task_id": "jira-auth:discovery",
        "kind": "jira_auth",
        "profile_name": None,
        "origin": None,
        "status": "failed",
        "started_at": now.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "error": str(exc),
        "error_type": type(exc).__name__,
    }


def _runner_failed_result(now: datetime, exc: Exception) -> dict[str, Any]:
    return {
        "task_id": "runner:error",
        "kind": "runner",
        "profile_name": None,
        "origin": None,
        "status": "failed",
        "started_at": now.isoformat(),
        "finished_at": now.isoformat(),
        "error": str(exc),
        "error_type": type(exc).__name__,
    }


def _cloud_job_failed_result(
    task_id: str,
    started_at: str,
    error: str,
    *,
    error_type: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "task_id": task_id,
        "kind": "cloud_job",
        "profile_name": None,
        "origin": None,
        "status": "failed",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
    if error_type:
        result["error_type"] = error_type
    return result


def _job_attempt_count(job: dict[str, Any]) -> int | None:
    value = job.get("attempt_count")
    if isinstance(value, bool):
        return None
    try:
        attempt_count = int(value)
    except (TypeError, ValueError):
        return None
    return attempt_count if attempt_count > 0 else None


def _api_error_message(response: dict[str, Any]) -> str:
    detail = str(response.get("detail") or "").strip()
    status_code = response.get("status_code")
    error = str(response.get("error") or "MemForge API request failed").strip()
    parts = [error]
    if status_code is not None:
        parts.append(f"status_code={status_code}")
    if detail:
        parts.append(detail)
    return ": ".join(parts)


def _skipped_result(task: LocalAgentTask, now: datetime, *, reason: str) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "kind": task.kind,
        "profile_name": task.profile_name,
        "origin": task.origin,
        "status": "skipped",
        "reason": reason,
        "started_at": now.isoformat(),
        "finished_at": now.isoformat(),
    }


def _task_due(task: LocalAgentTask, state: dict[str, Any], now: datetime) -> bool:
    tasks = state.get("tasks") if isinstance(state.get("tasks"), dict) else {}
    entry = tasks.get(task.task_id) if isinstance(tasks, dict) else None
    if not isinstance(entry, dict):
        return True
    if entry.get("last_status") == "failed":
        retry_seconds = min(task.interval_seconds, FAILED_TASK_RETRY_SECONDS)
        return _finished_at_elapsed(entry, now, retry_seconds)
    return _finished_at_elapsed(entry, now, task.interval_seconds)


def _finished_at_elapsed(entry: dict[str, Any], now: datetime, seconds: int) -> bool:
    finished_at = str(entry.get("last_finished_at") or "").strip()
    if not finished_at:
        return True
    try:
        previous = datetime.fromisoformat(finished_at.replace("Z", "+00:00"))
    except ValueError:
        return True
    if previous.tzinfo is None or previous.utcoffset() is None:
        previous = previous.replace(tzinfo=timezone.utc)
    return (now - previous).total_seconds() >= seconds


def _positive_seconds(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
