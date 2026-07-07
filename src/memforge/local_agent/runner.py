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
        default_sync_interval_seconds: int = 3600,
        jira_interval_seconds: int = 1800,
    ) -> None:
        self.adapter_config = adapter_config
        self.adapter_config_provider = adapter_config_provider
        self.state_store = state_store
        self.handlers = handlers
        self.jira_origins_provider = jira_origins_provider
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
