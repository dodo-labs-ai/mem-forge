"""Task discovery and execution contracts for the local agent daemon."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

TaskKind = Literal["kb_sync", "github_sync", "jira_auth"]


@dataclass(frozen=True)
class LocalAgentTask:
    task_id: str
    kind: TaskKind
    interval_seconds: int
    profile_name: str | None = None
    origin: str | None = None


@dataclass(frozen=True)
class LocalAgentHandlers:
    run_kb_profile: Callable[[str], dict[str, Any]]
    run_github_profile: Callable[[str], dict[str, Any]]
    run_jira_auth: Callable[[str, str | None], dict[str, Any]]


def discover_profile_tasks(
    adapter_config: dict[str, Any],
    *,
    default_interval_seconds: int,
) -> list[LocalAgentTask]:
    tasks: list[LocalAgentTask] = []
    interval = _positive_interval(default_interval_seconds)
    for name, profile in sorted(_profiles(adapter_config, "github").items()):
        if _linked(profile) and _daemon_enabled(profile):
            tasks.append(
                LocalAgentTask(
                    task_id=f"github:{name}",
                    kind="github_sync",
                    profile_name=name,
                    interval_seconds=_profile_interval(profile, interval),
                )
            )
    for name, profile in sorted(_profiles(adapter_config, "kb").items()):
        if _linked(profile) and _daemon_enabled(profile):
            tasks.append(
                LocalAgentTask(
                    task_id=f"kb:{name}",
                    kind="kb_sync",
                    profile_name=name,
                    interval_seconds=_profile_interval(profile, interval),
                )
            )
    return tasks


def discover_jira_auth_tasks(
    origins_response: dict[str, Any],
    *,
    default_interval_seconds: int,
) -> list[LocalAgentTask]:
    interval = _positive_interval(default_interval_seconds)
    tasks: list[LocalAgentTask] = []
    origins = origins_response.get("origins")
    if not isinstance(origins, list):
        return tasks
    for entry in origins:
        if not isinstance(entry, dict):
            continue
        origin = str(entry.get("origin") or "").strip().rstrip("/")
        if not origin:
            continue
        configured = entry.get("configured") is True
        status = str(entry.get("status") or "").strip()
        if not configured and not status:
            continue
        tasks.append(
            LocalAgentTask(
                task_id=f"jira-auth:{origin}",
                kind="jira_auth",
                origin=origin,
                interval_seconds=interval,
            )
        )
    return tasks


def run_local_agent_task(
    task: LocalAgentTask,
    handlers: LocalAgentHandlers,
    *,
    previous_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if task.kind == "kb_sync":
        if not task.profile_name:
            raise ValueError("kb_sync task is missing profile_name")
        return handlers.run_kb_profile(task.profile_name)
    if task.kind == "github_sync":
        if not task.profile_name:
            raise ValueError("github_sync task is missing profile_name")
        return handlers.run_github_profile(task.profile_name)
    if task.kind == "jira_auth":
        if not task.origin:
            raise ValueError("jira_auth task is missing origin")
        return handlers.run_jira_auth(task.origin, _previous_cookie_hash(previous_result))
    raise ValueError(f"unsupported local agent task kind: {task.kind}")


def _previous_cookie_hash(previous_result: dict[str, Any] | None) -> str | None:
    if not isinstance(previous_result, dict):
        return None
    payload = previous_result.get("payload")
    if not isinstance(payload, dict):
        return None
    value = str(payload.get("cookie_hash") or "").strip()
    return value or None


def _profiles(adapter_config: dict[str, Any], key: str) -> dict[str, dict[str, Any]]:
    raw = adapter_config.get(key)
    if not isinstance(raw, dict):
        return {}
    return {str(name): profile for name, profile in raw.items() if isinstance(profile, dict)}


def _linked(profile: dict[str, Any]) -> bool:
    return bool(str(profile.get("source_id") or "").strip())


def _daemon_enabled(profile: dict[str, Any]) -> bool:
    value = profile.get("daemon_enabled", True)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _profile_interval(profile: dict[str, Any], default: int) -> int:
    return _positive_interval(profile.get("daemon_interval_seconds"), default=default)


def _positive_interval(value: Any, *, default: int = 3600) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
