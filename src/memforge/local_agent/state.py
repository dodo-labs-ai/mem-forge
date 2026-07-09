"""Durable local state for the MemForge local agent daemon."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

STATE_VERSION = 1


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LocalAgentStateStore:
    """Small JSON state store with atomic writes.

    The state is intentionally local-only. It records task outcomes so status
    commands and future daemon restarts can reason about what happened without
    introducing a local database dependency for v1.
    """

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser()

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"version": STATE_VERSION, "tasks": {}}
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError:
            return {"version": STATE_VERSION, "tasks": {}}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._quarantine_corrupt_state()
            return {"version": STATE_VERSION, "tasks": {}}
        if not isinstance(payload, dict):
            self._quarantine_corrupt_state()
            return {"version": STATE_VERSION, "tasks": {}}
        if payload.get("version") != STATE_VERSION:
            self._quarantine_corrupt_state()
            return {"version": STATE_VERSION, "tasks": {}}
        tasks = payload.get("tasks")
        if not isinstance(tasks, dict):
            tasks = {}
        daemon = payload.get("daemon")
        cleaned: dict[str, Any] = {"version": STATE_VERSION, "tasks": tasks}
        if isinstance(daemon, dict):
            cleaned["daemon"] = daemon
        return cleaned

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        cleaned = {
            "version": STATE_VERSION,
            "tasks": payload.get("tasks") if isinstance(payload.get("tasks"), dict) else {},
        }
        if isinstance(payload.get("daemon"), dict):
            cleaned["daemon"] = payload["daemon"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(self.path.parent), prefix=f".{self.path.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(cleaned, handle, indent=2, sort_keys=True)
                handle.write("\n")
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return cleaned

    def record_result(self, task_id: str, result: dict[str, Any]) -> dict[str, Any]:
        payload = self.load()
        tasks = dict(payload.get("tasks") or {})
        previous = tasks.get(task_id) if isinstance(tasks.get(task_id), dict) else {}
        run_count = int(previous.get("run_count") or 0) + 1
        stored_result = _compact_result(result)
        tasks[task_id] = {
            "run_count": run_count,
            "last_status": stored_result.get("status"),
            "last_started_at": stored_result.get("started_at"),
            "last_finished_at": stored_result.get("finished_at"),
            "last_error": stored_result.get("error"),
            "last_result": stored_result,
            "updated_at": utc_now_iso(),
        }
        next_payload: dict[str, Any] = {"version": STATE_VERSION, "tasks": tasks}
        if isinstance(payload.get("daemon"), dict):
            next_payload["daemon"] = payload["daemon"]
        return self.save(next_payload)

    def record_running(self, task_id: str, result: dict[str, Any]) -> dict[str, Any]:
        payload = self.load()
        tasks = dict(payload.get("tasks") or {})
        previous = tasks.get(task_id) if isinstance(tasks.get(task_id), dict) else {}
        stored_result = _compact_result(result)
        tasks[task_id] = {
            "run_count": int(previous.get("run_count") or 0),
            "last_status": "running",
            "last_started_at": stored_result.get("started_at"),
            "last_finished_at": None,
            "last_error": None,
            "last_result": stored_result,
            "updated_at": utc_now_iso(),
        }
        next_payload: dict[str, Any] = {"version": STATE_VERSION, "tasks": tasks}
        if isinstance(payload.get("daemon"), dict):
            next_payload["daemon"] = payload["daemon"]
        return self.save(next_payload)

    def record_daemon_heartbeat(
        self,
        *,
        pid: int,
        started_at: str,
        command: list[str],
        target: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = self.load()
        daemon = {
            "pid": int(pid),
            "started_at": started_at,
            "updated_at": utc_now_iso(),
            "command": list(command),
        }
        if isinstance(target, dict):
            daemon["target"] = deepcopy(target)
        return self.save(
            {
                "version": STATE_VERSION,
                "tasks": payload.get("tasks") if isinstance(payload.get("tasks"), dict) else {},
                "daemon": daemon,
            }
        )

    def _quarantine_corrupt_state(self) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        for index in range(100):
            suffix = f".corrupt-{timestamp}" if index == 0 else f".corrupt-{timestamp}-{index}"
            target = self.path.with_name(f"{self.path.name}{suffix}")
            if target.exists():
                continue
            try:
                self.path.replace(target)
            except OSError:
                return
            return


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    stored = deepcopy({key: value for key, value in result.items() if key != "payload"})
    payload = result.get("payload")
    if isinstance(payload, dict):
        stored["payload"] = _compact_payload(payload)
    return stored


def _compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    preserved_keys = {
        "profile",
        "repo_url",
        "ref",
        "root",
        "vault_id",
        "source_id",
        "counts",
        "action",
        "ok",
        "cookie_hash",
        "error",
        "detail",
        "status_code",
        "sync_started",
        "sync_error",
        "audit_log_path",
        "operation",
        "leased_count",
    }
    compact = {key: deepcopy(payload[key]) for key in preserved_keys if key in payload}

    pushed = payload.get("pushed")
    if isinstance(pushed, list):
        compact["pushed_count"] = len(pushed)

    failed = payload.get("failed")
    if isinstance(failed, list):
        compact["failed_count"] = len(failed)
        if failed:
            compact["first_failed"] = deepcopy(failed[0])

    skipped_existing = payload.get("skipped_existing")
    if isinstance(skipped_existing, list):
        compact["skipped_existing_count"] = len(skipped_existing)
    return compact
