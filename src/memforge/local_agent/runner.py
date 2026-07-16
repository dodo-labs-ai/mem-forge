"""Local agent daemon runner."""

from __future__ import annotations

from datetime import datetime, timezone
import inspect
import threading
import time
from typing import Any, Callable

from memforge.local_agent.state import LocalAgentStateStore
from memforge.sync_progress import normalize_sync_progress_snapshot

DEFAULT_CLOUD_JOB_LEASE_SECONDS = 60
DEFAULT_CLOUD_JOB_HEARTBEAT_INTERVAL_SECONDS = 20
DEFAULT_CLOUD_JOB_PROGRESS_FLUSH_SECONDS = 2


class CloudJobLeaseLost(RuntimeError):
    """Raised when Cloud rejects the daemon's authority to execute a leased job."""


class LocalAgentRunner:
    def __init__(
        self,
        *,
        state_store: LocalAgentStateStore,
        cloud_job_handler: Callable[..., dict[str, Any]],
        cloud_jobs_provider: Callable[..., dict[str, Any]] | None = None,
        cloud_job_completer: Callable[[str, int, str, dict[str, Any], str | None], dict[str, Any]] | None = None,
        cloud_job_heartbeat: Callable[..., dict[str, Any]] | None = None,
        cloud_job_lease_seconds: int = DEFAULT_CLOUD_JOB_LEASE_SECONDS,
        cloud_job_heartbeat_interval_seconds: int = DEFAULT_CLOUD_JOB_HEARTBEAT_INTERVAL_SECONDS,
    ) -> None:
        self.state_store = state_store
        self.cloud_job_handler = cloud_job_handler
        self.cloud_jobs_provider = cloud_jobs_provider
        self.cloud_job_completer = cloud_job_completer
        self.cloud_job_heartbeat = cloud_job_heartbeat
        self.cloud_job_lease_seconds = _positive_seconds(
            cloud_job_lease_seconds,
            default=DEFAULT_CLOUD_JOB_LEASE_SECONDS,
        )
        self.cloud_job_heartbeat_interval_seconds = _positive_seconds(
            cloud_job_heartbeat_interval_seconds,
            default=DEFAULT_CLOUD_JOB_HEARTBEAT_INTERVAL_SECONDS,
        )

    def run_once(
        self,
        *,
        now: datetime | None = None,
        wait_seconds: int = 0,
    ) -> dict[str, Any]:
        """Lease and execute one batch of server-owned local-agent jobs."""
        results = self._run_cloud_jobs(
            now=now or datetime.now(timezone.utc),
            wait_seconds=wait_seconds,
        )
        return _runner_report(results)

    def _run_cloud_jobs(
        self,
        *,
        now: datetime,
        wait_seconds: int,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        cloud_jobs, cloud_discovery_result = self._lease_cloud_jobs(now, wait_seconds=wait_seconds)
        if cloud_discovery_result is not None:
            self.state_store.record_result(cloud_discovery_result["task_id"], cloud_discovery_result)
            results.append(cloud_discovery_result)
        for job in cloud_jobs:
            result = self._run_cloud_job(job, now)
            self.state_store.record_result(result["task_id"], result)
            results.append(result)
        return results

    def run_forever(
        self,
        *,
        poll_interval_seconds: int = 60,
        cloud_job_wait_seconds: int = 0,
        stop_after_iterations: int | None = None,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] | None = None,
    ) -> None:
        iterations = 0
        while True:
            iteration_failed = False
            try:
                cloud_results = self._run_cloud_jobs(
                    now=datetime.now(timezone.utc),
                    wait_seconds=max(int(cloud_job_wait_seconds), 0),
                )
            except Exception as exc:
                iteration_failed = True
                cloud_results = []
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
            if (
                int(cloud_job_wait_seconds) <= 0
                or self.cloud_jobs_provider is None
                or iteration_failed
                or _cloud_lease_failed(cloud_results)
            ):
                sleep(max(int(poll_interval_seconds), 1))

    def _run_cloud_job(self, job: dict[str, Any], now: datetime) -> dict[str, Any]:
        job_id = str(job.get("job_id") or "").strip()
        task_id = f"cloud-job:{job_id or 'unknown'}"
        started_at = now.isoformat()
        if not job_id:
            return _cloud_job_failed_result(task_id, started_at, "cloud job is missing job_id")
        attempt_count = _job_attempt_count(job)
        if attempt_count is None:
            return _cloud_job_failed_result(task_id, started_at, "cloud job is missing attempt_count")
        try:
            self.state_store.record_running(
                task_id,
                {
                    "task_id": task_id,
                    "kind": "cloud_job",
                    "profile_name": None,
                    "origin": job.get("source_id"),
                    "status": "running",
                    "started_at": started_at,
                    "payload": {
                        "source_id": job.get("source_id"),
                        "operation": job.get("operation"),
                    },
                },
            )
        except Exception:
            # Running-state visibility is best-effort; completing the cloud job is the source of truth.
            pass
        try:
            with _CloudJobLeaseHeartbeat(
                heartbeat=self.cloud_job_heartbeat,
                job_id=job_id,
                attempt_count=attempt_count,
                lease_seconds=self.cloud_job_lease_seconds,
                interval_seconds=self.cloud_job_heartbeat_interval_seconds,
            ) as heartbeat:
                payload = _call_cloud_job_handler(
                    self.cloud_job_handler,
                    job,
                    heartbeat.report_progress,
                )
            latest_progress = heartbeat.latest_progress
            if latest_progress is not None:
                payload = {**payload, "progress": latest_progress}
            heartbeat_errors = heartbeat.errors
            if heartbeat_errors:
                payload = dict(payload)
                payload["heartbeat_errors"] = heartbeat_errors
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
            completion_error = self._complete_cloud_job(
                job_id,
                attempt_count,
                "failed",
                {"retryable": isinstance(exc, (ConnectionError, TimeoutError))},
                error,
            )
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

    def _lease_cloud_jobs(
        self,
        now: datetime,
        *,
        wait_seconds: int = 0,
    ) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        if self.cloud_jobs_provider is None:
            return [], None
        try:
            response = _call_cloud_jobs_provider(
                self.cloud_jobs_provider,
                wait_seconds=wait_seconds,
                lease_seconds=self.cloud_job_lease_seconds,
            )
        except Exception as exc:  # a failed lease must not stop the daemon loop
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
            return [], _cloud_job_failed_result(
                "cloud-jobs:lease",
                now.isoformat(),
                "malformed local-agent lease response: jobs must be a list",
                error_type="CloudJobLeaseError",
            )
        valid_jobs = [job for job in jobs if isinstance(job, dict)]
        finished_at = datetime.now(timezone.utc).isoformat()
        return valid_jobs, {
            "task_id": "cloud-jobs:lease",
            "kind": "cloud_job_lease",
            "profile_name": None,
            "origin": None,
            "status": "success",
            "started_at": now.isoformat(),
            "finished_at": finished_at,
            "payload": {"leased_count": len(valid_jobs)},
        }

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

def _runner_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "ok",
        "counts": {
            "total": len(results),
            "success": sum(1 for item in results if item["status"] == "success"),
            "failed": sum(1 for item in results if item["status"] == "failed"),
        },
        "results": results,
    }


def _cloud_lease_failed(results: list[dict[str, Any]]) -> bool:
    return any(
        result.get("task_id") == "cloud-jobs:lease" and result.get("status") == "failed"
        for result in results
    )


class _CloudJobLeaseHeartbeat:
    def __init__(
        self,
        *,
        heartbeat: Callable[..., dict[str, Any]] | None,
        job_id: str,
        attempt_count: int,
        lease_seconds: int,
        interval_seconds: int,
        progress_flush_seconds: int = DEFAULT_CLOUD_JOB_PROGRESS_FLUSH_SECONDS,
    ) -> None:
        self._heartbeat = heartbeat
        self._job_id = job_id
        self._attempt_count = attempt_count
        self._lease_seconds = lease_seconds
        self._interval_seconds = interval_seconds
        self._progress_flush_seconds = min(progress_flush_seconds, interval_seconds)
        self._stop = threading.Event()
        self._progress_lock = threading.Lock()
        self._progress: dict[str, Any] | None = None
        self._progress_revision = 0
        self._sent_progress_revision = 0
        self._thread: threading.Thread | None = None
        self.errors: list[str] = []

    def __enter__(self) -> _CloudJobLeaseHeartbeat:
        if self._heartbeat is None:
            return self
        self._send_heartbeat(required=True)
        self._thread = threading.Thread(
            target=self._run,
            name=f"memforge-cloud-job-heartbeat-{self._job_id}",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=1)
        with self._progress_lock:
            progress_dirty = self._progress_revision > self._sent_progress_revision
        if progress_dirty:
            self._send_heartbeat()

    def _run(self) -> None:
        next_lease_heartbeat = time.monotonic() + self._interval_seconds
        while not self._stop.wait(self._progress_flush_seconds):
            with self._progress_lock:
                progress_dirty = self._progress_revision > self._sent_progress_revision
            lease_due = time.monotonic() >= next_lease_heartbeat
            if not progress_dirty and not lease_due:
                continue
            self._send_heartbeat()
            if lease_due:
                next_lease_heartbeat = time.monotonic() + self._interval_seconds

    def report_progress(self, progress: dict[str, Any]) -> None:
        with self._progress_lock:
            self._progress = normalize_sync_progress_snapshot(progress)
            self._progress_revision += 1

    @property
    def latest_progress(self) -> dict[str, Any] | None:
        with self._progress_lock:
            return dict(self._progress) if self._progress is not None else None

    def _send_heartbeat(self, *, required: bool = False) -> None:
        if self._heartbeat is None:
            return
        try:
            with self._progress_lock:
                progress = dict(self._progress) if self._progress is not None else None
                progress_revision = self._progress_revision
            response = _call_cloud_job_heartbeat(
                self._heartbeat,
                self._job_id,
                self._attempt_count,
                self._lease_seconds,
                progress,
            )
            if isinstance(response, dict) and response.get("error"):
                error = _api_error_message(response)
                self.errors.append(error)
                if required:
                    raise CloudJobLeaseLost(error)
            else:
                with self._progress_lock:
                    self._sent_progress_revision = max(
                        self._sent_progress_revision,
                        progress_revision,
                    )
        except CloudJobLeaseLost:
            raise
        except Exception as exc:
            self.errors.append(str(exc))
            if required:
                raise CloudJobLeaseLost(str(exc)) from exc


def _call_cloud_job_handler(
    handler: Callable[..., dict[str, Any]],
    job: dict[str, Any],
    report_progress: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    try:
        signature = inspect.signature(handler)
    except (TypeError, ValueError):
        return handler(job)
    if "report_progress" in signature.parameters:
        return handler(job, report_progress=report_progress)
    return handler(job)


def _call_cloud_job_heartbeat(
    heartbeat: Callable[..., dict[str, Any]],
    job_id: str,
    attempt_count: int,
    lease_seconds: int,
    progress: dict[str, Any] | None,
) -> dict[str, Any]:
    try:
        signature = inspect.signature(heartbeat)
    except (TypeError, ValueError):
        return heartbeat(job_id, attempt_count, lease_seconds)
    if "progress" in signature.parameters:
        return heartbeat(job_id, attempt_count, lease_seconds, progress=progress)
    return heartbeat(job_id, attempt_count, lease_seconds)


def _call_cloud_jobs_provider(
    provider: Callable[..., dict[str, Any]],
    *,
    wait_seconds: int,
    lease_seconds: int,
) -> dict[str, Any]:
    try:
        signature = inspect.signature(provider)
    except (TypeError, ValueError):
        return provider()
    kwargs: dict[str, Any] = {}
    if "wait_seconds" in signature.parameters:
        kwargs["wait_seconds"] = wait_seconds
    if "lease_seconds" in signature.parameters:
        kwargs["lease_seconds"] = lease_seconds
    if kwargs:
        return provider(**kwargs)
    return provider()


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


def _positive_seconds(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default
