"""A failed source sync run is retried only when its cause proves the failure transient.

The rule is ``failure_retryable``, the one every stage shares: a provider that
did not answer or answered 408, 409, 429 or 5xx, a storage error the store
classifies as transient, and an exception that declares itself retryable. A
credential the provider rejects, a missing repository and a programming error
fail the same way on every attempt, so the run fails at once.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable

import httpx
import pytest

import memforge.runtime as runtime
from memforge.config import AppConfig
from memforge.genes.toolkit.github_api import GitHubRateLimitError
from memforge.genes.toolkit.http import ConnectorUnreachableError
from memforge.models import SourceSyncRun, SyncState
from memforge.storage.database import Database
from tests.test_document_absence import JIRA_CONFIG
from tests.test_stored_input_currency import SOURCE_ID, Harness, JiraProvider
from tests.test_stored_input_currency import db as db


def _status_error(status: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "https://github.example.com/api/v3/repos/acme/handbook/commits/main")
    return httpx.HTTPStatusError(
        f"Client error '{status}' for url '{request.url}'",
        request=request,
        response=httpx.Response(status, request=request),
    )


def _raised_from(error: BaseException, cause: BaseException) -> BaseException:
    error.__cause__ = cause
    return error


FAILURES = [
    pytest.param(_status_error(401), False, id="credential_rejected"),
    pytest.param(_status_error(404), False, id="repository_missing"),
    pytest.param(KeyError("obs-1"), False, id="programming_error"),
    pytest.param(
        runtime.SourceSyncActivityConflict("source activity heartbeat stopped"), False, id="boundary_crossed"
    ),
    pytest.param(_status_error(503), True, id="provider_unavailable"),
    pytest.param(_status_error(429), True, id="rate_limited"),
    pytest.param(ConnectorUnreachableError("GitHub could not be reached after 4 attempts"), True, id="unreachable"),
    pytest.param(GitHubRateLimitError("GitHub's rate limit resets in 40 minutes"), True, id="rate_limit_resets_later"),
    pytest.param(sqlite3.OperationalError("database is locked"), True, id="database_locked"),
    pytest.param(
        _raised_from(RuntimeError("Failed to list Confluence pages"), ConnectorUnreachableError("VPN down")),
        True,
        id="wrapped_unreachable",
    ),
    pytest.param(_raised_from(RuntimeError("Jira search failed"), _status_error(503)), True, id="wrapped_unavailable"),
    pytest.param(_raised_from(RuntimeError("Jira search failed"), _status_error(401)), False, id="wrapped_rejected"),
]


async def _jira_source(db: Database) -> None:
    await db.upsert_source(
        id=SOURCE_ID, type="jira", name="Jira", config_json=json.dumps(JIRA_CONFIG),
        access_policy="workspace", owner_user_id="dev",
    )


async def _worker_run(db: Database, monkeypatch, outcome: Callable[[], SyncState | None]) -> SourceSyncRun:
    """One manual run of the Jira Source, whose sync ends as ``outcome`` says, as the worker leaves it."""

    await _jira_source(db)
    queued = await db.enqueue_source_sync_run(source_id=SOURCE_ID, trigger="manual")

    class Runtime:
        async def build_sync_runtime(self, *args, **kwargs):
            return object()

        async def run_source_sync(self, **kwargs):
            return outcome()

    async def release(run, source):
        return None

    worker = runtime.SourceSyncWorker(db, AppConfig(), runtime_provider=Runtime(), worker_id="worker-a")
    monkeypatch.setattr(worker, "_release_collection", release)
    assert (await worker.run_once()).run_id == queued.run_id
    return await db.get_source_sync_run(queued.run_id)


def _raising(failure: BaseException) -> Callable[[], SyncState]:
    def outcome() -> SyncState:
        raise failure

    return outcome


@pytest.mark.asyncio
@pytest.mark.parametrize(("failure", "retried"), FAILURES)
async def test_a_run_that_fails_outside_the_pipeline_is_retried_only_when_the_failure_is_transient(
    db, monkeypatch, failure, retried
):
    ended = await _worker_run(db, monkeypatch, _raising(failure))

    assert ended.status == ("pending" if retried else "failed")
    assert (ended.next_attempt_at is not None) is retried


@pytest.mark.asyncio
@pytest.mark.parametrize(("failure", "retried"), FAILURES)
async def test_a_failed_run_declares_itself_retryable_only_when_the_failure_is_transient(db, failure, retried):
    await _jira_source(db)

    class RefusingProvider(JiraProvider):
        async def authenticate(self) -> None:
            raise failure

    harness = Harness(db)
    harness.provider = RefusingProvider()

    state = await harness.sync()

    assert (state.last_sync_status, state.failure_retryable) == ("failed", retried)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(runtime.SourcePausedError("Source is paused"), id="paused"),
        pytest.param(runtime.SourceNotActiveError("Source is being deleted"), id="not_active"),
    ],
)
async def test_a_run_whose_source_cannot_sync_fails_without_a_retry(db, monkeypatch, failure):
    ended = await _worker_run(db, monkeypatch, _raising(failure))

    assert (ended.status, ended.next_attempt_at) == ("failed", None)


@pytest.mark.asyncio
async def test_a_run_that_returns_no_result_fails_without_a_retry(db, monkeypatch):
    ended = await _worker_run(db, monkeypatch, lambda: None)

    assert (ended.status, ended.error_message, ended.next_attempt_at) == (
        "failed", "sync completed without final state", None,
    )
