"""Durable worker for service-owned offline Agent Evaluation Runs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from memforge.evals.offline_evaluation import (
    AgentEvaluationRun,
    AgentEvaluationRunExecution,
    AgentEvaluationRunStatus,
    OfflineAgentEvaluation,
    OfflineEvaluationStore,
)


logger = logging.getLogger(__name__)

OfflineEvaluationFactory = Callable[
    [AgentEvaluationRun],
    Awaitable[OfflineAgentEvaluation],
]


class OfflineEvaluationWorker:
    """Claim and execute one durable offline run at a time."""

    def __init__(
        self,
        store: OfflineEvaluationStore,
        *,
        evaluation_factory: OfflineEvaluationFactory,
        worker_id: str,
        lease_seconds: float = 120.0,
    ) -> None:
        if not worker_id:
            raise ValueError("offline evaluation worker requires worker_id")
        if lease_seconds <= 0:
            raise ValueError("offline evaluation lease must be positive")
        self._store = store
        self._evaluation_factory = evaluation_factory
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds

    async def run_once(self) -> AgentEvaluationRunExecution | None:
        now = datetime.now(UTC)
        execution = await self._store.claim_agent_evaluation_run(
            worker_id=self._worker_id,
            now=now.isoformat(),
            lease_expires_at=(now + timedelta(seconds=self._lease_seconds)).isoformat(),
        )
        if execution is None:
            return None
        if not execution.lease_token:
            raise RuntimeError("claimed evaluation execution has no lease token")
        run = await self._store.get_agent_evaluation_run(execution.run_id)
        if run is None:
            raise RuntimeError("claimed evaluation execution has no run")
        if run.status is not AgentEvaluationRunStatus.RUNNING:
            await self._store.finish_agent_evaluation_run(
                run,
                lease_token=execution.lease_token,
                finished_at=datetime.now(UTC).isoformat(),
            )
            return execution

        stop = asyncio.Event()
        lease_lost = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._heartbeat(
                run_id=run.run_id,
                lease_token=execution.lease_token,
                stop=stop,
                lease_lost=lease_lost,
            )
        )
        try:
            evaluation = await self._evaluation_factory(run)
            await evaluation.execute_admitted_run(
                run.run_id,
                lease_token=execution.lease_token,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            if lease_lost.is_set():
                logger.warning(
                    "Offline evaluation worker lost lease for run %s",
                    run.run_id,
                )
            else:
                logger.exception("Offline evaluation worker failed run %s", run.run_id)
                failed = replace(
                    run,
                    status=AgentEvaluationRunStatus.FAILED,
                    completed_at=datetime.now(UTC).isoformat(),
                )
                await self._store.finish_agent_evaluation_run(
                    failed,
                    lease_token=execution.lease_token,
                    finished_at=failed.completed_at,
                )
        finally:
            stop.set()
            await heartbeat
        return execution

    async def run_forever(self, *, poll_seconds: float = 5.0) -> None:
        if poll_seconds <= 0:
            raise ValueError("offline evaluation poll interval must be positive")
        while True:
            execution = await self.run_once()
            if execution is None:
                await asyncio.sleep(poll_seconds)

    async def _heartbeat(
        self,
        *,
        run_id: str,
        lease_token: str,
        stop: asyncio.Event,
        lease_lost: asyncio.Event,
    ) -> None:
        interval = max(0.1, self._lease_seconds / 3)
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
                return
            except TimeoutError:
                now = datetime.now(UTC)
                renewed = await self._store.heartbeat_agent_evaluation_run(
                    run_id=run_id,
                    lease_token=lease_token,
                    now=now.isoformat(),
                    lease_expires_at=(
                        now + timedelta(seconds=self._lease_seconds)
                    ).isoformat(),
                )
                if not renewed:
                    lease_lost.set()
                    return
