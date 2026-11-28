"""Moving where a Source's acquisition runs (ADR 0058).

The Source row and its local-agent jobs live in different stores in some
deployments, so a move is three steps, each in its own store. It refuses
while a daemon holds the Source's ``document_acquisition`` job, changes the
Source, and then cancels the Source's queued jobs planned for the previous
executor. A job leased between the first two steps names the previous
device; the server refuses its submissions, because that device no longer
executes the Source.
"""

from __future__ import annotations

from typing import Any, Protocol

from memforge.local_agent.job_store import LocalAgentJobStore
from memforge.source_activity import SourceAcquisitionLeased


class _ExecutorChangingDatabase(Protocol):
    async def change_source_executor(
        self,
        source_id: str,
        *,
        executor: str,
        config_json: str,
        execution_owner_user_id: str | None = None,
        execution_device_id: str | None = None,
    ) -> Any: ...


async def move_source_executor(
    db: _ExecutorChangingDatabase,
    jobs: LocalAgentJobStore,
    source_id: str,
    *,
    executor: str,
    config_json: str,
    execution_owner_user_id: str | None = None,
    execution_device_id: str | None = None,
) -> list[str]:
    """Move the Source to ``executor`` and return the queued jobs this cancelled.

    Raises :class:`SourceAcquisitionLeased` while a daemon holds the Source's
    ``document_acquisition`` job, and whatever ``change_source_executor``
    raises: :class:`memforge.source_activity.SourceSyncRunActive` during a run
    and ``LookupError`` for an unknown Source.
    """

    leased_job_id = await jobs.leased_document_acquisition_job(source_id)
    if leased_job_id is not None:
        raise SourceAcquisitionLeased(f"source acquisition job is leased: {leased_job_id}", job_id=leased_job_id)
    device_id = execution_device_id if executor == "device" else None
    await db.change_source_executor(
        source_id,
        executor=executor,
        config_json=config_json,
        execution_owner_user_id=execution_owner_user_id,
        execution_device_id=device_id,
    )
    return await jobs.cancel_queued_source_jobs(source_id, execution_device_id=device_id)
