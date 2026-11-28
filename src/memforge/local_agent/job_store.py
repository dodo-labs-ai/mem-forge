"""The store of local-agent jobs and devices that a workspace's server code reads and writes.

A deployment keeps its daemon jobs and devices where one daemon can reach
the jobs of every workspace it serves, which need not be the workspace's own
database: the self-hosted server keeps them in its one database, MemForge
Cloud in its service metadata. Workspace code therefore reaches jobs and
devices only through this protocol, and never in a transaction with Source
rows. A rule that relates a job to its Source is checked where both are
read: a ``document_acquisition`` job names its execution device when it is
queued, and a submission under its lease is accepted only while that device
still executes the Source (ADR 0058).

An implementation serves one workspace; ``workspace_id`` arguments name it.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from memforge.local_agent.device_report import LocalAgentDevice
from memforge.local_agent.source_contract import LocalAgentLease


@runtime_checkable
class LocalAgentJobStore(Protocol):
    async def enqueue_local_agent_job(
        self,
        *,
        job_id: str,
        source_id: str,
        source_type: str,
        operation: str,
        payload: Mapping[str, Any],
        created_by_user_id: str,
        execution_owner_user_id: str,
        workspace_id: str,
        device_id: str | None = None,
        trigger: str = "manual",
        retry_job_id: str | None = None,
        next_attempt_at: datetime | None = None,
    ) -> tuple[str, bool]:
        """Queue a sync job, or coalesce it with the Source's current one; return its id and whether it is new.

        ``device_id`` is the one device of the execution owner that may lease
        the job; the caller names a ``document_acquisition`` job's Source's
        execution device. A job coalesces only with one that names the same
        device. ``next_attempt_at`` is the earliest time a daemon may lease a
        job this call creates. Raises ``ValueError`` for a retry target that
        does not match.
        """
        ...

    async def insert_local_agent_job(
        self,
        *,
        job_id: str,
        source_id: str,
        source_type: str,
        operation: str,
        payload: Mapping[str, Any],
        created_by_user_id: str,
        execution_owner_user_id: str,
        workspace_id: str,
        device_id: str | None = None,
    ) -> bool:
        """Queue a setup job; ``device_id`` names the one device of the owner that may lease it.

        A job without a device goes to whichever of the owner's devices
        leases first.
        """
        ...

    async def get_local_agent_job(self, job_id: str) -> dict[str, Any] | None: ...

    async def list_current_local_agent_jobs(self, *, workspace_id: str, user_id: str) -> list[dict[str, Any]]:
        """The current or else latest sync job of each of the workspace's Sources the user executes."""
        ...

    async def current_local_agent_lease(
        self,
        *,
        job_id: str,
        attempt_count: int,
        source_id: str,
        lease_owner_user_id: str,
        source_config_revision: str,
    ) -> LocalAgentLease | None:
        """The attempt's lease while it is current for this Source, owner and configuration, else ``None``."""
        ...

    async def requeue_local_agent_job_uncounted(
        self,
        *,
        job_id: str,
        attempt_count: int,
        now: datetime | None = None,
    ) -> bool:
        """Queue a leased attempt's job again without counting the attempt; return whether it was still leased."""
        ...

    async def stop_local_agent_source_jobs(
        self,
        *,
        source_id: str,
        now: datetime | None = None,
    ) -> list[str]:
        """Fail the Source's queued and leased sync jobs as stopped by a person; return their ids."""
        ...

    async def leased_document_acquisition_job(self, source_id: str) -> str | None:
        """The id of a ``document_acquisition`` job of the Source a daemon holds unexpired, if any."""
        ...

    async def cancel_queued_source_jobs(
        self,
        source_id: str,
        *,
        execution_device_id: str | None,
        now: datetime | None = None,
    ) -> list[str]:
        """Fail the Source's queued jobs planned for another executor as cancelled by a Source change; return them.

        These are its queued sync jobs, and its queued setup jobs that name a
        device other than ``execution_device_id``, the Source's execution
        device, or any device when it has none.
        """
        ...

    async def list_local_agent_devices(self, user_id: str) -> list[LocalAgentDevice]:
        """Each device of the user, as its latest report left it, most recently seen first."""
        ...
