"""What the routes a daemon calls read and write (ADR 0029, ADR 0058).

A MemForge daemon serves one user across every workspace that user's jobs
come from. Its routes reach the jobs and devices through a
:class:`LocalAgentBroker`, and each workspace's device Sources through
:class:`DaemonWorkspaces`. The self-hosted server answers both from its one
database; MemForge Cloud keeps jobs and devices in its service metadata and
Sources in each workspace's database.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from memforge.local_agent.device_report import DeviceReport, LocalAgentDevice


class LocalAgentBroker(Protocol):
    """The jobs and devices of every workspace a daemon serves.

    A daemon serves the workspaces its user belongs to, the ones
    :class:`DaemonWorkspaces` names; the broker leases only their jobs.
    """

    async def lease_local_agent_jobs(
        self,
        *,
        user_id: str,
        device: DeviceReport,
        acquisition_contracts: Mapping[str, int],
        limit: int,
        lease_seconds: int,
    ) -> list[dict[str, Any]]:
        """Keep the device's report and lease it up to ``limit`` of its user's due jobs.

        A job that names a device is leased only by that device. A
        ``document_acquisition`` job is leased only while the device's report
        names its Source ready and the device's acquisition contract for its
        type equals ``acquisition_contracts``, the server's; the attempt
        records that contract and the device that holds it. A store that
        cannot lease now, as when a concurrent lease wins, leases nothing; the
        daemon asks again.
        """
        ...

    async def heartbeat_local_agent_job(
        self,
        *,
        job_id: str,
        user_id: str,
        attempt_count: int,
        lease_seconds: int,
        device: DeviceReport,
        progress: Mapping[str, Any] | None = None,
    ) -> bool:
        """Keep the device's report and extend the attempt's lease while this device still holds it."""
        ...

    async def complete_local_agent_job(
        self,
        *,
        job_id: str,
        user_id: str,
        device_id: str,
        attempt_count: int,
        status: str,
        result: Mapping[str, Any],
        error: str | None,
        retryable: bool = False,
    ) -> bool:
        """End the attempt ``device_id`` holds; ``False`` when that device no longer holds it."""
        ...

    async def get_local_agent_job(self, job_id: str) -> dict[str, Any] | None: ...

    async def list_local_agent_devices(self, user_id: str) -> list[LocalAgentDevice]:
        """Each device of the user, as its latest report left it, most recently seen first."""
        ...

    async def remove_local_agent_device(self, user_id: str, device_id: str) -> list[str]:
        """Forget one of the user's devices and fail its jobs as cancelled; return those jobs.

        These are the queued jobs that name it and the attempts it holds.
        Raises ``LookupError`` when the user has no such device.
        """
        ...


class DeviceSourceDirectory(Protocol):
    """One workspace's Sources whose acquisition runs on a device."""

    async def list_device_sources(
        self, execution_owner_user_id: str, *, device_id: str | None = None
    ) -> list[dict[str, Any]]:
        """The workspace's Sources whose acquisition runs on a device of this execution owner, by id.

        With ``device_id``, only the Sources that device executes.
        """
        ...

    async def release_execution_device(self, execution_owner_user_id: str, device_id: str) -> list[str]:
        """Clear a removed device as the execution device of the owner's Sources; return those Sources."""
        ...


class DaemonWorkspaces(Protocol):
    """The workspaces whose jobs a user's daemon takes: those the user belongs to."""

    async def served_workspaces(self, user_id: str) -> Mapping[str, DeviceSourceDirectory]:
        """Each workspace whose jobs the user's daemon takes, by id, with its device Sources."""
        ...


class OneWorkspace:
    """The one workspace a self-hosted server serves its daemons."""

    def __init__(self, workspace_id: str, sources: DeviceSourceDirectory) -> None:
        self._workspaces = {workspace_id: sources}

    async def served_workspaces(self, user_id: str) -> Mapping[str, DeviceSourceDirectory]:
        return self._workspaces
