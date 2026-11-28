"""The routes a MemForge daemon calls on its server (ADR 0029, ADR 0058).

A daemon serves its user across every workspace that user's jobs come from:
it leases, renews and ends their jobs, reports its device, and reads which
Sources each of the user's devices executes. The routes are answered once,
in front of the workspaces, and read only the app's
:class:`~memforge.local_agent.broker.LocalAgentBroker`
(``app.state.local_agent_broker``) and
:class:`~memforge.local_agent.broker.DaemonWorkspaces`
(``app.state.daemon_workspaces``), and the caller through
``app.state.principal_resolver``. The self-hosted server answers them from
its one database; MemForge Cloud mounts the same router over its service
metadata and its workspaces' databases.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from memforge.release import installed_version
from memforge.local_agent.broker import DaemonWorkspaces, LocalAgentBroker
from memforge.local_agent.device_lease import EXECUTED_SOURCES_FIELD
from memforge.local_agent.device_report import (
    DEVICE_ID_MAX_CHARS,
    DeviceReport,
    declared_acquisition_contracts,
)
from memforge.local_agent.document_acquisition import acquisition_job_config
from memforge.local_agent.readiness import LOCAL_AGENT_DEVICE_STALE_SECONDS
from memforge.local_agent.source_contract import (
    LOCAL_AGENT_SYNC_OPERATIONS,
    SourceSyncRunReceiptError,
    source_sync_run_id_from_completion,
)
from memforge.server.principal import resolve_request_principal
from memforge.sync_progress import normalize_sync_progress_snapshot

LOCAL_AGENT_HOST_PREFIX = "/api/cloud/local-agent"
DEVICE_REPORT_REQUIRED = "device_report_required"

LocalAgentJobStatus = Literal["queued", "leased", "succeeded", "failed"]


class LocalAgentJobLeaseRequest(BaseModel):
    limit: int = Field(default=5, ge=1, le=20)
    lease_seconds: int = Field(default=60, ge=1, le=3600)
    wait_seconds: int = Field(default=0, ge=0, le=25)
    device: dict[str, Any] | None = None


class LocalAgentJobHeartbeatRequest(BaseModel):
    attempt_count: int = Field(ge=1)
    lease_seconds: int = Field(default=60, ge=1, le=3600)
    progress: dict[str, Any] | None = None
    device: dict[str, Any] | None = None


class LocalAgentJobCompleteRequest(BaseModel):
    status: Literal["succeeded", "failed"]
    attempt_count: int = Field(ge=1)
    device_id: str = Field(
        min_length=1,
        max_length=DEVICE_ID_MAX_CHARS,
        description="The device that holds the attempt; only the device that leased it completes it.",
    )
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(default=None, max_length=2000)


class LocalAgentJobResponse(BaseModel):
    job_id: str
    workspace_id: str
    source_id: str
    source_type: str
    operation: str
    status: LocalAgentJobStatus
    payload: dict[str, Any]
    execution_owner_user_id: str
    result: dict[str, Any] = Field(
        description=(
            "The daemon's report: sync progress while the job runs, then its completion result. "
            "Empty before the first report."
        ),
    )
    last_error: str | None
    next_attempt_at: str | None
    leased_until: str | None
    attempt_count: int
    created_at: str
    updated_at: str
    finished_at: str | None


class LocalAgentDeviceStatus(BaseModel):
    device_id: str
    name: str
    version: str
    acquisition_contracts: dict[str, int] = Field(
        description=(
            "The acquisition contract of each document type, as the device's latest report declares it. A device "
            "acquires a type only while its contract equals the server's."
        ),
    )
    status: Literal["online", "offline"]
    last_seen_at: str


class LocalAgentDeviceSource(BaseModel):
    """A Source whose acquisition runs on one of the caller's devices, with its configuration without secrets."""

    id: str
    name: str
    type: str
    config: dict[str, Any]
    execution_device_id: str | None = Field(
        description="The caller's device that executes the Source; null while the caller has chosen none.",
    )


class LocalAgentDeviceRemovedResponse(BaseModel):
    """The removed device and the Sources that no longer have an execution device."""

    device_id: str
    source_ids: list[str]


class LocalAgentServerContracts(BaseModel):
    """The server's MemForge version and the acquisition contract of each document type."""

    version: str
    acquisition_contracts: dict[str, int]


class LocalAgentDaemonStatusResponse(BaseModel):
    """The caller's devices, whether any of them is online, the caller's device Sources and the server's contracts."""

    status: Literal["online", "offline"]
    last_seen_at: str | None
    devices: list[LocalAgentDeviceStatus]
    sources: list[LocalAgentDeviceSource]
    server: LocalAgentServerContracts
    checked_at: str
    stale_after_seconds: int


def shape_local_agent_job(job: Mapping[str, Any]) -> dict[str, Any]:
    """A job as its routes answer it."""

    return {
        "job_id": job["job_id"],
        "workspace_id": job["workspace_id"],
        "source_id": job["source_id"],
        "source_type": job["source_type"],
        "operation": job["operation"],
        "status": job["status"],
        "payload": job.get("payload") or {},
        "execution_owner_user_id": job["execution_owner_user_id"],
        "result": job.get("result") or {},
        "last_error": job.get("last_error"),
        "next_attempt_at": job.get("next_attempt_at"),
        "leased_until": job.get("leased_until"),
        "attempt_count": int(job.get("attempt_count") or 0),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "finished_at": job.get("finished_at"),
    }


def _device_report(payload: Mapping[str, Any] | None) -> DeviceReport:
    """The device report a lease or job heartbeat carries; 422 when it is missing or malformed.

    A daemon that sends none predates the device report and takes no job
    until its device installs the server's MemForge version, which the
    answer names.
    """

    if payload is None:
        raise HTTPException(
            status_code=422,
            detail={
                "error": DEVICE_REPORT_REQUIRED,
                "message": (
                    "This MemForge daemon sends no device report, so it takes no jobs. "
                    f"Install MemForge {installed_version()} on this device and restart its daemon."
                ),
            },
        )
    try:
        return DeviceReport.from_payload(payload)
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _device_source_entry(source: Mapping[str, Any]) -> dict[str, Any]:
    """A device Source as a daemon reads it: its type, its configuration without secrets, and its execution device."""

    return {
        "id": source["id"],
        "type": source["type"],
        "config": acquisition_job_config(str(source["type"]), source.get("config") or {}),
        "execution_device_id": source.get("execution_device_id"),
    }


def _require_sync_run_receipt(job: Mapping[str, Any], req: LocalAgentJobCompleteRequest) -> None:
    """A leased sync job succeeds naming the SourceSyncRun it started, and a succeeded one only that run; 409 otherwise.

    A dry run starts no run, and an attempt that already ended otherwise
    answers its job's status.
    """

    if req.status != "succeeded" or job["operation"] not in LOCAL_AGENT_SYNC_OPERATIONS:
        return
    if (job.get("payload") or {}).get("dry_run") or job["status"] not in {"leased", "succeeded"}:
        return
    try:
        run_id = source_sync_run_id_from_completion(req.result)
        if job["status"] == "succeeded" and source_sync_run_id_from_completion(job.get("result")) != run_id:
            raise SourceSyncRunReceiptError("a completed sync job names another source_sync_run_id")
    except SourceSyncRunReceiptError as exc:
        raise HTTPException(status_code=409, detail="local_agent_source_sync_run_receipt_invalid") from exc


def _broker(request: Request) -> LocalAgentBroker:
    return request.app.state.local_agent_broker


def _workspaces(request: Request) -> DaemonWorkspaces:
    return request.app.state.daemon_workspaces


async def _device_sources(request: Request, owner: str, *, device_id: str | None = None) -> list[dict[str, Any]]:
    """The owner's device Sources in every workspace the owner's daemon serves."""

    directories = (await _workspaces(request).served_workspaces(owner)).values()
    found = await asyncio.gather(
        *(directory.list_device_sources(owner, device_id=device_id) for directory in directories)
    )
    return [source for sources in found for source in sources]


def build_local_agent_host_router() -> APIRouter:
    router = APIRouter(prefix=LOCAL_AGENT_HOST_PREFIX, tags=["local-agent"])

    @router.post("/jobs/lease")
    async def lease_local_agent_jobs(req: LocalAgentJobLeaseRequest, request: Request):
        """Lease the device its due jobs.

        The answer also names the server's version and acquisition contracts,
        and the Sources this device executes, each with its type and its
        configuration without secrets. The device reports its readiness for
        each of them, and deletes the credentials it holds for any other
        Source: one deleted, moved to the server or moved to another device.
        """

        requester = resolve_request_principal(request)
        device = _device_report(req.device)
        contracts = declared_acquisition_contracts()
        answer: dict[str, Any] = {
            "jobs": [],
            "server": {"version": installed_version(), "acquisition_contracts": contracts},
            EXECUTED_SOURCES_FIELD: [
                _device_source_entry(source)
                for source in await _device_sources(request, requester, device_id=device.device_id)
            ],
        }
        broker = _broker(request)
        deadline = time.monotonic() + req.wait_seconds
        while True:
            jobs = await broker.lease_local_agent_jobs(
                user_id=requester,
                device=device,
                acquisition_contracts=contracts,
                limit=req.limit,
                lease_seconds=req.lease_seconds,
            )
            if jobs or req.wait_seconds <= 0 or time.monotonic() >= deadline:
                return {**answer, "jobs": [shape_local_agent_job(job) for job in jobs]}
            if await request.is_disconnected():
                return answer
            await asyncio.sleep(min(0.5, max(deadline - time.monotonic(), 0)))

    @router.post("/jobs/{job_id}/heartbeat")
    async def heartbeat_local_agent_job(job_id: str, req: LocalAgentJobHeartbeatRequest, request: Request):
        try:
            progress = normalize_sync_progress_snapshot(req.progress) if req.progress is not None else None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        wrote = await _broker(request).heartbeat_local_agent_job(
            job_id=job_id,
            user_id=resolve_request_principal(request),
            attempt_count=req.attempt_count,
            lease_seconds=req.lease_seconds,
            device=_device_report(req.device),
            progress=progress,
        )
        if not wrote:
            raise HTTPException(status_code=404, detail="local_agent_job_not_found")
        return {"ok": True, "job_id": job_id, "status": "leased"}

    @router.post("/jobs/{job_id}/complete")
    async def complete_local_agent_job(job_id: str, req: LocalAgentJobCompleteRequest, request: Request):
        """End the device's attempt; completing an attempt that already ended answers the job's status."""

        requester = resolve_request_principal(request)
        broker = _broker(request)
        job = await broker.get_local_agent_job(job_id)
        if job is None or job["execution_owner_user_id"] != requester:
            raise HTTPException(status_code=404, detail="local_agent_job_not_found")
        if int(job.get("attempt_count") or 0) != req.attempt_count:
            raise HTTPException(status_code=404, detail="local_agent_job_not_found")
        _require_sync_run_receipt(job, req)
        wrote = await broker.complete_local_agent_job(
            job_id=job_id,
            user_id=requester,
            device_id=req.device_id,
            attempt_count=req.attempt_count,
            status=req.status,
            result=req.result,
            error=req.error,
            retryable=bool(req.result.get("retryable")),
        )
        current = await broker.get_local_agent_job(job_id)
        if current is None:
            raise HTTPException(status_code=404, detail="local_agent_job_not_found")
        status = str(current["status"])
        if not wrote and (
            int(current.get("attempt_count") or 0) != req.attempt_count
            or status not in {"queued", "succeeded", "failed"}
        ):
            raise HTTPException(status_code=404, detail="local_agent_job_not_found")
        return {"ok": True, "job_id": job_id, "status": status}

    @router.get("/jobs/{job_id}", response_model=LocalAgentJobResponse)
    async def read_local_agent_job(job_id: str, request: Request):
        """A job the caller created or executes, in a workspace the caller's daemon still serves."""

        requester = resolve_request_principal(request)
        job = await _broker(request).get_local_agent_job(job_id)
        if (
            job is None
            or requester not in {job["created_by_user_id"], job["execution_owner_user_id"]}
            or job["workspace_id"] not in await _workspaces(request).served_workspaces(requester)
        ):
            raise HTTPException(status_code=404, detail="local_agent_job_not_found")
        return shape_local_agent_job(job)

    @router.get("/status", response_model=LocalAgentDaemonStatusResponse)
    async def read_local_agent_status(request: Request):
        """Each of the caller's devices by its latest report, online while that report is recent.

        It also names the caller's device Sources with the device that
        executes each, and the server's version and acquisition contracts,
        from which a device tells, per Source it executes, whether it is
        ready, waits for a credential or a folder, or waits for an update.
        """

        requester = resolve_request_principal(request)
        now = datetime.now(timezone.utc)
        devices = [
            {
                "device_id": device.report.device_id,
                "name": device.report.name,
                "version": device.report.version,
                "acquisition_contracts": dict(sorted(device.report.acquisition_contracts.items())),
                "status": (
                    "online"
                    if (now - device.last_seen_at).total_seconds() <= LOCAL_AGENT_DEVICE_STALE_SECONDS
                    else "offline"
                ),
                "last_seen_at": device.last_seen_at.isoformat(),
            }
            for device in await _broker(request).list_local_agent_devices(requester)
        ]
        return {
            "status": "online" if any(device["status"] == "online" for device in devices) else "offline",
            "last_seen_at": devices[0]["last_seen_at"] if devices else None,
            "devices": devices,
            "sources": [
                {**_device_source_entry(source), "name": source["name"]}
                for source in await _device_sources(request, requester)
            ],
            "server": {"version": installed_version(), "acquisition_contracts": declared_acquisition_contracts()},
            "checked_at": now.isoformat(),
            "stale_after_seconds": LOCAL_AGENT_DEVICE_STALE_SECONDS,
        }

    @router.delete("/devices/{device_id}", response_model=LocalAgentDeviceRemovedResponse)
    async def remove_local_agent_device(device_id: str, request: Request):
        """Forget one of the caller's devices.

        Each Source it executed, in every workspace the caller's daemon
        serves, loses its execution device and shows Finish setup until its
        owner chooses another device in Where it runs. A Source of a
        workspace the caller has left keeps naming the device, and no job of
        it is leased, since a daemon serves only its user's workspaces. The
        device's queued jobs are cancelled. A daemon that still runs on the
        device appears again with its next report, without its Sources.
        """

        requester = resolve_request_principal(request)
        broker = _broker(request)
        if not any(device.report.device_id == device_id for device in await broker.list_local_agent_devices(requester)):
            raise HTTPException(status_code=404, detail="local_agent_device_not_found")
        # The Sources go first: once none names the device, no new job does.
        source_ids: list[str] = []
        for directory in (await _workspaces(request).served_workspaces(requester)).values():
            source_ids.extend(await directory.release_execution_device(requester, device_id))
        try:
            await broker.remove_local_agent_device(requester, device_id)
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="local_agent_device_not_found") from exc
        return {"device_id": device_id, "source_ids": source_ids}

    return router
