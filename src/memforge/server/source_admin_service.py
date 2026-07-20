"""Canonical admin source response assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from memforge.genes import source_type_supports_sync
from memforge.local_agent.source_contract import (
    execution_owner_user_id,
    is_local_agent_backed_source,
    source_execution_descriptor,
)
from memforge.memory.lifecycle_plan import (
    CutoverFindingStatus,
    LifecycleBackfillJob,
    LifecycleBackfillJobStatus,
    LifecycleGateState,
)
from memforge.source_access import (
    SourceAccessPolicy,
    source_access_policy,
    source_is_discoverable,
    source_owner_user_id,
)
from memforge.sync_progress import source_progress_unit, source_sync_progress_from_pipeline
from memforge.storage.admin_source import SourceAdminReader

WORKSPACE_ADMIN_ROLE = "workspace_admin"
MEMBER_ROLE = "member"
VIEWER_ROLE = "viewer"
LOCAL_WORKSPACE_ROLE = WORKSPACE_ADMIN_ROLE
MANAGED_SOURCE_TYPES = frozenset({"agent_session"})


def normalize_workspace_role(role: str | None) -> str:
    value = str(role or "").strip()
    return value if value in {WORKSPACE_ADMIN_ROLE, MEMBER_ROLE, VIEWER_ROLE} else MEMBER_ROLE


def _is_managed_source_type(source_type: str) -> bool:
    return source_type in MANAGED_SOURCE_TYPES


def source_viewer_relationship(
    source: dict[str, Any], *, viewer_id: str, viewer_role: str
) -> str:
    if source_owner_user_id(source) == viewer_id:
        return "owner"
    if viewer_role == WORKSPACE_ADMIN_ROLE:
        return WORKSPACE_ADMIN_ROLE
    return normalize_workspace_role(viewer_role)


def can_manage_source(
    source: dict[str, Any], *, viewer_id: str, viewer_role: str
) -> bool:
    if not source_is_discoverable(source, viewer_id=viewer_id):
        return False
    if source_owner_user_id(source) == viewer_id:
        return True
    return (
        source_access_policy(source) is SourceAccessPolicy.WORKSPACE
        and viewer_role == WORKSPACE_ADMIN_ROLE
    )


def source_ownership_and_capabilities(
    source: dict[str, Any], *, viewer_id: str, viewer_role: str
) -> tuple[dict[str, Any], dict[str, bool]]:
    relationship = source_viewer_relationship(
        source, viewer_id=viewer_id, viewer_role=viewer_role
    )
    can_manage = can_manage_source(source, viewer_id=viewer_id, viewer_role=viewer_role)
    managed_source = _is_managed_source_type(str(source.get("type") or ""))
    sync_supported = source_type_supports_sync(str(source.get("type") or ""))
    local_agent_backed = is_local_agent_backed_source(source)
    execution_owner = execution_owner_user_id(source)
    can_execute_locally = execution_owner is not None and execution_owner == viewer_id
    ownership = {
        "created_by_user_id": source.get("created_by_user_id"),
        "owner_user_id": source_owner_user_id(source),
        "execution_owner_user_id": execution_owner,
        "viewer_role": viewer_role,
        "viewer_relationship": relationship,
    }
    capabilities = {
        "can_subscribe": True,
        "can_configure": can_manage and not managed_source,
        "can_configure_connection": (
            can_execute_locally if local_agent_backed else can_manage and not managed_source
        ),
        "can_sync": (
            sync_supported
            and (
                can_execute_locally
                if local_agent_backed
                else can_manage and not managed_source
            )
        ),
        "can_force_resync": (
            sync_supported
            and (
                can_execute_locally
                if local_agent_backed
                else can_manage and not managed_source
            )
        ),
        "can_delete": can_manage and not managed_source,
        "can_change_access": can_manage,
    }
    return ownership, capabilities


def _sync_is_running(sync_service: Any, source_id: str) -> bool:
    return bool(sync_service is not None and sync_service.is_running(source_id))


def _durable_sync_payload(run: Any) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    status = str(run.status)
    if status == "running" and (
        run.lease_expires_at is None or run.lease_expires_at <= now
    ):
        status = "recovering"
    elif status == "pending" and run.next_attempt_at is not None:
        status = "recovering"
    return {
        "run_id": run.run_id,
        "status": status,
        "trigger": run.trigger,
        "force_full_sync": run.force_full_sync,
        "started_at": run.started_at.isoformat() if run.started_at else None,
        "finished_at": run.completed_at.isoformat() if run.completed_at else None,
        "next_attempt_at": (
            run.next_attempt_at.isoformat() if run.next_attempt_at else None
        ),
        "recovery_count": run.recovery_count,
        "error_message": run.error_message,
        "progress": run.progress,
        "progress_revision": run.progress_revision,
        "progress_updated_at": (
            run.progress_updated_at.isoformat() if run.progress_updated_at else None
        ),
    }


def _lifecycle_maintenance_payload(job: LifecycleBackfillJob) -> dict[str, Any]:
    return {
        "status": job.status.value,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.completed_at,
    }


async def _current_lifecycle_maintenance_payload(
    reader: SourceAdminReader,
    *,
    source_id: str,
    latest_job: LifecycleBackfillJob | None,
) -> dict[str, Any] | None:
    if latest_job is None:
        return None
    if latest_job.status is not LifecycleBackfillJobStatus.FAILED:
        return _lifecycle_maintenance_payload(latest_job)

    gate = await reader.get_lifecycle_gate(source_id)
    if gate.state is not LifecycleGateState.ENABLED:
        return _lifecycle_maintenance_payload(latest_job)
    open_findings = await reader.list_lifecycle_cutover_findings(
        source_id,
        status=CutoverFindingStatus.OPEN,
    )
    if open_findings:
        return _lifecycle_maintenance_payload(latest_job)
    vector_tasks = await reader.list_lifecycle_vector_tasks(
        source_id=source_id,
        limit=1,
    )
    if vector_tasks:
        return _lifecycle_maintenance_payload(latest_job)
    return None


async def list_source_admin_rows(
    reader: SourceAdminReader,
    *,
    sync_service: Any,
    viewer_id: str,
    viewer_role: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    viewer_role = normalize_workspace_role(viewer_role)
    for source in await reader.list_sources():
        if not source_is_discoverable(source, viewer_id=viewer_id):
            continue
        row = dict(source)
        source_id = str(row.get("id") or "")
        ownership, capabilities = source_ownership_and_capabilities(
            row, viewer_id=viewer_id, viewer_role=viewer_role
        )
        row["ownership"] = ownership
        row["capabilities"] = capabilities
        row["execution"] = source_execution_descriptor(
            str(row.get("type") or ""),
            row.get("config"),
        )
        enabled_for_me = await reader.is_source_enabled_for_user(source_id, viewer_id)
        row["subscription"] = {"enabled": enabled_for_me}
        row["enabled_for_me"] = enabled_for_me
        row["pinned_for_me"] = await reader.is_source_pinned_for_user(
            source_id, viewer_id
        )
        row["memory_count"] = await reader.count_source_memories(
            source_id,
            include_private=True,
            owner_user_id=viewer_id,
        )
        row["doc_count"] = await reader.count_documents(source=source_id)
        row["access_transition"] = await reader.get_active_source_access_transition(
            source_id
        )
        lifecycle_jobs = await reader.list_lifecycle_backfill_jobs(
            source_id,
            limit=1,
        )
        row["lifecycle_maintenance"] = await _current_lifecycle_maintenance_payload(
            reader,
            source_id=source_id,
            latest_job=lifecycle_jobs[0] if lifecycle_jobs else None,
        )
        row.setdefault("client", None)
        if not source_type_supports_sync(str(row.get("type") or "")):
            row["sync"] = None
            rows.append(row)
            continue
        durable_run = await reader.get_latest_source_sync_run(source_id=source_id)
        if durable_run is not None and durable_run.status in {"pending", "running"}:
            row["sync"] = _durable_sync_payload(durable_run)
        elif _sync_is_running(sync_service, source_id):
            progress = sync_service.progress.get(source_id, {})
            progress_snapshot = source_sync_progress_from_pipeline(
                {
                    "phase": progress.get("phase"),
                    "current": progress.get("docs_processed", 0),
                    "total": progress.get("docs_total", 0),
                    "docs_updated": progress.get("docs_updated", 0),
                    "docs_failed": progress.get("docs_failed", 0),
                    "memories_extracted": progress.get("memories_extracted", 0),
                },
                source_type=str(row.get("type") or ""),
            )
            row["sync"] = {
                "status": "running",
                "phase": progress.get("phase"),
                "started_at": progress.get("started_at"),
                "finished_at": None,
                "docs_processed": progress.get("docs_processed", 0),
                "docs_total": progress.get("docs_total", 0),
                "docs_updated": progress.get("docs_updated", 0),
                "docs_failed": progress.get("docs_failed", 0),
                "memories_extracted": progress.get("memories_extracted", 0),
                "docs_stored": row["doc_count"],
                "memories_stored": row["memory_count"],
                "current_title": progress.get("title"),
                "error_message": None,
                "progress": progress_snapshot,
            }
        else:
            history = await reader.get_sync_history(source=source_id, limit=1)
            if history:
                latest = history[0]
                row["sync"] = {
                    "status": latest.get("status", "success"),
                    "started_at": latest.get("started_at"),
                    "finished_at": latest.get("finished_at"),
                    "docs_processed": latest.get("docs_processed", 0),
                    "docs_updated": latest.get("docs_updated", 0),
                    "docs_failed": latest.get("docs_failed", 0),
                    "memories_extracted": latest.get("memories_extracted", 0),
                    "error_message": latest.get("error_message"),
                    "failed_docs": latest.get("failed_docs", []),
                    "progress": {
                        "schema_version": 1,
                        "phase": "processing",
                        "progress": {
                            "completed": latest.get("docs_processed", 0),
                            "unit": source_progress_unit(str(row.get("type") or "")),
                        },
                        "counts": {
                            "changed": latest.get("docs_updated", 0),
                            "failed": latest.get("docs_failed", 0),
                            "memories_created": latest.get("memories_extracted", 0),
                        },
                    },
                }
            elif durable_run is not None:
                row["sync"] = _durable_sync_payload(durable_run)
            else:
                row["sync"] = None
        rows.append(row)
    return rows
