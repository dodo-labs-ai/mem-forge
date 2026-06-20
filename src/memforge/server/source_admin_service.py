"""Canonical admin source response assembly."""

from __future__ import annotations

from typing import Any

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
    if str(source.get("created_by_user_id") or "") == viewer_id:
        return "creator"
    if viewer_role == WORKSPACE_ADMIN_ROLE:
        return WORKSPACE_ADMIN_ROLE
    return normalize_workspace_role(viewer_role)


def can_manage_source(
    source: dict[str, Any], *, viewer_id: str, viewer_role: str
) -> bool:
    if _is_managed_source_type(str(source.get("type") or "")):
        return False
    if viewer_role == WORKSPACE_ADMIN_ROLE:
        return True
    created_by = source.get("created_by_user_id")
    if not created_by:
        return False
    return viewer_role == MEMBER_ROLE and str(created_by) == viewer_id


def source_ownership_and_capabilities(
    source: dict[str, Any], *, viewer_id: str, viewer_role: str
) -> tuple[dict[str, Any], dict[str, bool]]:
    relationship = source_viewer_relationship(
        source, viewer_id=viewer_id, viewer_role=viewer_role
    )
    can_manage = can_manage_source(source, viewer_id=viewer_id, viewer_role=viewer_role)
    ownership = {
        "created_by_user_id": source.get("created_by_user_id"),
        "viewer_role": viewer_role,
        "viewer_relationship": relationship,
    }
    capabilities = {
        "can_subscribe": True,
        "can_configure": can_manage,
        "can_sync": can_manage,
        "can_force_resync": can_manage,
        "can_delete": can_manage,
    }
    return ownership, capabilities


def _sync_is_running(sync_service: Any, source_id: str) -> bool:
    return bool(sync_service is not None and sync_service.is_running(source_id))


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
        row = dict(source)
        source_id = str(row.get("id") or "")
        ownership, capabilities = source_ownership_and_capabilities(
            row, viewer_id=viewer_id, viewer_role=viewer_role
        )
        row["ownership"] = ownership
        row["capabilities"] = capabilities
        enabled_for_me = await reader.is_source_enabled_for_user(source_id, viewer_id)
        row["subscription"] = {"enabled": enabled_for_me}
        row["enabled_for_me"] = enabled_for_me
        row["memory_count"] = await reader.count_source_memories(
            source_id,
            include_private=True,
            owner_user_id=viewer_id,
        )
        row["doc_count"] = await reader.count_documents(source=source_id)
        row.setdefault("client", None)
        if _sync_is_running(sync_service, source_id):
            progress = sync_service.progress.get(source_id, {})
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
                }
            else:
                row["sync"] = None
        rows.append(row)
    return rows
