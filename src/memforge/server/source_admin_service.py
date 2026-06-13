"""Canonical admin source response assembly."""

from __future__ import annotations

from typing import Any

from memforge.storage.admin_source import SourceAdminReader


def _sync_is_running(sync_service: Any, source_id: str) -> bool:
    return bool(sync_service is not None and sync_service.is_running(source_id))


async def list_source_admin_rows(
    reader: SourceAdminReader,
    *,
    sync_service: Any,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source in await reader.list_sources():
        row = dict(source)
        source_id = str(row.get("id") or "")
        row["memory_count"] = await reader.count_source_memories(source_id)
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
