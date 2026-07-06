"""Seed deterministic SQLite corpora from retrieval golden fixture manifests."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from memforge.models import DocumentRecord, Memory, content_hash
from memforge.storage.database import Database


FIXED_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


async def seed_sqlite_fixture(
    *,
    db_path: Path,
    fixture: Mapping[str, Any],
) -> Database:
    """Create a SQLite database and seed one golden fixture."""

    db = Database(str(db_path))
    await db.connect()
    try:
        await _seed_sources(db, fixture)
        await _seed_documents(db, fixture)
        await _seed_memories(db, fixture)
        await _seed_memory_sources(db, fixture)
        await _seed_source_subscriptions(db, fixture)
    except Exception:
        await db.close()
        raise
    return db


async def _seed_sources(db: Database, fixture: Mapping[str, Any]) -> None:
    for source in fixture.get("sources") or ():
        await db.upsert_source(
            str(source["id"]),
            str(_value_or_default(source, "type", "generic")),
            str(_value_or_default(source, "name", source["id"])),
            json.dumps(_value_or_default(source, "config", {})),
            status=str(_value_or_default(source, "status", "active")),
        )


async def _seed_documents(db: Database, fixture: Mapping[str, Any]) -> None:
    for document in fixture.get("documents") or ():
        doc_id = str(document["doc_id"])
        await db.upsert_document(
            DocumentRecord(
                doc_id=doc_id,
                source=str(document["source_id"]),
                source_url=str(_value_or_default(document, "source_url", f"https://eval.example/{doc_id}")),
                title=str(_value_or_default(document, "title", doc_id)),
                space_or_project=str(_value_or_default(document, "space_or_project", "")),
                author=str(_value_or_default(document, "author", "eval")),
                last_modified=FIXED_NOW,
                labels=list(_value_or_default(document, "labels", ())),
                version=str(_value_or_default(document, "version", "1")),
                content_hash=str(_value_or_default(document, "content_hash", f"h-{doc_id}")),
                token_count=int(_value_or_default(document, "token_count", 1)),
                raw_content_uri=None,
                raw_content_type="text/plain",
                normalized_content_uri=None,
                pdf_content_uri=None,
                last_synced=FIXED_NOW,
                client=document.get("client"),
            )
        )


async def _seed_memories(db: Database, fixture: Mapping[str, Any]) -> None:
    for memory in fixture.get("memories") or ():
        memory_id = str(memory["id"]) if isinstance(memory, Mapping) else str(memory)
        content = str(
            memory.get("content")
            if isinstance(memory, Mapping) and memory.get("content") is not None
            else f"Evaluation memory {memory_id}."
        )
        await db.insert_memory(
            Memory(
                id=memory_id,
                memory_type=str(_value_or_default(memory, "memory_type", "fact")) if isinstance(memory, Mapping) else "fact",
                content=content,
                content_hash=content_hash(content),
                confidence=float(_value_or_default(memory, "confidence", 0.9)) if isinstance(memory, Mapping) else 0.9,
                visibility=str(_value_or_default(memory, "visibility", "workspace")) if isinstance(memory, Mapping) else "workspace",
                owner_user_id=_value_or_default(memory, "owner_user_id", None) if isinstance(memory, Mapping) else None,
                project_key=memory.get("project_key") if isinstance(memory, Mapping) else None,
                repo_identifier=_value_or_default(memory, "repo_identifier", None) if isinstance(memory, Mapping) else None,
                entity_refs=list(_value_or_default(memory, "entity_refs", ())) if isinstance(memory, Mapping) else [],
                tags=list(_value_or_default(memory, "tags", ())) if isinstance(memory, Mapping) else [],
                status=str(_value_or_default(memory, "status", "active")) if isinstance(memory, Mapping) else "active",
                created_at=FIXED_NOW,
                updated_at=FIXED_NOW,
            )
        )


async def _seed_memory_sources(db: Database, fixture: Mapping[str, Any]) -> None:
    for support in fixture.get("memory_sources") or ():
        await db.add_memory_source(
            str(support["memory_id"]),
            str(support["doc_id"]),
            str(_value_or_default(support, "source_type", "generic")),
            support.get("excerpt"),
            support_kind=str(_value_or_default(support, "support_kind", "extracted")),
            source_updated_at=FIXED_NOW,
        )


async def _seed_source_subscriptions(db: Database, fixture: Mapping[str, Any]) -> None:
    for row in fixture.get("source_subscriptions") or ():
        await db.set_source_subscription(
            str(row["source_id"]),
            str(row["user_id"]),
            bool(row["enabled"]),
        )


def _value_or_default(data: Mapping[str, Any], key: str, default: Any) -> Any:
    value = data.get(key, default)
    return default if value is None else value
