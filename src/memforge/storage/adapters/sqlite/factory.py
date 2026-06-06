"""One construction point for the SQLite adapters bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from memforge.storage.database import Database
from memforge.storage.adapters.sqlite.keyword import SqliteKeywordSearch
from memforge.storage.adapters.sqlite.relational import SqliteRelationalStore
from memforge.storage.adapters.sqlite.vector import SqliteVectorStore

if TYPE_CHECKING:
    from memforge.memory.audit import MemoryAuditLogger

__all__ = ["SqliteAdapters", "build_sqlite_adapters"]


@dataclass(frozen=True)
class SqliteAdapters:
    """The three adapter handles bound to one datastore."""

    relational: SqliteRelationalStore
    keyword: SqliteKeywordSearch
    vector: SqliteVectorStore


def build_sqlite_adapters(
    db: Database,
    memory_collection: Any,
    *,
    audit_logger: "MemoryAuditLogger | None" = None,
) -> SqliteAdapters:
    """Bind a Database plus its memories Chroma collection into the adapters.

    The optional audit logger is threaded into the relational store so the
    promote-to-workspace guard can record attempts without the row channel
    constructing its own logger.
    """
    return SqliteAdapters(
        relational=SqliteRelationalStore(db, audit_logger=audit_logger),
        keyword=SqliteKeywordSearch(db),
        vector=SqliteVectorStore(memory_collection),
    )
