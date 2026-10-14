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
    """Bind a Database and a Memory vector index into the adapters.

    ``memory_collection`` is an open Chroma collection, or the
    ``SqliteVectorStore`` to use as it is.

    The optional audit logger is threaded into the relational store so the
    promote-to-workspace guard can record attempts without the row channel
    constructing its own logger.
    """
    vector = (
        memory_collection
        if isinstance(memory_collection, SqliteVectorStore)
        else SqliteVectorStore(memory_collection)
    )
    return SqliteAdapters(
        relational=SqliteRelationalStore(db, audit_logger=audit_logger),
        keyword=SqliteKeywordSearch(db),
        vector=vector,
    )
