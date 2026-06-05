"""One construction point for the SQLite adapters bundle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memforge.storage.database import Database
from memforge.storage.adapters.sqlite.keyword import SqliteKeywordSearch
from memforge.storage.adapters.sqlite.relational import SqliteRelationalStore
from memforge.storage.adapters.sqlite.vector import SqliteVectorStore

__all__ = ["SqliteAdapters", "build_sqlite_adapters"]


@dataclass(frozen=True)
class SqliteAdapters:
    """The three adapter handles bound to one datastore."""

    relational: SqliteRelationalStore
    keyword: SqliteKeywordSearch
    vector: SqliteVectorStore


def build_sqlite_adapters(db: Database, memory_collection: Any) -> SqliteAdapters:
    """Bind a Database plus its memories Chroma collection into the adapters."""
    return SqliteAdapters(
        relational=SqliteRelationalStore(db),
        keyword=SqliteKeywordSearch(db),
        vector=SqliteVectorStore(memory_collection),
    )
