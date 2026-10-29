"""Code outside the storage package reaches a store only through its methods.

Every workspace store, SQLite in this repository and HANA in the Cloud edition,
implements the same methods. Raw SQL against one store's connection works on
that store alone, so a caller that reads ``database.db`` or a store's query
helpers would fail on every other store.
"""

from __future__ import annotations

import ast
from pathlib import Path

import memforge

STORE_NAMES = frozenset({"db", "_db", "database", "_database", "store", "_store"})
CONNECTION_ATTRIBUTES = frozenset({"db", "conn", "connection"})
QUERY_HELPERS = frozenset({"_fetch_all", "_fetch_one", "execute_fetchall"})


def _terminal_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _offenders() -> list[str]:
    package = Path(memforge.__file__).parent
    offenders = []
    for path in sorted(package.rglob("*.py")):
        relative = path.relative_to(package)
        if relative.parts[0] == "storage":
            continue
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if not isinstance(node, ast.Attribute):
                continue
            if node.attr in QUERY_HELPERS or (
                node.attr in CONNECTION_ATTRIBUTES and _terminal_name(node.value) in STORE_NAMES
            ):
                offenders.append(f"{relative}:{node.lineno}: {ast.unparse(node)}")
    return offenders


def test_no_code_outside_storage_queries_a_store_connection() -> None:
    assert _offenders() == []
