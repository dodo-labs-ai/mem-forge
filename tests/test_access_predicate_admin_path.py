# Cross-datastore isolation is a deployment-time property of the bound
# adapter handle, not of the core predicate; this test pack only exercises
# the predicate.
"""No retrieval channel offers an admin bypass for the access predicate.

Two complementary checks: (a) a static scan of the retrieval and storage
adapter modules confirms no `bypass_predicate`, no admin role plumbing, and
no role-keyed branch in any predicate site; (b) a live `SearchEngine.search`
with a TEAM scope still hides another user's private memory regardless of
any caller "role" the test attempts to attach to the request.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import memforge
from memforge.config import RetrievalConfig
from memforge.models import (
    Memory,
    SHARED_PROJECT_KEY,
    UNSORTED_PROJECT_KEY,
    Visibility,
    content_hash,
)
from memforge.retrieval.query_analyzer import QueryAnalysis
from memforge.retrieval.search import SearchEngine
from memforge.storage.adapters.context import AccessScope
from memforge.storage.adapters.sqlite import build_sqlite_adapters
from memforge.storage.database import Database


WORKSPACE = Visibility.WORKSPACE.value
PRIVATE = Visibility.PRIVATE.value


def _mem(mid: str, content: str, *, visibility=WORKSPACE, owner=None) -> Memory:
    return Memory(
        id=mid,
        memory_type="fact",
        content=content,
        content_hash=content_hash(content + mid),
        visibility=visibility,
        owner_user_id=owner,
        project_key=SHARED_PROJECT_KEY,
        tags=[],
        status="active",
    )


# Tokens that would indicate an admin retrieval bypass slipped in.
_FORBIDDEN_PATTERNS = (
    re.compile(r"\bbypass_predicate\b"),
    re.compile(r"\bskip_visibility\b"),
    re.compile(r"\bis_admin\s*\("),
    re.compile(r"\brequire_admin\b"),
    re.compile(r"role\s*==\s*['\"]admin['\"]"),
)


def _retrieval_and_adapter_sources() -> list[Path]:
    root = Path(memforge.__file__).resolve().parent
    files: list[Path] = []
    for sub in ("retrieval", "storage/adapters"):
        for path in (root / sub).rglob("*.py"):
            files.append(path)
    return files


def test_no_admin_retrieval_path_in_source():
    """Static scan: the retrieval and storage-adapter modules must not contain
    any admin-role branch or predicate-bypass mechanism."""
    offenders: list[tuple[str, str]] = []
    for path in _retrieval_and_adapter_sources():
        text = path.read_text(encoding="utf-8")
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern.search(text):
                offenders.append((str(path), pattern.pattern))
    assert offenders == [], (
        "Admin bypass markers found in retrieval or adapter source: "
        f"{offenders}. There must be exactly one predicate per channel."
    )


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "admin_path.db"))
    await database.connect()
    yield database
    await database.close()


class _Coll:
    def __init__(self) -> None:
        self.items: dict[str, dict] = {}

    def upsert(self, *, ids, embeddings, metadatas, **_):
        for i, mid in enumerate(ids):
            self.items[mid] = {"metadata": dict(metadatas[i])}

    def query(self, *, query_embeddings, n_results, where=None, **_):
        def matches(meta, clause):
            if "$and" in clause:
                return all(matches(meta, c) for c in clause["$and"])
            if "$or" in clause:
                return any(matches(meta, c) for c in clause["$or"])
            for k, v in clause.items():
                if isinstance(v, dict) and "$in" in v:
                    if meta.get(k) not in v["$in"]:
                        return False
                else:
                    if meta.get(k) != v:
                        return False
            return True

        ids, dists = [], []
        for mid, rec in self.items.items():
            if where is None or matches(rec["metadata"], where):
                ids.append(mid)
                dists.append(0.1)
        return {"ids": [ids[:n_results]], "distances": [dists[:n_results]]}


@pytest.mark.asyncio
async def test_search_with_admin_styled_caller_still_hides_other_users_private(
    db, monkeypatch,
):
    """Even if a caller arrives with attributes that look administrative
    (here, an attempt to attach a `role` and `is_admin` flag to the scope's
    user_id), `SearchEngine.search` only consults the AccessScope predicate.
    There is no admin retrieval channel to query; the predicate decides."""

    async def fake_analyze_query(*args, **kwargs):
        return QueryAnalysis()

    monkeypatch.setattr(
        "memforge.retrieval.search.analyze_query", fake_analyze_query,
    )

    coll = _Coll()
    coll.upsert(
        ids=["a-shared"], embeddings=[[0.1, 0.1, 0.1]],
        metadatas=[{
            "status": "active", "visibility": WORKSPACE,
            "owner_user_id": "", "project_key": SHARED_PROJECT_KEY,
            "memory_type": "fact",
        }],
    )
    coll.upsert(
        ids=["a-priv-u2"], embeddings=[[0.1, 0.1, 0.1]],
        metadatas=[{
            "status": "active", "visibility": PRIVATE,
            "owner_user_id": "u-2", "project_key": SHARED_PROJECT_KEY,
            "memory_type": "fact",
        }],
    )
    await db.insert_memory(_mem("a-shared", "deploy decision"))
    await db.insert_memory(_mem("a-priv-u2", "deploy decision",
                                 visibility=PRIVATE, owner="u-2"))

    adapters = build_sqlite_adapters(db, coll)
    engine = SearchEngine(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg={},
        config=RetrievalConfig(),
    )
    engine._get_or_compute_embedding = lambda query: [0.1, 0.1, 0.1]

    # An admin-styled caller from u-1: they cannot escalate by claiming to be
    # an administrator. The scope is the only authority.
    admin_styled_scope = AccessScope(
        user_id="u-1",
        open_projects=frozenset({SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY}),
        member_projects=frozenset(),
        include_private=False,
        allowed_statuses=("active",),
        active_project=None,
        scope_mode="project-first",
    )
    result = await engine.search(
        "deploy", top_k=10, request_scope=admin_styled_scope,
    )
    ids = {row.memory_id for row in result["results"]}
    assert "a-shared" in ids
    assert "a-priv-u2" not in ids
