"""Storage-layer batched lookup of each memory's source triples, used to pick the
leading source glyph in the admin UI without an N+1 query."""

from datetime import datetime, timezone

import pytest

from memforge.models import Memory, content_hash
from memforge.server.admin_api import _pick_origin_source_type
from memforge.storage.database import Database


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "origin-source.db"))
    await database.connect()
    yield database
    await database.close()


def _memory(mem_id: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=mem_id,
        memory_type="fact",
        content=f"content for {mem_id}",
        content_hash=content_hash(mem_id),
        project_key="PAY",
        confidence=0.9,
        created_at=now,
        updated_at=now,
        status="active",
    )


async def _insert_doc(db: Database, doc_id: str, client: str | None = None) -> None:
    now = datetime.now(timezone.utc).isoformat()
    await db.db.execute(
        """INSERT INTO documents
           (doc_id, source, source_url, title, space_or_project, last_modified, version, content_hash, last_synced, client)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (doc_id, "src", f"http://test/{doc_id}", doc_id, "PAY", now, "1", f"hash-{doc_id}", now, client),
    )
    await db.db.commit()


async def _add_source(
    db: Database,
    memory_id: str,
    doc_id: str,
    source_type: str,
    support_kind: str,
    added_at: str,
    client: str | None = None,
) -> None:
    await _insert_doc(db, doc_id, client=client)
    await db.add_memory_source(
        memory_id, doc_id, source_type, "excerpt", support_kind=support_kind, source_updated_at=None
    )
    # add_memory_source stamps added_at to now(); override it so ordering is testable.
    await db.db.execute(
        "UPDATE memory_sources SET added_at = ? WHERE memory_id = ? AND doc_id = ?",
        (added_at, memory_id, doc_id),
    )
    await db.db.commit()


async def test_empty_input_returns_empty_map(db: Database):
    assert await db.get_origin_source_pairs([]) == {}


async def test_groups_pairs_oldest_first_and_omits_sourceless(db: Database):
    for mem_id in ("mem-a", "mem-b", "mem-c"):
        await db.insert_memory(_memory(mem_id))
    # mem-a's corroborated jira source is added before its extracted confluence one.
    await _add_source(db, "mem-a", "doc-jira", "jira", "corroborated", "2026-01-01T00:00:00")
    await _add_source(db, "mem-a", "doc-conf", "confluence", "extracted", "2026-01-02T00:00:00")
    # mem-b has one source; mem-c has none.
    await _add_source(db, "mem-b", "doc-teams", "teams", "corroborated", "2026-01-01T00:00:00")

    pairs = await db.get_origin_source_pairs(["mem-a", "mem-b", "mem-c"])

    # Each triple is (source_type, support_kind, client). Non-agent-session docs have client=None.
    assert pairs["mem-a"] == [
        ("jira", "corroborated", None),
        ("confluence", "extracted", None),
    ]
    assert pairs["mem-b"] == [("teams", "corroborated", None)]
    assert "mem-c" not in pairs

    # The extracted origin wins even though it was attached later.
    source_type_a, client_a = _pick_origin_source_type(pairs["mem-a"])
    assert source_type_a == "confluence"
    assert client_a is None

    source_type_b, client_b = _pick_origin_source_type(pairs["mem-b"])
    assert source_type_b == "teams"
    assert client_b is None


async def test_client_is_returned_for_agent_session_documents(db: Database):
    """Client from documents.client is included in the triple for agent-session docs."""
    await db.insert_memory(_memory("mem-codex"))
    await _add_source(
        db,
        "mem-codex",
        "doc-agent-sess",
        "agent_session",
        "extracted",
        "2026-01-01T00:00:00",
        client="codex",
    )

    pairs = await db.get_origin_source_pairs(["mem-codex"])

    assert pairs["mem-codex"] == [("agent_session", "extracted", "codex")]
    source_type, client = _pick_origin_source_type(pairs["mem-codex"])
    assert source_type == "agent_session"
    assert client == "codex"
