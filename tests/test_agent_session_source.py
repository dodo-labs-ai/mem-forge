from __future__ import annotations

from pathlib import Path

import pytest

from memforge.agent_sessions import (
    agent_session_source_id,
    build_agent_session_doc_id,
    ensure_agent_session_source,
    submit_agent_hook_receipt,
)
from memforge.genes import GENE_REGISTRY, create_gene, source_type_supports_sync
from memforge.genes.agent_session_gene import AgentSessionGene
from memforge.storage.database import Database


@pytest.fixture
async def db(tmp_path: Path):
    database = Database(str(tmp_path / "agent_sessions.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_agent_session_sources_are_private_and_partitioned_by_owner(db: Database):
    alice = await ensure_agent_session_source(db, client="codex", owner_user_id="alice")
    bob = await ensure_agent_session_source(db, client="codex", owner_user_id="bob")

    assert alice["id"] == agent_session_source_id("codex", "alice")
    assert bob["id"] == agent_session_source_id("codex", "bob")
    assert alice["id"] != bob["id"]
    assert alice["owner_user_id"] == "alice"
    assert bob["owner_user_id"] == "bob"
    assert alice["access_policy"] == bob["access_policy"] == "private"
    assert alice["config"] == bob["config"] == {"client": "codex"}


def test_same_agent_window_for_two_users_has_distinct_receipt_identity():
    window = {
        "client": "codex",
        "session_id": "same-session",
        "trigger": "Stop",
        "workspace": "/workspace/repo",
        "history_window_kind": "boundary",
        "history_window_start": "evt-1",
        "history_window_end": "evt-2",
        "window_hash": "sha256:same",
    }

    alice = build_agent_session_doc_id(owner_user_id="alice", **window)
    bob = build_agent_session_doc_id(owner_user_id="bob", **window)

    assert alice != bob
    assert alice == build_agent_session_doc_id(owner_user_id="alice", **window)


@pytest.mark.asyncio
async def test_agent_session_gene_is_registered_without_sync_or_discovery():
    gene = create_gene("agent_session", {"client": "codex"}, agent_session_source_id("codex", "alice"))

    assert GENE_REGISTRY["agent_session"] is AgentSessionGene
    assert AgentSessionGene.metadata().execution_kinds == ()
    assert not source_type_supports_sync("agent_session")
    assert [item async for item in gene.discover()] == []


@pytest.mark.asyncio
async def test_submit_agent_hook_receipt_deduplicates_same_hook(db: Database):
    first = await submit_agent_hook_receipt(
        db=db,
        client="codex",
        session_id="sess-repeat",
        hook="Stop",
        workspace="/workspace/mem-forge",
        repo="mem-forge",
        branch="main",
        commit_sha="abc123",
        metadata={"has_transcript_path": True},
        submitted_at="2026-05-26T08:00:00+00:00",
    )
    second = await submit_agent_hook_receipt(
        db=db,
        client="codex",
        session_id="sess-repeat",
        hook="Stop",
        workspace="/workspace/mem-forge",
        repo="mem-forge",
        branch="main",
        commit_sha="abc123",
        metadata={"has_transcript_path": False},
        submitted_at="2026-05-26T08:05:00+00:00",
    )

    receipts = await db.list_agent_hook_receipts(session_id="sess-repeat")

    assert first["receipt_id"] == second["receipt_id"]
    assert len(receipts) == 1
    assert receipts[0]["receipt_id"] == first["receipt_id"]
    assert receipts[0]["metadata"] == {"has_transcript_path": False}
    assert receipts[0]["submitted_at"] == "2026-05-26T08:05:00+00:00"
