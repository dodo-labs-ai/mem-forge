from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from memforge.agent_sessions import (
    agent_session_source_id,
    ensure_agent_session_source,
    submit_agent_hook_receipt,
    submit_agent_session_document,
)
from memforge.config import AppConfig
from memforge.genes import GENE_REGISTRY, create_gene
from memforge.genes.agent_session_gene import AgentSessionGene
from memforge.storage.database import Database


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    return cfg


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "agent_sessions.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_agent_session_sources_are_private_and_partitioned_by_owner(
    db: Database,
    tmp_path: Path,
):
    cfg = _config(tmp_path)

    alice = await ensure_agent_session_source(
        db,
        cfg,
        client="codex",
        owner_user_id="alice",
    )
    bob = await ensure_agent_session_source(
        db,
        cfg,
        client="codex",
        owner_user_id="bob",
    )

    assert alice["id"] != bob["id"]
    assert alice["owner_user_id"] == "alice"
    assert bob["owner_user_id"] == "bob"
    assert alice["access_policy"] == bob["access_policy"] == "private"
    assert alice["config"]["documents_dir"] != bob["config"]["documents_dir"]
    assert Path(alice["config"]["documents_dir"]).name == alice["id"]
    assert Path(bob["config"]["documents_dir"]).name == bob["id"]


@pytest.mark.asyncio
async def test_same_agent_window_for_two_users_has_distinct_document_identity(
    db: Database,
    tmp_path: Path,
):
    cfg = _config(tmp_path)
    payload = {
        "db": db,
        "config": cfg,
        "client": "codex",
        "session_id": "same-session",
        "trigger": "Stop",
        "document_markdown": "## Outcome\nA user-owned durable finding.",
        "workspace": "/workspace/repo",
    }

    alice = await submit_agent_session_document(**payload, user_id="alice")
    bob = await submit_agent_session_document(**payload, user_id="bob")

    assert alice["doc_id"] != bob["doc_id"]
    assert alice["source_id"] == agent_session_source_id("codex", "alice")
    assert bob["source_id"] == agent_session_source_id("codex", "bob")
    assert Path(alice["document_uri"]).is_relative_to(
        Path((await db.get_source(alice["source_id"]))["config"]["documents_dir"])
    )
    assert Path(bob["document_uri"]).is_relative_to(
        Path((await db.get_source(bob["source_id"]))["config"]["documents_dir"])
    )


@pytest.mark.asyncio
async def test_submit_agent_session_document_records_receipt_and_source_package(db: Database, tmp_path: Path):
    cfg = _config(tmp_path)

    result = await submit_agent_session_document(
        db=db,
        config=cfg,
        user_id="user-owner",
        client="codex",
        session_id="sess-123",
        trigger="Stop",
        document_markdown="# Session Summary\n\n## User-Confirmed Decisions\n- Use generated session documents.",
        workspace="/workspace/mem-forge",
        repo="mem-forge",
        branch="main",
        commit_sha="abc123",
        history_window_kind="session",
        history_window_start="2026-05-21T10:00:00+00:00",
        history_window_end="2026-05-21T11:00:00+00:00",
        submitted_at="2026-05-21T11:30:00+00:00",
    )

    source = await db.get_source(agent_session_source_id("codex", "user-owner"))
    receipt = await db.get_agent_session_receipt(result["doc_id"])
    package = json.loads(Path(result["document_uri"]).read_text(encoding="utf-8"))

    assert source is not None
    assert source["type"] == "agent_session"
    assert receipt is not None
    assert receipt["client"] == "codex"
    assert receipt["session_id"] == "sess-123"
    assert receipt["document_hash"] == result["document_hash"]
    assert package["package_kind"] == "agent_session_document"
    assert package["content_role"] == "generated_summary"
    assert package["markdown"].startswith("# Session Summary")
    assert "source_updated_at" not in package
    assert "source_updated_at" not in receipt["metadata"]
    documents_root = Path(source["config"]["documents_dir"])
    doc_path = Path(result["document_uri"])
    # Without an admin-bound `project_binding`, agent-session documents
    # land under the UNSORTED bucket. An admin can later attach a
    # `by_field` binding on `repo` to route per-project.
    assert doc_path == documents_root / "unsorted" / f"{result['doc_id']}.json"


@pytest.mark.asyncio
async def test_submit_agent_session_document_records_explicit_source_updated_at(
    db: Database,
    tmp_path: Path,
):
    cfg = _config(tmp_path)

    result = await submit_agent_session_document(
        db=db,
        config=cfg,
        user_id="user-owner",
        client="codex",
        session_id="sess-observed-at",
        trigger="Stop",
        document_markdown="## Outcome\nA historical session produced a durable finding.",
        workspace="/workspace/mem-forge",
        submitted_at="2026-06-23T22:00:00+00:00",
        source_updated_at="2026-06-20T04:23:51Z",
    )

    receipt = await db.get_agent_session_receipt(result["doc_id"])
    package = json.loads(Path(result["document_uri"]).read_text(encoding="utf-8"))

    assert receipt is not None
    assert receipt["submitted_at"] == "2026-06-23T22:00:00+00:00"
    assert "source_updated_at" not in receipt["metadata"]
    assert package["last_modified"] == "2026-06-23T22:00:00+00:00"
    assert package["source_updated_at"] == "2026-06-20T04:23:51+00:00"


@pytest.mark.asyncio
async def test_agent_session_gene_discovers_and_normalizes_submitted_documents(db: Database, tmp_path: Path):
    cfg = _config(tmp_path)
    submitted = await submit_agent_session_document(
        db=db,
        config=cfg,
        user_id="user-owner",
        client="claude-code",
        session_id="sess-456",
        trigger="PreCompact",
        document_markdown="## Outcome\nConfirmed the generated document path.",
        workspace="/workspace/mem-forge",
        repo="mem-forge",
        branch="codex/agent-session",
        commit_sha="def456",
        history_window_kind="compaction",
        history_window_start="2026-05-21T12:00:00+00:00",
        history_window_end="2026-05-21T12:30:00+00:00",
    )
    source = await db.get_source(agent_session_source_id("claude-code", "user-owner"))
    gene = AgentSessionGene(config=source["config"], source_id=source["id"])

    items = [item async for item in gene.discover()]
    assert [item.item_id for item in items] == [submitted["doc_id"]]
    assert items[0].source_url.startswith("agent-session://claude-code/sess-456/")
    # No binding on the source: project resolves to UNSORTED rather than
    # leaking the workspace basename or the repo into the project key.
    assert items[0].space_or_project == "UNSORTED"

    raw = await gene.fetch(items[0])
    normalized = await gene.normalize(raw)

    assert "Client: claude-code" not in normalized.markdown_body
    assert "Session ID: sess-456" not in normalized.markdown_body
    assert "Workspace: /workspace/mem-forge" not in normalized.markdown_body
    assert "Confirmed the generated document path." in normalized.markdown_body
    assert normalized.source_semantics["source_kind"] == "generated_agent_summary"
    assert normalized.source_semantics["client"] == "claude-code"
    assert normalized.source_semantics["workspace"] == "/workspace/mem-forge"
    assert normalized.source_semantics["trigger"] == "PreCompact"


@pytest.mark.asyncio
async def test_agent_session_gene_normalize_exposes_explicit_source_updated_at(db: Database, tmp_path: Path):
    cfg = _config(tmp_path)
    submitted = await submit_agent_session_document(
        db=db,
        config=cfg,
        user_id="user-owner",
        client="codex",
        session_id="sess-source-updated",
        trigger="Stop",
        document_markdown="## Outcome\nA historical session produced a durable finding.",
        workspace="/workspace/mem-forge",
        submitted_at="2026-06-23T22:00:00+00:00",
        source_updated_at="2026-06-20T04:23:51Z",
    )
    source = await db.get_source(agent_session_source_id("codex", "user-owner"))
    gene = AgentSessionGene(config=source["config"], source_id=source["id"])
    [item] = [item async for item in gene.discover()]

    normalized = await gene.normalize(await gene.fetch(item))

    assert item.item_id == submitted["doc_id"]
    assert normalized.source_semantics["source_updated_at"] == "2026-06-20T04:23:51+00:00"


@pytest.mark.asyncio
async def test_agent_session_gene_ignores_receipt_metadata_source_updated_at(
    db: Database,
    tmp_path: Path,
):
    cfg = _config(tmp_path)
    submitted = await submit_agent_session_document(
        db=db,
        config=cfg,
        user_id="user-owner",
        client="codex",
        session_id="sess-stale-metadata-source-updated",
        trigger="Stop",
        document_markdown="## Outcome\nNo source updated timestamp was available.",
        workspace="/workspace/mem-forge",
        submitted_at="2026-06-23T22:00:00+00:00",
        metadata={"source_updated_at": "2026-06-20T04:23:51Z"},
    )
    source = await db.get_source(agent_session_source_id("codex", "user-owner"))
    receipt = await db.get_agent_session_receipt(submitted["doc_id"])
    gene = AgentSessionGene(config=source["config"], source_id=source["id"])
    [item] = [item async for item in gene.discover()]

    normalized = await gene.normalize(await gene.fetch(item))

    assert "source_updated_at" not in receipt["metadata"]
    assert "source_updated_at" not in json.loads(Path(submitted["document_uri"]).read_text(encoding="utf-8"))
    assert normalized.source_semantics["source_updated_at"] is None


@pytest.mark.asyncio
async def test_agent_session_gene_incremental_discovery_uses_submitted_timestamp(db: Database, tmp_path: Path):
    cfg = _config(tmp_path)
    await submit_agent_session_document(
        db=db,
        config=cfg,
        user_id="user-owner",
        client="codex",
        session_id="sess-789",
        trigger="Stop",
        document_markdown="## Outcome\nOlder summary.",
        workspace="/workspace/mem-forge",
        history_window_kind="session",
        submitted_at="2026-05-21T09:00:00+00:00",
    )
    source = await db.get_source(agent_session_source_id("codex", "user-owner"))
    gene = create_gene("agent_session", source["config"], source["id"])

    items = [
        item
        async for item in gene.discover(
            since=datetime(2026, 5, 21, 10, 0, tzinfo=timezone.utc),
        )
    ]

    assert items == []
    assert GENE_REGISTRY["agent_session"] is AgentSessionGene


@pytest.mark.asyncio
async def test_agent_session_gene_ignores_hook_receipts(db: Database, tmp_path: Path):
    cfg = _config(tmp_path)
    await submit_agent_hook_receipt(
        db=db,
        client="codex",
        session_id="sess-receipt",
        hook="Stop",
        workspace="/workspace/mem-forge",
        repo="mem-forge",
        branch="main",
        commit_sha="abc123",
        metadata={"has_transcript_path": True},
    )
    await submit_agent_session_document(
        db=db,
        config=cfg,
        user_id="user-owner",
        client="codex",
        session_id="sess-summary",
        trigger="TaskComplete",
        document_markdown="## Durable Findings\n- Summary documents are source material.",
        workspace="/workspace/mem-forge",
        repo="mem-forge",
    )
    source = await db.get_source(agent_session_source_id("codex", "user-owner"))
    gene = create_gene("agent_session", source["config"], source["id"])

    items = [item async for item in gene.discover()]

    assert len(items) == 1
    assert "sess-summary" in items[0].source_url


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


@pytest.mark.asyncio
async def test_agent_session_gene_skips_legacy_hook_capture_packages(db: Database, tmp_path: Path):
    cfg = _config(tmp_path)
    submitted = await submit_agent_session_document(
        db=db,
        config=cfg,
        user_id="user-owner",
        client="codex",
        session_id="sess-valid-stop-summary",
        trigger="Stop",
        document_markdown="## Durable Findings\n- Stop-triggered explicit summaries are valid source material.",
        workspace="/workspace/mem-forge",
        repo="mem-forge",
    )
    source = await db.get_source(agent_session_source_id("codex", "user-owner"))
    documents_dir = Path(source["config"]["documents_dir"])
    legacy_package = {
        "doc_id": "agent-session-codex-legacy-stop-hook-capture",
        "title": "Agent Session Hook Capture: codex Stop",
        "source_url": "agent-session://codex/legacy/stop/agent-session-codex-legacy-stop-hook-capture",
        "last_modified": "2026-05-25T12:00:00+00:00",
        "space_or_project": "mem-forge",
        "version": "legacy",
        "markdown": (
            "# Agent Session Hook Capture\n\n"
            "MemForge received a `Stop` lifecycle hook. "
            "This document records lifecycle metadata only."
        ),
        "receipt": {
            "client": "codex",
            "session_id": "legacy",
            "trigger": "Stop",
            "source_kind": "generated_agent_summary",
            "metadata": {"hook_event_name": "Stop"},
        },
    }
    (documents_dir / f"{legacy_package['doc_id']}.json").write_text(
        json.dumps(legacy_package),
        encoding="utf-8",
    )

    gene = create_gene("agent_session", source["config"], source["id"])
    items = [item async for item in gene.discover()]

    assert [item.item_id for item in items] == [submitted["doc_id"]]


@pytest.mark.asyncio
async def test_agent_session_gene_omits_receipt_metadata_from_normalized_markdown(db: Database, tmp_path: Path):
    cfg = _config(tmp_path)
    submitted = await submit_agent_session_document(
        db=db,
        config=cfg,
        user_id="user-owner",
        client="codex",
        session_id="sess-pathsafe",
        trigger="TaskComplete",
        document_markdown="## Durable Findings\n- Path metadata is kept out of LLM-visible markdown.",
        workspace="/workspace/mem-forge",
        repo="mem-forge",
        metadata={
            "has_transcript_path": True,
            "transcript_path": "/private/tmp/transcript.jsonl",
            "turn_id": "turn-1",
        },
    )
    source = await db.get_source(agent_session_source_id("codex", "user-owner"))
    gene = create_gene("agent_session", source["config"], source["id"])
    items = [item async for item in gene.discover()]
    item = next(item for item in items if item.item_id == submitted["doc_id"])

    normalized = await gene.normalize(await gene.fetch(item))

    assert "has_transcript_path: True" not in normalized.markdown_body
    assert "turn_id: turn-1" not in normalized.markdown_body
    assert "- transcript_path:" not in normalized.markdown_body
    assert "/private/tmp/transcript.jsonl" not in normalized.markdown_body
    assert normalized.source_semantics["client"] == "codex"
    assert normalized.source_semantics["session_id"] == "sess-pathsafe"


@pytest.mark.asyncio
async def test_agent_session_gene_keeps_operational_sections_out_of_extraction_markdown(
    db: Database,
    tmp_path: Path,
):
    cfg = _config(tmp_path)
    submitted = await submit_agent_session_document(
        db=db,
        config=cfg,
        user_id="user-owner",
        client="codex",
        session_id="sess-quality",
        trigger="TaskComplete",
        document_markdown=(
            "## Outcome\n"
            "Implemented recursive fit-first unitization for full-document extraction.\n\n"
            "## Durable Findings\n"
            "- Oversized documents are recursively partitioned by Markdown section subtree.\n\n"
            "## Validation\n"
            "- `/opt/homebrew/bin/uv run --extra dev pytest -q` passed.\n\n"
            "## Runtime Notes\n"
            "- Started MemForge API on `http://127.0.0.1:8765`.\n\n"
            "## Evidence\n"
            "- Tested through Codex MCP submit_agent_session_document.\n\n"
            "## Rejected Ideas\n"
            "- Do not index raw hook receipts as source documents.\n"
        ),
        workspace="/workspace/mem-forge",
        repo="mem-forge",
    )
    source = await db.get_source(agent_session_source_id("codex", "user-owner"))
    gene = create_gene("agent_session", source["config"], source["id"])
    items = [item async for item in gene.discover()]
    item = next(item for item in items if item.item_id == submitted["doc_id"])

    normalized = await gene.normalize(await gene.fetch(item))

    assert "Implemented recursive fit-first unitization" in normalized.markdown_body
    assert "Oversized documents are recursively partitioned" in normalized.markdown_body
    assert "Do not index raw hook receipts" in normalized.markdown_body
    assert "pytest -q" not in normalized.markdown_body
    assert "127.0.0.1:8765" not in normalized.markdown_body
    assert "submit_agent_session_document" not in normalized.markdown_body
