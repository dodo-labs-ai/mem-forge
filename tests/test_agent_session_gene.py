from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from memforge.agent_sessions import submit_agent_session_document
from memforge.agent_sessions import submit_agent_hook_receipt
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
async def test_submit_agent_session_document_records_receipt_and_source_package(db: Database, tmp_path: Path):
    cfg = _config(tmp_path)

    result = await submit_agent_session_document(
        db=db,
        config=cfg,
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

    source = await db.get_source("src-agent-sessions")
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
    documents_root = Path(source["config"]["documents_dir"])
    doc_path = Path(result["document_uri"])
    assert doc_path == documents_root / "mem-forge" / f"{result['doc_id']}.json"


@pytest.mark.asyncio
async def test_agent_session_gene_discovers_and_normalizes_submitted_documents(db: Database, tmp_path: Path):
    cfg = _config(tmp_path)
    submitted = await submit_agent_session_document(
        db=db,
        config=cfg,
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
    source = await db.get_source("src-agent-sessions")
    gene = AgentSessionGene(config=source["config"], source_id=source["id"])

    items = [item async for item in gene.discover()]
    assert [item.item_id for item in items] == [submitted["doc_id"]]
    assert items[0].source_url.startswith("agent-session://claude-code/sess-456/")
    assert items[0].space_or_project == "mem-forge"

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
async def test_agent_session_gene_incremental_discovery_uses_submitted_timestamp(db: Database, tmp_path: Path):
    cfg = _config(tmp_path)
    await submit_agent_session_document(
        db=db,
        config=cfg,
        client="codex",
        session_id="sess-789",
        trigger="Stop",
        document_markdown="## Outcome\nOlder summary.",
        workspace="/workspace/mem-forge",
        history_window_kind="session",
        submitted_at="2026-05-21T09:00:00+00:00",
    )
    source = await db.get_source("src-agent-sessions")
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
        client="codex",
        session_id="sess-summary",
        trigger="TaskComplete",
        document_markdown="## Durable Findings\n- Summary documents are source material.",
        workspace="/workspace/mem-forge",
        repo="mem-forge",
    )
    source = await db.get_source("src-agent-sessions")
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
        client="codex",
        session_id="sess-valid-stop-summary",
        trigger="Stop",
        document_markdown="## Durable Findings\n- Stop-triggered explicit summaries are valid source material.",
        workspace="/workspace/mem-forge",
        repo="mem-forge",
    )
    source = await db.get_source("src-agent-sessions")
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
    source = await db.get_source("src-agent-sessions")
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
    source = await db.get_source("src-agent-sessions")
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
