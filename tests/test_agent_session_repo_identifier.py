from __future__ import annotations

import json
from pathlib import Path

import pytest

from memforge.agent_sessions import (
    agent_session_source_id,
    submit_agent_session_document,
)
from memforge.config import AppConfig
from memforge.genes.agent_session_gene import AgentSessionGene
from memforge.repo_identity import normalize_repo_identifier
from memforge.storage.database import Database


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    return cfg


def test_normalize_repo_identifier_prefers_canonical_remote_slug():
    assert (
        normalize_repo_identifier("git@github.tools.sap:HCM/memforge-cloud.git")
        == "github.tools.sap/hcm/memforge-cloud"
    )
    assert normalize_repo_identifier("https://github.com/shno-labs/mem-forge.git") == "github.com/shno-labs/mem-forge"
    assert normalize_repo_identifier("HTTPS://GitHub.com/Shno-Labs/Mem-Forge.GIT?ref=main") == (
        "github.com/shno-labs/mem-forge"
    )
    assert normalize_repo_identifier("mem-inception") == "mem-inception"
    assert normalize_repo_identifier(None) is None


@pytest.mark.asyncio
async def test_submit_agent_session_document_persists_repo_identifier(tmp_path: Path):
    db = Database(str(tmp_path / "repo-id.db"))
    await db.connect()
    try:
        cfg = _config(tmp_path)
        result = await submit_agent_session_document(
            db=db,
            config=cfg,
            user_id="user-owner",
            client="codex",
            session_id="sess-repo-id",
            trigger="Stop",
            document_markdown="# Session Summary\n\n- The repo identity matters.",
            workspace="/workspace/memforge-cloud",
            repo="git@github.tools.sap:HCM/memforge-cloud.git",
        )

        package = json.loads(Path(result["document_uri"]).read_text(encoding="utf-8"))

        assert package["receipt"]["metadata"]["repo_identifier"] == ("github.tools.sap/hcm/memforge-cloud")
        assert result["receipt"]["metadata"]["repo_identifier"] == ("github.tools.sap/hcm/memforge-cloud")
    finally:
        await db.close()


@pytest.mark.asyncio
async def test_agent_session_gene_exposes_repo_identifier_in_source_semantics(tmp_path: Path):
    documents_dir = tmp_path / "agent-session-submissions"
    package_dir = documents_dir / "unsorted"
    package_dir.mkdir(parents=True)
    package = {
        "package_kind": "agent_session_document",
        "content_role": "generated_summary",
        "doc_id": "agent-session-codex-sess-stop",
        "title": "Agent Session: codex sess Stop",
        "source_url": "agent-session://codex/sess/stop/doc",
        "last_modified": "2026-06-01T12:00:00+00:00",
        "space_or_project": "UNSORTED",
        "version": "v1",
        "markdown": "# Summary\n\n- Durable repo-level finding.",
        "receipt": {
            "doc_id": "agent-session-codex-sess-stop",
            "source_id": agent_session_source_id("codex", "user-owner"),
            "client": "codex",
            "session_id": "sess",
            "trigger": "Stop",
            "workspace": "/workspace/memforge-cloud",
            "repo": "git@github.tools.sap:HCM/memforge-cloud.git",
            "branch": "main",
            "commit_sha": "abc123",
            "history_window_kind": "session",
            "history_window_start": None,
            "history_window_end": None,
            "submitted_at": "2026-06-01T12:00:00+00:00",
            "document_hash": "deadbeef",
            "source_kind": "generated_agent_summary",
            "document_uri": str(package_dir / "agent-session-codex-sess-stop.json"),
            "metadata": {
                "repo_identifier": "github.tools.sap/hcm/memforge-cloud",
            },
            "updated_at": "2026-06-01T12:00:00+00:00",
        },
    }
    package_path = package_dir / "agent-session-codex-sess-stop.json"
    package_path.write_text(json.dumps(package), encoding="utf-8")

    gene = AgentSessionGene(
        config={"documents_dir": str(documents_dir), "client": "codex"},
        source_id=agent_session_source_id("codex", "user-owner"),
    )
    item = [item async for item in gene.discover()][0]
    raw = await gene.fetch(item)
    normalized = await gene.normalize(raw)

    assert normalized.source_semantics["repo_identifier"] == ("github.tools.sap/hcm/memforge-cloud")
