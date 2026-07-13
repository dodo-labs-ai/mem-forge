"""Regression test: per-client agent-session sources must not cross-pollinate.

Every coding client and owner pair gets its own private source row. This test
also keeps the gene-level client filter explicit for manually constructed
package directories.

The gene must instead filter by ``receipt.client`` so an
``AgentSessionGene`` bound to the codex source only discovers codex packages,
and the same for claude-code. This test encodes that contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from memforge.agent_sessions import agent_session_source_id
from memforge.genes.agent_session_gene import AgentSessionGene


def _write_package(
    documents_dir: Path,
    *,
    client: str,
    session_id: str,
    trigger: str,
    workspace: str,
    repo: str,
) -> str:
    """Write a client-shaped agent-session package and return its doc_id.

    Mirrors the package layout produced by submit_agent_session_document:
    keys "package_kind", "content_role", "doc_id", "title", "source_url",
    "last_modified", "space_or_project", "version", "markdown", "receipt".
    """
    doc_id = f"agent-session-{client}-{session_id}-{trigger}"
    source_url = f"agent-session://{client}/{session_id}/{trigger}/{doc_id}"
    package = {
        "package_kind": "agent_session_document",
        "content_role": "generated_summary",
        "doc_id": doc_id,
        "title": f"Agent Session: {client} {session_id} {trigger}",
        "source_url": source_url,
        "last_modified": "2026-06-01T12:00:00+00:00",
        "space_or_project": repo,
        "version": "v1",
        "markdown": (f"## Durable Findings\n- {client} session {session_id} produced a summary worth keeping.\n"),
        "receipt": {
            "doc_id": doc_id,
            "source_id": agent_session_source_id(client, "user-owner"),
            "client": client,
            "session_id": session_id,
            "trigger": trigger,
            "workspace": workspace,
            "repo": repo,
            "branch": "main",
            "commit_sha": "abc123",
            "history_window_kind": "session",
            "history_window_start": None,
            "history_window_end": None,
            "submitted_at": "2026-06-01T12:00:00+00:00",
            "document_hash": "deadbeef",
            "source_kind": "generated_agent_summary",
            "source_url": source_url,
            "metadata": {},
            "updated_at": "2026-06-01T12:00:00+00:00",
        },
    }
    package_dir = documents_dir / repo
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / f"{doc_id}.json").write_text(json.dumps(package), encoding="utf-8")
    return doc_id


@pytest.mark.asyncio
async def test_agent_session_gene_only_discovers_its_own_clients_packages(tmp_path: Path):
    """A per-client AgentSessionGene must skip packages from other clients.

    The shared documents_dir contains one codex package and one claude-code
    package. The codex-bound gene must yield only the codex doc.
    """
    documents_dir = tmp_path / "agent-session-submissions"
    documents_dir.mkdir(parents=True, exist_ok=True)

    codex_doc_id = _write_package(
        documents_dir,
        client="codex",
        session_id="sess-codex-1",
        trigger="TaskComplete",
        workspace="/workspace/mem-forge",
        repo="mem-forge",
    )
    _write_package(
        documents_dir,
        client="claude-code",
        session_id="sess-claude-1",
        trigger="Stop",
        workspace="/workspace/mem-forge",
        repo="mem-forge",
    )

    codex_source_id = agent_session_source_id("codex", "user-owner")
    gene = AgentSessionGene(
        config={
            "documents_dir": str(documents_dir),
            "client": "codex",
        },
        source_id=codex_source_id,
    )

    items = [item async for item in gene.discover()]

    discovered_ids = [item.item_id for item in items]
    discovered_clients = [item.author for item in items]

    assert discovered_ids == [codex_doc_id], f"codex-bound gene leaked other clients' documents: {discovered_ids}"
    assert discovered_clients == ["codex"], f"codex-bound gene picked up non-codex receipts: {discovered_clients}"
