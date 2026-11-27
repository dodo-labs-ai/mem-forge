"""A local Markdown Source reads a folder on its device (ADR 0058, design §10.3).

Each run walks the folder once: the walk lists every file the configuration
includes, by its path relative to the folder, and that listing serves both
the run's discovery and its existence listing. A folder the walk cannot
read fails the run and removes nothing; a file it lists but cannot read
fails only its own Document, which stays as it was.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from memforge.genes import local_markdown_gene
from memforge.genes.local_markdown_gene import LOCAL_FILE_MAX_BYTES, LocalMarkdownGene
from memforge.local_agent.document_identity import build_local_markdown_doc_id
from tests.test_device_executor_move import OWNER, Deployment, DeviceType, _Runtime

SOURCE_ID = "src-notes"
VAULT_ID = "notes"
NOTES = {
    "ideas.md": "# Ideas\n\nShip smaller releases.\n",
    "team/plans.md": "# Plans\n\nMove the queue on Friday.\n",
    "team/standup.md": "# Standup\n\nAda reviews the release notes.\n",
}
_NO_PROVIDER = SimpleNamespace(
    _Provider=lambda listing: SimpleNamespace(http=SimpleNamespace(serving=lambda module: nullcontext()))
)


def _doc_id(relative_path: str) -> str:
    return build_local_markdown_doc_id(source_id=SOURCE_ID, vault_id=VAULT_ID, relative_path=relative_path)


@pytest.fixture
def folder(tmp_path) -> Path:
    root = (tmp_path / "notes").resolve()
    for relative_path, text in NOTES.items():
        (root / relative_path).parent.mkdir(parents=True, exist_ok=True)
        (root / relative_path).write_text(text)
    return root


@pytest.fixture
def notes(tmp_path, folder) -> Iterator[Deployment]:
    kind = DeviceType(
        "local_markdown", _NO_PROVIDER, local_markdown_gene, {"root": str(folder), "vault_id": VAULT_ID}, ""
    )
    deployment = Deployment(tmp_path, kind)
    asyncio.run(
        deployment.db.upsert_source(
            id=SOURCE_ID,
            type="local_markdown",
            name="Notes",
            config_json=json.dumps(kind.config),
            access_policy="workspace",
            owner_user_id=OWNER,
        )
    )
    try:
        with TestClient(deployment.app) as client:
            deployment.client = client
            yield deployment
    finally:
        asyncio.run(deployment.db.close())


def _sync(deployment: Deployment):
    runtime = _Runtime(deployment.db, deployment.app.state.document_store, deployment.extractor)
    gene = LocalMarkdownGene(config=dict(deployment.kind.config), source_id=SOURCE_ID)
    return asyncio.run(runtime.orchestrator().sync_gene(gene=gene, source_name="Notes", source_id=SOURCE_ID))


def test_one_walk_serves_the_runs_discovery_and_its_listing(folder, monkeypatch):
    walks = []
    walk = local_markdown_gene._walk_folder

    def counted(*args):
        walks.append(args[0])
        return walk(*args)

    monkeypatch.setattr(local_markdown_gene, "_walk_folder", counted)
    gene = LocalMarkdownGene(config={"root": str(folder), "vault_id": VAULT_ID}, source_id=SOURCE_ID)

    async def run():
        discovered = [item async for item in gene.discover(since=None)]
        return discovered, await gene.list_scope()

    discovered, listing = asyncio.run(run())

    assert walks == [folder]
    assert sorted(item.extra["relative_path"] for item in discovered) == sorted(NOTES)
    assert listing.doc_ids == frozenset(_doc_id(path) for path in NOTES)


def test_the_listing_names_each_file_by_its_path_and_reads_no_file(folder, monkeypatch):
    def unread(self, relative_path):
        raise AssertionError(f"the listing read {relative_path}")

    monkeypatch.setattr(LocalMarkdownGene, "_read_file", unread)
    gene = LocalMarkdownGene(config={"root": str(folder), "vault_id": VAULT_ID}, source_id=SOURCE_ID)

    listing = asyncio.run(gene.list_scope())

    assert listing.doc_ids == frozenset(_doc_id(path) for path in NOTES)


def test_a_run_extracts_each_file_once_and_an_unchanged_folder_extracts_nothing(notes):
    assert _sync(notes).last_sync_status == "success"
    assert sorted(notes.documents(SOURCE_ID)) == sorted(_doc_id(path) for path in NOTES)
    assert notes.documents(SOURCE_ID)[_doc_id("ideas.md")][0] == "Ideas"
    notes.extractor.fragment_calls.clear()

    assert _sync(notes).last_sync_status == "success"

    assert notes.extracted_documents(SOURCE_ID) == []


def test_a_renamed_folder_lists_new_paths_and_its_old_paths_go(notes, folder):
    _sync(notes)

    (folder / "team").rename(folder / "squad")
    state = _sync(notes)

    assert state.last_sync_status == "success"
    assert sorted(notes.documents(SOURCE_ID)) == sorted(
        _doc_id(path) for path in ("ideas.md", "squad/plans.md", "squad/standup.md")
    )


def test_a_folder_that_is_gone_fails_the_run_and_removes_nothing(notes, folder):
    _sync(notes)
    held = notes.documents(SOURCE_ID)

    shutil.rmtree(folder)
    state = _sync(notes)

    assert state.last_sync_status == "failed"
    assert notes.documents(SOURCE_ID) == held


@pytest.mark.skipif(os.geteuid() == 0, reason="the superuser reads any directory")
def test_a_directory_the_walk_cannot_read_fails_the_run_and_removes_nothing(notes, folder):
    _sync(notes)
    held = notes.documents(SOURCE_ID)

    (folder / "team").chmod(0)
    try:
        state = _sync(notes)
    finally:
        (folder / "team").chmod(0o755)

    assert state.last_sync_status == "failed"
    assert notes.documents(SOURCE_ID) == held


@pytest.mark.parametrize(
    "spoil",
    [
        pytest.param(lambda path: path.write_bytes(b"#" * (LOCAL_FILE_MAX_BYTES + 1)), id="too-large"),
        pytest.param(lambda path: path.write_bytes(b"# Plans\n\n\xff\xfe broken\n"), id="not-utf-8"),
        pytest.param(
            lambda path: path.chmod(0),
            id="unreadable",
            marks=pytest.mark.skipif(os.geteuid() == 0, reason="the superuser reads any file"),
        ),
    ],
)
def test_a_file_the_run_cannot_read_fails_only_its_document_which_stays(notes, folder, spoil):
    _sync(notes)
    held = notes.documents(SOURCE_ID)
    plans = folder / "team" / "plans.md"

    spoil(plans)
    try:
        state = _sync(notes)
    finally:
        plans.chmod(0o644)

    assert state.docs_failed == 1
    assert state.last_sync_status == "partial"
    assert notes.documents(SOURCE_ID) == held
