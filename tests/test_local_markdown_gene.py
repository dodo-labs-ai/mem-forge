"""A local Markdown Source reads a folder on its execution device (ADR 0058, design §10.3).

Each run walks the folder once: the walk lists every file the configuration
includes, by its path relative to the folder, and that listing serves both
the run's discovery and its existence listing. The owner's daemon runs the
walk and uploads only the bodies the server does not hold; the server
extracts only what changed. A folder the walk cannot read fails the run and
removes nothing; a file it lists but cannot read fails only its own
Document, which stays as it was.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from collections.abc import Iterator
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from memforge.genes import local_markdown_gene
from memforge.genes.local_markdown_gene import LOCAL_FILE_MAX_BYTES, LocalMarkdownGene
from memforge.local_agent.device_report import declared_acquisition_contracts
from memforge.local_agent.document_identity import build_local_markdown_doc_id
from memforge.local_agent.source_contract import DOCUMENT_ACQUISITION_OPERATION
from tests.device_report_fixture import DEVICE_ID, connect_device, device_report
from tests.test_device_acquisition import _routed_client
from tests.test_device_executor_move import OWNER, Deployment, DeviceType

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
def notes(tmp_path, folder, monkeypatch) -> Iterator[Deployment]:
    """A local Markdown Source of the owner, executed by the owner's laptop."""

    from tests.test_daemon_service import FakeKeyring

    monkeypatch.setitem(sys.modules, "keyring", FakeKeyring())
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
            created_by_user_id=OWNER,
            execution_owner_user_id=OWNER,
            executor="device",
            execution_device_id=DEVICE_ID,
        )
    )
    connect_device(deployment.db, OWNER)
    try:
        with TestClient(deployment.app) as client:
            deployment.client = client
            yield deployment
    finally:
        asyncio.run(deployment.db.close())


@pytest.fixture
def uploaded_bodies(monkeypatch) -> list[str]:
    """The relative path of each file whose body the daemon uploaded, in upload order."""

    from memforge.tool_client import ToolClient

    uploaded: list[str] = []
    upload = ToolClient.upload_acquired_input

    def recorded(self, *, package, body, **kwargs):
        if body is not None:
            uploaded.append(package["item"]["extra"]["relative_path"])
        return upload(self, package=package, body=body, **kwargs)

    monkeypatch.setattr(ToolClient, "upload_acquired_input", recorded)
    return uploaded


def _sync_now(deployment: Deployment) -> dict:
    """Sync now: the owner's daemon walks the folder and the server's worker processes what it acquired.

    Returns the daemon's job result; a job that fails on the device starts no run.
    """

    import memforge.main as main

    queued = deployment.client.post(f"/api/v1/sources/{SOURCE_ID}/sync")
    assert queued.status_code == 202, queued.text
    [job] = asyncio.run(
        deployment.db.lease_local_agent_jobs(
            user_id=OWNER,
            device=device_report([SOURCE_ID]),
            acquisition_contracts=declared_acquisition_contracts(),
            limit=1,
            lease_seconds=300,
        )
    )
    assert job["operation"] == DOCUMENT_ACQUISITION_OPERATION
    result = main._run_cloud_document_acquisition_job(job, _routed_client(deployment.client))
    if result.get("error") is None:
        deployment.sync()
    assert asyncio.run(
        deployment.db.complete_local_agent_job(
            job_id=job["job_id"],
            user_id=OWNER,
            device_id=DEVICE_ID,
            attempt_count=job["attempt_count"],
            status="failed" if result.get("error") else "succeeded",
            result=result,
            error=result.get("error"),
        )
    )
    return result


def _runs(deployment: Deployment) -> list[dict]:
    return asyncio.run(deployment.db.get_sync_history(source=SOURCE_ID, limit=100))


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


@pytest.mark.skipif(os.geteuid() == 0, reason="the superuser reads any directory")
def test_a_folder_exclude_leaves_out_is_never_read(folder):
    (folder / ".obsidian" / "plugins").mkdir(parents=True)
    (folder / ".obsidian" / "plugins" / "manifest.json").write_text("{}")
    (folder / ".obsidian" / "plugins").chmod(0)
    gene = LocalMarkdownGene(config={"root": str(folder), "vault_id": VAULT_ID}, source_id=SOURCE_ID)

    try:
        listing = asyncio.run(gene.list_scope())
    finally:
        (folder / ".obsidian" / "plugins").chmod(0o755)

    assert listing.doc_ids == frozenset(_doc_id(path) for path in NOTES)


def test_an_unchanged_folder_uploads_no_body_and_extracts_nothing(notes, folder, uploaded_bodies):
    first = _sync_now(notes)

    assert first["counts"] == {"discovered": 3, "uploaded": 3, "failed": 0}
    assert sorted(uploaded_bodies) == sorted(NOTES)
    assert sorted(notes.documents(SOURCE_ID)) == sorted(_doc_id(path) for path in NOTES)
    assert notes.documents(SOURCE_ID)[_doc_id("ideas.md")][0] == "Ideas"
    assert sorted(notes.extracted_documents(SOURCE_ID)) == sorted(_doc_id(path) for path in NOTES)
    held = notes.documents(SOURCE_ID)
    notes.extractor.fragment_calls.clear()
    uploaded_bodies.clear()

    assert _sync_now(notes).get("error") is None

    assert _runs(notes)[0]["status"] == "success"
    assert uploaded_bodies == []
    assert notes.extracted_documents(SOURCE_ID) == []
    assert notes.documents(SOURCE_ID) == held

    (folder / "team" / "plans.md").write_text("# Plans\n\nMove the queue on Monday.\n")
    _sync_now(notes)

    assert uploaded_bodies == ["team/plans.md"]
    assert notes.extracted_documents(SOURCE_ID) == [_doc_id("team/plans.md")]


def test_a_renamed_folders_files_are_read_under_their_new_paths_and_the_old_ones_go(notes, folder, uploaded_bodies):
    _sync_now(notes)
    uploaded_bodies.clear()

    (folder / "team").rename(folder / "squad")
    _sync_now(notes)

    assert _runs(notes)[0]["status"] == "success"
    assert sorted(notes.documents(SOURCE_ID)) == sorted(
        _doc_id(path) for path in ("ideas.md", "squad/plans.md", "squad/standup.md")
    )
    # The moved files' bodies are the ones the server already holds.
    assert uploaded_bodies == []


def test_a_folder_that_is_gone_fails_the_run_and_removes_nothing(notes, folder):
    _sync_now(notes)
    held = notes.documents(SOURCE_ID)
    runs = _runs(notes)

    shutil.rmtree(folder)
    result = _sync_now(notes)

    assert "does not exist" in result["error"]
    assert _runs(notes) == runs
    assert notes.documents(SOURCE_ID) == held


@pytest.mark.skipif(os.geteuid() == 0, reason="the superuser reads any directory")
def test_a_directory_the_walk_cannot_read_fails_the_run_and_removes_nothing(notes, folder):
    _sync_now(notes)
    held = notes.documents(SOURCE_ID)
    runs = _runs(notes)

    (folder / "team").chmod(0)
    try:
        result = _sync_now(notes)
    finally:
        (folder / "team").chmod(0o755)

    assert result["error_type"] == "PermissionError"
    assert _runs(notes) == runs
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
    _sync_now(notes)
    held = notes.documents(SOURCE_ID)
    plans = folder / "team" / "plans.md"

    spoil(plans)
    try:
        result = _sync_now(notes)
    finally:
        plans.chmod(0o644)

    assert result["counts"] == {"discovered": 3, "uploaded": 3, "failed": 1}
    latest = _runs(notes)[0]
    assert latest["status"] == "partial", latest
    assert latest["docs_failed"] == 1
    assert notes.documents(SOURCE_ID) == held
    assert [failure.document_id for failure in asyncio.run(notes.db.list_source_item_failures(source_id=SOURCE_ID))] == [
        _doc_id("team/plans.md")
    ]


def test_a_file_deleted_after_the_walk_fails_only_its_document_which_stays(notes, folder, monkeypatch):
    _sync_now(notes)
    held = notes.documents(SOURCE_ID)
    walk = local_markdown_gene._walk_folder

    def walk_then_delete(*args):
        listed = walk(*args)
        (folder / "team" / "plans.md").unlink()
        return listed

    monkeypatch.setattr(local_markdown_gene, "_walk_folder", walk_then_delete)
    result = _sync_now(notes)

    assert result["counts"] == {"discovered": 3, "uploaded": 3, "failed": 1}
    latest = _runs(notes)[0]
    assert latest["status"] == "partial", latest
    assert notes.documents(SOURCE_ID) == held
    [failure] = asyncio.run(notes.db.list_source_item_failures(source_id=SOURCE_ID))
    assert failure.document_id == _doc_id("team/plans.md")
    assert (failure.stage, failure.error_class) == ("fetch", "builtins.FileNotFoundError"), failure
