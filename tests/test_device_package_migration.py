"""Sources on a device package path move onto device acquisition (ADR 0058, design §10.9, gate of A4).

A Jira Source with local agent sync and a GitHub repository Source with local
push hold the packages their owner's daemon submitted. The migration's dry
run rebuilds the input the type's Gene reads from each package and refuses a
Source it cannot move; its apply replays the rebuilt inputs without forced
extraction and moves the Source to the device. Jira Units move to their
``jira-<KEY>`` Documents with their Support. The first device run afterwards
is an ordinary incremental run that extracts only the items that changed.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from collections.abc import Iterator
from datetime import datetime, timezone
from weakref import WeakValueDictionary

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.genes import github_repo_gene, jira_gene
from memforge.genes.github_repo_gene import GitHubRepoGene
from memforge.genes.jira_gene import JiraGene
from memforge.github_repo_utils import build_github_repo_doc_id, parse_github_repo_url
from memforge.models import Memory, slugify
from memforge.pipeline.package_migration import (
    apply_package_migration,
    package_path_sources,
    plan_package_migration,
)
from tests.contracts.document_fakes import ProviderListing
from tests.contracts.document_fakes import github_repo as github_repo_fakes
from tests.contracts.document_fakes import jira as jira_fakes
from tests.test_device_acquisition import DAEMON_SERVER
from tests.test_device_executor_move import OWNER, TEST_SOURCE_KEY, Deployment, DeviceType, _Runtime, _sync_now_on_the_device

SUBMITTED_AT = "2026-06-02T08:00:00+00:00"

JIRA = DeviceType(
    "jira",
    jira_fakes,
    jira_gene,
    {
        "base_url": jira_fakes.BASE_URL,
        "auth_mode": "pat",
        "pat": "token",
        "projects": ["PAY"],
        "issue_types": ["Story"],
    },
    "pat",
)
GITHUB = DeviceType(
    "github_repo",
    github_repo_fakes,
    github_repo_gene,
    {
        "repo_url": github_repo_fakes.REPO_URL,
        "ref": "main",
        "include_paths": ["docs/"],
        "include_extensions": ["md"],
        "pat": "repo-token",
    },
    "pat",
)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    from tests.test_daemon_service import FakeKeyring

    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
    monkeypatch.setitem(sys.modules, "keyring", FakeKeyring())
    monkeypatch.setattr("memforge.genes.atlassian_auth._ATLASSIAN_REQUEST_LIMITERS", {})
    monkeypatch.setattr("memforge.genes.toolkit.pacing._SHARED_LIMITERS", WeakValueDictionary())


@pytest.fixture
def jira(tmp_path, monkeypatch) -> Iterator[Deployment]:
    issue = jira_fakes._issue

    def rendered(key, zone):
        # Jira renders each issue's description as HTML beside its native text.
        answer = issue(key, zone)
        answer["renderedFields"] = {"description": f"<p>{answer['fields']['description']}</p>"}
        return answer

    monkeypatch.setattr(jira_fakes, "_issue", rendered)
    yield from _deployment(tmp_path, JIRA)


@pytest.fixture
def github(tmp_path) -> Iterator[Deployment]:
    yield from _deployment(tmp_path, GITHUB)


def _deployment(tmp_path, kind: DeviceType) -> Iterator[Deployment]:
    deployment = Deployment(tmp_path, kind)
    try:
        with TestClient(deployment.app) as client:
            deployment.client = client
            yield deployment
    finally:
        asyncio.run(deployment.db.close())


# ---------------------------------------------------------------------------
# The old package paths, as their daemons left them
# ---------------------------------------------------------------------------


def _package_path_jira_doc_id(source_id: str, issue_key: str) -> str:
    """The Document id Jira's package path gave an issue: the Source and the key, slugged and digested."""

    digest = hashlib.sha256(f"{source_id.strip()}|{issue_key.strip().upper()}".encode()).hexdigest()[:12]
    return "-".join(["jira", slugify(source_id)[:30], slugify(issue_key)[:30] or "issue", digest])


def _jira_package(source_id: str, stored, body: bytes) -> dict:
    """The package the daemon submitted for one issue: the issue as the Gene fetches it, wrapped."""

    payload = json.loads(body)
    key = str(payload["key"])
    return {
        "package_kind": "jira_document",
        "doc_id": stored.document_id,
        "issue_key": key,
        "base_url": jira_fakes.BASE_URL,
        "title": payload["fields"]["summary"],
        "source_url": f"{jira_fakes.BASE_URL}/browse/{key}",
        "version": payload["fields"]["updated"],
        "submitted_at": SUBMITTED_AT,
        "submitted_by": OWNER,
        "raw_payload": payload,
    }


def _github_package(source_id: str, stored, body: bytes) -> dict:
    """The package the daemon pushed for one file: its text, blob and commit time."""

    item = stored.item
    repo = parse_github_repo_url(github_repo_fakes.REPO_URL)
    return {
        "package_kind": "github_repo_document",
        "doc_id": stored.document_id,
        "title": item.extra["relative_path"],
        "source_url": item.source_url,
        "last_modified": SUBMITTED_AT,
        "version": item.extra["blob_sha"],
        "repo_url": repo["repo_url"],
        "repo_host": repo["host"],
        "repo_owner": repo["owner"],
        "repo_name": repo["repo"],
        "repo_ref": "main",
        "relative_path": item.extra["relative_path"],
        "blob_sha": item.extra["blob_sha"],
        "content_type": item.content_type,
        "markdown": body.decode(),
        "source_updated_at": item.extra[github_repo_gene.LAST_COMMIT_AT_KEY],
        "submitted_at": SUBMITTED_AT,
        "submitted_by": OWNER,
    }


PACKAGE_PATHS = {
    "jira": ({"auth_mode": "browser_cookie", "sync_mode": "local_agent"}, "local_agent_documents_dir", _jira_package),
    "github_repo": ({"connection_mode": "local_push"}, "documents_dir", _github_package),
}


async def _leave_on_the_package_path(db, doc_store, source_id: str, config: dict, inbox: str) -> None:
    """Store each Unit's input as the package its daemon submitted, and configure the Source's package path."""

    source = await db.get_source(source_id)
    path_fields, inbox_field, package = PACKAGE_PATHS[source["type"]]
    for doc_id in await db.list_indexed_doc_ids(source_id):
        unit = await db.find_source_unit_by_document_id(source_id, doc_id, current_only=True)
        stored = await db.get_source_unit_input(unit.id)
        body = doc_store.read_artifact(stored.raw_content_uri)
        packaged = package(source_id, stored, body)
        doc_store._resolve_artifact_path(stored.raw_content_uri).write_text(json.dumps(packaged))
    secret_free = {key: value for key, value in config.items() if key != "pat"}
    await db.upsert_source(
        id=source_id,
        type=source["type"],
        name=source["name"],
        config_json=json.dumps({**secret_free, **path_fields, inbox_field: inbox}),
        access_policy="workspace",
        owner_user_id=OWNER,
        created_by_user_id=OWNER,
        execution_owner_user_id=OWNER,
    )


def _on_the_package_path(deployment: Deployment, monkeypatch) -> str:
    """A Source its owner's daemon collected through its type's package path, as that path left it.

    The Source is synced from the fake provider, with Jira's issues named as
    the package path named them; each Unit's stored input then becomes the
    package the daemon submitted, and the configuration the package path's.
    """

    kind = deployment.kind
    source_id = f"src-{kind.source_type}"
    asyncio.run(
        deployment.db.upsert_source(
            id=source_id,
            type=kind.source_type,
            name="Team items",
            config_json=json.dumps(kind.config),
            access_policy="workspace",
            owner_user_id=OWNER,
            created_by_user_id=OWNER,
            execution_owner_user_id=OWNER,
        )
    )
    gene_cls = JiraGene if kind.source_type == "jira" else GitHubRepoGene
    with monkeypatch.context() as patch:
        if kind.source_type == "jira":
            patch.setattr(jira_gene, "_issue_doc_id", lambda key: _package_path_jira_doc_id(source_id, key))
        with deployment.serving():
            state = asyncio.run(
                _runtime(deployment).orchestrator().sync_gene(
                    gene=gene_cls(config=dict(kind.config), source_id=source_id),
                    source_name="Team items",
                    source_id=source_id,
                )
            )
    assert state.last_sync_status == "success", state.error_message
    asyncio.run(
        _leave_on_the_package_path(
            deployment.db, deployment.app.state.document_store, source_id, kind.config, f"/inbox/{source_id}"
        )
    )
    return source_id


def _runtime(deployment: Deployment) -> _Runtime:
    return _Runtime(deployment.db, deployment.app.state.document_store, deployment.extractor)


def _plan(deployment: Deployment, source_id: str):
    source = asyncio.run(deployment.db.get_source(source_id))
    return asyncio.run(plan_package_migration(deployment.db, deployment.app.state.document_store, source))


def _apply(deployment: Deployment, source_id: str):
    return asyncio.run(apply_package_migration(deployment.db, AppConfig(), source_id, runtime=_runtime(deployment)))


def _support_on(deployment: Deployment, doc_id: str) -> str:
    """A Memory the Source's extraction supports with ``doc_id``; returns its id."""

    memory = Memory(id=f"mem-{doc_id}", memory_type="fact", content=f"A fact of {doc_id}.", content_hash=f"h-{doc_id}")
    asyncio.run(deployment.db.insert_memory(memory))
    asyncio.run(
        deployment.db.add_memory_source(
            memory.id, doc_id, deployment.kind.source_type, "a fact", source_updated_at=datetime.now(timezone.utc)
        )
    )
    return memory.id


# ---------------------------------------------------------------------------
# Inventory and dry run
# ---------------------------------------------------------------------------


def test_the_inventory_names_only_sources_on_a_package_path(jira, monkeypatch):
    source_id = _on_the_package_path(jira, monkeypatch)
    asyncio.run(
        jira.db.upsert_source(
            id="src-server-jira",
            type="jira",
            name="Server Jira",
            config_json=json.dumps(JIRA.config),
            access_policy="workspace",
            owner_user_id=OWNER,
        )
    )

    assert [source["id"] for source in asyncio.run(package_path_sources(jira.db))] == [source_id]


def test_the_dry_run_rebuilds_every_jira_package_without_writing(jira, monkeypatch):
    source_id = _on_the_package_path(jira, monkeypatch)
    before = jira.documents(source_id)

    plan = _plan(jira, source_id)

    assert not plan.refused
    assert sorted(unit.document_id for unit in plan.units) == ["jira-PAY-1", "jira-PAY-2", "jira-PAY-3", "jira-PAY-4"]
    assert sorted(plan.moved_document_ids) == sorted(
        (_package_path_jira_doc_id(source_id, key), f"jira-{key}")
        for key in ("PAY-1", "PAY-2", "PAY-3", "PAY-4")
    )
    # The packages held the issues as the Gene fetches them: no Observation changes.
    assert plan.changed_document_ids == ()
    assert jira.documents(source_id) == before
    assert asyncio.run(jira.db.get_source(source_id))["executor"] == "server"


def test_the_dry_run_counts_and_apply_extracts_only_a_unit_whose_rewrite_changes_its_content(jira, monkeypatch):
    source_id = _on_the_package_path(jira, monkeypatch)
    old_pay_2 = _package_path_jira_doc_id(source_id, "PAY-2")

    def described_differently(package: dict) -> None:
        # PAY-2's stored package reads differently from its committed revision.
        text = "The rules PAY-2 records, as the package holds them."
        package["raw_payload"]["fields"]["description"] = text
        package["raw_payload"]["renderedFields"]["description"] = f"<p>{text}</p>"

    _rewrite_stored_package(jira, source_id, old_pay_2, described_differently)

    plan = _plan(jira, source_id)

    assert plan.changed_document_ids == (old_pay_2,)
    jira.extractor.fragment_calls.clear()
    assert _apply(jira, source_id).status == "migrated"
    assert jira.extracted_documents(source_id) == ["jira-PAY-2"]


def _rewrite_stored_package(deployment: Deployment, source_id: str, doc_id: str, change) -> None:
    """Change the package a Unit stored, as an earlier release may have stored it."""

    unit = asyncio.run(deployment.db.find_source_unit_by_document_id(source_id, doc_id, current_only=True))
    stored = asyncio.run(deployment.db.get_source_unit_input(unit.id))
    doc_store = deployment.app.state.document_store
    package = json.loads(doc_store.read_artifact(stored.raw_content_uri))
    change(package)
    doc_store._resolve_artifact_path(stored.raw_content_uri).write_text(json.dumps(package))


def test_the_dry_run_refuses_a_jira_package_without_its_update_time(jira, monkeypatch):
    source_id = _on_the_package_path(jira, monkeypatch)
    old_pay_3 = _package_path_jira_doc_id(source_id, "PAY-3")
    _rewrite_stored_package(jira, source_id, old_pay_3, lambda package: package["raw_payload"]["fields"].pop("updated"))

    plan = _plan(jira, source_id)

    assert plan.refused
    assert plan.unrebuildable == ((old_pay_3, "the package's issue has no fields.updated"),)
    result = _apply(jira, source_id)
    assert result.status == "refused"
    assert asyncio.run(jira.db.get_source(source_id))["config"]["sync_mode"] == "local_agent"
    assert old_pay_3 in jira.documents(source_id)


def test_the_dry_run_refuses_two_units_that_map_to_one_document(jira, monkeypatch):
    source_id = _on_the_package_path(jira, monkeypatch)
    old_pay_1, old_pay_2 = (_package_path_jira_doc_id(source_id, key) for key in ("PAY-1", "PAY-2"))
    # PAY-2's package names the issue PAY-1, as a device that read one key twice may have stored it.
    _rewrite_stored_package(jira, source_id, old_pay_2, lambda package: package["raw_payload"].update(key="PAY-1"))

    plan = _plan(jira, source_id)

    assert plan.refused
    assert plan.collisions == (("jira-PAY-1", tuple(sorted((old_pay_1, old_pay_2)))),)


def test_the_github_dry_run_shows_the_genes_ids_equal_the_stored_ones(github, monkeypatch):
    source_id = _on_the_package_path(github, monkeypatch)

    plan = _plan(github, source_id)

    assert not plan.refused
    assert plan.moved_document_ids == ()
    assert sorted(unit.document_id for unit in plan.units) == sorted(github.documents(source_id))
    assert plan.changed_document_ids == ()


def test_the_github_dry_run_refuses_a_file_the_configured_repository_names_differently(github, monkeypatch):
    source_id = _on_the_package_path(github, monkeypatch)
    # The Source's configuration now names another ref than its packages were pushed at.
    source = asyncio.run(github.db.get_source(source_id))
    asyncio.run(
        github.db.upsert_source(
            id=source_id,
            type="github_repo",
            name=source["name"],
            config_json=json.dumps({**source["config"], "ref": "release"}),
            access_policy="workspace",
            owner_user_id=OWNER,
        )
    )

    plan = _plan(github, source_id)

    assert plan.refused
    assert len(plan.id_changes_refused) == len(github.documents(source_id))


# ---------------------------------------------------------------------------
# Apply, recovery and the first device run
# ---------------------------------------------------------------------------


def test_a_moved_jira_source_keeps_its_support_and_extracts_only_the_issue_that_changed(jira, monkeypatch):
    source_id = _on_the_package_path(jira, monkeypatch)
    old_pay_1 = _package_path_jira_doc_id(source_id, "PAY-1")
    memory_id = _support_on(jira, old_pay_1)
    cursor = asyncio.run(jira.db.get_sync_state(source_id)).last_sync_at
    jira.extractor.fragment_calls.clear()

    result = _apply(jira, source_id)

    assert result.status == "migrated", result
    assert jira.extracted_documents(source_id) == []
    assert sorted(jira.documents(source_id)) == ["jira-PAY-1", "jira-PAY-2", "jira-PAY-3", "jira-PAY-4"]
    assert [source.doc_id for source in asyncio.run(jira.db.get_memory_sources(memory_id))] == ["jira-PAY-1"]
    moved = asyncio.run(jira.db.get_source(source_id))
    assert moved["executor"] == "device"
    assert "sync_mode" not in moved["config"] and "local_agent_documents_dir" not in moved["config"]
    # The replay kept the cursor and the Source is no longer on a package path.
    assert asyncio.run(jira.db.get_sync_state(source_id)).last_sync_at == cursor
    assert asyncio.run(package_path_sources(jira.db)) == []

    # PAY-2 is edited after the migration; nothing else changes.
    from memforge.local_agent.device_credentials import device_credentials

    # The package path signed in with the browser session; the owner signs in on the device.
    assert moved["config"]["auth_mode"] == "browser_cookie"
    device_credentials(DAEMON_SERVER).hold_session(source_id, "jira", jira_fakes.BASE_URL, "JSESSIONID=device")
    _jira_accepts_the_device_session(monkeypatch)
    issue_id, labels, _updated = jira_fakes.ISSUES["PAY-2"]
    monkeypatch.setitem(jira_fakes.ISSUES, "PAY-2", (issue_id, labels, datetime.now(timezone.utc)))
    issue = jira_fakes._issue

    def edited(key, zone):
        answer = issue(key, zone)
        if key == "PAY-2":
            answer["fields"]["description"] = "The rules PAY-2 records, revised."
            answer["renderedFields"]["description"] = "<p>The rules PAY-2 records, revised.</p>"
        return answer

    monkeypatch.setattr(jira_fakes, "_issue", edited)
    _sync_now_on_the_device(jira, source_id)

    assert jira.extracted_documents(source_id) == ["jira-PAY-2"]
    assert sorted(jira.documents(source_id)) == ["jira-PAY-1", "jira-PAY-2", "jira-PAY-3", "jira-PAY-4"]


def _jira_accepts_the_device_session(monkeypatch) -> None:
    from dataclasses import replace

    from memforge.auth import browser_session

    async def accept(origin, cookie_header, tls_config):
        return {"name": OWNER}

    browser_session.ensure_builtin_providers()
    provider = browser_session.get_provider("jira")
    monkeypatch.setitem(browser_session._PROVIDERS, "jira", replace(provider, validate_session=accept))


def test_a_moved_github_local_push_source_extracts_only_the_file_that_changed(github, monkeypatch):
    from tests.github_api_fixture import git_blob_sha
    from memforge.local_agent.device_credentials import device_credentials

    source_id = _on_the_package_path(github, monkeypatch)
    held = github.documents(source_id)
    github.extractor.fragment_calls.clear()

    result = _apply(github, source_id)

    assert result.status == "migrated", result
    assert github.extracted_documents(source_id) == []
    assert sorted(github.documents(source_id)) == sorted(held)
    moved = asyncio.run(github.db.get_source(source_id))
    assert moved["executor"] == "device"
    assert "connection_mode" not in moved["config"] and "documents_dir" not in moved["config"]

    device_credentials(DAEMON_SERVER).set(source_id, {"pat": "repo-token"})
    edited = {**github_repo_fakes.FILES, "docs/guide.md": b"# Guide\n\nRestart the queue before a release.\n"}
    monkeypatch.setattr(github_repo_fakes, "FILES", edited)
    github.provider = github_repo_fakes._Provider(ProviderListing.COMPLETE)
    _sync_now_on_the_device(github, source_id)

    guide = build_github_repo_doc_id(
        source_id=source_id, repo_url=github_repo_fakes.REPO_URL, repo_ref="main", relative_path="docs/guide.md"
    )
    assert github.extracted_documents(source_id) == [guide]
    assert github.documents(source_id)[guide][1] == git_blob_sha(edited["docs/guide.md"])


def test_an_interrupted_migration_continues_where_it_stopped(jira, monkeypatch):
    source_id = _on_the_package_path(jira, monkeypatch)
    jira.extractor.fragment_calls.clear()
    from memforge.pipeline import sync as sync_module

    processed = sync_module.GeneSyncOrchestrator._process_item
    calls = {"count": 0}

    async def interrupted(self, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 3:
            raise RuntimeError("the process stopped")
        return await processed(self, *args, **kwargs)

    monkeypatch.setattr(sync_module.GeneSyncOrchestrator, "_process_item", interrupted)
    monkeypatch.setattr(sync_module, "MAX_RETRIES", 1)

    first = _apply(jira, source_id)

    assert first.status == "replay_failed"
    # The Source keeps its package path; two issues already moved.
    assert asyncio.run(jira.db.get_source(source_id))["config"]["sync_mode"] == "local_agent"
    remaining = _plan(jira, source_id)
    assert not remaining.refused
    # Only the issues the first replay did not commit are planned again.
    assert remaining.units and remaining.current_units and len(remaining.units) + remaining.current_units == 4

    monkeypatch.setattr(sync_module.GeneSyncOrchestrator, "_process_item", processed)
    second = _apply(jira, source_id)

    assert second.status == "migrated", second
    assert sorted(jira.documents(source_id)) == ["jira-PAY-1", "jira-PAY-2", "jira-PAY-3", "jira-PAY-4"]
    assert jira.extracted_documents(source_id) == []
    assert asyncio.run(jira.db.get_source(source_id))["executor"] == "device"


def test_a_source_left_on_the_package_path_does_not_sync_until_it_moves(jira, monkeypatch):
    """Neither Sync now nor its schedule runs the Gene over a Source whose Units hold packages."""

    from datetime import timedelta

    from memforge.runtime import SyncService
    from memforge.scheduler import SyncScheduler

    source_id = _on_the_package_path(jira, monkeypatch)
    held = jira.documents(source_id)
    jira.extractor.fragment_calls.clear()

    queued = jira.client.post(f"/api/v1/sources/{source_id}/sync")
    assert queued.status_code == 202, queued.text
    jira.sync()
    due = datetime.now(timezone.utc) - timedelta(minutes=1)
    asyncio.run(jira.db.set_source_sync_schedule(source_id, enabled=True, interval_minutes=60, next_run_at=due))
    scheduler = SyncScheduler(
        jira.db,
        SyncService(jira.db, AppConfig(), workspace_id="default"),
        document_store=jira.app.state.document_store,
    )
    asyncio.run(scheduler._sync_due_sources())
    jira.sync()

    # Both runs failed before the Gene reached Jira; no jira-<KEY> Document appeared.
    assert [status for status, _ in jira.runs(source_id)][-2:] == ["failed", "failed"]
    # The Source shows the command that moves it.
    listed = jira.client.get("/api/v1/sources")
    assert listed.status_code == 200, listed.text
    [shown] = [source for source in listed.json()["data"] if source["id"] == source_id]
    assert shown["sync"]["status"] == "failed"
    assert "memforge maintenance migrate-device-packages" in shown["sync"]["error_message"]
    assert jira.documents(source_id) == held
    assert jira.extracted_documents(source_id) == []

    assert _apply(jira, source_id).status == "migrated"
    assert sorted(jira.documents(source_id)) == ["jira-PAY-1", "jira-PAY-2", "jira-PAY-3", "jira-PAY-4"]


# ---------------------------------------------------------------------------
# The command on a local installation
# ---------------------------------------------------------------------------


@pytest.fixture
def installation(tmp_path, monkeypatch):
    """A local installation whose GitHub repository Source is on the local push path; yields its config."""

    from rich.console import Console

    from memforge.storage.database import Database
    from memforge.storage.document_store import LocalDocumentStore
    from tests.test_sync_bookkeeping import ProjectionFragmentRecordingExtractor

    monkeypatch.setenv("MEMFORGE_BASE_DIR", str(tmp_path / "mem"))
    for name in ("MEMFORGE_STORAGE_DB_PATH", "MEMFORGE_STORAGE_CHROMA_PATH", "MEMFORGE_STORAGE_DOCS_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("memforge.main.console", Console(width=400))
    config = AppConfig()
    extractor = ProjectionFragmentRecordingExtractor()

    async def build_sync_runtime(self, db, config, **kwargs):
        return _Runtime(db, LocalDocumentStore(config.storage.docs_path), extractor)

    monkeypatch.setattr("memforge.runtime.DefaultRuntimeProvider.build_sync_runtime", build_sync_runtime)
    provider = github_repo_fakes._Provider(ProviderListing.COMPLETE)

    async def install() -> None:
        db = Database(str(config.storage.db_path))
        await db.connect()
        doc_store = LocalDocumentStore(config.storage.docs_path)
        try:
            await db.upsert_source(
                id="src-github",
                type="github_repo",
                name="Architecture",
                config_json=json.dumps(GITHUB.config),
                access_policy="workspace",
                owner_user_id=OWNER,
                execution_owner_user_id=OWNER,
            )
            runtime = _Runtime(db, doc_store, extractor)
            state = await runtime.orchestrator().sync_gene(
                gene=GitHubRepoGene(config=dict(GITHUB.config), source_id="src-github"),
                source_name="Architecture",
                source_id="src-github",
            )
            assert state.last_sync_status == "success", state.error_message
            await _leave_on_the_package_path(db, doc_store, "src-github", GITHUB.config, str(tmp_path / "inbox"))
        finally:
            await db.close()

    with provider.http.serving(github_repo_gene):
        asyncio.run(install())
    return config


def _installed_source(config: AppConfig) -> dict:
    from memforge.storage.database import Database

    async def read() -> dict:
        db = Database(str(config.storage.db_path))
        await db.connect()
        try:
            return await db.get_source("src-github")
        finally:
            await db.close()

    return asyncio.run(read())


def test_the_command_checks_then_applies_after_taking_a_recovery_point(installation):
    from click.testing import CliRunner

    from memforge.main import cli
    from memforge.storage.recovery_points import PRE_MIGRATION, RecoveryPoints

    checked = CliRunner().invoke(cli, ["maintenance", "migrate-device-packages", "--json"])

    assert checked.exit_code == 0, checked.output
    [row] = json.loads(checked.output)["sources"]
    assert (row["source_id"], row["units"], row["refused"]) == ("src-github", 3, False)
    assert _installed_source(installation)["executor"] == "server"
    assert RecoveryPoints.for_config(installation).list() == []

    applied = CliRunner().invoke(cli, ["maintenance", "migrate-device-packages", "--apply"])

    assert applied.exit_code == 0, applied.output
    [point] = RecoveryPoints.for_config(installation).list()
    assert point.reason == PRE_MIGRATION
    assert f"memforge backup restore {point.id}" in applied.output
    assert "moved to the device" in applied.output
    moved = _installed_source(installation)
    assert moved["executor"] == "device" and "connection_mode" not in moved["config"]
    again = CliRunner().invoke(cli, ["maintenance", "migrate-device-packages", "--apply"])
    assert again.exit_code == 0 and "No Source is on a device package path" in again.output


def test_the_command_applies_nothing_while_memforge_has_the_data_open(installation):
    from click.testing import CliRunner

    from memforge.main import cli
    from memforge.storage.database import Database
    from memforge.storage.recovery_points import RecoveryPoints

    async def hold_then_apply():
        running = Database(str(installation.storage.db_path), recovery_points=RecoveryPoints.for_config(installation))
        await running.connect()
        try:
            return await asyncio.to_thread(
                CliRunner().invoke, cli, ["maintenance", "migrate-device-packages", "--apply"]
            )
        finally:
            await running.close()

    result = asyncio.run(hold_then_apply())

    assert result.exit_code == 1
    assert "Another MemForge process has the data" in " ".join(result.output.split())
    assert _installed_source(installation)["executor"] == "server"
    assert RecoveryPoints.for_config(installation).list() == []


def test_the_check_names_each_cause_that_refuses_a_source(installation):
    from click.testing import CliRunner

    from memforge.main import cli

    # The Source now names another repository than the files it holds.
    from memforge.storage.database import Database

    async def rename_repository() -> None:
        db = Database(str(installation.storage.db_path))
        await db.connect()
        try:
            source = await db.get_source("src-github")
            config = {**source["config"], "repo_url": "https://github.example.com/acme/renamed"}
            await db.upsert_source(
                id="src-github",
                type="github_repo",
                name=source["name"],
                config_json=json.dumps(config),
                access_policy="workspace",
                owner_user_id=OWNER,
                execution_owner_user_id=OWNER,
            )
        finally:
            await db.close()

    asyncio.run(rename_repository())

    checked = CliRunner().invoke(cli, ["maintenance", "migrate-device-packages"])

    assert checked.exit_code == 1, checked.output
    assert "3 file(s) the configured repository names differently" in checked.output
    assert "not rebuildable" not in checked.output and "collision" not in checked.output
