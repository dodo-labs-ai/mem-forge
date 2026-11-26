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
from pathlib import Path
from weakref import WeakValueDictionary

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.genes import github_repo_gene, jira_gene
from memforge.genes.github_repo_gene import GitHubRepoGene
from memforge.genes.jira_gene import JiraGene
from memforge.github_repo_utils import build_github_repo_doc_id, github_content_type, parse_github_repo_url
from memforge.local_agent.document_identity import build_jira_doc_id
from memforge.models import Memory
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
    {"base_url": jira_fakes.BASE_URL, "auth_mode": "pat", "projects": ["PAY"], "issue_types": ["Story"]},
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
    },
    "pat",
)
JIRA_TOKEN = "token"


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


def _jira_packages(deployment: Deployment, source_id: str) -> dict[str, dict]:
    """What the daemon submitted for each PAY issue: the issue as the Gene fetches it, in a package."""

    async def fetched() -> dict[str, dict]:
        async with jira_fakes.FAKES.connect({**deployment.kind.config, "pat": JIRA_TOKEN}) as connection:
            gene = connection.gene
            payloads = {}
            async for item in gene.discover(None):
                payloads[item.extra["issue_key"]] = json.loads((await gene.fetch(item)).body)
            return payloads

    with deployment.serving():
        payloads = asyncio.run(fetched())
    return {
        key: {
            "package_kind": "jira_document",
            "doc_id": build_jira_doc_id(source_id=source_id, issue_key=key),
            "issue_key": key,
            "base_url": jira_fakes.BASE_URL,
            "title": payload["fields"]["summary"],
            "source_url": f"{jira_fakes.BASE_URL}/browse/{key}",
            "version": payload["fields"]["updated"],
            "submitted_at": SUBMITTED_AT,
            "submitted_by": OWNER,
            "raw_payload": payload,
        }
        for key, payload in payloads.items()
    }


def _github_packages(source_id: str) -> dict[str, dict]:
    """What the daemon pushed for each selected file: its text, blob and commit time."""

    from tests.github_api_fixture import git_blob_sha

    repo = parse_github_repo_url(github_repo_fakes.REPO_URL)
    packages = {}
    for path, body in github_repo_fakes.FILES.items():
        if not path.startswith("docs/") or not path.endswith(".md"):
            continue
        packages[path] = {
            "package_kind": "github_repo_document",
            "doc_id": build_github_repo_doc_id(
                source_id=source_id, repo_url=repo["repo_url"], repo_ref="main", relative_path=path
            ),
            "title": path,
            "source_url": f"{repo['repo_url']}/blob/main/{path}",
            "last_modified": SUBMITTED_AT,
            "version": git_blob_sha(body),
            "repo_url": repo["repo_url"],
            "repo_host": repo["host"],
            "repo_owner": repo["owner"],
            "repo_name": repo["repo"],
            "repo_ref": "main",
            "relative_path": path,
            "blob_sha": git_blob_sha(body),
            "content_type": github_content_type(path),
            "markdown": body.decode(),
            "source_updated_at": "2026-02-01T00:00:00+00:00",
            "submitted_at": SUBMITTED_AT,
            "submitted_by": OWNER,
        }
    return packages


def _on_the_package_path(deployment: Deployment, tmp_path: Path, packages_for) -> tuple[str, Path]:
    """Create the Source on its type's package path and sync what its daemon submitted to its inbox."""

    kind = deployment.kind
    source_id = f"src-{kind.source_type}"
    inbox = tmp_path / "inbox" / source_id
    inbox.mkdir(parents=True)
    path_fields = (
        {"sync_mode": "local_agent", "local_agent_documents_dir": str(inbox)}
        if kind.source_type == "jira"
        else {"connection_mode": "local_push", "documents_dir": str(inbox)}
    )
    asyncio.run(
        deployment.db.upsert_source(
            id=source_id,
            type=kind.source_type,
            name="Team items",
            config_json=json.dumps({**kind.config, **path_fields}),
            access_policy="workspace",
            owner_user_id=OWNER,
            created_by_user_id=OWNER,
            execution_owner_user_id=OWNER,
        )
    )
    _submit(inbox, packages_for(source_id))
    _sync_the_inbox(deployment, source_id)
    return source_id, inbox


def _submit(inbox: Path, packages: dict[str, dict]) -> None:
    for name, package in packages.items():
        (inbox / f"{hashlib.sha256(name.encode()).hexdigest()[:12]}.json").write_text(json.dumps(package))


def _sync_the_inbox(deployment: Deployment, source_id: str) -> None:
    source = asyncio.run(deployment.db.get_source(source_id))
    gene_cls = JiraGene if source["type"] == "jira" else GitHubRepoGene
    runtime = _Runtime(deployment.db, deployment.app.state.document_store, deployment.extractor)
    state = asyncio.run(
        runtime.orchestrator().sync_gene(
            gene=gene_cls(config=dict(source["config"]), source_id=source_id),
            source_name=source["name"],
            source_id=source_id,
        )
    )
    assert state.last_sync_status == "success", state.error_message


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


def test_the_inventory_names_only_sources_on_a_package_path(jira, tmp_path):
    source_id, _inbox = _on_the_package_path(jira, tmp_path, lambda source_id: _jira_packages(jira, source_id))
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


def test_the_dry_run_rebuilds_every_jira_package_without_writing(jira, tmp_path):
    source_id, _inbox = _on_the_package_path(jira, tmp_path, lambda source_id: _jira_packages(jira, source_id))
    before = jira.documents(source_id)

    plan = _plan(jira, source_id)

    assert not plan.refused
    assert sorted(unit.document_id for unit in plan.units) == ["jira-PAY-1", "jira-PAY-2", "jira-PAY-3", "jira-PAY-4"]
    assert sorted(plan.moved_document_ids) == sorted(
        (build_jira_doc_id(source_id=source_id, issue_key=key), f"jira-{key}")
        for key in ("PAY-1", "PAY-2", "PAY-3", "PAY-4")
    )
    # The packages held the issues as the Gene fetches them: no Observation changes.
    assert plan.changed_document_ids == ()
    assert jira.documents(source_id) == before
    assert asyncio.run(jira.db.get_source(source_id))["executor"] == "server"


def test_the_dry_run_counts_and_apply_extracts_only_a_unit_whose_rewrite_changes_its_content(jira, tmp_path):
    source_id, _inbox = _on_the_package_path(jira, tmp_path, lambda source_id: _jira_packages(jira, source_id))
    old_pay_2 = build_jira_doc_id(source_id=source_id, issue_key="PAY-2")

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


def test_the_dry_run_refuses_a_jira_package_without_its_update_time(jira, tmp_path):
    source_id, _inbox = _on_the_package_path(jira, tmp_path, lambda source_id: _jira_packages(jira, source_id))
    old_pay_3 = build_jira_doc_id(source_id=source_id, issue_key="PAY-3")
    _rewrite_stored_package(jira, source_id, old_pay_3, lambda package: package["raw_payload"]["fields"].pop("updated"))

    plan = _plan(jira, source_id)

    assert plan.refused
    assert plan.unrebuildable == ((old_pay_3, "the package's issue has no fields.updated"),)
    result = _apply(jira, source_id)
    assert result.status == "refused"
    assert asyncio.run(jira.db.get_source(source_id))["config"]["sync_mode"] == "local_agent"
    assert old_pay_3 in jira.documents(source_id)


def test_the_dry_run_refuses_two_units_that_map_to_one_document(jira, tmp_path):
    def packages(source_id: str) -> dict[str, dict]:
        built = _jira_packages(jira, source_id)
        # Two packages of one key under ids the device built differently.
        twin = json.loads(json.dumps(built["PAY-1"]))
        twin["doc_id"] = build_jira_doc_id(source_id=source_id, issue_key="PAY-1-twin")
        twin["raw_payload"]["id"] = "200001"
        built["PAY-1-twin"] = twin
        return built

    source_id, _inbox = _on_the_package_path(jira, tmp_path, packages)

    plan = _plan(jira, source_id)

    assert plan.refused
    assert plan.collisions == (
        (
            "jira-PAY-1",
            tuple(
                sorted(build_jira_doc_id(source_id=source_id, issue_key=key) for key in ("PAY-1", "PAY-1-twin"))
            ),
        ),
    )


def test_the_github_dry_run_shows_the_genes_ids_equal_the_stored_ones(github, tmp_path):
    source_id, _inbox = _on_the_package_path(github, tmp_path, _github_packages)

    plan = _plan(github, source_id)

    assert not plan.refused
    assert plan.moved_document_ids == ()
    assert sorted(unit.document_id for unit in plan.units) == sorted(github.documents(source_id))
    assert plan.changed_document_ids == ()


def test_the_github_dry_run_refuses_a_file_the_configured_repository_names_differently(github, tmp_path):
    source_id, _inbox = _on_the_package_path(github, tmp_path, _github_packages)
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


def test_a_moved_jira_source_keeps_its_support_and_extracts_only_the_issue_that_changed(jira, tmp_path, monkeypatch):
    source_id, _inbox = _on_the_package_path(jira, tmp_path, lambda source_id: _jira_packages(jira, source_id))
    old_pay_1 = build_jira_doc_id(source_id=source_id, issue_key="PAY-1")
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

    device_credentials(DAEMON_SERVER).set(source_id, {"pat": JIRA_TOKEN})
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


def test_a_moved_github_local_push_source_extracts_only_the_file_that_changed(github, tmp_path, monkeypatch):
    from tests.github_api_fixture import git_blob_sha
    from memforge.local_agent.device_credentials import device_credentials

    source_id, _inbox = _on_the_package_path(github, tmp_path, _github_packages)
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


def test_an_interrupted_migration_continues_where_it_stopped(jira, tmp_path, monkeypatch):
    source_id, _inbox = _on_the_package_path(jira, tmp_path, lambda source_id: _jira_packages(jira, source_id))
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
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _submit(inbox, _github_packages("src-github"))

    async def install() -> None:
        db = Database(str(config.storage.db_path))
        await db.connect()
        try:
            await db.upsert_source(
                id="src-github",
                type="github_repo",
                name="Architecture",
                config_json=json.dumps({**GITHUB.config, "connection_mode": "local_push", "documents_dir": str(inbox)}),
                access_policy="workspace",
                owner_user_id=OWNER,
                execution_owner_user_id=OWNER,
            )
            runtime = _Runtime(db, LocalDocumentStore(config.storage.docs_path), extractor)
            source = await db.get_source("src-github")
            state = await runtime.orchestrator().sync_gene(
                gene=GitHubRepoGene(config=dict(source["config"]), source_id="src-github"),
                source_name="Architecture",
                source_id="src-github",
            )
            assert state.last_sync_status == "success", state.error_message
        finally:
            await db.close()

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
