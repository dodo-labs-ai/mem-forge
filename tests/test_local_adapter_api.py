"""Tests for the local CLI adapter document push contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.local_agent.source_contract import LocalAgentLease, source_with_sync_inputs
from memforge.models import ContentItem, require_attested_content
from memforge.storage.database import Database
from memforge.storage.document_store import LocalDocumentStore


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.server.jwt_secret = "test-secret"
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    # These tests exercise the adapter HTTP/package boundary. Background
    # scheduling and durable-worker behavior have dedicated contract suites;
    # starting them here adds unrelated concurrent database activity and makes
    # TestClient shutdown nondeterministic on Linux runners.
    cfg.sync.scheduler_enabled = False
    cfg.sync.worker_enabled = False
    return cfg


def _connect_database(tmp_path: Path) -> Database:
    database = Database(str(tmp_path / "api.db"))
    asyncio.run(database.connect())
    return database


def _run_artifact_cleanup(database: Database, cfg: AppConfig) -> None:
    """Run the periodic artifact cleanup the scheduler drives."""
    from memforge.storage.source_cleanup import SourceArtifactCleanupService

    asyncio.run(SourceArtifactCleanupService(database, LocalDocumentStore(cfg.storage.docs_path)).run_pending())


def _project_source_inputs(database: Database, source: dict) -> dict:
    inputs = asyncio.run(database.list_source_sync_inputs(workspace_id="local", source_id=source["id"]))
    return source_with_sync_inputs(source, inputs)


def _create_local_markdown_source(
    client: TestClient, *, name: str = "Engineering notes", vault_id: str = "engineering"
) -> dict:
    response = client.post(
        "/api/v1/sources",
        json={
            "type": "local_markdown",
            "name": name,
            "access_policy": "private",
            "config": {
                "root": "/Users/test/engineering-notes",
                "vault_id": vault_id,
                "display_label": "Engineering notes",
            },
            "project_binding": {"mode": "fixed", "project_key": vault_id.upper()},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _create_unmapped_local_markdown_source(
    client: TestClient, *, name: str = "Engineering notes", vault_id: str = "engineering"
) -> dict:
    response = client.post(
        "/api/v1/sources",
        json={
            "type": "local_markdown",
            "name": name,
            "access_policy": "private",
            "config": {
                "root": "/Users/test/engineering-notes",
                "vault_id": vault_id,
                "display_label": "Engineering notes",
            },
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def _create_teams_source(client: TestClient, *, name: str = "Teams Channel") -> dict:
    response = client.post(
        "/api/v1/sources",
        json={
            "type": "teams",
            "name": name,
            "access_policy": "private",
            "config": {
                "region": "emea",
                "channels": ["19:channel@example.test"],
                "conversation_gap_minutes": 60,
            },
            "project_binding": {"mode": "fixed", "project_key": "TEAMS"},
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_create_local_markdown_source_populates_inbox_path(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            row = asyncio.run(database.get_source(source_id))
        assert row is not None
        assert row["type"] == "local_markdown"
        config = row["config"]
        assert config["vault_id"] == "engineering"
        documents_dir = Path(config["documents_dir"])
        assert documents_dir.exists()
        expected_root = Path(cfg.storage.docs_path).parent / "local-adapter-submissions"
        assert documents_dir.is_relative_to(expected_root)
    finally:
        asyncio.run(database.close())


def test_create_local_markdown_source_generates_internal_vault_id(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            response = client.post(
                "/api/v1/sources",
                json={
                    "type": "local_markdown",
                    "name": "Engineering notes",
                    "access_policy": "private",
                    "config": {
                        "root": "/Users/test/engineering-notes",
                        "display_label": "Engineering notes",
                    },
                },
            )
            assert response.status_code == 200, response.text
            source_id = response.json()["id"]
            row = asyncio.run(database.get_source(source_id))
        assert row is not None
        config = row["config"]
        assert config["vault_id"].startswith("local-")
        assert config["display_label"] == "Engineering notes"
        assert Path(config["documents_dir"]).exists()
    finally:
        asyncio.run(database.close())


def test_update_local_markdown_source_preserves_internal_vault_id(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client, vault_id="engineering")
            response = client.put(
                f"/api/v1/sources/{created['id']}",
                json={
                    "name": "Engineering docs",
                    "config": {
                        "root": "/Users/test/engineering-docs",
                        "display_label": "Engineering docs",
                    },
                },
            )
            assert response.status_code == 200, response.text
            row = asyncio.run(database.get_source(created["id"]))
        assert row is not None
        assert row["config"]["vault_id"] == "engineering"
        assert row["config"]["display_label"] == "Engineering docs"
    finally:
        asyncio.run(database.close())


def test_local_adapter_document_push_writes_package(tmp_path):
    from memforge.genes.local_markdown_gene import LocalMarkdownGene
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            response = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "decisions/cutoff.md",
                    "title": "Cutoff Decision",
                    "markdown_body": "# Cutoff\n\nDeploy on Tuesday.",
                    "raw_hash": "deadbeef",
                    "submitted_by": "cli-adapter",
                    "process_now": False,
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["source_id"] == source_id
        assert body["doc_id"].startswith("local-md-")
        assert body["package_uri"]
        assert body["sync_started"] is False
        assert body["relative_path"] == "decisions/cutoff.md"

        assert body["package_path"] is None
        package_path = Path(body["package_uri"])
        package = json.loads(package_path.read_text())
        assert package["package_kind"] == "local_markdown_document"
        assert package["vault_id"] == "engineering"
        assert package["relative_path"] == "decisions/cutoff.md"
        assert package["title"] == "Cutoff Decision"
        assert package["markdown"].startswith("# Cutoff")

        row = asyncio.run(database.get_source(source_id))
        assert row is not None
        inputs = asyncio.run(database.list_source_sync_inputs(workspace_id="local", source_id=source_id))
        assert len(inputs) == 1
        assert inputs[0].raw_uri == body["package_uri"]
        assert inputs[0].metadata["doc_id"] == body["doc_id"]
        assert inputs[0].metadata["manifest_entry"]["doc_id"] == body["doc_id"]
        assert inputs[0].metadata["package_sha256"] == body["package_sha256"]
        assert inputs[0].metadata["manifest_entry"]["package_sha256"] == body["package_sha256"]

        projected = _project_source_inputs(database, row)
        gene = LocalMarkdownGene(projected["config"], source_id)
        gene.bind_document_store(LocalDocumentStore(cfg.storage.docs_path))

        async def _read_package():
            await gene.authenticate()
            items = [item async for item in gene.discover(None)]
            raw = await gene.fetch(items[0])
            normalized = await gene.normalize(raw)
            return items, normalized

        items, normalized = asyncio.run(_read_package())
        assert items[0].extra["package_uri"] == body["package_uri"]
        assert normalized.markdown_body.startswith("# Cutoff")
    finally:
        asyncio.run(database.close())


def test_a_stored_local_markdown_package_is_read_by_its_own_kind(tmp_path):
    from memforge.genes.local_markdown_gene import LocalMarkdownGene

    gene = LocalMarkdownGene({"documents_dir": str(tmp_path)}, "src-local")
    item = ContentItem(
        item_id="local-md-empty", title="Empty", source_url="", last_modified=datetime.now(timezone.utc),
    )
    package = {
        "package_kind": "local_markdown_document",
        "relative_path": "notes/empty.md",
        "content_type": "text/markdown",
        "markdown": "",
    }

    stored = gene.raw_from_stored_input(item, json.dumps(package).encode("utf-8"), "application/json")
    normalized = asyncio.run(gene.normalize(stored))

    require_attested_content(stored, normalized)
    assert stored.empty_evidence == "local_markdown_package_attested_empty_file"
    other_kind = gene.raw_from_stored_input(
        item, json.dumps({**package, "package_kind": "github_repo_document"}).encode("utf-8"), "application/json",
    )
    assert (other_kind.authoritative_empty, other_kind.empty_evidence) == (False, None)
    with pytest.raises(ValueError, match="not a local_markdown_document package"):
        asyncio.run(gene.normalize(other_kind))


def test_duplicate_local_package_attests_the_retained_artifact_not_the_new_upload(
    tmp_path,
):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            first = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "decisions/retained.md",
                    "markdown_body": "# Decision\n\nKeep the retained artifact.",
                    "submitted_at": "2026-07-15T10:00:00+00:00",
                    "process_now": False,
                },
            )
            assert first.status_code == 200, first.text
            retained_uri = first.json()["package_uri"]
            retained_sha = hashlib.sha256(Path(retained_uri).read_bytes()).hexdigest()
            [legacy_input] = asyncio.run(
                database.list_source_sync_inputs(workspace_id="local", source_id=source_id)
            )
            legacy_metadata = dict(legacy_input.metadata)
            legacy_metadata.pop("package_sha256")
            manifest_entry = dict(legacy_metadata["manifest_entry"])
            manifest_entry.pop("package_sha256", None)
            legacy_metadata["manifest_entry"] = manifest_entry
            asyncio.run(
                database.db.execute(
                    "UPDATE source_sync_inputs SET metadata_json = ? WHERE input_id = ?",
                    (json.dumps(legacy_metadata, sort_keys=True), legacy_input.input_id),
                )
            )
            asyncio.run(database.db.commit())

            duplicate = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "decisions/retained.md",
                    "markdown_body": "# Decision\n\nKeep the retained artifact.",
                    "submitted_at": "2026-07-16T10:00:00+00:00",
                    "process_now": False,
                },
            )

        assert duplicate.status_code == 200, duplicate.text
        assert duplicate.json()["package_uri"] == retained_uri
        assert duplicate.json()["package_sha256"] == retained_sha
        [attested] = asyncio.run(database.list_source_sync_inputs(workspace_id="local", source_id=source_id))
        assert attested.input_id == legacy_input.input_id
        assert attested.raw_uri == retained_uri
        assert attested.metadata["package_sha256"] == retained_sha
        assert attested.metadata["manifest_entry"]["package_sha256"] == retained_sha
        assert Path(retained_uri).exists()
        _run_artifact_cleanup(database, cfg)
        assert list(Path(cfg.storage.docs_path).rglob("*package*.json")) == [
            Path(retained_uri)
        ]
        assert asyncio.run(database.list_source_artifact_cleanup_tasks()) == []
    finally:
        asyncio.run(database.close())


@pytest.mark.parametrize("retained_failure", ["corrupt", "missing"])
def test_duplicate_local_package_does_not_attest_an_invalid_retained_artifact(
    tmp_path,
    retained_failure,
):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            first = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "decisions/corrupt.md",
                    "markdown_body": "# Original",
                    "submitted_at": "2026-07-15T10:00:00+00:00",
                    "process_now": False,
                },
            )
            assert first.status_code == 200, first.text
            retained_uri = first.json()["package_uri"]
            [legacy_input] = asyncio.run(
                database.list_source_sync_inputs(workspace_id="local", source_id=source_id)
            )
            legacy_metadata = dict(legacy_input.metadata)
            legacy_metadata.pop("package_sha256")
            manifest_entry = dict(legacy_metadata["manifest_entry"])
            manifest_entry.pop("package_sha256", None)
            legacy_metadata["manifest_entry"] = manifest_entry
            asyncio.run(
                database.db.execute(
                    "UPDATE source_sync_inputs SET metadata_json = ? WHERE input_id = ?",
                    (json.dumps(legacy_metadata, sort_keys=True), legacy_input.input_id),
                )
            )
            asyncio.run(database.db.commit())
            if retained_failure == "corrupt":
                Path(retained_uri).write_bytes(b"{}")
            else:
                Path(retained_uri).unlink()

            duplicate = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "decisions/corrupt.md",
                    "markdown_body": "# Original",
                    "submitted_at": "2026-07-16T10:00:00+00:00",
                    "process_now": False,
                },
            )

        assert duplicate.status_code == 409, duplicate.text
        assert duplicate.json()["detail"] == (
            "source_lifecycle_local_replay_artifact_invalid"
        )
        [unchanged] = asyncio.run(database.list_source_sync_inputs(workspace_id="local", source_id=source_id))
        assert "package_sha256" not in unchanged.metadata
        assert "package_sha256" not in unchanged.metadata["manifest_entry"]
        _run_artifact_cleanup(database, cfg)
        if retained_failure == "corrupt":
            assert Path(retained_uri).read_bytes() == b"{}"
            assert list(Path(cfg.storage.docs_path).rglob("*package*.json")) == [
                Path(retained_uri)
            ]
        else:
            assert not Path(retained_uri).exists()
            assert list(Path(cfg.storage.docs_path).rglob("*package*.json")) == []
        assert asyncio.run(database.list_source_artifact_cleanup_tasks()) == []
    finally:
        asyncio.run(database.close())


def test_normal_local_source_fetch_rejects_tampered_canonical_package(tmp_path):
    from memforge.genes.local_markdown_gene import LocalMarkdownGene
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            response = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "tampered.md",
                    "markdown_body": "# Original",
                    "process_now": False,
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        artifact = Path(body["package_uri"])
        artifact.write_bytes(artifact.read_bytes() + b"\n")
        source = asyncio.run(database.get_source(source_id))
        projected = _project_source_inputs(database, source)
        gene = LocalMarkdownGene(projected["config"], source_id)
        gene.bind_document_store(LocalDocumentStore(cfg.storage.docs_path))

        async def fetch_package():
            items = [item async for item in gene.discover(None)]
            return await gene.fetch(items[0])

        with pytest.raises(
            ValueError,
            match="source_lifecycle_local_replay_artifact_invalid",
        ):
            asyncio.run(fetch_package())
    finally:
        asyncio.run(database.close())


def test_local_file_package_attests_explicit_empty_content(tmp_path):
    from memforge.genes.local_markdown_gene import LocalMarkdownGene
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            payload = {
                "vault_id": "engineering",
                "relative_path": "emptied.md",
                "markdown_body": "",
                "process_now": False,
            }
            response = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json=payload,
            )
        assert response.status_code == 200, response.text
        source = asyncio.run(database.get_source(source_id))
        projected = _project_source_inputs(database, source)
        gene = LocalMarkdownGene(projected["config"], source_id)
        gene.bind_document_store(LocalDocumentStore(cfg.storage.docs_path))

        async def fetch_package():
            items = [item async for item in gene.discover(None)]
            raw = await gene.fetch(items[0])
            normalized = await gene.normalize(raw)
            return raw, normalized

        raw, normalized = asyncio.run(fetch_package())
        assert raw.body.strip()
        assert raw.authoritative_empty is True
        assert raw.empty_evidence
        assert normalized.markdown_body == ""
    finally:
        asyncio.run(database.close())


def test_local_markdown_rejects_implicit_binary_artifact_intake(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            response = client.post(
                f"/api/v1/sources/{source_id}/adapter/artifacts",
                params={
                    "source_unit_key": "notes/readme.md",
                    "provider_key": "diagram.png",
                    "provider_revision": "sha-image",
                    "parent_observation_type": "document_body",
                    "parent_provider_key": "body",
                    "filename": "diagram.png",
                    "media_type": "image/png",
                    "local_agent_job_id": "test-local-agent-job",
                    "local_agent_attempt_count": 1,
                },
                content=b"\x89PNG\r\n\x1a\nnot-authoritative",
                headers={"content-type": "image/png"},
            )

        assert response.status_code == 400
        assert response.json()["detail"] == "source does not support local Artifact inputs"
    finally:
        asyncio.run(database.close())


def test_file_package_without_source_time_is_uploaded_once_more_under_the_source_time_contract(tmp_path):
    from memforge.local_agent.document_identity import build_local_markdown_doc_id
    from memforge.local_agent.source_contract import LOCAL_PACKAGE_CONTRACT_VERSION
    from memforge.models import content_hash
    from memforge.server.admin_api import create_admin_app

    commit_time = "2026-03-02T09:30:00+00:00"
    markdown_body = "# Payroll Processing\n\nArchitecture notes."
    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            relative_path = "payroll/README.md"
            doc_id = build_local_markdown_doc_id(
                source_id=source_id,
                vault_id="engineering",
                relative_path=relative_path,
            )
            revision = content_hash(markdown_body)
            package = {
                "vault_id": "engineering",
                "relative_path": relative_path,
                "markdown_body": markdown_body,
                "raw_hash": revision,
            }

            def manifest(job_id: str, contract_version: int | None) -> dict:
                body = {
                    "items": [{"doc_id": doc_id, "revision": revision, "change_kind": "upsert"}],
                    "coverage": "complete_snapshot",
                    "sync_snapshot_id": f"{job_id}:attempt:1",
                    "local_agent_job_id": job_id,
                }
                if contract_version is not None:
                    body["package_contract_version"] = contract_version
                response = client.post(f"/api/v1/sources/{source_id}/adapter/manifest", json=body)
                assert response.status_code == 200, response.text
                return response.json()

            # An agent that predates source times retained the file without one.
            assert manifest("old-agent", None)["required_doc_ids"] == [doc_id]
            untimed = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={**package, "local_agent_job_id": "old-agent"},
            )
            assert untimed.status_code == 200, untimed.text
            old_agent_again = manifest("old-agent-again", None)
            upgraded = manifest("upgraded-agent", LOCAL_PACKAGE_CONTRACT_VERSION)
            timed = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={**package, "source_updated_at": commit_time, "local_agent_job_id": "upgraded-agent"},
            )
            assert timed.status_code == 200, timed.text
            upgraded_again = manifest("upgraded-agent-again", LOCAL_PACKAGE_CONTRACT_VERSION)

        # The older agent keeps reusing what it retained.
        assert old_agent_again["required_doc_ids"] == []
        # The upgraded agent uploads the unchanged file once more, with its time.
        assert upgraded["required_doc_ids"] == [doc_id]
        assert upgraded_again["required_doc_ids"] == []
        assert upgraded_again["reused_count"] == 1

        def snapshot_input(job_id: str):
            [source_input] = asyncio.run(
                database.list_source_sync_inputs(
                    workspace_id="local",
                    source_id=source_id,
                    input_snapshot_id=f"{job_id}:attempt:1",
                )
            )
            return source_input

        untimed_input = snapshot_input("old-agent")
        timed_input = snapshot_input("upgraded-agent")
        # The timed package is its own input, not collapsed into the untimed one.
        assert timed_input.input_id != untimed_input.input_id
        assert timed_input.metadata["manifest_entry"]["source_updated_at"] == commit_time
        assert "source_updated_at" not in untimed_input.metadata["manifest_entry"]
        assert snapshot_input("upgraded-agent-again").input_id == timed_input.input_id
        timed_package = json.loads(LocalDocumentStore(cfg.storage.docs_path).read_artifact(timed_input.raw_uri))
        assert timed_package["source_updated_at"] == commit_time
    finally:
        asyncio.run(database.close())


def test_local_source_manifest_requests_changed_body_and_accepts_complete_removal(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            from memforge.local_agent.document_identity import build_local_markdown_doc_id

            source_id = _create_local_markdown_source(client)["id"]
            relative_path = "payroll/README.md"
            doc_id = build_local_markdown_doc_id(
                source_id=source_id, vault_id="engineering", relative_path=relative_path
            )
            first = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": relative_path,
                    "markdown_body": "# Payroll Processing",
                },
            )
            changed = client.post(
                f"/api/v1/sources/{source_id}/adapter/manifest",
                json={
                    "items": [{"doc_id": doc_id, "revision": "blob-sha-2", "change_kind": "upsert"}],
                    "coverage": "complete_snapshot",
                    "sync_snapshot_id": "changed-job:attempt:1",
                    "local_agent_job_id": "changed-job",
                    "local_agent_attempt_count": 1,
                },
            )
            changed_manifest_retry = client.post(
                f"/api/v1/sources/{source_id}/adapter/manifest",
                json={
                    "items": [{"doc_id": doc_id, "revision": "blob-sha-3", "change_kind": "upsert"}],
                    "coverage": "complete_snapshot",
                    "sync_snapshot_id": "changed-job:attempt:1",
                    "local_agent_job_id": "changed-job",
                    "local_agent_attempt_count": 1,
                },
            )
            removed = client.post(
                f"/api/v1/sources/{source_id}/adapter/manifest",
                json={
                    "items": [],
                    "coverage": "complete_snapshot",
                    "sync_snapshot_id": "removed-job:attempt:1",
                    "local_agent_job_id": "removed-job",
                    "local_agent_attempt_count": 1,
                },
            )
            # A device manifest is a complete snapshot of upserts; no other
            # coverage or change kind is accepted.
            refused = {
                (coverage, change_kind): client.post(
                    f"/api/v1/sources/{source_id}/adapter/manifest",
                    json={
                        "items": [{"doc_id": doc_id, "revision": "blob-sha-1", "change_kind": change_kind}],
                        "coverage": coverage,
                        "sync_snapshot_id": "refused-job:attempt:1",
                        "local_agent_job_id": "refused-job",
                        "local_agent_attempt_count": 1,
                    },
                )
                for coverage, change_kind in (
                    ("partial", "upsert"),
                    ("bounded_delta", "upsert"),
                    ("complete_snapshot", "tombstone"),
                )
            }

        assert first.status_code == 200, first.text
        assert changed.status_code == 200, changed.text
        assert changed.json()["required_doc_ids"] == [doc_id]
        assert changed.json()["reused_count"] == 0
        assert changed_manifest_retry.status_code == 409
        assert changed_manifest_retry.json()["detail"] == "source snapshot manifest changed"
        assert removed.status_code == 200, removed.text
        assert removed.json()["manifest_count"] == 0
        assert removed.json()["required_doc_ids"] == []
        assert asyncio.run(
            database.list_source_sync_inputs(
                workspace_id="local",
                source_id=source_id,
                input_snapshot_id="removed-job:attempt:1",
            )
        ) == []
        assert {key: response.status_code for key, response in refused.items()} == {
            key: 422 for key in refused
        }
        assert (
            asyncio.run(
                database.get_source_sync_snapshot_manifest_status(
                    source_id=source_id, workspace_id="local", snapshot_id="refused-job:attempt:1"
                )
            )
            is None
        )
    finally:
        asyncio.run(database.close())


def test_local_adapter_document_push_requires_execution_owner(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)

    async def _seed_source() -> None:
        await database.upsert_source(
            id="src-owned-local",
            type="local_markdown",
            name="Owned notes",
            config_json=json.dumps(
                {
                    "vault_id": "engineering",
                    "documents_dir": str(tmp_path / "local-inbox"),
                }
            ),
            created_by_user_id="alice",
            execution_owner_user_id="alice",
            access_policy="workspace",
            owner_user_id="alice",
        )

    asyncio.run(_seed_source())
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            principal_resolver=lambda request: "bob",
            workspace_role_resolver=lambda request: "member",
        )
        with LeaseAwareTestClient(app) as client:
            response = client.post(
                "/api/v1/sources/src-owned-local/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "decisions/cutoff.md",
                    "markdown_body": "# Cutoff\n\nDeploy on Tuesday.",
                    "process_now": False,
                },
            )

        assert response.status_code == 403
        assert response.json()["detail"] == "local_agent_sync_execution_owner_forbidden"
    finally:
        asyncio.run(database.close())


def test_local_adapter_revalidates_lease_before_committing_snapshot_membership(tmp_path):
    from memforge.server.admin_api import create_admin_app

    database = _connect_database(tmp_path)
    calls: list[tuple] = []

    async def validate(*args):
        calls.append(args)
        return await _allow_local_agent_lease()

    try:
        app = create_admin_app(
            db=database,
            config=_config(tmp_path),
            local_agent_lease_validator=validate,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            response = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "lease-check.md",
                    "markdown_body": "# Lease check",
                    "sync_snapshot_id": "test-local-agent-job:attempt:1",
                },
            )

        assert response.status_code == 200, response.text
        assert len(calls) == 2
    finally:
        asyncio.run(database.close())


def test_local_adapter_document_push_allows_unmapped_source(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_unmapped_local_markdown_source(client)
            source_id = created["id"]
            response = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "notes/unmapped.md",
                    "markdown_body": "# Unmapped\n\nStill ingest this.",
                    "process_now": False,
                },
            )
        assert response.status_code == 200, response.text
        body = response.json()
        row = asyncio.run(database.get_source(source_id))
        assert row is not None
        assert row["project_binding"] is None
        assert body["package_path"] is None
        assert Path(body["package_uri"]).exists()
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_is_idempotent_on_doc_id(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            payload = {
                "vault_id": "engineering",
                "relative_path": "notes/index.md",
                "markdown_body": "# Index\n\nTop level.",
                "process_now": False,
            }
            first = client.post(f"/api/v1/sources/{source_id}/adapter/packages", json=payload).json()
            second = client.post(f"/api/v1/sources/{source_id}/adapter/packages", json=payload).json()
        assert first["doc_id"] == second["doc_id"]
        assert first["package_path"] == second["package_path"]
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_attributes_input_to_app_workspace(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
            workspace_id="workspace-a",
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            response = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "notes/index.md",
                    "markdown_body": "# Index\n\nTop level.",
                    "process_now": False,
                },
            )

        assert response.status_code == 200, response.text
        inputs = asyncio.run(
            database.list_source_sync_inputs(
                source_id=source_id,
                workspace_id="workspace-a",
            )
        )
        assert len(inputs) == 1
        assert inputs[0].workspace_id == "workspace-a"
    finally:
        asyncio.run(database.close())


@pytest.mark.parametrize(
    "raw_payload",
    [
        {
            "conversation_id": "19:channel@example.test",
            "window_id": "window-a",
            "messages": [{"id": "message-a", "content": "Decision"}],
        },
        {
            "conversation_id": "19:other@example.test",
            "window_id": "window-a",
            "messages": [
                {
                    "id": "message-a",
                    "content": "Decision",
                    "time": "2026-07-16T09:00:00+00:00",
                }
            ],
        },
    ],
)
def test_teams_adapter_rejects_ambiguous_message_evidence(
    tmp_path,
    raw_payload,
):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_teams_source(client)["id"]
            response = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "conversation_id": "19:channel@example.test",
                    "window_id": "window-a",
                    "revision_hash": "revision-a",
                    "raw_payload": raw_payload,
                    "process_now": False,
                },
            )

        assert response.status_code == 400
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_rejects_vault_mismatch(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            response = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "wrong-vault",
                    "relative_path": "notes/x.md",
                    "markdown_body": "# X",
                    "process_now": False,
                },
            )
        assert response.status_code == 400
        assert "vault_id" in response.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_rejects_path_traversal(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            response = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "../escape.md",
                    "markdown_body": "# Nope",
                    "process_now": False,
                },
            )
        assert response.status_code == 400
        assert ".." in response.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_rejects_paused_source(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            assert client.put(f"/api/v1/sources/{source_id}", json={"status": "paused"}).status_code == 200
            response = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "notes.md",
                    "markdown_body": "# Notes",
                    "process_now": False,
                },
            )
        assert response.status_code == 400
        assert response.json()["detail"] == "Source is paused"
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_cleans_artifact_when_lease_expires_after_write(
    tmp_path,
):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    validation_count = 0

    async def validate_lease(*args, **kwargs) -> LocalAgentLease | None:
        nonlocal validation_count
        validation_count += 1
        return await _allow_local_agent_lease() if validation_count == 1 else None

    try:
        app = create_admin_app(
            db=database,
            config=cfg,
            local_agent_lease_validator=validate_lease,
        )
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            response = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "notes.md",
                    "markdown_body": "# Notes",
                    "process_now": False,
                },
            )

        assert response.status_code == 409
        assert response.json()["detail"] == "local_agent_lease_not_current"
        _run_artifact_cleanup(database, cfg)
        assert list(Path(cfg.storage.docs_path).rglob("*package*.json")) == []
        assert asyncio.run(database.list_source_artifact_cleanup_tasks()) == []
        assert asyncio.run(database.list_source_sync_inputs(workspace_id="local", source_id=source_id)) == []
    finally:
        asyncio.run(database.close())


def test_local_adapter_push_requires_local_adapter_source(tmp_path):
    """Pushing to a non-local-adapter source must be rejected."""
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)

    async def _seed_other_source() -> str:
        await database.upsert_source(
            id="src-other",
            type="confluence",
            name="Wiki",
            config_json=json.dumps({"base_url": "https://wiki.example.test"}),
            access_policy="private",
            owner_user_id="dev",
        )
        return "src-other"

    try:
        source_id = asyncio.run(_seed_other_source())
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            response = client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "x",
                    "relative_path": "x.md",
                    "markdown_body": "# X",
                    "process_now": False,
                },
            )
        assert response.status_code == 400
        assert "local adapter source" in response.json()["detail"]
    finally:
        asyncio.run(database.close())


def test_local_markdown_gene_discovers_pushed_packages(tmp_path):
    """The gene discovers the JSON packages the push endpoint writes."""
    from memforge.genes import create_gene
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            created = _create_local_markdown_source(client)
            source_id = created["id"]
            client.post(
                f"/api/v1/sources/{source_id}/adapter/packages",
                json={
                    "vault_id": "engineering",
                    "relative_path": "release.md",
                    "markdown_body": "# Release\n\nCut on Tuesday.",
                    "process_now": False,
                },
            )

            row = asyncio.run(database.get_source(source_id))
            projected = _project_source_inputs(database, row)
            gene = create_gene("local_markdown", projected["config"], source_id)
            gene.bind_document_store(LocalDocumentStore(cfg.storage.docs_path))
            asyncio.run(gene.authenticate())

        async def _collect():
            items = []
            async for item in gene.discover():
                items.append(item)
            return items

        items = asyncio.run(_collect())
        assert len(items) == 1
        assert items[0].title == "Release"
        assert items[0].space_or_project == "engineering"

        raw = asyncio.run(gene.fetch(items[0]))
        normalized = asyncio.run(gene.normalize(raw))
        assert normalized.markdown_body.startswith("# Release")
        assert normalized.source_semantics["vault_id"] == "engineering"
        assert normalized.source_semantics["relative_path"] == "release.md"
    finally:
        asyncio.run(database.close())


def test_to_markdown_conversions():
    """The gene's content-type converter handles every supported file type."""
    from memforge.genes.local_markdown_gene import _to_markdown

    assert _to_markdown("text/markdown", "# Keep\n\nx").startswith("# Keep")
    assert _to_markdown("text/plain", "plain") == "plain"
    assert _to_markdown("application/json", '{"a": 1}') == '```json\n{"a": 1}\n```\n'
    assert _to_markdown("text/html", "<h1>T</h1><p>hi</p>") == "# T\n\nhi"
    # content-type parameters and unknown types degrade to plain text
    assert _to_markdown("text/markdown; charset=utf-8", "# H") == "# H"
    assert _to_markdown("application/octet-stream", "raw") == "raw"


def test_local_markdown_gene_converts_by_content_type(tmp_path):
    """HTML and JSON pushes convert to markdown server-side; text passes through."""
    from memforge.genes import create_gene
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    try:
        app = create_admin_app(db=database, config=cfg, local_agent_lease_validator=_allow_local_agent_lease)
        with LeaseAwareTestClient(app) as client:
            source_id = _create_local_markdown_source(client)["id"]
            docs = [
                ("page.html", "text/html", "<h1>Decision</h1><p>Ship on <b>Tuesday</b>.</p>"),
                ("data.json", "application/json", '{"deploy": "tuesday"}'),
                ("note.txt", "text/plain", "plain reminder"),
            ]
            for rel, ctype, body in docs:
                resp = client.post(
                    f"/api/v1/sources/{source_id}/adapter/packages",
                    json={
                        "vault_id": "engineering",
                        "relative_path": rel,
                        "markdown_body": body,
                        "content_type": ctype,
                        "process_now": False,
                    },
                )
                assert resp.status_code == 200, resp.text

            row = asyncio.run(database.get_source(source_id))
            projected = _project_source_inputs(database, row)
            gene = create_gene("local_markdown", projected["config"], source_id)
            gene.bind_document_store(LocalDocumentStore(cfg.storage.docs_path))
            asyncio.run(gene.authenticate())

        async def _normalized_by_path():
            out = {}
            async for item in gene.discover():
                normalized = await gene.normalize(await gene.fetch(item))
                out[normalized.source_semantics["relative_path"]] = normalized
            return out

        by_path = asyncio.run(_normalized_by_path())
        assert by_path["page.html"].markdown_body == "# Decision\n\nShip on **Tuesday**."
        assert by_path["page.html"].source_semantics["content_type"] == "text/html"
        assert by_path["data.json"].markdown_body == '```json\n{"deploy": "tuesday"}\n```\n'
        assert by_path["note.txt"].markdown_body == "plain reminder"
    finally:
        asyncio.run(database.close())


async def _allow_local_agent_lease(*args, **kwargs) -> LocalAgentLease:
    return LocalAgentLease(payload={}, leased_at=datetime.now(timezone.utc))


class LeaseAwareTestClient(TestClient):
    def post(self, url, *args, **kwargs):
        if "/adapter/packages" in url or "/adapter/manifest" in url or url.endswith("/process"):
            body = dict(kwargs.get("json") or {})
            body.setdefault("local_agent_job_id", "test-local-agent-job")
            body.setdefault("local_agent_attempt_count", 1)
            kwargs["json"] = body
        return super().post(url, *args, **kwargs)


class _FailOnceThenCleanUp:
    """Fails the first sync-input record; runs artifact cleanup right before the next one.

    The second call lands in the window between writing an object and recording
    the reference to it, where cleanup must not be able to delete that object.
    """

    def __init__(self, database: Database, document_store: LocalDocumentStore) -> None:
        self.database = database
        self.document_store = document_store
        self.record = database.create_source_sync_input
        self.calls = 0
        self.cleaned = 0

    async def __call__(self, **kwargs):
        from memforge.storage.source_cleanup import SourceArtifactCleanupService

        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("sync input store unavailable")
        self.cleaned += await SourceArtifactCleanupService(self.database, self.document_store).run_pending()
        return await self.record(**kwargs)


def _cleanup_uris(database: Database) -> list[str]:
    return [task.artifact_uri for task in asyncio.run(database.list_source_artifact_cleanup_tasks(limit=10))]


def test_a_retried_artifact_upload_survives_cleanup_of_the_failed_attempt(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    document_store = LocalDocumentStore(cfg.storage.docs_path)
    image_bytes = b"\x89PNG\r\n\x1a\nretried-upload"
    params = {
        "source_unit_key": "Payroll Processing/architecture.png",
        "provider_key": "Payroll Processing/architecture.png",
        "provider_revision": "blob-image-sha",
        "parent_observation_type": "file_content",
        "parent_provider_key": "content",
        "filename": "architecture.png",
        "media_type": "image/png",
        "local_agent_job_id": "test-local-agent-job",
        "local_agent_attempt_count": 1,
    }
    try:
        app = create_admin_app(
            db=database, config=cfg, document_store=document_store,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        record = _FailOnceThenCleanUp(database, document_store)
        database.create_source_sync_input = record
        with LeaseAwareTestClient(app, raise_server_exceptions=False) as client:
            source_id = _create_teams_source(client)["id"]
            route = f"/api/v1/sources/{source_id}/adapter/artifacts"
            failed = client.post(route, params=params, content=image_bytes, headers={"content-type": "image/png"})
            [abandoned] = _cleanup_uris(database)
            retried = client.post(route, params=params, content=image_bytes, headers={"content-type": "image/png"})

        assert failed.status_code == 500
        assert retried.status_code == 200, retried.text
        assert record.cleaned == 1
        [retained] = asyncio.run(database.list_source_sync_inputs(workspace_id="local", source_id=source_id))
        assert retained.raw_uri != abandoned
        assert document_store.read_artifact(retained.raw_uri) == image_bytes
        assert document_store.get_artifact(abandoned, "image/png") is None
    finally:
        asyncio.run(database.close())


def test_a_retried_package_push_survives_cleanup_of_the_failed_attempt(tmp_path):
    from memforge.server.admin_api import create_admin_app

    cfg = _config(tmp_path)
    database = _connect_database(tmp_path)
    document_store = LocalDocumentStore(cfg.storage.docs_path)
    package = {
        "vault_id": "engineering",
        "relative_path": "payroll/overview.md",
        "markdown_body": "# Overview\n\nPayroll runs nightly.",
        "content_type": "text/markdown",
        # A retry of the same push carries the same payload, so its bytes are identical.
        "submitted_at": "2026-09-27T08:00:00+00:00",
        "submitted_by": "local-agent",
    }
    try:
        app = create_admin_app(
            db=database, config=cfg, document_store=document_store,
            local_agent_lease_validator=_allow_local_agent_lease,
        )
        record = _FailOnceThenCleanUp(database, document_store)
        database.create_source_sync_input = record
        with LeaseAwareTestClient(app, raise_server_exceptions=False) as client:
            source_id = _create_local_markdown_source(client)["id"]
            route = f"/api/v1/sources/{source_id}/adapter/packages"
            failed = client.post(route, json=package)
            [abandoned] = _cleanup_uris(database)
            retried = client.post(route, json=package)

        assert failed.status_code == 500
        assert retried.status_code == 200, retried.text
        assert record.cleaned == 1
        [retained] = asyncio.run(database.list_source_sync_inputs(workspace_id="local", source_id=source_id))
        assert retained.raw_uri == retried.json()["package_uri"] != abandoned
        assert document_store.get_artifact(retained.raw_uri, "application/json") is not None
        assert document_store.get_artifact(abandoned, "application/json") is None
    finally:
        asyncio.run(database.close())
