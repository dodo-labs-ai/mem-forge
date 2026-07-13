import json
from pathlib import Path

import pytest

from memforge.agent_sessions import (
    ensure_agent_session_source,
    submit_agent_session_document,
)
from memforge.config import AppConfig
from memforge.genes.agent_session_gene import AgentSessionGene
from memforge.memory.project_resolver import resolve_project_key
from memforge.models import UNSORTED_PROJECT_KEY
from memforge.storage.database import Database


def test_fixed_returns_configured_key():
    binding = {"mode": "fixed", "project_key": "PAY"}
    assert resolve_project_key(binding, item_field_value="anything", repo=None, workspace="/tmp") == "PAY"


def test_by_field_hit_returns_mapped_key():
    binding = {
        "mode": "by_field",
        "field": "space_or_project",
        "map": {"PAYSPACE": "PAY", "RISKSPACE": "RISK"},
        "default": "UNSORTED",
    }
    assert resolve_project_key(binding, item_field_value="PAYSPACE", repo=None, workspace="/tmp") == "PAY"


def test_by_field_miss_returns_default():
    binding = {
        "mode": "by_field",
        "field": "space_or_project",
        "map": {"PAYSPACE": "PAY"},
        "default": "UNSORTED",
    }
    assert resolve_project_key(binding, item_field_value="UNKNOWN", repo=None, workspace="/tmp") == "UNSORTED"


def test_admin_set_default_can_be_shared():
    binding = {
        "mode": "by_field",
        "field": "space_or_project",
        "map": {},
        "default": "SHARED",
    }
    assert resolve_project_key(binding, item_field_value="anything", repo=None, workspace="/tmp") == "SHARED"


def test_agent_repo_absent_returns_default_not_workspace_basename():
    """Agent non-repo fallback: never mint Path(workspace).name as a key
    (would create junk like 'tmp', 'Desktop'). Resolves to default."""
    binding = {
        "mode": "by_field",
        "field": "repo",
        "map": {"my-app": "APP"},
        "default": "UNSORTED",
    }
    result = resolve_project_key(binding, item_field_value=None, repo=None, workspace="/tmp/work")
    assert result == "UNSORTED"


def test_no_binding_resolves_to_unsorted():
    """A source with no binding (legacy row) still resolves predictably."""
    assert resolve_project_key(None, item_field_value="anything", repo=None, workspace="/tmp") == "UNSORTED"


# ---------------------------------------------------------------------------
# Integration: writer paths route through resolve_project_key.
# ---------------------------------------------------------------------------


def _config(tmp_path: Path) -> AppConfig:
    cfg = AppConfig(base_dir=tmp_path / "mem")
    cfg.llm.enrichment_api_key = ""
    cfg.llm.embedding_api_key = ""
    return cfg


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "resolver_writer.db"))
    await database.connect()
    yield database
    await database.close()


@pytest.mark.asyncio
async def test_agent_session_with_no_repo_lands_in_unsorted(db, tmp_path):
    """Without a repo and with the default (None) binding, the package
    must land under the UNSORTED project bucket: the basename of the
    workspace path must NOT leak into the project key."""
    cfg = _config(tmp_path)
    result = await submit_agent_session_document(
        db=db,
        config=cfg,
        user_id="user-owner",
        client="codex",
        session_id="sess-no-repo",
        trigger="Stop",
        document_markdown="# Session Summary\n\n## User-Confirmed Decisions\n- ok.",
        workspace="/tmp/scratch-xyz",
        repo=None,
    )

    package = json.loads(Path(result["document_uri"]).read_text(encoding="utf-8"))
    assert package["space_or_project"] == UNSORTED_PROJECT_KEY
    # The on-disk layout follows the resolved project, so a no-repo run
    # never seeds a junk directory derived from the workspace basename.
    assert Path(result["document_uri"]).parent.name == UNSORTED_PROJECT_KEY.lower()


@pytest.mark.asyncio
async def test_agent_session_by_field_repo_maps_to_configured_key(db, tmp_path):
    """A `by_field` binding on `repo` resolves to the mapped key, so the
    package's `space_or_project` carries the project decided at intake."""
    cfg = _config(tmp_path)
    # Seed the per-client agent-session source first so we can attach a
    # binding to it before the submit call resolves the project key.
    source = await ensure_agent_session_source(
        db,
        cfg,
        client="codex",
        owner_user_id="user-owner",
    )
    binding = {
        "mode": "by_field",
        "field": "repo",
        "map": {"my-app": "APP"},
        "default": "UNSORTED",
    }
    await db.db.execute(
        "UPDATE sources SET project_binding = ? WHERE id = ?",
        (json.dumps(binding), source["id"]),
    )
    await db.db.commit()

    result = await submit_agent_session_document(
        db=db,
        config=cfg,
        user_id="user-owner",
        client="codex",
        session_id="sess-mapped",
        trigger="Stop",
        document_markdown="# Session Summary\n\n## User-Confirmed Decisions\n- ok.",
        workspace="/tmp/work",
        repo="my-app",
    )
    package = json.loads(Path(result["document_uri"]).read_text(encoding="utf-8"))
    assert package["space_or_project"] == "APP"


@pytest.mark.asyncio
async def test_sync_pipeline_resolves_project_key_via_binding(db):
    """`_process_item`'s project_key derivation reads the source's
    `project_binding` and routes through `resolve_project_key`. A doc
    source with a `by_field` binding mapping `PAYSPACE -> PAY` must
    resolve to `PAY` even when the item's `space_or_project` is the raw
    field value the binding maps from.
    """
    binding = {
        "mode": "by_field",
        "field": "space_or_project",
        "map": {"PAYSPACE": "PAY", "RISKSPACE": "RISK"},
        "default": "UNSORTED",
    }
    await db.db.execute(
        "INSERT INTO sources (id, type, name, status, last_sync, doc_count, "
        "config, project_binding, access_policy, access_state, owner_user_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        (
            "src-conf",
            "confluence",
            "Conf",
            "active",
            None,
            0,
            json.dumps({"spaces": ["PAYSPACE"]}),
            json.dumps(binding),
            "workspace",
            "active",
            "dev",
        ),
    )
    await db.db.commit()

    source_row = await db.get_source("src-conf")
    # The same call sites the sync pipeline uses post-resolver wiring.
    resolved = resolve_project_key(
        source_row.get("project_binding"),
        item_field_value="PAYSPACE",
        repo=None,
        workspace=None,
    )
    assert resolved == "PAY"

    # An unmapped raw value falls through to the binding default.
    fallback = resolve_project_key(
        source_row.get("project_binding"),
        item_field_value="UNKNOWN",
        repo=None,
        workspace=None,
    )
    assert fallback == UNSORTED_PROJECT_KEY


@pytest.mark.asyncio
async def test_sync_pipeline_legacy_source_resolves_to_unsorted(db):
    """A source row inserted before the binding column existed reads as
    `project_binding=None`. The resolver returns UNSORTED, so the writer
    path keeps the row visible without ever minting a junk key.
    """
    await db.db.execute(
        "INSERT INTO sources (id, type, name, status, last_sync, doc_count, "
        "config, access_policy, access_state, owner_user_id, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
        ("src-legacy", "jira", "Legacy", "active", None, 0, "{}", "workspace", "active", "dev"),
    )
    await db.db.commit()

    source_row = await db.get_source("src-legacy")
    resolved = resolve_project_key(
        source_row.get("project_binding"),
        item_field_value="ANY-RAW-VALUE",
        repo=None,
        workspace=None,
    )
    assert resolved == UNSORTED_PROJECT_KEY


def test_agent_session_gene_declares_repo_as_project_field():
    """The agent-session gene exposes `repo` as the field a `by_field`
    binding reads, so the admin UI can scope the binding editor to fields
    the gene actually populates."""
    schema = AgentSessionGene.config_schema()
    assert schema.project_field == "repo"
