import json

import pytest

from memforge.genes.agent_session_gene import AgentSessionGene
from memforge.memory.project_resolver import (
    binding_names_project,
    binding_without_project,
    released_project_bindings,
    resolve_project_key,
)
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


@pytest.fixture
async def db(tmp_path):
    database = Database(str(tmp_path / "resolver_writer.db"))
    await database.connect()
    yield database
    await database.close()


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


@pytest.mark.parametrize(
    ("binding", "released"),
    [
        ({"mode": "fixed", "project_key": "PAY"}, None),
        (
            {"mode": "by_field", "field": "repo", "map": {"payroll": "PAY", "risk": "RISK"}, "default": "RISK"},
            {"mode": "by_field", "field": "repo", "map": {"risk": "RISK"}, "default": "RISK"},
        ),
        (
            {"mode": "by_field", "field": "repo", "default": "PAY"},
            {"mode": "by_field", "field": "repo", "default": UNSORTED_PROJECT_KEY},
        ),
    ],
)
def test_a_released_binding_no_longer_resolves_to_the_deleted_project(binding, released):
    assert binding_names_project(binding, "PAY")
    assert binding_without_project(binding, "PAY") == released
    assert not binding_names_project(released, "PAY")
    assert resolve_project_key(released, item_field_value=None, repo="payroll", workspace=None) != "PAY"


@pytest.mark.parametrize(
    "binding",
    [
        None,
        {"mode": "fixed", "project_key": "RISK"},
        {"mode": "by_field", "field": "repo", "map": {"risk": "RISK"}, "default": UNSORTED_PROJECT_KEY},
    ],
)
def test_bindings_to_other_projects_do_not_name_it(binding):
    assert not binding_names_project(binding, "PAY")


def test_released_project_bindings_reads_stored_json_and_skips_what_is_not_a_json_object():
    fixed = {"mode": "fixed", "project_key": "PAY"}
    stored = [
        ("src-json", json.dumps(fixed)),
        ("src-decoded", {"mode": "by_field", "field": "repo", "map": {"payroll": "PAY"}, "default": "PAY"}),
        ("src-other", json.dumps({"mode": "fixed", "project_key": "RISK"})),
        ("src-truncated", '{"mode": "fixed", "project_key": "PA'),
        ("src-list", '["PAY"]'),
        ("src-unbound", None),
    ]

    assert released_project_bindings(stored, "PAY") == [
        ("src-json", None),
        ("src-decoded", {"mode": "by_field", "field": "repo", "map": {}, "default": UNSORTED_PROJECT_KEY}),
    ]
