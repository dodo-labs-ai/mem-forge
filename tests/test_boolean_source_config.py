"""Boolean configuration fields of document connectors: read, checked on save and checked when a Gene is created.

A boolean field accepts real booleans, 0/1 and true/false, yes/no, on/off,
1/0 in any case; ``None`` and ``""`` read as the field's default. The admin
API refuses any other value with a 400 naming the field, a Gene refuses it
with ``SourceConfigurationError``, and a run that meets it is not retried.
Scope fingerprints keep reading stored values as they are.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from memforge.genes import GENE_REGISTRY
from memforge.genes.base import SourceConfigurationError
from memforge.genes.confluence_gene import ConfluenceGene
from memforge.genes.family import CollectionFamilyKind, gene_family
from memforge.genes.github_pages_gene import GitHubPagesGene
from memforge.genes.jira_gene import JiraGene, _issue_payload_from_search
from memforge.models import ConfigFieldType
from memforge.server.admin_api import create_admin_app
from memforge.source_projection_config import canonical_projection_scope, projection_scope_fingerprint
from memforge.source_secrets import prepare_source_config_for_storage
from memforge.storage.database import Database
from tests.test_startup_runtime import TEST_SOURCE_KEY, _config

PAGES_SITE = "https://pages.example.test/org/site"


def _document_boolean_fields() -> list[tuple[str, str]]:
    return [
        (source_type, field.key)
        for source_type, gene_cls in sorted(GENE_REGISTRY.items())
        if gene_family(gene_cls) is CollectionFamilyKind.DOCUMENT
        for field in gene_cls.config_schema().fields
        if field.field_type == ConfigFieldType.BOOLEAN
    ]


def test_every_document_boolean_field_is_known():
    assert _document_boolean_fields() == [
        ("confluence", "include_children"),
        ("github_pages", "sitemap_authoritative"),
        ("jira", "include_comments"),
    ]


@pytest.mark.parametrize(("source_type", "key"), _document_boolean_fields())
@pytest.mark.parametrize("value", ["maybe", "2", 2, []])
def test_creating_a_gene_refuses_an_odd_boolean(source_type, key, value):
    with pytest.raises(SourceConfigurationError, match=key):
        GENE_REGISTRY[source_type]({key: value}, "src-odd")


@pytest.mark.parametrize(("source_type", "key"), _document_boolean_fields())
@pytest.mark.parametrize("value", [True, False, 1, 0, "Yes", " off ", "1", None, ""])
def test_creating_a_gene_accepts_a_boolean(source_type, key, value):
    GENE_REGISTRY[source_type]({key: value}, "src-boolean")


@pytest.mark.parametrize(
    ("include_children", "lists_child"),
    [(None, True), ("", True), ("yes", True), (True, True), ("off", False), (0, False)],
)
def test_confluence_include_children_reads_through_bool_config(include_children, lists_child):
    gene = ConfluenceGene(
        {"base_url": "https://wiki.example.test", "page_tree_root": "1", "include_children": include_children},
        "src-confluence",
    )
    child = {"id": "2", "type": "page", "status": "current", "ancestors": [{"id": "1"}]}

    assert gene._page_tree_lists(child, set()) is lists_child


def _searched_issue() -> dict:
    fields = dict.fromkeys(("assignee", "description", "priority", "resolution", "status"))
    fields.update(labels=[], summary="Pay", updated="2026-01-01T00:00:00.000+0000")
    fields["comment"] = {"startAt": 0, "total": 0, "comments": []}
    return {"id": "1", "key": "PAY-1", "fields": fields, "changelog": {"startAt": 0, "total": 0, "histories": []}}


@pytest.mark.parametrize(
    ("include_comments", "included"),
    [(None, True), ("", True), ("yes", True), ("No", False), (False, False), (1, True)],
)
def test_jira_include_comments_defaults_to_true(include_comments, included):
    payload = _issue_payload_from_search(_searched_issue(), {"include_comments": include_comments})

    assert payload["_comments_included"] is included


@pytest.mark.parametrize(("value", "lists"), [(None, False), ("", False), ("On", True), ("0", False)])
def test_pages_sitemap_authoritative_reads_through_bool_config(value, lists):
    config = {"sync_mode": "subtree", "auth_mode": "none", "sitemap_authoritative": value}

    assert GitHubPagesGene.lists_scope(config) is lists


def test_jira_refuses_an_odd_include_comments_when_reading():
    with pytest.raises(SourceConfigurationError, match="include_comments"):
        _issue_payload_from_search(_searched_issue(), {"include_comments": "maybe"})
    with pytest.raises(SourceConfigurationError, match="include_comments"):
        JiraGene({"base_url": "https://jira.example.test", "include_comments": "sometimes"}, "src-jira")


@pytest.mark.parametrize(
    ("source_type", "config", "fingerprint"),
    [
        (
            "jira",
            {"query_mode": "simple", "projects": ["OPS"], "include_comments": "yes"},
            "bfeb26e5c4bfb477aa3df0936f7537c8cb2e348580e4a6ed2d1bd26ef762608b",
        ),
        (
            "jira",
            {"query_mode": "simple", "projects": ["OPS"], "include_comments": "1"},
            "ac26106f3942531e56777cfe806fb39d89017f7e832def04a81ba6bd22e0f864",
        ),
        (
            "confluence",
            {"sync_mode": "page_tree", "page_tree_root": "123", "include_children": "off"},
            "ea89b033b87c121225b8f0148b902a2fbb833208377e93287cc1e503aebc9696",
        ),
    ],
)
def test_scope_fingerprints_keep_stored_boolean_values_as_they_are(source_type, config, fingerprint):
    scope = canonical_projection_scope(source_type, config)

    assert {key: scope[key] for key in config if key in scope} == config
    assert projection_scope_fingerprint(scope) == fingerprint


# ---------------------------------------------------------------------------
# Admin API
# ---------------------------------------------------------------------------


def _create_confluence(tmp_path, include_children: object):
    with TestClient(create_admin_app(config=_config(tmp_path))) as client:
        return client.post(
            "/api/v1/sources",
            json={
                "type": "confluence",
                "name": "Engineering Wiki",
                "access_policy": "private",
                "config": {
                    "base_url": "https://wiki.example.test",
                    "sync_mode": "page_tree",
                    "page_tree_root": "123",
                    "pat": "wiki-pat-secret",
                    "include_children": include_children,
                },
            },
        )


@pytest.mark.parametrize("value", ["maybe", "2", 2])
def test_admin_source_create_refuses_an_odd_boolean_naming_the_field(tmp_path, monkeypatch, value):
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)

    response = _create_confluence(tmp_path, value)

    assert response.status_code == 400
    assert "include_children" in response.json()["detail"]


@pytest.mark.parametrize("value", [False, "No", "1", 0])
def test_admin_source_create_accepts_a_boolean(tmp_path, monkeypatch, value):
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)

    response = _create_confluence(tmp_path, value)

    assert response.status_code == 200, response.text


@pytest.fixture
async def db(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMFORGE_SECRET_KEY", TEST_SOURCE_KEY)
    database = Database(str(tmp_path / "boolean-config.db"))
    await database.connect()
    yield database
    await database.close()


async def _pages_source(db: Database, source_id: str, **fields: object) -> dict[str, object]:
    config = {
        "base_url": PAGES_SITE,
        "sync_mode": "subtree",
        "root_url": f"{PAGES_SITE}/guide",
        "auth_mode": "none",
        **fields,
    }
    await db.upsert_source(
        id=source_id,
        type="github_pages",
        name="Site",
        config_json=json.dumps(prepare_source_config_for_storage(config, secret_fields=("pat",))),
        access_policy="workspace",
        owner_user_id="dev",
    )
    return config


@pytest.mark.asyncio
async def test_admin_source_update_refuses_an_odd_boolean_naming_the_field(tmp_path, db):
    config = await _pages_source(db, "src-pages")

    with TestClient(create_admin_app(db=db, config=_config(tmp_path))) as client:
        refused = client.put("/api/v1/sources/src-pages", json={"config": {**config, "sitemap_authoritative": "sure"}})
        saved = client.put("/api/v1/sources/src-pages", json={"config": {**config, "sitemap_authoritative": "YES"}})

    assert refused.status_code == 400
    assert "sitemap_authoritative" in refused.json()["detail"]
    assert saved.status_code == 200, saved.text
    assert (await db.get_source("src-pages"))["config"]["sitemap_authoritative"] == "YES"


@pytest.mark.asyncio
async def test_reprocess_dry_run_refuses_a_stored_odd_boolean(tmp_path, db):
    await _pages_source(db, "src-odd-pages", sitemap_authoritative="sometimes")

    with TestClient(create_admin_app(db=db, config=_config(tmp_path))) as client:
        response = client.post(
            "/api/v1/sources/src-odd-pages/reprocess",
            json={"document_ids": ["github-pages-guide"], "dry_run": True},
        )

    assert response.status_code == 409
    assert "sitemap_authoritative" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Sync worker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_run_meeting_a_stored_odd_boolean_fails_without_a_retry(tmp_path, db):
    from memforge.runtime import SourceSyncWorker

    await _pages_source(db, "src-odd-pages", sitemap_authoritative="sometimes")
    run = await db.enqueue_source_sync_run(source_id="src-odd-pages", trigger="request")
    worker = SourceSyncWorker(db, _config(tmp_path), worker_id="test-worker")

    async def build_runtime():
        return SimpleNamespace(runtime_event_trace_sink=None)

    worker._build_runtime = build_runtime
    await worker.run_once()
    failed = await db.get_source_sync_run(run.run_id)

    assert failed is not None
    assert failed.status == "failed"
    assert "sitemap_authoritative" in (failed.error_message or "")
    assert failed.next_attempt_at is None
