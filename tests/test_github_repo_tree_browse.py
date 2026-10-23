"""The GitHub Repository folder browse: how it lists a tree, and how far it reads a truncated one."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from memforge.config import AppConfig
from memforge.genes import github_repo_gene
from memforge.genes.toolkit.github_api import BROWSE_TREE_REQUEST_BUDGET
from memforge.server.admin_api import _github_repo_tree_items
from memforge.storage.database import Database
from tests.github_api_fixture import FakeGitHub

REPO_URL = "https://github.example.test/payroll/architecture"
API = "https://github.example.test/api/v3/repos/payroll/architecture"
BROWSE_CONFIG = {"connection_mode": "cloud_pull", "repo_url": REPO_URL, "include_extensions": ["md"]}


def test_github_repo_tree_items_synthesizes_folders_and_files():
    items = _github_repo_tree_items(
        [
            {"path": "Payroll Processing/README.md", "type": "blob", "size": 42},
            {"path": "Payroll Processing/V2/Migration.md", "type": "blob", "size": 84},
            {"path": "Flexible Payroll", "type": "tree"},
            {"path": "Flexible Payroll/Overview.md", "type": "blob", "size": 11},
        ],
        limit=20,
        include_extensions={"md"},
    )

    assert [(item.type, item.path, item.size) for item in items] == [
        ("tree", "Flexible Payroll", None),
        ("tree", "Payroll Processing", None),
        ("tree", "Payroll Processing/V2", None),
        ("blob", "Flexible Payroll/Overview.md", 11),
        ("blob", "Payroll Processing/README.md", 42),
        ("blob", "Payroll Processing/V2/Migration.md", 84),
    ]


def test_github_repo_tree_items_returns_limit_plus_one_for_truncation_detection():
    items = _github_repo_tree_items(
        [
            {"path": "a/one.md", "type": "blob"},
            {"path": "b/two.md", "type": "blob"},
            {"path": "c/three.md", "type": "blob"},
        ],
        limit=2,
        include_extensions={"md"},
    )

    assert len(items) == 3


def test_github_repo_tree_items_omit_unsupported_files_and_empty_folders():
    items = _github_repo_tree_items(
        [
            {"path": "docs/README.md", "type": "blob"},
            {"path": "assets/logo.png", "type": "blob"},
            {"path": "assets", "type": "tree"},
        ],
        limit=20,
        include_extensions={"md"},
    )

    assert [(item.type, item.path) for item in items] == [
        ("tree", "docs"),
        ("blob", "docs/README.md"),
    ]


@pytest.fixture
async def database(tmp_path):
    db = Database(str(tmp_path / "runtime.db"))
    await db.connect()
    yield db
    await db.close()


def _browse(database, tmp_path, github: FakeGitHub, config: dict):
    from memforge.server.admin_api import create_admin_app

    app_config = AppConfig(base_dir=tmp_path / "mem")
    app_config.server.jwt_secret = "test-secret"
    app_config.sync.scheduler_enabled = False
    app_config.sync.worker_enabled = False
    with github.http.serving(github_repo_gene), TestClient(create_admin_app(db=database, config=app_config)) as client:
        return client.post("/api/v1/genes/github_repo/browse-tree", json={"config": config, "limit": 50})


@pytest.mark.asyncio
async def test_the_browse_reads_the_default_branch_and_a_truncated_tree_by_its_folders(database, tmp_path):
    github = FakeGitHub(API, files={"docs/guide.md": b"# Guide\n", "docs/runbooks/restart.md": b"# Restart\n"}, truncated={""})

    response = _browse(database, tmp_path, github, BROWSE_CONFIG)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["ref"] == "main"
    assert [(item["type"], item["path"]) for item in body["items"]] == [
        ("tree", "docs"),
        ("tree", "docs/runbooks"),
        ("blob", "docs/guide.md"),
        ("blob", "docs/runbooks/restart.md"),
    ]
    assert f"GET {API}/git/trees/main?recursive=1" in github.requests
    assert all(client.is_closed for client in github.http.clients)


@pytest.mark.asyncio
async def test_the_browse_answers_400_when_a_truncated_tree_needs_more_requests_than_it_may_spend(database, tmp_path):
    files = {f"area-{index:02}/README.md": b"# Area\n" for index in range(BROWSE_TREE_REQUEST_BUDGET)}
    github = FakeGitHub(API, files=files, truncated={""})

    response = _browse(database, tmp_path, github, {**BROWSE_CONFIG, "ref": "main"})

    assert response.status_code == 400
    assert response.json()["detail"] == (
        f"GitHub truncated the tree of payroll/architecture, and reading it folder by folder needs more than "
        f"{BROWSE_TREE_REQUEST_BUDGET} requests to browse. Type the folder or file paths to sync instead."
    )
    assert sum("/git/trees/" in request for request in github.requests) == BROWSE_TREE_REQUEST_BUDGET
    assert all(client.is_closed for client in github.http.clients)
