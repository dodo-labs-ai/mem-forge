"""Canonical scope, namespace and fingerprints per built-in source type.

A change here changes every stored Source's scope fingerprint and opens a scope
transition on each, so the expected values are fixed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from memforge.genes import GENE_REGISTRY, list_available_genes
from memforge.local_agent.source_contract import local_agent_sync_operation
from memforge.source_projection_config import (
    canonical_projection_scope,
    canonical_provider_namespace,
    projection_scope_fingerprint,
)
from memforge.source_types import source_type_descriptor
from memforge.sync_progress import source_progress_unit

_CONFIGS = {
    "confluence": [
        {
            "base_url": "https://wiki.example/",
            "sync_mode": "",
            "page_tree_root": "123",
            "spaces": "PAY, HR",
            "include_children": True,
        },
        {
            "base_url": "https://wiki.example",
            "sync_mode": "space",
            "spaces": ["HR", "PAY", "PAY"],
            "page_tree_root": "9",
            "exclude_labels": "draft",
        },
    ],
    "jira": [
        {
            "base_url": "https://jira.example/",
            "query_mode": "advanced",
            "jql": " project = PAY ",
            "projects": ["PAY"],
            "include_comments": True,
        },
        {
            "base_url": "https://jira.example",
            "projects": "PAY,HR",
            "jql_filter": "status != Done",
            "issue_types": ["Bug"],
            "jql": "x",
        },
        {"base_url": "https://jira.example", "projects": ["PAY"], "query_mode": ""},
    ],
    "github_repo": [
        {
            "repo_url": "https://github.example/org/repo/",
            "ref": "main",
            "include_paths": "docs, src",
            "include_extensions": [".md"],
        }
    ],
    "github_pages": [
        {
            "sync_mode": "subtree",
            "root_url": "https://pages.example/x/",
            "max_depth": 3,
            "exclude_url_patterns": ["b", "a"],
        },
        # How the pages are read is not part of the scope.
        {
            "base_url": "https://pages.example/x",
            "sync_mode": "single_page",
            "page_url": "https://pages.example/x/a/ ",
            "auth_mode": "github_pat",
            "max_pages": 50,
            "sitemap_authoritative": True,
        },
        {
            "sync_mode": "explicit_list",
            "pages": "https://pages.example/x/b, https://pages.example/x/a",
            "branch": "",
            "auth_mode": "none",
        },
        {"sync_mode": "subtree", "root_url": "https://pages.example/x/", "branch": " gh-pages "},
    ],
    "local_markdown": [{"vault_id": "v", "include": "docs/**, *.md", "exclude": []}],
    "teams": [{"conversation_ids": ["b", "a"], "rolling_retention_days": 365, "channels": "c1"}],
    "agent_session": [{"client": "codex"}],
}

_EXPECTED = {
    "agent_session": [[{"client": "codex"}, {}, "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a"]],
    "confluence": [
        [
            {"base_url": "https://wiki.example"},
            {"include_children": True, "page_tree_root": "123", "sync_mode": "page_tree"},
            "60fc16d7331fc7a54a0c473c24cf313a7a690173da7bc0830d163b2f6362c49d",
        ],
        [
            {"base_url": "https://wiki.example"},
            {"exclude_labels": ["draft"], "spaces": ["HR", "PAY"], "sync_mode": "space"},
            "2341c157d6a90c1c4b171c71867e736a0bed24f9f578a8604ff17d453fe5346b",
        ],
    ],
    "github_pages": [
        [
            {},
            {
                "exclude_url_patterns": ["a", "b"],
                "max_depth": 3,
                "root_url": "https://pages.example/x",
                "sync_mode": "subtree",
            },
            "0058d19fcd003ec009b08d061aac3ce6ea21e7c20f9f9b4eeeef4f1555baa03a",
        ],
        [
            {},
            {"page_url": "https://pages.example/x/a", "sync_mode": "single_page"},
            "dd1dc7bd874266b8add3dd8b113a35b809ae1e99d52c4fd992fc186ebc239e67",
        ],
        [
            {},
            {"pages": ["https://pages.example/x/a", "https://pages.example/x/b"], "sync_mode": "explicit_list"},
            "138e2e7529e46d0c35f3a5e15f80f5dbda276a802c48f4d0a2bce1b849c404b3",
        ],
        [
            {},
            {"branch": "gh-pages", "root_url": "https://pages.example/x", "sync_mode": "subtree"},
            "bb0a4f06115cc580f100efa91bd7dd5a66a0c00f2a53c5203d7e560eefd2d492",
        ],
    ],
    "github_repo": [
        [
            {"repo_url": "https://github.example/org/repo"},
            {"include_extensions": [".md"], "include_paths": ["docs", "src"], "ref": "main"},
            "cf6d646c7c0b4c7797fea6d5dda8ab6f0a230fa54f28fd982bf4cc134a1bd19a",
        ]
    ],
    "jira": [
        [
            {"base_url": "https://jira.example"},
            {"include_comments": True, "jql": "project = PAY", "query_mode": "advanced"},
            "d3c6eed76bfa8c9ff9acda68f157c34afee8dd94e9a68c96e51c4578b6a60852",
        ],
        [
            {"base_url": "https://jira.example"},
            {"issue_types": ["Bug"], "jql_filter": "status != Done", "projects": ["HR", "PAY"], "query_mode": "simple"},
            "e7d53081af440de59f1ac89404a3133847e1be10eeb9e3b250ad0e54739c5848",
        ],
        [
            {"base_url": "https://jira.example"},
            {"projects": ["PAY"], "query_mode": "simple"},
            "82ec6b18956f787310aa5aa20e4b4cb8f95fb513526801751488f581c003707a",
        ],
    ],
    "local_markdown": [
        [
            {"vault_id": "v"},
            {"include": ["*.md", "docs/**"]},
            "5a9239481ecca3317f1a37017ed5dfb0eea05cedee0b4cf2cc446e52ae9c6988",
        ]
    ],
    "teams": [
        [
            {},
            {"channels": ["c1"], "conversation_ids": ["a", "b"], "rolling_retention_days": 365},
            "32832b1315d6fb6331f24406501b2fedeeb47d5ec1fe1472e4c2a307686d8e0e",
        ]
    ],
}


@pytest.mark.parametrize("source_type", sorted(_CONFIGS))
def test_canonical_scope_and_namespace_are_fixed(source_type: str) -> None:
    actual = [
        [
            canonical_provider_namespace(source_type, config),
            canonical_projection_scope(source_type, config),
            projection_scope_fingerprint(canonical_projection_scope(source_type, config)),
        ]
        for config in _CONFIGS[source_type]
    ]

    assert actual == _EXPECTED[source_type]


def test_progress_nouns_and_available_genes() -> None:
    assert {name: source_progress_unit(name) for name in GENE_REGISTRY} == {
        "agent_session": "item",
        "confluence": "page",
        "github_pages": "page",
        "github_repo": "file",
        "jira": "issue",
        "local_markdown": "file",
        "teams": "message",
    }
    assert source_progress_unit("outlook") == "item"
    assert [metadata.name for metadata in list_available_genes()] == sorted(GENE_REGISTRY)


def test_a_source_type_name_matches_exactly() -> None:
    assert source_type_descriptor("Jira") is None
    assert source_type_descriptor(" jira ") is None
    assert "Jira" not in GENE_REGISTRY
    assert canonical_projection_scope("Jira", {"projects": ["PAY"]}) == {}
    # The local-agent contract accepts the loose spelling its callers send.
    assert local_agent_sync_operation(" JIRA ", {"sync_mode": "local_agent"}) == "jira_sync"


def test_a_built_in_gene_that_cannot_load_fails_the_import() -> None:
    probe = """
import importlib.abc, sys
class Refuse(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "memforge.genes.jira_gene":
            raise ImportError("broken connector")
        return None
sys.meta_path.insert(0, Refuse())
import memforge.genes
"""
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "broken connector" in result.stderr


def test_a_scope_default_names_a_scope_field() -> None:
    from memforge.source_types.descriptor import SourceTypeDescriptor

    with pytest.raises(ValueError, match="not a scope field"):
        SourceTypeDescriptor(
            source_type="example",
            gene="memforge.genes.jira_gene:JiraGene",
            scope_fields=("projects",),
            scope_defaults={"query_mode": "simple"},
        )
