"""MemorySearchRequest scope-mode coercion."""

from __future__ import annotations

import pytest

from memforge.server.admin_api import MemorySearchRequest


def test_project_first_without_active_project_coerces_to_workspace():
    """Project-aware ranking needs an active project. Omitting one falls
    through to flat workspace ranking so the default contract just works."""
    req = MemorySearchRequest(query="test")
    assert req.scope_mode == "workspace"
    assert req.active_project is None


def test_project_mode_without_active_project_coerces_to_workspace():
    req = MemorySearchRequest(query="test", scope_mode="project")
    assert req.scope_mode == "workspace"


def test_project_first_with_active_project_passes_through():
    req = MemorySearchRequest(query="test", active_project="PAY")
    assert req.scope_mode == "project-first"
    assert req.active_project == "PAY"


def test_project_mode_with_active_project_passes_through():
    req = MemorySearchRequest(query="test", scope_mode="project", active_project="PAY")
    assert req.scope_mode == "project"


def test_workspace_mode_passes_through_without_active_project():
    req = MemorySearchRequest(query="test", scope_mode="workspace")
    assert req.scope_mode == "workspace"


def test_invalid_scope_mode_rejected_by_literal():
    with pytest.raises(Exception):
        MemorySearchRequest(query="test", scope_mode="bogus")  # type: ignore[arg-type]


def test_source_filter_accepts_exact_registered_facets():
    req = MemorySearchRequest(
        query="agent memory last week",
        include_private=True,
        active_repo_identifier="github.tools.sap/hcm/memforge-cloud",
        source_filter={
            "source_types": ["agent_session"],
            "clients": ["codex"],
            "repo_identifiers": ["github.tools.sap/hcm/memforge-cloud"],
        },
    )

    assert req.include_private is True
    assert req.active_repo_identifier == "github.tools.sap/hcm/memforge-cloud"
    assert req.source_filter is not None
    assert req.source_filter.source_types == ["agent_session"]
    assert req.source_filter.clients == ["codex"]
    assert req.source_filter.repo_identifiers == [
        "github.tools.sap/hcm/memforge-cloud"
    ]


def test_source_filter_rejects_unknown_source_type():
    with pytest.raises(Exception):
        MemorySearchRequest(
            query="agent memory last week",
            source_filter={"source_types": ["agent-session"]},
        )


def test_source_filter_rejects_unknown_client():
    with pytest.raises(Exception):
        MemorySearchRequest(
            query="agent memory last week",
            source_filter={"clients": ["claude"]},
        )


def test_search_request_rejects_unknown_memory_type():
    with pytest.raises(Exception):
        MemorySearchRequest(query="test", memory_types=["bug"])  # type: ignore[list-item]


def test_search_request_rejects_unknown_status():
    with pytest.raises(Exception):
        MemorySearchRequest(query="test", status="resolved")
