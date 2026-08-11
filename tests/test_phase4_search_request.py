"""MemorySearchRequest scope-mode coercion."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from memforge.server.admin_api import MemorySearchRequest, RecentMemoryListRequest


def test_project_first_without_active_project_coerces_to_workspace():
    """Project-aware ranking needs an active project. Omitting one falls
    through to flat workspace ranking so the default contract just works."""
    req = MemorySearchRequest(query="test")
    assert req.scope_mode == "workspace"
    assert req.active_project is None


def test_search_request_defaults_to_twenty_results():
    req = MemorySearchRequest(query="test")

    assert req.top_k == 20


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


@pytest.mark.parametrize("intent", ["general_hybrid", "known_item", "relationship"])
def test_search_request_accepts_ranked_retrieval_intent(intent):
    req = MemorySearchRequest(query="payroll lock behavior", intent=intent)

    assert req.intent == intent


def test_search_request_rejects_unknown_retrieval_intent():
    with pytest.raises(Exception):
        MemorySearchRequest(query="payroll lock behavior", intent="semantic_lookup")


def test_queryless_compatibility_rejects_ranked_retrieval_intent():
    with pytest.raises(Exception, match="intent requires a ranked query"):
        MemorySearchRequest(
            intent="general_hybrid",
            source_filter={"source_ids": ["src-mounttai"]},
        )


def test_source_filter_accepts_exact_agent_session_facets():
    req = MemorySearchRequest(
        query="agent memory last week",
        include_private=True,
        active_repo_identifier="github.tools.sap/hcm/memforge-cloud",
        source_filter={
            "clients": ["codex"],
            "repo_identifiers": ["github.tools.sap/hcm/memforge-cloud"],
        },
    )

    assert req.include_private is True
    assert req.active_repo_identifier == "github.tools.sap/hcm/memforge-cloud"
    assert req.source_filter is not None
    assert req.source_filter.clients == ["codex"]
    assert req.source_filter.repo_identifiers == ["github.tools.sap/hcm/memforge-cloud"]


def test_source_filter_accepts_exact_source_ids():
    req = MemorySearchRequest(
        source_filter={"source_ids": ["src-mounttai", "src-sfpay"]},
        time_range={"start_date": "2026-06-19"},
    )

    assert req.query == ""
    assert req.source_filter is not None
    assert req.source_filter.source_ids == ["src-mounttai", "src-sfpay"]


def test_source_filter_rejects_empty_source_ids():
    with pytest.raises(Exception):
        MemorySearchRequest(
            source_filter={"source_ids": []},
            time_range={"start_date": "2026-06-19"},
        )


def test_queryless_search_requires_deterministic_filter():
    with pytest.raises(Exception):
        MemorySearchRequest()


def test_search_request_rejects_legacy_top_level_sources_filter():
    with pytest.raises(Exception):
        MemorySearchRequest(query="jira defects", sources=["Matterhorn Defects"])  # type: ignore[call-arg]


def test_source_filter_rejects_source_type_selector():
    with pytest.raises(Exception):
        MemorySearchRequest(
            query="agent memory last week",
            source_filter={"source_types": ["agent_session"]},
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


def test_time_range_accepts_date_only_open_ended_bounds():
    req = MemorySearchRequest(
        query="jira memories",
        time_range={"start_date": "2026-06-19", "date_type": "source_updated_at"},
    )

    assert req.time_range is not None
    assert req.time_range.start_date.isoformat() == "2026-06-19"
    assert req.time_range.end_date is None
    converted = req.time_range.to_time_range()
    assert converted.date_type == "source_updated_at"
    assert converted.after == datetime(2026, 6, 19, tzinfo=timezone.utc)
    assert converted.before is None


def test_time_range_defaults_to_source_updated_at_and_converts_end_date_half_open():
    req = MemorySearchRequest(
        query="recent memory updates",
        time_range={"end_date": "2026-06-26"},
    )

    assert req.time_range is not None
    converted = req.time_range.to_time_range()
    assert converted.date_type == "source_updated_at"
    assert converted.after is None
    assert converted.before == datetime(2026, 6, 27, tzinfo=timezone.utc)


def test_recent_memory_request_resolves_timezone_qualified_half_open_window_to_utc():
    req = RecentMemoryListRequest(
        source_ids=["src-jira"],
        memory_types=["decision"],
        time_range={
            "date_type": "source_updated_at",
            "start_at": "2026-07-21T00:00:00+08:00",
            "end_at": "2026-07-28T00:00:00+08:00",
        },
        page_size=25,
    )

    resolved = req.time_range.to_time_range()

    assert resolved.date_type == "source_updated_at"
    assert resolved.after == datetime(2026, 7, 20, 16, tzinfo=timezone.utc)
    assert resolved.before == datetime(2026, 7, 27, 16, tzinfo=timezone.utc)
    assert req.source_ids == ["src-jira"]
    assert req.memory_types == ["decision"]
    assert req.page_size == 25


@pytest.mark.parametrize(
    "payload",
    [
        {
            "time_range": {
                "start_at": "2026-07-21T00:00:00",
                "end_at": "2026-07-28T00:00:00+08:00",
            }
        },
        {
            "time_range": {
                "start_at": "2026-07-28T00:00:00+08:00",
                "end_at": "2026-07-21T00:00:00+08:00",
            }
        },
        {
            "query": "updates",
            "time_range": {
                "start_at": "2026-07-21T00:00:00+08:00",
                "end_at": "2026-07-28T00:00:00+08:00",
            },
        },
    ],
)
def test_recent_memory_request_rejects_ambiguous_window_or_semantic_query(payload):
    with pytest.raises(Exception):
        RecentMemoryListRequest(**payload)


@pytest.mark.parametrize(
    "time_range",
    [
        {},
        {"start_date": "2026-06-20T00:00:00Z"},
        {"after": "2026-06-20"},
        {"date_type": "created_at", "start_date": "2026-06-20"},
        {"start_date": "2026-06-21", "end_date": "2026-06-20"},
        {"start_date": "2026-02-30"},
    ],
)
def test_time_range_rejects_invalid_shapes(time_range):
    with pytest.raises(Exception):
        MemorySearchRequest(query="test", time_range=time_range)
