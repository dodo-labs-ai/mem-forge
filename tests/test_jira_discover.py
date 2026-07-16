"""Tests for Jira issue -> ContentItem mapping, including null optional fields."""

from __future__ import annotations

import pytest

from memforge.genes.jira_gene import _issue_content_item


def test_handles_null_optional_fields():
    # A real Jira issue can carry explicit nulls (no priority/assignee/etc set).
    issue = {
        "id": "10001",
        "key": "PROJ-1",
        "fields": {
            "summary": "Handle orphan data",
            "updated": "2026-06-01T00:00:00.000+0000",
            "priority": None,
            "status": None,
            "project": None,
            "issuetype": None,
            "assignee": None,
            "labels": None,
        },
    }
    item = _issue_content_item(issue, "https://jira.example")
    assert item.item_id == "jira-PROJ-1"
    assert item.title == "PROJ-1: Handle orphan data"
    assert item.space_or_project == ""
    assert item.author is None
    assert item.labels == []
    assert item.extra["status"] == ""
    assert item.extra["priority"] == ""
    assert item.extra["issue_type"] == ""


def test_maps_populated_fields():
    issue = {
        "id": "10002",
        "key": "PROJ-2",
        "fields": {
            "summary": "Migration job",
            "updated": "2026-06-01T12:00:00.000+0000",
            "priority": {"name": "High"},
            "status": {"name": "Open"},
            "project": {"key": "PROJ"},
            "issuetype": {"name": "Story"},
            "assignee": {"displayName": "Jane Doe"},
            "labels": ["a", "b"],
        },
    }
    item = _issue_content_item(issue, "https://jira.example")
    assert item.source_url == "https://jira.example/browse/PROJ-2"
    assert item.space_or_project == "PROJ"
    assert item.author == "Jane Doe"
    assert item.labels == ["a", "b"]
    assert item.extra["status"] == "Open"
    assert item.extra["priority"] == "High"
    assert item.extra["issue_type"] == "Story"


def test_rejects_offsetless_updated_timestamp():
    issue = {
        "id": "10003",
        "key": "PROJ-3",
        "fields": {
            "summary": "Offsetless timestamp",
            "updated": "2026-06-01T12:00:00.000",
        },
    }

    with pytest.raises(RuntimeError, match="timestamp has no timezone"):
        _issue_content_item(issue, "https://jira.example")
