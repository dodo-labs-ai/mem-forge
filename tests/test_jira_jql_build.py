"""Tests for Jira JQL construction in simple and advanced query modes."""

from __future__ import annotations

from datetime import datetime, timezone

from memforge.genes.jira_gene import _build_jql


def _since() -> datetime:
    return datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)


def test_simple_mode_builds_project_and_type_clauses():
    jql = _build_jql({"projects": ["PAY", "ARCH"], "issue_types": ["Epic", "Story"]}, None)
    assert jql == "project in (PAY,ARCH) AND issuetype in (Epic,Story) ORDER BY updated DESC"


def test_simple_mode_appends_filter_and_delta():
    jql = _build_jql(
        {"projects": ["PAY"], "issue_types": ["Bug"], "jql_filter": "labels = 'x'"},
        _since(),
    )
    assert jql == (
        "project in (PAY) AND issuetype in (Bug) AND (labels = 'x') "
        "AND updated >= '2026-06-01 12:00' ORDER BY updated DESC"
    )


def test_advanced_mode_injects_delta_before_user_order_by():
    user = 'project = PROJ AND type not in ("Test") ORDER BY Rank ASC'
    jql = _build_jql({"query_mode": "advanced", "jql": user}, _since())
    assert jql == ("(project = PROJ AND type not in (\"Test\")) AND updated >= '2026-06-01 12:00' ORDER BY Rank ASC")


def test_advanced_mode_defaults_order_when_user_has_none():
    jql = _build_jql({"query_mode": "advanced", "jql": "project = PROJ"}, _since())
    assert jql == "(project = PROJ) AND updated >= '2026-06-01 12:00' ORDER BY updated DESC"


def test_advanced_mode_without_since_keeps_user_order():
    user = "project = PROJ ORDER BY Rank ASC"
    jql = _build_jql({"query_mode": "advanced", "jql": user}, None)
    assert jql == "project = PROJ ORDER BY Rank ASC"


def test_advanced_mode_preserves_complex_clauses_verbatim():
    user = (
        'project = PROJ AND type not in ("Test", "Test Plan") '
        'AND ("Agile Team" = "GSHCMNextGenPay-Mount Tai") '
        'AND (fixVersion != "GLO_PAUSE" OR fixVersion is EMPTY) ORDER BY Rank ASC'
    )
    jql = _build_jql({"query_mode": "advanced", "jql": user}, _since())
    assert jql.startswith(
        '(project = PROJ AND type not in ("Test", "Test Plan") '
        'AND ("Agile Team" = "GSHCMNextGenPay-Mount Tai") '
        'AND (fixVersion != "GLO_PAUSE" OR fixVersion is EMPTY)) '
        "AND updated >= '2026-06-01 12:00'"
    )
    assert jql.endswith("ORDER BY Rank ASC")
