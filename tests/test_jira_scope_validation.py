"""Tests for Jira scope validation across simple and advanced query modes."""

from __future__ import annotations

import pytest

from memforge.server.admin_api import _validate_jira_scope_config


def test_simple_mode_requires_projects():
    with pytest.raises(ValueError):
        _validate_jira_scope_config({"query_mode": "simple"})


def test_default_mode_is_simple_and_requires_projects():
    with pytest.raises(ValueError):
        _validate_jira_scope_config({})


def test_simple_mode_ok_with_projects():
    _validate_jira_scope_config({"query_mode": "simple", "projects": ["PAY"]})


def test_advanced_mode_requires_jql():
    with pytest.raises(ValueError):
        _validate_jira_scope_config({"query_mode": "advanced", "jql": "   "})


def test_advanced_mode_ok_with_jql_and_no_projects():
    _validate_jira_scope_config({"query_mode": "advanced", "jql": "project = PROJ ORDER BY Rank ASC"})
