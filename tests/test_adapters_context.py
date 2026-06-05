"""AccessScope value object: the per-request caller context."""

from __future__ import annotations

import dataclasses

import pytest

from memforge.storage.adapters.context import AccessScope, LOCAL_DEV_USER_ID


def test_local_dev_user_id_is_the_single_local_caller():
    assert LOCAL_DEV_USER_ID == "dev"


def test_access_scope_is_frozen():
    scope = AccessScope(
        user_id=LOCAL_DEV_USER_ID,
        open_projects=frozenset({"SHARED"}),
        member_projects=frozenset({"SHARED"}),
        include_private=False,
        allowed_statuses=("active",),
        active_project=None,
        scope_mode="project-first",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        scope.user_id = "someone-else"  # type: ignore[misc]


def test_access_scope_carries_relevance_and_lifecycle_fields():
    scope = AccessScope(
        user_id=LOCAL_DEV_USER_ID,
        open_projects=frozenset({"SHARED", "PAY"}),
        member_projects=frozenset({"SHARED", "PAY"}),
        include_private=True,
        allowed_statuses=("active", "superseded"),
        active_project="PAY",
        scope_mode="workspace",
    )
    assert scope.active_project == "PAY"
    assert scope.scope_mode == "workspace"
    assert scope.include_private is True
    assert "superseded" in scope.allowed_statuses
