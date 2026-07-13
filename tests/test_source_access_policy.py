from __future__ import annotations

import pytest

from memforge.source_access import (
    SourceAccessPolicy,
    SourceAccessState,
    memory_visibility_for_source,
    source_is_discoverable,
)


def _source(
    *,
    policy: str,
    owner: str = "owner-a",
    state: str = "active",
) -> dict[str, str]:
    return {
        "access_policy": policy,
        "owner_user_id": owner,
        "access_state": state,
    }


def test_private_source_is_discoverable_only_by_owner() -> None:
    source = _source(policy=SourceAccessPolicy.PRIVATE.value)

    assert source_is_discoverable(source, viewer_id="owner-a") is True
    assert source_is_discoverable(source, viewer_id="member-b") is False


def test_workspace_source_is_discoverable_by_every_workspace_member() -> None:
    source = _source(policy=SourceAccessPolicy.WORKSPACE.value)

    assert source_is_discoverable(source, viewer_id="owner-a") is True
    assert source_is_discoverable(source, viewer_id="member-b") is True


def test_changing_source_is_temporarily_discoverable_only_by_owner() -> None:
    source = _source(
        policy=SourceAccessPolicy.WORKSPACE.value,
        state=SourceAccessState.CHANGING.value,
    )

    assert source_is_discoverable(source, viewer_id="owner-a") is True
    assert source_is_discoverable(source, viewer_id="member-b") is False


def test_orphaned_private_source_is_not_discoverable_even_by_prior_owner() -> None:
    source = _source(
        policy=SourceAccessPolicy.PRIVATE.value,
        state=SourceAccessState.ORPHANED_PRIVATE.value,
    )

    assert source_is_discoverable(source, viewer_id="owner-a") is False


def test_memory_visibility_is_materialized_from_source_policy() -> None:
    assert memory_visibility_for_source(_source(policy=SourceAccessPolicy.PRIVATE.value)) == ("private", "owner-a")
    assert memory_visibility_for_source(_source(policy=SourceAccessPolicy.WORKSPACE.value)) == ("workspace", None)


@pytest.mark.parametrize(
    "source",
    [
        {"owner_user_id": "owner-a", "access_state": "active"},
        {"access_policy": "workspace", "access_state": "active"},
        {
            "access_policy": "unexpected",
            "owner_user_id": "owner-a",
            "access_state": "active",
        },
    ],
)
def test_source_access_requires_explicit_valid_policy_and_owner(source: dict[str, str]) -> None:
    with pytest.raises(ValueError):
        memory_visibility_for_source(source)
