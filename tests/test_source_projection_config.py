from memforge.source_projection_config import (
    canonical_projection_scope,
    canonical_provider_namespace,
    projection_scope_transition_id,
)


def test_operational_and_secret_fields_do_not_change_projection_scope() -> None:
    first = canonical_projection_scope(
        "github_repo",
        {
            "repo_url": "https://github.com/acme/payroll",
            "ref": "main",
            "include_paths": ["src/**", "docs/**"],
            "max_files": 1000,
            "pat": "secret-a",
        },
    )
    second = canonical_projection_scope(
        "github_repo",
        {
            "repo_url": "https://github.com/acme/payroll",
            "ref": "main",
            "include_paths": ["docs/**", "src/**"],
            "max_files": 5000,
            "pat": "secret-b",
        },
    )

    assert first == second == {"ref": "main", "include_paths": ["docs/**", "src/**"]}


def test_namespace_is_separate_from_scope() -> None:
    config = {
        "base_url": "https://jira.example.test/",
        "projects": ["PAY"],
        "include_comments": True,
        "request_interval_ms": 100,
    }

    assert canonical_provider_namespace("jira", config) == {
        "base_url": "https://jira.example.test"
    }
    assert canonical_projection_scope("jira", config) == {
        "query_mode": "projects",
        "projects": ["PAY"],
        "include_comments": True,
    }


def test_teams_window_shape_settings_are_projection_scope() -> None:
    scope = canonical_projection_scope(
        "teams",
        {
            "region": "emea",
            "conversation_ids": ["19:b", "19:a"],
            "max_age_days": 14,
            "conversation_gap_minutes": 60,
            "max_block_messages": 100,
            "conversation_fetch_timeout_seconds": 300,
        },
    )

    assert scope == {
        "conversation_ids": ["19:a", "19:b"],
        "max_age_days": 14,
        "conversation_gap_minutes": 60,
        "max_block_messages": 100,
    }


def test_scope_transition_id_is_order_independent_and_target_specific() -> None:
    first = projection_scope_transition_id(
        "src-1",
        {"projects": ["A", "B"], "comments": True},
        {"projects": ["C"]},
    )
    retry = projection_scope_transition_id(
        "src-1",
        {"comments": True, "projects": ["A", "B"]},
        {"projects": ["C"]},
    )

    assert first == retry
    assert first != projection_scope_transition_id(
        "src-1",
        {"projects": ["A", "B"], "comments": True},
        {"projects": ["D"]},
    )


def test_scope_transition_id_distinguishes_a_repeated_transition_cycle() -> None:
    first_a_to_b = projection_scope_transition_id(
        "src-1",
        {"include_paths": ["docs"]},
        {"include_paths": ["other"]},
        predecessor_transition_id=None,
    )
    b_to_a = projection_scope_transition_id(
        "src-1",
        {"include_paths": ["other"]},
        {"include_paths": ["docs"]},
        predecessor_transition_id=first_a_to_b,
    )
    second_a_to_b = projection_scope_transition_id(
        "src-1",
        {"include_paths": ["docs"]},
        {"include_paths": ["other"]},
        predecessor_transition_id=b_to_a,
    )
    retry = projection_scope_transition_id(
        "src-1",
        {"include_paths": ["docs"]},
        {"include_paths": ["other"]},
        predecessor_transition_id=b_to_a,
    )

    assert second_a_to_b != first_a_to_b
    assert retry == second_a_to_b
