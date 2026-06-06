from memforge.models import (
    SHARED_PROJECT_KEY,
    UNSORTED_PROJECT_KEY,
    Visibility,
)
from memforge.retrieval.access_predicate import (
    is_visible,
    visible_chroma_where,
    visible_sql,
)
from memforge.storage.adapters.context import AccessScope

WORKSPACE = Visibility.WORKSPACE.value
PRIVATE = Visibility.PRIVATE.value


def _scope(*, include_private: bool, user_id: str = "u-1",
           active_project: str | None = None,
           statuses: tuple[str, ...] = ("active",),
           open_projects: frozenset[str] | None = None) -> AccessScope:
    return AccessScope(
        user_id=user_id,
        open_projects=open_projects or frozenset({SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY}),
        member_projects=frozenset(),
        include_private=include_private,
        allowed_statuses=statuses,
        active_project=active_project,
        scope_mode="project-first",
    )


def test_visible_sql_team_excludes_private_branch():
    fragment, params = visible_sql(_scope(include_private=False), "m")
    assert "m.visibility = ?" in fragment
    assert "owner_user_id" not in fragment  # private branch must not appear at all
    # status, visibility=workspace, and project_open must all be in params/fragment
    assert WORKSPACE in params and "active" in params


def test_visible_sql_personalized_includes_owned_private():
    fragment, params = visible_sql(_scope(include_private=True), "m")
    # both branches are present and ORed together inside one parenthesized clause
    assert "m.visibility = ?" in fragment
    assert "m.owner_user_id = ?" in fragment
    assert WORKSPACE in params and PRIVATE in params and "u-1" in params


def test_visible_chroma_where_team_only_workspace():
    where = visible_chroma_where(_scope(include_private=False), memory_types=None)
    # Workspace-only at the access tier: no private branch, no project_key
    # narrowing (the relational post-fusion re-check is the authority on
    # project openness). The "$and" wraps status + visibility=workspace.
    flat = where.get("$and", [where])
    assert any(c == {"visibility": WORKSPACE} for c in flat)
    assert not any(
        (isinstance(c, dict) and c.get("$or"))
        or "project_key" in (c if isinstance(c, dict) else {})
        for c in flat
    )


def test_visible_chroma_where_personalized_or_branches():
    where = visible_chroma_where(_scope(include_private=True), memory_types=None)
    # Top-level: $and(status, $or(workspace branch, private branch))
    or_clause = next(c for c in where["$and"] if "$or" in c)["$or"]
    assert any(b == {"visibility": WORKSPACE} for b in or_clause)
    assert any(
        b.get("$and") and {"visibility": PRIVATE} in b["$and"]
        and {"owner_user_id": "u-1"} in b["$and"]
        for b in or_clause
    )


def test_is_visible_default_deny_unknown_visibility():
    row = {"status": "active", "visibility": None, "owner_user_id": None,
           "project_key": SHARED_PROJECT_KEY}
    assert is_visible(row, _scope(include_private=True)) is False


def test_is_visible_team_excludes_caller_own_private():
    row = {"status": "active", "visibility": PRIVATE,
           "owner_user_id": "u-1", "project_key": SHARED_PROJECT_KEY}
    assert is_visible(row, _scope(include_private=False)) is False
    assert is_visible(row, _scope(include_private=True)) is True


def test_is_visible_personalized_excludes_other_users_private():
    row = {"status": "active", "visibility": PRIVATE,
           "owner_user_id": "u-2", "project_key": SHARED_PROJECT_KEY}
    assert is_visible(row, _scope(include_private=True, user_id="u-1")) is False


def test_is_visible_workspace_dangling_project_treated_as_open():
    # A workspace memory whose project_key has no row in the projects table
    # is treated as open (dangling fail-safe).
    row = {"status": "active", "visibility": WORKSPACE,
           "owner_user_id": None, "project_key": "DANGLING"}
    scope = _scope(include_private=False, open_projects=frozenset({SHARED_PROJECT_KEY,
                                                                   UNSORTED_PROJECT_KEY}))
    assert is_visible(row, scope, dangling_project_keys={"DANGLING"}) is True
