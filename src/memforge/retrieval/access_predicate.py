"""The single default-deny access predicate, shared across every retrieval channel.

One definition, three projections: a SQL WHERE fragment for the relational and
keyword channels, a Chroma where dict for the vector channel, and an in-process
predicate for the post-fusion re-check and tests. A row is visible iff its
status is allowed AND it is on the workspace branch (visibility='workspace',
which covers every workspace-visible row in the bound datastore) OR, only when
scope.include_private is set, the caller's own private branch (visibility='private'
AND owner_user_id = caller).

`scope_mode` decides whether project_key narrows the workspace branch. In
``project-first`` (the default) and ``workspace`` modes the workspace branch
keeps every project_key untouched and the ranker handles affinity weighting.
In ``project`` mode the workspace branch narrows upstream to the active project
plus SHARED: UNSORTED and other projects are pruned at the predicate, never
returned.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from memforge.models import (
    SHARED_PROJECT_KEY,
    Visibility,
)
from memforge.storage.adapters.context import AccessScope

__all__ = ["is_visible", "visible_chroma_where", "visible_sql"]


def _project_mode_keys(scope: AccessScope) -> tuple[str, ...]:
    """Return the project keys that satisfy the workspace branch in
    ``project`` mode. Only the active project plus SHARED qualify;
    UNSORTED and every other project are pruned.
    """
    keys: set[str] = {SHARED_PROJECT_KEY}
    if scope.active_project:
        keys.add(scope.active_project)
    return tuple(sorted(keys))


def visible_sql(scope: AccessScope, alias: str) -> tuple[str, list[Any]]:
    """Return (sql_fragment, params) for the WHERE of a SQL query against memories.

    The fragment is parenthesized and parameter-bound. Always include the status
    list, then the workspace branch, then the private branch only when allowed.
    """
    statuses = list(scope.allowed_statuses)
    params: list[Any] = []
    parts: list[str] = []

    status_placeholders = ",".join("?" for _ in statuses)
    parts.append(f"{alias}.status IN ({status_placeholders})")
    params.extend(statuses)

    if scope.scope_mode == "project":
        # Hard narrowing in project mode: the workspace branch admits a
        # candidate only when its project_key is the active project or
        # SHARED. Other rows (UNSORTED, RISK, dangling keys) are pruned
        # at the predicate.
        narrow_keys = list(_project_mode_keys(scope))
        project_placeholders = ",".join("?" for _ in narrow_keys)
        workspace_branch = (
            f"({alias}.visibility = ? AND "
            f"{alias}.project_key IN ({project_placeholders}))"
        )
        params.append(Visibility.WORKSPACE.value)
        params.extend(narrow_keys)
    else:
        # Project-first and workspace modes weight cross-project hits via
        # the ranker. The predicate keeps every workspace row visible
        # regardless of project_key, so adding a real `projects` row for
        # a project never silently drops candidates from results.
        workspace_branch = f"({alias}.visibility = ?)"
        params.append(Visibility.WORKSPACE.value)

    branches = [workspace_branch]

    if scope.include_private:
        branches.append(
            f"({alias}.visibility = ? AND {alias}.owner_user_id = ?)"
        )
        params.append(Visibility.PRIVATE.value)
        params.append(scope.user_id)

    parts.append("(" + " OR ".join(branches) + ")")
    return "(" + " AND ".join(parts) + ")", params


def visible_chroma_where(
    scope: AccessScope,
    memory_types: list[str] | None,
) -> Mapping[str, Any]:
    """Return a Chroma where dict equivalent to visible_sql at the access tier.

    Project narrowing is intentionally NOT encoded at the vector tier here:
    the post-fusion `filter_visible_ids` reapplies the full SQL predicate and
    is the authoritative re-check for any candidate the vector channel
    returns. The pre-filter only narrows by what Chroma can express safely
    (visibility, status, memory_type).
    """
    statuses = list(scope.allowed_statuses)
    clauses: list[Mapping[str, Any]] = []

    if memory_types:
        clauses.append({"memory_type": {"$in": list(memory_types)}})
    clauses.append({"status": {"$in": statuses}} if len(statuses) > 1
                   else {"status": statuses[0]})

    workspace_branch: Mapping[str, Any] = {"visibility": Visibility.WORKSPACE.value}

    if scope.include_private:
        private_branch: Mapping[str, Any] = {
            "$and": [
                {"visibility": Visibility.PRIVATE.value},
                {"owner_user_id": scope.user_id},
            ]
        }
        clauses.append({"$or": [workspace_branch, private_branch]})
    else:
        clauses.append(workspace_branch)

    return {"$and": clauses} if len(clauses) > 1 else clauses[0]


def is_visible(
    row: Mapping[str, Any],
    scope: AccessScope,
    *,
    dangling_project_keys: Iterable[str] | None = None,
) -> bool:
    """In-process predicate. Default-deny: unknown visibility is hidden.

    `dangling_project_keys` is accepted for backward-compat with callers
    that pass it, but it is never read: the SQL predicate no longer
    consults a dangling-key fallback because workspace visibility itself
    spans every project in the relevance-weighted modes.
    """
    del dangling_project_keys  # retained for API compatibility
    if row.get("status") not in scope.allowed_statuses:
        return False
    visibility = row.get("visibility")
    if visibility == Visibility.WORKSPACE.value:
        if scope.scope_mode == "project":
            project_key = row.get("project_key") or ""
            return project_key in _project_mode_keys(scope)
        return True
    if visibility == Visibility.PRIVATE.value and scope.include_private:
        return row.get("owner_user_id") == scope.user_id
    return False
