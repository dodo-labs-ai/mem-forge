"""The single default-deny access predicate, shared across every retrieval channel.

One definition, three projections: a SQL WHERE fragment for the relational and
keyword channels, a Chroma where dict for the vector channel, and an in-process
predicate for the post-fusion re-check and tests. A row is visible iff its
status is allowed AND it is on the workspace branch (visibility='workspace' AND
project_open) OR, only when scope.include_private is set, the caller's own
private branch (visibility='private' AND owner_user_id = caller).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from memforge.models import (
    SHARED_PROJECT_KEY,
    UNSORTED_PROJECT_KEY,
    Visibility,
)
from memforge.storage.adapters.context import AccessScope

__all__ = ["is_visible", "visible_chroma_where", "visible_sql"]


# Reserved keys that always count as open. project_open(key, scope) is
# satisfied iff key is in scope.open_projects, plus the dangling fail-safe.
_ALWAYS_OPEN_KEYS = frozenset({SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY})


def _open_project_keys(scope: AccessScope) -> tuple[str, ...]:
    keys = set(scope.open_projects) | _ALWAYS_OPEN_KEYS
    if scope.active_project:
        keys.add(scope.active_project)
    return tuple(sorted(keys))


def visible_sql(scope: AccessScope, alias: str) -> tuple[str, list[Any]]:
    """Return (sql_fragment, params) for the WHERE of a SQL query against memories.

    The fragment is parenthesized and parameter-bound. Always include the status
    list, then the workspace branch, then the private branch only when allowed.
    """
    statuses = list(scope.allowed_statuses)
    open_keys = list(_open_project_keys(scope))
    params: list[Any] = []
    parts: list[str] = []

    status_placeholders = ",".join("?" for _ in statuses)
    parts.append(f"{alias}.status IN ({status_placeholders})")
    params.extend(statuses)

    project_placeholders = ",".join("?" for _ in open_keys)
    workspace_branch = (
        f"({alias}.visibility = ? AND ("
        f"{alias}.project_key IN ({project_placeholders}) OR "
        f"{alias}.project_key NOT IN (SELECT project_key FROM projects)"
        "))"
    )
    branches = [workspace_branch]
    params.append(Visibility.WORKSPACE.value)
    params.extend(open_keys)

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

    Chroma's filter language has no NOT-IN-against-a-foreign-table primitive,
    so the workspace branch deliberately filters on `visibility` (and status,
    memory_type) only. Project narrowing is intentionally NOT encoded at the
    vector tier here: the post-fusion `filter_visible_ids` reapplies the full
    SQL predicate (including the dangling-project fallback) and is the
    authoritative re-check for any candidate the vector channel returns.
    Encoding `project_key IN (open_keys)` in Chroma alone would silently drop a
    legitimately-visible vector candidate whose project_key has no row in the
    `projects` table. The relational re-check decides; the vector pre-filter
    only narrows by what it can express safely.
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
    """In-process predicate. Default-deny: unknown visibility is hidden."""
    if row.get("status") not in scope.allowed_statuses:
        return False
    visibility = row.get("visibility")
    if visibility == Visibility.WORKSPACE.value:
        project_key = row.get("project_key") or ""
        if project_key in _open_project_keys(scope):
            return True
        if dangling_project_keys is not None and project_key in dangling_project_keys:
            return True
        return False
    if visibility == Visibility.PRIVATE.value and scope.include_private:
        return row.get("owner_user_id") == scope.user_id
    return False
