"""Stable Jira sub-observation identity shared by write, replay, and projection."""

from __future__ import annotations

from collections.abc import Mapping

JIRA_CORE_FIELD_KEYS = frozenset(
    {
        "summary",
        "description",
        "status",
        "priority",
        "assignee",
        "labels",
        "resolution",
        "updated",
    }
)


def validate_jira_observation_identities(payload: Mapping[str, object]) -> None:
    """Reject Jira snapshots that cannot preserve complete stable lineage."""

    issue_id = str(payload.get("id") or "").strip()
    issue_key = str(payload.get("key") or "").strip()
    if not issue_id.isdigit() or not issue_key:
        raise ValueError("Jira issue is missing a stable provider id or key")
    fields = payload.get("fields")
    if not isinstance(fields, Mapping):
        raise ValueError("Jira issue fields must be an object")
    missing_core_fields = sorted(JIRA_CORE_FIELD_KEYS.difference(fields))
    if missing_core_fields:
        raise ValueError(f"Jira issue is missing core fields: {', '.join(missing_core_fields)}")

    if "_comments" not in payload:
        raise ValueError("Jira comments collection evidence is missing")
    comments = payload.get("_comments")
    if not isinstance(comments, list):
        raise ValueError("Jira comments must be a list")
    comment_ids: set[str] = set()
    for comment in comments:
        comment_id = str(comment.get("id") or "").strip() if isinstance(comment, Mapping) else ""
        if not comment_id:
            raise ValueError("Jira comment is missing a stable provider id")
        if comment_id in comment_ids:
            raise ValueError(f"Jira comments contain duplicate provider id {comment_id}")
        comment_ids.add(comment_id)
    comments_included = payload.get("_comments_included")
    comments_total = payload.get("_comments_total")
    if not isinstance(comments_included, bool):
        raise ValueError("Jira comments scope evidence is missing")
    if (
        not isinstance(comments_total, int)
        or isinstance(comments_total, bool)
        or comments_total < len(comments)
    ):
        raise ValueError("Jira comments total is invalid")
    if not comments_included and (comments or comments_total != 0):
        raise ValueError("Jira excluded comments snapshot is inconsistent")
    truncated = payload.get("_comments_truncated")
    if comments_included and comments_total > len(comments):
        if not isinstance(truncated, Mapping) or (
            truncated.get("returned") != len(comments)
            or truncated.get("total") != comments_total
        ):
            raise ValueError("Jira truncated comments evidence is missing")
    elif truncated is not None:
        raise ValueError("Jira comments truncation evidence is inconsistent")

    changelog = payload.get("changelog")
    if not isinstance(changelog, Mapping):
        raise ValueError("Jira changelog must be an object")
    histories = changelog.get("histories")
    if not isinstance(histories, list):
        raise ValueError("Jira changelog histories must be a list")
    if changelog.get("startAt") != 0:
        raise ValueError("Jira changelog does not start at zero")
    history_total = changelog.get("total")
    if (
        not isinstance(history_total, int)
        or isinstance(history_total, bool)
        or history_total < len(histories)
    ):
        raise ValueError("Jira changelog total is invalid")
    history_ids: set[str] = set()
    for history in histories:
        history_id = str(history.get("id") or "").strip() if isinstance(history, Mapping) else ""
        if not history_id:
            raise ValueError("Jira changelog history is missing a stable provider id")
        if history_id in history_ids:
            raise ValueError(f"Jira changelog contains duplicate provider id {history_id}")
        history_ids.add(history_id)
