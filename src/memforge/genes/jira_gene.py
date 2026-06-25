"""Jira Gene — syncs issues from Jira via REST API.

Wraps the Jira REST API v2 to discover, fetch, and normalize
issues (with comments, links, status history) into comprehensive markdown.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone

import httpx

from memforge.genes.atlassian_auth import (
    atlassian_request_limiter,
    bearer_headers,
    request_with_rate_limit_retry,
    require_https_base_url,
    tls_verify,
)
from memforge.genes.base import Gene
from memforge.models import (
    ConfigField,
    ConfigFieldType,
    ConfigGroup,
    ContentItem,
    GeneConfigSchema,
    GeneMetadata,
    NormalizedContent,
    RawContent,
)

logger = logging.getLogger(__name__)

__all__ = ["JiraGene"]

DEFAULT_ISSUE_TYPES = ["Epic", "Story", "Bug", "Task"]
JIRA_AUTH_MODE_COOKIE = "browser_cookie"
JIRA_AUTH_MODE_PAT = "pat"
DEFAULT_REQUEST_INTERVAL_MS = 750
# Connect is quick even cold; a JQL search against a slow corporate Jira can take
# a while to return, so the read budget is generous.
JIRA_CONNECT_TIMEOUT_SECONDS = 10.0
JIRA_READ_TIMEOUT_SECONDS = 60.0
COMMENT_MAX_RESULTS = 100
HYDRATED_SEARCH_MAX_RESULTS = 25
JIRA_SEARCH_FIELDS = ["*all"]
JIRA_SEARCH_EXPAND = ["changelog", "renderedFields"]

JIRA_QUERY_MODE_SIMPLE = "simple"
JIRA_QUERY_MODE_ADVANCED = "advanced"
_DEFAULT_ORDER_BY = "ORDER BY updated DESC"
_ORDER_BY_RE = re.compile(r"\border\s+by\b", re.IGNORECASE)


def _delta_clause(since: datetime) -> str:
    return f"updated >= '{since.strftime('%Y-%m-%d %H:%M')}'"


def _augment_advanced_jql(raw_jql: str, since: datetime | None) -> str:
    """Use a user-authored JQL as-is, injecting the delta clause before ORDER BY.

    The user's query is authoritative: its ORDER BY (or a default) is preserved,
    and the incremental ``updated >=`` filter is AND-ed onto the where clause
    (wrapped in parentheses so a top-level OR keeps its meaning).
    """
    query = raw_jql.strip()
    match = _ORDER_BY_RE.search(query)
    if match:
        where = query[: match.start()].strip()
        order = query[match.start():].strip()
    else:
        where = query
        order = _DEFAULT_ORDER_BY
    if since:
        delta = _delta_clause(since)
        where = f"({where}) AND {delta}" if where else delta
    return f"{where} {order}".strip()


def _build_jql(config: dict, since: datetime | None) -> str:
    """Build the effective JQL for a sync from the source config.

    In ``advanced`` query mode the configured ``jql`` is authoritative. In
    ``simple`` mode the query is assembled from projects, issue types, and an
    optional refine filter.
    """
    if str(config.get("query_mode") or JIRA_QUERY_MODE_SIMPLE).strip().lower() == JIRA_QUERY_MODE_ADVANCED:
        return _augment_advanced_jql(str(config.get("jql") or ""), since)

    projects = config.get("projects", [])
    if isinstance(projects, str):
        projects = [p.strip() for p in projects.split(",") if p.strip()]
    issue_types = config.get("issue_types", DEFAULT_ISSUE_TYPES)
    if isinstance(issue_types, str):
        issue_types = [t.strip() for t in issue_types.split(",") if t.strip()]
    jql_filter = config.get("jql_filter", "")

    jql = f"project in ({','.join(projects)}) AND issuetype in ({','.join(issue_types)})"
    if jql_filter:
        jql += f" AND ({jql_filter})"
    if since:
        jql += f" AND {_delta_clause(since)}"
    jql += f" {_DEFAULT_ORDER_BY}"
    return jql


def _request_interval_seconds(config: dict) -> float:
    raw_value = config.get("request_interval_ms", DEFAULT_REQUEST_INTERVAL_MS)
    try:
        return max(float(raw_value), 0.0) / 1000.0
    except (TypeError, ValueError):
        return DEFAULT_REQUEST_INTERVAL_MS / 1000.0


def _bool_config(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "0", "no", "off"}
    return bool(value)


def _auth_mode(config: dict) -> str:
    configured = config.get("auth_mode")
    if configured is None and (config.get("pat") or config.get("pat_encrypted")):
        return JIRA_AUTH_MODE_PAT
    mode = str(configured or JIRA_AUTH_MODE_COOKIE).strip().lower()
    if mode not in {JIRA_AUTH_MODE_COOKIE, JIRA_AUTH_MODE_PAT}:
        raise ValueError("Jira Authentication Method must be Browser session or Personal access token")
    return mode


def _jira_headers(config: dict) -> dict[str, str]:
    mode = _auth_mode(config)
    if mode == JIRA_AUTH_MODE_PAT:
        return bearer_headers(config, "Jira")

    cookie = str(config.get("jira_cookie") or "").strip()
    if not cookie:
        raise ValueError("Jira browser session cookie is required")
    return {
        "Accept": "application/json",
        "Cookie": cookie,
    }


def _zero_quota_message(mode: str) -> str:
    if mode == JIRA_AUTH_MODE_PAT:
        return (
            "Jira PAT API quota is zero for this user or token. "
            "Use browser cookie authentication or ask Jira admins to enable REST API quota."
        )
    return (
        "Jira REST API quota is zero for this browser-cookie session. "
        "Ask Jira admins to enable REST API quota for this user or session."
    )


def _issue_payload_from_search(issue: dict, config: dict) -> dict:
    """Build the issue payload used by normalization from the hydrated search result."""
    payload = dict(issue)
    fields = payload.get("fields", {})
    comments = fields.get("comment") if isinstance(fields, dict) else None
    include_comments = _bool_config(config, "include_comments", True)

    if include_comments and isinstance(comments, dict):
        payload["_comments"] = comments.get("comments", [])
        comment_total = comments.get("total")
        if isinstance(comment_total, int) and comment_total > len(payload["_comments"]):
            payload["_comments_truncated"] = {
                "returned": len(payload["_comments"]),
                "total": comment_total,
            }
    else:
        payload["_comments"] = []

    return payload


def _issue_content_item(issue: dict, base_url: str) -> ContentItem:
    """Map a Jira search issue to a ContentItem, tolerating null optional fields.

    Jira returns ``priority``/``assignee``/etc. as explicit ``null`` when unset,
    so ``fields.get(key, {})`` is not safe (the key exists with a None value).
    """
    fields = issue.get("fields") or {}
    key = issue.get("key", "")
    updated_str = fields.get("updated", "")
    try:
        last_modified = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
        if last_modified.tzinfo is None or last_modified.utcoffset() is None:
            last_modified = last_modified.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        last_modified = datetime.now(timezone.utc)
    assignee = fields.get("assignee") or {}
    return ContentItem(
        item_id=f"jira-{key}",
        title=f"{key}: {fields.get('summary', 'Untitled')}",
        source_url=f"{base_url}/browse/{key}",
        last_modified=last_modified,
        content_type="application/json",
        space_or_project=(fields.get("project") or {}).get("key", ""),
        version=updated_str,
        author=assignee.get("displayName") if assignee else None,
        labels=fields.get("labels") or [],
        extra={
            "issue_key": key,
            "status": (fields.get("status") or {}).get("name", ""),
            "priority": (fields.get("priority") or {}).get("name", ""),
            "issue_type": (fields.get("issuetype") or {}).get("name", ""),
        },
    )


class JiraGene(Gene):
    """Jira data source gene.

    Discovers issues via JQL, fetches full issue data including comments
    and linked issues, normalizes to comprehensive markdown.
    """

    @classmethod
    def metadata(cls) -> GeneMetadata:
        return GeneMetadata(
            name="jira",
            display_name="Jira",
            description="Tickets, decisions, and work items",
            default_sync_interval_minutes=360,  # 6 hours
            auth_method="browser_cookie",
            data_shape="ticket",
        )

    @classmethod
    def config_schema(cls) -> GeneConfigSchema:
        return GeneConfigSchema(
            groups=[
                ConfigGroup(key="connection", label="Connection", order=0),
                ConfigGroup(key="scope", label="What to Sync", order=1),
            ],
            fields=[
                ConfigField(
                    key="base_url", label="Jira Base URL",
                    field_type=ConfigFieldType.URL, required=True,
                    placeholder="https://jira.example.com",
                    group="connection", order=0,
                ),
                ConfigField(
                    key="auth_mode", label="Authentication Method",
                    field_type=ConfigFieldType.SELECT, required=True,
                    options=[JIRA_AUTH_MODE_COOKIE, JIRA_AUTH_MODE_PAT],
                    default=JIRA_AUTH_MODE_COOKIE,
                    help_text=(
                        "Browser session uses the local CLI adapter "
                        "(`memforge adapter auth jira refresh --base-url ...`) to "
                        "capture cookies from your signed-in browser. Use this for "
                        "Enterprise Jira where REST API quota is not available. PAT "
                        "mode is only for Jira deployments that grant the user REST API "
                        "quota."
                    ),
                    group="connection", order=1,
                ),
                ConfigField(
                    key="query_mode", label="Query mode",
                    field_type=ConfigFieldType.SELECT, required=False,
                    options=[JIRA_QUERY_MODE_SIMPLE, JIRA_QUERY_MODE_ADVANCED],
                    default=JIRA_QUERY_MODE_SIMPLE,
                    help_text=(
                        "Simple builds the query from projects and issue types. "
                        "Advanced uses a full JQL you provide (authoritative)."
                    ),
                    group="scope", order=0,
                ),
                ConfigField(
                    key="projects", label="Projects to Sync",
                    field_type=ConfigFieldType.TAG_LIST, required=False,
                    placeholder="PAY, ARCH",
                    help_text="Comma-separated Jira project keys (simple mode)",
                    group="scope", order=1,
                ),
                ConfigField(
                    key="jql", label="JQL",
                    field_type=ConfigFieldType.STRING, required=False,
                    placeholder='project = PROJ AND type != "Test" ORDER BY Rank ASC',
                    help_text=(
                        "Full JQL used as-is in advanced mode. MemForge adds an "
                        "incremental updated-since filter before your ORDER BY."
                    ),
                    group="scope", order=2,
                ),
                ConfigField(
                    key="jql_filter", label="JQL Filter (optional)",
                    field_type=ConfigFieldType.STRING, required=False,
                    placeholder="labels = 'important'",
                    help_text="Additional JQL filter to refine results (simple mode)",
                    group="scope", order=3,
                ),
                ConfigField(
                    key="issue_types", label="Issue Types",
                    field_type=ConfigFieldType.MULTI_SELECT, required=False,
                    options=["Epic", "Story", "Bug", "Task", "Sub-task", "Defect"],
                    default="Epic,Story,Bug,Task",
                    group="scope", order=4,
                ),
                ConfigField(
                    key="include_comments", label="Include Comments",
                    field_type=ConfigFieldType.BOOLEAN, required=False,
                    default="true",
                    group="scope", order=5,
                ),
                ConfigField(
                    key="pat", label="Personal Access Token",
                    field_type=ConfigFieldType.SECRET, required=False,
                    help_text="Stored encrypted and sent as a bearer token when PAT mode is selected",
                    group="connection", order=2,
                ),
            ],
        )

    # -------------------------------------------------------------------
    # Instance methods
    # -------------------------------------------------------------------

    async def authenticate(self) -> None:
        """Prepare a Jira client with the configured authentication mode."""
        base_url = self.config.get("base_url", "").rstrip("/")
        if not base_url:
            raise ValueError("Jira base_url is required")
        require_https_base_url(base_url, "Jira")

        mode = _auth_mode(self.config)
        self._base_url = base_url
        self._auth_mode = mode
        client = httpx.AsyncClient(
            base_url=base_url,
            headers=_jira_headers(self.config),
            timeout=httpx.Timeout(JIRA_READ_TIMEOUT_SECONDS, connect=JIRA_CONNECT_TIMEOUT_SECONDS),
            follow_redirects=True,
            verify=tls_verify(self.config),
        )
        self._client = client
        self._request_limiter = atlassian_request_limiter(
            base_url,
            min_interval_seconds=_request_interval_seconds(self.config),
            owner_id=self.source_id,
        )
        self._hydrated_issues: dict[str, dict] = {}
        logger.info("Jira client prepared with %s auth: %s", mode, base_url)

    async def discover(self, since: datetime | None = None) -> AsyncIterator[ContentItem]:
        """Discover issues via JQL search."""
        jql = _build_jql(self.config, since)

        self._hydrated_issues = {}
        start_at = 0
        max_results = HYDRATED_SEARCH_MAX_RESULTS

        while True:
            try:
                resp = await self._request(
                    "POST",
                    "/rest/api/2/search",
                    json_body={
                        "jql": jql,
                        "startAt": start_at,
                        "maxResults": max_results,
                        "fields": JIRA_SEARCH_FIELDS,
                        "expand": JIRA_SEARCH_EXPAND,
                    },
                )
                data = resp.json()
            except Exception as e:
                logger.error("Jira search failed: %s", e)
                raise RuntimeError(f"Jira search failed: {e}") from e

            issues = data.get("issues", [])
            if not issues:
                break

            for issue in issues:
                key = issue.get("key", "")
                self._hydrated_issues[key] = issue
                yield _issue_content_item(issue, self._base_url)

            # Pagination
            if start_at + len(issues) >= data.get("total", 0):
                break
            start_at += max_results

    async def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict | None = None,
        json_body: dict | None = None,
    ) -> httpx.Response:
        try:
            return await request_with_rate_limit_retry(
                self._client,
                method,
                url,
                product_name="Jira",
                params=params,
                json_body=json_body,
                limiter=getattr(self, "_request_limiter", None),
                zero_quota_message=_zero_quota_message(getattr(self, "_auth_mode", _auth_mode(self.config))),
            )
        except httpx.HTTPStatusError as exc:
            if (
                getattr(self, "_auth_mode", _auth_mode(self.config)) == JIRA_AUTH_MODE_COOKIE
                and exc.response.status_code == 401
            ):
                raise RuntimeError("Jira browser session cookie expired or is not accepted. Refresh the cookie.") from exc
            raise

    async def fetch(self, item: ContentItem) -> RawContent:
        """Fetch full issue data with comments and links."""
        key = item.extra.get("issue_key", item.item_id.replace("jira-", ""))
        hydrated_issue = getattr(self, "_hydrated_issues", {}).get(key)
        if isinstance(hydrated_issue, dict):
            payload = _issue_payload_from_search(hydrated_issue, self.config)
            if payload.get("_comments_truncated"):
                await self._top_up_truncated_comments(key, payload)
            getattr(self, "_hydrated_issues", {}).pop(key, None)
            return RawContent(
                item=item,
                body=json.dumps(payload).encode("utf-8"),
                content_type="application/json",
            )

        resp = await self._request(
            "GET",
            f"/rest/api/2/issue/{key}",
            params={"expand": "changelog,renderedFields"},
        )
        data = resp.json()

        # Fetch comments separately if configured
        include_comments = _bool_config(self.config, "include_comments", True)
        if include_comments:
            comments_resp = await self._request(
                "GET",
                f"/rest/api/2/issue/{key}/comment",
                params={"maxResults": COMMENT_MAX_RESULTS},
            )
            data["_comments"] = comments_resp.json().get("comments", [])

        return RawContent(
            item=item,
            body=json.dumps(data).encode("utf-8"),
            content_type="application/json",
        )

    async def _top_up_truncated_comments(self, key: str, payload: dict) -> None:
        comments_resp = await self._request(
            "GET",
            f"/rest/api/2/issue/{key}/comment",
            params={"maxResults": COMMENT_MAX_RESULTS},
        )
        comment_data = comments_resp.json()
        comments = comment_data.get("comments", [])
        payload["_comments"] = comments
        total = comment_data.get("total")
        if isinstance(total, int) and total > len(comments):
            payload["_comments_truncated"] = {
                "returned": len(comments),
                "total": total,
            }
            logger.warning(
                "Jira comments truncated for %s: returned %d of %d comments",
                key,
                len(comments),
                total,
            )
        else:
            payload.pop("_comments_truncated", None)

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        """Convert Jira issue JSON to comprehensive markdown.

        Includes available summary, status, assignee, description, status history,
        issue links, subtasks, and comments so the enricher can extract memories
        from the issue payload.
        """
        data = json.loads(raw.body)
        fields = data.get("fields", {})
        key = raw.item.extra.get("issue_key", "")

        lines = []

        # Header
        issue_type = raw.item.extra.get("issue_type", "Task")
        status = raw.item.extra.get("status", "Unknown")
        priority = raw.item.extra.get("priority", "")
        assignee = raw.item.author or "Unassigned"

        lines.append(f"# [{issue_type}] {raw.item.title}")
        lines.append(f"**Status**: {status} | **Priority**: {priority} | **Assignee**: {assignee}")
        if raw.item.labels:
            lines.append(f"**Labels**: {', '.join(raw.item.labels)}")
        lines.append("")

        # Status history (from changelog)
        changelog = data.get("changelog", {}).get("histories", [])
        status_transitions = []
        for history in changelog:
            for item in history.get("items", []):
                if item.get("field") == "status":
                    created = history.get("created", "")
                    date_str = created[:10] if created else ""
                    status_transitions.append(
                        f"- {item.get('fromString', '?')} -> {item.get('toString', '?')} ({date_str})"
                    )

        if status_transitions:
            lines.append("## Status History")
            lines.extend(status_transitions)
            lines.append("")

        # Description
        description = fields.get("description", "") or ""
        if description:
            lines.append("## Description")
            lines.append(description)
            lines.append("")

        # Issue links
        issue_links = fields.get("issuelinks", [])
        if issue_links:
            lines.append("## Issue Links")
            for link in issue_links:
                link_type = link.get("type", {}).get("outward", link.get("type", {}).get("name", "related"))
                if "outwardIssue" in link:
                    target = link["outwardIssue"]
                    lines.append(f"- {link_type}: {target.get('key', '')} - {target.get('fields', {}).get('summary', '')}")
                elif "inwardIssue" in link:
                    target = link["inwardIssue"]
                    inward_type = link.get("type", {}).get("inward", "related to")
                    lines.append(f"- {inward_type}: {target.get('key', '')} - {target.get('fields', {}).get('summary', '')}")
            lines.append("")

        # Subtasks
        subtasks = fields.get("subtasks", [])
        if subtasks:
            lines.append("## Subtasks")
            for sub in subtasks:
                sub_key = sub.get("key", "")
                sub_summary = sub.get("fields", {}).get("summary", "")
                sub_status = sub.get("fields", {}).get("status", {}).get("name", "")
                lines.append(f"- {sub_key}: {sub_summary} ({sub_status})")
            lines.append("")

        # Comments
        comments = data.get("_comments", [])
        if comments:
            lines.append("## Comments")
            for comment in comments:
                author = comment.get("author", {}).get("displayName", "Unknown")
                created = comment.get("created", "")[:10]
                body = comment.get("body", "")
                lines.append(f"**{author}** ({created}):")
                lines.append(body)
                lines.append("")

        if data.get("_comments_truncated"):
            truncated = data["_comments_truncated"]
            lines.append("## Comment Sync Note")
            lines.append(
                f"Jira returned {truncated.get('returned', 0)} of {truncated.get('total', 0)} comments "
                "for this issue during sync."
            )
            lines.append("")

        markdown_body = "\n".join(lines)

        # Source semantics (structured data for filtering)
        source_semantics = {
            "issue_key": key,
            "status": status,
            "priority": priority,
            "issue_type": issue_type,
            "assignee": assignee,
            "labels": raw.item.labels,
            "status_transitions": [
                {"from": t.split(" -> ")[0].lstrip("- "), "to": t.split(" -> ")[1].split(" (")[0]}
                for t in status_transitions if " -> " in t
            ],
            "linked_issues": [
                link.get("outwardIssue", link.get("inwardIssue", {})).get("key", "")
                for link in issue_links
            ],
        }

        return NormalizedContent(
            item=raw.item,
            markdown_body=markdown_body,
            source_semantics=source_semantics,
        )
