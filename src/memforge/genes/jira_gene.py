"""Jira Gene — syncs issues from Jira via REST API.

Wraps the Jira REST API v2 to discover, fetch, and normalize
issues (with comments, links, status history) into comprehensive markdown.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from memforge.genes.atlassian_auth import (
    atlassian_request_limiter,
    bearer_headers,
    request_with_rate_limit_retry,
    require_https_base_url,
    stream_with_rate_limit_retry,
    tls_verify,
)
from memforge.genes.base import Gene
from memforge.genes.local_adapter_packages import (
    has_package_manifest,
    package_manifest,
    read_package_body,
)
from memforge.local_agent.jira_contract import validate_jira_observation_identities
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
from memforge.source_artifacts import (
    MAX_SOURCE_ARTIFACT_DESCRIPTORS_PER_UNIT,
    MAX_SOURCE_ARTIFACTS_PER_UNIT,
    MAX_SOURCE_ARTIFACT_STORAGE_BYTES,
    MAX_SOURCE_ARTIFACT_STORAGE_BYTES_PER_UNIT,
    RawSourceArtifact,
    SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES,
    SourceArtifactContractError,
    SourceArtifactDownload,
    normalize_source_artifact_media_type,
    parse_source_artifact_content_length,
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
JIRA_INVENTORY_FIELDS = ["summary", "updated", "project"]
JIRA_INVENTORY_MAX_RESULTS = 100
LOCAL_AGENT_JIRA_PACKAGE_KIND = "jira_document"

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


def _parse_local_package_dt(value: str) -> datetime:
    if not value:
        return datetime.fromtimestamp(0, timezone.utc)
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _issue_payload_from_search(issue: dict, config: dict) -> dict:
    """Build the issue payload used by normalization from the hydrated search result."""
    payload = dict(issue)
    fields = payload.get("fields")
    if not isinstance(fields, dict):
        raise RuntimeError("Jira issue response is missing fields")
    include_comments = _bool_config(config, "include_comments", True)

    if include_comments:
        comments = fields.get("comment")
        if not isinstance(comments, dict) or not isinstance(comments.get("comments"), list):
            raise RuntimeError("Jira issue response is missing the comments collection")
        if comments.get("startAt") != 0:
            raise RuntimeError("Jira issue comments collection does not start at zero")
        payload["_comments"] = comments["comments"]
        comment_total = comments.get("total")
        if (
            not isinstance(comment_total, int)
            or isinstance(comment_total, bool)
            or comment_total < len(payload["_comments"])
        ):
            raise RuntimeError("Jira issue comments total is invalid")
        if comment_total > len(payload["_comments"]):
            payload["_comments_truncated"] = {
                "returned": len(payload["_comments"]),
                "total": comment_total,
            }
        payload["_comments_total"] = comment_total
    else:
        payload["_comments"] = []
        payload["_comments_total"] = 0
    payload["_comments_included"] = include_comments

    _mark_changelog_completeness(payload)
    validate_jira_observation_identities(payload)
    return payload


def _mark_changelog_completeness(payload: dict) -> None:
    """Make Jira's embedded changelog pagination explicit to projection."""

    changelog = payload.get("changelog")
    if not isinstance(changelog, dict):
        raise RuntimeError("Jira issue response is missing changelog")
    histories = changelog.get("histories")
    if not isinstance(histories, list):
        raise RuntimeError("Jira issue changelog histories are invalid")
    returned = len(histories)
    total = changelog.get("total")
    if not isinstance(total, int) or isinstance(total, bool) or total < returned:
        raise RuntimeError("Jira issue changelog total is invalid")
    if total > returned:
        payload["_changelog_truncated"] = {
            "returned": returned,
            "total": total,
        }


def _issue_content_item(issue: dict, base_url: str) -> ContentItem:
    """Map a Jira search issue to a ContentItem, tolerating null optional fields.

    Jira returns ``priority``/``assignee``/etc. as explicit ``null`` when unset,
    so ``fields.get(key, {})`` is not safe (the key exists with a None value).
    """
    fields = issue["fields"]
    key = str(issue["key"])
    updated_str = fields["updated"]
    try:
        last_modified = datetime.fromisoformat(updated_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError) as exc:
        raise RuntimeError(f"Jira issue {key} has an invalid updated timestamp") from exc
    if last_modified.tzinfo is None or last_modified.utcoffset() is None:
        raise RuntimeError(f"Jira issue {key} updated timestamp has no timezone")
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
            "issue_id": str(issue["id"]),
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
            execution_kinds=("server", "local_agent"),
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
                    key="sync_mode", label="Sync Location",
                    field_type=ConfigFieldType.SELECT, required=False,
                    options=["cloud", "local_agent"],
                    default="cloud",
                    help_text=(
                        "Cloud runs Jira sync from the MemForge service. Local agent runs Jira sync "
                        "from the machine where the daemon is running, useful for VPN-only Jira."
                    ),
                    group="connection", order=2,
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
                    group="connection", order=3,
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
        sync_mode = str(self.config.get("sync_mode") or "cloud").strip().lower()
        if sync_mode == "local_agent":
            if has_package_manifest(self.config):
                self._client = None
                self._request_limiter = None
                self._hydrated_issues = {}
                logger.info("Jira client using local-agent package manifest")
                return
            local_documents_dir = self._local_agent_documents_dir()
            if local_documents_dir is None:
                raise ValueError("Jira local-agent sync requires a local package inbox")
            local_documents_dir.mkdir(parents=True, exist_ok=True)
            self._client = None
            self._request_limiter = None
            self._hydrated_issues = {}
            logger.info("Jira client using local-agent package inbox: %s", local_documents_dir)
            return
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
        if str(self.config.get("sync_mode") or "cloud").strip().lower() == "local_agent":
            manifest = self._local_agent_package_manifest()
            if has_package_manifest(self.config):
                async for item in self._discover_local_agent_package_manifest(manifest, since):
                    yield item
                return
            local_documents_dir = self._local_agent_documents_dir()
            if local_documents_dir is None:
                raise ValueError("Jira local-agent sync requires a local package inbox")
            async for item in self._discover_local_agent_packages(local_documents_dir, since):
                yield item
            return

        self._hydrated_issues = {}
        async for item in self._discover_remote_search(
            since,
            fields=JIRA_SEARCH_FIELDS,
            expand=JIRA_SEARCH_EXPAND,
            max_results=HYDRATED_SEARCH_MAX_RESULTS,
            cache_hydrated=True,
        ):
            yield item

    async def discover_inventory(self) -> AsyncIterator[ContentItem]:
        """Discover a complete lightweight issue inventory for daemon planning."""
        if str(self.config.get("sync_mode") or "cloud").strip().lower() == "local_agent":
            raise ValueError("Jira inventory discovery requires direct provider access")
        self._hydrated_issues = {}
        async for item in self._discover_remote_search(
            None,
            fields=JIRA_INVENTORY_FIELDS,
            expand=None,
            max_results=JIRA_INVENTORY_MAX_RESULTS,
            cache_hydrated=False,
        ):
            yield item

    async def _discover_remote_search(
        self,
        since: datetime | None,
        *,
        fields: list[str],
        expand: list[str] | None,
        max_results: int,
        cache_hydrated: bool,
    ) -> AsyncIterator[ContentItem]:
        jql = _build_jql(self.config, since)
        seen_issue_ids: set[str] = set()
        seen_issue_keys: set[str] = set()
        expected_total: int | None = None
        start_at = 0

        while True:
            try:
                request_body: dict[str, Any] = {
                    "jql": jql,
                    "startAt": start_at,
                    "maxResults": max_results,
                    "fields": fields,
                }
                if expand:
                    request_body["expand"] = expand
                resp = await self._request(
                    "POST",
                    "/rest/api/2/search",
                    json_body=request_body,
                )
                data = resp.json()
            except Exception as e:
                logger.error("Jira search failed: %s", e)
                raise RuntimeError(f"Jira search failed: {e}") from e

            if not isinstance(data, dict) or "issues" not in data or not isinstance(data.get("issues"), list):
                raise RuntimeError("Jira search response is missing an issues list")
            issues = data["issues"]
            if any(not isinstance(issue, dict) for issue in issues):
                raise RuntimeError("Jira search response contains an invalid issue record")
            total = data.get("total")
            if not isinstance(total, int) or isinstance(total, bool) or total < 0:
                raise RuntimeError("Jira search response is missing a valid total")
            if expected_total is None:
                expected_total = total
            elif total != expected_total:
                raise RuntimeError("Jira search total changed during pagination")
            response_start = data.get("startAt")
            if not isinstance(response_start, int) or isinstance(response_start, bool) or response_start != start_at:
                raise RuntimeError("Jira search response startAt does not match the requested page")
            if not issues:
                if start_at < total:
                    raise RuntimeError("Jira search ended before the declared total was reached")
                break

            for issue in issues:
                issue_id = str(issue.get("id") or "").strip()
                key = str(issue.get("key") or "").strip()
                issue_fields = issue.get("fields")
                if not issue_id.isdigit() or not key or not isinstance(issue_fields, dict):
                    raise RuntimeError("Jira search issue is missing a stable id, key, or fields")
                if issue_id in seen_issue_ids or key in seen_issue_keys:
                    raise RuntimeError("Jira search returned duplicate issue identity")
                seen_issue_ids.add(issue_id)
                seen_issue_keys.add(key)
                item = _issue_content_item(issue, self._base_url)
                if cache_hydrated:
                    self._hydrated_issues[key] = issue
                else:
                    item.extra["attest_materialized_revision"] = True
                yield item

            # Pagination
            if start_at + len(issues) >= total:
                break
            start_at += len(issues)
        if len(seen_issue_ids) != (expected_total or 0):
            raise RuntimeError("Jira search unique issue count did not match total")
        self.attest_discovery_complete("jira_search_total_exhausted")

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
        if item.extra.get("package_uri") or item.extra.get("package_path"):
            return RawContent(
                item=item,
                body=read_package_body(self, item, source_label="Jira"),
                content_type="application/json",
            )

        key = item.extra.get("issue_key", item.item_id.replace("jira-", ""))
        hydrated_issue = getattr(self, "_hydrated_issues", {}).get(key)
        if isinstance(hydrated_issue, dict):
            payload = _issue_payload_from_search(hydrated_issue, self.config)
            if payload.get("_comments_truncated"):
                await self._top_up_truncated_comments(key, payload)
            getattr(self, "_hydrated_issues", {}).pop(key, None)
            artifacts = await self._fetch_source_artifacts(payload)
            return RawContent(
                item=item,
                body=json.dumps(payload).encode("utf-8"),
                content_type="application/json",
                artifacts=artifacts,
            )

        resp = await self._request(
            "GET",
            f"/rest/api/2/issue/{key}",
            params={"expand": "changelog,renderedFields"},
        )
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError("Jira issue response must be an object")
        if str(data.get("key") or "").strip() != str(key):
            raise RuntimeError("Jira issue response identity mismatch")
        _mark_changelog_completeness(data)

        # Fetch comments separately if configured
        include_comments = _bool_config(self.config, "include_comments", True)
        if include_comments:
            comments_resp = await self._request(
                "GET",
                f"/rest/api/2/issue/{key}/comment",
                params={"maxResults": COMMENT_MAX_RESULTS},
            )
            comment_data = comments_resp.json()
            data["_comments"] = self._validated_comment_page(comment_data)
            total = comment_data["total"]
            data["_comments_total"] = total
            data["_comments_included"] = True
            if total > len(data["_comments"]):
                data["_comments_truncated"] = {
                    "returned": len(data["_comments"]),
                    "total": total,
                }
        else:
            data["_comments"] = []
            data["_comments_total"] = 0
            data["_comments_included"] = False
        if item.extra.get("attest_materialized_revision") is True:
            await self._attest_materialized_revision(key, str(item.version or ""))
        validate_jira_observation_identities(data)
        artifacts = await self._fetch_source_artifacts(data)

        return RawContent(
            item=item,
            body=json.dumps(data).encode("utf-8"),
            content_type="application/json",
            artifacts=artifacts,
        )

    async def _fetch_source_artifacts(self, issue: dict) -> tuple[RawSourceArtifact, ...]:
        """Return bounded issue attachment descriptors."""

        fields = issue.get("fields") if isinstance(issue.get("fields"), dict) else {}
        raw_descriptors = fields.get("attachment")
        provider_descriptors = raw_descriptors if isinstance(raw_descriptors, list) else []
        if len(provider_descriptors) > MAX_SOURCE_ARTIFACT_DESCRIPTORS_PER_UNIT:
            raise SourceArtifactContractError(
                "Jira issue exceeds the Source Artifact descriptor scan limit"
            )
        descriptors: list[tuple[dict, str]] = []
        for descriptor in provider_descriptors:
            if not isinstance(descriptor, dict):
                raise SourceArtifactContractError(
                    "Jira attachment response contains an invalid record"
                )
            media_type = normalize_source_artifact_media_type(descriptor.get("mimeType"))
            if media_type in SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES:
                descriptors.append((descriptor, media_type))
        if len(descriptors) > MAX_SOURCE_ARTIFACTS_PER_UNIT:
            raise SourceArtifactContractError(
                f"Jira issue exceeds {MAX_SOURCE_ARTIFACTS_PER_UNIT} supported Artifact limit"
            )
        comments = issue.get("_comments") if isinstance(issue.get("_comments"), list) else []
        issue_id = str(issue.get("id") or "").strip()
        artifacts: list[RawSourceArtifact] = []
        declared_bytes = 0
        for descriptor, media_type in descriptors:
            size_value = descriptor.get("size")
            if not isinstance(size_value, int) or isinstance(size_value, bool) or size_value < 0:
                raise SourceArtifactContractError("Jira attachment is missing a valid file size")
            if size_value > MAX_SOURCE_ARTIFACT_STORAGE_BYTES:
                raise SourceArtifactContractError(
                    f"Jira attachment exceeds {MAX_SOURCE_ARTIFACT_STORAGE_BYTES} byte storage limit"
                )
            declared_bytes += size_value
            if declared_bytes > MAX_SOURCE_ARTIFACT_STORAGE_BYTES_PER_UNIT:
                raise SourceArtifactContractError(
                    "Jira attachments exceed the Source Unit storage aggregate limit"
                )
            attachment_id = str(descriptor.get("id") or "").strip()
            filename = str(descriptor.get("filename") or "").strip()
            provider_revision = str(descriptor.get("created") or "immutable").strip()
            content_url = str(descriptor.get("content") or "").strip()
            if not attachment_id or not filename or not content_url:
                raise SourceArtifactContractError("Jira attachment identity is incomplete")
            request_path = self._attachment_request_path(content_url)
            parent_type, parent_key = self._jira_attachment_parent(
                filename=filename,
                issue_id=issue_id,
                comments=comments,
            )
            artifacts.append(
                RawSourceArtifact(
                    provider_key=attachment_id,
                    parent_observation_type=parent_type,
                    parent_provider_key=parent_key,
                    provider_revision=provider_revision,
                    filename=filename,
                    media_type=media_type,
                    declared_size_bytes=size_value,
                    locator={
                        "attachment_id": attachment_id,
                        "request_path": request_path,
                    },
                )
            )
        return tuple(artifacts)

    @asynccontextmanager
    async def open_source_artifact(self, artifact: RawSourceArtifact):
        """Open one Jira attachment without buffering its body."""

        request_path = str(artifact.locator.get("request_path") or "").strip()
        if not request_path:
            raise SourceArtifactContractError("Jira attachment locator is incomplete")
        async with stream_with_rate_limit_retry(
            self._client,
            "GET",
            request_path,
            product_name="Jira",
            limiter=getattr(self, "_request_limiter", None),
            zero_quota_message=_zero_quota_message(
                getattr(self, "_auth_mode", _auth_mode(self.config))
            ),
        ) as response:
            yield SourceArtifactDownload(
                chunks=response.aiter_bytes(),
                media_type=response.headers.get("content-type"),
                content_length=parse_source_artifact_content_length(
                    response.headers.get("content-length")
                ),
                content_encoding=response.headers.get("content-encoding"),
            )

    def _attachment_request_path(self, content_url: str) -> str:
        parsed = urlsplit(content_url)
        if parsed.query or parsed.fragment:
            raise SourceArtifactContractError("Jira attachment URL cannot contain query or fragment")
        if parsed.scheme or parsed.netloc:
            base = urlsplit(self._base_url)
            if parsed.scheme != base.scheme or parsed.netloc != base.netloc:
                raise SourceArtifactContractError("Jira attachment URL is outside the configured origin")
            return parsed.path
        if not content_url.startswith("/"):
            raise SourceArtifactContractError("Jira attachment URL must be absolute or root-relative")
        return content_url

    @staticmethod
    def _jira_attachment_parent(
        *,
        filename: str,
        issue_id: str,
        comments: list[object],
    ) -> tuple[str, str]:
        matches = [
            str(comment.get("id"))
            for comment in comments
            if isinstance(comment, dict)
            and filename in str(comment.get("body") or "")
            and str(comment.get("id") or "").strip()
        ]
        if len(matches) == 1:
            return "comment", matches[0]
        return "issue_core", f"{issue_id}:core"

    async def _attest_materialized_revision(self, key: str, expected_revision: str) -> None:
        response = await self._request(
            "GET",
            f"/rest/api/2/issue/{key}",
            params={"fields": "updated"},
        )
        payload = response.json()
        fields = payload.get("fields") if isinstance(payload, dict) else None
        current_revision = (
            str(fields.get("updated") or "").strip()
            if isinstance(fields, dict)
            else ""
        )
        if (
            not isinstance(payload, dict)
            or str(payload.get("key") or "").strip() != key
            or not current_revision
            or current_revision != expected_revision
        ):
            raise RuntimeError(
                f"Jira issue {key} changed during materialization; retry inventory"
            )

    async def _top_up_truncated_comments(self, key: str, payload: dict) -> None:
        comments_resp = await self._request(
            "GET",
            f"/rest/api/2/issue/{key}/comment",
            params={"maxResults": COMMENT_MAX_RESULTS},
        )
        comment_data = comments_resp.json()
        comments = self._validated_comment_page(comment_data)
        payload["_comments"] = comments
        total = comment_data["total"]
        payload["_comments_total"] = total
        payload["_comments_included"] = True
        if total > len(comments):
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
        validate_jira_observation_identities(payload)

    @staticmethod
    def _validated_comment_page(data: object) -> list[dict]:
        if not isinstance(data, dict) or not isinstance(data.get("comments"), list):
            raise RuntimeError("Jira comments response is missing a comments list")
        comments = data["comments"]
        if any(not isinstance(comment, dict) for comment in comments):
            raise RuntimeError("Jira comments response contains an invalid record")
        total = data.get("total")
        if not isinstance(total, int) or isinstance(total, bool) or total < len(comments):
            raise RuntimeError("Jira comments response total is invalid")
        if data.get("startAt") != 0:
            raise RuntimeError("Jira comments response does not start at zero")
        return comments

    async def normalize(self, raw: RawContent) -> NormalizedContent:
        """Convert Jira issue JSON to comprehensive markdown.

        Includes available summary, status, assignee, description, status history,
        issue links, subtasks, and comments so the enricher can extract memories
        from the issue payload.
        """
        package = json.loads(raw.body)
        data = package
        package_semantics: dict[str, Any] = {}
        if package.get("package_kind") == LOCAL_AGENT_JIRA_PACKAGE_KIND:
            raw_payload = package.get("raw_payload")
            if not isinstance(raw_payload, dict):
                raise ValueError("Jira local-agent package is missing raw_payload")
            data = raw_payload
            package_semantics = {
                "source_kind": "jira",
                "base_url": package.get("base_url"),
                "issue_key": package.get("issue_key"),
                "raw_hash": package.get("raw_hash"),
                "submitted_at": package.get("submitted_at"),
                "submitted_by": package.get("submitted_by"),
            }
        fields = data.get("fields", {})
        key = str(package_semantics.get("issue_key") or raw.item.extra.get("issue_key") or data.get("key") or "")

        lines = []

        # Header
        issue_type = raw.item.extra.get("issue_type") or (fields.get("issuetype") or {}).get("name") or "Task"
        status = raw.item.extra.get("status") or (fields.get("status") or {}).get("name") or "Unknown"
        priority = raw.item.extra.get("priority") or (fields.get("priority") or {}).get("name") or ""
        assignee_field = fields.get("assignee") or {}
        assignee = raw.item.author or assignee_field.get("displayName") or "Unassigned"

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
            **package_semantics,
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

    def _local_agent_documents_dir(self) -> Path | None:
        configured = str(self.config.get("local_agent_documents_dir") or "").strip()
        return Path(configured).expanduser() if configured else None

    def _local_agent_package_manifest(self) -> list[dict]:
        return package_manifest(self.config)

    async def _discover_local_agent_package_manifest(
        self,
        manifest: list[dict],
        since: datetime | None,
    ) -> AsyncIterator[ContentItem]:
        for entry in sorted(
            manifest,
            key=lambda item: (str(item.get("last_modified") or ""), str(item.get("doc_id") or "")),
        ):
            package_uri = str(entry.get("package_uri") or "").strip()
            if not package_uri:
                continue
            last_modified = _parse_local_package_dt(str(entry.get("last_modified") or ""))
            if since and last_modified <= since:
                continue
            issue_key = str(entry.get("issue_key") or "").strip()
            yield ContentItem(
                item_id=str(entry.get("doc_id") or issue_key),
                title=str(entry.get("title") or issue_key),
                source_url=str(entry.get("source_url") or ""),
                last_modified=last_modified,
                content_type="application/json",
                space_or_project=str(entry.get("space_or_project") or issue_key.split("-", 1)[0]),
                version=str(entry.get("version") or ""),
                author=entry.get("submitted_by"),
                labels=["jira"],
                extra={
                    "package_uri": package_uri,
                    "package_path": entry.get("package_path"),
                    "package_sha256": entry.get("package_sha256"),
                    "input_sha256": entry.get("input_sha256"),
                    "issue_key": issue_key,
                },
            )

    async def _discover_local_agent_packages(
        self,
        documents_dir: Path,
        since: datetime | None,
    ) -> AsyncIterator[ContentItem]:
        for package_path in sorted(documents_dir.rglob("*.json")):
            try:
                package = json.loads(package_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                logger.warning("Skipping unreadable Jira local-agent package: %s", package_path)
                continue
            if package.get("package_kind") != LOCAL_AGENT_JIRA_PACKAGE_KIND:
                continue
            last_modified = _parse_local_package_dt(str(package.get("last_modified") or ""))
            if since and last_modified <= since:
                continue
            issue_key = str(package.get("issue_key") or "").strip()
            yield ContentItem(
                item_id=str(package.get("doc_id") or issue_key),
                title=str(package.get("title") or issue_key),
                source_url=str(package.get("source_url") or ""),
                last_modified=last_modified,
                content_type="text/markdown",
                space_or_project=str(package.get("space_or_project") or issue_key.split("-", 1)[0]),
                version=str(package.get("version") or ""),
                author=package.get("author"),
                labels=["jira"],
                extra={"package_path": str(package_path), "issue_key": issue_key},
            )
