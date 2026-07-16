from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from memforge.genes.jira_gene import JiraGene
from memforge.models import ContentItem


def _search_page(issues: list[dict], *, total: int | None = None, start_at: int = 0) -> dict:
    return {
        "startAt": start_at,
        "total": len(issues) if total is None else total,
        "issues": issues,
    }


def _jira_issue(
    key: str,
    *,
    issue_id: str | None = None,
    field_overrides: dict | None = None,
    comments: list[dict] | None = None,
    comment_total: int | None = None,
    histories: list[dict] | None = None,
    changelog_total: int | None = None,
) -> dict:
    normalized_comments = [
        {"id": str(index + 1), **comment}
        for index, comment in enumerate(comments or [])
    ]
    normalized_histories = [
        {"id": str(index + 1), **history}
        for index, history in enumerate(histories or [])
    ]
    fields = {
        "summary": key,
        "description": None,
        "status": None,
        "priority": None,
        "assignee": None,
        "labels": [],
        "resolution": None,
        "updated": "2026-05-21T08:00:00.000+0000",
        "project": {"key": "PAY"},
        "issuetype": {"name": "Task"},
        "issuelinks": [],
        "subtasks": [],
        "comment": {
            "startAt": 0,
            "comments": normalized_comments,
            "total": len(normalized_comments) if comment_total is None else comment_total,
        },
        **(field_overrides or {}),
    }
    return {
        "id": issue_id or str(100000 + int(key.rsplit("-", 1)[-1])),
        "key": key,
        "fields": fields,
        "changelog": {
            "startAt": 0,
            "histories": normalized_histories,
            "total": len(normalized_histories) if changelog_total is None else changelog_total,
        },
    }


def test_jira_schema_hides_runtime_transport_fields_from_ui():
    fields = {field.key: field for field in JiraGene.config_schema().fields}

    assert fields["auth_mode"].required is True
    assert fields["auth_mode"].default == "browser_cookie"
    assert "jira_cookie" not in fields
    assert fields["pat"].required is False
    assert fields["pat"].advanced is False
    assert "tls_ca_bundle" not in fields
    assert "request_interval_ms" not in fields


class JsonResponse:
    def __init__(self, payload: dict, status_code: int = 200, headers: dict | None = None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://jira.example.test/rest/api/2/search")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("request failed", request=request, response=response)
        return None


class RecordingAsyncClient:
    instances: list["RecordingAsyncClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[tuple[str, str, dict]] = []
        self.closed = False
        RecordingAsyncClient.instances.append(self)

    async def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/comment"):
            return JsonResponse({"startAt": 0, "comments": [], "total": 0})
        if "/issue/" in url:
            return JsonResponse(_jira_issue(url.split("/issue/", 1)[1].split("/", 1)[0]))
        return JsonResponse(_search_page([]))

    async def get(self, url: str, **kwargs):
        return await self.request("GET", url, **kwargs)

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"issues": None, "total": 0},
        {"issues": [], "total": 1},
        {"issues": [], "total": "0"},
    ],
)
async def test_discovery_rejects_malformed_or_early_terminal_search_page(payload):
    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "pat": "token"},
        source_id="src-jira",
    )
    gene._base_url = "https://jira.example.test"
    gene._request = AsyncMock(return_value=JsonResponse(payload))

    with pytest.raises(RuntimeError, match="Jira search"):
        _ = [item async for item in gene.discover()]
    assert gene.discovery_complete is False


@pytest.mark.asyncio
async def test_discovery_rejects_duplicate_issue_identity_before_completion():
    issue = _jira_issue("PAY-1")
    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "pat": "token"},
        source_id="src-jira",
    )
    gene._base_url = "https://jira.example.test"
    gene._request = AsyncMock(return_value=JsonResponse(_search_page([issue, issue], total=2)))

    with pytest.raises(RuntimeError, match="duplicate issue identity"):
        _ = [item async for item in gene.discover()]
    assert gene.discovery_complete is False


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_collection", ["comments", "changelog"])
async def test_fetch_rejects_missing_requested_collection_evidence(missing_collection):
    issue = _jira_issue("PAY-1")
    if missing_collection == "comments":
        issue["fields"].pop("comment")
    else:
        issue.pop("changelog")

    class MissingCollectionClient(RecordingAsyncClient):
        async def request(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            return JsonResponse(_search_page([issue]))

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "include_comments": True},
        source_id="src-jira",
    )
    gene._client = MissingCollectionClient(base_url="https://jira.example.test")
    gene._base_url = "https://jira.example.test"
    item = [item async for item in gene.discover()][0]

    with pytest.raises((RuntimeError, ValueError)):
        await gene.fetch(item)


@pytest.mark.asyncio
async def test_fetch_rejects_duplicate_comment_provider_ids():
    issue = _jira_issue(
        "PAY-1",
        comments=[{"id": "same", "body": "one"}, {"id": "same", "body": "two"}],
    )

    class DuplicateCommentClient(RecordingAsyncClient):
        async def request(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            return JsonResponse(_search_page([issue]))

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"]},
        source_id="src-jira",
    )
    gene._client = DuplicateCommentClient(base_url="https://jira.example.test")
    gene._base_url = "https://jira.example.test"
    item = [item async for item in gene.discover()][0]

    with pytest.raises(ValueError, match="duplicate provider id"):
        await gene.fetch(item)


class RateLimitedThenSuccessClient(RecordingAsyncClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.responses = [
            JsonResponse({}, status_code=429),
            JsonResponse({}, status_code=429),
            JsonResponse(_search_page([])),
        ]

    async def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


class AlwaysRateLimitedClient(RecordingAsyncClient):
    async def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        return JsonResponse({}, status_code=429)


class ZeroQuotaClient(RecordingAsyncClient):
    async def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        return JsonResponse(
            {},
            status_code=429,
            headers={
                "X-RateLimit-Limit": "0",
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-FillRate": "0",
                "Retry-After": "9223372036854775807",
            },
        )


class IssueThenRateLimitedCommentClient(RecordingAsyncClient):
    async def request(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/comment"):
            return JsonResponse({}, status_code=429)
        return JsonResponse(_jira_issue("PAY-123"))


class AsyncNoop:
    async def __call__(self, delay: float) -> None:
        return None


@pytest.mark.asyncio
async def test_authenticate_prepares_bearer_pat_client_without_remote_probe(monkeypatch):
    RecordingAsyncClient.instances.clear()
    monkeypatch.setattr("memforge.genes.jira_gene.httpx.AsyncClient", RecordingAsyncClient)
    gene = JiraGene(
        config={
            "base_url": "https://jira.example.test/",
            "projects": ["PAY"],
            "auth_mode": "pat",
            "pat": "jira-pat",
        },
        source_id="src-jira",
    )

    await gene.authenticate()

    assert gene._base_url == "https://jira.example.test"
    assert RecordingAsyncClient.instances[-1].kwargs["base_url"] == "https://jira.example.test"
    assert RecordingAsyncClient.instances[-1].kwargs["headers"]["Authorization"] == "Bearer jira-pat"
    assert "Cookie" not in RecordingAsyncClient.instances[-1].kwargs["headers"]
    assert RecordingAsyncClient.instances[-1].kwargs["verify"] is True
    assert RecordingAsyncClient.instances[-1].calls == []


@pytest.mark.asyncio
async def test_authenticate_prepares_cookie_client_without_bearer_header(monkeypatch):
    RecordingAsyncClient.instances.clear()
    monkeypatch.setattr("memforge.genes.jira_gene.httpx.AsyncClient", RecordingAsyncClient)
    gene = JiraGene(
        config={
            "base_url": "https://jira.example.test/",
            "projects": ["PAY"],
            "auth_mode": "browser_cookie",
            "jira_cookie": "JSESSIONID=session; atlassian.xsrf.token=token",
        },
        source_id="src-jira",
    )

    await gene.authenticate()

    headers = RecordingAsyncClient.instances[-1].kwargs["headers"]
    assert headers["Cookie"] == "JSESSIONID=session; atlassian.xsrf.token=token"
    assert headers["Accept"] == "application/json"
    assert "Authorization" not in headers
    assert RecordingAsyncClient.instances[-1].calls == []


@pytest.mark.asyncio
async def test_authenticate_uses_configured_jira_ca_bundle(monkeypatch, tmp_path):
    RecordingAsyncClient.instances.clear()
    monkeypatch.setattr("memforge.genes.jira_gene.httpx.AsyncClient", RecordingAsyncClient)
    ca_bundle = tmp_path / "corp-ca.pem"
    ca_bundle.write_text("test-ca", encoding="utf-8")
    gene = JiraGene(
        config={
            "base_url": "https://jira.example.test/",
            "projects": ["PAY"],
            "auth_mode": "pat",
            "pat": "jira-pat",
            "tls_ca_bundle": str(ca_bundle),
        },
        source_id="src-jira",
    )

    await gene.authenticate()

    assert RecordingAsyncClient.instances[-1].kwargs["verify"] == str(ca_bundle)


@pytest.mark.asyncio
async def test_authenticate_rejects_insecure_jira_url():
    gene = JiraGene(
        config={"base_url": "http://jira.example.test", "projects": ["PAY"], "auth_mode": "pat", "pat": "jira-pat"},
        source_id="src-jira",
    )

    with pytest.raises(ValueError, match="HTTPS"):
        await gene.authenticate()


@pytest.mark.asyncio
async def test_authenticate_requires_jira_pat():
    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "auth_mode": "pat"},
        source_id="src-jira",
    )

    with pytest.raises(ValueError, match="PAT"):
        await gene.authenticate()


@pytest.mark.asyncio
async def test_authenticate_requires_jira_cookie_for_cookie_mode():
    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "auth_mode": "browser_cookie"},
        source_id="src-jira",
    )

    with pytest.raises(ValueError, match="Jira browser session cookie is required"):
        await gene.authenticate()


@pytest.mark.asyncio
async def test_cookie_mode_reports_expired_session_without_retry(monkeypatch):
    monkeypatch.setattr("memforge.genes.atlassian_auth.asyncio.sleep", AsyncNoop())

    class ExpiredCookieClient(RecordingAsyncClient):
        async def request(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            return JsonResponse({"message": "Client must be authenticated"}, status_code=401)

    gene = JiraGene(
        config={
            "base_url": "https://jira.example.test",
            "projects": ["PAY"],
            "auth_mode": "browser_cookie",
            "jira_cookie": "JSESSIONID=expired",
        },
        source_id="src-jira",
    )
    client = ExpiredCookieClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"

    with pytest.raises(RuntimeError, match="Jira browser session cookie expired"):
        [item async for item in gene.discover()]

    assert [call[0:2] for call in client.calls] == [("POST", "/rest/api/2/search")]


@pytest.mark.asyncio
async def test_pat_zero_quota_rate_limit_fails_fast(monkeypatch):
    monkeypatch.setattr("memforge.genes.atlassian_auth.asyncio.sleep", AsyncNoop())
    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "auth_mode": "pat", "pat": "jira-pat"},
        source_id="src-jira",
    )
    client = ZeroQuotaClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"

    with pytest.raises(RuntimeError, match="Jira PAT API quota is zero"):
        [item async for item in gene.discover()]

    assert [call[0:2] for call in client.calls] == [("POST", "/rest/api/2/search")]


@pytest.mark.asyncio
async def test_cookie_zero_quota_rate_limit_reports_cookie_context(monkeypatch):
    monkeypatch.setattr("memforge.genes.atlassian_auth.asyncio.sleep", AsyncNoop())
    gene = JiraGene(
        config={
            "base_url": "https://jira.example.test",
            "projects": ["PAY"],
            "auth_mode": "browser_cookie",
            "jira_cookie": "JSESSIONID=session",
        },
        source_id="src-jira",
    )
    client = ZeroQuotaClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"

    with pytest.raises(RuntimeError) as exc_info:
        [item async for item in gene.discover()]

    message = str(exc_info.value)
    assert "browser-cookie session" in message
    assert "PAT API quota" not in message
    assert "Use browser cookie authentication" not in message
    assert [call[0:2] for call in client.calls] == [("POST", "/rest/api/2/search")]


@pytest.mark.asyncio
async def test_discover_uses_configured_project_filter_and_issue_types():
    gene = JiraGene(
        config={
            "base_url": "https://jira.example.test",
            "projects": ["PAY"],
            "jql_filter": '"Agile Team" = "ExampleTeam-DeliveryBoard"',
            "issue_types": ["Story", "Bug"],
        },
        source_id="src-jira",
    )
    client = RecordingAsyncClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"

    items = [
        item
        async for item in gene.discover(
            since=datetime(2026, 5, 21, 8, 0, tzinfo=timezone.utc),
        )
    ]

    assert items == []
    assert client.calls[0][1] == "/rest/api/2/search"
    jql = client.calls[0][2]["json"]["jql"]
    assert "project in (PAY)" in jql
    assert "issuetype in (Story,Bug)" in jql
    assert '("Agile Team" = "ExampleTeam-DeliveryBoard")' in jql
    assert "updated >= '2026-05-21 08:00'" in jql


@pytest.mark.asyncio
async def test_discover_hydrates_search_result_so_fetch_uses_no_per_issue_requests():
    class SearchIssueClient(RecordingAsyncClient):
        async def request(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            return JsonResponse(
                _search_page(
                    [
                        _jira_issue(
                            "PAY-123",
                            issue_id="100123",
                            field_overrides={
                                "summary": "Hydrated issue",
                                "status": {"name": "In Progress"},
                                "priority": {"name": "High"},
                                "assignee": {"displayName": "Ada"},
                                "issuetype": {"name": "Story"},
                                "labels": ["architecture"],
                                "description": "Design context",
                            },
                            comments=[
                                {
                                    "author": {"displayName": "Grace"},
                                    "created": "2026-05-21T09:00:00.000+0000",
                                    "body": "Keep the low-request path.",
                                }
                            ],
                            changelog_total=3,
                        )
                    ]
                )
            )

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "include_comments": True},
        source_id="src-jira",
    )
    client = SearchIssueClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"

    items = [item async for item in gene.discover()]
    raw = await gene.fetch(items[0])
    normalized = await gene.normalize(raw)

    assert [call[0:2] for call in client.calls] == [("POST", "/rest/api/2/search")]
    search_body = client.calls[0][2]["json"]
    assert search_body["fields"] == ["*all"]
    assert search_body["expand"] == ["changelog", "renderedFields"]
    assert isinstance(search_body["fields"], list)
    assert "_search_issue" not in items[0].extra
    raw_payload = json.loads(raw.body)
    assert raw_payload["_comments"][0]["body"] == "Keep the low-request path."
    assert raw_payload["_changelog_truncated"] == {"returned": 0, "total": 3}
    assert "Design context" in normalized.markdown_body
    assert "Keep the low-request path." in normalized.markdown_body


@pytest.mark.asyncio
async def test_discover_paces_paginated_jira_search_requests(monkeypatch):
    import memforge.genes.atlassian_auth as atlassian_auth

    now = 1000.0
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return now

    async def fake_sleep(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    class TwoPageClient(RecordingAsyncClient):
        async def request(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            start_at = kwargs["json"]["startAt"]
            page_size = 1 if start_at == 100 else 50
            return JsonResponse(
                {
                    "startAt": start_at,
                    "total": 101,
                    "issues": [
                        _jira_issue(f"PAY-{index + 1}")
                        for index in range(start_at, start_at + page_size)
                    ],
                }
            )

    monkeypatch.setattr(atlassian_auth.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(atlassian_auth.asyncio, "sleep", fake_sleep)
    gene = JiraGene(
        config={
            "base_url": "https://jira.example.test",
            "projects": ["PAY"],
            "request_interval_ms": 750,
        },
        source_id="src-jira",
    )
    client = TwoPageClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"
    gene._request_limiter = atlassian_auth.AtlassianRequestLimiter(min_interval_seconds=0.75)

    items = [item async for item in gene.discover()]

    assert len(items) == 101
    assert items[0].item_id == "jira-PAY-1"
    assert items[-1].item_id == "jira-PAY-101"
    assert [call[0:2] for call in client.calls] == [
        ("POST", "/rest/api/2/search"),
        ("POST", "/rest/api/2/search"),
        ("POST", "/rest/api/2/search"),
    ]
    assert sleeps == [0.75, 0.75]


@pytest.mark.asyncio
async def test_fetch_tops_up_truncated_hydrated_comments_through_limiter():
    class SearchIssueClient(RecordingAsyncClient):
        async def request(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            if url.endswith("/comment"):
                return JsonResponse(
                    {
                        "startAt": 0,
                        "comments": [
                            {"id": "1", "author": {"displayName": "Grace"}, "created": "2026-05-21", "body": "First"},
                            {"id": "2", "author": {"displayName": "Ada"}, "created": "2026-05-22", "body": "Second"},
                        ],
                        "total": 2,
                    }
                )
            return JsonResponse(
                _search_page(
                    [
                        _jira_issue(
                            "PAY-123",
                            issue_id="100123",
                            field_overrides={"summary": "Hydrated issue"},
                            comments=[
                                {"author": {"displayName": "Grace"}, "created": "2026-05-21", "body": "First"}
                            ],
                            comment_total=2,
                        )
                    ]
                )
            )

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "include_comments": True},
        source_id="src-jira",
    )
    client = SearchIssueClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"
    gene._hydrated_issues = {}

    items = [item async for item in gene.discover()]
    raw = await gene.fetch(items[0])

    assert [call[0:2] for call in client.calls] == [
        ("POST", "/rest/api/2/search"),
        ("GET", "/rest/api/2/issue/PAY-123/comment"),
    ]
    assert [comment["body"] for comment in json.loads(raw.body)["_comments"]] == ["First", "Second"]


@pytest.mark.asyncio
async def test_fetch_rejects_malformed_comment_top_up_instead_of_clearing_truncation():
    issue = _jira_issue(
        "PAY-123",
        issue_id="100123",
        comments=[{"id": "1", "body": "First"}],
        comment_total=2,
    )

    class MalformedTopUpClient(RecordingAsyncClient):
        async def request(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            if url.endswith("/comment"):
                return JsonResponse({})
            return JsonResponse(_search_page([issue]))

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "include_comments": True},
        source_id="src-jira",
    )
    gene._client = MalformedTopUpClient(base_url="https://jira.example.test")
    gene._base_url = "https://jira.example.test"
    item = [item async for item in gene.discover()][0]

    with pytest.raises(RuntimeError, match="comments list"):
        await gene.fetch(item)


@pytest.mark.asyncio
async def test_atlassian_limiter_paces_after_request_exception(monkeypatch):
    import memforge.genes.atlassian_auth as atlassian_auth

    now = 1000.0
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return now

    async def fake_sleep(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    monkeypatch.setattr(atlassian_auth.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(atlassian_auth.asyncio, "sleep", fake_sleep)
    limiter = atlassian_auth.AtlassianRequestLimiter(min_interval_seconds=0.5)

    async def fail() -> JsonResponse:
        raise httpx.ReadTimeout("timeout")

    async def succeed() -> JsonResponse:
        return JsonResponse({})

    with pytest.raises(httpx.ReadTimeout):
        await limiter.run(fail)
    await limiter.run(succeed)

    assert sleeps == [0.5]


def test_atlassian_limiter_is_shared_by_base_host():
    from memforge.genes.atlassian_auth import atlassian_request_limiter, release_atlassian_request_limiter

    first = atlassian_request_limiter("https://jira.example.test", min_interval_seconds=0.75, owner_id="src-one")
    second = atlassian_request_limiter(
        "https://jira.example.test/plugins/servlet",
        min_interval_seconds=0.25,
        owner_id="src-two",
    )

    assert first is second
    release_atlassian_request_limiter("https://jira.example.test", owner_id="src-one")
    release_atlassian_request_limiter("https://jira.example.test", owner_id="src-two")


@pytest.mark.asyncio
async def test_releasing_limiter_owner_removes_strict_interval(monkeypatch):
    import memforge.genes.atlassian_auth as atlassian_auth

    now = 2500.0
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return now

    async def fake_sleep(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    monkeypatch.setattr(atlassian_auth.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(atlassian_auth.asyncio, "sleep", fake_sleep)
    limiter = atlassian_auth.atlassian_request_limiter(
        "https://release-jira.example.test",
        min_interval_seconds=5.0,
        owner_id="slow-source",
    )
    atlassian_auth.atlassian_request_limiter(
        "https://release-jira.example.test",
        min_interval_seconds=0.25,
        owner_id="fast-source",
    )
    atlassian_auth.release_atlassian_request_limiter(
        "https://release-jira.example.test",
        owner_id="slow-source",
    )

    async def succeed() -> JsonResponse:
        return JsonResponse({})

    await limiter.run(succeed)
    await limiter.run(succeed)

    assert sleeps == [0.25]
    atlassian_auth.release_atlassian_request_limiter(
        "https://release-jira.example.test",
        owner_id="fast-source",
    )


def test_releasing_last_limiter_owner_drops_cached_origin():
    import memforge.genes.atlassian_auth as atlassian_auth

    first = atlassian_auth.atlassian_request_limiter(
        "https://drop-jira.example.test",
        min_interval_seconds=3.0,
        owner_id="src-one",
    )
    atlassian_auth.release_atlassian_request_limiter(
        "https://drop-jira.example.test",
        owner_id="src-one",
    )
    second = atlassian_auth.atlassian_request_limiter(
        "https://drop-jira.example.test",
        min_interval_seconds=0.25,
        owner_id="src-two",
    )

    assert second is not first
    atlassian_auth.release_atlassian_request_limiter(
        "https://drop-jira.example.test",
        owner_id="src-two",
    )


@pytest.mark.asyncio
async def test_atlassian_limiter_uses_strictest_interval_for_same_host(monkeypatch):
    import memforge.genes.atlassian_auth as atlassian_auth

    now = 2000.0
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return now

    async def fake_sleep(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    monkeypatch.setattr(atlassian_auth.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(atlassian_auth.asyncio, "sleep", fake_sleep)
    limiter = atlassian_auth.atlassian_request_limiter(
        "https://strict-jira.example.test",
        min_interval_seconds=5.0,
        owner_id="slow-source",
    )
    same_limiter = atlassian_auth.atlassian_request_limiter(
        "https://strict-jira.example.test/browse/PAY-1",
        min_interval_seconds=0.1,
        owner_id="fast-source",
    )

    async def succeed() -> JsonResponse:
        return JsonResponse({})

    await limiter.run(succeed)
    await same_limiter.run(succeed)

    assert limiter is same_limiter
    assert sleeps == [5.0]


@pytest.mark.asyncio
async def test_request_limiter_honors_retry_after_on_degraded_response(monkeypatch):
    import memforge.genes.atlassian_auth as atlassian_auth

    now = 3000.0
    sleeps: list[float] = []

    def fake_monotonic() -> float:
        return now

    async def fake_sleep(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    class DegradedClient(RecordingAsyncClient):
        async def request(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            return JsonResponse({}, status_code=503, headers={"Retry-After": "3"})

    monkeypatch.setattr(atlassian_auth.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(atlassian_auth.asyncio, "sleep", fake_sleep)
    limiter = atlassian_auth.AtlassianRequestLimiter(min_interval_seconds=0.5)

    with pytest.raises(httpx.HTTPStatusError):
        await atlassian_auth.get_with_rate_limit_retry(
            DegradedClient(),
            "/rest/api/2/search",
            product_name="Jira",
            limiter=limiter,
        )

    await limiter.run(lambda: _async_json_response({}))

    assert sleeps == [3.0]


@pytest.mark.asyncio
async def test_discover_resets_hydrated_issue_cache_between_runs():
    class OneIssueClient(RecordingAsyncClient):
        async def request(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            issue_key = "PAY-1" if len(self.calls) == 1 else "PAY-2"
            return JsonResponse(
                _search_page([_jira_issue(issue_key)])
            )

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"]},
        source_id="src-jira",
    )
    gene._client = OneIssueClient(base_url="https://jira.example.test")
    gene._base_url = "https://jira.example.test"
    gene._hydrated_issues = {}

    first_run = [item async for item in gene.discover()]
    second_run = [item async for item in gene.discover()]

    assert [item.item_id for item in first_run] == ["jira-PAY-1"]
    assert [item.item_id for item in second_run] == ["jira-PAY-2"]
    assert set(gene._hydrated_issues) == {"PAY-2"}


async def _async_json_response(payload: dict) -> JsonResponse:
    return JsonResponse(payload)


@pytest.mark.asyncio
async def test_discover_retries_jira_rate_limit(monkeypatch):
    monkeypatch.setattr("memforge.genes.atlassian_auth.asyncio.sleep", AsyncNoop())
    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"]},
        source_id="src-jira",
    )
    client = RateLimitedThenSuccessClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"

    items = [item async for item in gene.discover()]

    assert items == []
    assert [call[1] for call in client.calls] == [
        "/rest/api/2/search",
        "/rest/api/2/search",
        "/rest/api/2/search",
    ]


@pytest.mark.asyncio
async def test_discover_reports_jira_rate_limit_after_retries(monkeypatch):
    monkeypatch.setattr("memforge.genes.atlassian_auth.asyncio.sleep", AsyncNoop())
    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"]},
        source_id="src-jira",
    )
    client = AlwaysRateLimitedClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"

    with pytest.raises(RuntimeError, match="Jira search failed: Jira rate limit persisted"):
        [item async for item in gene.discover()]

    assert [call[1] for call in client.calls] == [
        "/rest/api/2/search",
        "/rest/api/2/search",
        "/rest/api/2/search",
        "/rest/api/2/search",
    ]


@pytest.mark.asyncio
async def test_discover_raises_when_jira_search_is_unauthorized():
    class UnauthorizedClient(RecordingAsyncClient):
        async def request(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            return JsonResponse({}, status_code=401)

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"]},
        source_id="src-jira",
    )
    gene._client = UnauthorizedClient(base_url="https://jira.example.test")
    gene._base_url = "https://jira.example.test"

    with pytest.raises(RuntimeError, match="Jira search failed"):
        [item async for item in gene.discover()]


@pytest.mark.asyncio
async def test_fetch_respects_include_comments_config():
    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "include_comments": False},
        source_id="src-jira",
    )
    client = RecordingAsyncClient(base_url="https://jira.example.test")
    gene._client = client
    item = ContentItem(
        item_id="jira-PAY-123",
        title="PAY-123: Example issue",
        source_url="https://jira.example.test/browse/PAY-123",
        last_modified=datetime(2026, 5, 21, tzinfo=timezone.utc),
        content_type="application/json",
        extra={"issue_key": "PAY-123"},
    )

    await gene.fetch(item)

    assert [call[1] for call in client.calls] == ["/rest/api/2/issue/PAY-123"]


@pytest.mark.asyncio
async def test_fetch_reports_comment_rate_limit_when_comments_are_enabled(monkeypatch):
    monkeypatch.setattr("memforge.genes.atlassian_auth.asyncio.sleep", AsyncNoop())
    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "include_comments": True},
        source_id="src-jira",
    )
    client = IssueThenRateLimitedCommentClient(base_url="https://jira.example.test")
    gene._client = client
    item = ContentItem(
        item_id="jira-PAY-123",
        title="PAY-123: Example issue",
        source_url="https://jira.example.test/browse/PAY-123",
        last_modified=datetime(2026, 5, 21, tzinfo=timezone.utc),
        content_type="application/json",
        extra={"issue_key": "PAY-123"},
    )

    with pytest.raises(RuntimeError, match="Jira rate limit persisted"):
        await gene.fetch(item)

    assert [call[1] for call in client.calls] == [
        "/rest/api/2/issue/PAY-123",
        "/rest/api/2/issue/PAY-123/comment",
        "/rest/api/2/issue/PAY-123/comment",
        "/rest/api/2/issue/PAY-123/comment",
        "/rest/api/2/issue/PAY-123/comment",
    ]


@pytest.mark.asyncio
async def test_discover_retries_once_on_transient_timeout(monkeypatch):
    monkeypatch.setattr("memforge.genes.atlassian_auth.asyncio.sleep", AsyncNoop())

    class TimeoutThenSuccessClient(RecordingAsyncClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._timed_out = False

        async def request(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            if not self._timed_out:
                self._timed_out = True
                raise httpx.ReadTimeout("slow", request=httpx.Request(method, f"https://jira.example.test{url}"))
            return JsonResponse(_search_page([]))

    gene = JiraGene(
        config={
            "base_url": "https://jira.example.test",
            "projects": ["PAY"],
            "auth_mode": "browser_cookie",
            "jira_cookie": "JSESSIONID=ok",
        },
        source_id="src-jira",
    )
    client = TimeoutThenSuccessClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"

    items = [item async for item in gene.discover()]

    assert items == []  # search returned no issues once the retry succeeded
    assert [call[0:2] for call in client.calls] == [
        ("POST", "/rest/api/2/search"),
        ("POST", "/rest/api/2/search"),
    ]


@pytest.mark.asyncio
async def test_discover_retries_on_transient_connect_error(monkeypatch):
    monkeypatch.setattr("memforge.genes.atlassian_auth.asyncio.sleep", AsyncNoop())

    class ConnectErrorThenSuccessClient(RecordingAsyncClient):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._failed = False

        async def request(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            if not self._failed:
                self._failed = True
                raise httpx.ConnectError("flaky", request=httpx.Request(method, f"https://jira.example.test{url}"))
            return JsonResponse(_search_page([]))

    gene = JiraGene(
        config={
            "base_url": "https://jira.example.test",
            "projects": ["PAY"],
            "auth_mode": "browser_cookie",
            "jira_cookie": "JSESSIONID=ok",
        },
        source_id="src-jira",
    )
    client = ConnectErrorThenSuccessClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"

    items = [item async for item in gene.discover()]

    assert items == []  # search returned no issues once the connect retry succeeded
    assert [call[0:2] for call in client.calls] == [
        ("POST", "/rest/api/2/search"),
        ("POST", "/rest/api/2/search"),
    ]
