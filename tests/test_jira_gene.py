from __future__ import annotations

import json
import ssl
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from memforge.genes import atlassian_auth, jira_gene
from memforge.genes.jira_gene import JiraGene, JiraQueryRejected
from memforge.genes.toolkit.http import ConnectorHttpClient
from memforge.models import AbsenceAnswer, ContentItem
from memforge.source_artifacts import RawSourceArtifact, SourceArtifactContractError
from tests.connector_http_fixture import AnsweringClient, FakeConnectorHttp
from tests.tls_bundle_fixture import BUNDLE_CA_COUNT, trusted_ca_count, write_ca_bundle


@pytest.fixture(autouse=True)
def own_origin_limiters(monkeypatch):
    """Give each test its own Atlassian origin limiters, so no pacing carries over from another test."""
    monkeypatch.setattr("memforge.genes.atlassian_auth._ATLASSIAN_REQUEST_LIMITERS", {})


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


class JsonResponse(httpx.Response):
    def __init__(self, payload: dict, status_code: int = 200, headers: dict | None = None):
        super().__init__(status_code, json=payload, headers=headers)


class BinaryResponse(httpx.Response):
    def __init__(self, content: bytes, content_type: str):
        super().__init__(200, content=content, headers={"content-type": content_type})


def _recording_http() -> tuple[list[str], FakeConnectorHttp]:
    """A fake Jira that answers every request with 200 and records its path."""
    paths: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={})

    return paths, FakeConnectorHttp(handle)


class RecordingAsyncClient(AnsweringClient):
    """A Jira answering searches with no issues and each issue read with that issue; it records every call."""

    def __init__(self, base_url: str = "https://jira.example.test") -> None:
        super().__init__(base_url=base_url)
        self.calls: list[tuple[str, str, dict]] = []

    async def answer(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/comment"):
            return JsonResponse({"startAt": 0, "comments": [], "total": 0})
        if "/issue/" in url:
            return JsonResponse(_jira_issue(url.split("/issue/", 1)[1].split("/", 1)[0]))
        return JsonResponse(_search_page([]))


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
    with pytest.raises(RuntimeError, match="Jira search"):
        await gene.list_scope()


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
    with pytest.raises(RuntimeError, match="duplicate issue identity"):
        await gene.list_scope()


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_collection", ["comments", "changelog"])
async def test_fetch_rejects_missing_requested_collection_evidence(missing_collection):
    issue = _jira_issue("PAY-1")
    if missing_collection == "comments":
        issue["fields"].pop("comment")
    else:
        issue.pop("changelog")

    class MissingCollectionClient(RecordingAsyncClient):
        async def answer(self, method: str, url: str, **kwargs):
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
        async def answer(self, method: str, url: str, **kwargs):
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


@pytest.mark.asyncio
async def test_authenticate_prepares_bearer_pat_client_without_remote_probe():
    paths, http = _recording_http()
    gene = JiraGene(
        config={
            "base_url": "https://jira.example.test/",
            "projects": ["PAY"],
            "auth_mode": "pat",
            "pat": "jira-pat",
        },
        source_id="src-jira",
    )

    with http.serving(jira_gene):
        await gene.authenticate()
    await gene.close()

    assert gene._base_url == "https://jira.example.test"
    [options] = http.options
    assert options["base_url"] == "https://jira.example.test"
    assert options["headers"]["Authorization"] == "Bearer jira-pat"
    assert "Cookie" not in options["headers"]
    assert options["verify"] is True
    assert options["follow_redirects"] is True
    assert paths == []


@pytest.mark.asyncio
async def test_authenticate_prepares_cookie_client_without_bearer_header():
    paths, http = _recording_http()
    gene = JiraGene(
        config={
            "base_url": "https://jira.example.test/",
            "projects": ["PAY"],
            "auth_mode": "browser_cookie",
            "jira_cookie": "JSESSIONID=session; atlassian.xsrf.token=token",
        },
        source_id="src-jira",
    )

    with http.serving(jira_gene):
        await gene.authenticate()
    await gene.close()

    headers = http.options[-1]["headers"]
    assert headers["Cookie"] == "JSESSIONID=session; atlassian.xsrf.token=token"
    assert headers["Accept"] == "application/json"
    assert "Authorization" not in headers
    assert paths == []


@pytest.mark.asyncio
async def test_authenticate_uses_configured_jira_ca_bundle(tmp_path):
    _paths, http = _recording_http()
    ca_bundle = write_ca_bundle(tmp_path / "corp-ca.pem")
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

    with http.serving(jira_gene):
        await gene.authenticate()
    await gene.close()

    verify = http.options[-1]["verify"]
    assert isinstance(verify, ssl.SSLContext)
    assert verify.verify_mode == ssl.CERT_REQUIRED
    assert trusted_ca_count(verify) == BUNDLE_CA_COUNT


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
async def test_cookie_mode_reports_expired_session_without_retry():
    class ExpiredCookieClient(RecordingAsyncClient):
        async def answer(self, method: str, url: str, **kwargs):
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


def _zero_quota(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        429,
        headers={
            "X-RateLimit-Limit": "0",
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-FillRate": "0",
            "Retry-After": "9223372036854775807",
        },
    )


async def _authenticated_gene(config: dict, handler) -> tuple[JiraGene, FakeConnectorHttp, list[str]]:
    """A Gene authenticated through its real client, answered by ``handler``; the paths it requested, in order."""
    paths: list[str] = []

    def recording(request: httpx.Request):
        paths.append(f"{request.method} {request.url.path}")
        return handler(request)

    http = FakeConnectorHttp(recording)
    gene = JiraGene(config={"base_url": "https://jira.example.test", "projects": ["PAY"], **config}, source_id="src-jira")
    with http.serving(jira_gene):
        await gene.authenticate()
    return gene, http, paths


@pytest.mark.asyncio
async def test_pat_zero_quota_rate_limit_fails_fast():
    gene, _http, paths = await _authenticated_gene({"auth_mode": "pat", "pat": "jira-pat"}, _zero_quota)
    try:
        with pytest.raises(RuntimeError, match="Jira PAT API quota is zero"):
            [item async for item in gene.discover()]
    finally:
        await gene.close()

    assert paths == ["POST /rest/api/2/search"]


@pytest.mark.asyncio
async def test_cookie_zero_quota_rate_limit_reports_cookie_context():
    gene, _http, paths = await _authenticated_gene(
        {"auth_mode": "browser_cookie", "jira_cookie": "JSESSIONID=session"}, _zero_quota
    )
    try:
        with pytest.raises(RuntimeError) as exc_info:
            [item async for item in gene.discover()]
    finally:
        await gene.close()

    message = str(exc_info.value)
    assert "browser-cookie session" in message
    assert "PAT API quota" not in message
    assert "Use browser cookie authentication" not in message
    assert paths == ["POST /rest/api/2/search"]


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
        async def answer(self, method: str, url: str, **kwargs):
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
                                    "renderedBody": "<p>Keep the low-request path.</p>",
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


# Jira Data Center embeds an issue's whole changelog; this is more histories than
# the single 100-history page a capped changelog would carry.
LONG_CHANGELOG_HISTORY_COUNT = 150
# A changelog cut short by a Jira administrator's response limit.
LIMITED_CHANGELOG_HISTORY_COUNT = 100


def _histories(count: int) -> list[dict]:
    return [
        {"created": "2026-05-21T08:00:00.000+0000", "items": [{"field": "Sprint"}]}
        for _ in range(count)
    ]


def _project_jira(raw, normalized):
    from memforge.pipeline.source_projection_adapters import project_source_item

    return project_source_item(
        source_id="src-jira",
        source_type="jira",
        run_id="run-jira-changelog",
        item=raw.item,
        raw=raw,
        normalized=normalized,
    )


def _changelog_keys(projection) -> list[str]:
    return [
        observation.provider_key
        for observation in projection.observations
        if observation.observation_type == "changelog"
    ]


class _LongChangelogClient(RecordingAsyncClient):
    """Serves PAY-7 with its whole long changelog embedded, as Jira Data Center does."""

    async def answer(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        issue = _jira_issue("PAY-7", issue_id="100007", histories=_histories(LONG_CHANGELOG_HISTORY_COUNT))
        if url.endswith("/comment"):
            return JsonResponse({"startAt": 0, "comments": [], "total": 0})
        if url.startswith("/rest/api/2/issue/"):
            return JsonResponse(issue)
        return JsonResponse(_search_page([issue]))


def _pay_7_item(**extra) -> ContentItem:
    return ContentItem(
        item_id="jira-PAY-7",
        title="PAY-7: PAY-7",
        source_url="https://jira.example.test/browse/PAY-7",
        last_modified=datetime(2026, 5, 21, tzinfo=timezone.utc),
        content_type="application/json",
        version="2026-05-21T08:00:00.000+0000",
        extra={"issue_key": "PAY-7", **extra},
    )


async def _read_hydrated_by_search(gene: JiraGene):
    items = [item async for item in gene.discover()]
    return await gene.fetch(items[0])


async def _read_rediscovered(gene: JiraGene):
    current = await gene.rediscover(_pay_7_item(issue_id="100007"))
    return await gene.fetch(current)


async def _read_per_issue(gene: JiraGene):
    return await gene.fetch(_pay_7_item())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("read_issue", "expected_calls"),
    [
        (_read_hydrated_by_search, [("POST", "/rest/api/2/search")]),
        (_read_rediscovered, [("POST", "/rest/api/2/search")]),
        (
            _read_per_issue,
            [("GET", "/rest/api/2/issue/PAY-7"), ("GET", "/rest/api/2/issue/PAY-7/comment")],
        ),
    ],
    ids=["search", "rediscover", "per-issue"],
)
async def test_every_issue_read_projects_the_whole_embedded_changelog_without_extra_requests(
    read_issue,
    expected_calls,
):
    from memforge.source_projection import ProjectionCoverage

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "include_comments": True},
        source_id="src-jira",
    )
    client = _LongChangelogClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"
    gene._hydrated_issues = {}

    raw = await read_issue(gene)
    projection = _project_jira(raw, await gene.normalize(raw))

    assert [call[0:2] for call in client.calls] == expected_calls
    assert "_changelog_truncated" not in json.loads(raw.body)
    assert _changelog_keys(projection) == [
        str(index + 1) for index in range(LONG_CHANGELOG_HISTORY_COUNT)
    ]
    assert projection.coverage is ProjectionCoverage.COMPLETE_SNAPSHOT


@pytest.mark.asyncio
async def test_limited_changelog_keeps_returned_histories_and_projects_partial_coverage():
    from memforge.source_projection import ProjectionCoverage

    class LimitedChangelogClient(RecordingAsyncClient):
        async def answer(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            return JsonResponse(
                _search_page(
                    [
                        _jira_issue(
                            "PAY-7",
                            issue_id="100007",
                            histories=_histories(LIMITED_CHANGELOG_HISTORY_COUNT),
                            changelog_total=LONG_CHANGELOG_HISTORY_COUNT,
                        )
                    ]
                )
            )

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "include_comments": True},
        source_id="src-jira",
    )
    client = LimitedChangelogClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"

    raw = await _read_hydrated_by_search(gene)
    projection = _project_jira(raw, await gene.normalize(raw))

    # Jira Data Center has no changelog page to ask for, so nothing more is requested.
    assert [call[0:2] for call in client.calls] == [("POST", "/rest/api/2/search")]
    assert json.loads(raw.body)["_changelog_truncated"] == {
        "returned": LIMITED_CHANGELOG_HISTORY_COUNT,
        "total": LONG_CHANGELOG_HISTORY_COUNT,
    }
    assert _changelog_keys(projection) == [
        str(index + 1) for index in range(LIMITED_CHANGELOG_HISTORY_COUNT)
    ]
    assert projection.coverage is ProjectionCoverage.PARTIAL_PROJECTION


@pytest.mark.asyncio
async def test_fetch_describes_jira_image_attachment_with_comment_parent():
    image = b"\x89PNG\r\n\x1a\nknown-jira-image"

    class AttachmentClient(RecordingAsyncClient):
        async def answer(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            if url == "/rest/api/2/attachment/content/7001":
                return BinaryResponse(image, "image/png")
            return JsonResponse(
                _search_page(
                    [
                        _jira_issue(
                            "PAY-123",
                            issue_id="100123",
                            field_overrides={
                                "attachment": [
                                    {
                                        "id": "7001",
                                        "filename": "screenshot.png",
                                        "mimeType": "image/png",
                                        "size": len(image),
                                        "created": "2026-05-21T09:01:00Z",
                                        "content": (
                                            "https://jira.example.test"
                                            "/rest/api/2/attachment/content/7001"
                                        ),
                                    }
                                ]
                            },
                            comments=[
                                {
                                    "id": "comment-9",
                                    "author": {"displayName": "Grace"},
                                    "created": "2026-05-21T09:00:00Z",
                                    "body": "The result is shown in screenshot.png.",
                                    "renderedBody": "<p>The result is shown in screenshot.png.</p>",
                                }
                            ],
                        )
                    ]
                )
            )

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "include_comments": True},
        source_id="src-jira",
    )
    client = AttachmentClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"

    items = [item async for item in gene.discover()]
    raw = await gene.fetch(items[0])

    assert len(raw.artifacts) == 1
    artifact = raw.artifacts[0]
    assert artifact.provider_key == "7001"
    assert artifact.parent_observation_type == "comment"
    assert artifact.parent_provider_key == "comment-9"
    assert artifact.media_type == "image/png"
    assert artifact.declared_size_bytes == len(image)
    assert artifact.locator["request_path"] == "/rest/api/2/attachment/content/7001"
    assert [call[1] for call in client.calls] == ["/rest/api/2/search"]


@pytest.mark.asyncio
async def test_jira_secure_attachment_stream_uses_id_based_locator():
    image = b"\xff\xd8\xffknown-jira-image"

    class AttachmentClient(RecordingAsyncClient):
        async def answer(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            return JsonResponse(
                _search_page(
                    [
                        _jira_issue(
                            "PAY-123",
                            issue_id="100123",
                            field_overrides={
                                "attachment": [
                                    {
                                        "id": "7001",
                                        "filename": "200%_zoom.jpg",
                                        "mimeType": "image/jpeg",
                                        "size": len(image),
                                        "created": "2026-05-21T09:01:00Z",
                                        "content": (
                                            "https://jira.example.test"
                                            "/secure/attachment/7001/200%25_zoom.jpg"
                                        ),
                                    }
                                ]
                            },
                        )
                    ]
                )
            )

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"]},
        source_id="src-jira",
    )
    discovery_client = AttachmentClient(base_url="https://jira.example.test")
    gene._client = discovery_client
    gene._base_url = "https://jira.example.test"

    items = [item async for item in gene.discover()]
    raw = await gene.fetch(items[0])
    artifact = raw.artifacts[0]

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/secure/attachment/7001/attachment-7001"
        return httpx.Response(
            200,
            content=image,
            headers={"content-type": "image/jpeg", "content-length": str(len(image))},
        )

    gene._client = FakeConnectorHttp(handler).client("https://jira.example.test", product_name="Jira")
    gene._auth_mode = "pat"
    try:
        assert artifact.locator["request_path"] == "/secure/attachment/7001/attachment-7001"
        async with gene.open_source_artifact(artifact) as download:
            assert b"".join([chunk async for chunk in download.chunks]) == image
    finally:
        await gene.close()


@pytest.mark.asyncio
async def test_jira_secure_attachment_rejects_descriptor_url_identity_mismatch():
    class AttachmentClient(RecordingAsyncClient):
        async def answer(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            return JsonResponse(
                _search_page(
                    [
                        _jira_issue(
                            "PAY-123",
                            issue_id="100123",
                            field_overrides={
                                "attachment": [
                                    {
                                        "id": "7001",
                                        "filename": "screenshot.jpg",
                                        "mimeType": "image/jpeg",
                                        "size": 4,
                                        "created": "2026-05-21T09:01:00Z",
                                        "content": (
                                            "https://jira.example.test"
                                            "/secure/attachment/9999/screenshot.jpg"
                                        ),
                                    }
                                ]
                            },
                        )
                    ]
                )
            )

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"]},
        source_id="src-jira",
    )
    gene._client = AttachmentClient(base_url="https://jira.example.test")
    gene._base_url = "https://jira.example.test"

    items = [item async for item in gene.discover()]

    with pytest.raises(
        SourceArtifactContractError,
        match="Jira attachment URL identity does not match its descriptor",
    ):
        await gene.fetch(items[0])


@pytest.mark.asyncio
async def test_open_source_artifact_streams_jira_attachment():
    image = b"\x89PNG\r\n\x1a\nknown-jira-image"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rest/api/2/attachment/content/7001"
        return httpx.Response(
            200,
            content=image,
            headers={"content-type": "image/png", "content-length": str(len(image))},
        )

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"]},
        source_id="src-jira",
    )
    gene._client = FakeConnectorHttp(handler).client("https://jira.example.test", product_name="Jira")
    gene._auth_mode = "pat"
    artifact = RawSourceArtifact(
        provider_key="7001",
        parent_observation_type="comment",
        parent_provider_key="comment-9",
        provider_revision="1",
        filename="screenshot.png",
        media_type="image/png",
        declared_size_bytes=len(image),
        locator={"request_path": "/rest/api/2/attachment/content/7001"},
    )
    try:
        async with gene.open_source_artifact(artifact) as download:
            assert b"".join([chunk async for chunk in download.chunks]) == image
            assert download.content_length == len(image)
    finally:
        await gene.close()


@pytest.mark.asyncio
async def test_fetch_does_not_count_unsupported_jira_attachments_against_image_budget():
    image = b"\x89PNG\r\n\x1a\nbounded-jira-image"
    unsupported = [
        {
            "id": str(8000 + index),
            "filename": f"document-{index}.docx",
            "mimeType": (
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            "size": 100,
            "created": "2026-05-21T09:00:00Z",
            "content": (
                "https://jira.example.test"
                f"/rest/api/2/attachment/content/{8000 + index}"
            ),
        }
        for index in range(20)
    ]
    supported = {
        "id": "9001",
        "filename": "screenshot.png",
        "mimeType": "image/png",
        "size": len(image),
        "created": "2026-05-21T09:01:00Z",
        "content": "https://jira.example.test/rest/api/2/attachment/content/9001",
    }

    class AttachmentClient(RecordingAsyncClient):
        async def answer(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            return JsonResponse(
                _search_page(
                    [
                        _jira_issue(
                            "PAY-123",
                            issue_id="100123",
                            field_overrides={"attachment": [*unsupported, supported]},
                        )
                    ]
                )
            )

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"]},
        source_id="src-jira",
    )
    client = AttachmentClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"

    items = [item async for item in gene.discover()]
    raw = await gene.fetch(items[0])

    assert [artifact.provider_key for artifact in raw.artifacts] == ["9001"]
    assert [call[1] for call in client.calls] == ["/rest/api/2/search"]


@pytest.mark.asyncio
async def test_jira_attachment_inventory_is_not_rejected_by_descriptor_count():
    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"]},
        source_id="src-jira",
    )
    issue = _jira_issue(
        "PAY-123",
        issue_id="100123",
        field_overrides={
            "attachment": [
                {
                    "id": str(8000 + index),
                    "filename": f"document-{index}.bin",
                    "mimeType": "application/octet-stream",
                    "size": 1,
                    "created": "2026-05-21T09:00:00Z",
                    "content": f"/rest/api/2/attachment/content/{8000 + index}",
                }
                for index in range(201)
            ]
        },
    )

    assert await gene._fetch_source_artifacts(issue) == ()


@pytest.mark.asyncio
async def test_discover_paces_paginated_jira_search_requests():
    now = 1000.0
    sleeps: list[float] = []

    def clock() -> float:
        return now

    async def fake_sleep(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    searches: list[int] = []

    def handle(request: httpx.Request) -> httpx.Response:
        start_at = json.loads(request.content)["startAt"]
        searches.append(start_at)
        page_size = 1 if start_at == 100 else 50
        issues = [_jira_issue(f"PAY-{index + 1}") for index in range(start_at, start_at + page_size)]
        return httpx.Response(200, json={"startAt": start_at, "total": 101, "issues": issues})

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], "request_interval_ms": 750},
        source_id="src-jira",
    )
    gene._client = ConnectorHttpClient(
        product_name="Jira",
        base_url="https://jira.example.test",
        timeout=5.0,
        follow_redirects=True,
        limiter=atlassian_auth.AtlassianRequestLimiter(min_interval_seconds=0.75, clock=clock),
        sleep=fake_sleep,
        transport=httpx.MockTransport(handle),
    )
    gene._base_url = "https://jira.example.test"

    try:
        items = [item async for item in gene.discover()]
    finally:
        await gene.close()

    assert len(items) == 101
    assert items[0].item_id == "jira-PAY-1"
    assert items[-1].item_id == "jira-PAY-101"
    assert searches == [0, 50, 100]
    assert sleeps == [0.75, 0.75]


@pytest.mark.asyncio
async def test_fetch_tops_up_truncated_hydrated_comments_through_limiter():
    class SearchIssueClient(RecordingAsyncClient):
        async def answer(self, method: str, url: str, **kwargs):
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
        async def answer(self, method: str, url: str, **kwargs):
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
async def test_discover_resets_hydrated_issue_cache_between_runs():
    class OneIssueClient(RecordingAsyncClient):
        async def answer(self, method: str, url: str, **kwargs):
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


@pytest.mark.asyncio
async def test_discover_inventory_uses_lightweight_complete_search_without_hydrating_cache():
    issue = _jira_issue("PAY-1")
    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"]},
        source_id="src-jira",
    )
    gene._base_url = "https://jira.example.test"
    gene._hydrated_issues = {"stale": {}}
    gene._request = AsyncMock(return_value=JsonResponse(_search_page([issue])))

    items = [item async for item in gene.discover_inventory()]

    assert [item.extra["issue_key"] for item in items] == ["PAY-1"]
    assert items[0].extra["attest_materialized_revision"] is True
    assert gene._hydrated_issues == {}
    request = gene._request.await_args.kwargs["json_body"]
    assert request["fields"] == ["summary", "updated", "project"]
    assert "expand" not in request


@pytest.mark.asyncio
async def test_discover_inventory_preserves_lightweight_fields_across_pages():
    class TwoPageInventoryClient(RecordingAsyncClient):
        async def answer(self, method: str, url: str, **kwargs):
            self.calls.append((method, url, kwargs))
            start_at = kwargs["json"]["startAt"]
            issue = _jira_issue("PAY-1" if start_at == 0 else "PAY-2")
            return JsonResponse(_search_page([issue], total=2, start_at=start_at))

    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"]},
        source_id="src-jira",
    )
    client = TwoPageInventoryClient(base_url="https://jira.example.test")
    gene._client = client
    gene._base_url = "https://jira.example.test"

    items = [item async for item in gene.discover_inventory()]

    assert [item.extra["issue_key"] for item in items] == ["PAY-1", "PAY-2"]
    assert [call[2]["json"]["fields"] for call in client.calls] == [
        ["summary", "updated", "project"],
        ["summary", "updated", "project"],
    ]


class InventorySearchClient(RecordingAsyncClient):
    """Answers each search by its fields: the inventory, the full read, and the revision check.

    Like Jira Data Center, a search writes ``updated`` at whole seconds and an
    issue read writes it with milliseconds.
    """

    def __init__(self, *, full: dict, attested: dict, comments: list[dict] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.full = full
        self.attested = attested
        self.comments = comments or []

    async def answer(self, method: str, url: str, **kwargs):
        self.calls.append((method, url, kwargs))
        if url == "/rest/api/2/search":
            fields = kwargs["json"]["fields"]
            if fields == ["updated"]:
                return JsonResponse(_search_page([self.attested]))
            return JsonResponse(_search_page([self.full]))
        if url.endswith("/comment"):
            return JsonResponse({"startAt": 0, "comments": self.comments, "total": len(self.comments)})
        key = url.split("/issue/", 1)[1].split("/", 1)[0]
        return JsonResponse(_jira_issue(key, field_overrides={"updated": "2026-05-21T08:00:00.517+0000"}))


def _inventory_gene(client: RecordingAsyncClient, **config) -> JiraGene:
    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], **config},
        source_id="src-jira",
    )
    gene._client = client
    gene._base_url = "https://jira.example.test"
    return gene


def _search_fields(client: RecordingAsyncClient) -> list[object]:
    return [call[2]["json"]["fields"] if call[1] == "/rest/api/2/search" else call[1] for call in client.calls]


@pytest.mark.asyncio
async def test_inventory_fetch_reads_the_issue_through_search():
    issue = _jira_issue("PAY-1")
    client = InventorySearchClient(full=issue, attested=issue, base_url="https://jira.example.test")
    gene = _inventory_gene(client, include_comments=True)
    item = [item async for item in gene.discover_inventory()][0]

    raw = await gene.fetch(item)

    assert json.loads(raw.body)["fields"]["updated"] == item.version == "2026-05-21T08:00:00.000+0000"
    # The full read carries complete comments, so nothing is read apart to check again.
    assert _search_fields(client) == [["summary", "updated", "project"], ["*all"]]
    scope = "(project in (PAY) AND issuetype in (Epic,Story,Bug,Task))"
    assert client.calls[1][2]["json"]["jql"] == f"issuekey = 100001 AND {scope}"


@pytest.mark.asyncio
async def test_inventory_fetch_rejects_an_issue_changed_before_its_full_read():
    original = _jira_issue("PAY-1")
    updated = _jira_issue("PAY-1", field_overrides={"updated": "2026-05-21T08:01:00.000+0000"})

    class ChangedBeforeReadClient(InventorySearchClient):
        async def answer(self, method: str, url: str, **kwargs):
            if url == "/rest/api/2/search" and kwargs["json"]["fields"] == ["summary", "updated", "project"]:
                self.calls.append((method, url, kwargs))
                return JsonResponse(_search_page([original]))
            return await super().answer(method, url, **kwargs)

    client = ChangedBeforeReadClient(full=updated, attested=updated, base_url="https://jira.example.test")
    gene = _inventory_gene(client)
    item = [item async for item in gene.discover_inventory()][0]

    with pytest.raises(RuntimeError, match="changed during materialization; retry inventory"):
        await gene.fetch(item)

    assert _search_fields(client) == [["summary", "updated", "project"], ["*all"]]


@pytest.mark.asyncio
async def test_inventory_fetch_rejects_an_issue_whose_key_changed_or_that_left_the_scope():
    listed = _jira_issue("PAY-1")
    moved = _jira_issue("OPS-7", issue_id="100001")

    class ListedThenReadClient(InventorySearchClient):
        async def answer(self, method: str, url: str, **kwargs):
            if url == "/rest/api/2/search" and kwargs["json"]["fields"] == ["summary", "updated", "project"]:
                self.calls.append((method, url, kwargs))
                return JsonResponse(_search_page([listed]))
            if url == "/rest/api/2/search" and self.full is None:
                self.calls.append((method, url, kwargs))
                return JsonResponse(_search_page([]))
            return await super().answer(method, url, **kwargs)

    for full in (moved, None):
        client = ListedThenReadClient(full=full, attested=listed, base_url="https://jira.example.test")
        gene = _inventory_gene(client)
        item = [item async for item in gene.discover_inventory()][0]

        with pytest.raises(RuntimeError, match="changed during materialization; retry inventory"):
            await gene.fetch(item)

        assert gene._hydrated_issues == {}


@pytest.mark.asyncio
async def test_inventory_fetch_rejects_comment_body_from_newer_issue_revision():
    original = _jira_issue("PAY-1", comments=[{"body": "Old comment"}], comment_total=2)
    updated = _jira_issue("PAY-1", field_overrides={"updated": "2026-05-21T08:01:00.000+0000"})
    client = InventorySearchClient(
        full=original,
        attested=updated,
        comments=[{"id": "comment-new", "body": "New comment"}],
        base_url="https://jira.example.test",
    )
    gene = _inventory_gene(client, include_comments=True)
    item = [item async for item in gene.discover_inventory()][0]

    with pytest.raises(RuntimeError, match="changed during materialization; retry inventory"):
        await gene.fetch(item)

    assert _search_fields(client) == [
        ["summary", "updated", "project"],
        ["*all"],
        "/rest/api/2/issue/PAY-1/comment",
        ["updated"],
    ]

def _answers(*responses: httpx.Response | Exception):
    """A handler answering each request with the next response, or raising the next error; the last one repeats."""
    queue = list(responses)

    def handle(request: httpx.Request) -> httpx.Response:
        answer = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    return handle


@pytest.mark.asyncio
async def test_discover_retries_jira_rate_limit():
    gene, http, paths = await _authenticated_gene(
        {"auth_mode": "pat", "pat": "jira-pat"},
        _answers(httpx.Response(429), httpx.Response(429), httpx.Response(200, json=_search_page([]))),
    )
    try:
        items = [item async for item in gene.discover()]
    finally:
        await gene.close()

    assert items == []
    assert paths == ["POST /rest/api/2/search"] * 3


@pytest.mark.asyncio
async def test_discover_reports_jira_rate_limit_after_retries():
    gene, _http, paths = await _authenticated_gene({"auth_mode": "pat", "pat": "jira-pat"}, _answers(httpx.Response(429)))
    try:
        with pytest.raises(RuntimeError, match="Jira search failed: Jira rate limit persisted"):
            [item async for item in gene.discover()]
    finally:
        await gene.close()

    assert paths == ["POST /rest/api/2/search"] * 4


@pytest.mark.asyncio
async def test_discover_retries_a_jira_search_server_error_as_a_safe_read():
    gene, _http, paths = await _authenticated_gene(
        {"auth_mode": "pat", "pat": "jira-pat"},
        _answers(httpx.Response(500), httpx.Response(503), httpx.Response(200, json=_search_page([]))),
    )
    try:
        items = [item async for item in gene.discover()]
    finally:
        await gene.close()

    assert items == []
    assert paths == ["POST /rest/api/2/search"] * 3


@pytest.mark.asyncio
async def test_discover_fails_when_a_jira_search_server_error_persists():
    gene, _http, paths = await _authenticated_gene({"auth_mode": "pat", "pat": "jira-pat"}, _answers(httpx.Response(502)))
    try:
        with pytest.raises(RuntimeError, match="Jira search failed: Jira answered 502 Bad Gateway after 4 attempts"):
            [item async for item in gene.discover()]
    finally:
        await gene.close()

    assert paths == ["POST /rest/api/2/search"] * 4


@pytest.mark.asyncio
async def test_discover_raises_when_jira_search_is_unauthorized():
    class UnauthorizedClient(RecordingAsyncClient):
        async def answer(self, method: str, url: str, **kwargs):
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
async def test_fetch_reports_comment_rate_limit_when_comments_are_enabled():
    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/comment"):
            return httpx.Response(429)
        return httpx.Response(200, json=_jira_issue("PAY-123"))

    gene, _http, paths = await _authenticated_gene(
        {"auth_mode": "pat", "pat": "jira-pat", "include_comments": True}, handle
    )
    item = ContentItem(
        item_id="jira-PAY-123",
        title="PAY-123: Example issue",
        source_url="https://jira.example.test/browse/PAY-123",
        last_modified=datetime(2026, 5, 21, tzinfo=timezone.utc),
        content_type="application/json",
        extra={"issue_key": "PAY-123"},
    )
    try:
        with pytest.raises(httpx.HTTPStatusError, match="Jira rate limit persisted"):
            await gene.fetch(item)
    finally:
        await gene.close()

    assert paths == ["GET /rest/api/2/issue/PAY-123", *["GET /rest/api/2/issue/PAY-123/comment"] * 4]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("slow"),
        httpx.ConnectError("flaky"),
    ],
    ids=["timeout", "connect-error"],
)
async def test_discover_retries_once_on_a_transient_transport_error(error):
    gene, http, paths = await _authenticated_gene(
        {"auth_mode": "browser_cookie", "jira_cookie": "JSESSIONID=ok"},
        _answers(error, httpx.Response(200, json=_search_page([]))),
    )
    try:
        items = [item async for item in gene.discover()]
    finally:
        await gene.close()

    assert items == []  # search returned no issues once the retry succeeded
    assert paths == ["POST /rest/api/2/search"] * 2
    assert http.waits[0] == 1.0


@pytest.mark.asyncio
async def test_discover_reports_an_unreachable_jira_with_the_vpn_hint():
    gene, _http, paths = await _authenticated_gene(
        {"auth_mode": "pat", "pat": "jira-pat"}, _answers(httpx.ConnectError("All connection attempts failed"))
    )
    try:
        with pytest.raises(RuntimeError, match="check VPN or network") as raised:
            [item async for item in gene.discover()]
    finally:
        await gene.close()

    assert "All connection attempts failed" in str(raised.value)
    assert paths == ["POST /rest/api/2/search"] * 4


def _transport_gene(handler, config: dict | None = None) -> JiraGene:
    gene = JiraGene(
        config={"base_url": "https://jira.example.test", "projects": ["PAY"], **(config or {})},
        source_id="src-jira",
    )
    gene._client = FakeConnectorHttp(handler).client("https://jira.example.test", product_name="Jira")
    gene._base_url = "https://jira.example.test"
    gene._auth_mode = "pat"
    gene._hydrated_issues = {}
    return gene


def _stored_issue(key: str, issue_id: str) -> ContentItem:
    return ContentItem(
        item_id=f"jira-{key}",
        title=key,
        source_url=f"https://jira.example.test/browse/{key}",
        last_modified=datetime(2026, 5, 21, tzinfo=timezone.utc),
        extra={"issue_id": issue_id, "issue_key": key},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "expected_jql"),
    [
        (
            {},
            "project in (PAY) AND issuetype in (Epic,Story,Bug,Task) ORDER BY key ASC",
        ),
        (
            {"query_mode": "advanced", "jql": "project = PAY AND updated >= -30d ORDER BY updated DESC"},
            "project = PAY AND updated >= -30d ORDER BY key ASC",
        ),
    ],
    ids=["simple", "advanced"],
)
async def test_scope_listing_pages_the_configured_jql_without_since_by_identifier(config, expected_jql):
    requests: list[dict] = []
    page_size = 2
    issues = [_jira_issue(f"PAY-{number}") for number in range(1, 4)]

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        start = body["startAt"]
        return httpx.Response(200, json=_search_page(issues[start:start + page_size], total=len(issues), start_at=start))

    gene = _transport_gene(handler, config)
    gene._hydrated_issues = {"PAY-9": {"cached": True}}
    try:
        listing = await gene.list_scope()
    finally:
        await gene.close()

    assert listing.kind.value == "query"
    assert listing.doc_ids == {"jira-PAY-1", "jira-PAY-2", "jira-PAY-3"}
    assert {body["jql"] for body in requests} == {expected_jql}
    assert all(body["fields"] == ["updated"] and "expand" not in body for body in requests)
    # Discovery's hydrated issues are kept for this run's fetch.
    assert gene._hydrated_issues == {"PAY-9": {"cached": True}}


@pytest.mark.asyncio
async def test_scope_listing_that_ends_before_its_total_raises():
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json=_search_page([] if body["startAt"] else [_jira_issue("PAY-1")], total=2, start_at=body["startAt"]))

    gene = _transport_gene(handler)
    try:
        with pytest.raises(RuntimeError, match="declared total"):
            await gene.list_scope()
    finally:
        await gene.close()


@pytest.mark.asyncio
async def test_local_agent_jira_is_listed_by_its_device_manifest():
    config = {"base_url": "https://jira.example.test", "sync_mode": "local_agent"}
    gene = JiraGene(config=config, source_id="src-jira")

    assert JiraGene.lists_scope(config) is True
    with pytest.raises(ValueError, match="snapshot manifest"):
        await gene.list_scope()


@pytest.mark.asyncio
async def test_confirming_unlisted_issues_keeps_every_issue_jira_still_has():
    searches: list[dict] = []
    reads: list[str] = []
    # PAY-1 is found by the batch search. PAY-2 is deleted, PAY-3 is refused
    # for this one issue and PAY-4 is found when read by id.
    status_by_id = {"100002": httpx.codes.NOT_FOUND, "100003": httpx.codes.FORBIDDEN, "100004": httpx.codes.OK}

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rest/api/2/search":
            body = json.loads(request.content)
            searches.append(body)
            return httpx.Response(200, json=_search_page([_jira_issue("PAY-1")]))
        issue_id = request.url.path.rsplit("/", 1)[-1]
        reads.append(issue_id)
        status = status_by_id[issue_id]
        return httpx.Response(status, json=_jira_issue("PAY-4") if status == httpx.codes.OK else {"errorMessages": []})

    gene = _transport_gene(handler)
    items = [_stored_issue(f"PAY-{number}", str(100000 + number)) for number in range(1, 5)]
    try:
        answers = await gene.confirm_absent(items)
    finally:
        await gene.close()

    assert answers == {
        "jira-PAY-1": AbsenceAnswer.EXISTS,
        "jira-PAY-2": AbsenceAnswer.ABSENT,
        "jira-PAY-3": AbsenceAnswer.HIDDEN,
        "jira-PAY-4": AbsenceAnswer.EXISTS,
    }
    assert [body["jql"] for body in searches] == ["issuekey in (100001, 100002, 100003, 100004)"]
    assert searches[0]["validateQuery"] is False
    assert reads == ["100002", "100003", "100004"]


@pytest.mark.asyncio
async def test_confirming_unlisted_issues_searches_in_batches_of_one_page():
    from memforge.genes.jira_gene import JIRA_INVENTORY_MAX_RESULTS

    batch_sizes: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        refs = body["jql"].removeprefix("issuekey in (").removesuffix(")").split(", ")
        batch_sizes.append(len(refs))
        return httpx.Response(200, json=_search_page([_jira_issue(f"PAY-{int(ref) - 100000}") for ref in refs]))

    gene = _transport_gene(handler)
    items = [_stored_issue(f"PAY-{number}", str(100000 + number)) for number in range(1, JIRA_INVENTORY_MAX_RESULTS + 2)]
    try:
        answers = await gene.confirm_absent(items)
    finally:
        await gene.close()

    assert set(answers.values()) == {AbsenceAnswer.EXISTS}
    assert set(answers) == {item.item_id for item in items}
    assert batch_sizes == [JIRA_INVENTORY_MAX_RESULTS, 1]


@pytest.mark.asyncio
async def test_confirming_unlisted_issues_fails_when_jira_rejects_the_credential():
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/rest/api/2/search":
            return httpx.Response(200, json=_search_page([]))
        return httpx.Response(httpx.codes.UNAUTHORIZED, json={"errorMessages": ["unauthorized"]})

    gene = _transport_gene(handler)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await gene.confirm_absent([_stored_issue("PAY-2", "100002")])
    finally:
        await gene.close()


@pytest.mark.asyncio
async def test_a_device_confirms_an_issue_by_the_key_the_server_names_it_by():
    searched: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        searched.append(json.loads(request.content)["jql"])
        return httpx.Response(200, json=_search_page([_jira_issue("PAY-1"), _jira_issue("PAY-5")]))

    # A device names its Documents by Source and issue key, not by the server's ``jira-KEY``.
    stored_with_id = _stored_issue("PAY-1", "100001")
    stored_by_key = ContentItem(
        item_id="jira-src-pay-5-abc",
        title="PAY-5",
        source_url="https://jira.example.test/browse/PAY-5",
        last_modified=datetime(2026, 10, 23, tzinfo=timezone.utc),
        extra={"issue_key": "PAY-5"},
    )
    assert JiraGene.provider_key(stored_with_id) == "100001"
    assert JiraGene.provider_key(stored_by_key) == "PAY-5"

    gene = _transport_gene(handler)
    items = [
        JiraGene.confirmation_item("jira-src-pay-1-abc", "100001"),
        JiraGene.confirmation_item("jira-src-pay-5-abc", "PAY-5"),
    ]
    try:
        answers = await gene.confirm_absent(items)
    finally:
        await gene.close()

    assert answers == {"jira-src-pay-1-abc": AbsenceAnswer.EXISTS, "jira-src-pay-5-abc": AbsenceAnswer.EXISTS}
    assert searched == ["issuekey in (100001, PAY-5)"]
    with pytest.raises(ValueError, match="neither an issue id nor an issue key"):
        JiraGene.confirmation_item("jira-src-x", "PAY-1) OR (project = SECRET")


@pytest.mark.asyncio
async def test_complete_hydrated_comments_without_rendering_fetch_attested_comment_snapshot():
    issue = _jira_issue("PAY-123", comments=[{"id": "c1", "body": "*Require approval.*"}])
    rendered = {**issue["fields"]["comment"]["comments"][0], "renderedBody": "<p><strong>Require approval.</strong></p>"}
    class CommentClient(RecordingAsyncClient):
        async def answer(self, method, url, **kwargs):
            self.calls.append((method, url, kwargs))
            assert method == "GET" and url == "/rest/api/2/issue/PAY-123/comment"
            assert kwargs["params"]["expand"] == "renderedBody"
            return JsonResponse({"startAt": 0, "total": 1, "comments": [rendered]})
    gene = JiraGene(config={"base_url": "https://jira.example.test", "include_comments": True}, source_id="src")
    gene._client = CommentClient(base_url="https://jira.example.test")
    gene._hydrated_issues = {"PAY-123": issue}
    from memforge.models import ContentItem
    item = ContentItem(item_id="jira-PAY-123", title="Rule", source_url=None, last_modified=datetime.now(timezone.utc), extra={"issue_key": "PAY-123"})
    raw = await gene.fetch(item)
    assert json.loads(raw.body)["_comments"] == [rendered]
    assert len(gene._client.calls) == 1


def _rediscovery_handler(searches: list[dict], issues_by_id: dict[str, dict]):
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/rest/api/2/search"
        body = json.loads(request.content)
        searches.append(body)
        issue_ref = body["jql"].removeprefix("issuekey = ").split(" ", 1)[0]
        issues = [issues_by_id[issue_ref]] if issue_ref in issues_by_id else []
        return httpx.Response(200, json=_search_page(issues))

    return handler


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("config", "expected_jql"),
    [
        (
            {"projects": ["PAY"], "issue_types": ["Story"], "jql_filter": "labels = important"},
            "issuekey = 100001 AND (project in (PAY) AND issuetype in (Story) AND (labels = important))",
        ),
        (
            {"query_mode": "advanced", "jql": "project = PAY OR labels = payroll ORDER BY Rank ASC"},
            "issuekey = 100001 AND (project = PAY OR labels = payroll)",
        ),
        ({"query_mode": "advanced", "jql": "ORDER BY created DESC"}, "issuekey = 100001"),
        ({"query_mode": "advanced", "jql": ""}, "issuekey = 100001"),
    ],
    ids=["simple-with-filter", "advanced-with-order-and-top-level-or", "advanced-order-only", "advanced-empty"],
)
async def test_rediscovery_searches_for_the_issue_within_the_configured_jql(config, expected_jql):
    searches: list[dict] = []
    gene = _transport_gene(_rediscovery_handler(searches, {"100001": _jira_issue("PAY-1", issue_id="100001")}), config)
    try:
        current = await gene.rediscover(_stored_issue("PAY-1", "100001"))
    finally:
        await gene.close()

    assert current.item_id == "jira-PAY-1"
    assert [body["jql"] for body in searches] == [expected_jql]
    assert searches[0]["validateQuery"] is False
    # Fetch reads the issue the search returned instead of asking again.
    assert set(gene._hydrated_issues) == {"PAY-1"}


@pytest.mark.asyncio
async def test_rediscovery_finds_an_issue_whose_key_changed_by_its_id():
    searches: list[dict] = []
    moved = _jira_issue("OPS-7", issue_id="100001")
    gene = _transport_gene(_rediscovery_handler(searches, {"100001": moved}), {"projects": ["PAY", "OPS"]})
    try:
        current = await gene.rediscover(_stored_issue("PAY-1", "100001"))
    finally:
        await gene.close()

    assert (current.item_id, current.extra["issue_id"]) == ("jira-OPS-7", "100001")


@pytest.mark.asyncio
async def test_rediscovery_gives_nothing_for_an_issue_outside_the_jql():
    searches: list[dict] = []
    gene = _transport_gene(_rediscovery_handler(searches, {}), {"projects": ["PAY"]})
    try:
        assert await gene.rediscover(_stored_issue("PAY-2", "100002")) is None
    finally:
        await gene.close()

    assert len(searches) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(500, json={"errorMessages": ["Internal server error"]}),
        httpx.Response(200, json=_search_page([], total=1)),
        httpx.Response(200, json=_search_page([_jira_issue("PAY-5", issue_id="100005")])),
    ],
    ids=["search-error", "ends-before-total", "another-issue"],
)
async def test_rediscovery_raises_when_the_search_fails_or_answers_inconsistently(response):
    async def handler(request: httpx.Request) -> httpx.Response:
        return response

    gene = _transport_gene(handler, {"projects": ["PAY"]})
    try:
        with pytest.raises(RuntimeError):
            await gene.rediscover(_stored_issue("PAY-1", "100001"))
    finally:
        await gene.close()


@pytest.mark.asyncio
async def test_rediscovery_by_an_old_key_accepts_the_issue_under_its_new_key():
    # Jira resolves the key an issue had before it moved, and answers with its current key.
    searches: list[dict] = []
    moved = _jira_issue("OPS-7", issue_id="100001")
    gene = _transport_gene(_rediscovery_handler(searches, {"PAY-1": moved}), {"projects": ["PAY", "OPS"]})
    stored = ContentItem(
        item_id="jira-PAY-1",
        title="PAY-1",
        source_url="https://jira.example.test/browse/PAY-1",
        last_modified=datetime(2026, 5, 21, tzinfo=timezone.utc),
        extra={"issue_key": "PAY-1"},
    )
    try:
        current = await gene.rediscover(stored)
    finally:
        await gene.close()

    assert [body["jql"].split(" AND ", 1)[0] for body in searches] == ["issuekey = PAY-1"]
    assert (current.item_id, current.extra["issue_id"]) == ("jira-OPS-7", "100001")


@pytest.mark.asyncio
@pytest.mark.parametrize("issue_ref", ["PAY-1) OR (project = HR", "pay-1", "PAY-", "1 OR 2"])
async def test_an_issue_reference_that_is_not_an_id_or_key_never_reaches_jql(issue_ref):
    searches: list[dict] = []
    gene = _transport_gene(_rediscovery_handler(searches, {}), {"projects": ["PAY"]})
    stored = _stored_issue("PAY-1", issue_ref)
    try:
        with pytest.raises(ValueError, match="neither an issue id nor an issue key"):
            await gene.rediscover(stored)
        with pytest.raises(ValueError, match="neither an issue id nor an issue key"):
            await gene.confirm_absent([stored])
    finally:
        await gene.close()

    assert searches == []


@pytest.mark.asyncio
async def test_search_refuses_a_page_whose_start_at_does_not_continue_the_walk():
    async def handler(request: httpx.Request) -> httpx.Response:
        start = json.loads(request.content)["startAt"]
        # The second page claims to start at 0 again, overlapping the first.
        issues = [_jira_issue("PAY-1")] if start == 0 else [_jira_issue("PAY-2")]
        return httpx.Response(200, json=_search_page(issues, total=2, start_at=0))

    gene = _transport_gene(handler)
    try:
        with pytest.raises(RuntimeError, match="Jira search response startAt does not match the requested page"):
            await gene.list_scope()
    finally:
        await gene.close()


@pytest.mark.asyncio
async def test_search_refuses_issues_beyond_the_total_before_yielding_that_page():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_search_page([_jira_issue("PAY-1"), _jira_issue("PAY-2")], total=1))

    gene = _transport_gene(handler)
    yielded: list[dict] = []
    try:
        with pytest.raises(RuntimeError, match="Jira search unique issue count did not match total"):
            async for issue in gene._search_issues("project = PAY", fields=["updated"], expand=None, max_results=50):
                yielded.append(issue)
    finally:
        await gene.close()
    assert yielded == []


@pytest.mark.asyncio
async def test_search_refused_by_jira_raises_query_rejected():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"errorMessages": ["The value 'NOPE' does not exist for the field 'project'."]})

    gene = _transport_gene(handler)
    try:
        with pytest.raises(JiraQueryRejected, match="Jira rejected the JQL"):
            await gene.list_scope()
    finally:
        await gene.close()
