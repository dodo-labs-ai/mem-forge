from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from memforge.genes.confluence_gene import ConfluenceGene, PREVIEW_DISCOVERY_LIMIT_CONFIG_KEY


def test_confluence_schema_hides_runtime_transport_fields_from_ui():
    fields = {field.key: field for field in ConfluenceGene.config_schema().fields}

    assert fields["pat"].required is True
    assert fields["pat"].advanced is False
    assert fields["base_url"].label == "Wiki URL"
    assert fields["spaces"].required is False
    assert fields["sync_mode"].advanced is False
    assert "api_prefix" not in fields
    assert "tls_ca_bundle" not in fields


def test_confluence_normalizes_corporate_wiki_page_url_to_page_tree_scope():
    config = {
        "base_url": "https://wiki.company.example/wiki/spaces/PAY/pages/5695886009/Flexible+Payroll",
        "pat": "confluence-pat",
    }

    ConfluenceGene.normalize_config(config)

    assert config["base_url"] == "https://wiki.company.example"
    assert config["api_prefix"] == "/wiki"
    assert config["spaces"] == ["PAY"]
    assert config["page_tree_root"] == "5695886009"
    assert config["sync_mode"] == "page_tree"


def test_confluence_normalizes_space_url_without_page_tree_root():
    config = {
        "base_url": "https://team.atlassian.net/wiki/spaces/ENG",
        "pat": "confluence-pat",
    }

    ConfluenceGene.normalize_config(config)

    assert config["base_url"] == "https://team.atlassian.net"
    assert config["api_prefix"] == "/wiki"
    assert config["spaces"] == ["ENG"]
    assert config.get("page_tree_root") is None
    assert "sync_mode" not in config


class RecordingAsyncClient:
    instances: list["RecordingAsyncClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[str] = []
        self.closed = False
        RecordingAsyncClient.instances.append(self)

    async def request(self, _method: str, url: str, **_kwargs):
        self.calls.append(url)
        return JsonResponse({"results": []})

    async def get(self, url: str, **kwargs):
        return await self.request("GET", url, **kwargs)

    async def aclose(self) -> None:
        self.closed = True


class FailingAuthClient(RecordingAsyncClient):
    async def request(self, _method: str, url: str, **_kwargs):
        self.calls.append(url)
        return JsonResponse({}, status_code=401)


class RootRestFallbackClient(RecordingAsyncClient):
    async def request(self, _method: str, url: str, **_kwargs):
        self.calls.append(url)
        if url == "/wiki/rest/api/space":
            return JsonResponse({}, status_code=404)
        return JsonResponse({"results": []})


class JsonResponse:
    def __init__(self, payload: dict, status_code: int = 200, headers: dict | None = None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://wiki.example.com/wiki/rest/api/content/123")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("request failed", request=request, response=response)


class NotFoundClient:
    async def request(self, _method: str, _url: str, **_kwargs):
        return JsonResponse({}, status_code=404)

    async def get(self, url: str, **kwargs):
        return await self.request("GET", url, **kwargs)


class RateLimitedThenSuccessClient:
    def __init__(self):
        self.calls: list[str] = []

    async def request(self, _method: str, url: str, **_kwargs):
        self.calls.append(url)
        if len(self.calls) == 1:
            return JsonResponse({}, status_code=429, headers={"Retry-After": "1"})
        return JsonResponse(
            {
                "id": "123",
                "title": "Root Page",
                "space": {"key": "PAY"},
                "version": {"number": 7, "when": "2026-05-24T00:00:00Z"},
                "_links": {"webui": "/display/PAY/Root+Page"},
            }
        )

    async def get(self, url: str, **kwargs):
        return await self.request("GET", url, **kwargs)


class AlwaysRateLimitedClient:
    def __init__(self):
        self.calls: list[str] = []

    async def request(self, _method: str, url: str, **_kwargs):
        self.calls.append(url)
        return JsonResponse({}, status_code=429, headers={"Retry-After": "1"})

    async def get(self, url: str, **kwargs):
        return await self.request("GET", url, **kwargs)


class PageTreeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def request(self, _method: str, url: str, **_kwargs):
        self.calls.append(url)
        if url.endswith("/content/root"):
            return JsonResponse(_page("root", "Root", "2026-05-20T00:00:00Z"))
        if url.endswith("/content/root/child/page"):
            return JsonResponse({"results": [_page("parent", "Unchanged Parent", "2026-05-20T00:00:00Z")]})
        if url.endswith("/content/parent/child/page"):
            return JsonResponse({"results": [_page("target", "Changed Child", "2026-05-26T14:51:21Z")]})
        if url.endswith("/content/target/child/page"):
            return JsonResponse({"results": []})
        return JsonResponse({"results": []})

    async def get(self, url: str, **kwargs):
        return await self.request("GET", url, **kwargs)


class PreviewLimitClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []

    async def request(self, _method: str, url: str, **kwargs):
        self.requests.append((url, kwargs.get("params") or {}))
        if url.endswith("/content/root"):
            return JsonResponse(_page("root", "Root", "2026-05-20T00:00:00Z"))
        if url.endswith("/content/root/child/page"):
            return JsonResponse(
                {
                    "results": [
                        _page("child-1", "Child 1", "2026-05-21T00:00:00Z"),
                        _page("child-2", "Child 2", "2026-05-22T00:00:00Z"),
                        _page("child-3", "Child 3", "2026-05-23T00:00:00Z"),
                    ],
                }
            )
        return JsonResponse({"results": []})

    async def get(self, url: str, **kwargs):
        return await self.request("GET", url, **kwargs)


def _page(page_id: str, title: str, when: str) -> dict:
    return {
        "id": page_id,
        "title": title,
        "space": {"key": "PAY"},
        "version": {"number": 1, "when": when},
        "metadata": {"labels": {"results": []}},
        "_links": {"webui": f"/display/PAY/{title.replace(' ', '+')}"},
    }


@pytest.mark.asyncio
async def test_authenticate_uses_bearer_pat_and_normalizes_ui_confluence_wiki_url(monkeypatch):
    RecordingAsyncClient.instances.clear()
    monkeypatch.setattr("memforge.genes.confluence_gene.httpx.AsyncClient", RecordingAsyncClient)

    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.test/wiki/",
            "spaces": ["PAY"],
            "pat": "confluence-pat",
        },
        source_id="src-confluence",
    )

    await gene.authenticate()

    assert gene._base_url == "https://wiki.example.test"
    assert RecordingAsyncClient.instances[-1].kwargs["base_url"] == "https://wiki.example.test"
    assert RecordingAsyncClient.instances[-1].kwargs["headers"]["Authorization"] == "Bearer confluence-pat"
    assert RecordingAsyncClient.instances[-1].kwargs["verify"] is True
    assert RecordingAsyncClient.instances[-1].calls == ["/wiki/rest/api/space"]


@pytest.mark.asyncio
async def test_authenticate_uses_configured_confluence_ca_bundle(monkeypatch, tmp_path):
    RecordingAsyncClient.instances.clear()
    monkeypatch.setattr("memforge.genes.confluence_gene.httpx.AsyncClient", RecordingAsyncClient)
    ca_bundle = tmp_path / "corp-ca.pem"
    ca_bundle.write_text("test-ca", encoding="utf-8")

    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.test/wiki/",
            "spaces": ["PAY"],
            "pat": "confluence-pat",
            "tls_ca_bundle": str(ca_bundle),
        },
        source_id="src-confluence",
    )

    await gene.authenticate()

    assert RecordingAsyncClient.instances[-1].kwargs["verify"] == str(ca_bundle)


@pytest.mark.asyncio
async def test_authenticate_falls_back_to_root_rest_api_for_plain_confluence_url(monkeypatch):
    RecordingAsyncClient.instances.clear()
    monkeypatch.setattr("memforge.genes.confluence_gene.httpx.AsyncClient", RootRestFallbackClient)

    gene = ConfluenceGene(
        config={
            "base_url": "https://confluence.example.test",
            "spaces": ["PAY"],
            "pat": "confluence-pat",
        },
        source_id="src-confluence",
    )

    await gene.authenticate()

    assert gene._base_url == "https://confluence.example.test"
    assert gene._api_prefix == ""
    assert RecordingAsyncClient.instances[-1].calls == [
        "/wiki/rest/api/space",
        "/rest/api/space",
    ]


@pytest.mark.asyncio
async def test_authenticate_rejects_insecure_confluence_url():
    gene = ConfluenceGene(
        config={"base_url": "http://wiki.example.test", "spaces": ["PAY"], "pat": "confluence-pat"},
        source_id="src-confluence",
    )

    with pytest.raises(ValueError, match="HTTPS"):
        await gene.authenticate()


@pytest.mark.asyncio
async def test_authenticate_requires_confluence_pat():
    gene = ConfluenceGene(
        config={"base_url": "https://wiki.example.test", "spaces": ["PAY"]},
        source_id="src-confluence",
    )

    with pytest.raises(ValueError, match="PAT"):
        await gene.authenticate()


@pytest.mark.asyncio
async def test_authenticate_closes_confluence_client_when_pat_validation_fails(monkeypatch):
    RecordingAsyncClient.instances.clear()
    monkeypatch.setattr("memforge.genes.confluence_gene.httpx.AsyncClient", FailingAuthClient)
    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.test",
            "spaces": ["PAY"],
            "pat": "expired-pat",
        },
        source_id="src-confluence",
    )

    with pytest.raises(httpx.HTTPStatusError):
        await gene.authenticate()

    assert RecordingAsyncClient.instances[-1].closed is True
    assert not hasattr(gene, "_client")


@pytest.mark.asyncio
async def test_page_tree_discovery_raises_when_root_page_cannot_be_fetched():
    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.com",
            "page_tree_root": "123",
            "include_children": True,
        },
        source_id="src-confluence",
    )
    gene._base_url = "https://wiki.example.com"
    gene._api_prefix = "/wiki"
    gene._client = NotFoundClient()

    with pytest.raises(RuntimeError, match="Failed to fetch Confluence page 123"):
        [
            item
            async for item in gene.discover(
                since=datetime(2026, 5, 20, tzinfo=timezone.utc),
            )
        ]


@pytest.mark.asyncio
async def test_page_tree_discovery_retries_confluence_rate_limit(monkeypatch):
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("memforge.genes.atlassian_auth.asyncio.sleep", fake_sleep)
    client = RateLimitedThenSuccessClient()
    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.com",
            "page_tree_root": "123",
            "include_children": False,
        },
        source_id="src-confluence",
    )
    gene._base_url = "https://wiki.example.com"
    gene._api_prefix = "/wiki"
    gene._client = client

    items = [
        item
        async for item in gene.discover(
            since=datetime(2026, 5, 20, tzinfo=timezone.utc),
        )
    ]

    assert [item.title for item in items] == ["Root Page"]
    assert client.calls == ["/wiki/rest/api/content/123", "/wiki/rest/api/content/123"]
    assert sleep_calls == [1.0]


@pytest.mark.asyncio
async def test_page_tree_discovery_reports_confluence_rate_limit_after_retries(monkeypatch):
    async def fake_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("memforge.genes.atlassian_auth.asyncio.sleep", fake_sleep)
    client = AlwaysRateLimitedClient()
    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.com",
            "page_tree_root": "123",
            "include_children": False,
        },
        source_id="src-confluence",
    )
    gene._base_url = "https://wiki.example.com"
    gene._api_prefix = "/wiki"
    gene._client = client

    with pytest.raises(RuntimeError, match="Confluence rate limit persisted"):
        [
            item
            async for item in gene.discover(
                since=datetime(2026, 5, 20, tzinfo=timezone.utc),
            )
        ]

    assert len(client.calls) == 4


@pytest.mark.asyncio
async def test_page_tree_discovery_traverses_unchanged_parent_to_find_changed_child():
    client = PageTreeClient()
    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.com",
            "page_tree_root": "root",
            "include_children": True,
        },
        source_id="src-confluence",
    )
    gene._base_url = "https://wiki.example.com"
    gene._api_prefix = "/wiki"
    gene._client = client

    items = [
        item
        async for item in gene.discover(
            since=datetime(2026, 5, 25, tzinfo=timezone.utc),
        )
    ]

    assert [item.title for item in items] == ["Changed Child"]
    assert "/wiki/rest/api/content/parent/child/page" in client.calls


@pytest.mark.asyncio
async def test_page_tree_preview_limits_child_page_request_size():
    client = PreviewLimitClient()
    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.com",
            "page_tree_root": "root",
            "include_children": True,
            PREVIEW_DISCOVERY_LIMIT_CONFIG_KEY: 3,
        },
        source_id="preview-confluence",
    )
    gene._base_url = "https://wiki.example.com"
    gene._api_prefix = "/wiki"
    gene._client = client

    items = [item async for item in gene.discover(since=None)]

    assert [item.title for item in items] == ["Root", "Child 1", "Child 2"]
    child_requests = [params for url, params in client.requests if url.endswith("/content/root/child/page")]
    assert child_requests == [{"start": 0, "limit": 2, "expand": "version,metadata.labels"}]
