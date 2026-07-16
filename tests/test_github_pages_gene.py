from __future__ import annotations

import base64
import hashlib
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest
import requests

from memforge.genes import GENE_REGISTRY
from memforge.genes.github_pages_gene import GitHubPagesGene
from memforge.models import ContentItem, RawContent


class HtmlResponse:
    def __init__(
        self,
        text: str,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        url: str = "https://github-pages.example.test/pages/org/repo/path/",
    ) -> None:
        self.text = text
        self.content = text.encode("utf-8")
        self.status_code = status_code
        self.headers = headers or {}
        self.url = url
        self.history = []

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError("request failed")

    def json(self) -> dict:
        return {}


class RecordingAsyncClient:
    instances: list["RecordingAsyncClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[tuple[str, str]] = []
        self.closed = False
        RecordingAsyncClient.instances.append(self)

    async def head(self, url: str, **_kwargs):
        self.calls.append(("HEAD", url))
        return HtmlResponse(
            "",
            headers={
                "Last-Modified": "Tue, 26 May 2026 14:51:21 GMT",
                "ETag": '"abc123"',
            },
            url=url,
        )

    async def get(self, url: str, **_kwargs):
        self.calls.append(("GET", url))
        return HtmlResponse(
            "<main><h1>Process Tracking</h1><p>Body</p></main>",
            headers={
                "Last-Modified": "Tue, 26 May 2026 14:51:21 GMT",
                "ETag": '"abc123"',
            },
            url=url,
        )

    async def aclose(self) -> None:
        self.closed = True


class RepoApiResponse(HtmlResponse):
    def __init__(self, payload, **kwargs):
        super().__init__("", **kwargs)
        self._payload = payload

    def json(self):
        return self._payload


class RepoApiClient(RecordingAsyncClient):
    async def get(self, url: str, **_kwargs):
        self.calls.append(("GET", url))
        if url.endswith("/api/v3/repos/org/repo"):
            return RepoApiResponse({"default_branch": "main"}, url=url)
        if url.endswith("/api/v3/repos/org/repo/git/trees/main?recursive=1"):
            return RepoApiResponse(
                {
                    "truncated": False,
                    "tree": [
                        {
                            "path": "docs/cloud-native-platform/process-tracking.md",
                            "type": "blob",
                            "sha": "blob-sha-123",
                        }
                    ]
                },
                url=url,
            )
        if url.endswith(
            "/api/v3/repos/org/repo/commits?sha=main&path=docs/cloud-native-platform/process-tracking.md&per_page=1"
        ):
            return RepoApiResponse([{"commit": {"committer": {"date": "2026-05-26T14:51:21Z"}}}], url=url)
        if url.endswith("/api/v3/repos/org/repo/contents/docs/cloud-native-platform/process-tracking.md?ref=main"):
            body = b"# Process Tracking\n\nUse `/api/` for REST clients."
            encoded = base64.b64encode(body).decode()
            return RepoApiResponse(
                {"sha": "blob-sha-123", "content": encoded, "encoding": "base64", "size": len(body)},
                url=url,
            )
        return await super().get(url, **_kwargs)


class RunbooksRepoApiClient(RecordingAsyncClient):
    async def get(self, url: str, **_kwargs):
        self.calls.append(("GET", url))
        if url.endswith("/api/v3/repos/example-org/runbooks"):
            return RepoApiResponse({"default_branch": "main"}, url=url)
        if url.endswith("/api/v3/repos/example-org/runbooks/git/trees/main?recursive=1"):
            return RepoApiResponse(
                {
                    "truncated": False,
                    "tree": [
                        {
                            "path": "docs/runbooks/Process Tracking/stuck-lock.md",
                            "type": "blob",
                            "sha": "stuck-lock-sha",
                        },
                        {
                            "path": "docs/runbooks/Process Tracking/timed-out-process-instance.md",
                            "type": "blob",
                            "sha": "timed-out-sha",
                        },
                        {
                            "path": "docs/runbooks/Other Runbook/overview.md",
                            "type": "blob",
                            "sha": "other-sha",
                        },
                        {
                            "path": "docs/runbooks/Process Tracking/assets/diagram.png",
                            "type": "blob",
                            "sha": "asset-sha",
                        },
                    ]
                },
                url=url,
            )
        if "/api/v3/repos/example-org/runbooks/commits?sha=main&path=" in url:
            return RepoApiResponse([{"commit": {"committer": {"date": "2026-05-26T14:51:21Z"}}}], url=url)
        if url.endswith(
            "/api/v3/repos/example-org/runbooks/contents/docs/runbooks/Process%20Tracking/stuck-lock.md?ref=main"
        ):
            body = b"# Stuck Lock\n\nUnlock the process."
            encoded = base64.b64encode(body).decode()
            return RepoApiResponse(
                {"sha": "stuck-lock-sha", "content": encoded, "encoding": "base64", "size": len(body)},
                url=url,
            )
        return await super().get(url, **_kwargs)


class TruncatedRepoApiClient(RepoApiClient):
    async def get(self, url: str, **_kwargs):
        if url.endswith("/api/v3/repos/org/repo/git/trees/main?recursive=1"):
            return RepoApiResponse(
                {
                    "truncated": True,
                    "tree": [
                        {
                            "path": "docs/cloud-native-platform/process-tracking.md",
                            "type": "blob",
                            "sha": "blob-sha-123",
                        }
                    ],
                },
                url=url,
            )
        return await super().get(url, **_kwargs)


class MissingTreeRepoApiClient(RepoApiClient):
    async def get(self, url: str, **_kwargs):
        if url.endswith("/api/v3/repos/org/repo/git/trees/main?recursive=1"):
            return RepoApiResponse({}, url=url)
        return await super().get(url, **_kwargs)


class SitemapClient(RecordingAsyncClient):
    async def get(self, url: str, **_kwargs):
        self.calls.append(("GET", url))
        if url.endswith("/sitemap.xml"):
            return HtmlResponse(
                """<?xml version="1.0" encoding="UTF-8"?>
                <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
                  <url>
                    <loc>https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/</loc>
                    <lastmod>2026-05-26T14:51:21Z</lastmod>
                  </url>
                  <url>
                    <loc>https://github-pages.example.test/pages/org/repo/cloud-native-platform/locking/</loc>
                    <lastmod>2026-05-25T00:00:00Z</lastmod>
                  </url>
                  <url>
                    <loc>https://github-pages.example.test/pages/org/repo/general/overview/</loc>
                    <lastmod>2026-05-24T00:00:00Z</lastmod>
                  </url>
                </urlset>""",
                url=url,
            )
        return await super().get(url, **_kwargs)


def test_github_pages_gene_is_registered_and_schema_is_practical():
    assert GENE_REGISTRY["github_pages"] is GitHubPagesGene

    fields = {field.key: field for field in GitHubPagesGene.config_schema().fields}

    assert "base_url" not in fields
    assert fields["auth_mode"].options == ["github_pat", "none"]
    assert fields["sync_mode"].options == ["single_page", "subtree", "explicit_list"]
    assert fields["pat"].required is False
    assert fields["page_url"].required is False
    assert fields["root_url"].required is False
    assert fields["pages"].required is False
    assert fields["max_pages"].default == "200"
    assert fields["max_depth"].advanced is True
    assert fields["max_pages"].advanced is True
    assert fields["exclude_url_patterns"].advanced is True


@pytest.mark.asyncio
async def test_single_page_discovery_uses_canonical_url_metadata(monkeypatch):
    RecordingAsyncClient.instances.clear()
    monkeypatch.setattr("memforge.genes.github_pages_gene._RequestsAsyncClient", RecordingAsyncClient)

    gene = GitHubPagesGene(
        config={
            "auth_mode": "none",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/#tracking",
        },
        source_id="src-pages",
    )

    await gene.authenticate()
    items = [item async for item in gene.discover()]

    assert len(items) == 1
    item = items[0]
    assert item.item_id.startswith("github-pages-")
    assert item.title == "process tracking"
    assert item.source_url == "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking"
    assert item.last_modified == datetime(2026, 5, 26, 14, 51, 21, tzinfo=timezone.utc)
    assert item.version == "sha256:" + hashlib.sha256(
        b"<main><h1>Process Tracking</h1><p>Body</p></main>"
    ).hexdigest()
    assert RecordingAsyncClient.instances[-1].calls == [
        ("GET", "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking")
    ]


@pytest.mark.asyncio
async def test_github_pat_authentication_uses_bearer_header(monkeypatch):
    RecordingAsyncClient.instances.clear()
    monkeypatch.setattr("memforge.genes.github_pages_gene._RequestsAsyncClient", RecordingAsyncClient)

    gene = GitHubPagesGene(
        config={
            "base_url": "https://github-pages.example.test/pages/org/repo/",
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/",
        },
        source_id="src-pages",
    )

    await gene.authenticate()

    assert RecordingAsyncClient.instances[-1].kwargs["headers"]["Authorization"] == "Bearer github-secret"


@pytest.mark.asyncio
async def test_github_pat_discovers_and_fetches_repository_markdown_for_page_url(monkeypatch):
    RepoApiClient.instances.clear()
    monkeypatch.setattr("memforge.genes.github_pages_gene._RequestsAsyncClient", RepoApiClient)

    gene = GitHubPagesGene(
        config={
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/",
        },
        source_id="src-pages",
    )

    await gene.authenticate()
    items = [item async for item in gene.discover()]
    raw = await gene.fetch(items[0])
    normalized = await gene.normalize(raw)

    assert (
        items[0].source_url == "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking"
    )
    assert items[0].extra["repo_path"] == "docs/cloud-native-platform/process-tracking.md"
    assert items[0].version == "blob-sha-123"
    assert raw.content_type == "text/markdown"
    assert "# Process Tracking" in normalized.markdown_body
    assert "Repository Path: docs/cloud-native-platform/process-tracking.md" in normalized.markdown_body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {"encoding": "base64", "size": 0},
        {"content": None, "encoding": "base64", "size": 0},
        {"content": "", "encoding": "utf-8", "size": 0},
    ],
)
async def test_github_pat_fetch_rejects_missing_or_invalid_content_contract(payload):
    gene = GitHubPagesGene(
        config={
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/page/",
        },
        source_id="src-pages",
    )
    response = HtmlResponse("")
    response.json = lambda: payload
    gene._client = AsyncMock()
    gene._client.get.return_value = response
    item = ContentItem(
        item_id="github-pages-page",
        title="Page",
        source_url="https://github-pages.example.test/pages/org/repo/page",
        last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
        content_type="text/markdown",
        version="blob-a",
        extra={"repo_api_url": "https://github.example/api/content", "repo_path": "docs/page.md"},
    )

    with pytest.raises(RuntimeError):
        await gene.fetch(item)


@pytest.mark.asyncio
async def test_github_pat_decodes_encoded_page_url_when_matching_repository_markdown(monkeypatch):
    RunbooksRepoApiClient.instances.clear()
    monkeypatch.setattr("memforge.genes.github_pages_gene._RequestsAsyncClient", RunbooksRepoApiClient)

    gene = GitHubPagesGene(
        config={
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "single_page",
            "page_url": "https://github.example.test/pages/example-org/runbooks/runbooks/Process%20Tracking/stuck-lock",
        },
        source_id="src-pages",
    )

    await gene.authenticate()
    items = [item async for item in gene.discover()]
    raw = await gene.fetch(items[0])

    assert len(items) == 1
    assert items[0].source_url == (
        "https://github.example.test/pages/example-org/runbooks/runbooks/Process%20Tracking/stuck-lock"
    )
    assert items[0].extra["repo_path"] == "docs/runbooks/Process Tracking/stuck-lock.md"
    assert raw.body.startswith(b"# Stuck Lock")


@pytest.mark.asyncio
async def test_github_pat_subtree_discovers_repository_markdown_without_crawling_pages(monkeypatch):
    RunbooksRepoApiClient.instances.clear()
    monkeypatch.setattr("memforge.genes.github_pages_gene._RequestsAsyncClient", RunbooksRepoApiClient)

    gene = GitHubPagesGene(
        config={
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "subtree",
            "root_url": "https://github.example.test/pages/example-org/runbooks/runbooks/Process%20Tracking",
        },
        source_id="src-pages",
    )

    await gene.authenticate()
    items = [item async for item in gene.discover()]

    assert [item.extra["repo_path"] for item in items] == [
        "docs/runbooks/Process Tracking/stuck-lock.md",
        "docs/runbooks/Process Tracking/timed-out-process-instance.md",
    ]
    assert [item.source_url for item in items] == [
        "https://github.example.test/pages/example-org/runbooks/runbooks/Process%20Tracking/stuck-lock",
        "https://github.example.test/pages/example-org/runbooks/runbooks/Process%20Tracking/timed-out-process-instance",
    ]
    assert all(item.version.endswith("-sha") for item in items)
    assert not any("/sitemap.xml" in url for _, url in RunbooksRepoApiClient.instances[-1].calls)


@pytest.mark.asyncio
async def test_github_pat_subtree_fails_closed_when_recursive_tree_is_truncated(monkeypatch):
    RecordingAsyncClient.instances.clear()
    monkeypatch.setattr(
        "memforge.genes.github_pages_gene._RequestsAsyncClient",
        TruncatedRepoApiClient,
    )
    gene = GitHubPagesGene(
        config={
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "subtree",
            "root_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/",
        },
        source_id="src-pages",
    )

    await gene.authenticate()
    with pytest.raises(RuntimeError, match="truncated"):
        _ = [item async for item in gene.discover()]


@pytest.mark.asyncio
async def test_github_pat_subtree_rejects_missing_tree_without_completion_evidence(monkeypatch):
    monkeypatch.setattr(
        "memforge.genes.github_pages_gene._RequestsAsyncClient",
        MissingTreeRepoApiClient,
    )
    gene = GitHubPagesGene(
        config={
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "subtree",
            "root_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/",
        },
        source_id="src-pages",
    )

    await gene.authenticate()
    with pytest.raises(RuntimeError, match="truncated=false"):
        _ = [item async for item in gene.discover()]
    assert gene.discovery_complete is False


@pytest.mark.asyncio
async def test_github_pat_subtree_rejects_duplicate_repository_paths(monkeypatch):
    class DuplicateTreeClient(RepoApiClient):
        async def get(self, url: str, **_kwargs):
            if url.endswith("/api/v3/repos/org/repo/git/trees/main?recursive=1"):
                return RepoApiResponse(
                    {
                        "truncated": False,
                        "tree": [
                            {"path": "docs/a.md", "type": "blob", "sha": "sha-a"},
                            {"path": "docs/a.md", "type": "blob", "sha": "sha-b"},
                        ],
                    },
                    url=url,
                )
            return await super().get(url, **_kwargs)

    monkeypatch.setattr("memforge.genes.github_pages_gene._RequestsAsyncClient", DuplicateTreeClient)
    gene = GitHubPagesGene(
        config={
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "subtree",
            "root_url": "https://github-pages.example.test/pages/org/repo/",
        },
        source_id="src-pages",
    )

    await gene.authenticate()
    with pytest.raises(RuntimeError, match="duplicate path"):
        _ = [item async for item in gene.discover()]
    assert gene.discovery_complete is False


@pytest.mark.asyncio
async def test_github_pat_subtree_honors_max_pages(monkeypatch):
    RunbooksRepoApiClient.instances.clear()
    monkeypatch.setattr("memforge.genes.github_pages_gene._RequestsAsyncClient", RunbooksRepoApiClient)

    gene = GitHubPagesGene(
        config={
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "subtree",
            "root_url": "https://github.example.test/pages/example-org/runbooks/runbooks/Process%20Tracking",
            "max_pages": 1,
        },
        source_id="src-pages",
    )

    await gene.authenticate()
    with pytest.raises(RuntimeError, match="max_pages=1"):
        [item async for item in gene.discover()]


@pytest.mark.asyncio
async def test_github_pat_authentication_requires_pat():
    gene = GitHubPagesGene(
        config={
            "base_url": "https://github-pages.example.test/pages/org/repo/",
            "auth_mode": "github_pat",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/",
        },
        source_id="src-pages",
    )

    with pytest.raises(ValueError, match="PAT"):
        await gene.authenticate()


@pytest.mark.asyncio
async def test_subtree_discovery_prefers_sitemap_and_filters_to_root(monkeypatch):
    RecordingAsyncClient.instances.clear()
    monkeypatch.setattr("memforge.genes.github_pages_gene._RequestsAsyncClient", SitemapClient)

    gene = GitHubPagesGene(
        config={
            "base_url": "https://github-pages.example.test/pages/org/repo/",
            "auth_mode": "none",
            "sync_mode": "subtree",
            "root_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/",
        },
        source_id="src-pages",
    )

    await gene.authenticate()
    items = [item async for item in gene.discover()]

    assert [item.title for item in items] == ["locking", "process tracking"]
    assert all("/cloud-native-platform/" in item.source_url for item in items)
    assert items[0].last_modified == datetime(2026, 5, 25, 0, 0, tzinfo=timezone.utc)
    assert items[0].version.startswith("sha256:")
    assert gene.discovery_complete is False


@pytest.mark.asyncio
async def test_subtree_bfs_never_grants_source_wide_absence_authority(monkeypatch):
    class BfsClient(RecordingAsyncClient):
        async def get(self, url: str, **_kwargs):
            self.calls.append(("GET", url))
            if url.endswith("/sitemap.xml"):
                return HtmlResponse("missing", status_code=404, url=url)
            return HtmlResponse("<main><p>Reachable page</p></main>", url=url)

    monkeypatch.setattr("memforge.genes.github_pages_gene._RequestsAsyncClient", BfsClient)
    gene = GitHubPagesGene(
        config={
            "base_url": "https://github-pages.example.test/pages/org/repo/",
            "auth_mode": "none",
            "sync_mode": "subtree",
            "root_url": "https://github-pages.example.test/pages/org/repo/",
        },
        source_id="src-pages",
    )

    await gene.authenticate()
    items = [item async for item in gene.discover()]

    assert len(items) == 1
    assert gene.discovery_complete is False


@pytest.mark.asyncio
async def test_http_fetch_rejects_body_or_final_url_drift_from_discovery(monkeypatch):
    class DriftingClient(RecordingAsyncClient):
        fetch_count = 0

        async def get(self, url: str, **_kwargs):
            self.calls.append(("GET", url))
            self.fetch_count += 1
            if self.fetch_count == 1:
                return HtmlResponse("<main>Revision B</main>", url=url)
            return HtmlResponse("<main>Revision A</main>", url=url)

    monkeypatch.setattr("memforge.genes.github_pages_gene._RequestsAsyncClient", DriftingClient)
    gene = GitHubPagesGene(
        config={
            "auth_mode": "none",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/page/",
        },
        source_id="src-pages",
    )
    await gene.authenticate()
    [item] = [item async for item in gene.discover()]

    with pytest.raises(RuntimeError, match="changed between discovery and fetch"):
        await gene.fetch(item)


@pytest.mark.asyncio
async def test_subtree_sitemap_requires_explicit_authoritative_contract(monkeypatch):
    RecordingAsyncClient.instances.clear()
    monkeypatch.setattr("memforge.genes.github_pages_gene._RequestsAsyncClient", SitemapClient)
    gene = GitHubPagesGene(
        config={
            "base_url": "https://github-pages.example.test/pages/org/repo/",
            "auth_mode": "none",
            "sync_mode": "subtree",
            "root_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/",
            "sitemap_authoritative": True,
        },
        source_id="src-pages",
    )

    await gene.authenticate()
    _ = [item async for item in gene.discover()]

    assert gene.discovery_complete is True
    assert gene.discovery_completion_reason == "github_pages_authoritative_sitemap_exhausted"


@pytest.mark.asyncio
async def test_subtree_rejects_duplicate_canonical_sitemap_urls(monkeypatch):
    class DuplicateSitemapClient(SitemapClient):
        async def get(self, url: str, **_kwargs):
            if url.endswith("/sitemap.xml"):
                page = "https://github-pages.example.test/pages/org/repo/cloud-native-platform/locking/"
                return HtmlResponse(
                    f"<urlset><url><loc>{page}</loc></url><url><loc>{page}</loc></url></urlset>",
                    url=url,
                )
            return await super().get(url, **_kwargs)

    monkeypatch.setattr("memforge.genes.github_pages_gene._RequestsAsyncClient", DuplicateSitemapClient)
    gene = GitHubPagesGene(
        config={
            "base_url": "https://github-pages.example.test/pages/org/repo/",
            "auth_mode": "none",
            "sync_mode": "subtree",
            "root_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/",
            "sitemap_authoritative": True,
        },
        source_id="src-pages",
    )

    await gene.authenticate()
    with pytest.raises(RuntimeError, match="duplicate canonical URLs"):
        _ = [item async for item in gene.discover()]
    assert gene.discovery_complete is False


@pytest.mark.asyncio
async def test_normalize_extracts_main_article_and_removes_page_chrome():
    gene = GitHubPagesGene(
        config={
            "base_url": "https://github-pages.example.test/pages/org/repo/",
            "auth_mode": "none",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/",
        },
        source_id="src-pages",
    )
    await gene.authenticate()
    gene._client = RecordingAsyncClient()
    item = await gene._content_item_for_url(
        "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/",
        metadata_headers={},
    )
    raw = RawContent(
        item=item,
        content_type="text/html",
        body=b"""
    <html>
      <body>
        <header>NextGenPayroll Documentation</header>
        <nav>Home General Cloud Native Platform</nav>
        <aside class="md-sidebar">Process Tracking</aside>
        <main>
          <article>
            <h1>Process Tracking</h1>
            <h2>Access to REST Endpoints</h2>
            <p>Use <code>/api/</code> for REST clients and <code>/ui/</code> for browser access.</p>
            <pre><code>GET /process-tracker/v1/</code></pre>
          </article>
        </main>
        <aside class="md-sidebar--secondary">Table of contents</aside>
      </body>
    </html>
    """,
    )

    normalized = await gene.normalize(raw)

    assert "# Process Tracking" in normalized.markdown_body
    assert "## Source Metadata" in normalized.markdown_body
    assert "Access to REST Endpoints" in normalized.markdown_body
    assert "`/api/`" in normalized.markdown_body
    assert "GET /process-tracker/v1/" in normalized.markdown_body
    assert "NextGenPayroll Documentation" not in normalized.markdown_body
    assert "Table of contents" not in normalized.markdown_body
    assert normalized.source_semantics["source_type"] == "github_pages"
    assert normalized.source_semantics["page_url"].endswith("/cloud-native-platform/process-tracking")


@pytest.mark.asyncio
async def test_authoritative_empty_github_page_stays_empty_after_normalization():
    gene = GitHubPagesGene(
        config={
            "base_url": "https://github-pages.example.test/pages/org/repo/",
            "auth_mode": "none",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/empty/",
        },
        source_id="src-pages",
    )
    gene._base_url = "https://github-pages.example.test/pages/org/repo"
    item = ContentItem(
        item_id="github-pages-empty",
        title="Empty page",
        source_url="https://github-pages.example.test/pages/org/repo/empty",
        last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
    )
    raw = RawContent(
        item=item,
        content_type="text/html",
        body=b"",
        authoritative_empty=True,
        empty_evidence="github_pages_http_successful_empty_response",
    )

    normalized = await gene.normalize(raw)

    assert normalized.markdown_body == ""
    assert normalized.source_semantics["source_type"] == "github_pages"
