from __future__ import annotations

import hashlib
import ssl
from collections.abc import Callable
from datetime import datetime, timezone

import httpx
import pytest

from memforge.genes import GENE_REGISTRY, github_pages_gene
from memforge.genes.base import SourceConfigurationError
from memforge.genes.github_pages_gene import GitHubPagesGene
from memforge.genes.toolkit.github_api import TREE_REQUEST_BUDGET, GitHubTreeBudgetExceeded
from memforge.models import ContentItem, RawContent
from tests.connector_http_fixture import FakeConnectorHttp
from tests.github_api_fixture import FakeGitHub, git_blob_sha
from tests.tls_bundle_fixture import BUNDLE_CA_COUNT, trusted_ca_count, write_ca_bundle

Answer = Callable[[httpx.Request], httpx.Response]

PAGE_HEADERS = {"Last-Modified": "Tue, 26 May 2026 14:51:21 GMT", "ETag": '"abc123"'}
PAGE_HTML = "<main><h1>Process Tracking</h1><p>Body</p></main>"

PROCESS_TRACKING = b"# Process Tracking\n\nUse `/api/` for REST clients."
ORG_REPO_FILES = {"docs/cloud-native-platform/process-tracking.md": PROCESS_TRACKING}
RUNBOOKS_FILES = {
    "docs/runbooks/Process Tracking/stuck-lock.md": b"# Stuck Lock\n\nUnlock the process.",
    "docs/runbooks/Process Tracking/timed-out-process-instance.md": b"# Timed Out\n",
    "docs/runbooks/Other Runbook/overview.md": b"# Overview\n",
    "docs/runbooks/Process Tracking/assets/diagram.png": b"\x89PNG\r\n",
}
RUNBOOKS_API = "https://github.example.test/api/v3/repos/example-org/runbooks"


def page(text: str = PAGE_HTML, *, status: int = 200, headers: dict[str, str] | None = None) -> Answer:
    """An answer serving one page, built afresh for each request."""
    return lambda _request: httpx.Response(status, text=text, headers=headers or {})


def not_found(_request: httpx.Request) -> httpx.Response:
    return httpx.Response(404, text="missing")


class Site:
    """The provider behind a Pages Gene: a GitHub repository's REST API, the published pages, or both.

    ``pages`` answers a URL; ``fallback`` answers any other request.
    ``requests`` lists each request as ``(method, url)``.
    """

    def __init__(
        self,
        pages: dict[str, Answer] | None = None,
        *,
        github: FakeGitHub | None = None,
        fallback: Answer = not_found,
    ) -> None:
        self.pages = pages or {}
        self.github = github
        self.fallback = fallback
        self.requests: list[tuple[str, str]] = []
        self.http = FakeConnectorHttp(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, str(request.url)))
        if self.github is not None and (answer := self.github.answer(request)) is not None:
            return answer
        return self.pages.get(str(request.url), self.fallback)(request)

    async def connect(self, config: dict) -> GitHubPagesGene:
        gene = GitHubPagesGene(config=config, source_id="src-pages")
        with self.http.serving(github_pages_gene):
            await gene.authenticate()
        return gene


def org_repo_site(api_origin: str = "https://github-pages.example.test", **github_options) -> Site:
    return Site(github=FakeGitHub(f"{api_origin}/api/v3/repos/org/repo", files=ORG_REPO_FILES, **github_options))


def runbooks_site(**github_options) -> Site:
    return Site(github=FakeGitHub(RUNBOOKS_API, files=RUNBOOKS_FILES, **github_options))


def published_site() -> Site:
    """A site whose every page answers the Process Tracking page."""
    return Site(fallback=page(headers=PAGE_HEADERS))


SITEMAP = """<?xml version="1.0" encoding="UTF-8"?>
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
</urlset>"""
SITE = "https://github-pages.example.test/pages/org/repo/"
SITEMAP_URL = f"{SITE}sitemap.xml"


def sitemap_site(sitemap: str = SITEMAP) -> Site:
    return Site({SITEMAP_URL: page(sitemap)}, fallback=page(headers=PAGE_HEADERS))


SUBTREE_PAT = {"auth_mode": "github_pat", "pat": "github-secret", "sync_mode": "subtree"}
RUNBOOKS_ROOT = "https://github.example.test/pages/example-org/runbooks/runbooks/Process%20Tracking"


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
async def test_single_page_discovery_uses_canonical_url_metadata():
    site = published_site()
    gene = await site.connect(
        {
            "auth_mode": "none",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/#tracking",
        }
    )

    items = [item async for item in gene.discover()]

    assert len(items) == 1
    item = items[0]
    assert item.item_id.startswith("github-pages-")
    assert item.title == "process tracking"
    assert item.source_url == "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking"
    assert item.last_modified == datetime(2026, 5, 26, 14, 51, 21, tzinfo=timezone.utc)
    assert item.extra["content_updated_at"] == "2026-05-26T14:51:21+00:00"
    assert item.version == "sha256:" + hashlib.sha256(PAGE_HTML.encode()).hexdigest()
    assert site.requests == [
        ("GET", "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking")
    ]


@pytest.mark.asyncio
async def test_github_pat_authentication_uses_bearer_header():
    site = published_site()
    await site.connect(
        {
            "base_url": "https://github-pages.example.test/pages/org/repo/",
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/",
        }
    )

    assert site.http.options[-1]["headers"]["Authorization"] == "Bearer github-secret"


@pytest.mark.asyncio
async def test_the_client_follows_redirects_and_trusts_requests_ca_bundle(tmp_path, monkeypatch):
    bundle = write_ca_bundle(tmp_path / "corp-ca.pem")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(bundle))
    site = published_site()

    await site.connect({"auth_mode": "none", "sync_mode": "single_page", "page_url": f"{SITE}page/"})

    options = site.http.options[-1]
    assert options["follow_redirects"] is True
    assert isinstance(options["verify"], ssl.SSLContext)
    assert trusted_ca_count(options["verify"]) == BUNDLE_CA_COUNT


@pytest.mark.asyncio
async def test_a_redirect_to_the_github_login_page_fails_instead_of_syncing_it():
    login = "https://github-pages.example.test/login"
    site = Site(
        {f"{SITE}page": lambda _request: httpx.Response(302, headers={"Location": login})},
        fallback=page("<form>Sign in</form>"),
    )
    gene = await site.connect({"auth_mode": "none", "sync_mode": "single_page", "page_url": f"{SITE}page/"})

    with pytest.raises(RuntimeError, match="returned the login page"):
        _ = [item async for item in gene.discover()]


@pytest.mark.asyncio
async def test_github_pat_discovers_and_fetches_repository_markdown_for_page_url():
    site = org_repo_site()
    gene = await site.connect(
        {
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/",
        }
    )

    items = [item async for item in gene.discover()]
    raw = await gene.fetch(items[0])
    normalized = await gene.normalize(raw)

    assert (
        items[0].source_url == "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking"
    )
    assert items[0].extra["repo_path"] == "docs/cloud-native-platform/process-tracking.md"
    assert items[0].version == git_blob_sha(PROCESS_TRACKING)
    assert raw.content_type == "text/markdown"
    assert "# Process Tracking" in normalized.markdown_body
    assert "Repository Path: docs/cloud-native-platform/process-tracking.md" in normalized.markdown_body
    assert normalized.source_semantics["repo_identifier"] == "github-pages.example.test/org/repo"


@pytest.mark.asyncio
async def test_github_pat_supports_enterprise_pages_subdomain_isolation():
    site = org_repo_site("https://github.example.test")
    gene = await site.connect(
        {
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "single_page",
            "page_url": "https://pages.github.example.test/org/repo/cloud-native-platform/process-tracking/",
        }
    )

    items = [item async for item in gene.discover()]
    normalized = await gene.normalize(await gene.fetch(items[0]))

    assert items[0].source_url == (
        "https://pages.github.example.test/org/repo/cloud-native-platform/process-tracking"
    )
    assert items[0].extra["repo_api_url"].startswith(
        "https://github.example.test/api/v3/repos/org/repo/"
    )
    assert normalized.source_semantics["site_url"] == "https://pages.github.example.test/org/repo"
    assert normalized.source_semantics["repo_identifier"] == "github.example.test/org/repo"


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
    site = Site(fallback=lambda _request: httpx.Response(200, json=payload))
    gene = await site.connect(
        {
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/page/",
        }
    )
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
async def test_github_pat_decodes_encoded_page_url_when_matching_repository_markdown():
    site = runbooks_site()
    gene = await site.connect(
        {
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "single_page",
            "page_url": "https://github.example.test/pages/example-org/runbooks/runbooks/Process%20Tracking/stuck-lock",
        }
    )

    items = [item async for item in gene.discover()]
    raw = await gene.fetch(items[0])

    assert len(items) == 1
    assert items[0].source_url == (
        "https://github.example.test/pages/example-org/runbooks/runbooks/Process%20Tracking/stuck-lock"
    )
    assert items[0].extra["repo_path"] == "docs/runbooks/Process Tracking/stuck-lock.md"
    assert raw.body.startswith(b"# Stuck Lock")


@pytest.mark.asyncio
async def test_github_pat_subtree_discovers_repository_markdown_without_crawling_pages():
    site = runbooks_site()
    gene = await site.connect({**SUBTREE_PAT, "root_url": RUNBOOKS_ROOT})

    items = [item async for item in gene.discover()]

    assert [item.extra["repo_path"] for item in items] == [
        "docs/runbooks/Process Tracking/stuck-lock.md",
        "docs/runbooks/Process Tracking/timed-out-process-instance.md",
    ]
    assert [item.source_url for item in items] == [
        f"{RUNBOOKS_ROOT}/stuck-lock",
        f"{RUNBOOKS_ROOT}/timed-out-process-instance",
    ]
    assert [item.version for item in items] == [git_blob_sha(RUNBOOKS_FILES[item.extra["repo_path"]]) for item in items]
    assert not any("/sitemap.xml" in url for _, url in site.requests)


@pytest.mark.asyncio
async def test_github_pat_subtree_preserves_enterprise_pages_subdomain_urls():
    site = runbooks_site()
    gene = await site.connect(
        {**SUBTREE_PAT, "root_url": "https://pages.github.example.test/example-org/runbooks/runbooks/Process%20Tracking"}
    )

    items = [item async for item in gene.discover()]

    assert [item.source_url for item in items] == [
        "https://pages.github.example.test/example-org/runbooks/runbooks/Process%20Tracking/stuck-lock",
        "https://pages.github.example.test/example-org/runbooks/runbooks/Process%20Tracking/timed-out-process-instance",
    ]
    assert any(url.startswith(f"{RUNBOOKS_API}/") for _, url in site.requests)


@pytest.mark.asyncio
async def test_github_pat_subtree_reads_a_truncated_branch_tree_by_its_subtrees():
    # GitHub truncates the root and every folder down to the runbooks, so the walk descends level by level.
    site = runbooks_site(truncated={"", "docs", "docs/runbooks"})
    gene = await site.connect({**SUBTREE_PAT, "root_url": RUNBOOKS_ROOT})

    items = [item async for item in gene.discover()]

    assert [item.extra["repo_path"] for item in items] == [
        "docs/runbooks/Process Tracking/stuck-lock.md",
        "docs/runbooks/Process Tracking/timed-out-process-instance.md",
    ]
    assert ("GET", f"{RUNBOOKS_API}/git/trees/{site.github.tree_sha('docs/runbooks')}") in site.requests


@pytest.mark.asyncio
async def test_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pend():
    files = {**RUNBOOKS_FILES, **{f"vendor-{index:04}/README.md": b"# Vendored\n" for index in range(TREE_REQUEST_BUDGET)}}
    site = Site(github=FakeGitHub(RUNBOOKS_API, files=files, truncated={""}))
    gene = await site.connect({**SUBTREE_PAT, "root_url": RUNBOOKS_ROOT})

    with pytest.raises(GitHubTreeBudgetExceeded, match=f"more than {TREE_REQUEST_BUDGET} requests"):
        _ = [item async for item in gene.discover()]

    assert sum("/git/trees/" in url for _, url in site.requests) == TREE_REQUEST_BUDGET


@pytest.mark.asyncio
async def test_github_pat_reads_a_configured_branch_named_with_a_slash_as_one_path_segment():
    site = org_repo_site(default_branch="release/2026")
    gene = await site.connect(
        {
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/",
            "branch": "release/2026",
        }
    )

    [item] = [item async for item in gene.discover()]

    assert item.extra["repo_branch"] == "release/2026"
    assert any(url.endswith("/git/trees/release%2F2026?recursive=1") for _, url in site.requests)


@pytest.mark.asyncio
async def test_configured_pages_of_one_repository_read_its_branch_tree_once_per_connection():
    site = runbooks_site(truncated={""})
    root = "https://github.example.test/pages/example-org/runbooks/runbooks"
    gene = await site.connect(
        {
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "explicit_list",
            "pages": [
                f"{root}/Process%20Tracking/stuck-lock",
                f"{root}/Process%20Tracking/timed-out-process-instance",
                f"{root}/Other%20Runbook/overview",
            ],
        }
    )

    def reads() -> tuple[int, int]:
        branch_reads = sum(url == RUNBOOKS_API for _, url in site.requests)
        root_tree_reads = sum(url.endswith("/git/trees/main?recursive=1") for _, url in site.requests)
        return branch_reads, root_tree_reads

    listing = await gene.list_scope()
    discovered = {item.item_id async for item in gene.discover()}

    assert discovered == listing.doc_ids and len(discovered) == 3
    assert reads() == (1, 1)

    # A new connection reads the branch afresh.
    with site.http.serving(github_pages_gene):
        await gene.authenticate()
    _ = [item async for item in gene.discover()]
    assert reads() == (2, 2)


@pytest.mark.asyncio
async def test_github_pat_subtree_rejects_missing_tree_without_completion_evidence():
    site = org_repo_site(answers={"/git/trees/main?recursive=1": {}})
    gene = await site.connect({**SUBTREE_PAT, "root_url": f"{SITE}cloud-native-platform/"})

    with pytest.raises(RuntimeError, match="truncated=false"):
        _ = [item async for item in gene.discover()]
    with pytest.raises(RuntimeError, match="truncated=false"):
        await gene.list_scope()


@pytest.mark.asyncio
async def test_github_pat_subtree_rejects_duplicate_repository_paths():
    duplicate = {
        "truncated": False,
        "tree": [
            {"path": "docs/a.md", "type": "blob", "sha": "sha-a"},
            {"path": "docs/a.md", "type": "blob", "sha": "sha-b"},
        ],
    }
    site = org_repo_site(answers={"/git/trees/main?recursive=1": duplicate})
    gene = await site.connect({**SUBTREE_PAT, "root_url": SITE})

    with pytest.raises(RuntimeError, match="duplicate path"):
        _ = [item async for item in gene.discover()]
    with pytest.raises(RuntimeError, match="duplicate path"):
        await gene.list_scope()


@pytest.mark.asyncio
async def test_github_pat_subtree_honors_max_pages():
    site = runbooks_site()
    gene = await site.connect({**SUBTREE_PAT, "root_url": RUNBOOKS_ROOT, "max_pages": 1})

    with pytest.raises(
        SourceConfigurationError,
        match=r"found 2 pages.*Max Pages \(1\).*Increase Max Pages.*narrow",
    ):
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


SUBTREE_NONE = {"base_url": SITE, "auth_mode": "none", "sync_mode": "subtree"}


@pytest.mark.asyncio
async def test_subtree_discovery_prefers_sitemap_and_filters_to_root():
    site = sitemap_site()
    gene = await site.connect({**SUBTREE_NONE, "root_url": f"{SITE}cloud-native-platform/"})

    items = [item async for item in gene.discover()]

    assert [item.title for item in items] == ["locking", "process tracking"]
    assert all("/cloud-native-platform/" in item.source_url for item in items)
    assert items[0].last_modified == datetime(2026, 5, 25, 0, 0, tzinfo=timezone.utc)
    assert items[0].extra["content_updated_at"] == "2026-05-25T00:00:00+00:00"
    assert items[0].version.startswith("sha256:")
    assert GitHubPagesGene.lists_scope(gene.config) is False


@pytest.mark.asyncio
async def test_subtree_bfs_never_grants_source_wide_absence_authority():
    site = Site({SITEMAP_URL: not_found}, fallback=page("<main><p>Reachable page</p></main>"))
    gene = await site.connect({**SUBTREE_NONE, "root_url": SITE})

    items = [item async for item in gene.discover()]

    assert len(items) == 1
    assert GitHubPagesGene.lists_scope(gene.config) is False


# Latin-1 pages: read as UTF-8, "Caf\u00e9" would become "Caf\ufffd".
LATIN1_PAGE = (
    '<html><head><meta charset="iso-8859-1"></head>'
    f'<body><p>Caf\u00e9</p><a href="{SITE}guide/">Guide</a><a href="{SITE}caf\u00e9/">Caf\u00e9</a></body></html>'
)
LATIN1_SITEMAP = (
    '<?xml version="1.0" encoding="ISO-8859-1"?>'
    '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"><!-- Caf\u00e9 -->'
    f"<url><loc>{SITE}guide/</loc><lastmod>2026-05-25T00:00:00Z</lastmod></url>"
    f"<url><loc>{SITE}caf\u00e9/</loc></url></urlset>"
)


def latin1(text: str, content_type: str) -> Answer:
    return lambda _request: httpx.Response(200, content=text.encode("latin-1"), headers={"content-type": content_type})


def test_links_and_sitemap_locations_are_read_in_the_encoding_the_page_declares():
    assert github_pages_gene._links_from_html(LATIN1_PAGE.encode("latin-1"), base_url=SITE) == [
        f"{SITE}guide/",
        f"{SITE}caf\u00e9/",
    ]
    # A charset the response declares comes first.
    utf8_page = '<html><body><a href="caf\u00e9/">x</a></body></html>'.encode()
    assert github_pages_gene._links_from_html(utf8_page, base_url=SITE, encoding="utf-8") == [f"{SITE}caf\u00e9/"]
    assert github_pages_gene._urls_from_sitemap(LATIN1_SITEMAP.encode("latin-1")) == (
        "urlset",
        [f"{SITE}guide", f"{SITE}caf\u00e9"],
    )


@pytest.mark.asyncio
async def test_the_link_crawl_reads_a_page_that_is_not_utf8():
    site = Site(
        {SITEMAP_URL: not_found, SITE.rstrip("/"): latin1(LATIN1_PAGE, "text/html"), f"{SITE}caf%C3%A9": not_found},
        fallback=page(),
    )
    gene = await site.connect({**SUBTREE_NONE, "root_url": SITE, "max_depth": 1})

    # The café page is gone; what matters is that its link was read by its own name and followed.
    with pytest.raises(httpx.HTTPStatusError):
        _ = [item async for item in gene.discover()]

    assert ("GET", f"{SITE}guide") in site.requests
    assert ("GET", f"{SITE}caf%C3%A9") in site.requests


@pytest.mark.asyncio
async def test_an_authoritative_sitemap_that_is_not_utf8_lists_its_pages():
    sitemap = LATIN1_SITEMAP.replace(f"<url><loc>{SITE}caf\u00e9/</loc></url>", "")
    site = Site({SITEMAP_URL: latin1(sitemap, "application/xml")}, fallback=page())
    gene = await site.connect({**SUBTREE_NONE, "root_url": SITE, "sitemap_authoritative": True})

    items = [item async for item in gene.discover()]

    assert [item.source_url for item in items] == [f"{SITE}guide"]
    assert items[0].extra["content_updated_at"] == "2026-05-25T00:00:00+00:00"


@pytest.mark.asyncio
async def test_http_fetch_rejects_body_or_final_url_drift_from_discovery():
    revisions = iter(["<main>Revision B</main>", "<main>Revision A</main>"])
    site = Site(fallback=lambda _request: httpx.Response(200, text=next(revisions)))
    gene = await site.connect({"auth_mode": "none", "sync_mode": "single_page", "page_url": f"{SITE}page/"})
    [item] = [item async for item in gene.discover()]

    with pytest.raises(RuntimeError, match="changed between discovery and fetch"):
        await gene.fetch(item)


@pytest.mark.asyncio
async def test_subtree_sitemap_requires_explicit_authoritative_contract():
    site = sitemap_site()
    gene = await site.connect(
        {**SUBTREE_NONE, "root_url": f"{SITE}cloud-native-platform/", "sitemap_authoritative": True}
    )

    items = [item async for item in gene.discover()]

    assert GitHubPagesGene.lists_scope(gene.config) is True
    listing = await gene.list_scope()
    assert listing.kind.value == "existence"
    assert listing.doc_ids == {item.item_id for item in items}


@pytest.mark.asyncio
async def test_subtree_rejects_duplicate_canonical_sitemap_urls():
    duplicate = f"{SITE}cloud-native-platform/locking/"
    site = sitemap_site(f"<urlset><url><loc>{duplicate}</loc></url><url><loc>{duplicate}</loc></url></urlset>")
    gene = await site.connect(
        {**SUBTREE_NONE, "root_url": f"{SITE}cloud-native-platform/", "sitemap_authoritative": True}
    )

    with pytest.raises(RuntimeError, match="duplicate canonical URLs"):
        _ = [item async for item in gene.discover()]
    with pytest.raises(RuntimeError, match="duplicate canonical URLs"):
        await gene.list_scope()


@pytest.mark.asyncio
async def test_a_closed_gene_does_not_reconnect_on_its_own():
    site = published_site()
    gene = await site.connect({"auth_mode": "none", "sync_mode": "single_page", "page_url": f"{SITE}page/"})
    await gene.close()

    with site.http.serving(github_pages_gene):
        health = await gene.health_check()
        with pytest.raises(RuntimeError, match="call authenticate"):
            _ = [item async for item in gene.discover()]

    assert health["healthy"] is False
    assert "call authenticate" in health["error"]
    assert len(site.http.clients) == 1
    assert site.requests == []


@pytest.mark.asyncio
async def test_the_health_check_asks_the_site_with_head():
    site = published_site()
    gene = await site.connect({"auth_mode": "none", "sync_mode": "single_page", "page_url": f"{SITE}page/"})

    assert await gene.health_check() == {"healthy": True, "status_code": 200}
    assert site.requests == [("HEAD", SITE.rstrip("/"))]


@pytest.mark.asyncio
async def test_normalize_extracts_main_article_and_removes_page_chrome():
    gene = await published_site().connect(
        {
            "base_url": "https://github-pages.example.test/pages/org/repo/",
            "auth_mode": "none",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/",
        }
    )
    item = await gene._content_item_for_url(
        "https://github-pages.example.test/pages/org/repo/cloud-native-platform/process-tracking/",
    )
    raw = RawContent(
        item=item,
        content_type="text/html",
        body=b"""
    <html>
      <body>
        <header>AcmePayroll Documentation</header>
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
    assert "AcmePayroll Documentation" not in normalized.markdown_body
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
    assert "repo_identifier" not in normalized.source_semantics


@pytest.mark.asyncio
async def test_authoritative_empty_repository_backed_page_preserves_repo_identifier():
    gene = GitHubPagesGene(
        config={
            "base_url": "https://github-pages.example.test/pages/org/repo/",
            "auth_mode": "github_pat",
            "pat": "github-secret",
            "sync_mode": "single_page",
            "page_url": "https://github-pages.example.test/pages/org/repo/empty/",
        },
        source_id="src-pages",
    )
    gene._base_url = "https://github-pages.example.test/pages/org/repo"
    item = ContentItem(
        item_id="github-pages-empty-repo",
        title="Empty repository page",
        source_url="https://github-pages.example.test/pages/org/repo/empty",
        last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
        extra={
            "repo_api_url": "https://github-pages.example.test/api/v3/repos/org/repo/contents/docs/empty.md?ref=main",
            "repo_path": "docs/empty.md",
        },
    )
    raw = RawContent(
        item=item,
        content_type="text/markdown",
        body=b"",
        authoritative_empty=True,
        empty_evidence="github_contents_api_successful_empty_blob",
    )

    normalized = await gene.normalize(raw)

    assert normalized.markdown_body == ""
    assert normalized.source_semantics["repo_identifier"] == "github-pages.example.test/org/repo"


@pytest.mark.asyncio
async def test_incremental_github_pat_subtree_lists_every_page_the_branch_holds():
    config = {**SUBTREE_PAT, "root_url": RUNBOOKS_ROOT}
    full = await runbooks_site().connect(dict(config))
    every_page = {item.item_id async for item in full.discover()}
    gene = await runbooks_site().connect(dict(config))

    changed = [item async for item in gene.discover(since=datetime(2100, 1, 1, tzinfo=timezone.utc))]
    listing = await gene.list_scope()

    assert changed == []
    assert listing.kind.value == "existence"
    assert listing.doc_ids == every_page
    assert len(every_page) == 2


@pytest.mark.parametrize(
    ("config", "lists"),
    [
        ({"sync_mode": "single_page"}, True),
        ({"sync_mode": "explicit_list", "auth_mode": "none"}, True),
        ({"sync_mode": "subtree"}, True),
        ({"sync_mode": "subtree", "auth_mode": "github_pat"}, True),
        ({"sync_mode": "subtree", "auth_mode": "none"}, False),
        ({"sync_mode": "subtree", "auth_mode": "none", "sitemap_authoritative": "false"}, False),
        ({"sync_mode": "subtree", "auth_mode": "none", "sitemap_authoritative": True}, True),
    ],
    ids=[
        "single-page",
        "explicit-list",
        "subtree-default-auth",
        "subtree-branch-tree",
        "subtree-crawl-or-sitemap",
        "subtree-sitemap-not-authoritative",
        "subtree-authoritative-sitemap",
    ],
)
def test_configured_pages_and_a_subtree_read_from_a_complete_inventory_list_their_scope(config, lists):
    assert GitHubPagesGene.lists_scope(config) is lists


def _failing_sitemap(failure: str) -> Answer:
    def answer(request: httpx.Request) -> httpx.Response:
        if failure == "missing":
            return httpx.Response(404, text="missing")
        if failure == "transport":
            raise httpx.ConnectError("connection reset", request=request)
        return httpx.Response(
            200,
            text=(
                '<?xml version="1.0"?><sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                f"<sitemap><loc>{SITE}sitemap-1.xml</loc></sitemap></sitemapindex>"
            ),
        )

    return answer


def _no_crawl(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"the link crawl read {request.url}")


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "transport", "not-a-urlset"])
@pytest.mark.parametrize("step", ["list_scope", "discover"])
async def test_an_authoritative_sitemap_that_cannot_be_read_fails_and_never_falls_back_to_the_crawl(failure, step):
    site = Site({SITEMAP_URL: _failing_sitemap(failure)}, fallback=_no_crawl)
    gene = await site.connect({**SUBTREE_NONE, "root_url": SITE, "sitemap_authoritative": True})

    with pytest.raises(Exception) as raised:
        if step == "list_scope":
            await gene.list_scope()
        else:
            _ = [item async for item in gene.discover()]

    assert not isinstance(raised.value, AssertionError)
    # Only the sitemap is read; a refused connection is tried again before the run fails.
    assert {url for _, url in site.requests} == {SITEMAP_URL}
    assert len(site.requests) == (4 if failure == "transport" else 1)


def _configured_pages(request: httpx.Request) -> httpx.Response:
    """Serves every configured page except ``missing``."""
    if str(request.url).rstrip("/").endswith("missing"):
        return httpx.Response(404, text="missing")
    return httpx.Response(200, text="<html><body><h1>Page</h1></body></html>")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "config",
    [
        {"sync_mode": "single_page", "page_url": f"{SITE}a/"},
        {"sync_mode": "explicit_list", "pages": [f"{SITE}a/", f"{SITE}b/"]},
    ],
    ids=["single-page", "explicit-list"],
)
async def test_configured_pages_are_an_existence_listing_with_or_without_discovery(config):
    config = {"base_url": SITE, "auth_mode": "none", **config}
    discovered = await Site(fallback=_configured_pages).connect(dict(config))
    every_page = {item.item_id async for item in discovered.discover()}
    after_discovery = await discovered.list_scope()
    standalone = await Site(fallback=_configured_pages).connect(dict(config))

    assert after_discovery.kind.value == "existence"
    assert after_discovery.doc_ids == every_page
    assert await standalone.list_scope() == after_discovery


@pytest.mark.asyncio
async def test_a_configured_page_that_cannot_be_read_fails_the_listing():
    gene = await Site(fallback=_configured_pages).connect(
        {"base_url": SITE, "auth_mode": "none", "sync_mode": "explicit_list", "pages": [f"{SITE}a/", f"{SITE}missing/"]}
    )

    with pytest.raises(httpx.HTTPStatusError):
        await gene.list_scope()
