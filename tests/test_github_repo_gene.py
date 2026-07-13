from __future__ import annotations

import base64
from datetime import datetime, timezone

import pytest

from memforge.genes import GENE_REGISTRY
from memforge.genes.github_repo_gene import GitHubRepoGene
from memforge.github_repo_utils import build_github_repo_doc_id


class GithubResponse:
    def __init__(self, payload, *, status_code: int = 200, url: str = "https://github.example.test/api/v3") -> None:
        self._payload = payload
        self.status_code = status_code
        self.url = url
        self.headers = {}
        self.content = b""
        self.text = ""

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"request failed: {self.status_code}")


class RepoApiClient:
    instances: list["RepoApiClient"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.calls: list[tuple[str, str]] = []
        RepoApiClient.instances.append(self)

    async def get(self, url: str):
        self.calls.append(("GET", url))
        if url.endswith("/api/v3/repos/payroll/architecture"):
            return GithubResponse({"default_branch": "main"}, url=url)
        if url.endswith("/api/v3/repos/payroll/architecture/git/trees/main?recursive=1"):
            return GithubResponse(
                {
                    "tree": [
                        {
                            "path": "Payroll Processing/README.md",
                            "type": "blob",
                            "sha": "readme-sha",
                            "size": 42,
                        },
                        {
                            "path": "Payroll Processing/images/diagram.png",
                            "type": "blob",
                            "sha": "image-sha",
                            "size": 2048,
                        },
                        {
                            "path": "Flexible Payroll/README.md",
                            "type": "blob",
                            "sha": "flex-sha",
                            "size": 50,
                        },
                        {
                            "path": "Payroll Processing V2/Main Algorithm.md",
                            "type": "blob",
                            "sha": "v2-sha",
                            "size": 55,
                        },
                    ]
                },
                url=url,
            )
        if url.endswith("/api/v3/repos/payroll/architecture/contents/Payroll%20Processing/README.md?ref=main"):
            encoded = base64.b64encode(b"# Payroll Processing\n\nCreate tasks carefully.").decode()
            return GithubResponse({"content": encoded}, url=url)
        raise AssertionError(f"unexpected URL: {url}")

    async def aclose(self) -> None:
        pass


def test_github_repo_gene_is_registered_and_schema_is_repo_oriented():
    assert GENE_REGISTRY["github_repo"] is GitHubRepoGene

    meta = GitHubRepoGene.metadata()
    assert meta.name == "github_repo"
    assert meta.display_name == "GitHub Repository"
    assert meta.auth_method == "github_repo"
    assert meta.data_shape == "document"

    fields = {field.key: field for field in GitHubRepoGene.config_schema().fields}
    assert fields["connection_mode"].options == ["cloud_pull", "local_push"]
    assert fields["repo_url"].required is True
    assert fields["ref"].default == "main"
    assert fields["include_paths"].required is False
    assert fields["exclude_paths"].required is False
    assert "repo_path" not in fields
    assert fields["include_extensions"].default == "md, markdown, txt, adoc, rst"
    assert fields["max_files"].default == "500"
    assert fields["pat"].required is False


@pytest.mark.asyncio
async def test_cloud_pull_discovers_scoped_markdown_and_fetches_content(monkeypatch):
    RepoApiClient.instances.clear()
    monkeypatch.setattr("memforge.genes.github_repo_gene._RequestsAsyncClient", RepoApiClient)

    gene = GitHubRepoGene(
        config={
            "connection_mode": "cloud_pull",
            "repo_url": "https://github.example.test/payroll/architecture",
            "ref": "main",
            "include_paths": [],
            "exclude_paths": ["Flexible Payroll", "Payroll Processing V2"],
            "include_extensions": ["md"],
            "max_files": 10,
        },
        source_id="src-github-repo",
    )

    await gene.authenticate()
    items = [item async for item in gene.discover()]

    assert [item.extra["relative_path"] for item in items] == ["Payroll Processing/README.md"]
    item = items[0]
    assert item.item_id == build_github_repo_doc_id(
        source_id="src-github-repo",
        repo_url="https://github.example.test/payroll/architecture",
        repo_ref="main",
        relative_path="Payroll Processing/README.md",
    )
    assert item.title == "README"
    assert item.source_url == (
        "https://github.example.test/payroll/architecture/blob/main/Payroll%20Processing/README.md"
    )
    assert item.version == "readme-sha"
    assert item.space_or_project == "payroll/architecture"
    assert item.extra["repo_host"] == "github.example.test"
    assert item.extra["repo_owner"] == "payroll"
    assert item.extra["repo_name"] == "architecture"
    assert item.extra["repo_ref"] == "main"
    assert item.extra["blob_sha"] == "readme-sha"

    raw = await gene.fetch(item)
    normalized = await gene.normalize(raw)

    assert raw.content_type == "text/markdown"
    assert normalized.markdown_body.startswith("# Payroll Processing")
    assert normalized.source_semantics == {
        "source_type": "github_repo",
        "connection_mode": "cloud_pull",
        "repo_url": "https://github.example.test/payroll/architecture",
        "repo_host": "github.example.test",
        "repo_owner": "payroll",
        "repo_name": "architecture",
        "repo_ref": "main",
        "relative_path": "Payroll Processing/README.md",
        "blob_sha": "readme-sha",
        "content_type": "text/markdown",
        "canonical_url": "https://github.example.test/payroll/architecture/blob/main/Payroll%20Processing/README.md",
    }


def test_normalize_config_canonicalizes_repository_scope() -> None:
    config = {
        "connection_mode": "local_push",
        "repo_url": "https://github.example.test/payroll/architecture",
        "include_paths": ["docs/current/guide.md", "docs"],
        "exclude_paths": ["docs/archive/old.md", "docs/archive"],
    }

    GitHubRepoGene.normalize_config(config)

    assert config["include_paths"] == ["docs"]
    assert config["exclude_paths"] == ["docs/archive"]


@pytest.mark.asyncio
async def test_cloud_pull_rejects_truncated_git_tree(monkeypatch):
    class TruncatedTreeClient(RepoApiClient):
        async def get(self, url: str):
            self.calls.append(("GET", url))
            if url.endswith("/api/v3/repos/payroll/architecture/git/trees/main?recursive=1"):
                return GithubResponse({"tree": [], "truncated": True}, url=url)
            raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("memforge.genes.github_repo_gene._RequestsAsyncClient", TruncatedTreeClient)
    gene = GitHubRepoGene(
        config={
            "connection_mode": "cloud_pull",
            "repo_url": "https://github.example.test/payroll/architecture",
            "ref": "main",
        },
        source_id="src-github-repo",
    )

    await gene.authenticate()
    with pytest.raises(RuntimeError, match="truncated"):
        [item async for item in gene.discover()]


@pytest.mark.asyncio
async def test_cloud_pull_rejects_contents_api_non_base64_payload(monkeypatch):
    class LargeFileClient(RepoApiClient):
        async def get(self, url: str):
            self.calls.append(("GET", url))
            if url.endswith("/api/v3/repos/payroll/architecture/git/trees/main?recursive=1"):
                return GithubResponse(
                    {
                        "tree": [
                            {
                                "path": "Payroll Processing/Large.md",
                                "type": "blob",
                                "sha": "large-sha",
                                "size": 2_000_000,
                            }
                        ]
                    },
                    url=url,
                )
            if url.endswith("/api/v3/repos/payroll/architecture/contents/Payroll%20Processing/Large.md?ref=main"):
                return GithubResponse({"encoding": "none", "content": "", "size": 2_000_000}, url=url)
            raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("memforge.genes.github_repo_gene._RequestsAsyncClient", LargeFileClient)
    gene = GitHubRepoGene(
        config={
            "connection_mode": "cloud_pull",
            "repo_url": "https://github.example.test/payroll/architecture",
            "ref": "main",
            "include_paths": ["Payroll Processing/"],
            "include_extensions": ["md"],
        },
        source_id="src-github-repo",
    )

    await gene.authenticate()
    items = [item async for item in gene.discover()]
    with pytest.raises(RuntimeError, match="base64"):
        await gene.fetch(items[0])


@pytest.mark.asyncio
async def test_cloud_pull_rejects_malformed_base64_payload(monkeypatch):
    class MalformedContentClient(RepoApiClient):
        async def get(self, url: str):
            self.calls.append(("GET", url))
            if url.endswith("/api/v3/repos/payroll/architecture/git/trees/main?recursive=1"):
                return GithubResponse(
                    {
                        "tree": [
                            {
                                "path": "Payroll Processing/Broken.md",
                                "type": "blob",
                                "sha": "broken-sha",
                                "size": 10,
                            }
                        ]
                    },
                    url=url,
                )
            if url.endswith("/api/v3/repos/payroll/architecture/contents/Payroll%20Processing/Broken.md?ref=main"):
                return GithubResponse({"encoding": "base64", "content": "!!!!", "size": 10}, url=url)
            raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr("memforge.genes.github_repo_gene._RequestsAsyncClient", MalformedContentClient)
    gene = GitHubRepoGene(
        config={
            "connection_mode": "cloud_pull",
            "repo_url": "https://github.example.test/payroll/architecture",
            "ref": "main",
            "include_paths": ["Payroll Processing/"],
            "include_extensions": ["md"],
        },
        source_id="src-github-repo",
    )

    await gene.authenticate()
    items = [item async for item in gene.discover()]
    with pytest.raises(RuntimeError, match="base64"):
        await gene.fetch(items[0])


@pytest.mark.asyncio
async def test_cloud_pull_uses_optional_pat_as_bearer_header(monkeypatch):
    RepoApiClient.instances.clear()
    monkeypatch.setattr("memforge.genes.github_repo_gene._RequestsAsyncClient", RepoApiClient)

    gene = GitHubRepoGene(
        config={
            "connection_mode": "cloud_pull",
            "repo_url": "https://github.com/shno-labs/mem-forge",
            "pat": "secret-token",
        },
        source_id="src-github-repo",
    )

    await gene.authenticate()

    headers = RepoApiClient.instances[-1].kwargs["headers"]
    assert headers["Authorization"] == "Bearer secret-token"


@pytest.mark.asyncio
async def test_local_push_discovers_and_normalizes_pushed_package(tmp_path):
    package = tmp_path / "github-repo-doc.json"
    encoded_time = datetime(2026, 7, 7, 9, 30, tzinfo=timezone.utc).isoformat()
    package.write_text(
        """
{
  "package_kind": "github_repo_document",
  "doc_id": "github-repo-src-doc",
  "title": "Architecture README",
  "source_url": "https://github.example.test/payroll/architecture/blob/main/README.md",
  "last_modified": "%s",
  "space_or_project": "payroll/architecture",
  "version": "blob-sha-1",
  "repo_url": "https://github.example.test/payroll/architecture",
  "repo_host": "github.example.test",
  "repo_owner": "payroll",
  "repo_name": "architecture",
  "repo_ref": "main",
  "relative_path": "README.md",
  "blob_sha": "blob-sha-1",
  "content_type": "text/markdown",
  "raw_hash": "raw-hash",
  "submitted_at": "%s",
  "submitted_by": "cli",
  "markdown": "# Architecture\\n\\nMemory source design."
}
""".strip()
        % (encoded_time, encoded_time),
        encoding="utf-8",
    )

    gene = GitHubRepoGene(
        config={
            "connection_mode": "local_push",
            "repo_url": "https://github.example.test/payroll/architecture",
            "documents_dir": str(tmp_path),
        },
        source_id="src-github-repo",
    )

    await gene.authenticate()
    items = [item async for item in gene.discover()]

    assert len(items) == 1
    assert items[0].title == "Architecture README"
    assert items[0].extra["relative_path"] == "README.md"

    normalized = await gene.normalize(await gene.fetch(items[0]))

    assert normalized.markdown_body == "# Architecture\n\nMemory source design."
    assert normalized.source_semantics["source_type"] == "github_repo"
    assert normalized.source_semantics["connection_mode"] == "local_push"
    assert normalized.source_semantics["relative_path"] == "README.md"


@pytest.mark.asyncio
async def test_local_push_explicit_empty_manifest_does_not_fall_back_to_inbox(tmp_path):
    (tmp_path / "stale.json").write_text(
        '{"package_kind":"github_repo_document","doc_id":"stale"}',
        encoding="utf-8",
    )
    gene = GitHubRepoGene(
        config={
            "connection_mode": "local_push",
            "repo_url": "https://github.example.test/payroll/architecture",
            "documents_dir": str(tmp_path),
            "local_agent_package_manifest": [],
        },
        source_id="src-github-repo",
    )

    await gene.authenticate()
    items = [item async for item in gene.discover()]

    assert items == []


@pytest.mark.asyncio
async def test_local_push_discovery_filters_packages_by_current_scope(tmp_path):
    encoded_time = datetime(2026, 7, 7, 9, 30, tzinfo=timezone.utc).isoformat()
    for name, relative_path, repo_ref in [
        ("keep.json", "Payroll Processing/README.md", "main"),
        ("wrong-path.json", "Flexible Payroll/README.md", "main"),
        ("wrong-ref.json", "Payroll Processing/Old.md", "feature"),
        ("wrong-extension.json", "Payroll Processing/diagram.png", "main"),
    ]:
        (tmp_path / name).write_text(
            """
{
  "package_kind": "github_repo_document",
  "doc_id": "%s",
  "title": "%s",
  "source_url": "https://github.example.test/payroll/architecture/blob/%s/%s",
  "last_modified": "%s",
  "space_or_project": "payroll/architecture",
  "version": "blob-sha-1",
  "repo_url": "https://github.example.test/payroll/architecture",
  "repo_host": "github.example.test",
  "repo_owner": "payroll",
  "repo_name": "architecture",
  "repo_ref": "%s",
  "relative_path": "%s",
  "blob_sha": "blob-sha-1",
  "content_type": "text/markdown",
  "submitted_at": "%s",
  "submitted_by": "cli",
  "markdown": "# Doc"
}
""".strip()
            % (name, relative_path, repo_ref, relative_path, encoded_time, repo_ref, relative_path, encoded_time),
            encoding="utf-8",
        )

    gene = GitHubRepoGene(
        config={
            "connection_mode": "local_push",
            "repo_url": "https://github.example.test/payroll/architecture",
            "ref": "main",
            "include_paths": ["Payroll Processing/"],
            "include_extensions": ["md"],
            "documents_dir": str(tmp_path),
        },
        source_id="src-github-repo",
    )

    await gene.authenticate()
    items = [item async for item in gene.discover()]

    assert [item.extra["relative_path"] for item in items] == ["Payroll Processing/README.md"]


@pytest.mark.asyncio
async def test_local_push_discovery_enforces_max_files_on_current_scope(tmp_path):
    encoded_time = datetime(2026, 7, 7, 9, 30, tzinfo=timezone.utc).isoformat()
    for name, relative_path in [
        ("one.json", "Payroll Processing/One.md"),
        ("two.json", "Payroll Processing/Two.md"),
        ("ignored.json", "Flexible Payroll/Ignored.md"),
    ]:
        (tmp_path / name).write_text(
            """
{
  "package_kind": "github_repo_document",
  "doc_id": "%s",
  "title": "%s",
  "source_url": "https://github.example.test/payroll/architecture/blob/main/%s",
  "last_modified": "%s",
  "space_or_project": "payroll/architecture",
  "version": "blob-sha-1",
  "repo_url": "https://github.example.test/payroll/architecture",
  "repo_host": "github.example.test",
  "repo_owner": "payroll",
  "repo_name": "architecture",
  "repo_ref": "main",
  "relative_path": "%s",
  "blob_sha": "blob-sha-1",
  "content_type": "text/markdown",
  "submitted_at": "%s",
  "submitted_by": "cli",
  "markdown": "# Doc"
}
""".strip()
            % (name, relative_path, relative_path, encoded_time, relative_path, encoded_time),
            encoding="utf-8",
        )

    gene = GitHubRepoGene(
        config={
            "connection_mode": "local_push",
            "repo_url": "https://github.example.test/payroll/architecture",
            "ref": "main",
            "include_paths": ["Payroll Processing/"],
            "include_extensions": ["md"],
            "max_files": 1,
            "documents_dir": str(tmp_path),
        },
        source_id="src-github-repo",
    )

    await gene.authenticate()
    with pytest.raises(RuntimeError, match="max_files"):
        [item async for item in gene.discover()]
