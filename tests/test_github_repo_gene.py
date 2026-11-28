from __future__ import annotations

import json
import re
import ssl
from datetime import datetime, timedelta, timezone

import httpx
import pytest

from memforge.genes import GENE_REGISTRY, github_repo_gene
from memforge.genes.github_repo_gene import GitHubRepoGene
from memforge.genes.toolkit.github_api import GitHubRateLimitError
from memforge.github_repo_utils import build_github_repo_doc_id
from memforge.models import ContentItem, require_attested_content
from memforge.source_artifacts import SourceArtifactContractError
from tests.github_api_fixture import FakeGitHub, git_blob_sha
from tests.tls_bundle_fixture import BUNDLE_CA_COUNT, trusted_ca_count, write_ca_bundle

REPO_URL = "https://github.example.test/payroll/architecture"
API = "https://github.example.test/api/v3/repos/payroll/architecture"
README = b"# Payroll Processing\n\nCreate tasks carefully."
FILES = {
    "Payroll Processing/README.md": README,
    "Payroll Processing/images/diagram.png": b"\x89PNG\r\n\x1a\n" + b"0" * 2040,
    "Flexible Payroll/README.md": b"# Flexible Payroll\n",
    "Payroll Processing V2/Main Algorithm.md": b"# Main Algorithm\n",
}
REPO_CONFIG = {"repo_url": REPO_URL, "ref": "main"}
FILE_COMMIT_TIME = "2026-05-01T10:00:00+00:00"


async def _connected(github: FakeGitHub, config: dict) -> GitHubRepoGene:
    gene = GitHubRepoGene(config=config, source_id="src-github-repo")
    with github.http.serving(github_repo_gene):
        await gene.authenticate()
    return gene


def _commit_time_requests(github: FakeGitHub) -> list[str]:
    return [request for request in github.requests if "/commits?" in request]


def test_github_repo_gene_is_registered_and_schema_is_repo_oriented():
    assert GENE_REGISTRY["github_repo"] is GitHubRepoGene

    meta = GitHubRepoGene.metadata()
    assert meta.name == "github_repo"
    assert meta.display_name == "GitHub Repository"
    assert meta.auth_method == "github_repo"
    assert meta.data_shape == "document"

    fields = {field.key: field for field in GitHubRepoGene.config_schema().fields}
    assert "connection_mode" not in fields
    assert fields["repo_url"].required is True
    assert fields["ref"].default == "main"
    assert fields["include_paths"].required is False
    assert fields["exclude_paths"].required is False
    assert "repo_path" not in fields
    assert fields["include_extensions"].default == "md, markdown, txt, adoc, rst"
    assert fields["max_files"].default == "500"
    assert fields["pat"].required is False
    assert fields["pat"].visible_when == ()


@pytest.mark.asyncio
async def test_cloud_pull_discovers_scoped_markdown_and_fetches_content():
    github = FakeGitHub(API, files=FILES)
    gene = await _connected(
        github,
        {
            **REPO_CONFIG,
            "include_paths": [],
            "exclude_paths": ["Flexible Payroll", "Payroll Processing V2"],
            "include_extensions": ["md"],
            "max_files": 10,
        },
    )

    items = [item async for item in gene.discover()]

    assert [item.extra["relative_path"] for item in items] == ["Payroll Processing/README.md"]
    item = items[0]
    assert item.item_id == build_github_repo_doc_id(
        source_id="src-github-repo",
        repo_url=REPO_URL,
        repo_ref="main",
        relative_path="Payroll Processing/README.md",
    )
    assert item.title == "README"
    assert item.source_url == (
        "https://github.example.test/payroll/architecture/blob/main/Payroll%20Processing/README.md"
    )
    assert item.version == "e3e42b115cf369088b26398e3506f9eebbed519b"
    assert item.space_or_project == "payroll/architecture"
    assert item.extra["repo_host"] == "github.example.test"
    assert item.extra["repo_owner"] == "payroll"
    assert item.extra["repo_name"] == "architecture"
    assert item.extra["repo_ref"] == "main"
    assert item.extra["blob_sha"] == "e3e42b115cf369088b26398e3506f9eebbed519b"
    assert item.extra["repo_blob_url"] == f"{API}/git/blobs/e3e42b115cf369088b26398e3506f9eebbed519b"
    assert item.extra["file_identity_contract"] == "repository_path"
    assert "previous_filename" not in item.extra

    raw = await gene.fetch(item)
    normalized = await gene.normalize(raw)

    assert raw.content_type == "text/markdown"
    assert normalized.markdown_body.startswith("# Payroll Processing")
    assert (
        f"GET {API}/commits?sha=commit-main&path=Payroll%20Processing/README.md&per_page=1" in github.requests
    )
    assert normalized.source_semantics == {
        "source_updated_at": FILE_COMMIT_TIME,
        "source_type": "github_repo",
        "repo_url": REPO_URL,
        "repo_identifier": "github.example.test/payroll/architecture",
        "repo_host": "github.example.test",
        "repo_owner": "payroll",
        "repo_name": "architecture",
        "repo_ref": "main",
        "relative_path": "Payroll Processing/README.md",
        "blob_sha": "e3e42b115cf369088b26398e3506f9eebbed519b",
        "content_type": "text/markdown",
        "canonical_url": "https://github.example.test/payroll/architecture/blob/main/Payroll%20Processing/README.md",
    }


async def _discovered_cloud_pull_item(github: FakeGitHub) -> tuple[GitHubRepoGene, ContentItem]:
    gene = await _connected(
        github,
        {**REPO_CONFIG, "include_paths": ["Payroll Processing/"], "include_extensions": ["md"], "max_files": 10},
    )
    [item] = [item async for item in gene.discover()]
    return gene, item


@pytest.mark.asyncio
async def test_cloud_pull_keeps_the_commit_time_recorded_for_an_unchanged_blob():
    github = FakeGitHub(API, files=FILES)
    gene, item = await _discovered_cloud_pull_item(github)
    item.stored_extra = {"blob_sha": item.extra["blob_sha"], "last_commit_at": "2026-04-01T08:00:00+00:00"}

    normalized = await gene.normalize(await gene.fetch(item))

    assert _commit_time_requests(github) == []
    assert normalized.source_semantics["source_updated_at"] == "2026-04-01T08:00:00+00:00"


@pytest.mark.asyncio
async def test_cloud_pull_asks_again_for_a_changed_blob_or_a_missing_time():
    for recorded in ({"blob_sha": "previous-blob", "last_commit_at": "2026-04-01T08:00:00+00:00"}, {"last_commit_at": None}):
        github = FakeGitHub(API, files=FILES)
        gene, item = await _discovered_cloud_pull_item(github)
        item.stored_extra = {"blob_sha": item.extra["blob_sha"], **recorded}

        normalized = await gene.normalize(await gene.fetch(item))

        assert len(_commit_time_requests(github)) == 1
        assert normalized.source_semantics["source_updated_at"] == FILE_COMMIT_TIME


_README_COMMITS = "/commits?sha=commit-main&path=Payroll%20Processing/README.md&per_page=1"


@pytest.mark.asyncio
async def test_cloud_pull_syncs_the_file_without_a_time_when_github_refuses_the_commit_lookup():
    refused = httpx.Response(403, json={"message": "Resource not accessible by integration"})
    github = FakeGitHub(API, files=FILES, answers={_README_COMMITS: refused})
    gene, item = await _discovered_cloud_pull_item(github)

    normalized = await gene.normalize(await gene.fetch(item))

    assert normalized.markdown_body.startswith("# Payroll Processing")
    assert normalized.source_semantics["source_updated_at"] is None
    # A refusal is not a rate limit: it is asked once, and nothing is waited for.
    assert len(_commit_time_requests(github)) == 1
    assert github.http.waits == []


@pytest.mark.asyncio
async def test_a_permission_refusal_of_the_tree_fails_discovery_without_retrying():
    refused = httpx.Response(403, json={"message": "Resource not accessible by integration"})
    github = FakeGitHub(API, files=FILES, answers={"/git/trees/tree-main?recursive=1": refused})
    gene = await _connected(github, dict(REPO_CONFIG))

    with pytest.raises(httpx.HTTPStatusError) as raised:
        _ = [item async for item in gene.discover()]

    assert raised.value.response.status_code == 403
    assert "rate limit" not in str(raised.value).lower()
    assert sum("/git/trees/" in request for request in github.requests) == 1
    assert github.http.waits == []


@pytest.mark.asyncio
async def test_a_rate_limit_that_resets_soon_is_waited_out():
    reset = int((datetime.now(timezone.utc) + timedelta(seconds=20)).timestamp())
    limited = httpx.Response(
        403,
        headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(reset)},
        json={"message": "API rate limit exceeded"},
    )
    github = FakeGitHub(API, files=FILES)
    answers = iter([limited])

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/commits/main"):
            limited_once = next(answers, None)
            if limited_once is not None:
                github.requests.append(f"{request.method} {request.url}")
                return limited_once
        return github.handle(request)

    github.http.transport = httpx.MockTransport(handle)
    gene = await _connected(github, {**REPO_CONFIG, "include_extensions": ["md"]})

    items = [item async for item in gene.discover()]

    assert len(items) == 3
    [wait] = github.http.waits
    assert 15 <= wait <= 21
    assert sum(request.endswith("/commits/main") for request in github.requests) == 2


@pytest.mark.asyncio
async def test_a_rate_limit_that_resets_later_than_a_sync_waits_fails_with_its_reset_time():
    reset = datetime.now(timezone.utc).replace(microsecond=0) + timedelta(hours=1)
    limited = httpx.Response(
        403,
        headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(reset.timestamp()))},
        json={"message": "API rate limit exceeded"},
    )
    github = FakeGitHub(API, files=FILES, answers={"/commits/main": limited})
    gene = await _connected(github, dict(REPO_CONFIG))

    with pytest.raises(GitHubRateLimitError, match=r"GitHub rate limit reached .* it resets at ") as raised:
        _ = [item async for item in gene.discover()]

    assert f"{reset:%Y-%m-%d %H:%M:%S} UTC" in str(raised.value)
    assert github.http.waits == []


@pytest.mark.asyncio
async def test_cloud_pull_materializes_explicitly_selected_image_blob():
    image_bytes = b"\x89PNG\r\n\x1a\nstable-github-image"
    github = FakeGitHub(API, files={"docs/architecture.png": image_bytes})
    gene = await _connected(github, {**REPO_CONFIG, "include_extensions": ["png"]})

    item = [item async for item in gene.discover()][0]
    raw = await gene.fetch(item)
    normalized = await gene.normalize(raw)

    assert item.content_type == "image/png"
    assert raw.body == b""
    assert raw.authoritative_empty is True
    assert normalized.markdown_body == ""
    assert normalized.source_semantics["source_updated_at"] == FILE_COMMIT_TIME
    assert len(raw.artifacts) == 1
    artifact = raw.artifacts[0]
    assert artifact.provider_key == "docs/architecture.png"
    assert artifact.parent_observation_type == "file_content"
    assert artifact.parent_provider_key == "content"
    assert artifact.provider_revision == git_blob_sha(image_bytes)
    assert artifact.declared_size_bytes == len(image_bytes)

    async with gene.open_source_artifact(artifact) as download:
        observed = b"".join([chunk async for chunk in download.chunks])

    assert observed == image_bytes
    assert f"GET {API}/git/blobs/{git_blob_sha(image_bytes)}" in github.requests
    assert all("/contents/" not in request for request in github.requests)


@pytest.mark.asyncio
async def test_raw_blob_reads_ask_for_the_raw_media_type_and_identity_encoding():
    github = FakeGitHub(API, files=FILES)
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return github.handle(request)

    github.http.transport = httpx.MockTransport(handle)
    gene, item = await _discovered_cloud_pull_item(github)

    await gene.fetch(item)

    [blob_read] = [request for request in seen if "/git/blobs/" in request.url.path]
    assert blob_read.headers["Accept"] == "application/vnd.github.raw+json"
    assert blob_read.headers["Accept-Encoding"] == "identity"


@pytest.mark.asyncio
async def test_cloud_pull_rejects_a_raw_blob_sent_with_a_content_encoding():
    github = FakeGitHub(API, files=FILES)
    readme_blob = f"/git/blobs/{git_blob_sha(README)}"
    github.answers[readme_blob] = httpx.Response(200, stream=httpx.ByteStream(README), headers={"content-encoding": "br"})
    gene, item = await _discovered_cloud_pull_item(github)

    with pytest.raises(RuntimeError, match="unsupported content encoding"):
        await gene.fetch(item)


@pytest.mark.asyncio
async def test_cloud_pull_rejects_explicitly_selected_unsupported_image():
    github = FakeGitHub(API, files={"docs/architecture.svg": b"<svg/>"})
    gene = await _connected(github, {**REPO_CONFIG, "include_extensions": ["svg"]})

    item = [item async for item in gene.discover()][0]

    assert item.content_type == "image/svg+xml"
    with pytest.raises(SourceArtifactContractError, match="unsupported binary media type"):
        await gene.fetch(item)


def test_normalize_config_canonicalizes_repository_scope() -> None:
    config = {
        "repo_url": "https://github.example.test/payroll/architecture",
        "include_paths": ["docs/current/guide.md", "docs"],
        "exclude_paths": ["docs/archive/old.md", "docs/archive"],
    }

    GitHubRepoGene.normalize_config(config)

    assert config["include_paths"] == ["docs"]
    assert config["exclude_paths"] == ["docs/archive"]


@pytest.mark.asyncio
async def test_cloud_pull_reads_a_truncated_tree_by_its_subtrees():
    github = FakeGitHub(API, files=FILES, truncated={""})
    gene = await _connected(github, {**REPO_CONFIG, "include_extensions": ["md"]})

    items = [item async for item in gene.discover()]

    assert sorted(item.extra["relative_path"] for item in items) == [
        "Flexible Payroll/README.md",
        "Payroll Processing V2/Main Algorithm.md",
        "Payroll Processing/README.md",
    ]
    assert f"GET {API}/git/trees/tree-main" in github.requests
    assert f"GET {API}/git/trees/{github.tree_sha('Payroll Processing')}?recursive=1" in github.requests


@pytest.mark.asyncio
async def test_cloud_pull_stops_reading_a_truncated_tree_once_it_selects_more_than_max_files():
    files = {f"area-{index}/{name}.md": f"# {index} {name}\n".encode() for index in range(5) for name in ("a", "b")}
    github = FakeGitHub(API, files=files, truncated={""})
    gene = await _connected(github, {**REPO_CONFIG, "max_files": 3})

    with pytest.raises(RuntimeError) as raised:
        await gene.list_scope()

    # The message both admin UIs parse; reading stopped at the folder that passed the limit, so 4 is a lower bound.
    assert re.fullmatch(r"GitHub Repository discovery matched at least 4 files, exceeding max_files=3", str(raised.value))
    assert not any(github.tree_sha("area-2") in request for request in github.requests)


@pytest.mark.asyncio
async def test_cloud_pull_counts_every_selected_file_of_a_complete_tree_against_max_files():
    files = {f"docs/{index}.md": b"# Doc\n" for index in range(4)}
    github = FakeGitHub(API, files=files)
    gene = await _connected(github, {**REPO_CONFIG, "max_files": 3})

    with pytest.raises(RuntimeError, match=r"^GitHub Repository discovery matched 4 files, exceeding max_files=3$"):
        await gene.list_scope()


@pytest.mark.asyncio
async def test_cloud_pull_rejects_missing_git_tree_without_completion_evidence():
    github = FakeGitHub(API, files=FILES, answers={"/git/trees/tree-main?recursive=1": {}})
    gene = await _connected(github, dict(REPO_CONFIG))

    with pytest.raises(RuntimeError, match="truncated=false"):
        _ = [item async for item in gene.discover()]
    with pytest.raises(RuntimeError, match="truncated=false"):
        await gene.list_scope()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tree_payload",
    [
        {"truncated": None, "tree": []},
        {
            "truncated": False,
            "tree": [
                {"path": "docs/a.md", "type": "blob", "sha": "sha-a", "mode": "100644"},
                {"path": "docs/a.md", "type": "blob", "sha": "sha-b", "mode": "100644"},
            ],
        },
        {"truncated": False, "tree": [{"path": "docs/a.md", "type": "blob", "sha": "", "mode": "100644"}]},
    ],
)
async def test_cloud_pull_rejects_tree_without_unique_stable_inventory(tree_payload):
    github = FakeGitHub(API, files=FILES, answers={"/git/trees/tree-main?recursive=1": tree_payload})
    gene = await _connected(github, dict(REPO_CONFIG))

    with pytest.raises(RuntimeError):
        _ = [item async for item in gene.discover()]
    with pytest.raises(RuntimeError):
        await gene.list_scope()


@pytest.mark.asyncio
async def test_cloud_pull_uses_optional_pat_as_bearer_header():
    github = FakeGitHub(API, files=FILES)
    await _connected(github, {**REPO_CONFIG, "pat": "secret-token"})
    await _connected(github, dict(REPO_CONFIG))

    with_token, without_token = github.http.options
    assert with_token["headers"]["Authorization"] == "Bearer secret-token"
    assert "Authorization" not in without_token["headers"]


@pytest.mark.asyncio
async def test_cloud_pull_reads_the_token_from_its_environment_variable(monkeypatch):
    monkeypatch.setenv("GHE_TOKEN", "env-token")
    github = FakeGitHub(API, files=FILES)
    await _connected(github, {**REPO_CONFIG, "pat_env_var": "GHE_TOKEN"})

    assert github.http.options[-1]["headers"]["Authorization"] == "Bearer env-token"


@pytest.mark.asyncio
async def test_cloud_pull_follows_redirects_and_trusts_requests_ca_bundle(tmp_path, monkeypatch):
    bundle = write_ca_bundle(tmp_path / "corp-ca.pem")
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", str(bundle))
    github = FakeGitHub(API, files=FILES)

    await _connected(github, dict(REPO_CONFIG))

    options = github.http.options[-1]
    assert options["follow_redirects"] is True
    assert isinstance(options["verify"], ssl.SSLContext)
    assert trusted_ca_count(options["verify"]) == BUNDLE_CA_COUNT


@pytest.mark.asyncio
async def test_cloud_pull_uses_the_default_trust_store_without_a_bundle(monkeypatch):
    monkeypatch.delenv("REQUESTS_CA_BUNDLE", raising=False)
    github = FakeGitHub(API, files=FILES)

    await _connected(github, dict(REPO_CONFIG))

    assert github.http.options[-1]["verify"] is True


@pytest.mark.asyncio
async def test_a_closed_gene_does_not_reconnect_on_its_own():
    github = FakeGitHub(API, files=FILES)
    gene = await _connected(github, dict(REPO_CONFIG))
    await gene.close()

    with github.http.serving(github_repo_gene), pytest.raises(RuntimeError, match="call authenticate"):
        await gene.list_scope()

    assert len(github.http.clients) == 1
    assert github.http.clients[0].is_closed
    assert not any("/git/trees/" in request for request in github.requests)


@pytest.mark.asyncio
async def test_a_gene_that_never_authenticated_does_not_connect_on_its_own():
    github = FakeGitHub(API, files=FILES)
    gene = GitHubRepoGene(config=dict(REPO_CONFIG), source_id="src-github-repo")

    with github.http.serving(github_repo_gene), pytest.raises(RuntimeError, match="call authenticate"):
        await gene.list_scope()

    assert github.http.clients == []


@pytest.mark.asyncio
@pytest.mark.parametrize("package_kind", [None, "teams_window_document"])
async def test_a_json_repository_file_is_read_as_file_text(package_kind):
    body = {"repo_url": "https://github.example.test/other/repo", "markdown": "not a package"}
    if package_kind is not None:
        body["package_kind"] = package_kind
    item = ContentItem(
        item_id="github-repo-src-config",
        title="config",
        source_url="https://github.example.test/payroll/architecture/blob/main/config.json",
        last_modified=datetime(2026, 7, 7, tzinfo=timezone.utc),
        content_type="application/json",
        extra={
            "repo_url": "https://github.example.test/payroll/architecture",
            "relative_path": "config.json",
        },
    )
    gene = GitHubRepoGene(
        config={"repo_url": "https://github.example.test/payroll/architecture"},
        source_id="src-github-repo",
    )
    text = json.dumps(body).encode("utf-8")

    stored = gene.raw_from_stored_input(item, text, "application/json")
    normalized = await gene.normalize(stored)

    assert (stored.authoritative_empty, stored.empty_evidence) == (False, None)
    assert "not a package" in normalized.markdown_body
    assert normalized.source_semantics["repo_identifier"] == "github.example.test/payroll/architecture"
    assert normalized.source_semantics["relative_path"] == "config.json"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "attested"),
    [("image/png", True), ("text/markdown", False)],
    ids=["image_file", "text_file"],
)
async def test_a_stored_image_file_is_rebuilt_as_the_empty_text_its_fetch_attested(content_type, attested):
    item = ContentItem(
        item_id="github-repo-src-diagram",
        title="diagram",
        source_url="https://github.example.test/payroll/architecture/blob/main/docs/diagram.png",
        last_modified=datetime(2026, 7, 7, tzinfo=timezone.utc),
        content_type=content_type,
        extra={
            "repo_url": "https://github.example.test/payroll/architecture",
            "relative_path": "docs/diagram.png",
        },
    )
    gene = GitHubRepoGene(
        config={"repo_url": "https://github.example.test/payroll/architecture"},
        source_id="src-github-repo",
    )

    stored = gene.raw_from_stored_input(item, b"", content_type)

    assert stored.authoritative_empty is attested
    if attested:
        require_attested_content(stored, await gene.normalize(stored))
        assert stored.empty_evidence == "github_binary_file_has_no_text_body"
    else:
        with pytest.raises(ValueError, match="did not attest"):
            require_attested_content(stored, await gene.normalize(stored))


@pytest.mark.asyncio
async def test_incremental_cloud_pull_lists_the_whole_tree_as_an_existence_listing():
    github = FakeGitHub(API, files=FILES)
    gene = await _connected(github, {**REPO_CONFIG, "include_extensions": ["md"], "max_files": 10})

    _ = [item async for item in gene.discover(since=datetime(2026, 9, 30, tzinfo=timezone.utc))]
    listing = await gene.list_scope()

    assert listing.kind.value == "existence"
    assert listing.doc_ids == {
        build_github_repo_doc_id(
            source_id="src-github-repo",
            repo_url=REPO_URL,
            repo_ref="main",
            relative_path=path,
        )
        for path in ("Payroll Processing/README.md", "Flexible Payroll/README.md", "Payroll Processing V2/Main Algorithm.md")
    }
