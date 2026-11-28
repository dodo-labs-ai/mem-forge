"""GitHub collection contracts at the provider transport boundary."""

import base64
import hashlib

import httpx
import pytest

from memforge.genes import github_repo_gene
from memforge.genes.github_repo_gene import GitHubRepoGene
from tests.connector_http_fixture import FakeConnectorHttp


HELLO = b"hello\n"
HELLO_SHA = "ce013625030ba8dba906f756967f9e9ca394464a"
COMMIT_DATE = "2026-05-01T10:00:00Z"
COMMIT_TIME = "2026-05-01T10:00:00+00:00"


def commits_listing(date):
    return [{"sha": "file-commit", "commit": {"committer": {"date": date}}}]


class Response:
    def __init__(self, payload=None, body=b"", headers=None):
        self.payload = payload
        self.body = body
        self.headers = headers if headers is not None else {"content-length": str(len(body))}

    def json(self):
        return self.payload

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for start in range(0, len(self.body), chunk_size):
            yield self.body[start:start + chunk_size]

    def close(self):
        pass


class RepositorySession:
    def __init__(self):
        self.headers = {}
        self.calls = []
        self.body = HELLO
        self.sha = HELLO_SHA
        self.size = len(HELLO)
        self.mode = "100644"
        self.entry_type = "blob"

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if "/commits/main" in url:
            return Response({"sha": "commit-one", "commit": {"tree": {"sha": "tree-one"}}})
        if "/commits?" in url:
            assert "sha=commit-one&path=README.md&per_page=1" in url
            return Response(commits_listing(COMMIT_DATE))
        if "/git/trees/" in url:
            return Response({"truncated": False, "tree": [{
                "path": "README.md", "type": self.entry_type, "mode": self.mode,
                "sha": self.sha, "size": self.size,
            }]})
        if "/contents/" in url:
            # Same advertised identity, transformed Contents response.
            return Response({"sha": self.sha, "size": self.size,
                             "encoding": "base64", "content": base64.b64encode(b"transformed").decode()})
        if "/git/blobs/" in url:
            assert kwargs["headers"]["Accept"] == "application/vnd.github.raw+json"
            return Response(body=self.body)
        raise AssertionError(url)

    def close(self):
        pass


class _Chunks(httpx.AsyncByteStream):
    """A raw body handed to the client chunk by chunk, read only as far as the client reads."""

    def __init__(self, response: Response) -> None:
        self._response = response

    async def __aiter__(self):
        for chunk in self._response.iter_content(65_536):
            yield chunk


async def _cloud_gene(session: RepositorySession) -> GitHubRepoGene:
    """A Gene whose GitHub requests ``session`` answers."""

    def handle(request: httpx.Request) -> httpx.Response:
        response = session.get(str(request.url), headers=request.headers)
        if response.payload is not None:
            return httpx.Response(200, json=response.payload)
        # Header values as the bytes a server sends, which may be any UTF-8.
        headers = [(name.encode(), value.encode()) for name, value in response.headers.items()]
        return httpx.Response(200, headers=headers, stream=_Chunks(response))

    gene = GitHubRepoGene(
        config={"repo_url": "https://github.com/example/repo", "ref": "main"},
        source_id="src-test",
    )
    with FakeConnectorHttp(handle).serving(github_repo_gene):
        await gene.authenticate()
    return gene


@pytest.fixture
async def cloud_collection():
    session = RepositorySession()
    gene = await _cloud_gene(session)
    yield gene, session
    await gene.close()


@pytest.mark.asyncio
async def test_collection_reads_original_blob_despite_transformed_contents(cloud_collection):
    gene, session = cloud_collection
    item = [item async for item in gene.discover()][0]
    raw = await gene.fetch(item)
    assert raw.body == HELLO
    normalized = await gene.normalize(raw)
    assert normalized.markdown_body == "hello\n"
    # The file's source time is its latest commit at the collection commit, not the sync time.
    assert normalized.source_semantics["source_updated_at"] == COMMIT_TIME
    urls = [url for url, _ in session.calls]
    assert any("/git/trees/tree-one?" in url for url in urls)
    assert any(url.endswith("/git/blobs/" + HELLO_SHA) for url in urls)
    assert not any("/contents/" in url for url in urls)
    assert item.extra["repo_ref"] == "main"


@pytest.mark.asyncio
@pytest.mark.parametrize("headers", [{}, {"content-length": "6"}])
async def test_optional_sizes_do_not_prevent_identity_verification(cloud_collection, headers):
    gene, session = cloud_collection
    session.size = None
    original_get = session.get
    session.get = lambda url, **kwargs: (Response(body=HELLO, headers=headers)
                                        if "/git/blobs/" in url else original_get(url, **kwargs))
    item = [item async for item in gene.discover()][0]
    assert (await gene.fetch(item)).body == HELLO


@pytest.mark.asyncio
@pytest.mark.parametrize("length", ["", "+6", "-6", "６", "6.0"])
async def test_supplied_invalid_http_lengths_fail_before_body_read(cloud_collection, length):
    gene, session = cloud_collection
    original_get = session.get
    session.get = lambda url, **kwargs: (Response(body=HELLO, headers={"content-length": length})
                                        if "/git/blobs/" in url else original_get(url, **kwargs))
    item = [item async for item in gene.discover()][0]
    with pytest.raises(RuntimeError, match="transport length"):
        await gene.fetch(item)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [-1, True, "6"])
async def test_invalid_inventory_size_does_not_allow_a_body(cloud_collection, size):
    gene, session = cloud_collection
    session.size = size
    item = [item async for item in gene.discover()][0]
    with pytest.raises(RuntimeError, match="size is invalid"):
        await gene.fetch(item)


@pytest.mark.asyncio
@pytest.mark.parametrize("body", [b"", b"hello\n", "中文 — hello\n".encode(), b'{"value": 1}'])
async def test_verified_text_and_empty_files_keep_their_identity(cloud_collection, body):
    gene, session = cloud_collection
    session.body = body
    session.size = len(body)
    session.sha = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    item = [item async for item in gene.discover()][0]
    if body.startswith(b"{"):
        item.content_type = "application/json"
    raw = await gene.fetch(item)
    normalized = await gene.normalize(raw)
    assert raw.body == body
    assert normalized.source_semantics["blob_sha"] == session.sha
    assert raw.authoritative_empty is (not body)


@pytest.mark.asyncio
@pytest.mark.parametrize("body,size,error", [
    (b"short", 6, "size mismatch"),
    (b"extra!!", 6, "size mismatch"),
    (b"other\n", 6, "hash mismatch"),
    (b'{"content":"aGVsbG8K"}', None, "hash mismatch"),
])
async def test_raw_response_cannot_change_pinned_identity(cloud_collection, body, size, error):
    gene, session = cloud_collection
    session.size = size
    item = [item async for item in gene.discover()][0]
    session.body = body
    with pytest.raises(RuntimeError, match=error):
        await gene.fetch(item)


@pytest.mark.asyncio
async def test_raw_bytes_verified_but_mixed_encoding_is_not_repaired(cloud_collection):
    gene, session = cloud_collection
    session.body = "UTF-8 — ".encode() + b"\xd6"
    session.size = len(session.body)
    session.sha = hashlib.sha1(b"blob " + str(session.size).encode() + b"\0" + session.body).hexdigest()
    item = [item async for item in gene.discover()][0]
    raw = await gene.fetch(item)
    assert raw.body == session.body
    with pytest.raises(ValueError, match="invalid UTF-8 at byte"):
        await gene.normalize(raw)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, "160000"])
async def test_selected_non_regular_files_are_not_treated_as_text(cloud_collection, mode):
    gene, session = cloud_collection
    session.mode = mode
    if mode == "160000":
        session.entry_type = "commit"
    with pytest.raises(RuntimeError, match="unsupported file mode"):
        [item async for item in gene.discover()]


@pytest.mark.asyncio
async def test_malformed_symlink_tree_entry_fails_instead_of_disappearing(cloud_collection):
    gene, session = cloud_collection
    session.mode = "120000"
    session.entry_type = "tree"
    with pytest.raises(RuntimeError, match="not represented by a blob"):
        [item async for item in gene.discover()]


@pytest.mark.asyncio
async def test_cloud_collection_resolves_selected_text_symlink_at_pinned_tree():
    link_target = b"docs/CONTRIBUTIONS.md"
    link_sha = hashlib.sha1(
        b"blob " + str(len(link_target)).encode() + b"\0" + link_target
    ).hexdigest()
    target_body = b"# Contributions\n\nUse reviewed changes.\n"
    target_sha = hashlib.sha1(
        b"blob " + str(len(target_body)).encode() + b"\0" + target_body
    ).hexdigest()

    class SymlinkSession(RepositorySession):
        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if "/commits/main" in url:
                return Response({"sha": "commit-one", "commit": {"tree": {"sha": "tree-one"}}})
            if "/commits?" in url:
                return Response(commits_listing(
                    "2026-06-02T08:00:00Z" if "path=docs/CONTRIBUTIONS.md" in url else COMMIT_DATE
                ))
            if "/git/trees/" in url:
                return Response({"truncated": False, "tree": [
                    {"path": "README.md", "type": "blob", "mode": "120000",
                     "sha": link_sha, "size": len(link_target)},
                    {"path": "docs/CONTRIBUTIONS.md", "type": "blob", "mode": "100644",
                     "sha": target_sha, "size": len(target_body)},
                ]})
            if url.endswith("/git/blobs/" + link_sha):
                return Response(body=link_target)
            if url.endswith("/git/blobs/" + target_sha):
                return Response(body=target_body)
            raise AssertionError(url)

    session = SymlinkSession()
    gene = await _cloud_gene(session)

    items = [item async for item in gene.discover()]
    readme = next(item for item in items if item.extra["relative_path"] == "README.md")
    raw = await gene.fetch(readme)
    normalized = await gene.normalize(raw)

    assert readme.version == target_sha
    assert readme.extra["blob_sha"] == target_sha
    assert readme.extra["resolved_relative_path"] == "docs/CONTRIBUTIONS.md"
    assert readme.extra["symlink_chain"] == [{
        "path": "README.md", "blob_sha": link_sha, "target_path": "docs/CONTRIBUTIONS.md",
    }]
    assert raw.body == target_body
    assert normalized.source_semantics["relative_path"] == "README.md"
    assert normalized.source_semantics["resolved_relative_path"] == "docs/CONTRIBUTIONS.md"
    # A symlinked file changes when either the link or its target changes.
    assert normalized.source_semantics["source_updated_at"] == "2026-06-02T08:00:00+00:00"


@pytest.mark.asyncio
async def test_cloud_collection_rejects_text_symlink_to_binary_target():
    link_target = b"assets/photo.png"
    link_sha = hashlib.sha1(
        b"blob " + str(len(link_target)).encode() + b"\0" + link_target
    ).hexdigest()

    class BinaryTargetSession(RepositorySession):
        def get(self, url, **kwargs):
            self.calls.append((url, kwargs))
            if "/commits/main" in url:
                return Response({"sha": "commit-one", "commit": {"tree": {"sha": "tree-one"}}})
            if "/git/trees/" in url:
                return Response({"truncated": False, "tree": [
                    {"path": "README.md", "type": "blob", "mode": "120000",
                     "sha": link_sha, "size": len(link_target)},
                    {"path": "assets/photo.png", "type": "blob", "mode": "100644",
                     "sha": "b" * 40, "size": 8},
                ]})
            if url.endswith("/git/blobs/" + link_sha):
                return Response(body=link_target)
            raise AssertionError(url)

    session = BinaryTargetSession()
    gene = await _cloud_gene(session)

    with pytest.raises(RuntimeError, match="binary Artifact"):
        [item async for item in gene.discover()]


@pytest.mark.asyncio
async def test_stream_stops_before_buffering_more_than_declared_size(cloud_collection):
    gene, session = cloud_collection
    response = Response(headers={})
    chunks_read = []

    def chunks(chunk_size):
        for chunk in (b"hello\n", b"extra", b"must not read"):
            chunks_read.append(chunk)
            yield chunk

    response.iter_content = chunks
    original_get = session.get
    session.get = lambda url, **kwargs: response if "/git/blobs/" in url else original_get(url, **kwargs)
    item = [item async for item in gene.discover()][0]
    with pytest.raises(RuntimeError, match="exceeds declared size"):
        await gene.fetch(item)
    assert chunks_read == [b"hello\n", b"extra"]


@pytest.mark.asyncio
async def test_text_larger_than_four_mib_keeps_existing_input_range(cloud_collection):
    gene, session = cloud_collection
    session.body = b"valid text\n" * 500_000
    session.size = len(session.body)
    session.sha = hashlib.sha1(b"blob " + str(session.size).encode() + b"\0" + session.body).hexdigest()
    item = [item async for item in gene.discover()][0]
    raw = await gene.fetch(item)
    assert raw.body == session.body
    assert (await gene.normalize(raw)).markdown_body == session.body.decode()


@pytest.mark.asyncio
async def test_unknown_lengths_still_bound_actual_stream_bytes(cloud_collection, monkeypatch):
    monkeypatch.setattr("memforge.github_repo_utils.GITHUB_BLOB_MAX_BYTES", 5)
    gene, session = cloud_collection
    session.size = None
    original_get = session.get
    session.get = lambda url, **kwargs: (Response(body=HELLO, headers={})
                                        if "/git/blobs/" in url else original_get(url, **kwargs))
    item = [item async for item in gene.discover()][0]
    with pytest.raises(RuntimeError, match="provider byte limit"):
        await gene.fetch(item)


