"""GitHub collection contracts at the provider transport boundary."""

import base64
import json
import subprocess
import sys
import hashlib
from tempfile import TemporaryFile

import pytest

from memforge.genes.github_repo_gene import GitHubRepoGene
from memforge import main
from tests.test_cli_agent_tools import FakeToolClient, _cloud_test_client


HELLO = b"hello\n"
HELLO_SHA = "ce013625030ba8dba906f756967f9e9ca394464a"


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
            assert kwargs["stream"] is True
            assert kwargs["headers"]["Accept"] == "application/vnd.github.raw+json"
            return Response(body=self.body)
        raise AssertionError(url)

    def close(self):
        pass


@pytest.fixture
def cloud_collection(monkeypatch):
    session = RepositorySession()
    monkeypatch.setattr("memforge.genes.github_repo_gene.requests.Session", lambda: session)
    gene = GitHubRepoGene(config={"connection_mode": "cloud_pull",
                                "repo_url": "https://github.com/example/repo", "ref": "main"},
                          source_id="src-test")
    return gene, session


@pytest.mark.asyncio
async def test_collection_reads_original_blob_despite_transformed_contents(cloud_collection):
    gene, session = cloud_collection
    item = [item async for item in gene.discover()][0]
    raw = await gene.fetch(item)
    assert raw.body == HELLO
    assert (await gene.normalize(raw)).markdown_body == "hello\n"
    urls = [url for url, _ in session.calls]
    assert any("/git/trees/tree-one?" in url for url in urls)
    assert any(url.endswith("/git/blobs/" + HELLO_SHA) for url in urls)
    assert not any("/contents/" in url for url in urls)
    assert item.extra["repo_ref"] == "main"


@pytest.mark.parametrize("body,size,error", [
    (HELLO, 6, None),
    (b"", 0, None),
    ("中文 — hello\n".encode(), None, None),
    (b"bad\xd6", 4, "invalid UTF-8"),
    (HELLO, 7, "size mismatch"),
    (HELLO, 5, "exceeds declared size"),
])
def test_daemon_collects_raw_blob_and_uploads_exact_text(monkeypatch, body, size, error):
    session = RepositorySession()
    session.body = body
    session.size = size
    session.sha = hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()
    real_popen = subprocess.Popen

    def metadata(cmd, **kwargs):
        response = session.get("https://api.github.com/" + cmd[2])
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(response.json()), stderr="")

    def raw_process(cmd, **kwargs):
        assert cmd[:2] == ["gh", "api"]
        assert cmd[2].endswith("/git/blobs/" + session.sha)
        assert "Accept: application/vnd.github.raw+json" in cmd
        with TemporaryFile() as input_file:
            input_file.write(body)
            input_file.seek(0)
            return real_popen([sys.executable, "-c",
                               "import shutil,sys; shutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer)"],
                              stdin=input_file, **kwargs)

    monkeypatch.setattr(main.subprocess, "run", metadata)
    monkeypatch.setattr(main.subprocess, "Popen", raw_process)
    FakeToolClient.reset({"doc_id": "doc", "document_hash": "hash"})
    result = main._run_cloud_local_agent_job({
        "job_id": "laj-test", "attempt_count": 1, "workspace_id": "workspace-a",
        "operation": "github_repo_sync", "source_id": "src-test",
        "payload": {"repo_url": "https://github.com/example/repo", "ref": "main"},
    }, _cloud_test_client())
    if error:
        assert result["counts"]["failed"] == 1, result
        assert error in result["error"]
        assert not any(call[0] in ("push_github_repo_document", "start_source_processing")
                       for call in FakeToolClient.calls)
        return
    assert result["counts"]["pushed"] == 1, result
    [call] = [call for call in FakeToolClient.calls if call[0] == "push_github_repo_document"]
    assert call[1]["markdown_body"] == body.decode()
    assert call[1]["blob_sha"] == session.sha


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
async def test_cloud_collection_resolves_selected_text_symlink_at_pinned_tree(monkeypatch):
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
    monkeypatch.setattr("memforge.genes.github_repo_gene.requests.Session", lambda: session)
    gene = GitHubRepoGene(
        config={"connection_mode": "cloud_pull", "repo_url": "https://github.com/example/repo", "ref": "main"},
        source_id="src-test",
    )

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


@pytest.mark.asyncio
async def test_cloud_collection_rejects_text_symlink_to_binary_target(monkeypatch):
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
    monkeypatch.setattr("memforge.genes.github_repo_gene.requests.Session", lambda: session)
    gene = GitHubRepoGene(
        config={"connection_mode": "cloud_pull", "repo_url": "https://github.com/example/repo", "ref": "main"},
        source_id="src-test",
    )

    with pytest.raises(RuntimeError, match="binary Artifact"):
        [item async for item in gene.discover()]


def test_daemon_collection_resolves_selected_text_symlink_with_cloud_parity(monkeypatch):
    link_target = b"docs/CONTRIBUTIONS.md"
    link_sha = hashlib.sha1(
        b"blob " + str(len(link_target)).encode() + b"\0" + link_target
    ).hexdigest()
    target_body = b"# Contributions\n\nUse reviewed changes.\n"
    target_sha = hashlib.sha1(
        b"blob " + str(len(target_body)).encode() + b"\0" + target_body
    ).hexdigest()
    real_popen = subprocess.Popen

    def metadata(cmd, **kwargs):
        endpoint = cmd[2]
        if "/commits/main" in endpoint:
            payload = {"sha": "commit-one", "commit": {"tree": {"sha": "tree-one"}}}
        elif "/git/trees/" in endpoint:
            payload = {"truncated": False, "tree": [
                {"path": "README.md", "type": "blob", "mode": "120000",
                 "sha": link_sha, "size": len(link_target)},
                {"path": "docs/CONTRIBUTIONS.md", "type": "blob", "mode": "100644",
                 "sha": target_sha, "size": len(target_body)},
            ]}
        else:
            raise AssertionError(endpoint)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    def raw_process(cmd, **kwargs):
        if cmd[2].endswith("/git/blobs/" + link_sha):
            body = link_target
        elif cmd[2].endswith("/git/blobs/" + target_sha):
            body = target_body
        else:
            raise AssertionError(cmd[2])
        with TemporaryFile() as input_file:
            input_file.write(body)
            input_file.seek(0)
            return real_popen(
                [sys.executable, "-c", "import shutil,sys; shutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer)"],
                stdin=input_file,
                **kwargs,
            )

    monkeypatch.setattr(main.subprocess, "run", metadata)
    monkeypatch.setattr(main.subprocess, "Popen", raw_process)
    FakeToolClient.reset({"doc_id": "doc", "document_hash": "hash"})

    result = main._run_cloud_local_agent_job({
        "job_id": "laj-symlink", "attempt_count": 1, "workspace_id": "workspace-a",
        "operation": "github_repo_sync", "source_id": "src-test",
        "payload": {"repo_url": "https://github.com/example/repo", "ref": "main"},
    }, _cloud_test_client())

    assert result["counts"] == {"selected": 2, "reused": 0, "fetched": 2, "pushed": 2, "failed": 0}
    call = next(
        call for call in FakeToolClient.calls
        if call[0] == "push_github_repo_document" and call[1]["relative_path"] == "README.md"
    )
    assert call[1]["markdown_body"] == target_body.decode()
    assert call[1]["blob_sha"] == target_sha
    assert call[1]["resolved_relative_path"] == "docs/CONTRIBUTIONS.md"
    assert call[1]["symlink_chain"] == [{
        "path": "README.md", "blob_sha": link_sha, "target_path": "docs/CONTRIBUTIONS.md",
    }]


def test_daemon_collection_rejects_text_symlink_to_binary_before_manifest(monkeypatch):
    link_target = b"assets/photo.png"
    link_sha = hashlib.sha1(
        b"blob " + str(len(link_target)).encode() + b"\0" + link_target
    ).hexdigest()
    real_popen = subprocess.Popen

    def metadata(cmd, **kwargs):
        endpoint = cmd[2]
        if "/commits/main" in endpoint:
            payload = {"sha": "commit-one", "commit": {"tree": {"sha": "tree-one"}}}
        elif "/git/trees/" in endpoint:
            payload = {"truncated": False, "tree": [
                {"path": "README.md", "type": "blob", "mode": "120000",
                 "sha": link_sha, "size": len(link_target)},
                {"path": "assets/photo.png", "type": "blob", "mode": "100644",
                 "sha": "b" * 40, "size": 8},
            ]}
        else:
            raise AssertionError(endpoint)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    def raw_process(cmd, **kwargs):
        assert cmd[2].endswith("/git/blobs/" + link_sha)
        with TemporaryFile() as input_file:
            input_file.write(link_target)
            input_file.seek(0)
            return real_popen(
                [sys.executable, "-c", "import shutil,sys; shutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer)"],
                stdin=input_file,
                **kwargs,
            )

    monkeypatch.setattr(main.subprocess, "run", metadata)
    monkeypatch.setattr(main.subprocess, "Popen", raw_process)
    FakeToolClient.reset({})
    with pytest.raises(main.click.ClickException, match="binary Artifact"):
        main._run_cloud_local_agent_job({
            "job_id": "laj-binary-link", "attempt_count": 1, "workspace_id": "workspace-a",
            "operation": "github_repo_sync", "source_id": "src-test",
            "payload": {"repo_url": "https://github.com/example/repo", "ref": "main"},
        }, _cloud_test_client())
    assert not any(
        call[0] in {"prepare_local_source_snapshot", "start_source_processing"}
        for call in FakeToolClient.calls
    )

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


@pytest.mark.parametrize("failure", ["overflow", "timeout", "network", "http"])
def test_daemon_raw_transfer_failure_closes_process_and_never_processes(monkeypatch, failure):
    session = RepositorySession()
    real_popen = subprocess.Popen
    processes = []

    def metadata(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(
            session.get("https://api.github.com/" + cmd[2]).json()), stderr="")

    scripts = {
        "overflow": "import sys,time; sys.stdout.buffer.write(b'too long'); sys.stdout.flush(); time.sleep(60)",
        "timeout": "import time; time.sleep(60)",
        "network": "import sys; sys.stderr.write('error connecting to GitHub'); sys.exit(1)",
        "http": "import sys; sys.stderr.write('HTTP 403 Forbidden'); sys.exit(1)",
    }

    def raw_process(cmd, **kwargs):
        process = real_popen([sys.executable, "-c", scripts[failure]], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(main.subprocess, "run", metadata)
    monkeypatch.setattr(main.subprocess, "Popen", raw_process)
    monkeypatch.setattr(main, "GITHUB_RAW_TRANSFER_TIMEOUT_SECONDS", 0.5)
    FakeToolClient.reset({})
    job = {
        "job_id": "laj-error", "attempt_count": 1, "workspace_id": "workspace-a",
        "operation": "github_repo_sync", "source_id": "src-test",
        "payload": {"repo_url": "https://github.com/example/repo", "ref": "main"},
    }
    if failure in {"timeout", "network"}:
        # The runner owns the existing network-backoff classification.
        with pytest.raises(main.GitHubProviderConnectionError):
            main._run_cloud_local_agent_job(job, _cloud_test_client())
    else:
        result = main._run_cloud_local_agent_job(job, _cloud_test_client())
        assert result["counts"]["failed"] == 1
    [process] = processes
    assert process.poll() is not None
    assert process.stdout.closed and process.stderr.closed
    assert not any(call[0] == "start_source_processing" for call in FakeToolClient.calls)


def test_daemon_selected_gitlink_is_not_an_empty_complete_snapshot(monkeypatch):
    session = RepositorySession()
    session.mode = "160000"
    session.entry_type = "commit"

    def metadata(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(
            session.get("https://api.github.com/" + cmd[2]).json()), stderr="")

    monkeypatch.setattr(main.subprocess, "run", metadata)
    FakeToolClient.reset({})
    with pytest.raises(main.click.ClickException, match="unsupported file mode"):
        main._run_cloud_local_agent_job({
            "job_id": "laj-gitlink", "attempt_count": 1, "workspace_id": "workspace-a",
            "operation": "github_repo_sync", "source_id": "src-test",
            "payload": {"repo_url": "https://github.com/example/repo", "ref": "main",
                        "include_paths": ["README.md"]},
        }, _cloud_test_client())
    assert not any(call[0] in {"prepare_local_source_snapshot", "start_source_processing"}
                   for call in FakeToolClient.calls)


def test_daemon_malformed_symlink_is_not_an_empty_complete_snapshot(monkeypatch):
    session = RepositorySession()
    session.mode = "120000"
    session.entry_type = "tree"

    def metadata(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(
            session.get("https://api.github.com/" + cmd[2]).json()), stderr="")

    monkeypatch.setattr(main.subprocess, "run", metadata)
    FakeToolClient.reset({})
    with pytest.raises(main.click.ClickException, match="not represented by a blob"):
        main._run_cloud_local_agent_job({
            "job_id": "laj-malformed-symlink", "attempt_count": 1, "workspace_id": "workspace-a",
            "operation": "github_repo_sync", "source_id": "src-test",
            "payload": {"repo_url": "https://github.com/example/repo", "ref": "main",
                        "include_paths": ["README.md"]},
        }, _cloud_test_client())
    assert not any(call[0] in {"prepare_local_source_snapshot", "start_source_processing"}
                   for call in FakeToolClient.calls)
