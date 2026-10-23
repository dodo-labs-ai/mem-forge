"""A fake GitHub REST API for one repository, answered through ``httpx.MockTransport``.

Describe the repository by its files; the fake serves what a GitHub
connector reads, with real Git blob shas so blob reads verify::

    github = FakeGitHub(
        "https://github.example.test/api/v3/repos/payroll/architecture",
        files={"docs/guide.md": b"# Guide", "src/app.py": b"print()"},
        truncated={""},   # GitHub truncates the recursive tree of the root
    )
    with github.http.serving(github_repo_gene):
        await gene.authenticate()
    ...
    assert "GET .../git/trees/tree-main" in github.requests

It answers, under its API URL:

- the repository, naming ``default_branch``;
- ``commits/<branch>``, the branch's commit and root tree (``tree-main`` for
  ``main``);
- ``git/trees/<sha or branch>``, recursive or one level, truncated for the
  folders in ``truncated`` (``""`` is the root);
- ``commits?sha=…&path=…``, one commit at ``commit_times[path]`` or
  :data:`COMMIT_DATE`;
- ``git/blobs/<sha>``, the raw body;
- ``contents/<path>?ref=…``, the base64 body with its blob sha.

``answers`` replaces the answer to one request, keyed by its path and query
below the API URL, such as ``"/git/trees/tree-main?recursive=1"``: a
``httpx.Response``, or a JSON payload. Any other request is a 404.
``requests`` lists every request as ``"GET <url>"``.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterable, Mapping
from urllib.parse import parse_qs, unquote, urlsplit

import httpx

from tests.connector_http_fixture import FakeConnectorHttp

COMMIT_DATE = "2026-05-01T10:00:00Z"
SYMLINK_MODE = "120000"


def git_blob_sha(body: bytes) -> str:
    """The Git object id of a blob holding ``body``."""
    return hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()


class FakeGitHub:
    """One repository behind a fake GitHub REST API; see the module docstring."""

    def __init__(
        self,
        api_url: str,
        *,
        files: Mapping[str, bytes],
        symlinks: Iterable[str] = (),
        default_branch: str = "main",
        truncated: Iterable[str] = (),
        commit_times: Mapping[str, str] | None = None,
        answers: Mapping[str, object] | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.files = dict(files)
        self.symlinks = set(symlinks)
        self.default_branch = default_branch
        self.truncated = set(truncated)
        self.commit_times = dict(commit_times or {})
        self.answers = dict(answers or {})
        self.requests: list[str] = []
        self.http = FakeConnectorHttp(self.handle)
        self._folders = self._list_folders()
        self._folder_by_sha = {self.tree_sha(folder): folder for folder in self._folders}
        self._body_by_sha = {git_blob_sha(body): body for body in self.files.values()}

    # ------------------------------------------------------------------
    # The repository's shape
    # ------------------------------------------------------------------

    def tree_sha(self, folder: str) -> str:
        """The tree sha of ``folder``; the root of branch ``main`` is ``tree-main``."""
        if not folder:
            return f"tree-{self.default_branch}"
        return hashlib.sha1(f"tree {folder}".encode()).hexdigest()

    def _list_folders(self) -> list[str]:
        found = {""}
        for path in self.files:
            parts = path.split("/")[:-1]
            found.update("/".join(parts[:index]) for index in range(1, len(parts) + 1))
        return sorted(found)

    def children(self, folder: str) -> list[dict]:
        """The entries directly inside ``folder``, paths relative to it."""
        prefix = f"{folder}/" if folder else ""
        entries: list[dict] = []
        for child in self._folders:
            if child and child.startswith(prefix) and "/" not in child[len(prefix):]:
                entries.append({"path": child[len(prefix):], "type": "tree", "sha": self.tree_sha(child), "mode": "040000"})
        for path, body in sorted(self.files.items()):
            if path.startswith(prefix) and "/" not in path[len(prefix):]:
                entries.append({
                    "path": path[len(prefix):],
                    "type": "blob",
                    "sha": git_blob_sha(body),
                    "mode": SYMLINK_MODE if path in self.symlinks else "100644",
                    "size": len(body),
                })
        return entries

    def recursive(self, folder: str) -> list[dict]:
        """Every entry below ``folder``, paths relative to it, folders before their contents."""
        entries: list[dict] = []
        for entry in self.children(folder):
            entries.append(entry)
            if entry["type"] == "tree":
                child = f"{folder}/{entry['path']}" if folder else entry["path"]
                entries.extend({**nested, "path": f"{entry['path']}/{nested['path']}"} for nested in self.recursive(child))
        return entries

    def entry(self, path: str) -> dict:
        """The blob entry of ``path`` as a full tree lists it."""
        folder, _, name = path.rpartition("/")
        [found] = [child for child in self.children(folder) if child["path"] == name]
        return {**found, "path": path}

    # ------------------------------------------------------------------
    # Answering
    # ------------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(f"{request.method} {request.url}")
        answer = self.answer(request)
        return answer if answer is not None else httpx.Response(404, json={"message": "Not Found"})

    def answer(self, request: httpx.Request) -> httpx.Response | None:
        """The answer to a request under the API URL, or ``None`` for any other URL."""
        url = str(request.url)
        if not url.startswith(self.api_url):
            return None
        below = url[len(self.api_url):]
        if below in self.answers:
            answer = self.answers[below]
            if isinstance(answer, httpx.Response):
                # A fresh copy each time, since the client reads and closes every response it gets.
                return httpx.Response(answer.status_code, headers=answer.headers, stream=answer.stream)
            return httpx.Response(200, json=answer)
        parts = urlsplit(below)
        path, query = unquote(parts.path), parse_qs(parts.query)
        if path == "":
            return httpx.Response(200, json={"default_branch": self.default_branch})
        if path == "/commits":
            changed = self.commit_times.get(query["path"][0], COMMIT_DATE)
            return httpx.Response(200, json=[{"sha": "file-commit", "commit": {"committer": {"date": changed}}}])
        if path.startswith("/commits/"):
            ref = path.removeprefix("/commits/")
            if ref != self.default_branch:
                return None
            return httpx.Response(200, json={"sha": f"commit-{ref}", "commit": {"tree": {"sha": self.tree_sha("")}}})
        if path.startswith("/git/trees/"):
            return self._tree(path.removeprefix("/git/trees/"), recursive="recursive" in query)
        if path.startswith("/git/blobs/"):
            body = self._body(path.removeprefix("/git/blobs/"))
            if body is None:
                return None
            return httpx.Response(200, content=body, headers={"content-type": "application/octet-stream"})
        if path.startswith("/contents/"):
            repo_path = path.removeprefix("/contents/")
            if repo_path not in self.files:
                return None
            body = self.files[repo_path]
            return httpx.Response(200, json={
                "sha": git_blob_sha(body),
                "content": base64.b64encode(body).decode(),
                "encoding": "base64",
                "size": len(body),
            })
        return None

    def _tree(self, tree_ish: str, *, recursive: bool) -> httpx.Response | None:
        if tree_ish == self.default_branch:
            tree_ish = self.tree_sha("")
        folder = self._folder_by_sha.get(tree_ish)
        if folder is None:
            return None
        if recursive and folder in self.truncated:
            return httpx.Response(200, json={"sha": tree_ish, "truncated": True, "tree": self.recursive(folder)[:1]})
        entries = self.recursive(folder) if recursive else self.children(folder)
        return httpx.Response(200, json={"sha": tree_ish, "truncated": False, "tree": entries})

    def _body(self, blob_sha: str) -> bytes | None:
        return self._body_by_sha.get(blob_sha)
