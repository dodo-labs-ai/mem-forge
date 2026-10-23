"""Read a GitHub repository through its REST API: default branch, commit, complete tree and file commit times.

A connector builds its :class:`~memforge.genes.toolkit.http.ConnectorHttpClient`
in ``authenticate`` with GitHub's rate-limit rule, then reads one repository
through :class:`GitHubApi`::

    from memforge.genes.toolkit.github_api import GitHubApi, GitHubRepository, github_rate_limit_wait
    from memforge.genes.toolkit.http import ConnectorHttpClient

    self._client = ConnectorHttpClient(
        product_name="GitHub",
        headers={"Accept": "application/vnd.github+json"},
        timeout=30.0,
        follow_redirects=True,
        rate_limit=github_rate_limit_wait,
    )
    github = GitHubApi(self._client, GitHubRepository("https://github.com", "octo-org", "handbook"))
    branch = await github.default_branch()             # "main"
    commit = await github.commit(branch)               # GitHubCommit(sha="9fb0…", tree_sha="4b82…")
    entries = await github.tree(commit.tree_sha)       # every blob, tree and submodule, with full paths
    changed = await github.latest_commit_time("docs/setup.md", ref=commit.sha)  # "2026-05-01T10:00:00+00:00"

**Where the API lives.** github.com serves it at ``https://api.github.com``;
GitHub Enterprise Server serves it at ``<origin>/api/v3``. Owner, repository,
refs and paths are percent-encoded into the URL, so a branch such as
``release/2026`` is one path segment.

**A complete tree.** ``tree`` returns every entry of the repository, or
raises. GitHub answers a recursive tree request with at most 100,000 entries
(and 7 MB), and says ``truncated: true`` when it left some out. The walk then
reads that tree one level down and asks for each child folder's recursive
tree, going a further level down only into a child that is itself truncated.
A repository with one oversized ``vendor/`` folder costs::

    GET git/trees/<root>?recursive=1          truncated
    GET git/trees/<root>                      the root's own entries
    GET git/trees/<docs>?recursive=1          complete
    GET git/trees/<vendor>?recursive=1        truncated
    GET git/trees/<vendor>                    vendor's own entries
    GET git/trees/<vendor/a>?recursive=1      complete
    ...

The whole repository is walked even when a Source selects only some folders,
because a selected symbolic link resolves anywhere in the complete tree
(ADR 0011). Two bounds keep a walk finite, and both raise:

- ``max_requests`` caps the tree requests of one walk
  (:class:`GitHubTreeBudgetExceeded`). A sync allows
  :data:`TREE_REQUEST_BUDGET`; the admin folder browse, which a person waits
  for, allows :data:`BROWSE_TREE_REQUEST_BUDGET`.
- ``selects`` with ``max_selected`` stops the walk as soon as more entries
  are selected than the Source allows (:class:`GitHubTreeSelectionExceeded`),
  so an oversized scope fails before the rest of the repository is read.

**Rate limits.** GitHub answers an exhausted rate limit with 403 or 429.
:func:`github_rate_limit_wait` tells the client how long to wait: the
``Retry-After`` seconds, or the time until ``x-ratelimit-reset`` when
``x-ratelimit-remaining`` is 0, or a minute for a secondary rate limit that
names neither. A wait within :data:`RATE_LIMIT_MAX_WAIT_SECONDS` is waited out
and retried; a longer one raises :class:`GitHubRateLimitError`, which says
when the limit resets. A 403 without these signals is a refusal and is
returned to the connector unchanged.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from memforge.genes.toolkit.http import DEFAULT_RETRY_POLICY, ConnectorHttpClient, retry_after_seconds
from memforge.github_repo_utils import (
    github_latest_commit_time,
    github_path_commits_query,
    normalize_github_relative_path,
    validate_github_tree_payload,
)

__all__ = [
    "BROWSE_TREE_REQUEST_BUDGET",
    "RATE_LIMIT_MAX_WAIT_SECONDS",
    "TREE_REQUEST_BUDGET",
    "GitHubApi",
    "GitHubCommit",
    "GitHubRateLimitError",
    "GitHubRepository",
    "GitHubTreeBudgetExceeded",
    "GitHubTreeSelectionExceeded",
    "github_rate_limit_wait",
]

TREE_REQUEST_BUDGET = 500
"""Tree requests one sync may spend reading a truncated tree.

A tenth of the 5,000 REST requests an hour GitHub grants an authenticated
user, which leaves the rest of the hour's quota for the per-file commit and
blob reads that follow. A walk costs one recursive request for each folder
it reads, plus one one-level listing for each truncated folder, so only a
repository past GitHub's 100,000-entry limit whose truncated folders hold
hundreds of folders reaches it.
"""

BROWSE_TREE_REQUEST_BUDGET = 20
"""Tree requests the admin folder browse may spend, so a person waits seconds, not minutes.

It reads a truncated root and up to 18 child folders; a larger repository is
scoped by typing its paths.
"""

RATE_LIMIT_MAX_WAIT_SECONDS = DEFAULT_RETRY_POLICY.max_wait_seconds
"""The longest rate-limit wait a request waits out; the client's cap on any wait."""

# GitHub's advice for a secondary rate limit that names no wait: at least a minute.
_SECONDARY_RATE_LIMIT_WAIT_SECONDS = 60.0
_RATE_LIMIT_STATUSES = frozenset({httpx.codes.FORBIDDEN, httpx.codes.TOO_MANY_REQUESTS})


class GitHubRateLimitError(RuntimeError):
    """GitHub's rate limit resets later than a request waits; the message says when."""


class GitHubTreeBudgetExceeded(RuntimeError):
    """Reading a truncated tree needs more requests than the walk may spend."""


class GitHubTreeSelectionExceeded(RuntimeError):
    """The walk selected more entries than allowed; ``selected`` counts them when it stopped.

    ``exact`` says whether that is every entry the tree selects: it is for a
    tree GitHub returned whole, and a lower bound for one read folder by
    folder, whose walk stops at the first folder that passes the limit::

        GitHubTreeSelectionExceeded(4, 3, exact=False)  # "selects at least 4 entries, exceeding 3"
    """

    def __init__(self, selected: int, max_selected: int, *, exact: bool) -> None:
        count = str(selected) if exact else f"at least {selected}"
        super().__init__(f"The GitHub tree selects {count} entries, exceeding {max_selected}")
        self.selected = selected
        self.max_selected = max_selected
        self.exact = exact


@dataclass(frozen=True)
class GitHubRepository:
    """One repository on github.com or a GitHub Enterprise Server, named by its web origin::

        GitHubRepository("https://github.com", "octo-org", "handbook").api_url
        # "https://api.github.com/repos/octo-org/handbook"
        GitHubRepository("https://github.example.com", "payroll", "docs").api_url
        # "https://github.example.com/api/v3/repos/payroll/docs"
    """

    origin: str
    owner: str
    name: str

    @property
    def api_url(self) -> str:
        """The repository's REST API root."""
        repo_path = f"repos/{quote(self.owner, safe='')}/{quote(self.name, safe='')}"
        if urlsplit(self.origin).netloc.lower() == "github.com":
            return f"https://api.github.com/{repo_path}"
        return f"{self.origin.rstrip('/')}/api/v3/{repo_path}"

    @property
    def full_name(self) -> str:
        """``owner/name``, as GitHub writes it."""
        return f"{self.owner}/{self.name}"

    def blob_url(self, blob_sha: str) -> str:
        """The Git Blobs URL of one immutable blob."""
        return f"{self.api_url}/git/blobs/{quote(blob_sha, safe='')}"


@dataclass(frozen=True)
class GitHubCommit:
    """A commit and the root tree it pins."""

    sha: str
    tree_sha: str


class GitHubApi:
    """Reads one repository through a connector's client; see the module docstring.

    ``label`` names the repository in validation errors, such as "GitHub
    Pages repository". Every request goes through ``client``, so the client's
    retries, TLS, redirects and rate-limit rule apply to each.
    """

    def __init__(
        self,
        client: ConnectorHttpClient,
        repository: GitHubRepository,
        *,
        label: str = "GitHub repository",
    ) -> None:
        self.client = client
        self.repository = repository
        self.label = label

    async def default_branch(self) -> str:
        """The repository's default branch, such as ``"main"``; raises when GitHub names none."""
        payload = await _json(self.client, self.repository.api_url)
        branch = str(payload.get("default_branch") or "").strip() if isinstance(payload, dict) else ""
        if not branch:
            raise RuntimeError(f"{self.label} response is missing default_branch")
        return branch

    async def commit(self, ref: str) -> GitHubCommit:
        """The commit ``ref`` (a branch, tag or sha) names now, with its root tree; raises when either is missing."""
        payload = await _json(self.client, f"{self.repository.api_url}/commits/{quote(ref, safe='')}")
        payload = payload if isinstance(payload, dict) else {}
        commit = payload.get("commit")
        tree = commit.get("tree") if isinstance(commit, dict) else None
        commit_sha = str(payload.get("sha") or "")
        tree_sha = str(tree.get("sha") or "") if isinstance(tree, dict) else ""
        if not commit_sha or not tree_sha:
            raise RuntimeError("GitHub did not return an immutable commit and root tree for this collection")
        return GitHubCommit(sha=commit_sha, tree_sha=tree_sha)

    async def tree(
        self,
        tree_ish: str,
        *,
        max_requests: int = TREE_REQUEST_BUDGET,
        selects: Callable[[Mapping[str, Any]], bool] | None = None,
        max_selected: int | None = None,
    ) -> list[dict[str, Any]]:
        """Every entry of the tree ``tree_ish`` names (a tree sha, or a branch, tag or commit), with full paths.

        Each entry is GitHub's own (``path``, ``type``, ``sha``, ``mode``,
        and ``size`` for a blob), its ``path`` relative to the repository
        root. A truncated tree is walked folder by folder (module docstring)::

            entries = await github.tree(commit.tree_sha, selects=is_markdown, max_selected=500)

        Raises ``RuntimeError`` when a response is not a complete, valid
        tree, :class:`GitHubTreeBudgetExceeded` after ``max_requests`` tree
        requests, and :class:`GitHubTreeSelectionExceeded` once ``selects``
        accepts more than ``max_selected`` entries.
        """
        walk = _TreeWalk(self, max_requests=max_requests, selects=selects, max_selected=max_selected)
        return await walk.entries(tree_ish, prefix="")

    async def latest_commit_time(self, path: str, *, ref: str) -> str | None:
        """When the latest commit reachable from ``ref`` changed ``path``, in ISO 8601 UTC.

        ``None`` when GitHub lists no such commit. A refused request raises
        ``httpx.HTTPStatusError``, so the connector decides what an unknown
        time means::

            await github.latest_commit_time("docs/setup.md", ref="main")  # "2026-05-01T10:00:00+00:00"
        """
        query = github_path_commits_query(relative_path=path, ref=ref)
        return github_latest_commit_time(await _json(self.client, f"{self.repository.api_url}/commits?{query}"))


async def _json(client: ConnectorHttpClient, url: str) -> object:
    response = await client.get(url)
    response.raise_for_status()
    return response.json()


class _TreeWalk:
    """One bounded walk of a tree: its request count and how many entries it selected so far."""

    def __init__(
        self,
        github: GitHubApi,
        *,
        max_requests: int,
        selects: Callable[[Mapping[str, Any]], bool] | None,
        max_selected: int | None,
    ) -> None:
        self.github = github
        self.max_requests = max_requests
        self.selects = selects
        self.max_selected = max_selected
        self.requests = 0
        self.selected = 0
        self.truncated = False
        self.paths: set[str] = set()

    async def entries(self, tree_ish: str, *, prefix: str) -> list[dict[str, Any]]:
        """The entries under ``tree_ish``, their paths prefixed by the folder ``prefix`` it sits at."""
        recursive = await self.read(tree_ish, recursive=True)
        if not (isinstance(recursive, Mapping) and recursive.get("truncated") is True):
            return self.keep(recursive, prefix, one_level=False)
        self.truncated = True
        entries: list[dict[str, Any]] = []
        for entry in self.keep(await self.read(tree_ish, recursive=False), prefix, one_level=True):
            entries.append(entry)
            if entry.get("type") == "tree":
                entries.extend(await self.entries(str(entry["sha"]), prefix=str(entry["path"])))
        return entries

    async def read(self, tree_ish: str, *, recursive: bool) -> object:
        if self.requests >= self.max_requests:
            raise GitHubTreeBudgetExceeded(
                f"GitHub truncated the tree of {self.github.repository.full_name}, and reading it folder by folder "
                f"needs more than {self.max_requests} requests"
            )
        self.requests += 1
        url = f"{self.github.repository.api_url}/git/trees/{quote(tree_ish, safe='')}"
        return await _json(self.github.client, f"{url}?recursive=1" if recursive else url)

    def keep(self, payload: object, prefix: str, *, one_level: bool) -> list[dict[str, Any]]:
        """Validate one complete tree response, root its paths at ``prefix``, and count what it selects.

        A one-level listing names each entry by its bare name, and no path
        may appear twice anywhere in the walk; either fault fails closed.
        """
        label = self.github.label
        try:
            entries = validate_github_tree_payload(payload, label=label)
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
        rooted: list[dict[str, Any]] = []
        for entry in entries:
            name = normalize_github_relative_path(str(entry["path"])).rstrip("/")
            if one_level and "/" in name:
                raise RuntimeError(f"{label} tree entry has an invalid path")
            path = f"{prefix}/{name}" if prefix else name
            if path in self.paths:
                raise RuntimeError(f"{label} tree response contains duplicate path {path!r}")
            self.paths.add(path)
            rooted.append({**entry, "path": path} if prefix else entry)
        if self.selects is not None and self.max_selected is not None:
            self.selected += sum(1 for entry in rooted if self.selects(entry))
            if self.selected > self.max_selected:
                raise GitHubTreeSelectionExceeded(self.selected, self.max_selected, exact=not self.truncated)
        return rooted


def github_rate_limit_wait(response: httpx.Response) -> float | None:
    """GitHub's rate-limit rule for :class:`ConnectorHttpClient`'s ``rate_limit`` hook.

    For a 403 or 429 it returns the seconds GitHub asks to wait, in GitHub's
    order:

    - ``Retry-After``;
    - with ``x-ratelimit-remaining: 0``, the time until ``x-ratelimit-reset``
      (epoch seconds);
    - for a body that names a rate limit, such as a secondary rate limit,
      one minute.

    It raises :class:`GitHubRateLimitError` when that wait is longer than
    :data:`RATE_LIMIT_MAX_WAIT_SECONDS`, and returns ``None`` for any other
    response, a 403 that refuses access included. The body of a streamed
    response is not read here, so a secondary rate limit that a streamed
    blob read signals only in its body is not detected and fails as a
    refusal::

        github_rate_limit_wait(httpx.Response(403, headers={"Retry-After": "30"}))   # 30.0
        github_rate_limit_wait(httpx.Response(403, json={"message": "Resource not accessible"}))  # None
    """
    if response.status_code not in _RATE_LIMIT_STATUSES:
        return None
    now = datetime.now(timezone.utc)
    wait = retry_after_seconds(response)
    if wait is not None:
        resets_at = now + timedelta(seconds=wait)
    elif (reset := _exhausted_quota_reset(response)) is not None:
        resets_at = reset
        wait = max((reset - now).total_seconds(), 0.0)
    elif _names_a_rate_limit(response):
        wait = _SECONDARY_RATE_LIMIT_WAIT_SECONDS
        resets_at = now + timedelta(seconds=wait)
    else:
        return None
    if wait > RATE_LIMIT_MAX_WAIT_SECONDS:
        raise GitHubRateLimitError(
            f"GitHub rate limit reached{_for_request(response)}; it resets at {resets_at:%Y-%m-%d %H:%M:%S} UTC, later than a sync waits "
            f"({RATE_LIMIT_MAX_WAIT_SECONDS:.0f} s). Retry after that time."
        )
    return wait


def _for_request(response: httpx.Response) -> str:
    """`` for GET <url>`` naming the request a response answers, or nothing for a response built without one."""
    try:
        request = response.request
    except RuntimeError:
        return ""
    return f" for {request.method} {request.url}"


def _exhausted_quota_reset(response: httpx.Response) -> datetime | None:
    """When an exhausted primary quota resets, or ``None`` when the quota is not exhausted or the reset unreadable."""
    if response.headers.get("x-ratelimit-remaining", "").strip() != "0":
        return None
    reset = response.headers.get("x-ratelimit-reset", "").strip()
    if not (reset.isascii() and reset.isdigit()):
        return None
    try:
        return datetime.fromtimestamp(int(reset), tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _names_a_rate_limit(response: httpx.Response) -> bool:
    """Whether a body already read says it is a rate limit; a streamed body is never read here."""
    try:
        body = response.content
    except httpx.ResponseNotRead:
        return False
    try:
        payload = json.loads(body)
    except ValueError:
        return False
    message = payload.get("message") if isinstance(payload, dict) else None
    return isinstance(message, str) and "rate limit" in message.lower()
