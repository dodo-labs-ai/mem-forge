"""github_api: where the REST API lives, how a truncated tree is walked and bounded, and GitHub's rate-limit rule."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx
import pytest

from memforge.genes.toolkit.github_api import (
    RATE_LIMIT_MAX_WAIT_SECONDS,
    GitHubApi,
    GitHubRateLimitError,
    GitHubRepository,
    GitHubTreeBudgetExceeded,
    GitHubTreeSelectionExceeded,
    github_rate_limit_wait,
)
from tests.github_api_fixture import FakeGitHub

API = "https://github.example.test/api/v3/repos/payroll/architecture"
REPOSITORY = GitHubRepository("https://github.example.test", "payroll", "architecture")
FILES = {
    "README.md": b"# Architecture\n",
    "docs/guide.md": b"# Guide\n",
    "docs/runbooks/restart.md": b"# Restart\n",
    "vendor/lib-a/index.js": b"a\n",
    "vendor/lib-a/util/strings.js": b"s\n",
    "vendor/lib-b/index.js": b"b\n",
}


def _api(github: FakeGitHub) -> GitHubApi:
    client = github.http.client("", product_name="GitHub", rate_limit=github_rate_limit_wait)
    return GitHubApi(client, REPOSITORY)


def _tree_requests(github: FakeGitHub) -> list[str]:
    """Each tree request as ``<folder>`` or ``<folder> recursive``; the root is ``/``."""
    folder_by_sha = {github.tree_sha(folder): folder or "/" for folder in github._folders}
    described = []
    for request in github.requests:
        if "/git/trees/" not in request:
            continue
        tree_ish, _, query = request.split("/git/trees/", 1)[1].partition("?")
        described.append(f"{folder_by_sha[tree_ish]}{' recursive' if query else ''}")
    return described


def _paths(entries: list[dict]) -> list[str]:
    return sorted(entry["path"] for entry in entries)


# ---------------------------------------------------------------------------
# Where the API lives
# ---------------------------------------------------------------------------


def test_github_com_is_read_at_api_github_com_and_an_enterprise_server_at_api_v3():
    assert GitHubRepository("https://github.com", "octo-org", "handbook").api_url == (
        "https://api.github.com/repos/octo-org/handbook"
    )
    assert GitHubRepository("https://github.example.com:8443", "payroll", "docs").api_url == (
        "https://github.example.com:8443/api/v3/repos/payroll/docs"
    )
    assert REPOSITORY.blob_url("abc123") == f"{API}/git/blobs/abc123"


@pytest.mark.asyncio
async def test_a_ref_in_the_path_is_one_percent_encoded_segment():
    github = FakeGitHub(API, files=FILES, default_branch="release/2026")
    api = _api(github)

    await api.commit("release/2026")
    await api.tree("release/2026")
    await api.latest_commit_time("docs/runbooks/restart notes.md", ref="release/2026")

    assert github.requests == [
        f"GET {API}/commits/release%2F2026",
        f"GET {API}/git/trees/release%2F2026?recursive=1",
        # A query may carry "/" as it is.
        f"GET {API}/commits?sha=release/2026&path=docs/runbooks/restart%20notes.md&per_page=1",
    ]


@pytest.mark.asyncio
async def test_the_default_branch_commit_and_commit_time_are_read_from_the_api():
    github = FakeGitHub(API, files=FILES, commit_times={"docs/guide.md": "2026-06-02T08:00:00Z"})
    api = _api(github)

    assert await api.default_branch() == "main"
    assert (await api.commit("main")).tree_sha == "tree-main"
    assert await api.latest_commit_time("docs/guide.md", ref="commit-main") == "2026-06-02T08:00:00+00:00"


@pytest.mark.asyncio
async def test_a_commit_time_github_refuses_raises_and_one_it_does_not_list_is_none():
    refused = httpx.Response(403, json={"message": "Resource not accessible by integration"})
    github = FakeGitHub(
        API,
        files=FILES,
        answers={
            "/commits?sha=main&path=secret.md&per_page=1": refused,
            "/commits?sha=main&path=new.md&per_page=1": [],
        },
    )
    api = _api(github)

    with pytest.raises(httpx.HTTPStatusError):
        await api.latest_commit_time("secret.md", ref="main")
    assert await api.latest_commit_time("new.md", ref="main") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "read", "message"),
    [
        ({"name": "architecture"}, "default_branch", "missing default_branch"),
        ({"sha": "commit-main", "commit": {}}, "commit", "immutable commit and root tree"),
    ],
)
async def test_a_repository_or_commit_without_its_identity_raises(answer, read, message):
    github = FakeGitHub(API, files=FILES, answers={"": answer, "/commits/main": answer})

    with pytest.raises(RuntimeError, match=message):
        await (_api(github).default_branch() if read == "default_branch" else _api(github).commit("main"))


# ---------------------------------------------------------------------------
# A complete tree
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_complete_tree_is_read_in_one_request():
    github = FakeGitHub(API, files=FILES)

    entries = await _api(github).tree("tree-main")

    assert entries == github.recursive("")
    assert _tree_requests(github) == ["/ recursive"]


@pytest.mark.asyncio
async def test_a_truncated_tree_is_read_by_the_recursive_trees_of_its_folders():
    github = FakeGitHub(API, files=FILES, truncated={""})

    entries = await _api(github).tree("tree-main")

    assert _paths(entries) == _paths(github.recursive(""))
    assert {entry["path"]: entry["sha"] for entry in entries} == {
        entry["path"]: entry["sha"] for entry in github.recursive("")
    }
    assert _tree_requests(github) == ["/ recursive", "/", "docs recursive", "vendor recursive"]


@pytest.mark.asyncio
async def test_a_folder_that_is_itself_truncated_is_read_level_by_level_and_only_that_folder():
    github = FakeGitHub(API, files=FILES, truncated={"", "vendor"})

    entries = await _api(github).tree("tree-main")

    assert _paths(entries) == _paths(github.recursive(""))
    assert _tree_requests(github) == [
        "/ recursive",
        "/",
        "docs recursive",  # complete: read whole, never level by level
        "vendor recursive",
        "vendor",
        "vendor/lib-a recursive",
        "vendor/lib-b recursive",
    ]


@pytest.mark.asyncio
async def test_a_folder_listing_github_truncates_fails_closed():
    level_truncated = {"sha": "tree-main", "truncated": True, "tree": []}
    github = FakeGitHub(API, files=FILES, truncated={""}, answers={"/git/trees/tree-main": level_truncated})

    with pytest.raises(RuntimeError, match="did not attest truncated=false"):
        await _api(github).tree("tree-main")


@pytest.mark.asyncio
async def test_a_folder_listing_with_a_duplicate_path_fails_closed():
    duplicate = {
        "truncated": False,
        "tree": [
            {"path": "index.js", "type": "blob", "sha": "sha-a", "mode": "100644"},
            {"path": "index.js", "type": "blob", "sha": "sha-b", "mode": "100644"},
        ],
    }
    github = FakeGitHub(API, files=FILES, truncated={""})
    github.answers[f"/git/trees/{github.tree_sha('vendor')}?recursive=1"] = duplicate

    with pytest.raises(RuntimeError, match="duplicate path 'index.js'"):
        await _api(github).tree("tree-main")


@pytest.mark.asyncio
async def test_a_one_level_listing_that_names_a_nested_path_fails_closed():
    # A malformed listing of the root: "docs/guide.md" beside the folder "docs" would be read twice.
    nested = {"truncated": False, "tree": [*FakeGitHub(API, files=FILES).children(""), {
        "path": "docs/guide.md", "type": "blob", "sha": "sha-guide", "mode": "100644",
    }]}
    github = FakeGitHub(API, files=FILES, truncated={""}, answers={"/git/trees/tree-main": nested})

    with pytest.raises(RuntimeError, match="tree entry has an invalid path"):
        await _api(github).tree("tree-main")


@pytest.mark.asyncio
async def test_a_submodule_is_kept_as_an_entry_and_never_walked_into():
    github = FakeGitHub(API, files=FILES, truncated={"", "vendor"}, submodules={"vendor/sdk": "c0ffee", "themes": "decade"})

    entries = await _api(github).tree("tree-main")

    submodules = {entry["path"]: entry["sha"] for entry in entries if entry["type"] == "commit"}
    assert submodules == {"vendor/sdk": "c0ffee", "themes": "decade"}
    assert not any(sha in request for sha in ("c0ffee", "decade") for request in github.requests)


@pytest.mark.asyncio
async def test_a_folder_named_with_url_characters_is_read_and_rooted_by_its_own_name():
    odd = "notes #1? 100% done"
    github = FakeGitHub(API, files={**FILES, f"{odd}/plan.md": b"# Plan\n", f"{odd}/deep/more.md": b"# More\n"}, truncated={""})

    entries = await _api(github).tree("tree-main")

    assert {f"{odd}/plan.md", f"{odd}/deep", f"{odd}/deep/more.md"} <= {entry["path"] for entry in entries}
    assert _paths(entries) == _paths(github.recursive(""))


@pytest.mark.asyncio
async def test_a_folder_github_cannot_find_during_a_walk_fails_the_walk():
    github = FakeGitHub(API, files=FILES, truncated={""})
    github.answers[f"/git/trees/{github.tree_sha('vendor')}?recursive=1"] = httpx.Response(404, json={"message": "Not Found"})

    with pytest.raises(httpx.HTTPStatusError, match="404"):
        await _api(github).tree("tree-main")


@pytest.mark.asyncio
async def test_a_walk_that_needs_more_requests_than_its_budget_fails_closed():
    github = FakeGitHub(API, files=FILES, truncated={"", "vendor"})

    with pytest.raises(GitHubTreeBudgetExceeded, match=r"payroll/architecture.*more than 4 requests"):
        await _api(github).tree("tree-main", max_requests=4)

    assert len(_tree_requests(github)) == 4


@pytest.mark.asyncio
async def test_a_walk_stops_at_the_first_folder_that_selects_more_than_allowed():
    files = {f"area-{index}/{name}.md": b"# Doc\n" for index in range(5) for name in ("a", "b")}
    github = FakeGitHub(API, files=files, truncated={""})

    with pytest.raises(GitHubTreeSelectionExceeded) as raised:
        await _api(github).tree(
            "tree-main",
            selects=lambda entry: entry["type"] == "blob",
            max_selected=3,
        )

    assert (raised.value.selected, raised.value.max_selected, raised.value.exact) == (4, 3, False)
    assert "at least 4" in str(raised.value)
    assert _tree_requests(github) == ["/ recursive", "/", "area-0 recursive", "area-1 recursive"]


@pytest.mark.asyncio
async def test_a_complete_tree_counts_every_selected_entry():
    files = {f"docs/{index}.md": b"# Doc\n" for index in range(5)}
    github = FakeGitHub(API, files=files)

    with pytest.raises(GitHubTreeSelectionExceeded) as raised:
        await _api(github).tree("tree-main", selects=lambda entry: entry["type"] == "blob", max_selected=3)

    assert (raised.value.selected, raised.value.exact) == (5, True)


# ---------------------------------------------------------------------------
# Rate limits
# ---------------------------------------------------------------------------


def _reset_in(seconds: float) -> str:
    return str(int((datetime.now(timezone.utc) + timedelta(seconds=seconds)).timestamp()))


def test_a_403_or_429_with_retry_after_waits_that_long():
    assert github_rate_limit_wait(httpx.Response(403, headers={"Retry-After": "30"})) == 30.0
    assert github_rate_limit_wait(httpx.Response(429, headers={"Retry-After": "5"})) == 5.0


def test_an_exhausted_quota_waits_until_it_resets():
    wait = github_rate_limit_wait(
        httpx.Response(403, headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": _reset_in(20)})
    )
    assert wait is not None and 15 <= wait <= 20


def test_a_secondary_rate_limit_that_names_no_wait_waits_a_minute():
    response = httpx.Response(403, json={"message": "You have exceeded a secondary rate limit."})
    assert github_rate_limit_wait(response) == 60.0


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(403, json={"message": "Resource not accessible by integration"}),
        httpx.Response(403, headers={"x-ratelimit-remaining": "12", "x-ratelimit-reset": _reset_in(3600)}),
        httpx.Response(404, headers={"Retry-After": "5"}),
        httpx.Response(429),
    ],
    ids=["refusal", "quota-left", "other-status", "bare-429"],
)
def test_a_refusal_or_another_status_is_not_a_github_rate_limit(response):
    assert github_rate_limit_wait(response) is None


@pytest.mark.parametrize(
    "headers",
    [
        {"x-ratelimit-remaining": "0", "x-ratelimit-reset": _reset_in(3600)},
        {"Retry-After": str(int(RATE_LIMIT_MAX_WAIT_SECONDS) + 60)},
    ],
    ids=["quota-resets-later", "retry-after-longer"],
)
def test_a_rate_limit_longer_than_the_cap_raises_with_its_reset_time(headers):
    request = httpx.Request("GET", f"{API}/git/trees/tree-main")
    with pytest.raises(GitHubRateLimitError, match=r"GitHub rate limit reached for GET .*; it resets at .* UTC"):
        github_rate_limit_wait(httpx.Response(403, headers=headers, request=request))


@pytest.mark.asyncio
async def test_the_client_waits_out_a_github_rate_limit_and_retries():
    limited = httpx.Response(403, headers={"Retry-After": "7"}, json={"message": "secondary rate limit"})
    answers = iter([limited])
    github = FakeGitHub(API, files=FILES)

    def handle(request: httpx.Request) -> httpx.Response:
        return next(answers, None) or github.handle(request)

    github.http.transport = httpx.MockTransport(handle)

    assert await _api(github).default_branch() == "main"
    assert github.http.waits == [7.0]


@pytest.mark.asyncio
async def test_a_rate_limit_that_persists_raises_as_a_rate_limit_not_a_refusal():
    limited = httpx.Response(403, headers={"Retry-After": "1"}, json={"message": "API rate limit exceeded"})
    github = FakeGitHub(API, files=FILES, answers={"": limited})

    with pytest.raises(httpx.HTTPStatusError, match="GitHub rate limit persisted after 4 attempts") as raised:
        await _api(github).default_branch()

    assert raised.value.response.status_code == 403
    assert github.http.waits == [1.0, 1.0, 1.0]


@pytest.mark.asyncio
async def test_a_permission_refusal_is_returned_once_and_raised_by_the_reader():
    refused = httpx.Response(403, json={"message": "Must have admin rights to Repository."})
    github = FakeGitHub(API, files=FILES, answers={"": refused})

    with pytest.raises(httpx.HTTPStatusError, match="403 Forbidden") as raised:
        await _api(github).default_branch()

    assert "rate limit" not in str(raised.value).lower()
    assert len(github.requests) == 1
    assert github.http.waits == []
