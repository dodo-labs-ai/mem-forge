"""A fake Jira: issues in projects PAY and OPS, and a JQL search that understands the suite's queries.

PAY-1 and PAY-2 carry the label ``payroll``; PAY-3 does not; OPS-1 is in
another project. The simple configuration selects project PAY, the advanced
one ``project = PAY AND labels = payroll``. Issue 100009 is gone.

The search answers a scope query only when it carries the configured where
clause, filtered by an ``updated >=`` clause when present; ``issuekey in
(...)`` by issue id or key; and ``issuekey = X AND (<where>)`` with issue X
when it exists and is in scope.
"""

from __future__ import annotations

import functools
import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from unittest.mock import patch

import httpx

from memforge.genes.jira_gene import JiraGene
from memforge.models import ContentItem
from tests.contracts.document_fakes import (
    LONG_AGO,
    RECENTLY,
    DocumentConnectorFakes,
    FakeConnection,
    ListingConfig,
    ProviderListing,
)

BASE_URL = "https://jira.example.test"
_REAL_ASYNC_CLIENT = httpx.AsyncClient

# issue key -> (issue id, labels, updated)
ISSUES: dict[str, tuple[str, tuple[str, ...], datetime]] = {
    "PAY-1": ("100001", ("payroll",), LONG_AGO),
    "PAY-2": ("100002", ("payroll",), RECENTLY),
    "PAY-3": ("100003", (), RECENTLY),
    "OPS-1": ("100004", ("payroll",), RECENTLY),
}
GONE_ISSUE = ("PAY-9", "100009")

_CONNECTION = {"base_url": BASE_URL, "auth_mode": "pat", "pat": "token"}
SIMPLE = {**_CONNECTION, "projects": ["PAY"], "issue_types": ["Story"]}
ADVANCED = {**_CONNECTION, "query_mode": "advanced", "jql": "project = PAY AND labels = payroll ORDER BY Rank ASC"}

# The where clause each configuration's JQL carries, and the issues it selects.
_SCOPES = {
    "project in (PAY) AND issuetype in (Story)": lambda key: key.startswith("PAY-"),
    "project = PAY AND labels = payroll": lambda key: key.startswith("PAY-") and "payroll" in ISSUES[key][1],
}
_UPDATED_SINCE = re.compile(r"updated >= '(\d{4}-\d{2}-\d{2} \d{2}:\d{2})'")


def _issue(key: str) -> dict:
    issue_id, labels, updated = ISSUES[key]
    return {
        "id": issue_id,
        "key": key,
        "fields": {
            "summary": key,
            "updated": updated.isoformat(),
            "project": {"key": key.split("-", 1)[0]},
            "issuetype": {"name": "Story"},
            "labels": list(labels),
            "status": None,
            "priority": None,
            "assignee": None,
        },
    }


def _by_ref(ref: str) -> str | None:
    return next((key for key, (issue_id, *_rest) in ISSUES.items() if ref in (key, issue_id)), None)


class _Provider:
    def __init__(self, listing: ProviderListing, read_status: int | None = None) -> None:
        self.listing = listing
        self.read_status = read_status
        self.requests: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(f"{request.method} {request.url.path}")
        if request.url.path == "/rest/api/2/search":
            return self._search(json.loads(request.content))
        ref = request.url.path.removeprefix("/rest/api/2/issue/")
        if self.read_status is not None:
            headers = {"Retry-After": "0"} if self.read_status == 429 else {}
            body = _issue("PAY-1") if self.read_status == 200 else {"errorMessages": []}
            return httpx.Response(self.read_status, json=body, headers=headers)
        key = _by_ref(ref)
        if key is None:
            return httpx.Response(404, json={"errorMessages": ["Issue Does Not Exist"]})
        return httpx.Response(200, json=_issue(key))

    def _search(self, body: dict) -> httpx.Response:
        jql = str(body["jql"])
        start = int(body["startAt"])
        if (refs := re.fullmatch(r"issuekey in \((.*)\)", jql)) is not None:
            if self.read_status is not None:
                return _page([], start, 0)
            keys = [key for ref in refs.group(1).split(", ") if (key := _by_ref(ref)) is not None]
            return _page([_issue(key) for key in keys], start, len(keys))
        where = next((where for where in _SCOPES if where in jql), None)
        if where is None:
            return httpx.Response(400, json={"errorMessages": [f"unexpected JQL: {jql}"]})
        if (single := re.fullmatch(r"issuekey = (\S+) AND \((.*)\)", jql)) is not None:
            key = _by_ref(single.group(1))
            matched = [key] if key is not None and _SCOPES[where](key) else []
        else:
            if self.listing is ProviderListing.FAILED:
                return httpx.Response(500, json={"errorMessages": ["Internal server error"]})
            since = _UPDATED_SINCE.search(jql)
            matched = [
                key
                for key in ISSUES
                if _SCOPES[where](key)
                and (since is None or ISSUES[key][2] >= _jql_time(since.group(1)))
            ]
        if self.listing is ProviderListing.TRUNCATED:
            # Jira declares every match but stops after the first.
            return _page([_issue(key) for key in matched[:1]] if start == 0 else [], start, len(matched))
        page_size = int(body["maxResults"])
        return _page([_issue(key) for key in matched[start:start + page_size]], start, len(matched))


def _jql_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


def _page(issues: list[dict], start: int, total: int) -> httpx.Response:
    return httpx.Response(200, json={"startAt": start, "total": total, "issues": issues})


class JiraFakes(DocumentConnectorFakes):
    listing_configs = {
        "simple": ListingConfig(SIMPLE, walks_scope=False),
        "advanced": ListingConfig(ADVANCED, walks_scope=False),
    }
    device_configs = {
        "local-agent": {"base_url": BASE_URL, "auth_mode": "browser_cookie", "sync_mode": "local_agent", "projects": ["PAY"]},
    }

    @asynccontextmanager
    async def connect(
        self,
        config: Mapping[str, object],
        listing: ProviderListing = ProviderListing.COMPLETE,
    ) -> AsyncIterator[FakeConnection]:
        provider = _Provider(listing)
        gene = await _authenticated(dict(config), provider)
        try:
            yield FakeConnection(gene=gene, requests=provider.requests)
        finally:
            await gene._client.aclose()

    def manifest_entries(self, config: Mapping[str, object]) -> list[dict]:
        return [
            {
                "doc_id": f"jira-{key}",
                "issue_key": key,
                "title": key,
                "version": ISSUES[key][2].isoformat(),
                "last_modified": ISSUES[key][2].isoformat(),
                "change_kind": "upsert",
            }
            for key in ("PAY-1", "PAY-2")
        ]

    def rediscover_candidates(self, config: Mapping[str, object]) -> list[ContentItem]:
        return [
            *(_stored_issue(key, issue_id) for key, (issue_id, *_rest) in ISSUES.items()),
            _stored_issue(*GONE_ISSUE),
        ]

    @asynccontextmanager
    async def reading_by_id(self, status: int) -> AsyncIterator[tuple[JiraGene, Sequence[ContentItem]]]:
        provider = _Provider(ProviderListing.COMPLETE, read_status=status)
        gene = await _authenticated(dict(SIMPLE), provider)
        try:
            yield gene, [_stored_issue("PAY-1", "100001")]
        finally:
            await gene._client.aclose()


async def _authenticated(config: dict, provider: _Provider) -> JiraGene:
    gene = JiraGene(config=config, source_id="src-contract")
    client = functools.partial(_REAL_ASYNC_CLIENT, transport=httpx.MockTransport(provider.handle))
    with patch("httpx.AsyncClient", client):
        await gene.authenticate()
    gene._request_limiter = None
    return gene


def _stored_issue(key: str, issue_id: str) -> ContentItem:
    return ContentItem(
        item_id=f"jira-{key}",
        title=key,
        source_url=f"{BASE_URL}/browse/{key}",
        last_modified=LONG_AGO,
        extra={"issue_id": issue_id, "issue_key": key},
    )


FAKES = JiraFakes()
