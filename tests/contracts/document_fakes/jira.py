"""A fake Jira: issues in projects PAY and OPS, and a JQL search that understands the suite's queries.

PAY-1, PAY-2 and PAY-4 carry the label ``payroll``; PAY-3 does not; OPS-1 is
in another project. PAY-4 was edited in the cursor's minute. The simple
configuration selects project PAY, the advanced one ``project = PAY AND
labels = payroll``. Issue 100009 is gone.

The search answers a scope query only when it carries the configured where
clause, filtered by an ``updated >=`` clause when present; ``issuekey in
(...)`` by issue id or key; and ``issuekey = X AND (<where>)`` with issue X
when it exists and is in scope; any other single-issue search is rejected,
as Jira rejects an ORDER BY inside the parentheses.

Like Jira, the fake reads an ``updated >= 'YYYY-MM-DD HH:MM'`` literal as a
wall-clock minute in the account's profile timezone, which
``/rest/api/2/myself`` answers as ``timeZone``, and answers each issue's
``updated`` in that timezone. A minute that names two instants around a
daylight-saving change is read as the later one, the stricter reading. The
profile is west of UTC by default, where a literal written in UTC hides edits.
"""

from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx

from memforge.genes import jira_gene
from memforge.genes.jira_gene import JiraGene
from memforge.models import ContentItem
from tests.connector_http_fixture import FakeConnectorHttp
from tests.contracts.document_fakes import (
    IN_THE_CURSOR_MINUTE,
    LONG_AGO,
    RECENTLY,
    DocumentConnectorFakes,
    FakeConnection,
    ListingConfig,
    ProviderListing,
)

BASE_URL = "https://jira.example.test"
PROFILE_TIMEZONE = "America/Los_Angeles"

# issue key -> (issue id, labels, updated)
ISSUES: dict[str, tuple[str, tuple[str, ...], datetime]] = {
    "PAY-1": ("100001", ("payroll",), LONG_AGO),
    "PAY-2": ("100002", ("payroll",), RECENTLY),
    "PAY-3": ("100003", (), RECENTLY),
    "OPS-1": ("100004", ("payroll",), RECENTLY),
    "PAY-4": ("100005", ("payroll",), IN_THE_CURSOR_MINUTE),
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


def _issue(key: str, profile_zone: ZoneInfo) -> dict:
    issue_id, labels, updated = ISSUES[key]
    local = updated.astimezone(profile_zone)
    return {
        "id": issue_id,
        "key": key,
        "fields": {
            "summary": key,
            # As Jira writes it: "2026-03-01T01:30:40.000-0800".
            "updated": f"{local:%Y-%m-%dT%H:%M:%S}.000{local:%z}",
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
    def __init__(
        self,
        listing: ProviderListing,
        read_status: int | None = None,
        profile_timezone: str = PROFILE_TIMEZONE,
    ) -> None:
        self.listing = listing
        self.read_status = read_status
        self.profile_zone = ZoneInfo(profile_timezone)
        self.requests: list[str] = []
        self.http = FakeConnectorHttp(self.handle)

    def connection(self, gene: JiraGene) -> FakeConnection:
        return FakeConnection(gene=gene, requests=self.requests, clients=self.http.clients, waits=self.http.waits)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(f"{request.method} {request.url.path}")
        if request.url.path == "/rest/api/2/search":
            return self._search(json.loads(request.content))
        if request.url.path == "/rest/api/2/myself":
            return httpx.Response(200, json={"name": "contract", "timeZone": self.profile_zone.key})
        ref = request.url.path.removeprefix("/rest/api/2/issue/")
        if self.read_status is not None:
            # A wait over the client's cap on 429, and none on 503, which then backs off.
            headers = {"Retry-After": "600"} if self.read_status == httpx.codes.TOO_MANY_REQUESTS else {}
            body = self._issue("PAY-1") if self.read_status == 200 else {"errorMessages": []}
            return httpx.Response(self.read_status, json=body, headers=headers)
        key = _by_ref(ref)
        if key is None:
            return httpx.Response(404, json={"errorMessages": ["Issue Does Not Exist"]})
        return httpx.Response(200, json=self._issue(key))

    def _search(self, body: dict) -> httpx.Response:
        jql = str(body["jql"])
        start = int(body["startAt"])
        if (refs := re.fullmatch(r"issuekey in \((.*)\)", jql)) is not None:
            if self.read_status is not None:
                return _page([], start, 0)
            keys = [key for ref in refs.group(1).split(", ") if (key := _by_ref(ref)) is not None]
            return _page([self._issue(key) for key in keys], start, len(keys))
        where = next((where for where in _SCOPES if where in jql), None)
        if where is None:
            return httpx.Response(400, json={"errorMessages": [f"unexpected JQL: {jql}"]})
        if jql.startswith("issuekey = "):
            # Jira rejects anything but a bare where clause inside the parentheses, such as an ORDER BY.
            single = re.fullmatch(rf"issuekey = (\S+) AND \({re.escape(where)}\)", jql)
            if single is None:
                return httpx.Response(400, json={"errorMessages": [f"invalid JQL: {jql}"]})
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
                and (since is None or ISSUES[key][2] >= self._jql_time(since.group(1)))
            ]
        if self.listing is ProviderListing.TRUNCATED:
            # Jira declares every match but stops after the first.
            return _page([self._issue(key) for key in matched[:1]] if start == 0 else [], start, len(matched))
        page_size = int(body["maxResults"])
        return _page([self._issue(key) for key in matched[start:start + page_size]], start, len(matched))

    def _issue(self, key: str) -> dict:
        return _issue(key, self.profile_zone)

    def _jql_time(self, value: str) -> datetime:
        """A JQL date literal as Jira reads it: a minute in the profile timezone, at its later reading."""
        wall = datetime.strptime(value, "%Y-%m-%d %H:%M")
        return max(wall.replace(tzinfo=self.profile_zone, fold=fold).astimezone(timezone.utc) for fold in (0, 1))


def _page(issues: list[dict], start: int, total: int) -> httpx.Response:
    return httpx.Response(200, json={"startAt": start, "total": total, "issues": issues})


class JiraFakes(DocumentConnectorFakes):
    # Retry-After: 600 is capped at 60 s; a 503 without it backs off 2 s.
    retry_waits = {429: (60.0, 60.0), 503: (2.0, 2.0)}
    listing_configs = {
        "simple": ListingConfig(SIMPLE, walks_scope=False, edited_in_the_cursor_minute="jira-PAY-4"),
        "advanced": ListingConfig(ADVANCED, walks_scope=False, edited_in_the_cursor_minute="jira-PAY-4"),
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
            yield provider.connection(gene)
        finally:
            await gene.close()

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
            for key in ("PAY-1", "PAY-2", "PAY-4")
        ]

    def rediscover_candidates(self, config: Mapping[str, object]) -> list[ContentItem]:
        return [
            *(_stored_issue(key, issue_id) for key, (issue_id, *_rest) in ISSUES.items()),
            _stored_issue(*GONE_ISSUE),
        ]

    @asynccontextmanager
    async def reading_by_id(self, status: int) -> AsyncIterator[tuple[FakeConnection, Sequence[ContentItem]]]:
        provider = _Provider(ProviderListing.COMPLETE, read_status=status)
        gene = await _authenticated(dict(SIMPLE), provider)
        try:
            yield provider.connection(gene), [_stored_issue("PAY-1", "100001")]
        finally:
            await gene.close()


async def _authenticated(config: dict, provider: _Provider) -> JiraGene:
    gene = JiraGene(config=config, source_id="src-contract")
    with provider.http.serving(jira_gene):
        await gene.authenticate()
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
