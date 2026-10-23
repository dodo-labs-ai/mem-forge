"""Jira writes an incremental search's date literal in the account's profile timezone.

Jira reads ``updated >= '2026-03-01 01:30'`` as a wall-clock minute in the
searching account's profile timezone, which ``GET /rest/api/2/myself``
answers. These tests drive the real ``JiraGene`` against a fake Jira and read
the JQL it sends.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
import pytest

from memforge.genes import jira_gene
from memforge.genes.jira_gene import JiraGene, _delta_clause, _readable_no_later_than
from tests.connector_http_fixture import FakeConnectorHttp

BASE_URL = "https://jira.example.test"
CONFIG = {"base_url": BASE_URL, "auth_mode": "pat", "pat": "token", "projects": ["PAY"], "issue_types": ["Story"]}
# Ten seconds into a minute, as an incremental cursor usually is.
SINCE = datetime(2026, 3, 1, 9, 30, 10, tzinfo=timezone.utc)
NEW_YORK = ZoneInfo("America/New_York")


@pytest.fixture(autouse=True)
def own_origin_limiters(monkeypatch):
    """Give each test its own Atlassian origin limiters, so no pacing carries over from another test."""
    monkeypatch.setattr("memforge.genes.atlassian_auth._ATLASSIAN_REQUEST_LIMITERS", {})


class FakeJira:
    """Answers ``/myself`` with ``profile`` (or ``myself``, a response or an exception), and every search with no issues."""

    def __init__(self, profile: dict | None = None, *, myself: httpx.Response | Exception | None = None) -> None:
        self.profile = profile if profile is not None else {"name": "me", "timeZone": "America/Los_Angeles"}
        self.myself = myself
        self.requests: list[str] = []
        self.jql: list[str] = []
        self.http = FakeConnectorHttp(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(f"{request.method} {request.url.path}")
        if request.url.path == "/rest/api/2/myself":
            if isinstance(self.myself, Exception):
                raise self.myself
            return self.myself if self.myself is not None else httpx.Response(200, json=self.profile)
        body = json.loads(request.content)
        self.jql.append(body["jql"])
        return httpx.Response(200, json={"startAt": body["startAt"], "total": 0, "issues": []})

    @property
    def myself_reads(self) -> int:
        return self.requests.count("GET /rest/api/2/myself")

    async def gene(self, config: dict | None = None) -> JiraGene:
        gene = JiraGene(config=dict(config or CONFIG), source_id="src-jira")
        with self.http.serving(jira_gene):
            await gene.authenticate()
        return gene


async def _incremental_jql(jira: FakeJira, since: datetime = SINCE) -> str:
    gene = await jira.gene()
    try:
        _ = [item async for item in gene.discover(since)]
    finally:
        await gene.close()
    return jira.jql[-1]


# ---------------------------------------------------------------------------
# The literal is the cursor in the profile timezone, rounded down to the minute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_profile_west_of_utc_reads_the_cursor_minute_in_its_own_time():
    jql = await _incremental_jql(FakeJira({"timeZone": "America/Los_Angeles"}))

    # 09:30:10 UTC is 01:30:10 PST. Written as '09:30', Jira would read 09:30
    # PST, 17:30 UTC, and hide eight hours of edits.
    assert "updated >= '2026-03-01 01:30'" in jql


@pytest.mark.asyncio
async def test_a_profile_east_of_utc_reads_the_cursor_minute_in_its_own_time():
    jql = await _incremental_jql(FakeJira({"timeZone": "Asia/Kolkata"}))

    # 09:30:10 UTC is 15:00:10 IST (UTC+05:30).
    assert "updated >= '2026-03-01 15:00'" in jql


@pytest.mark.asyncio
async def test_a_utc_profile_rounds_the_cursor_down_to_its_minute():
    jql = await _incremental_jql(FakeJira({"timeZone": "UTC"}))

    assert "updated >= '2026-03-01 09:30'" in jql


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "profile",
    [{"name": "me"}, {"name": "me", "timeZone": None}, {"name": "me", "timeZone": ""}, {"timeZone": "Mars/Olympus"}],
    ids=["missing", "null", "empty", "unknown"],
)
async def test_a_profile_without_a_known_timezone_writes_the_literal_at_utc_minus_12(profile, caplog):
    with caplog.at_level(logging.WARNING, logger="memforge.genes.jira_gene"):
        jql = await _incremental_jql(FakeJira(profile))

    # 09:30:10 UTC is 21:30:10 the day before at UTC-12. Any profile timezone
    # reads that minute as 09:30 UTC or earlier, so nothing is hidden.
    assert "updated >= '2026-02-28 21:30'" in jql
    assert "UTC-12" in caplog.text


# ---------------------------------------------------------------------------
# Daylight-saving changes never make Jira read a later instant than the cursor
# ---------------------------------------------------------------------------


def test_an_ambiguous_minute_at_the_earlier_cursor_steps_back_an_hour():
    # 05:30:20 UTC is 01:30:20 EDT on 2026-11-01; 01:30 also names 06:30 UTC
    # (EST), which would hide an hour, so the literal steps back to 00:30 EDT.
    since = datetime(2026, 11, 1, 5, 30, 20, tzinfo=timezone.utc)

    assert _delta_clause(since, NEW_YORK) == "updated >= '2026-11-01 00:30'"


def test_an_ambiguous_minute_at_the_later_cursor_is_kept():
    # 06:30:20 UTC is 01:30:20 EST, the second 01:30; both readings of 01:30
    # (05:30 and 06:30 UTC) are at or before it.
    since = datetime(2026, 11, 1, 6, 30, 20, tzinfo=timezone.utc)

    assert _delta_clause(since, NEW_YORK) == "updated >= '2026-11-01 01:30'"


def test_a_non_existent_minute_steps_back_an_hour():
    # 02:30 on 2026-03-08 never happens in New York; Jira may move it forward to
    # 03:30 EDT, 07:30 UTC, after 07:00 UTC, so it steps back to 01:30 EST.
    instant = datetime(2026, 3, 8, 7, 0, tzinfo=timezone.utc)

    assert _readable_no_later_than(datetime(2026, 3, 8, 2, 30), NEW_YORK, instant) == datetime(2026, 3, 8, 1, 30)


@pytest.mark.parametrize(
    ("since", "literal"),
    [
        (datetime(2026, 3, 8, 6, 59, 30, tzinfo=timezone.utc), "2026-03-08 01:59"),  # 01:59:30 EST
        (datetime(2026, 3, 8, 7, 0, 30, tzinfo=timezone.utc), "2026-03-08 03:00"),  # 03:00:30 EDT
    ],
    ids=["before", "after"],
)
def test_a_cursor_around_clocks_going_forward_names_an_existing_minute(since, literal):
    assert _delta_clause(since, NEW_YORK) == f"updated >= '{literal}'"


# ---------------------------------------------------------------------------
# /myself: read once per connection, only for an incremental search
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_profile_is_read_once_per_connection_and_again_after_authenticate_or_close():
    jira = FakeJira()
    gene = await jira.gene()
    try:
        assert jira.requests == []  # authenticate asks nothing

        _ = [item async for item in gene.discover(SINCE)]
        _ = [item async for item in gene.discover(SINCE)]
        assert jira.myself_reads == 1

        with jira.http.serving(jira_gene):
            await gene.authenticate()
        _ = [item async for item in gene.discover(SINCE)]
        assert jira.myself_reads == 2

        await gene.close()
        with jira.http.serving(jira_gene):
            await gene.authenticate()
        _ = [item async for item in gene.discover(SINCE)]
        assert jira.myself_reads == 3
    finally:
        await gene.close()


@pytest.mark.asyncio
async def test_a_search_without_since_never_reads_the_profile():
    jira = FakeJira()
    gene = await jira.gene()
    try:
        _ = [item async for item in gene.discover(None)]
        _ = [item async for item in gene.discover_inventory()]
        await gene.list_scope()
    finally:
        await gene.close()

    assert jira.myself_reads == 0
    assert all("updated >=" not in jql for jql in jira.jql)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "myself",
    [
        httpx.Response(500, json={"errorMessages": ["Internal server error"]}),
        httpx.Response(400, json={"errorMessages": ["Bad request"]}),
        httpx.Response(200, text="<html>not json</html>"),
        httpx.Response(200, json=["not", "a", "profile"]),
        httpx.ConnectError("connection refused"),
    ],
    ids=["server-error", "bad-request", "not-json", "not-a-profile", "unreachable"],
)
async def test_a_failed_profile_read_raises_runtime_error(myself):
    jira = FakeJira(myself=myself)
    gene = await jira.gene()
    try:
        with pytest.raises(RuntimeError) as exc_info:
            _ = [item async for item in gene.discover(SINCE)]
    finally:
        await gene.close()

    # Neither a refused configuration (ValueError) nor a configuration error
    # (SourceConfigurationError, a RuntimeError subclass): the run is retried.
    assert type(exc_info.value) is RuntimeError
    assert "/myself" in str(exc_info.value)
    assert jira.jql == []


@pytest.mark.asyncio
async def test_an_expired_browser_session_on_myself_asks_for_a_fresh_cookie():
    jira = FakeJira(myself=httpx.Response(401, json={}))
    config = {**CONFIG, "auth_mode": "browser_cookie", "jira_cookie": "JSESSIONID=expired"}
    config.pop("pat")
    gene = await jira.gene(config)
    try:
        with pytest.raises(RuntimeError, match="Refresh the cookie") as exc_info:
            _ = [item async for item in gene.discover(SINCE)]
    finally:
        await gene.close()

    assert type(exc_info.value) is RuntimeError
