"""Read Notion's public API: the requests, the response shapes and the data-source query windows.

:class:`NotionApi` sends every request through the Gene's
:class:`~memforge.genes.toolkit.http.ConnectorHttpClient`, which carries the
pinned ``Notion-Version``, the connection's shared request rate and the
retry policy. It answers in Notion's own terms (pages, blocks, databases and
data sources) and leaves what an answer means for the listing to the Gene::

    api = NotionApi(client)
    response = await api.get(f"/v1/pages/{page_id}")
    if response.status_code == 404:
        ...                                   # the Gene decides
    page = notion_object(response, "page", page_id)

    async for block in api.block_children(page_id):
        ...

Notion ids are UUIDs. Responses write them with dashes; MemForge keys them as
32 lowercase hex digits (:func:`notion_id`), which the API also accepts.
"""

from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from memforge.genes.toolkit.http import ConnectorHttpClient
from memforge.genes.toolkit.paging import CursorPage, PagingErrors, paged_by_cursor
from memforge.source_time import utc_minute_floor

__all__ = [
    "NOTION_API_ORIGIN",
    "NOTION_REQUESTS_PER_MINUTE",
    "NOTION_VERSION",
    "NotionApi",
    "NotionMarkdown",
    "NotionRequestBlockedError",
    "notion_blocked_integration",
    "notion_error_code",
    "notion_id",
    "notion_link_id",
    "notion_object",
    "notion_parent",
    "notion_time",
]

NOTION_API_ORIGIN = "https://api.notion.com"
NOTION_VERSION = "2026-03-11"
"""The ``Notion-Version`` every request sends. Under it, ``in_trash`` marks a
trashed object and ``is_archived`` an archived page; ``archived`` is a
deprecated alias of ``in_trash`` and is never read."""
NOTION_REQUESTS_PER_MINUTE = 180
"""Notion's average budget per connection on most plans: three requests a second."""

_PAGE_SIZE = 100
_RESULT_LIMIT_REACHED = "query_result_limit_reached"
_REQUEST_BLOCKED = "public_api_request_blocked"
_DASHED_OR_PLAIN_ID = re.compile(r"^[0-9a-fA-F]{8}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{4}-?[0-9a-fA-F]{12}$")
# The id that ends a link's last path segment, after a ``Title-`` slug or alone.
_TRAILING_ID = re.compile(
    r"(?:^|-)([0-9a-fA-F]{32}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)


@dataclass(frozen=True)
class NotionMarkdown:
    """One answer of the Markdown endpoint for a page or block.

    ``truncated`` is Notion's own flag, and ``unknown_block_ids`` the 32-hex
    ids it names for blocks the answer left out, each written in
    ``markdown`` as an ``<unknown>`` tag. What they mean for the page's
    coverage is the Gene's to decide.
    """

    markdown: str
    truncated: bool
    unknown_block_ids: tuple[str, ...]


class NotionRequestBlockedError(RuntimeError):
    """Notion blocked this connection's requests and says never to retry them."""


def notion_id(value: object) -> str | None:
    """A Notion id as 32 lowercase hex digits, or ``None`` when ``value`` is not one::

        notion_id("1f3c2a4b-5d6e-4f70-8a9b-0c1d2e3f4a5b")  # "1f3c2a4b5d6e4f708a9b0c1d2e3f4a5b"
        notion_id("1F3C2A4B5D6E4F708A9B0C1D2E3F4A5B")      # "1f3c2a4b5d6e4f708a9b0c1d2e3f4a5b"
        notion_id("Roadmap")                               # None
    """
    text = str(value or "").strip()
    if not _DASHED_OR_PLAIN_ID.match(text):
        return None
    return text.replace("-", "").lower()


def notion_link_id(value: object, *, open_page: bool) -> str | None:
    """The 32-hex id a pasted Notion link or id names, or ``None`` when it names none.

    The id is read from the link's last path segment, after an optional
    ``Title-`` slug (a slug alone reads too), whatever the host:
    ``www.notion.so``, ``notion.so``, ``<team>.notion.site`` and
    ``app.notion.com`` links read alike, and a link without a scheme reads as
    one with ``https://``. With ``open_page``, a page open over a database
    view (``?p=<id>``) is the page named; a view (``?v=<id>``) is never one::

        notion_link_id("https://www.notion.so/team/Roadmap-1f3c2a4b5d6e4f708a9b0c1d2e3f4a5b", open_page=True)
        # "1f3c2a4b5d6e4f708a9b0c1d2e3f4a5b"
        notion_link_id("https://app.notion.com/p/9c0d…?v=5e6f…&p=1f3c…", open_page=True)   # "1f3c…"
        notion_link_id("https://app.notion.com/p/9c0d…?v=5e6f…&p=1f3c…", open_page=False)  # "9c0d…"
        notion_link_id("1f3c2a4b-5d6e-4f70-8a9b-0c1d2e3f4a5b", open_page=False)            # "1f3c2a4b…"
        notion_link_id("Roadmap", open_page=True)                                         # None
    """
    text = str(value or "").strip()
    if (bare := notion_id(text)) is not None:
        return bare
    if "/" in text or "?" in text:
        link = urlsplit(text if "://" in text else f"https://{text}")
        if open_page:
            for opened in parse_qs(link.query).get("p", []):
                if (page_id := notion_id(opened)) is not None:
                    return page_id
        segment = link.path.rstrip("/").rsplit("/", 1)[-1]
    else:
        segment = text
    match = _TRAILING_ID.search(segment)
    return match.group(1).replace("-", "").lower() if match else None


def notion_time(value: object, what: str) -> datetime:
    """An aware time from a Notion ISO 8601 field such as ``last_edited_time``; ``what`` words the error."""
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Notion {what} has no time")
    try:
        moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as exc:
        raise RuntimeError(f"Notion {what} has an unreadable time {value!r}") from exc
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise RuntimeError(f"Notion {what} time {value!r} has no offset")
    return moment


def notion_parent(record: dict[str, Any], what: str) -> tuple[str, str | None]:
    """The physical parent of a page, block, database or data source, as ``(kind, id32)``.

    ``kind`` is ``"page"``, ``"block"``, ``"database"``, ``"data_source"`` or
    ``"workspace"`` (with id ``None``). A row's parent names its data source
    and that data source's database; the database is the parent, because a
    data source lives in exactly one::

        notion_parent({"parent": {"type": "page_id", "page_id": "1f3c…"}}, "page")   # ("page", "1f3c…")
        notion_parent({"parent": {"type": "data_source_id", "data_source_id": "9c…",
                                  "database_id": "b7…"}}, "row")                   # ("database", "b7…")
    """
    parent = record.get("parent")
    if not isinstance(parent, dict):
        raise RuntimeError(f"Notion {what} has no parent")
    parent_type = parent.get("type")
    if parent_type == "workspace":
        return "workspace", None
    if parent_type == "data_source_id" and notion_id(parent.get("database_id")):
        return "database", notion_id(parent.get("database_id"))
    kinds = {"page_id": "page", "block_id": "block", "database_id": "database", "data_source_id": "data_source"}
    kind = kinds.get(str(parent_type))
    parent_id = notion_id(parent.get(str(parent_type))) if kind else None
    if kind is None or parent_id is None:
        raise RuntimeError(f"Notion {what} has a parent of an unknown shape: {parent_type!r}")
    return kind, parent_id


def notion_object(response: httpx.Response, object_type: str, object_id: str) -> dict[str, Any]:
    """The object a successful read returned, checked to be ``object_type`` with id ``object_id``.

    An error status raises ``httpx.HTTPStatusError``; a body of another shape
    or identity raises ``RuntimeError``.
    """
    response.raise_for_status()
    record = _json_object(response, f"{object_type} {object_id}")
    if record.get("object") != object_type or notion_id(record.get("id")) != notion_id(object_id):
        raise RuntimeError(f"Notion answered the read of {object_type} {object_id} with another object")
    return record


def notion_error_code(response: httpx.Response) -> str | None:
    """The ``code`` of a Notion error body, such as ``"validation_error"``, or ``None`` when it names none."""
    return _error_code_and_reason(response)[0]


def notion_blocked_integration(response: httpx.Response) -> None:
    """The ``fail_fast`` hook: stop at once when Notion answers 429 ``public_api_request_blocked``."""
    if response.status_code != httpx.codes.TOO_MANY_REQUESTS:
        return
    code, reason = _error_code_and_reason(response)
    if _REQUEST_BLOCKED in (code, reason):
        raise NotionRequestBlockedError(
            "Notion blocked this connection's requests (public_api_request_blocked) and asks not to "
            "retry them. Check the connection in Notion's integration settings, then sync again."
        )


class NotionApi:
    """The Notion requests a Gene sends, through one ``ConnectorHttpClient``; see the module docstring."""

    def __init__(self, client: ConnectorHttpClient) -> None:
        self._client = client

    async def get(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        """``GET`` ``path``; an error status is returned, and an exhausted 429 names Notion's reason."""
        return await self._send("GET", path, params=params)

    async def block_children(self, block_id: str) -> AsyncIterator[dict[str, Any]]:
        """Every child block of a page or block, a hundred at a time, checked to be blocks.

        A 404 raises ``httpx.HTTPStatusError`` like any other error status, so
        the caller can tell it apart.
        """

        async def fetch(cursor: str | None) -> CursorPage[dict[str, Any]]:
            params: dict[str, Any] = {"page_size": _PAGE_SIZE}
            if cursor:
                params["start_cursor"] = cursor
            response = await self.get(f"/v1/blocks/{block_id}/children", params=params)
            response.raise_for_status()
            return _cursor_page(_json_object(response, f"children of block {block_id}"), f"children of block {block_id}")

        async for block in paged_by_cursor(fetch, errors=_paging_errors(f"children of block {block_id}")):
            if block.get("object") != "block" or notion_id(block.get("id")) is None or not block.get("type"):
                raise RuntimeError(f"Notion listed a child of block {block_id} that is not a block")
            yield block

    async def page_markdown(self, block_id: str) -> NotionMarkdown | None:
        """The enhanced Markdown of a page, or of one block read again, or ``None`` when Notion answers 404.

        The Markdown version is pinned (``markdown_version=v1``), and meeting
        note transcripts are left out (``include_transcript=false``), rather
        than left to Notion's defaults. Any other error status raises
        ``httpx.HTTPStatusError``; an answer of another shape raises
        ``RuntimeError``.
        """
        response = await self.get(
            f"/v1/pages/{block_id}/markdown",
            params={"markdown_version": "v1", "include_transcript": "false"},
        )
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        response.raise_for_status()
        data = _json_object(response, f"Markdown of {block_id}")
        markdown, truncated, unknown = data.get("markdown"), data.get("truncated"), data.get("unknown_block_ids")
        if not isinstance(markdown, str) or not isinstance(truncated, bool) or not isinstance(unknown, list):
            raise RuntimeError(f"Notion answered the read of the Markdown of {block_id} in another shape")
        ids = tuple(notion_id(value) for value in unknown)
        if any(value is None for value in ids):
            raise RuntimeError(f"Notion named an unknown block of {block_id} without a block id")
        return NotionMarkdown(markdown=markdown, truncated=truncated, unknown_block_ids=ids)  # type: ignore[arg-type]

    async def property_items(self, page_id: str, property_id: str) -> AsyncIterator[dict[str, Any]]:
        """Every item of one page property, a hundred at a time.

        A page object holds at most 25 references of a ``title``,
        ``rich_text``, ``people`` or ``relation`` value; this reads them
        all. Each item names its value under its ``type``, as
        ``{"type": "relation", "relation": {"id": "…"}}``.
        """
        what = f"property {property_id} of page {page_id}"

        async def fetch(cursor: str | None) -> CursorPage[dict[str, Any]]:
            params: dict[str, Any] = {"page_size": _PAGE_SIZE}
            if cursor:
                params["start_cursor"] = cursor
            response = await self.get(f"/v1/pages/{page_id}/properties/{property_id}", params=params)
            response.raise_for_status()
            return _cursor_page(_json_object(response, what), what)

        async for item in paged_by_cursor(fetch, errors=_paging_errors(what)):
            if item.get("object") != "property_item" or not item.get("type"):
                raise RuntimeError(f"Notion listed an item of {what} that is not a property item")
            yield item

    async def query_data_source(self, data_source_id: str) -> AsyncIterator[dict[str, Any]]:
        """Every result of a data source, past Notion's 10,000-result cap, each yielded once.

        A query returns at most 10,000 results and then marks its pages
        ``request_status: incomplete`` with ``query_result_limit_reached``. So
        each query sorts by ``created_time`` ascending, and when one is cut at
        the cap, the next starts ``on_or_after`` the minute of the last result
        seen. Results in that boundary minute are read twice and yielded once::

            window 1: no filter                       -> 10,000 results, the last created 09:30:40
            window 2: created_time on_or_after 09:30  -> the rest

        A result is a row (``object: "page"``) or, in a wiki, a nested data
        source (``object: "data_source"``); both take part in the cut. These
        raise, so the listing fails rather than lists less:

        - a window that would start at the same minute as the one before it
          (10,000 results created in one minute cannot be windowed);
        - ``incomplete`` for any other reason;
        - a result repeated inside one window, or out of ``created_time`` order;
        - a result of another object type, or without an id or ``created_time``.

        Archived rows are left out, as Notion's query does by default.
        Windows are a transport detail: nothing is kept between runs.
        """
        listed: set[str] = set()
        window_start: datetime | None = None
        while True:
            last_created: datetime | None = None
            cut_at_cap = False
            seen_in_window: set[str] = set()

            async def fetch(cursor: str | None, start: datetime | None = window_start) -> CursorPage[dict[str, Any]]:
                nonlocal cut_at_cap
                body: dict[str, Any] = {
                    "sorts": [{"timestamp": "created_time", "direction": "ascending"}],
                    "page_size": _PAGE_SIZE,
                }
                if start is not None:
                    body["filter"] = {
                        "timestamp": "created_time",
                        "created_time": {"on_or_after": start.isoformat().replace("+00:00", "Z")},
                    }
                if cursor:
                    body["start_cursor"] = cursor
                response = await self._send("POST", f"/v1/data_sources/{data_source_id}/query", json=body)
                response.raise_for_status()
                data = _json_object(response, f"query of data source {data_source_id}")
                status = data.get("request_status")
                if isinstance(status, dict) and status.get("type") == "incomplete":
                    if status.get("incomplete_reason") != _RESULT_LIMIT_REACHED:
                        raise RuntimeError(
                            f"Notion answered the query of data source {data_source_id} incompletely "
                            f"({status.get('incomplete_reason')!r})"
                        )
                    cut_at_cap = True
                return _cursor_page(data, f"query of data source {data_source_id}")

            async for result in paged_by_cursor(fetch, errors=_paging_errors(f"query of data source {data_source_id}")):
                result_id = notion_id(result.get("id"))
                if result.get("object") not in {"page", "data_source"} or result_id is None:
                    raise RuntimeError(
                        f"Notion's query of data source {data_source_id} returned a result that is "
                        "neither a page nor a data source"
                    )
                if result_id in seen_in_window:
                    raise RuntimeError(f"Notion's query of data source {data_source_id} returned {result_id} twice")
                seen_in_window.add(result_id)
                created = notion_time(result.get("created_time"), f"result {result_id}")
                if last_created is not None and created < last_created:
                    raise RuntimeError(f"Notion's query of data source {data_source_id} is not in created_time order")
                last_created = created
                if result_id not in listed:
                    listed.add(result_id)
                    yield result

            if not cut_at_cap:
                return
            next_start = utc_minute_floor(last_created) if last_created is not None else None
            if next_start is None or (window_start is not None and next_start <= window_start):
                raise RuntimeError(
                    f"Notion data source {data_source_id} holds more than 10,000 results created in one "
                    "minute, so its query cannot be read in created_time windows"
                )
            window_start = next_start

    async def _send(self, method: str, path: str, **options: Any) -> httpx.Response:
        try:
            # Every Notion request only reads, so a POST query is replayed like a GET.
            return await self._client.request(method, path, safe_read=True, **options)
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != httpx.codes.TOO_MANY_REQUESTS:
                raise
            _code, reason = _error_code_and_reason(exc.response)
            if not reason:
                raise
            raise httpx.HTTPStatusError(
                f"{exc} Notion named the limit {reason}.", request=exc.request, response=exc.response
            ) from exc


def _json_object(response: httpx.Response, what: str) -> dict[str, Any]:
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Notion answered the read of {what} with a body that is not JSON") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"Notion answered the read of {what} with JSON that is not an object")
    return data


def _cursor_page(data: dict[str, Any], what: str) -> CursorPage[dict[str, Any]]:
    results = data.get("results")
    has_more = data.get("has_more")
    if not isinstance(results, list) or not all(isinstance(item, dict) for item in results):
        raise RuntimeError(f"Notion answered the {what} without a list of results")
    if not isinstance(has_more, bool):
        raise RuntimeError(f"Notion answered the {what} without has_more")
    next_cursor = data.get("next_cursor")
    return CursorPage(items=results, has_more=has_more, next_cursor=next_cursor if isinstance(next_cursor, str) else None)


def _paging_errors(what: str) -> PagingErrors:
    return PagingErrors(
        error_type=RuntimeError,
        repeated_next=f"Notion repeated a cursor in the {what}",
        missing_next=f"Notion said more of the {what} follows but gave no cursor",
    )


def _error_code_and_reason(response: httpx.Response) -> tuple[str | None, str | None]:
    """Notion's error ``code`` and ``additional_data.rate_limit_reason``, when the body names them."""
    try:
        body = response.json()
    except (ValueError, httpx.ResponseNotRead):
        return None, None
    if not isinstance(body, dict):
        return None, None
    additional = body.get("additional_data")
    reason = additional.get("rate_limit_reason") if isinstance(additional, dict) else None
    code = body.get("code")
    return (code if isinstance(code, str) else None), (reason if isinstance(reason, str) else None)
