"""Walk a provider's paged listing to its end, or raise.

A listing is evidence for absence: a document the walk did not see may be
removed. So a walk sees every page exactly once or raises; it never stops
quietly part way. The two pagers here hold those rules once, and a connector
only says how to fetch and read one page:

- :func:`paged_by_offset` for ``start``/``limit`` APIs, such as Confluence and
  Jira;
- :func:`paged_by_cursor` for ``next_cursor``/``has_more`` APIs, such as
  Notion.

Both are async generators that yield the items of every page, in order::

    from memforge.genes.toolkit.paging import OffsetPage, paged_by_offset

    async def fetch(start: int) -> OffsetPage[dict]:
        data = (await client.get("/rest/api/content", params={"start": start, "limit": 50})).json()
        return OffsetPage(start=data["start"], items=data["results"], next_link=data["_links"].get("next"))

    async for record in paged_by_offset(fetch):
        ...

``fetch`` checks the shape of one response (a results list, records that are
objects) with the connector's own messages. The pager checks how the pages fit
together. When a page breaks one of its rules, it raises
``errors.error_type`` with the message :class:`PagingErrors` gives for that
rule, so each connector keeps its own exception class and wording.

Stopping early is safe: leave the ``async for`` (``break`` or ``return``) and
no further page is fetched.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Hashable, Iterable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

__all__ = ["CursorPage", "OffsetPage", "PagingErrors", "paged_by_cursor", "paged_by_offset"]

T = TypeVar("T")


@dataclass(frozen=True)
class PagingErrors:
    """The exception a pager raises for each broken rule, worded by the connector.

    ``error_type`` is the exception class. Every other field is the message for
    one rule; a field left out keeps its generic message. A connector whose
    listing failures must surface as its own error names that class and words
    the rules it can meet::

        PagingErrors(
            error_type=SourceArtifactContractError,
            start_mismatch="Confluence attachment pagination is inconsistent",
            early_end="Confluence attachment response cannot advance pagination",
            repeated_next="Confluence attachment pagination contains a cursor cycle",
        )

    The rules:

    - ``start_mismatch`` (offset): a page does not start where the previous
      one ended, which is a gap or an overlap. The first page must start at 0.
    - ``total_changed`` (offset): a page declares a different ``total`` from
      the first page, or declares one when the first page did not.
    - ``early_end`` (offset): an empty page arrives while the listing says more
      follow: before the declared ``total``, or with a next link.
    - ``repeated_item`` (offset, with ``item_ids``): an item identity the walk
      has already yielded.
    - ``count_mismatch`` (offset): the pages hold more items than the declared
      ``total``.
    - ``repeated_next`` (offset and cursor): a next link or cursor the walk
      has already followed, which would loop.
    - ``missing_next`` (cursor): a page says ``has_more`` but gives no cursor.
    """

    error_type: type[Exception] = RuntimeError
    start_mismatch: str = "Paged listing returned a page that does not start where the previous page ended"
    total_changed: str = "Paged listing changed its declared total between pages"
    early_end: str = "Paged listing ended before its declared end"
    repeated_item: str = "Paged listing returned an item twice"
    count_mismatch: str = "Paged listing returned more items than its declared total"
    repeated_next: str = "Paged listing repeated a next link or cursor"
    missing_next: str = "Paged listing says more pages follow but gives no cursor"


@dataclass(frozen=True)
class OffsetPage(Generic[T]):
    """One page of an offset listing, as ``fetch(start)`` read it from the provider.

    - ``start``: the offset the response says this page begins at (Confluence
      ``start``, Jira ``startAt``), passed exactly as returned. The pager
      checks it, including that it is an integer.
    - ``items``: the page's records, in order. Their number is the page's
      size, and the next page is asked for at ``start + len(items)``.
    - ``total``: how many items the whole listing declares (Jira ``total``),
      or ``None`` when the provider declares no total.
    - ``next_link``: the link to the next page (Confluence ``_links.next``),
      or ``None`` on the last page. It is read only when ``total`` is ``None``.
    """

    start: object
    items: Sequence[T]
    total: int | None = None
    next_link: str | None = None


@dataclass(frozen=True)
class CursorPage(Generic[T]):
    """One page of a cursor listing, as ``fetch(cursor)`` read it from the provider.

    - ``items``: the page's records, in order.
    - ``has_more``: whether another page follows.
    - ``next_cursor``: the cursor that asks for the next page; required while
      ``has_more`` is true, and ignored otherwise.
    """

    items: Sequence[T]
    has_more: bool
    next_cursor: str | None = None


async def paged_by_offset(
    fetch: Callable[[int], Awaitable[OffsetPage[T]]],
    *,
    first: OffsetPage[T] | None = None,
    item_ids: Callable[[T], Iterable[Hashable]] | None = None,
    errors: PagingErrors = PagingErrors(),
) -> AsyncIterator[T]:
    """Yield every item of an offset listing, asking for each page at the offset the previous one ended.

    ``fetch(start)`` requests the page at ``start`` and returns it as an
    :class:`OffsetPage`. It chooses the page size itself, and may choose a
    different one for each page: servers cap the size they return anyway, so
    the pager only trusts what each page returned. The walk ends on the
    provider's own end signal:

    - **a declared total** (Jira): the walk ends once ``total`` items are
      yielded. An empty page before then is an early end, and more items
      than ``total`` is a count mismatch;
    - **no next link** (Confluence), when the page declares no total: the walk
      ends on the first page without a next link. An empty page with a next
      link is an early end, and a next link the walk has already followed is
      a loop.

    Every page must start where the previous one ended (the first at 0), and
    every page must declare the same total. A page is checked whole before
    any of its items is yielded. :class:`PagingErrors` names each rule.

    - ``first``: a page the caller has already read, as ``fetch(0)`` would
      return it. The walk checks it and starts from it, so ``fetch`` is first
      called for the second page. Confluence returns a page's first
      attachments inside the page itself, for example.
    - ``item_ids``: every identity of one item, each of which must be new to
      the walk. Jira gives an issue's id and its key, so neither may repeat.

    A page size that shrinks as a preview fills up is computed inside
    ``fetch``, which is called only when the next page is needed::

        emitted = 0

        async def fetch(start: int) -> OffsetPage[dict]:
            limit = min(50, preview_limit - emitted)
            ...

        async for record in paged_by_offset(fetch):
            yield record
            emitted += 1
            if emitted >= preview_limit:
                return
    """
    page = first if first is not None else await fetch(0)
    offset = 0
    total = page.total
    seen_ids: set[Hashable] = set()
    followed_links: set[str] = set()
    while True:
        if page.total != total:
            raise errors.error_type(errors.total_changed)
        if not isinstance(page.start, int) or isinstance(page.start, bool) or page.start != offset:
            raise errors.error_type(errors.start_mismatch)
        items = page.items
        more_declared = offset < total if total is not None else bool(page.next_link)
        if not items:
            if more_declared:
                raise errors.error_type(errors.early_end)
            return
        if item_ids is not None:
            for item in items:
                for identity in item_ids(item):
                    if identity in seen_ids:
                        raise errors.error_type(errors.repeated_item)
                    seen_ids.add(identity)
        end = offset + len(items)
        if total is not None:
            if end > total:
                raise errors.error_type(errors.count_mismatch)
            last = end == total
        else:
            last = not page.next_link
            if not last:
                if page.next_link in followed_links:
                    raise errors.error_type(errors.repeated_next)
                followed_links.add(page.next_link)
        for item in items:
            yield item
        if last:
            return
        offset = end
        page = await fetch(offset)


async def paged_by_cursor(
    fetch: Callable[[str | None], Awaitable[CursorPage[T]]],
    *,
    errors: PagingErrors = PagingErrors(),
) -> AsyncIterator[T]:
    """Yield every item of a cursor listing, following ``next_cursor`` while ``has_more`` is true.

    ``fetch(cursor)`` requests one page and returns it as a
    :class:`CursorPage`; the first call passes ``None``. The walk ends on the
    first page whose ``has_more`` is false. A page that says ``has_more`` but
    gives no cursor raises ``missing_next``, and a cursor the walk has already
    sent raises ``repeated_next``, because following it would loop. A page is
    checked before any of its items is yielded::

        from memforge.genes.toolkit.paging import CursorPage, paged_by_cursor

        async def fetch(cursor: str | None) -> CursorPage[dict]:
            body = {"page_size": 100} | ({"start_cursor": cursor} if cursor else {})
            data = (await client.post("/v1/search", json=body)).json()
            return CursorPage(items=data["results"], has_more=data["has_more"], next_cursor=data["next_cursor"])

        async for result in paged_by_cursor(fetch):
            ...
    """
    cursor: str | None = None
    sent_cursors: set[str] = set()
    while True:
        page = await fetch(cursor)
        if page.has_more:
            if not page.next_cursor:
                raise errors.error_type(errors.missing_next)
            if page.next_cursor in sent_cursors:
                raise errors.error_type(errors.repeated_next)
        for item in page.items:
            yield item
        if not page.has_more:
            return
        cursor = page.next_cursor
        sent_cursors.add(cursor)
