"""The connector toolkit's pagers: a listing is walked to its end exactly once, or the walk raises."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

import pytest

from memforge.genes.toolkit.paging import CursorPage, OffsetPage, PagingErrors, paged_by_cursor, paged_by_offset


class ListingBroken(Exception):
    """A connector's own error type, to show the caller chooses what a pager raises."""


ERRORS = PagingErrors(
    error_type=ListingBroken,
    start_mismatch="start mismatch",
    total_changed="total changed",
    early_end="early end",
    repeated_item="repeated item",
    count_mismatch="count mismatch",
    repeated_next="repeated next",
    missing_next="missing next",
)


class OffsetProvider:
    """Answers ``fetch(start)`` with prepared pages, keyed by the start asked for, and records each request."""

    def __init__(self, pages: dict[int, OffsetPage[str]]) -> None:
        self.pages = pages
        self.requests: list[int] = []

    async def fetch(self, start: int) -> OffsetPage[str]:
        self.requests.append(start)
        return self.pages[start]


async def _collect(items: AsyncIterator[str]) -> list[str]:
    return [item async for item in items]


def _by_total(*pages: tuple[object, Sequence[str]], total: int) -> dict[int, OffsetPage[str]]:
    pages_by_start: dict[int, OffsetPage[str]] = {}
    offset = 0
    for start, items in pages:
        pages_by_start[offset] = OffsetPage(start=start, items=items, total=total)
        offset += len(items)
    return pages_by_start


async def test_a_declared_total_is_walked_by_the_size_each_page_returned():
    # The caller asks for 50 a page; the server caps pages at 2 and 1.
    provider = OffsetProvider(_by_total((0, ["a", "b"]), (2, ["c", "d"]), (4, ["e"]), total=5))

    assert await _collect(paged_by_offset(provider.fetch, errors=ERRORS)) == ["a", "b", "c", "d", "e"]
    assert provider.requests == [0, 2, 4]


async def test_a_listing_without_a_total_ends_on_the_first_page_without_a_next_link():
    provider = OffsetProvider(
        {
            0: OffsetPage(start=0, items=["a", "b"], next_link="/content?start=2"),
            2: OffsetPage(start=2, items=["c"], next_link=None),
        }
    )

    assert await _collect(paged_by_offset(provider.fetch, errors=ERRORS)) == ["a", "b", "c"]
    assert provider.requests == [0, 2]


async def test_an_empty_page_without_a_next_link_ends_a_listing_without_a_total():
    provider = OffsetProvider(
        {
            0: OffsetPage(start=0, items=["a", "b"], next_link="/content?start=2"),
            2: OffsetPage(start=2, items=[], next_link=None),
        }
    )

    assert await _collect(paged_by_offset(provider.fetch, errors=ERRORS)) == ["a", "b"]


async def test_a_declared_total_ends_the_walk_without_asking_past_it():
    provider = OffsetProvider(_by_total((0, ["a", "b"]), total=2))

    assert await _collect(paged_by_offset(provider.fetch, errors=ERRORS)) == ["a", "b"]
    assert provider.requests == [0]


async def test_an_empty_listing_with_a_total_of_zero_yields_nothing():
    provider = OffsetProvider({0: OffsetPage(start=0, items=[], total=0)})

    assert await _collect(paged_by_offset(provider.fetch, errors=ERRORS)) == []


@pytest.mark.parametrize(
    "second_start",
    [3, 1],
    ids=["gap", "overlap"],
)
async def test_a_page_that_does_not_start_where_the_previous_one_ended_raises(second_start):
    provider = OffsetProvider(_by_total((0, ["a", "b"]), (second_start, ["c", "d"]), total=4))

    with pytest.raises(ListingBroken, match="start mismatch"):
        await _collect(paged_by_offset(provider.fetch, errors=ERRORS))


@pytest.mark.parametrize("start", [5, "0", None, False], ids=["not-zero", "string", "missing", "boolean"])
async def test_the_first_page_must_start_at_the_integer_zero(start):
    provider = OffsetProvider({0: OffsetPage(start=start, items=["a"], total=1)})

    with pytest.raises(ListingBroken, match="start mismatch"):
        await _collect(paged_by_offset(provider.fetch, errors=ERRORS))


async def test_an_empty_page_before_the_declared_total_is_an_early_end():
    provider = OffsetProvider(_by_total((0, ["a", "b"]), (2, []), total=3))

    with pytest.raises(ListingBroken, match="early end"):
        await _collect(paged_by_offset(provider.fetch, errors=ERRORS))


async def test_an_empty_page_with_a_next_link_is_an_early_end():
    provider = OffsetProvider(
        {
            0: OffsetPage(start=0, items=["a"], next_link="/content?start=1"),
            1: OffsetPage(start=1, items=[], next_link="/content?start=2"),
        }
    )

    with pytest.raises(ListingBroken, match="early end"):
        await _collect(paged_by_offset(provider.fetch, errors=ERRORS))


@pytest.mark.parametrize(
    "second_page",
    [
        OffsetPage(start=2, items=["c"], total=4),
        OffsetPage(start=2, items=["c"], total=None, next_link=None),
    ],
    ids=["total-grew", "total-dropped"],
)
async def test_a_total_that_changes_mid_walk_raises(second_page):
    provider = OffsetProvider({0: OffsetPage(start=0, items=["a", "b"], total=3), 2: second_page})

    with pytest.raises(ListingBroken, match="total changed"):
        await _collect(paged_by_offset(provider.fetch, errors=ERRORS))


async def test_a_total_that_appears_after_the_first_page_raises():
    provider = OffsetProvider(
        {
            0: OffsetPage(start=0, items=["a"], next_link="/content?start=1"),
            1: OffsetPage(start=1, items=["b"], total=2),
        }
    )

    with pytest.raises(ListingBroken, match="total changed"):
        await _collect(paged_by_offset(provider.fetch, errors=ERRORS))


async def test_more_items_than_the_declared_total_raises_before_any_of_them_is_yielded():
    provider = OffsetProvider({0: OffsetPage(start=0, items=["a", "b", "c"], total=2)})
    yielded: list[str] = []

    with pytest.raises(ListingBroken, match="count mismatch"):
        async for item in paged_by_offset(provider.fetch, errors=ERRORS):
            yielded.append(item)
    assert yielded == []


async def test_a_repeated_item_identity_raises_across_pages():
    provider = OffsetProvider(_by_total((0, ["a", "b"]), (2, ["b"]), total=3))

    with pytest.raises(ListingBroken, match="repeated item"):
        await _collect(paged_by_offset(provider.fetch, item_ids=lambda item: (item,), errors=ERRORS))


async def test_every_identity_of_an_item_must_be_new():
    # Two issues with different ids but the same key.
    issues = [{"id": "1", "key": "PAY-1"}, {"id": "2", "key": "PAY-1"}]

    async def fetch(start: int) -> OffsetPage[dict]:
        return OffsetPage(start=start, items=issues, total=2)

    with pytest.raises(ListingBroken, match="repeated item"):
        _ = [
            issue
            async for issue in paged_by_offset(
                fetch, item_ids=lambda issue: (("id", issue["id"]), ("key", issue["key"])), errors=ERRORS
            )
        ]


async def test_a_page_is_checked_whole_before_any_of_its_items_is_yielded():
    provider = OffsetProvider(_by_total((0, ["a"]), (1, ["b", "a"]), total=3))
    yielded: list[str] = []

    with pytest.raises(ListingBroken, match="repeated item"):
        async for item in paged_by_offset(provider.fetch, item_ids=lambda item: (item,), errors=ERRORS):
            yielded.append(item)
    assert yielded == ["a"]


async def test_a_next_link_the_walk_has_already_followed_raises():
    provider = OffsetProvider(
        {
            0: OffsetPage(start=0, items=["a"], next_link="/content?page=2"),
            1: OffsetPage(start=1, items=["b"], next_link="/content?page=2"),
        }
    )

    with pytest.raises(ListingBroken, match="repeated next"):
        await _collect(paged_by_offset(provider.fetch, errors=ERRORS))
    assert provider.requests == [0, 1]


async def test_a_walk_can_start_from_a_page_the_caller_already_read():
    provider = OffsetProvider({2: OffsetPage(start=2, items=["c"], next_link=None)})
    first = OffsetPage(start=0, items=["a", "b"], next_link="/attachments?start=2")

    assert await _collect(paged_by_offset(provider.fetch, first=first, errors=ERRORS)) == ["a", "b", "c"]
    assert provider.requests == [2]


async def test_a_page_the_caller_already_read_is_checked_like_any_other():
    provider = OffsetProvider({})

    with pytest.raises(ListingBroken, match="start mismatch"):
        await _collect(paged_by_offset(provider.fetch, first=OffsetPage(start=1, items=["a"]), errors=ERRORS))
    assert provider.requests == []


async def test_fetch_can_shrink_each_page_as_a_preview_fills_and_the_walk_stops_when_the_caller_leaves():
    preview_limit = 3
    emitted = 0
    limits: list[int] = []
    records = ["a", "b", "c", "d", "e"]

    async def fetch(start: int) -> OffsetPage[str]:
        limit = min(2, preview_limit - emitted)
        limits.append(limit)
        return OffsetPage(start=start, items=records[start:start + limit], next_link="/next" if start else "/first")

    previewed: list[str] = []
    async for record in paged_by_offset(fetch, errors=ERRORS):
        previewed.append(record)
        emitted += 1
        if emitted >= preview_limit:
            break

    assert previewed == ["a", "b", "c"]
    assert limits == [2, 1]


async def test_without_named_errors_a_pager_raises_runtime_error_with_a_generic_message():
    async def fetch(start: int) -> OffsetPage[str]:
        return OffsetPage(start=start + 1, items=["a"])

    with pytest.raises(RuntimeError, match="does not start where the previous page ended"):
        await _collect(paged_by_offset(fetch))


class CursorProvider:
    """Answers ``fetch(cursor)`` with prepared pages, keyed by the cursor sent, and records each request."""

    def __init__(self, pages: dict[str | None, CursorPage[str]]) -> None:
        self.pages = pages
        self.requests: list[str | None] = []

    async def fetch(self, cursor: str | None) -> CursorPage[str]:
        self.requests.append(cursor)
        return self.pages[cursor]


async def test_a_cursor_listing_is_followed_until_has_more_is_false():
    provider = CursorProvider(
        {
            None: CursorPage(items=["a", "b"], has_more=True, next_cursor="c1"),
            "c1": CursorPage(items=[], has_more=True, next_cursor="c2"),
            "c2": CursorPage(items=["c"], has_more=False, next_cursor="ignored"),
        }
    )

    assert await _collect(paged_by_cursor(provider.fetch, errors=ERRORS)) == ["a", "b", "c"]
    assert provider.requests == [None, "c1", "c2"]


@pytest.mark.parametrize("next_cursor", [None, ""], ids=["missing", "empty"])
async def test_a_missing_cursor_while_has_more_raises_before_the_page_is_yielded(next_cursor):
    provider = CursorProvider({None: CursorPage(items=["a"], has_more=True, next_cursor=next_cursor)})
    yielded: list[str] = []

    with pytest.raises(ListingBroken, match="missing next"):
        async for item in paged_by_cursor(provider.fetch, errors=ERRORS):
            yielded.append(item)
    assert yielded == []


async def test_a_cursor_cycle_raises():
    provider = CursorProvider(
        {
            None: CursorPage(items=["a"], has_more=True, next_cursor="c1"),
            "c1": CursorPage(items=["b"], has_more=True, next_cursor="c2"),
            "c2": CursorPage(items=["c"], has_more=True, next_cursor="c1"),
        }
    )

    with pytest.raises(ListingBroken, match="repeated next"):
        await _collect(paged_by_cursor(provider.fetch, errors=ERRORS))
    assert provider.requests == [None, "c1", "c2"]


async def test_a_cursor_pager_raises_runtime_error_by_default():
    provider = CursorProvider({None: CursorPage(items=[], has_more=True)})

    with pytest.raises(RuntimeError, match="gives no cursor"):
        await _collect(paged_by_cursor(provider.fetch))
