"""The Notion Gene's answers for one page by id: ``confirm_absent`` and ``rediscover``."""

from __future__ import annotations

from datetime import datetime, timezone
from weakref import WeakValueDictionary

import httpx
import pytest

from memforge.genes import notion_gene
from memforge.genes.notion_gene import NotionGene
from memforge.models import AbsenceAnswer, ContentItem
from tests.connector_http_fixture import FakeConnectorHttp
from tests.notion_fake import TOKEN, NotionFake

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def own_rate_limiters(monkeypatch):
    monkeypatch.setattr("memforge.genes.toolkit.pacing._SHARED_LIMITERS", WeakValueDictionary())


async def connect(fake: NotionFake, **config) -> NotionGene:
    gene = NotionGene({"token": TOKEN, **config}, "notion-test")
    with FakeConnectorHttp(fake.handle).serving(notion_gene):
        await gene.authenticate()
    return gene


def stored(page_id: str) -> ContentItem:
    return ContentItem(
        item_id=f"notion-{page_id}",
        title="Stored",
        source_url=f"https://app.notion.com/p/{page_id}",
        last_modified=datetime(2026, 3, 1, tzinfo=timezone.utc),
        extra={"page_id": page_id},
    )


async def answer(fake: NotionFake, page_id: str) -> AbsenceAnswer:
    gene = await connect(fake, root_pages=["f" * 32])
    item = stored(page_id)
    return (await gene.confirm_absent([item]))[item.item_id]


# ---------------------------------------------------------------------------
# confirm_absent
# ---------------------------------------------------------------------------


async def test_a_trashed_page_is_absent() -> None:
    fake = NotionFake()
    page = fake.add_page("Trashed", in_trash=True)

    assert await answer(fake, page) is AbsenceAnswer.ABSENT


async def test_an_archived_page_exists() -> None:
    fake = NotionFake()
    page = fake.add_page("Archived", is_archived=True)

    assert await answer(fake, page) is AbsenceAnswer.EXISTS


async def test_a_page_moved_out_of_the_tree_exists() -> None:
    fake = NotionFake()
    page = fake.add_page("Moved to another workspace page", parent=("page", fake.add_page("Elsewhere")))

    assert await answer(fake, page) is AbsenceAnswer.EXISTS


async def test_a_page_no_longer_shared_is_absent() -> None:
    assert await answer(NotionFake(), "c" * 32) is AbsenceAnswer.ABSENT


@pytest.mark.parametrize(("status", "expected"), [(410, AbsenceAnswer.ABSENT), (403, AbsenceAnswer.HIDDEN)])
async def test_gone_and_forbidden_answers(status: int, expected: AbsenceAnswer) -> None:
    fake = NotionFake()
    page = fake.add_page("Page")
    fake.fail("GET", f"/v1/pages/{page}", status)

    assert await answer(fake, page) is expected


async def test_a_page_turned_into_a_wiki_exists() -> None:
    fake = NotionFake()
    page = fake.add_page("Team wiki", parent=("page", fake.add_page("Root")))
    fake.turn_into_wiki(page)

    assert await answer(fake, page) is AbsenceAnswer.EXISTS
    fake.databases[page]["in_trash"] = True
    assert await answer(fake, page) is AbsenceAnswer.ABSENT


@pytest.mark.parametrize(
    ("status", "body"),
    [(400, None), (400, {"object": "error", "code": "validation_error"}), (401, None)],
    ids=["400", "400-validation-error-on-a-page", "401"],
)
async def test_a_400_answer_raises(status: int, body: dict | None) -> None:
    fake = NotionFake()
    page = fake.add_page("Page")
    fake.fail("GET", f"/v1/pages/{page}", status, json_body=body)

    with pytest.raises(httpx.HTTPStatusError):
        await answer(fake, page)


# ---------------------------------------------------------------------------
# rediscover
# ---------------------------------------------------------------------------


async def test_rediscover_climbs_parents_to_a_configured_root() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    toggle = fake.add_block(root, "toggle")
    child = fake.add_page("Child", parent=("block", toggle))
    grandchild = fake.add_page("Grandchild", parent=("page", child), edited="2026-03-02T10:15:00.000Z")
    gene = await connect(fake, root_pages=[root])

    item = await gene.rediscover(stored(grandchild))

    assert item is not None
    assert item.item_id == f"notion-{grandchild}"
    assert item.version == "2026-03-02T10:15:00.000Z"
    assert item.extra == {"page_id": grandchild, "parent_page_id": child, "parent_kind": "page"}
    assert (await gene.rediscover(stored(child))).extra["parent_page_id"] == root
    assert (await gene.rediscover(stored(root))) is not None


async def test_rediscover_reaches_a_row_through_its_database() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    database, (source,) = fake.add_database(root)
    row = fake.add_row(database, source, "Row")
    under_row = fake.add_page("Under the row", parent=("page", row))
    root_database, (root_source,) = fake.add_database(None)
    root_row = fake.add_row(root_database, root_source, "Row of a root database")
    gene = await connect(fake, root_pages=[root], databases=[root_database])

    found = await gene.rediscover(stored(row))
    assert found is not None
    assert found.extra == {"page_id": row, "parent_page_id": database, "parent_kind": "database"}
    assert await gene.rediscover(stored(under_row)) is not None
    assert await gene.rediscover(stored(root_row)) is not None


async def test_rediscover_reaches_a_row_of_a_database_nested_in_a_wiki() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    wiki, (_wiki_source,) = fake.add_database(root, wiki=True)
    nested, nested_source = fake.add_nested_database(wiki)
    row = fake.add_row(nested, nested_source, "Use HANA for Cloud")
    gene = await connect(fake, root_pages=[root])

    found = await gene.rediscover(stored(row))

    assert found is not None
    assert found.extra == {"page_id": row, "parent_page_id": nested, "parent_kind": "database"}
    assert f"notion-{row}" in (await gene.list_scope()).doc_ids


@pytest.mark.parametrize("container", ["column", "toggle"])
@pytest.mark.parametrize("parented_to", ["block", "page"])
async def test_the_walk_and_rediscover_agree_on_a_database_inside_a_column_or_toggle(container, parented_to) -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    holder = (
        fake.add_block(fake.add_block(root, "column_list"), "column")
        if container == "column"
        else fake.add_block(root, "toggle")
    )
    database, (source,) = fake.add_database(holder if parented_to == "block" else root, named_by=holder)
    row = fake.add_row(database, source, "Row")
    gene = await connect(fake, root_pages=[root])

    assert f"notion-{row}" in (await gene.list_scope()).doc_ids
    found = await gene.rediscover(stored(row))
    assert found is not None
    assert found.extra == {"page_id": row, "parent_page_id": database, "parent_kind": "database"}


async def test_rediscover_climbs_through_a_parent_turned_into_a_wiki() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    former_page = fake.add_page("Team wiki", parent=("page", root))
    child = fake.add_page("Kept from the page body", parent=("page", former_page))
    fake.turn_into_wiki(former_page)
    gene = await connect(fake, root_pages=[root])

    found = await gene.rediscover(stored(child))

    assert found is not None
    assert found.extra == {"page_id": child, "parent_page_id": former_page, "parent_kind": "database"}
    assert f"notion-{child}" in (await gene.list_scope()).doc_ids
    assert await gene.rediscover(stored(former_page)) is None


async def test_rediscover_returns_none_outside_every_root() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    outside = fake.add_page("Outside", parent=("page", fake.add_page("Elsewhere")))
    hidden_parent = fake.add_page("Under a page the connection cannot see", parent=("page", "d" * 32))
    elsewhere = fake.add_page("Elsewhere too")
    database, (source,) = fake.add_database(elsewhere, named_by=root)
    linked_row = fake.add_row(database, source, "Row shown through a view")
    gene = await connect(fake, root_pages=[root])

    assert await gene.rediscover(stored(outside)) is None
    assert await gene.rediscover(stored(hidden_parent)) is None
    assert await gene.rediscover(stored(linked_row)) is None


async def test_rediscover_returns_none_for_a_trashed_page() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    trashed = fake.add_page("Trashed", parent=("page", root), in_trash=True)
    under_trashed = fake.add_page("Under a trashed page", parent=("page", trashed))
    gene = await connect(fake, root_pages=[root])

    assert await gene.rediscover(stored(trashed)) is None
    assert await gene.rediscover(stored(under_trashed)) is None
    assert await gene.rediscover(stored("e" * 32)) is None


async def test_rediscover_returns_none_for_an_archived_row() -> None:
    fake = NotionFake()
    database, (source,) = fake.add_database(None)
    archived = fake.add_row(database, source, "Archived", is_archived=True)
    under_archived = fake.add_page("Under an archived row", parent=("page", archived))
    gene = await connect(fake, databases=[database])

    assert await gene.rediscover(stored(archived)) is None
    assert await gene.rediscover(stored(under_archived)) is None
    assert {f"notion-{archived}", f"notion-{under_archived}"}.isdisjoint((await gene.list_scope()).doc_ids)


async def test_rediscover_returns_none_below_an_archived_or_trashed_root() -> None:
    fake = NotionFake()
    archived_root = fake.add_page("Archived root", is_archived=True)
    under_archived = fake.add_page("Under the archived root", parent=("page", archived_root))
    trashed_root = fake.add_page("Trashed root", in_trash=True)
    under_trashed = fake.add_page("Under the trashed root", parent=("page", trashed_root))
    gene = await connect(fake, root_pages=[archived_root, trashed_root])

    assert await gene.rediscover(stored(under_archived)) is None
    assert await gene.rediscover(stored(under_trashed)) is None
    archived_only = await connect(fake, root_pages=[archived_root])
    assert (await archived_only.list_scope()).doc_ids == set()


async def test_rediscover_matches_the_walk_for_a_page_shown_through_a_synced_duplicate() -> None:
    fake = NotionFake()
    outside = fake.add_page("Outside")
    original = fake.add_block(outside, "synced_block")
    shown = fake.add_page("Shown through the duplicate", parent=("block", original))
    root = fake.add_page("Root")
    fake.add_block(root, "synced_block", synced_from=original, has_children=True)
    gene = await connect(fake, root_pages=[root])

    assert f"notion-{shown}" not in (await gene.list_scope()).doc_ids
    assert await gene.rediscover(stored(shown)) is None


async def test_rediscover_reads_a_shared_ancestor_once() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    section = fake.add_page("Section", parent=("page", root))
    first = fake.add_page("First", parent=("page", section))
    second = fake.add_page("Second", parent=("page", section))
    gene = await connect(fake, root_pages=[root])

    assert await gene.rediscover(stored(first)) is not None
    assert await gene.rediscover(stored(second)) is not None
    assert fake.count("GET", f"/v1/pages/{section}") == 1

    await gene.close()
    assert gene._ancestor_verdicts == {}


async def test_rediscover_raises_beyond_its_parent_bound() -> None:
    fake = NotionFake()
    parent = fake.add_page("Top")
    for index in range(65):
        parent = fake.add_page(f"Level {index}", parent=("page", parent))
    gene = await connect(fake, root_pages=["f" * 32])

    with pytest.raises(RuntimeError, match="longer than 64 ancestors"):
        await gene.rediscover(stored(parent))
