"""The Notion Gene's walk: what it lists, what it skips, and what fails the listing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from weakref import WeakValueDictionary

import httpx
import pytest

from memforge.genes import notion_gene
from memforge.genes.base import SourceConfigurationError
from memforge.genes.notion_api import NOTION_VERSION, NotionRequestBlockedError
from memforge.genes.notion_gene import NotionGene
from memforge.genes.toolkit import pacing
from memforge.models import AbsenceAnswer, ContentItem, ScopeListingKind
from tests.connector_http_fixture import FakeConnectorHttp
from tests.notion_fake import TOKEN, NotionFake, dashed

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def own_rate_limiters(monkeypatch):
    monkeypatch.setattr("memforge.genes.toolkit.pacing._SHARED_LIMITERS", WeakValueDictionary())


async def connect(fake: NotionFake, **config) -> tuple[NotionGene, FakeConnectorHttp]:
    gene = NotionGene({"token": TOKEN, **config}, "notion-test")
    http = FakeConnectorHttp(fake.handle)
    with http.serving(notion_gene):
        await gene.authenticate()
    return gene, http


async def listed(gene: NotionGene) -> set[str]:
    listing = await gene.list_scope()
    assert listing.kind is ScopeListingKind.QUERY
    return set(listing.doc_ids)


def docs(*page_ids: str) -> set[str]:
    return {f"notion-{page_id}" for page_id in page_ids}


def query_bodies(fake: NotionFake, source: str) -> list[dict]:
    return [
        json.loads(request.content)
        for request in fake.requests
        if request.method == "POST" and request.url.path == f"/v1/data_sources/{source}/query"
    ]


# ---------------------------------------------------------------------------
# What the walk reaches
# ---------------------------------------------------------------------------


async def test_the_walk_reaches_child_pages_inside_toggles_and_columns() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    toggle = fake.add_block(root, "toggle")
    in_toggle = fake.add_page("In a toggle", parent=("block", toggle))
    column_list = fake.add_block(root, "column_list")
    column = fake.add_block(column_list, "column")
    in_column = fake.add_page("In a column", parent=("block", column))
    heading = fake.add_block(root, "heading_2")
    in_heading = fake.add_page("Under a toggleable heading", parent=("block", heading))
    callout = fake.add_block(root, "callout")
    in_callout = fake.add_page("In a callout", parent=("block", callout))
    original = fake.add_block(root, "synced_block")
    in_original = fake.add_page("In an original synced block", parent=("block", original))
    item = fake.add_block(root, "bulleted_list_item")
    in_item = fake.add_page("Under a list item", parent=("block", item))
    database, (source,) = fake.add_database(column)
    row = fake.add_row(database, source, "Row of a database in a column")
    fake.add_block(root, "paragraph")
    gene, _http = await connect(fake, root_pages=[root])

    assert await listed(gene) == docs(root, in_toggle, in_column, in_heading, in_callout, in_original, in_item, row)


async def test_the_walk_recurses_into_database_rows_and_their_child_pages() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    database, (source,) = fake.add_database(root)
    row = fake.add_row(database, source, "Row")
    under_row = fake.add_page("Under the row", parent=("page", row))
    inner_database, (inner_source,) = fake.add_database(row)
    inner_row = fake.add_row(inner_database, inner_source, "Inner row")
    gene, _http = await connect(fake, root_pages=[root])

    assert await listed(gene) == docs(root, row, under_row, inner_row)


async def test_each_database_adds_every_data_source() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    database, (first, second) = fake.add_database(root, sources=2)
    rows = [fake.add_row(database, first, "A"), fake.add_row(database, second, "B")]
    root_database, (third, fourth) = fake.add_database(None, sources=2)
    rows += [fake.add_row(root_database, third, "C"), fake.add_row(root_database, fourth, "D")]
    gene, _http = await connect(fake, root_pages=[root], databases=[dashed(root_database)])

    assert await listed(gene) == docs(root, *rows)


async def test_a_database_nested_in_a_wiki_is_reached_through_the_wikis_child_blocks() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    wiki, (wiki_source,) = fake.add_database(root, wiki=True)
    entry = fake.add_wiki_entry(wiki, wiki_source, "Coding standards")
    nested, nested_source = fake.add_nested_database(wiki)
    nested_row = fake.add_row(nested, nested_source, "Use HANA for Cloud")

    for config in ({"root_pages": [root]}, {"databases": [wiki]}):
        fake.requests.clear()
        gene, _http = await connect(fake, **config)
        found = [item.item_id async for item in gene.discover(None)]

        assert sorted(found) == sorted(docs(entry, nested_row) | (docs(root) if "root_pages" in config else set()))
        assert fake.count("GET", f"/v1/blocks/{wiki}/children") == 1
        assert fake.count("POST", f"/v1/data_sources/{nested_source}/query") == 1
        assert fake.count("GET", f"/v1/pages/{entry}") == 0  # listed from the query, not read again


async def test_a_data_source_returned_by_a_wiki_query_is_queried_too() -> None:
    fake = NotionFake()
    wiki, (wiki_source,) = fake.add_database(None, wiki=True)
    wiki_page = fake.add_wiki_entry(wiki, wiki_source, "Wiki page", created="2026-03-01T09:00:00.000Z")
    nested, nested_source = fake.add_nested_data_source_result(wiki_source, created="2026-03-01T09:01:00.000Z")
    nested_row = fake.add_row(nested, nested_source, "Nested row")
    gene, _http = await connect(fake, databases=[wiki])

    assert await listed(gene) == docs(wiki_page, nested_row)
    assert fake.count("POST", f"/v1/data_sources/{nested_source}/query") == 1


async def test_a_root_page_turned_into_a_wiki_is_walked_as_a_database_root() -> None:
    fake = NotionFake()
    root = fake.add_page("Team wiki")
    wiki_source = fake.turn_into_wiki(root)
    entry = fake.add_wiki_entry(root, wiki_source, "Glossary")
    nested, nested_source = fake.add_nested_database(root)
    nested_row = fake.add_row(nested, nested_source, "Decision")
    gene, _http = await connect(fake, root_pages=[root])

    assert await listed(gene) == docs(entry, nested_row)


async def test_a_page_reached_twice_is_read_once() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    child = fake.add_page("Child", parent=("page", root))
    gene, _http = await connect(fake, root_pages=[root, child])

    assert await listed(gene) == docs(root, child)
    assert fake.count("GET", f"/v1/blocks/{child}/children") == 1


async def test_the_walk_does_not_descend_into_a_synced_duplicate() -> None:
    fake = NotionFake()
    outside = fake.add_page("Outside the roots")
    original = fake.add_block(outside, "synced_block")
    shown = fake.add_page("Shown through the duplicate", parent=("block", original))
    root = fake.add_page("Root")
    duplicate = fake.add_block(root, "synced_block", synced_from=original, has_children=True)
    gene, _http = await connect(fake, root_pages=[root])

    assert await listed(gene) == docs(root)
    assert f"/v1/blocks/{duplicate}/children" not in fake.paths()
    assert f"/v1/pages/{shown}" not in fake.paths()


async def test_deeply_nested_blocks_do_not_exhaust_the_stack() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    container = root
    for _ in range(2_000):
        container = fake.add_block(container, "toggle")
    deepest = fake.add_page("Deepest", parent=("block", container))
    gene, _http = await connect(fake, root_pages=[root])

    assert await listed(gene) == docs(root, deepest)


async def test_a_cyclic_block_reference_terminates() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    toggle = fake.add_block(root, "toggle")
    inner = fake.add_block(toggle, "toggle")
    fake.children[inner].append(toggle)  # a provider bug: a block lists its own ancestor
    gene, _http = await connect(fake, root_pages=[root])

    assert await listed(gene) == docs(root)
    assert fake.count("GET", f"/v1/blocks/{toggle}/children") == 1


async def test_more_than_100_children_are_read_through_start_cursor() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    top = [fake.add_page(f"Top {index}", parent=("page", root)) for index in range(150)]
    toggle = fake.add_block(root, "toggle")
    nested = [fake.add_page(f"Nested {index}", parent=("block", toggle)) for index in range(120)]
    gene, _http = await connect(fake, root_pages=[root])

    assert await listed(gene) == docs(root, *top, *nested)
    for container in (root, toggle):
        reads = [r for r in fake.requests if r.url.path == f"/v1/blocks/{container}/children"]
        assert [r.url.params.get("start_cursor") for r in reads] == [None, "100"]
        assert all(r.url.params["page_size"] == "100" for r in reads)


async def test_an_archived_page_is_not_listed() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    archived = fake.add_page("Archived", parent=("page", root), is_archived=True)
    fake.add_page("Under the archived page", parent=("page", archived))
    database, (source,) = fake.add_database(root)
    fake.add_row(database, source, "Archived row", is_archived=True)
    kept_row = fake.add_row(database, source, "Row")
    gene, _http = await connect(fake, root_pages=[root])

    assert await listed(gene) == docs(root, kept_row)


async def test_the_deprecated_archived_field_is_never_read() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    child = fake.add_page("Child", parent=("page", root), archived=True)
    gene, _http = await connect(fake, root_pages=[root])

    assert await listed(gene) == docs(root, child)
    item = ContentItem(item_id=f"notion-{child}", title="", source_url="", last_modified=datetime.now(timezone.utc))
    assert await gene.confirm_absent([item]) == {item.item_id: AbsenceAnswer.EXISTS}


async def test_the_walk_costs_one_call_per_page_and_container() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    toggle = fake.add_block(root, "toggle")
    fake.add_page("In the toggle", parent=("block", toggle))
    fake.add_page("Child", parent=("page", root))
    outside = fake.add_page("Outside")
    original = fake.add_block(outside, "synced_block")
    fake.add_block(root, "synced_block", synced_from=original, has_children=True)
    fake.add_block(root, "paragraph")
    gene, _http = await connect(fake, root_pages=[root])
    fake.requests.clear()

    await listed(gene)

    pages, containers, child_page_reads = 3, 1, 2
    root_read = 1
    assert len(fake.requests) == pages + containers + child_page_reads + root_read


# ---------------------------------------------------------------------------
# Data-source windows
# ---------------------------------------------------------------------------


async def test_a_single_complete_window_does_not_query_again() -> None:
    fake = NotionFake()
    database, (source,) = fake.add_database(None)
    rows = [fake.add_row(database, source, f"Row {index}") for index in range(3)]
    gene, _http = await connect(fake, databases=[database])

    assert await listed(gene) == docs(*rows)
    (body,) = query_bodies(fake, source)
    assert body["sorts"] == [{"timestamp": "created_time", "direction": "ascending"}]
    assert "filter" not in body


async def test_a_data_source_over_the_result_cap_is_read_in_created_time_windows() -> None:
    fake = NotionFake()
    fake.result_cap = 3
    database, (source,) = fake.add_database(None)
    rows = [fake.add_row(database, source, f"Row {minute}", created=f"2026-03-01T09:{minute:02d}:00.000Z") for minute in range(7)]
    gene, _http = await connect(fake, databases=[database])

    assert await listed(gene) == docs(*rows)
    filters = [body.get("filter") for body in query_bodies(fake, source)]
    assert filters == [
        None,
        {"timestamp": "created_time", "created_time": {"on_or_after": "2026-03-01T09:02:00Z"}},
        {"timestamp": "created_time", "created_time": {"on_or_after": "2026-03-01T09:04:00Z"}},
    ]


async def test_rows_in_a_window_boundary_minute_are_listed_once() -> None:
    fake = NotionFake()
    fake.result_cap = 3
    database, (source,) = fake.add_database(None)
    times = ["09:00", "09:01", "09:01", "09:02", "09:03"]
    rows = [fake.add_row(database, source, f"Row {i}", created=f"2026-03-01T{t}:00.000Z") for i, t in enumerate(times)]
    gene, _http = await connect(fake, databases=[database])
    found: list[str] = []
    async for item in gene.discover(None):
        found.append(item.item_id)

    assert sorted(found) == sorted(docs(*rows))
    assert len(found) == len(set(found))
    assert len(query_bodies(fake, source)) == 3  # the 09:01 rows were read in two windows


async def test_a_window_advances_on_a_nested_data_source_boundary() -> None:
    fake = NotionFake()
    fake.result_cap = 2
    wiki, (wiki_source,) = fake.add_database(None)
    first = fake.add_row(wiki, wiki_source, "First", created="2026-03-01T09:00:00.000Z")
    nested, nested_source = fake.add_nested_data_source_result(wiki_source, created="2026-03-01T09:05:00.000Z")
    later = fake.add_row(wiki, wiki_source, "Later", created="2026-03-01T09:07:00.000Z")
    nested_row = fake.add_row(nested, nested_source, "Nested row")
    gene, _http = await connect(fake, databases=[wiki])

    assert await listed(gene) == docs(first, later, nested_row)
    assert query_bodies(fake, wiki_source)[1]["filter"]["created_time"] == {"on_or_after": "2026-03-01T09:05:00Z"}


async def test_a_window_that_cannot_advance_fails_the_listing() -> None:
    fake = NotionFake()
    fake.result_cap = 2
    database, (source,) = fake.add_database(None)
    for index in range(3):
        fake.add_row(database, source, f"Row {index}", created=f"2026-03-01T09:00:{index:02d}.000Z")
    gene, _http = await connect(fake, databases=[database])

    with pytest.raises(RuntimeError, match="cannot be read in created_time windows"):
        await gene.list_scope()


async def test_an_incomplete_answer_for_another_reason_fails_the_listing() -> None:
    fake = NotionFake()
    fake.incomplete_reason = "something_else"
    database, (source,) = fake.add_database(None)
    fake.add_row(database, source, "Row")
    gene, _http = await connect(fake, databases=[database])

    with pytest.raises(RuntimeError, match="incompletely"):
        await gene.list_scope()


async def test_a_row_repeated_inside_one_window_fails_the_listing() -> None:
    fake = NotionFake()
    database, (source,) = fake.add_database(None)
    row = fake.add_row(database, source, "Row")
    fake.results[source].append(row)
    gene, _http = await connect(fake, databases=[database])

    with pytest.raises(RuntimeError, match="twice"):
        await gene.list_scope()


async def test_a_query_result_of_another_object_type_fails_the_listing() -> None:
    fake = NotionFake()
    database, (source,) = fake.add_database(None)
    row = fake.add_row(database, source, "Row")
    fake.pages[row]["object"] = "block"
    gene, _http = await connect(fake, databases=[database])

    with pytest.raises(RuntimeError, match="neither a page nor a data source"):
        await gene.list_scope()


# ---------------------------------------------------------------------------
# 404 below a root, and databases outside the tree
# ---------------------------------------------------------------------------


async def test_a_child_that_answers_404_mid_walk_is_skipped_and_confirmed_later() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    kept = fake.add_page("Kept", parent=("page", root))
    deleted = fake.add_page("Deleted during the walk", parent=("page", root))
    fake.fail("GET", f"/v1/pages/{deleted}", 404, times=2)
    gene, _http = await connect(fake, root_pages=[root])

    assert await listed(gene) == docs(root, kept)
    assert fake.count("GET", f"/v1/pages/{deleted}") == 2
    item = ContentItem(item_id=f"notion-{deleted}", title="", source_url="", last_modified=datetime.now(timezone.utc))
    del fake.pages[deleted]
    assert await gene.confirm_absent([item]) == {item.item_id: AbsenceAnswer.ABSENT}


async def test_a_404_that_the_read_by_id_contradicts_fails_the_listing() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    child = fake.add_page("Child", parent=("page", root))
    fake.add_page("Grandchild", parent=("page", child))
    fake.fail("GET", f"/v1/blocks/{child}/children", 404)
    gene, _http = await connect(fake, root_pages=[root])

    with pytest.raises(RuntimeError, match="the listing cannot skip it"):
        await gene.list_scope()


async def test_a_child_database_the_connection_cannot_read_is_skipped() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    database, (source,) = fake.add_database(root)
    fake.add_row(database, source, "Row of a database deleted during the walk")
    fake.fail("GET", f"/v1/databases/{database}", 404, times=2)
    gene, _http = await connect(fake, root_pages=[root])

    assert await listed(gene) == docs(root)
    assert fake.count("GET", f"/v1/databases/{database}") == 2
    assert query_bodies(fake, source) == []


async def test_a_linked_view_lists_nothing_and_is_counted(caplog) -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    view = fake.add_linked_view(root)
    gene, _http = await connect(fake, root_pages=[root])

    with caplog.at_level("INFO", logger="memforge.genes.notion_gene"):
        assert await listed(gene) == docs(root)

    assert fake.count("GET", f"/v1/databases/{view}") == 1
    assert "skipped_not_found=0" in caplog.text
    assert "empty_databases=1" in caplog.text


async def test_rows_of_a_database_outside_the_tree_are_not_listed() -> None:
    fake = NotionFake()
    elsewhere = fake.add_page("Elsewhere")
    root = fake.add_page("Root")
    database, (source,) = fake.add_database(elsewhere, named_by=root)
    fake.add_row(database, source, "Row living elsewhere")
    gene, _http = await connect(fake, root_pages=[root])

    assert await listed(gene) == docs(root)
    assert query_bodies(fake, source) == []


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------


async def test_a_root_page_that_answers_404_is_a_configuration_error() -> None:
    gene, _http = await connect(NotionFake(), root_pages=["a" * 32])

    with pytest.raises(SourceConfigurationError, match="root page a{32} does not exist or is not shared"):
        await gene.list_scope()


async def test_a_root_database_that_answers_404_is_a_configuration_error() -> None:
    gene, _http = await connect(NotionFake(), databases=["b" * 32])

    with pytest.raises(SourceConfigurationError, match="database b{32} does not exist or is not shared"):
        await gene.list_scope()


async def test_a_trashed_root_is_a_configuration_error() -> None:
    fake = NotionFake()
    root = fake.add_page("Root", in_trash=True)
    gene, _http = await connect(fake, root_pages=[root])

    with pytest.raises(SourceConfigurationError, match="is in the trash"):
        await gene.list_scope()


async def test_a_root_that_answers_403_fails_the_listing() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    fake.fail("GET", f"/v1/pages/{root}", 403)
    gene, _http = await connect(fake, root_pages=[root])

    with pytest.raises(httpx.HTTPStatusError) as raised:
        await gene.list_scope()
    assert not isinstance(raised.value, SourceConfigurationError)
    assert raised.value.response.status_code == 403


# ---------------------------------------------------------------------------
# Fail closed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.parametrize("where", ["children", "child page", "database", "data source"])
async def test_401_and_403_below_a_root_fail_the_listing(status: int, where: str) -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    child = fake.add_page("Child", parent=("page", root))
    database, (source,) = fake.add_database(root)
    path = {
        "children": ("GET", f"/v1/blocks/{root}/children"),
        "child page": ("GET", f"/v1/pages/{child}"),
        "database": ("GET", f"/v1/databases/{database}"),
        "data source": ("POST", f"/v1/data_sources/{source}/query"),
    }[where]
    fake.fail(*path, status)
    gene, _http = await connect(fake, root_pages=[root])

    with pytest.raises(httpx.HTTPStatusError) as raised:
        await gene.list_scope()
    assert raised.value.response.status_code == status
    assert gene._discovered_scope is None


async def test_a_403_data_source_fails_the_listing_and_is_never_empty() -> None:
    fake = NotionFake()
    database, (source,) = fake.add_database(None)
    fake.add_row(database, source, "Row")
    fake.fail("POST", f"/v1/data_sources/{source}/query", 403)
    gene, _http = await connect(fake, databases=[database])

    with pytest.raises(httpx.HTTPStatusError):
        await gene.list_scope()


async def test_a_500_on_block_children_fails_the_listing() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    fake.fail("GET", f"/v1/blocks/{root}/children", 500, times=4)
    gene, _http = await connect(fake, root_pages=[root])

    with pytest.raises(httpx.HTTPStatusError, match="answered 500"):
        await gene.list_scope()


async def test_a_400_invalid_start_cursor_fails_the_listing() -> None:
    fake = NotionFake()
    fake.cursors_expire = True
    root = fake.add_page("Root")
    for index in range(101):
        fake.add_page(f"Child {index}", parent=("page", root))
    gene, _http = await connect(fake, root_pages=[root])

    with pytest.raises(httpx.HTTPStatusError) as raised:
        await gene.list_scope()
    assert raised.value.response.status_code == 400


async def test_an_exhausted_retry_fails_the_listing() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    fake.fail("GET", f"/v1/blocks/{root}/children", 503, times=4)
    gene, _http = await connect(fake, root_pages=[root])

    with pytest.raises(httpx.HTTPStatusError):
        await gene.list_scope()
    assert fake.count("GET", f"/v1/blocks/{root}/children") == 4


async def test_has_more_without_a_cursor_fails_the_listing() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    fake.fail(
        "GET",
        f"/v1/blocks/{root}/children",
        200,
        json_body={"object": "list", "results": [], "has_more": True, "next_cursor": None},
    )
    gene, _http = await connect(fake, root_pages=[root])

    with pytest.raises(RuntimeError, match="gave no cursor"):
        await gene.list_scope()


async def test_search_is_never_called() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    database, (source,) = fake.add_database(root)
    fake.add_row(database, source, "Row")
    gene, _http = await connect(fake, root_pages=[root])

    async for _item in gene.discover(None):
        pass
    await gene.list_scope()

    assert "/v1/search" not in fake.paths()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


async def test_discover_rounds_since_down_to_the_minute() -> None:
    fake = NotionFake()
    root = fake.add_page("Root", edited="2026-03-01T09:00:00.000Z")
    in_minute = fake.add_page("Edited in the cursor minute", parent=("page", root), edited="2026-03-01T09:30:00.000Z")
    fake.add_page("Edited before", parent=("page", root), edited="2026-03-01T09:29:00.000Z")
    gene, _http = await connect(fake, root_pages=[root])

    found = [item async for item in gene.discover(datetime(2026, 3, 1, 9, 30, 10, tzinfo=timezone.utc))]

    assert [item.item_id for item in found] == [f"notion-{in_minute}"]
    assert gene.filter_summary(datetime(2026, 3, 1, 9, 30, 10, tzinfo=timezone.utc)) == (
        "last_edited_time >= 2026-03-01T09:30:00+00:00"
    )
    assert len((await gene.list_scope()).doc_ids) == 3


async def test_discovered_items_carry_the_page_and_its_nearest_container() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    toggle = fake.add_block(root, "toggle")
    child = fake.add_page("Child", parent=("block", toggle), edited="2026-03-01T09:30:00.000Z")
    database, (source,) = fake.add_database(root)
    row = fake.add_row(database, source, "Row")
    gene, _http = await connect(fake, root_pages=[root])

    items = {item.item_id: item async for item in gene.discover(None)}

    page = items[f"notion-{child}"]
    assert page.title == "Child"
    assert page.source_url == f"https://app.notion.com/p/{child}"
    assert page.version == "2026-03-01T09:30:00.000Z"
    assert page.last_modified == datetime(2026, 3, 1, 9, 30, tzinfo=timezone.utc)
    assert page.extra == {"page_id": child, "parent_page_id": root, "parent_kind": "page"}
    assert items[f"notion-{row}"].extra == {"page_id": row, "parent_page_id": database, "parent_kind": "database"}
    assert items[f"notion-{root}"].extra == {"page_id": root, "parent_page_id": None, "parent_kind": None}


async def test_a_preview_stops_at_its_limit_and_records_no_listing() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    for index in range(5):
        fake.add_page(f"Child {index}", parent=("page", root))
    gene, _http = await connect(fake, root_pages=[root], _memforge_preview_limit=2)

    found = [item async for item in gene.discover(None)]

    assert len(found) == 2
    assert gene._discovered_scope is None


async def test_list_scope_refuses_a_preview() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    gene, _http = await connect(fake, root_pages=[root], _memforge_preview_limit=2)

    with pytest.raises(RuntimeError, match="preview"):
        await gene.list_scope()


async def test_a_listing_reuses_the_discovery_walk() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    gene, _http = await connect(fake, root_pages=[root])
    async for _item in gene.discover(None):
        pass
    walked = len(fake.requests)

    assert await listed(gene) == docs(root)
    assert len(fake.requests) == walked


# ---------------------------------------------------------------------------
# Connection, pacing and retries
# ---------------------------------------------------------------------------


async def test_every_request_sends_the_pinned_notion_version() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    database, (source,) = fake.add_database(root)
    fake.add_row(database, source, "Row")
    gene, http = await connect(fake, root_pages=[root])
    await gene.list_scope()

    assert fake.requests
    assert {request.headers["Notion-Version"] for request in fake.requests} == {NOTION_VERSION}
    api_options, file_options = http.options
    assert api_options["base_url"] == "https://api.notion.com"
    assert api_options["follow_redirects"] is False
    assert "Authorization" not in file_options["headers"]
    assert file_options.get("limiter") is None


async def test_a_rejected_token_is_a_configuration_error() -> None:
    gene = NotionGene({"token": "not-the-token", "root_pages": ["a" * 32]}, "notion-test")
    with FakeConnectorHttp(NotionFake().handle).serving(notion_gene):
        with pytest.raises(SourceConfigurationError, match="rejected the connection token"):
            await gene.authenticate()
    await gene.close()


@pytest.mark.parametrize("status", [500, 503, 529])
async def test_a_data_source_query_is_replayed_as_a_safe_read(status: int) -> None:
    fake = NotionFake()
    database, (source,) = fake.add_database(None)
    row = fake.add_row(database, source, "Row")
    fake.fail("POST", f"/v1/data_sources/{source}/query", status)
    gene, _http = await connect(fake, databases=[database])

    assert await listed(gene) == docs(row)
    assert fake.count("POST", f"/v1/data_sources/{source}/query") == 2


async def test_a_blocked_integration_is_not_retried() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    fake.fail(
        "GET",
        f"/v1/pages/{root}",
        429,
        json_body={
            "object": "error",
            "code": "rate_limited",
            "additional_data": {"rate_limit_reason": "public_api_request_blocked"},
        },
    )
    gene, _http = await connect(fake, root_pages=[root])

    with pytest.raises(NotionRequestBlockedError, match="asks not to retry"):
        await gene.list_scope()
    assert fake.count("GET", f"/v1/pages/{root}") == 1


async def test_a_rate_limit_error_names_notions_reason() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    fake.fail(
        "GET",
        f"/v1/pages/{root}",
        429,
        times=4,
        headers={"Retry-After": "1"},
        json_body={
            "object": "error",
            "code": "rate_limited",
            "additional_data": {"rate_limit_reason": "public_api_space_request_rate_limit"},
        },
    )
    gene, http = await connect(fake, root_pages=[root])

    with pytest.raises(httpx.HTTPStatusError, match="rate limit persisted.*public_api_space_request_rate_limit"):
        await gene.list_scope()
    assert 1.0 in http.waits


async def test_requests_share_one_limiter_per_token() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    first, first_http = await connect(fake, root_pages=[root])
    second, second_http = await connect(fake, root_pages=[root])

    first_limiter = first_http.options[0]["limiter"]
    assert first_limiter is second_http.options[0]["limiter"]
    assert first_limiter.interval_seconds == pytest.approx(60 / 180)
    assert all(TOKEN not in key for key in pacing._SHARED_LIMITERS)
    await first.close()
    await second.close()
