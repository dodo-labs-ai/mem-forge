"""An in-memory Notion workspace that answers the API the Notion Gene reads.

Build a workspace, then serve it to a Gene through ``FakeConnectorHttp``::

    fake = NotionFake()
    root = fake.add_page("Handbook")
    toggle = fake.add_block(root, "toggle")
    child = fake.add_page("Inside the toggle", parent=("block", toggle))
    database, (source,) = fake.add_database(root)
    row = fake.add_row(database, source, "First row")

    gene = NotionGene({"token": "notion-test-token", "root_pages": [root]}, "notion-test")
    http = FakeConnectorHttp(fake.handle)
    with http.serving(notion_gene):
        await gene.authenticate()

It models what the Gene relies on, in the shapes Notion answered in the live
checks: pages, blocks and their children, synced originals and duplicates,
databases with several data sources, wikis (entries with a database parent,
and nested databases reachable only as the wiki database's child blocks),
pages turned into wikis (a page read answers 400 "is a database, not a
page"), linked views (a database with no data sources), minute-truncated
times, the 10,000-result cap (``result_cap``) with ``request_status``,
archived and trashed pages, queued failures (:meth:`fail`), and cursors
that stop working (``cursors_expire``). Ids are 32-hex internally and
written with dashes in answers, as Notion writes them. ``requests`` records
every request.
"""

from __future__ import annotations

import json
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Any

import httpx

TOKEN = "notion-test-token"
DEFAULT_TIME = "2026-03-01T09:00:00.000Z"


def dashed(id32: str) -> str:
    return str(uuid.UUID(id32))


def new_id() -> str:
    return uuid.uuid4().hex


class NotionFake:
    """One Notion workspace; see the module docstring."""

    def __init__(self) -> None:
        self.pages: dict[str, dict[str, Any]] = {}
        self.blocks: dict[str, dict[str, Any]] = {}
        self.children: dict[str, list[str]] = defaultdict(list)
        self.databases: dict[str, dict[str, Any]] = {}
        self.data_sources: dict[str, dict[str, Any]] = {}
        self.results: dict[str, list[str]] = defaultdict(list)
        self.failures: dict[tuple[str, str], list[httpx.Response]] = defaultdict(list)
        self.requests: list[httpx.Request] = []
        self.result_cap = 10_000
        self.incomplete_reason: str | None = None
        self.cursors_expire = False

    # ------------------------------------------------------------------
    # Building the workspace
    # ------------------------------------------------------------------

    def add_page(
        self,
        title: str,
        *,
        parent: tuple[str, str] | tuple[str] = ("workspace",),
        created: str = DEFAULT_TIME,
        edited: str = DEFAULT_TIME,
        in_trash: bool = False,
        is_archived: bool = False,
        page_id: str | None = None,
        **fields: Any,
    ) -> str:
        """A page under ``("workspace",)``, ``("page", id)`` or ``("block", id)``; a child page block is added to its parent."""
        page_id = page_id or new_id()
        self.pages[page_id] = {
            "object": "page",
            "id": dashed(page_id),
            "created_time": created,
            "last_edited_time": edited,
            "in_trash": in_trash,
            "is_archived": is_archived,
            "parent": _parent_json(parent),
            "properties": {"title": {"id": "title", "type": "title", "title": [{"plain_text": title}]}},
            **fields,
        }
        if parent[0] in {"page", "block"}:
            self.children[parent[1]].append(page_id)
            self.blocks[page_id] = {"type": "child_page", "child_page": {"title": title}, "parent": parent}
        return page_id

    def add_block(
        self,
        parent: str,
        block_type: str,
        *,
        synced_from: str | None = None,
        block_id: str | None = None,
        has_children: bool | None = None,
    ) -> str:
        """A block appended to page or block ``parent``. ``has_children`` follows its children unless given."""
        block_id = block_id or new_id()
        body: dict[str, Any] = {}
        if block_type == "synced_block":
            body = {"synced_from": {"type": "block_id", "block_id": dashed(synced_from)} if synced_from else None}
        self.blocks[block_id] = {
            "type": block_type,
            block_type: body,
            "parent": ("page" if parent in self.pages else "block", parent),
            "has_children": has_children,
        }
        self.children[parent].append(block_id)
        return block_id

    def add_database(
        self,
        parent: str | None = None,
        *,
        sources: int = 1,
        parent_kind: str | None = None,
        named_by: str | None = None,
        created: str = DEFAULT_TIME,
        wiki: bool = False,
        database_id: str | None = None,
    ) -> tuple[str, list[str]]:
        """A database with ``sources`` data sources (``database_type: "wiki"`` when ``wiki``).

        Its parent is page or block ``parent`` (the workspace when ``None``;
        ``parent_kind="database"`` for a database nested in a wiki), and a
        ``child_database`` block is added under ``named_by`` (default: the
        parent). Pass ``named_by`` different from ``parent`` for a database
        shown inside the tree but living elsewhere.
        """
        database_id = database_id or new_id()
        source_ids = [new_id() for _ in range(sources)]
        if parent is None:
            parent_ref: tuple[str, ...] = ("workspace",)
        else:
            parent_ref = (parent_kind or ("page" if parent in self.pages else "block"), parent)
        self.databases[database_id] = {
            "object": "database",
            "id": dashed(database_id),
            "in_trash": False,
            "parent": _parent_json(parent_ref),
            "database_type": "wiki" if wiki else "database",
            "data_sources": [{"id": dashed(source_id), "name": f"Source {index}"} for index, source_id in enumerate(source_ids)],
        }
        for source_id in source_ids:
            self._add_data_source(source_id, database_id, created=created)
        container = named_by or parent
        if container is not None:
            self.blocks[database_id] = {"type": "child_database", "child_database": {"title": "Database"}, "parent": parent_ref}
            self.children[container].append(database_id)
        return database_id, source_ids

    def add_linked_view(self, host: str) -> str:
        """A linked view on page ``host``: a ``child_database`` whose database has no data sources."""
        database_id, _sources = self.add_database(host, sources=0)
        return database_id

    def add_row(
        self,
        database: str,
        source: str,
        title: str,
        *,
        created: str = DEFAULT_TIME,
        edited: str | None = None,
        is_archived: bool = False,
        in_trash: bool = False,
    ) -> str:
        """A row page of ``source`` in ``database``."""
        row_id = new_id()
        self.pages[row_id] = {
            "object": "page",
            "id": dashed(row_id),
            "created_time": created,
            "last_edited_time": edited or created,
            "in_trash": in_trash,
            "is_archived": is_archived,
            "parent": {"type": "data_source_id", "data_source_id": dashed(source), "database_id": dashed(database)},
            "properties": {"Name": {"id": "title", "type": "title", "title": [{"plain_text": title}]}},
        }
        self.results[source].append(row_id)
        return row_id

    def add_wiki_entry(self, wiki: str, source: str, title: str, *, created: str = DEFAULT_TIME) -> str:
        """An entry of wiki database ``wiki``: a result of its data source, with the wiki as its parent,
        and a ``child_page`` among the wiki database's child blocks."""
        entry_id = self.add_row(wiki, source, title, created=created)
        self.pages[entry_id]["parent"] = _parent_json(("database", wiki))
        self.blocks[entry_id] = {"type": "child_page", "child_page": {"title": title}, "parent": ("database", wiki)}
        self.children[wiki].append(entry_id)
        return entry_id

    def add_nested_database(self, wiki: str) -> tuple[str, str]:
        """A database nested in wiki database ``wiki``, as Notion answers it: its parent is the wiki,
        it is a ``child_database`` among the wiki's child blocks, and no query of the wiki returns it."""
        database_id, (source_id,) = self.add_database(wiki, parent_kind="database")
        return database_id, source_id

    def add_nested_data_source_result(self, wiki_source: str, *, created: str = DEFAULT_TIME) -> tuple[str, str]:
        """A nested database in the shape Notion's query reference documents and the live checks never saw:
        its data source is a result of the wiki's data source."""
        wiki_database = notion_id32(self.data_sources[wiki_source]["parent"]["database_id"])
        database_id, (source_id,) = self.add_database(wiki_database, parent_kind="database", created=created)
        self.blocks.pop(database_id)
        self.children[wiki_database].remove(database_id)
        self.results[wiki_source].append(source_id)
        return database_id, source_id

    def turn_into_wiki(self, page_id: str) -> str:
        """Turn page ``page_id`` into a wiki: the id becomes a wiki database with the page's parent
        and child blocks, and reading it as a page answers 400. Returns the wiki's data source."""
        page = self.pages.pop(page_id)
        database = {
            "object": "database",
            "id": dashed(page_id),
            "in_trash": page["in_trash"],
            "parent": page["parent"],
            "database_type": "wiki",
            "data_sources": [],
        }
        self.databases[page_id] = database
        source_id = new_id()
        database["data_sources"].append({"id": dashed(source_id), "name": "Wiki"})
        self._add_data_source(source_id, page_id, created=page["created_time"])
        if page_id in self.blocks:
            self.blocks[page_id] = {**self.blocks[page_id], "type": "child_database", "child_database": {"title": "Wiki"}}
            self.blocks[page_id].pop("child_page", None)
        return source_id

    def _add_data_source(self, source_id: str, database_id: str, *, created: str) -> None:
        self.data_sources[source_id] = {
            "object": "data_source",
            "id": dashed(source_id),
            "created_time": created,
            "last_edited_time": created,
            "in_trash": False,
            "parent": {"type": "database_id", "database_id": dashed(database_id)},
        }

    def fail(
        self,
        method: str,
        path: str,
        status: int,
        *,
        times: int = 1,
        json_body: Any = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Answer the next ``times`` requests of ``method`` on ``path`` with ``status``."""
        for _ in range(times):
            self.failures[(method, path)].append(httpx.Response(status, json=json_body or {"object": "error"}, headers=headers))

    # ------------------------------------------------------------------
    # Inspecting requests
    # ------------------------------------------------------------------

    def paths(self, method: str | None = None) -> list[str]:
        return [request.url.path for request in self.requests if method is None or request.method == method]

    def count(self, method: str, path: str) -> int:
        return sum(1 for request in self.requests if request.method == method and request.url.path == path)

    # ------------------------------------------------------------------
    # The API
    # ------------------------------------------------------------------

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        queued = self.failures.get((request.method, path))
        if queued:
            return queued.pop(0)
        if request.headers.get("Authorization") != f"Bearer {TOKEN}":
            return httpx.Response(401, json={"object": "error", "code": "unauthorized"})
        parts = path.strip("/").split("/")
        if request.method == "GET" and parts == ["v1", "users", "me"]:
            return httpx.Response(200, json={"object": "user", "id": dashed(new_id()), "type": "bot"})
        if request.method == "GET" and len(parts) == 3 and parts[1] == "pages":
            if notion_id32(parts[2]) in self.databases:
                return httpx.Response(
                    400,
                    json={
                        "object": "error",
                        "code": "validation_error",
                        "message": f"Provided ID {parts[2]} is a database, not a page. Use the retrieve database API instead.",
                    },
                )
            return self._object(self.pages.get(notion_id32(parts[2])))
        if request.method == "GET" and len(parts) == 3 and parts[1] == "blocks":
            return self._object(self._block(notion_id32(parts[2])))
        if request.method == "GET" and len(parts) == 4 and parts[1] == "blocks" and parts[3] == "children":
            return self._children(notion_id32(parts[2]), request)
        if request.method == "GET" and len(parts) == 3 and parts[1] == "databases":
            return self._object(self.databases.get(notion_id32(parts[2])))
        if request.method == "GET" and len(parts) == 3 and parts[1] == "data_sources":
            return self._object(self.data_sources.get(notion_id32(parts[2])))
        if request.method == "POST" and len(parts) == 4 and parts[1] == "data_sources" and parts[3] == "query":
            return self._query(notion_id32(parts[2]), json.loads(request.content or b"{}"))
        return httpx.Response(400, json={"object": "error", "code": "invalid_request_url"})

    @staticmethod
    def _object(record: dict[str, Any] | None) -> httpx.Response:
        if record is None:
            return httpx.Response(404, json={"object": "error", "code": "object_not_found"})
        return httpx.Response(200, json=record)

    def _block(self, block_id: str) -> dict[str, Any] | None:
        block = self.blocks.get(block_id)
        if block is None:
            return None
        kind, parent_id = block["parent"][0], block["parent"][-1]
        has_children = block.get("has_children")
        if has_children is None:
            has_children = bool(self.children.get(block_id))
        record = {
            "object": "block",
            "id": dashed(block_id),
            "type": block["type"],
            block["type"]: block[block["type"]],
            "has_children": has_children,
            "in_trash": False,
            "parent": _parent_json((kind, parent_id) if kind != "workspace" else ("workspace",)),
        }
        return record

    def _children(self, container: str, request: httpx.Request) -> httpx.Response:
        if container not in self.pages and container not in self.blocks and container not in self.databases:
            return httpx.Response(404, json={"object": "error", "code": "object_not_found"})
        if self.cursors_expire and request.url.params.get("start_cursor"):
            return httpx.Response(400, json={"object": "error", "code": "validation_error"})
        child_ids = self.children.get(container, [])
        start = int(request.url.params.get("start_cursor", "0"))
        size = int(request.url.params.get("page_size", "100"))
        window = child_ids[start : start + size]
        more = start + size < len(child_ids)
        return httpx.Response(
            200,
            json={
                "object": "list",
                "results": [self._block(child_id) for child_id in window],
                "has_more": more,
                "next_cursor": str(start + size) if more else None,
                "type": "block",
            },
        )

    def _query(self, source_id: str, body: dict[str, Any]) -> httpx.Response:
        if source_id not in self.data_sources:
            return httpx.Response(404, json={"object": "error", "code": "object_not_found"})
        records = []
        for result_id in self.results.get(source_id, []):
            record = self.pages.get(result_id) or self.data_sources[result_id]
            if record["object"] == "page" and (record["in_trash"] or record["is_archived"]):
                continue
            records.append(record)
        on_or_after = ((body.get("filter") or {}).get("created_time") or {}).get("on_or_after")
        if on_or_after:
            records = [record for record in records if _instant(record["created_time"]) >= _instant(on_or_after)]
        records.sort(key=lambda record: _instant(record["created_time"]))
        incomplete_reason = self.incomplete_reason
        if len(records) > self.result_cap:
            records = records[: self.result_cap]
            incomplete_reason = incomplete_reason or "query_result_limit_reached"
        start = int(body.get("start_cursor") or 0)
        size = int(body.get("page_size") or 100)
        window = records[start : start + size]
        more = start + size < len(records)
        answer: dict[str, Any] = {
            "object": "list",
            "results": window,
            "has_more": more,
            "next_cursor": str(start + size) if more else None,
            "type": "page_or_data_source",
        }
        if incomplete_reason:
            answer["request_status"] = {"type": "incomplete", "incomplete_reason": incomplete_reason}
        return httpx.Response(200, json=answer)


def notion_id32(value: str) -> str:
    return value.replace("-", "").lower()


def _parent_json(parent: tuple[str, ...]) -> dict[str, Any]:
    kind = parent[0]
    if kind == "workspace":
        return {"type": "workspace", "workspace": True}
    key = {"page": "page_id", "block": "block_id", "database": "database_id"}[kind]
    return {"type": key, key: dashed(parent[1])}


def _instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
