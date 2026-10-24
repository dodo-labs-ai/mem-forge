"""The Notion Gene's fetch: the page body, left-out blocks, row properties, files and the raw envelope."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from weakref import WeakValueDictionary

import httpx
import pytest

from memforge.genes import notion_gene
from memforge.genes.notion_files import notion_file_reference
from memforge.genes.notion_gene import NotionGene, notion_raw_envelope
from memforge.models import ContentItem, RawContent, require_attested_content
from memforge.source_artifacts import RawSourceArtifact
from tests.connector_http_fixture import FakeConnectorHttp
from tests.notion_fake import FILE_HOST, TOKEN, NotionFake, dashed, new_id

pytestmark = pytest.mark.asyncio


@pytest.fixture(autouse=True)
def own_rate_limiters(monkeypatch):
    monkeypatch.setattr("memforge.genes.toolkit.pacing._SHARED_LIMITERS", WeakValueDictionary())


async def connect(fake: NotionFake, root: str) -> tuple[NotionGene, FakeConnectorHttp]:
    gene = NotionGene({"token": TOKEN, "root_pages": [root]}, "notion-test")
    http = FakeConnectorHttp(fake.handle)
    with http.serving(notion_gene):
        await gene.authenticate()
    return gene, http


def item_for(page_id: str) -> ContentItem:
    return ContentItem(
        item_id=f"notion-{page_id}",
        title="Page",
        source_url=f"https://app.notion.com/p/{page_id}",
        last_modified=datetime(2026, 3, 1, 9, 0, tzinfo=timezone.utc),
        content_type="text/markdown",
        version="2026-03-01T09:00:00.000Z",
        extra={"page_id": page_id},
    )


async def fetch(fake: NotionFake, page_id: str) -> tuple[RawContent, NotionGene]:
    gene, _http = await connect(fake, page_id)
    return await gene.fetch(item_for(page_id)), gene


def envelope(raw: RawContent) -> dict:
    return json.loads(raw.body)


def unknown(page_id: str, block_id: str, alt: str | None = None) -> str:
    alt_attribute = f' alt="{alt}"' if alt else ""
    return f'<unknown url="https://app.notion.com/p/{page_id}#{block_id}"{alt_attribute}/>'


def markdown_reads(fake: NotionFake) -> list[str]:
    return [request.url.path for request in fake.requests if request.url.path.endswith("/markdown")]


def a_row(fake: NotionFake, properties: dict | None = None) -> str:
    root = fake.add_page("Root")
    database, (source,) = fake.add_database(root)
    row = fake.add_row(database, source, "Row")
    if properties is not None:
        fake.pages[row]["properties"] = properties
    return row


# ---------------------------------------------------------------------------
# The body
# ---------------------------------------------------------------------------


async def test_fetch_reads_the_body_from_the_markdown_endpoint_at_v1() -> None:
    fake = NotionFake()
    page = fake.add_page("Handbook")
    fake.set_markdown(page, "# Handbook\n\nShip on Friday.")

    raw, _gene = await fetch(fake, page)

    assert envelope(raw)["markdown"] == "# Handbook\n\nShip on Friday."
    (request,) = [request for request in fake.requests if request.url.path == f"/v1/pages/{page}/markdown"]
    assert request.url.params["markdown_version"] == "v1"


async def test_fetch_pins_include_transcript() -> None:
    fake = NotionFake()
    page = fake.add_page("Meeting")

    await fetch(fake, page)

    (request,) = [request for request in fake.requests if request.url.path == f"/v1/pages/{page}/markdown"]
    assert request.url.params["include_transcript"] == "false"


async def test_each_unknown_block_is_read_again_and_spliced_in_place() -> None:
    fake = NotionFake()
    page = fake.add_page("Large")
    first, second = new_id(), new_id()
    fake.set_markdown(
        page,
        f"Intro\n{unknown(page, first)}\nMiddle\n<details>\n<summary>T</summary>\n\t{unknown(page, second)}\n</details>",
        truncated=True,
        unknown_block_ids=[first, second],
    )
    fake.set_markdown(first, "First block")
    fake.set_markdown(second, "Second block\n\tIts child")

    raw, _gene = await fetch(fake, page)

    body = envelope(raw)
    assert body["markdown"] == (
        "Intro\nFirst block\nMiddle\n<details>\n<summary>T</summary>\n\tSecond block\n\t\tIts child\n</details>"
    )
    assert body["truncated"] is False


async def test_unknown_blocks_inside_a_re_read_are_read_again() -> None:
    fake = NotionFake()
    page = fake.add_page("Large")
    outer, inner = new_id(), new_id()
    fake.set_markdown(page, unknown(page, outer), truncated=True, unknown_block_ids=[outer])
    fake.set_markdown(outer, f"Outer\n{unknown(outer, inner)}", truncated=True, unknown_block_ids=[inner])
    fake.set_markdown(inner, "Inner")

    raw, _gene = await fetch(fake, page)

    assert envelope(raw)["markdown"] == "Outer\nInner"
    assert envelope(raw)["truncated"] is False


async def test_a_page_whose_unknown_blocks_all_resolve_is_complete() -> None:
    fake = NotionFake()
    page = fake.add_page("Large")
    ids = [new_id() for _ in range(3)]
    fake.set_markdown(page, "\n".join(unknown(page, block_id) for block_id in ids), truncated=True, unknown_block_ids=ids)
    for index, block_id in enumerate(ids):
        fake.set_markdown(block_id, f"Block {index}")

    raw, _gene = await fetch(fake, page)

    assert envelope(raw)["truncated"] is False
    assert envelope(raw)["markdown"] == "Block 0\nBlock 1\nBlock 2"


async def test_an_unshared_child_block_does_not_mark_the_page_partial() -> None:
    fake = NotionFake()
    page = fake.add_page("Partly shared")
    hidden = new_id()
    fake.set_markdown(page, f"Visible\n{unknown(page, hidden)}", truncated=True, unknown_block_ids=[hidden])

    raw, _gene = await fetch(fake, page)

    body = envelope(raw)
    assert body["markdown"] == "Visible\n"
    assert body["truncated"] is False
    assert body["unknown_not_found"] == 1


async def test_unknown_block_reads_beyond_their_bound_mark_the_page_truncated() -> None:
    fake = NotionFake()
    page = fake.add_page("Huge")
    ids = [new_id() for _ in range(201)]
    fake.set_markdown(page, "\n".join(unknown(page, block_id) for block_id in ids), truncated=True, unknown_block_ids=ids)
    for block_id in ids:
        fake.set_markdown(block_id, "Text")

    raw, _gene = await fetch(fake, page)

    assert envelope(raw)["truncated"] is True
    assert len(markdown_reads(fake)) == 1 + 200


async def test_a_truncated_answer_with_nothing_to_follow_marks_the_page_truncated() -> None:
    fake = NotionFake()
    page = fake.add_page("Cut")
    fake.set_markdown(page, "Everything Notion sent", truncated=True)

    raw, _gene = await fetch(fake, page)

    assert envelope(raw)["truncated"] is True


async def test_a_block_cut_again_by_its_own_re_read_marks_the_page_truncated() -> None:
    fake = NotionFake()
    page = fake.add_page("Cut twice")
    block = new_id()
    fake.set_markdown(page, unknown(page, block), truncated=True, unknown_block_ids=[block])
    fake.set_markdown(block, unknown(block, block), truncated=True, unknown_block_ids=[block])

    raw, _gene = await fetch(fake, page)

    assert envelope(raw)["truncated"] is True
    assert len(markdown_reads(fake)) == 2


@pytest.mark.parametrize(
    ("alt", "block_type", "content", "link"),
    [
        (
            "bookmark",
            "bookmark",
            {"url": "https://example.com/spec", "caption": [{"plain_text": "The spec"}]},
            "[The spec](https://example.com/spec)",
        ),
        ("bookmark", "bookmark", {"url": "https://example.com/spec", "caption": []}, "[https://example.com/spec](https://example.com/spec)"),
        ("external_object_instance", "link_preview", {"url": "https://github.com/o/r/pull/1"}, "[https://github.com/o/r/pull/1](https://github.com/o/r/pull/1)"),
        ("link_preview", "link_preview", {"url": "https://figma.com/file/x"}, "[https://figma.com/file/x](https://figma.com/file/x)"),
    ],
)
async def test_a_bookmark_is_read_through_the_block_api_and_becomes_a_link_to_its_target(
    alt: str, block_type: str, content: dict, link: str
) -> None:
    fake = NotionFake()
    page = fake.add_page("Links")
    block = fake.add_block(page, block_type, content=content)
    fake.set_markdown(page, f"See {unknown(page, block, alt)}", truncated=alt != "bookmark", unknown_block_ids=[] if alt == "bookmark" else [block])

    raw, _gene = await fetch(fake, page)

    body = envelope(raw)
    assert body["markdown"] == f"See {link}"
    assert "app.notion.com" not in body["markdown"]
    assert body["truncated"] is False
    assert fake.count("GET", f"/v1/blocks/{block}") == 1
    assert len(markdown_reads(fake)) == 1


async def test_an_embed_renders_from_its_own_tag_without_a_read() -> None:
    fake = NotionFake()
    page = fake.add_page("Embed")
    fake.set_markdown(page, '<embed src="https://youtube.com/watch?v=1"></embed>')

    raw, gene = await fetch(fake, page)

    assert [path for path in fake.paths() if "/blocks/" in path] == []
    assert "[https://youtube.com/watch?v=1](https://youtube.com/watch?v=1)" in (await gene.normalize(raw)).markdown_body


@pytest.mark.parametrize("listed", [True, False])
@pytest.mark.parametrize("alt", ["breadcrumb", "button", "template", "unsupported"])
async def test_an_unsupported_block_does_not_mark_the_page_partial(alt: str, listed: bool) -> None:
    fake = NotionFake()
    page = fake.add_page("Unsupported")
    block = new_id()
    fake.set_markdown(page, f"Before\n{unknown(page, block, alt)}\nAfter", truncated=listed, unknown_block_ids=[block] if listed else [])

    raw, _gene = await fetch(fake, page)

    body = envelope(raw)
    assert body["markdown"] == f"Before\n[Notion block not shown: {alt}]\nAfter"
    assert body["truncated"] is False
    assert len(markdown_reads(fake)) == 1
    assert [path for path in fake.paths() if "/blocks/" in path] == []


async def test_a_link_preview_the_block_api_shows_as_unsupported_is_named_by_its_type() -> None:
    fake = NotionFake()
    page = fake.add_page("Preview")
    block = fake.add_block(page, "unsupported", content={"block_type": "external_object_instance"})
    fake.set_markdown(page, unknown(page, block, "external_object_instance"), truncated=True, unknown_block_ids=[block])

    raw, _gene = await fetch(fake, page)

    assert envelope(raw)["markdown"] == "[Notion block not shown: external_object_instance]"
    assert envelope(raw)["truncated"] is False


async def test_a_bookmark_that_answers_404_leaves_a_placeholder_and_is_not_partial() -> None:
    fake = NotionFake()
    page = fake.add_page("Gone bookmark")
    block = new_id()
    fake.set_markdown(page, unknown(page, block, "bookmark"))

    raw, _gene = await fetch(fake, page)

    body = envelope(raw)
    assert body["markdown"] == "[Notion block not shown: bookmark]"
    assert body["truncated"] is False
    assert body["unknown_not_found"] == 1


async def test_a_bookmark_read_that_fails_fails_the_item() -> None:
    fake = NotionFake()
    page = fake.add_page("Bookmark")
    block = fake.add_block(page, "bookmark", content={"url": "https://example.com"})
    fake.set_markdown(page, unknown(page, block, "bookmark"))
    fake.fail("GET", f"/v1/blocks/{block}", 403)
    gene, _http = await connect(fake, page)

    with pytest.raises(httpx.HTTPStatusError):
        await gene.fetch(item_for(page))


@pytest.mark.parametrize("alt", ["breadcrumb", None])
async def test_an_unsupported_block_tag_without_a_readable_id_fails_the_item(alt: str | None) -> None:
    fake = NotionFake()
    page = fake.add_page("Broken")
    alt_attribute = f' alt="{alt}"' if alt else ""
    fake.set_markdown(page, f'<unknown url="https://app.notion.com/p/not-an-id"{alt_attribute}/>', truncated=True)
    gene, _http = await connect(fake, page)

    with pytest.raises(RuntimeError, match="block id"):
        await gene.fetch(item_for(page))


async def test_a_page_trashed_since_discovery_fails_the_item() -> None:
    fake = NotionFake()
    root = fake.add_page("Root")
    page = fake.add_page("Trashed", parent=("page", root))
    gene, _http = await connect(fake, root)
    fake.pages[page]["in_trash"] = True
    fake.set_markdown(page, "Notion still serves a trashed page's body")

    with pytest.raises(RuntimeError, match="trash"):
        await gene.fetch(item_for(page))


# ---------------------------------------------------------------------------
# The raw envelope and empty bodies
# ---------------------------------------------------------------------------


async def test_the_raw_body_is_the_versioned_envelope() -> None:
    fake = NotionFake()
    page = fake.add_page("Plain page")
    fake.set_markdown(page, "Body")
    row = a_row(fake, {"Name": {"id": "title", "type": "title", "title": [{"plain_text": "Row"}]}})
    fake.set_markdown(row, "Row body")

    page_raw, _gene = await fetch(fake, page)
    row_raw, _gene = await fetch(fake, row)

    assert page_raw.content_type == "application/json"
    assert envelope(page_raw) == {"notion_raw": 1, "markdown": "Body", "truncated": False, "unknown_not_found": 0}
    assert envelope(row_raw) == {
        "notion_raw": 1,
        "markdown": "Row body",
        "properties": [{"name": "Name", "type": "title", "value": "Row"}],
        "truncated": False,
        "unknown_not_found": 0,
    }
    assert notion_raw_envelope(row_raw) == envelope(row_raw)


@pytest.mark.parametrize(
    "body",
    [
        b'{"notion_raw": 2, "markdown": "", "truncated": false, "unknown_not_found": 0}',
        b'{"notion_raw": 1, "markdown": 3, "truncated": false, "unknown_not_found": 0}',
        b'{"notion_raw": 1, "markdown": "", "truncated": false, "unknown_not_found": 0, "properties": [{"name": "A"}]}',
        b"not json",
    ],
)
async def test_an_envelope_of_another_version_or_shape_fails_the_item(body: bytes) -> None:
    gene = NotionGene({"token": TOKEN, "root_pages": [new_id()]}, "notion-test")
    raw = RawContent(item=item_for(new_id()), body=body, content_type="application/json")

    with pytest.raises(ValueError, match="envelope"):
        await gene.normalize(raw)


@pytest.mark.parametrize("markdown", ["", "<empty-block/>\n<table_of_contents/>\n"])
async def test_an_empty_page_is_attested_empty(markdown: str) -> None:
    fake = NotionFake()
    page = fake.add_page("Blank")
    fake.set_markdown(page, markdown)

    raw, gene = await fetch(fake, page)

    assert raw.authoritative_empty is True
    assert raw.empty_evidence == "notion_markdown_api_successful_empty_page_body"
    normalized = await gene.normalize(raw)
    assert normalized.markdown_body == ""
    require_attested_content(raw, normalized)


async def test_a_truncated_empty_answer_is_not_attested_empty() -> None:
    fake = NotionFake()
    page = fake.add_page("Cut and blank")
    fake.set_markdown(page, "", truncated=True)

    raw, gene = await fetch(fake, page)

    assert raw.authoritative_empty is False
    assert raw.empty_evidence is None
    with pytest.raises(ValueError):
        require_attested_content(raw, await gene.normalize(raw))


async def test_a_row_body_holds_no_properties() -> None:
    fake = NotionFake()
    row = a_row(
        fake,
        {
            "Name": {"id": "title", "type": "title", "title": [{"plain_text": "Launch"}]},
            "Status": {"id": "s", "type": "status", "status": {"name": "Shipped"}},
        },
    )

    raw, gene = await fetch(fake, row)
    normalized = await gene.normalize(raw)

    assert normalized.markdown_body == ""
    assert "Shipped" not in raw.body.decode().split('"properties"')[0]
    assert envelope(raw)["properties"][1] == {"name": "Status", "type": "status", "value": "Shipped"}
    assert raw.authoritative_empty is True
    require_attested_content(raw, normalized)


async def test_normalize_reads_the_envelope_and_the_page_time() -> None:
    fake = NotionFake()
    page = fake.add_page("Callout")
    fake.set_markdown(page, '<callout icon="💡">\n\tShip on Friday\n</callout>')

    raw, gene = await fetch(fake, page)
    normalized = await gene.normalize(raw)

    assert normalized.markdown_body == "> 💡 Ship on Friday"
    assert normalized.source_semantics == {"source_updated_at": "2026-03-01T09:00:00+00:00"}


# ---------------------------------------------------------------------------
# Row properties
# ---------------------------------------------------------------------------


async def test_a_property_cut_at_25_references_is_read_in_full() -> None:
    fake = NotionFake()
    related = [new_id() for _ in range(30)]
    people = [{"object": "user", "id": dashed(new_id()), "name": f"User {index}"} for index in range(30)]
    row = a_row(
        fake,
        {
            "Name": {"id": "title", "type": "title", "title": [{"plain_text": "Row"}]},
            "Links": {"id": "rel%3A", "type": "relation", "relation": [{"id": dashed(page)} for page in related[:25]], "has_more": True},
            "Team": {"id": "ppl", "type": "people", "people": people[:25]},
            "Few": {"id": "few", "type": "relation", "relation": [{"id": dashed(related[0])}], "has_more": False},
        },
    )
    fake.set_property_items(row, "rel%3A", [{"type": "relation", "relation": {"id": dashed(page)}} for page in related])
    fake.set_property_items(row, "ppl", [{"type": "people", "people": person} for person in people])

    raw, _gene = await fetch(fake, row)

    values = {record["name"]: record["value"] for record in envelope(raw)["properties"]}
    assert values["Links"] == ", ".join(related)
    assert values["Team"] == ", ".join(f"User {index}" for index in range(30))
    assert values["Few"] == related[0]
    assert not [request for request in fake.requests if request.url.path.endswith("/properties/few")]


async def test_property_values_are_rendered_into_records_sorted_by_name() -> None:
    fake = NotionFake()
    row = a_row(
        fake,
        {
            "Zeta": {"id": "z", "type": "checkbox", "checkbox": False},
            "Name": {"id": "title", "type": "title", "title": [{"plain_text": "Ro"}, {"plain_text": "w"}]},
            "Alpha": {"id": "a", "type": "button", "button": {}},
        },
    )

    raw, _gene = await fetch(fake, row)

    assert envelope(raw)["properties"] == [
        {"name": "Alpha", "type": "button", "value": ""},
        {"name": "Name", "type": "title", "value": "Row"},
        {"name": "Zeta", "type": "checkbox", "value": "false"},
    ]


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


async def test_a_signed_file_url_becomes_a_stable_artifact_reference() -> None:
    fake = NotionFake()
    page = fake.add_page("Media")
    url = fake.add_file("diagram.png", b"\x89PNG")
    key = url.split("?")[0].removeprefix(f"https://{FILE_HOST}/")
    fake.set_markdown(page, f"![Diagram]({url})")

    raw, _gene = await fetch(fake, page)

    assert envelope(raw)["markdown"] == f"![Diagram](notion-file:{key})"
    assert raw.artifacts == (
        RawSourceArtifact(
            provider_key=key,
            parent_observation_type="page_body",
            parent_provider_key=f"{page}:body",
            provider_revision=key,
            filename="diagram.png",
            media_type="image/png",
            locator={"file_key": key},
        ),
    )


async def test_a_files_property_artifact_belongs_to_the_properties_observation() -> None:
    fake = NotionFake()
    url = fake.add_file("Q3%20plan.pdf", b"%PDF", content_type="application/pdf")
    key = url.split("?")[0].removeprefix(f"https://{FILE_HOST}/")
    row = a_row(
        fake,
        {"Attachments": {"id": "f", "type": "files", "files": [{"name": "Q3 plan.pdf", "type": "file", "file": {"url": url}}]}},
    )

    raw, _gene = await fetch(fake, row)

    (artifact,) = raw.artifacts
    assert (artifact.parent_observation_type, artifact.parent_provider_key) == ("properties", f"{row}:properties")
    assert (artifact.filename, artifact.media_type) == ("Q3 plan.pdf", "application/pdf")
    assert envelope(raw)["properties"] == [{"name": "Attachments", "type": "files", "value": f"notion-file:{key}"}]


async def test_a_file_referenced_twice_yields_one_artifact() -> None:
    fake = NotionFake()
    url = fake.add_file("diagram.png", b"\x89PNG")
    path = url.split("?")[0].removeprefix(f"https://{FILE_HOST}")
    row = a_row(
        fake,
        {"Files": {"id": "f", "type": "files", "files": [{"name": "diagram.png", "type": "file", "file": {"url": fake.sign(path)}}]}},
    )
    fake.set_markdown(row, f"![One]({url})\n<file src=\"{url.replace('&', '&amp;')}\">Again</file>")

    raw, _gene = await fetch(fake, row)

    (artifact,) = raw.artifacts
    assert artifact.parent_observation_type == "page_body"
    assert artifact.parent_provider_key == f"{row}:body"


async def test_a_file_of_a_type_memforge_does_not_store_stays_a_reference_without_an_artifact() -> None:
    fake = NotionFake()
    page = fake.add_page("Archive")
    url = fake.add_file("bundle.zip", b"PK", content_type="application/zip")
    fake.set_markdown(page, f'<file src="{url}">bundle</file>')

    raw, _gene = await fetch(fake, page)

    assert raw.artifacts == ()
    assert "notion-file:" in envelope(raw)["markdown"]


async def test_the_stored_body_holds_no_url_signature() -> None:
    fake = NotionFake()
    page = fake.add_page("Signatures")
    notion_url = fake.add_file("photo.jpg", b"JPEG", content_type="image/jpeg")
    other = "https://other-bucket.s3.amazonaws.com/x/y.png?X-Amz-Signature=abc&X-Amz-Credential=c"
    row = a_row(
        fake,
        {"Files": {"id": "f", "type": "files", "files": [{"name": "y", "type": "external", "external": {"url": other}}]}},
    )
    fake.set_markdown(page, f"![Photo]({notion_url}) and [mirror]({other})")
    fake.set_markdown(row, f'<pdf src="{other}"></pdf>')

    page_raw, _gene = await fetch(fake, page)
    row_raw, _gene = await fetch(fake, row)

    for raw in (page_raw, row_raw):
        assert "x-amz" not in raw.body.decode().lower()
        for artifact in raw.artifacts:
            assert "x-amz" not in json.dumps(dict(artifact.locator)).lower()
    assert "[mirror](https://other-bucket.s3.amazonaws.com/x/y.png)" in envelope(page_raw)["markdown"]


async def test_an_external_file_stays_a_link() -> None:
    fake = NotionFake()
    row = a_row(
        fake,
        {"Files": {"id": "f", "type": "files", "files": [{"name": "Spec", "type": "external", "external": {"url": "https://example.com/spec.pdf"}}]}},
    )
    fake.set_markdown(row, "![Logo](https://example.com/logo.png)")

    raw, _gene = await fetch(fake, row)

    assert raw.artifacts == ()
    assert envelope(raw)["markdown"] == "![Logo](https://example.com/logo.png)"
    assert envelope(raw)["properties"][0]["value"] == "https://example.com/spec.pdf"


async def test_an_artifact_is_streamed_without_the_notion_token() -> None:
    fake = NotionFake()
    page = fake.add_page("Media")
    fake.set_markdown(page, f"![Diagram]({fake.add_file('diagram.png', b'PNG bytes')})")
    gene, http = await connect(fake, page)
    raw = await gene.fetch(item_for(page))

    async with gene.open_source_artifact(raw.artifacts[0]) as download:
        content = b"".join([chunk async for chunk in download.chunks])
        media_type = download.media_type

    assert (content, media_type) == (b"PNG bytes", "image/png")
    (file_request,) = [request for request in fake.requests if request.url.host == FILE_HOST]
    assert "Authorization" not in file_request.headers
    file_client = next(client for client, options in zip(http.clients, http.options) if options["product_name"] == "Notion files")
    assert gene._file_client is file_client
    assert "Authorization" not in (http.options[http.clients.index(file_client)]["headers"] or {})


@pytest.mark.parametrize("status", [400, 403])
async def test_a_file_host_refusal_fails_the_item(status: int) -> None:
    fake = NotionFake()
    page = fake.add_page("Media")
    url = fake.add_file("diagram.png", b"PNG")
    fake.set_markdown(page, f"![Diagram]({url})")
    gene, _http = await connect(fake, page)
    raw = await gene.fetch(item_for(page))
    fake.failures[("GET", url.split("?")[0].removeprefix(f"https://{FILE_HOST}"))].append(
        httpx.Response(status, text="<Error><Code>AccessDenied</Code></Error>")
    )

    with pytest.raises(RuntimeError, match=f"answered {status}") as raised:
        async with gene.open_source_artifact(raw.artifacts[0]):
            pass

    assert "X-Amz" not in str(raised.value)


async def test_an_expired_signature_fails_and_a_fresh_fetch_streams_the_file() -> None:
    fake = NotionFake()
    page = fake.add_page("Media")
    url = fake.add_file("diagram.png", b"PNG")
    fake.set_markdown(page, f"![Diagram]({url})")
    gene, _http = await connect(fake, page)
    raw = await gene.fetch(item_for(page))
    path = url.split("?")[0].removeprefix(f"https://{FILE_HOST}")
    fake.set_markdown(page, f"![Diagram]({fake.sign(path)})")  # the old signature stops working

    with pytest.raises(RuntimeError, match="answered 403"):
        async with gene.open_source_artifact(raw.artifacts[0]):
            pass
    raw = await gene.fetch(item_for(page))
    async with gene.open_source_artifact(raw.artifacts[0]) as download:
        assert b"".join([chunk async for chunk in download.chunks]) == b"PNG"


async def test_an_artifact_this_connection_did_not_fetch_fails_the_item() -> None:
    fake = NotionFake()
    page = fake.add_page("Media")
    gene, _http = await connect(fake, page)
    artifact = RawSourceArtifact(
        provider_key="a/b/c.png",
        parent_observation_type="page_body",
        parent_provider_key=f"{page}:body",
        provider_revision="a/b/c.png",
        filename="c.png",
        media_type="image/png",
    )

    with pytest.raises(RuntimeError, match="no signed URL"):
        async with gene.open_source_artifact(artifact):
            pass


async def test_close_releases_the_api_and_file_clients() -> None:
    fake = NotionFake()
    page = fake.add_page("Media")
    fake.set_markdown(page, f"![Diagram]({fake.add_file('diagram.png', b'PNG')})")
    gene, http = await connect(fake, page)
    raw = await gene.fetch(item_for(page))

    await gene.close()

    assert len(http.clients) == 2
    assert all(client.is_closed for client in http.clients)
    assert gene._signed_files == {}
    with pytest.raises(RuntimeError, match="no signed URL"):
        async with gene.open_source_artifact(raw.artifacts[0]):
            pass


@pytest.mark.parametrize(
    ("url", "key"),
    [
        (
            "https://prod-files-secure.s3.us-west-2.amazonaws.com/4c1e2f3a-5b6c-4d7e-8f90-a1b2c3d4e5f6/"
            "9a7b6c5d-4e3f-4a1b-8c9d-0e1f2a3b4c5d/diagram.png?X-Amz-Signature=abc",
            "4c1e2f3a-5b6c-4d7e-8f90-a1b2c3d4e5f6/9a7b6c5d-4e3f-4a1b-8c9d-0e1f2a3b4c5d/diagram.png",
        ),
        (
            "https://s3.us-west-2.amazonaws.com/secure.notion-static.com/9a7b6c5d-4e3f-4a1b-8c9d-0e1f2a3b4c5d/a.pdf?X-Amz-Signature=abc",
            "secure.notion-static.com/9a7b6c5d-4e3f-4a1b-8c9d-0e1f2a3b4c5d/a.pdf",
        ),
        ("https://prod-files-secure.s3.us-west-2.amazonaws.com/not/a/notion-file.png", None),
        ("https://example.com/4c1e2f3a-5b6c-4d7e-8f90-a1b2c3d4e5f6/9a7b6c5d-4e3f-4a1b-8c9d-0e1f2a3b4c5d/x.png", None),
    ],
)
async def test_notion_file_urls_are_recognized_by_host_and_path_shape(url: str, key: str | None) -> None:
    assert notion_file_reference(url) == key
