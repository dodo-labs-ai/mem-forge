from __future__ import annotations

import ssl
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import httpx
import pytest

from memforge.genes import confluence_gene
from memforge.genes.base import SourceConfigurationError
from memforge.genes.confluence_gene import ConfluenceGene, PREVIEW_DISCOVERY_LIMIT_CONFIG_KEY
from memforge.models import AbsenceAnswer, ConfigCondition, ContentItem
from memforge.source_artifacts import RawSourceArtifact, SourceArtifactContractError
from tests.connector_http_fixture import AnsweringClient, FakeConnectorHttp
from tests.tls_bundle_fixture import BUNDLE_CA_COUNT, trusted_ca_count, write_ca_bundle


@pytest.fixture(autouse=True)
def unpaced_origin_requests(monkeypatch):
    """Give each test its own origin limiters that leave no gap between requests."""
    monkeypatch.setattr("memforge.genes.atlassian_auth._ATLASSIAN_REQUEST_LIMITERS", {})
    monkeypatch.setattr("memforge.genes.confluence_gene.CONFLUENCE_REQUEST_INTERVAL_SECONDS", 0.0)


def test_confluence_schema_hides_runtime_transport_fields_from_ui():
    fields = {field.key: field for field in ConfluenceGene.config_schema().fields}

    assert fields["pat"].required is True
    assert fields["pat"].advanced is False
    assert fields["base_url"].label == "Wiki URL"
    assert fields["spaces"].required is True
    assert fields["spaces"].visible_when == (ConfigCondition("sync_mode", ("space",)),)
    assert fields["sync_mode"].default == "space"
    assert fields["sync_mode"].advanced is False
    assert "api_prefix" not in fields
    assert "tls_ca_bundle" not in fields


def test_confluence_normalizes_corporate_wiki_page_url_to_page_tree_scope():
    config = {
        "base_url": "https://wiki.company.example/wiki/spaces/PAY/pages/5695886009/Flexible+Payroll",
        "pat": "confluence-pat",
    }

    ConfluenceGene.normalize_config(config)

    assert config["base_url"] == "https://wiki.company.example"
    assert config["api_prefix"] == "/wiki"
    assert config["spaces"] == ["PAY"]
    assert config["page_tree_root"] == "5695886009"
    assert config["sync_mode"] == "page_tree"


def test_confluence_normalizes_space_url_without_page_tree_root():
    config = {
        "base_url": "https://team.atlassian.net/wiki/spaces/ENG",
        "pat": "confluence-pat",
    }

    ConfluenceGene.normalize_config(config)

    assert config["base_url"] == "https://team.atlassian.net"
    assert config["api_prefix"] == "/wiki"
    assert config["spaces"] == ["ENG"]
    assert config.get("page_tree_root") is None
    assert config["sync_mode"] == "space"


def _space_probe_http(space_status: int = 200, *, root_space_status: int = 200) -> tuple[list[str], FakeConnectorHttp]:
    """A fake Confluence answering the REST probe under ``/wiki`` with ``space_status`` and at the root with ``root_space_status``."""
    paths: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        status = space_status if request.url.path.startswith("/wiki/") else root_space_status
        return httpx.Response(status, json={"results": []} if status == 200 else {})

    return paths, FakeConnectorHttp(handle)


class JsonResponse(httpx.Response):
    def __init__(self, payload: dict, status_code: int = 200, headers: dict | None = None):
        super().__init__(status_code, json=payload, headers=headers)


class BinaryResponse(httpx.Response):
    def __init__(self, content: bytes, content_type: str):
        super().__init__(200, content=content, headers={"content-type": content_type})


class NotFoundClient(AnsweringClient):
    async def answer(self, _method: str, _url: str, **_kwargs):
        return JsonResponse({}, status_code=404)


class PageTreeClient(AnsweringClient):
    def __init__(self) -> None:
        super().__init__(base_url="https://wiki.example.com")
        self.calls: list[str] = []

    async def answer(self, _method: str, url: str, **_kwargs):
        self.calls.append(url)
        if url.endswith("/content/root"):
            return JsonResponse(_page("root", "Root", "2026-05-20T00:00:00Z"))
        if url.endswith("/content/root/child/page"):
            return JsonResponse(
                _page_batch([_page("parent", "Unchanged Parent", "2026-05-20T00:00:00Z")])
            )
        if url.endswith("/content/parent/child/page"):
            return JsonResponse(
                _page_batch([_page("target", "Changed Child", "2026-05-26T14:51:21Z")])
            )
        if url.endswith("/content/target/child/page"):
            return JsonResponse(_page_batch([]))
        return JsonResponse(_page_batch([]))


class PreviewLimitClient(AnsweringClient):
    def __init__(self) -> None:
        super().__init__(base_url="https://wiki.example.com")
        self.requests: list[tuple[str, dict]] = []

    async def answer(self, _method: str, url: str, **kwargs):
        self.requests.append((url, kwargs.get("params") or {}))
        if url.endswith("/content/root"):
            return JsonResponse(_page("root", "Root", "2026-05-20T00:00:00Z"))
        if url.endswith("/content/root/child/page"):
            return JsonResponse(
                _page_batch(
                    [
                        _page("child-1", "Child 1", "2026-05-21T00:00:00Z"),
                        _page("child-2", "Child 2", "2026-05-22T00:00:00Z"),
                        _page("child-3", "Child 3", "2026-05-23T00:00:00Z"),
                    ]
                )
            )
        return JsonResponse(_page_batch([]))


def _page(page_id: str, title: str, when: str) -> dict:
    return {
        "id": page_id,
        "title": title,
        "space": {"key": "PAY"},
        "version": {"number": 1, "when": when},
        "metadata": {"labels": {"results": []}},
        "_links": {"webui": f"/display/PAY/{title.replace(' ', '+')}"},
    }


def _page_batch(results: list[dict], *, start: int = 0, next_link: str | None = None) -> dict:
    links = {} if next_link is None else {"next": next_link}
    return {"results": results, "start": start, "size": len(results), "_links": links}


def _attachment(
    attachment_id: str,
    filename: str,
    *,
    media_type: str = "image/png",
) -> dict:
    return {
        "id": attachment_id,
        "title": filename,
        "version": {"number": 1},
        "extensions": {"mediaType": media_type},
        "_links": {"download": f"/download/attachments/123/{attachment_id}"},
    }


async def _fetch_current_page(
    body_html: str,
    attachments: tuple[dict, ...],
):
    gene = ConfluenceGene(
        config={"base_url": "https://wiki.example.test", "spaces": ["PAY"], "pat": "token"},
        source_id="src-confluence",
    )
    gene._api_prefix = "/wiki"
    gene._get = AsyncMock(
        return_value=JsonResponse(
            {
                "id": "123",
                "version": {"number": 7},
                "body": {"storage": {"value": body_html}},
                "ancestors": [],
                "space": {"key": "PAY"},
                "children": {
                    "attachment": {
                        "results": list(attachments),
                        "start": 0,
                        "size": len(attachments),
                        "_links": {},
                    }
                },
            }
        )
    )
    item = ContentItem(
        item_id="confluence-123",
        title="Page",
        source_url="https://wiki.example.test/wiki/pages/123",
        last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
        space_or_project="PAY",
        version="7",
        extra={"page_id": "123"},
    )
    return await gene.fetch(item)


@pytest.mark.asyncio
async def test_semantically_empty_confluence_page_stays_empty_after_normalization():
    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.test/wiki/",
            "spaces": ["PAY"],
            "pat": "confluence-pat",
        },
        source_id="src-confluence",
    )
    gene._api_prefix = "/wiki"
    gene._get = AsyncMock(
        return_value=JsonResponse(
            {
                "id": "123",
                "version": {"number": 7},
                "body": {"storage": {"value": "<p></p>"}},
                "ancestors": [],
                "space": {"key": "PAY"},
                "children": {
                    "attachment": {
                        "results": [],
                        "start": 0,
                        "size": 0,
                        "_links": {},
                    }
                },
            }
        )
    )
    item = ContentItem(
        item_id="confluence-123",
        title="Empty page",
        source_url="https://wiki.example.test/wiki/pages/123",
        last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
        space_or_project="PAY",
        version="7",
        extra={"page_id": "123"},
    )

    raw = await gene.fetch(item)
    normalized = await gene.normalize(raw)

    assert raw.authoritative_empty is True
    assert raw.empty_evidence == "confluence_content_api_successful_semantically_empty_storage_body"
    assert normalized.markdown_body == ""
    assert normalized.source_semantics["semantic_markdown"] == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"body": {}},
        {"body": {"storage": {}}},
        {"body": {"storage": {"value": None}}},
    ],
)
async def test_fetch_rejects_missing_or_invalid_storage_body(payload):
    gene = ConfluenceGene(
        config={"base_url": "https://wiki.example.test", "spaces": ["PAY"], "pat": "token"},
        source_id="src-confluence",
    )
    gene._api_prefix = "/wiki"
    gene._get = AsyncMock(
        return_value=JsonResponse({"id": "123", "version": {"number": 7}, **payload})
    )
    item = ContentItem(
        item_id="confluence-123",
        title="Page",
        source_url="https://wiki.example.test/wiki/pages/123",
        last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
        space_or_project="PAY",
        version="7",
        extra={"page_id": "123"},
    )

    with pytest.raises(RuntimeError, match="body.storage.value"):
        await gene.fetch(item)


@pytest.mark.asyncio
async def test_fetch_does_not_promote_confluence_reported_size_to_exact_byte_contract():
    image = b"\x89PNG\r\n\x1a\nknown-confluence-image"
    gene = ConfluenceGene(
        config={"base_url": "https://wiki.example.test", "spaces": ["PAY"], "pat": "token"},
        source_id="src-confluence",
    )
    gene._api_prefix = "/wiki"
    gene._get = AsyncMock(
        side_effect=[
            JsonResponse(
                {
                    "id": "123",
                    "version": {"number": 7},
                    "body": {
                        "storage": {
                            "value": (
                                "<p>See diagram.</p>"
                                '<ac:image><ri:attachment ri:filename="diagram.png" /></ac:image>'
                            )
                        }
                    },
                    "ancestors": [],
                    "space": {"key": "PAY"},
                    "children": {
                        "attachment": {
                            "results": [
                                {
                                    "id": "att-42",
                                    "title": "diagram.png",
                                    "version": {"number": 3},
                                    "extensions": {
                                        "mediaType": "image/png",
                                        "fileSize": len(image) - 3,
                                    },
                                    "_links": {"download": "/download/attachments/123/diagram.png"},
                                }
                            ],
                            "start": 0,
                            "size": 1,
                            "_links": {},
                        }
                    },
                }
            ),
        ]
    )
    item = ContentItem(
        item_id="confluence-123",
        title="Page",
        source_url="https://wiki.example.test/wiki/pages/123",
        last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
        space_or_project="PAY",
        version="7",
        extra={"page_id": "123"},
    )

    raw = await gene.fetch(item)

    assert len(raw.artifacts) == 1
    artifact = raw.artifacts[0]
    assert artifact.provider_key == "att-42"
    assert artifact.parent_observation_type == "page_body"
    assert artifact.parent_provider_key == "123:body"
    assert artifact.provider_revision == "3"
    assert artifact.media_type == "image/png"
    assert artifact.declared_size_bytes is None
    assert artifact.locator["download_path"] == "/download/attachments/123/diagram.png"
    assert gene._get.await_count == 1


@pytest.mark.asyncio
async def test_open_source_artifact_streams_confluence_attachment():
    image = b"\x89PNG\r\n\x1a\nknown-confluence-image"

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/download/attachments/123/diagram.png"
        return httpx.Response(
            200,
            content=image,
            headers={"content-type": "image/png", "content-length": str(len(image))},
        )

    gene = ConfluenceGene(
        config={"base_url": "https://wiki.example.test", "spaces": ["PAY"], "pat": "token"},
        source_id="src-confluence",
    )
    gene._client = FakeConnectorHttp(handler).client("https://wiki.example.test", product_name="Confluence")
    artifact = RawSourceArtifact(
        provider_key="att-42",
        parent_observation_type="page_body",
        parent_provider_key="123:body",
        provider_revision="3",
        filename="diagram.png",
        media_type="image/png",
        declared_size_bytes=len(image),
        locator={"download_path": "/download/attachments/123/diagram.png"},
    )
    try:
        async with gene.open_source_artifact(artifact) as download:
            assert b"".join([chunk async for chunk in download.chunks]) == image
            assert download.content_length == len(image)
    finally:
        await gene.close()


@pytest.mark.asyncio
async def test_fetch_does_not_count_unsupported_confluence_attachments_against_image_budget():
    image = b"\x89PNG\r\n\x1a\nbounded-confluence-image"
    unsupported = [
        {
            "id": f"document-{index}",
            "title": f"document-{index}.docx",
            "version": {"number": 1},
            "extensions": {
                "mediaType": (
                    "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document"
                ),
                "fileSize": 100,
            },
            "_links": {"download": f"/download/attachments/123/document-{index}.docx"},
        }
        for index in range(20)
    ]
    supported = {
        "id": "image-1",
        "title": "diagram.png",
        "version": {"number": 2},
        "extensions": {
            "mediaType": "image/png",
            "fileSize": len(image),
        },
        "_links": {"download": "/download/attachments/123/diagram.png"},
    }
    gene = ConfluenceGene(
        config={"base_url": "https://wiki.example.test", "spaces": ["PAY"], "pat": "token"},
        source_id="src-confluence",
    )
    gene._api_prefix = "/wiki"
    gene._get = AsyncMock(
        side_effect=[
            JsonResponse(
                {
                    "id": "123",
                    "version": {"number": 7},
                    "body": {
                        "storage": {
                            "value": (
                                "<p>See diagram.</p>"
                                '<ac:image><ri:attachment ri:filename="diagram.png" /></ac:image>'
                            )
                        }
                    },
                    "ancestors": [],
                    "space": {"key": "PAY"},
                    "children": {
                        "attachment": {
                            "results": [*unsupported, supported],
                            "start": 0,
                            "size": 21,
                            "_links": {},
                        }
                    },
                }
            ),
        ]
    )
    item = ContentItem(
        item_id="confluence-123",
        title="Page",
        source_url="https://wiki.example.test/wiki/pages/123",
        last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
        space_or_project="PAY",
        version="7",
        extra={"page_id": "123"},
    )

    raw = await gene.fetch(item)

    assert [artifact.provider_key for artifact in raw.artifacts] == ["image-1"]
    assert gene._get.await_count == 1


@pytest.mark.asyncio
async def test_fetch_materializes_only_attachments_referenced_as_current_body_images():
    raw = await _fetch_current_page(
        '<ac:image><ri:attachment ri:filename="current.png" /></ac:image>',
        (
            _attachment("att-current", "current.png"),
            _attachment("att-residue", "removed-from-body.png"),
        ),
    )

    assert [artifact.provider_key for artifact in raw.artifacts] == ["att-current"]


@pytest.mark.asyncio
async def test_fetch_matches_body_filename_to_entity_encoded_attachment_title():
    raw = await _fetch_current_page(
        (
            '<ac:image><ri:attachment '
            'ri:filename="Screenshot at 1.44.11&#8239;PM.png" /></ac:image>'
        ),
        (
            _attachment(
                "att-current",
                "Screenshot at 1.44.11&#8239;PM.png",
            ),
        ),
    )

    assert [artifact.provider_key for artifact in raw.artifacts] == ["att-current"]


@pytest.mark.asyncio
async def test_fetch_rejects_attachment_titles_that_collide_after_entity_decoding():
    with pytest.raises(SourceArtifactContractError, match="cannot be resolved"):
        await _fetch_current_page(
            (
                '<ac:image><ri:attachment '
                'ri:filename="Screenshot at 1.44.11&#8239;PM.png" /></ac:image>'
            ),
            (
                _attachment(
                    "att-encoded",
                    "Screenshot at 1.44.11&#8239;PM.png",
                ),
                _attachment(
                    "att-unicode",
                    "Screenshot at 1.44.11\u202fPM.png",
                ),
            ),
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "attachment_results",
    [
        (),
        (
            _attachment("att-1", "missing.png"),
            _attachment("att-2", "missing.png"),
        ),
    ],
)
async def test_fetch_rejects_an_unresolved_current_body_image_reference(
    attachment_results,
):
    with pytest.raises(SourceArtifactContractError, match="cannot be resolved"):
        await _fetch_current_page(
            '<ac:image><ri:attachment ri:filename="missing.png" /></ac:image>',
            attachment_results,
        )


@pytest.mark.asyncio
async def test_fetch_does_not_bind_an_explicit_other_page_image_to_local_inventory():
    raw = await _fetch_current_page(
        (
            '<ac:image><ri:attachment ri:filename="local.png" /></ac:image>'
            '<ac:image><ri:attachment ri:filename="same-name.png">'
            '<ri:page ri:space-key="PAY" ri:content-title="Other Page" />'
            "</ri:attachment></ac:image>"
        ),
        (
            _attachment("att-local", "local.png"),
            _attachment("att-wrong-parent", "same-name.png"),
        ),
    )

    assert [artifact.provider_key for artifact in raw.artifacts] == ["att-local"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "container_markup",
    [
        '<ri:content-entity ri:content-id="123" />',
        '<ri:page ri:space-key="PAY" ri:content-title="Page" />',
    ],
)
async def test_fetch_accepts_an_attachment_with_the_current_page_as_explicit_container(
    container_markup,
):
    raw = await _fetch_current_page(
        (
            '<ac:image><ri:attachment ri:filename="current.png">'
            f"{container_markup}"
            "</ri:attachment></ac:image>"
        ),
        (_attachment("att-current", "current.png"),),
    )

    assert [artifact.provider_key for artifact in raw.artifacts] == ["att-current"]


@pytest.mark.asyncio
async def test_fetch_resolves_a_local_default_gallery_to_current_image_inventory():
    raw = await _fetch_current_page(
        '<ac:structured-macro ac:name="gallery" />',
        (
            _attachment("att-image", "gallery.png"),
            _attachment("att-pdf", "linked.pdf", media_type="application/pdf"),
        ),
    )

    assert [artifact.provider_key for artifact in raw.artifacts] == ["att-image"]


@pytest.mark.asyncio
async def test_fetch_applies_gallery_include_and_exclude_membership_parameters():
    raw = await _fetch_current_page(
        (
            '<ac:structured-macro ac:name="gallery">'
            '<ac:parameter ac:name="include">'
            'keep.png, "also,keep.png", drop.png'
            "</ac:parameter>"
            '<ac:parameter ac:name="exclude">drop.png</ac:parameter>'
            "</ac:structured-macro>"
        ),
        tuple(
            _attachment(f"att-{index}", filename)
            for index, filename in enumerate(
                ("keep.png", "also,keep.png", "drop.png", "other.png")
            )
        ),
    )

    assert [artifact.provider_key for artifact in raw.artifacts] == ["att-0", "att-1"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("page", "Other Page"),
        ("includeLabel", "architecture"),
        ("excludeLabel", "obsolete"),
    ],
)
async def test_fetch_rejects_gallery_membership_that_needs_provider_metadata(
    parameter,
    value,
):
    with pytest.raises(SourceArtifactContractError, match="provider metadata"):
        await _fetch_current_page(
            (
                '<ac:structured-macro ac:name="gallery">'
                f'<ac:parameter ac:name="{parameter}">{value}</ac:parameter>'
                "</ac:structured-macro>"
            ),
            (),
        )


@pytest.mark.asyncio
async def test_fetch_materializes_a_current_body_attachment_link_but_not_inventory_residue():
    raw = await _fetch_current_page(
        (
            '<ac:link><ri:attachment ri:filename="current.pdf" />'
            "<ac:plain-text-link-body>Current PDF</ac:plain-text-link-body>"
            "</ac:link>"
        ),
        (
            _attachment("att-current-pdf", "current.pdf", media_type="application/pdf"),
            _attachment("att-residue-pdf", "residue.pdf", media_type="application/pdf"),
        ),
    )

    assert [artifact.provider_key for artifact in raw.artifacts] == ["att-current-pdf"]


@pytest.mark.asyncio
async def test_confluence_attachment_inventory_is_not_rejected_by_descriptor_count():
    gene = ConfluenceGene(
        config={"base_url": "https://wiki.example.test", "spaces": ["PAY"], "pat": "token"},
        source_id="src-confluence",
    )
    gene._api_prefix = "/wiki"
    first_page = {
        "results": [
            {
                "id": f"document-{index}",
                "title": f"document-{index}.docx",
                "version": {"number": 1},
                "extensions": {
                    "mediaType": "application/octet-stream",
                    "fileSize": 1,
                },
                "_links": {"download": f"/download/attachments/123/document-{index}.docx"},
            }
            for index in range(201)
        ],
        "start": 0,
        "size": 201,
        "_links": {},
    }

    assert await gene._fetch_source_artifacts(
        "123",
        first_page=first_page,
        body_html="",
        page_title="Page",
        space_key="PAY",
    ) == ()


@pytest.mark.asyncio
async def test_space_discovery_rejects_missing_results_without_completion_evidence():
    gene = ConfluenceGene(
        config={"base_url": "https://wiki.example.test", "spaces": ["PAY"], "pat": "token"},
        source_id="src-confluence",
    )
    gene._api_prefix = "/wiki"
    gene._get = AsyncMock(return_value=JsonResponse({}))

    with pytest.raises(RuntimeError, match="missing a results list"):
        _ = [item async for item in gene.discover()]
    with pytest.raises(RuntimeError, match="missing a results list"):
        await gene.list_scope()


@pytest.mark.asyncio
async def test_authenticate_uses_bearer_pat_and_normalizes_ui_confluence_wiki_url():
    paths, http = _space_probe_http()

    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.test/wiki/",
            "spaces": ["PAY"],
            "pat": "confluence-pat",
        },
        source_id="src-confluence",
    )

    with http.serving(confluence_gene):
        await gene.authenticate()
    await gene.close()

    assert gene._base_url == "https://wiki.example.test"
    [options] = http.options
    assert options["base_url"] == "https://wiki.example.test"
    assert options["headers"]["Authorization"] == "Bearer confluence-pat"
    assert options["verify"] is True
    assert options["follow_redirects"] is True
    assert paths == ["/wiki/rest/api/space"]


@pytest.mark.asyncio
async def test_authenticate_uses_configured_confluence_ca_bundle(tmp_path):
    _paths, http = _space_probe_http()
    ca_bundle = write_ca_bundle(tmp_path / "corp-ca.pem")

    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.test/wiki/",
            "spaces": ["PAY"],
            "pat": "confluence-pat",
            "tls_ca_bundle": str(ca_bundle),
        },
        source_id="src-confluence",
    )

    with http.serving(confluence_gene):
        await gene.authenticate()
    await gene.close()

    verify = http.options[-1]["verify"]
    assert isinstance(verify, ssl.SSLContext)
    assert verify.verify_mode == ssl.CERT_REQUIRED
    assert trusted_ca_count(verify) == BUNDLE_CA_COUNT


@pytest.mark.asyncio
async def test_authenticate_falls_back_to_root_rest_api_for_plain_confluence_url():
    paths, http = _space_probe_http(404)

    gene = ConfluenceGene(
        config={
            "base_url": "https://confluence.example.test",
            "spaces": ["PAY"],
            "pat": "confluence-pat",
        },
        source_id="src-confluence",
    )

    with http.serving(confluence_gene):
        await gene.authenticate()
    await gene.close()

    assert gene._base_url == "https://confluence.example.test"
    assert gene._api_prefix == ""
    assert paths == [
        "/wiki/rest/api/space",
        "/rest/api/space",
    ]


@pytest.mark.asyncio
async def test_authenticate_rejects_insecure_confluence_url():
    gene = ConfluenceGene(
        config={"base_url": "http://wiki.example.test", "spaces": ["PAY"], "pat": "confluence-pat"},
        source_id="src-confluence",
    )

    with pytest.raises(ValueError, match="HTTPS"):
        await gene.authenticate()


@pytest.mark.asyncio
async def test_authenticate_requires_confluence_pat():
    gene = ConfluenceGene(
        config={"base_url": "https://wiki.example.test", "spaces": ["PAY"]},
        source_id="src-confluence",
    )

    with pytest.raises(ValueError, match="PAT"):
        await gene.authenticate()


@pytest.mark.asyncio
async def test_authenticate_closes_confluence_client_when_pat_validation_fails():
    paths, http = _space_probe_http(401)
    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.test",
            "spaces": ["PAY"],
            "pat": "expired-pat",
        },
        source_id="src-confluence",
    )

    with http.serving(confluence_gene), pytest.raises(httpx.HTTPStatusError):
        await gene.authenticate()

    assert paths == ["/wiki/rest/api/space"]
    assert [client.is_closed for client in http.clients] == [True]
    assert not hasattr(gene, "_client")


@pytest.mark.asyncio
async def test_a_page_tree_root_confluence_does_not_return_is_a_configuration_error():
    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.com",
            "page_tree_root": "123",
            "include_children": True,
        },
        source_id="src-confluence",
    )
    gene._base_url = "https://wiki.example.com"
    gene._api_prefix = "/wiki"
    gene._client = NotFoundClient()

    with pytest.raises(SourceConfigurationError, match="Page Tree Root 123 does not exist"):
        [
            item
            async for item in gene.discover(
                since=datetime(2026, 5, 20, tzinfo=timezone.utc),
            )
        ]


async def _authenticated_root_page_gene(*root_answers: httpx.Response) -> tuple[ConfluenceGene, FakeConnectorHttp, list[str]]:
    """A page-tree Gene authenticated through its real client; reads of root page 123 answer ``root_answers`` in turn."""
    answers = list(root_answers)
    reads: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/wiki/rest/api/space":
            return httpx.Response(200, json={"results": []})
        reads.append(request.url.path)
        return answers.pop(0) if len(answers) > 1 else answers[0]

    http = FakeConnectorHttp(handle)
    gene = ConfluenceGene(
        config={"base_url": "https://wiki.example.com", "page_tree_root": "123", "include_children": False, "pat": "t"},
        source_id="src-confluence",
    )
    with http.serving(confluence_gene):
        await gene.authenticate()
    return gene, http, reads


@pytest.mark.asyncio
async def test_page_tree_discovery_retries_confluence_rate_limit():
    root = {
        "id": "123",
        "title": "Root Page",
        "space": {"key": "PAY"},
        "version": {"number": 7, "when": "2026-05-24T00:00:00Z"},
        "_links": {"webui": "/display/PAY/Root+Page"},
    }
    gene, http, reads = await _authenticated_root_page_gene(
        httpx.Response(429, headers={"Retry-After": "1"}), httpx.Response(200, json=root)
    )
    try:
        items = [item async for item in gene.discover(since=datetime(2026, 5, 20, tzinfo=timezone.utc))]
    finally:
        await gene.close()

    assert [item.title for item in items] == ["Root Page"]
    assert reads == ["/wiki/rest/api/content/123", "/wiki/rest/api/content/123"]
    assert http.waits[0] == 1.0


@pytest.mark.asyncio
async def test_page_tree_discovery_reports_confluence_rate_limit_after_retries():
    gene, _http, reads = await _authenticated_root_page_gene(httpx.Response(429, headers={"Retry-After": "1"}))
    try:
        with pytest.raises(RuntimeError, match="Confluence rate limit persisted"):
            _ = [item async for item in gene.discover(since=datetime(2026, 5, 20, tzinfo=timezone.utc))]
    finally:
        await gene.close()

    assert len(reads) == 4


@pytest.mark.asyncio
async def test_page_tree_discovery_retries_a_confluence_server_error():
    root = _page("123", "Root Page", "2026-05-24T00:00:00Z")
    gene, http, reads = await _authenticated_root_page_gene(httpx.Response(502), httpx.Response(200, json=root))
    try:
        items = [item async for item in gene.discover(since=datetime(2026, 5, 20, tzinfo=timezone.utc))]
    finally:
        await gene.close()

    assert [item.title for item in items] == ["Root Page"]
    assert len(reads) == 2


@pytest.mark.asyncio
async def test_confluence_reports_zero_rest_quota_without_retrying():
    zero_quota = httpx.Response(429, headers={"X-RateLimit-Limit": "0", "X-RateLimit-FillRate": "0"})
    gene, _http, reads = await _authenticated_root_page_gene(zero_quota)
    try:
        with pytest.raises(RuntimeError, match="Confluence REST API quota is zero"):
            _ = [item async for item in gene.discover(since=datetime(2026, 5, 20, tzinfo=timezone.utc))]
    finally:
        await gene.close()

    assert len(reads) == 1


@pytest.mark.asyncio
async def test_page_tree_discovery_traverses_unchanged_parent_to_find_changed_child():
    client = PageTreeClient()
    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.com",
            "page_tree_root": "root",
            "include_children": True,
        },
        source_id="src-confluence",
    )
    gene._base_url = "https://wiki.example.com"
    gene._api_prefix = "/wiki"
    gene._client = client

    items = [
        item
        async for item in gene.discover(
            since=datetime(2026, 5, 25, tzinfo=timezone.utc),
        )
    ]

    assert [item.title for item in items] == ["Changed Child"]
    assert "/wiki/rest/api/content/parent/child/page" in client.calls


@pytest.mark.asyncio
async def test_page_tree_preview_limits_child_page_request_size():
    client = PreviewLimitClient()
    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.com",
            "page_tree_root": "root",
            "include_children": True,
            PREVIEW_DISCOVERY_LIMIT_CONFIG_KEY: 3,
        },
        source_id="preview-confluence",
    )
    gene._base_url = "https://wiki.example.com"
    gene._api_prefix = "/wiki"
    gene._client = client

    items = [item async for item in gene.discover(since=None)]

    assert [item.title for item in items] == ["Root", "Child 1", "Child 2"]
    child_requests = [params for url, params in client.requests if url.endswith("/content/root/child/page")]
    assert child_requests == [{"start": "0", "limit": "2", "expand": "version,metadata.labels,space"}]


@pytest.mark.parametrize(
    "payload",
    [
        {"results": [], "start": 0, "size": 0},
        {"results": [], "start": 0, "size": 1, "_links": {}},
    ],
)
def test_confluence_pagination_envelope_is_required_for_completion(payload):
    with pytest.raises(RuntimeError):
        ConfluenceGene._validated_page_results(
            payload,
            context="test",
        )


@pytest.mark.asyncio
async def test_confluence_listing_that_does_not_start_where_the_walk_asked_is_refused():
    gene = ConfluenceGene(
        config={"base_url": "https://wiki.example.test", "spaces": ["PAY"], "pat": "token"},
        source_id="src-confluence",
    )
    gene._api_prefix = "/wiki"
    gene._get = AsyncMock(return_value=JsonResponse({"results": [], "start": 1, "size": 0, "_links": {}}))

    with pytest.raises(RuntimeError, match="Confluence space PAY response has inconsistent pagination start"):
        _ = [item async for item in gene.discover()]


@pytest.mark.asyncio
async def test_page_tree_rejects_duplicate_page_identity_before_completion():
    class DuplicatePageClient(PageTreeClient):
        async def answer(self, _method: str, url: str, **_kwargs):
            if url.endswith("/content/root"):
                return JsonResponse(_page("root", "Root", "2026-05-20T00:00:00Z"))
            if url.endswith("/content/root/child/page"):
                duplicate = _page("child", "Child", "2026-05-21T00:00:00Z")
                return JsonResponse(_page_batch([duplicate, duplicate]))
            return JsonResponse(_page_batch([]))

    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.com",
            "page_tree_root": "root",
            "include_children": True,
        },
        source_id="src-confluence",
    )
    gene._base_url = "https://wiki.example.com"
    gene._api_prefix = "/wiki"
    gene._client = DuplicatePageClient()

    with pytest.raises(RuntimeError, match="duplicate page id"):
        _ = [item async for item in gene.discover()]
    with pytest.raises(RuntimeError, match="duplicate page id"):
        await gene.list_scope()


@pytest.mark.asyncio
async def test_page_tree_rejects_page_without_stable_version():
    class MissingVersionClient(PageTreeClient):
        async def answer(self, _method: str, url: str, **_kwargs):
            if url.endswith("/content/root"):
                page = _page("root", "Root", "2026-05-20T00:00:00Z")
                page.pop("version")
                return JsonResponse(page)
            return JsonResponse(_page_batch([]))

    gene = ConfluenceGene(
        config={
            "base_url": "https://wiki.example.com",
            "page_tree_root": "root",
            "include_children": True,
        },
        source_id="src-confluence",
    )
    gene._base_url = "https://wiki.example.com"
    gene._api_prefix = "/wiki"
    gene._client = MissingVersionClient()

    with pytest.raises(RuntimeError, match="version metadata"):
        _ = [item async for item in gene.discover()]
    with pytest.raises(RuntimeError, match="version metadata"):
        await gene.list_scope()


def _page_tree_gene(client) -> ConfluenceGene:
    gene = ConfluenceGene(
        config={"base_url": "https://wiki.example.com", "page_tree_root": "root", "include_children": True},
        source_id="src-confluence",
    )
    gene._base_url = "https://wiki.example.com"
    gene._api_prefix = "/wiki"
    gene._client = client
    return gene


@pytest.mark.asyncio
async def test_incremental_page_tree_discovery_lists_every_walked_page_as_a_query():
    gene = _page_tree_gene(PageTreeClient())

    _ = [item async for item in gene.discover(since=datetime(2026, 5, 25, tzinfo=timezone.utc))]
    listing = await gene.list_scope()

    assert listing.kind.value == "query"
    assert listing.doc_ids == {"confluence-root", "confluence-parent", "confluence-target"}


@pytest.mark.asyncio
async def test_preview_limited_discovery_does_not_list_the_scope():
    gene = _page_tree_gene(PreviewLimitClient())
    gene.config[PREVIEW_DISCOVERY_LIMIT_CONFIG_KEY] = 1

    _ = [item async for item in gene.discover()]

    with pytest.raises(RuntimeError, match="preview"):
        await gene.list_scope()


@pytest.mark.asyncio
async def test_confirming_unlisted_pages_keeps_a_page_moved_out_of_the_tree():
    reads: list[str] = []
    # "moved" was moved out of the page tree, "deleted" is in the trash and
    # "restricted" is refused for this one page.
    status_by_page = {"moved": httpx.codes.OK, "deleted": httpx.codes.NOT_FOUND, "restricted": httpx.codes.FORBIDDEN}

    async def handler(request: httpx.Request) -> httpx.Response:
        page_id = request.url.path.rsplit("/", 1)[-1]
        reads.append(page_id)
        status = status_by_page[page_id]
        return httpx.Response(status, json=_page(page_id, "Page", "2026-05-20T00:00:00Z") if status == httpx.codes.OK else {})

    gene = _page_tree_gene(FakeConnectorHttp(handler).client("https://wiki.example.com", product_name="Confluence"))
    items = [
        ContentItem(
            item_id=f"confluence-{page_id}",
            title="Page",
            source_url=f"https://wiki.example.com/pages/{page_id}",
            last_modified=datetime(2026, 5, 20, tzinfo=timezone.utc),
            extra={"page_id": page_id},
        )
        for page_id in status_by_page
    ]
    try:
        answers = await gene.confirm_absent(items)
    finally:
        await gene.close()

    assert answers == {
        "confluence-moved": AbsenceAnswer.EXISTS,
        "confluence-deleted": AbsenceAnswer.ABSENT,
        "confluence-restricted": AbsenceAnswer.HIDDEN,
    }
    assert reads == list(status_by_page)


@pytest.mark.asyncio
async def test_confirming_unlisted_pages_fails_when_confluence_rejects_the_credential():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(httpx.codes.UNAUTHORIZED, json={})

    gene = _page_tree_gene(FakeConnectorHttp(handler).client("https://wiki.example.com", product_name="Confluence"))
    item = ContentItem(
        item_id="confluence-page",
        title="Page",
        source_url="https://wiki.example.com/pages/page",
        last_modified=datetime(2026, 5, 20, tzinfo=timezone.utc),
        extra={"page_id": "page"},
    )
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await gene.confirm_absent([item])
    finally:
        await gene.close()


@pytest.mark.asyncio
async def test_space_discovery_rejects_a_repeated_next_link():
    gene = ConfluenceGene(
        config={"base_url": "https://wiki.example.test", "spaces": ["PAY"], "pat": "token"},
        source_id="src-confluence",
    )
    gene._api_prefix = "/wiki"
    gene._base_url = "https://wiki.example.test"
    gene._get = AsyncMock(
        side_effect=[
            JsonResponse(_page_batch([_page("1", "One", "2026-05-20T00:00:00Z")], next_link="/content?next")),
            JsonResponse(_page_batch([_page("2", "Two", "2026-05-20T00:00:00Z")], start=1, next_link="/content?next")),
        ]
    )

    with pytest.raises(RuntimeError, match="Confluence space PAY response repeats a next link"):
        _ = [item async for item in gene.discover()]


def _attachment_batch(attachments: list[dict], *, start: int, next_link: str | None) -> dict:
    links = {} if next_link is None else {"next": next_link}
    return {"results": attachments, "start": start, "size": len(attachments), "_links": links}


def _paged_attachment_gene(first_page: dict, *attachment_pages: dict) -> ConfluenceGene:
    """A gene whose page read embeds ``first_page`` and whose attachment listing answers ``attachment_pages`` in turn."""
    gene = ConfluenceGene(
        config={"base_url": "https://wiki.example.test", "spaces": ["PAY"], "pat": "token"},
        source_id="src-confluence",
    )
    gene._api_prefix = "/wiki"
    page = {
        "id": "123",
        "version": {"number": 7},
        "body": {
            "storage": {
                "value": "".join(
                    f'<ac:image><ri:attachment ri:filename="{name}" /></ac:image>' for name in ("a.png", "b.png", "c.png")
                )
            }
        },
        "ancestors": [],
        "space": {"key": "PAY"},
        "children": {"attachment": first_page},
    }
    gene._get = AsyncMock(side_effect=[JsonResponse(page), *(JsonResponse(listed) for listed in attachment_pages)])
    return gene


def _attachment_page_item() -> ContentItem:
    return ContentItem(
        item_id="confluence-123",
        title="Page",
        source_url="https://wiki.example.test/wiki/pages/123",
        last_modified=datetime(2026, 7, 16, tzinfo=timezone.utc),
        space_or_project="PAY",
        version="7",
        extra={"page_id": "123"},
    )


def _attachment_requests(gene: ConfluenceGene) -> list[tuple[str, dict | None]]:
    return [
        (call.args[0], call.kwargs.get("params"))
        for call in gene._get.await_args_list
        if call.args[0].endswith("/child/attachment")
    ]


@pytest.mark.asyncio
async def test_fetch_lists_attachments_from_the_embedded_page_across_fetched_pages():
    gene = _paged_attachment_gene(
        _attachment_batch([_attachment("att-a", "a.png")], start=0, next_link="/attachments?start=1"),
        _attachment_batch([_attachment("att-b", "b.png")], start=1, next_link="/attachments?start=2"),
        _attachment_batch([_attachment("att-c", "c.png")], start=2, next_link=None),
    )

    raw = await gene.fetch(_attachment_page_item())

    assert sorted(artifact.provider_key for artifact in raw.artifacts) == ["att-a", "att-b", "att-c"]
    # The embedded first page is not asked for again.
    assert _attachment_requests(gene) == [
        ("/wiki/rest/api/content/123/child/attachment", {"start": 1, "limit": 100, "expand": "version,extensions"}),
        ("/wiki/rest/api/content/123/child/attachment", {"start": 2, "limit": 100, "expand": "version,extensions"}),
    ]


@pytest.mark.asyncio
async def test_fetch_refuses_an_attachment_listing_that_repeats_a_next_link():
    gene = _paged_attachment_gene(
        _attachment_batch([_attachment("att-a", "a.png")], start=0, next_link="/attachments?page=2"),
        _attachment_batch([_attachment("att-b", "b.png")], start=1, next_link="/attachments?page=2"),
    )

    with pytest.raises(SourceArtifactContractError, match="Confluence attachment pagination contains a cursor cycle"):
        await gene.fetch(_attachment_page_item())


@pytest.mark.asyncio
async def test_fetch_refuses_an_attachment_page_that_leaves_a_gap():
    gene = _paged_attachment_gene(
        _attachment_batch([_attachment("att-a", "a.png")], start=0, next_link="/attachments?start=1"),
        _attachment_batch([_attachment("att-c", "c.png")], start=2, next_link=None),
    )

    with pytest.raises(SourceArtifactContractError, match="Confluence attachment pagination is inconsistent"):
        await gene.fetch(_attachment_page_item())


@pytest.mark.asyncio
async def test_fetch_refuses_an_empty_attachment_page_with_a_next_link():
    gene = _paged_attachment_gene(
        _attachment_batch([_attachment("att-a", "a.png")], start=0, next_link="/attachments?start=1"),
        _attachment_batch([], start=1, next_link="/attachments?start=2"),
    )

    with pytest.raises(SourceArtifactContractError, match="Confluence attachment response cannot advance pagination"):
        await gene.fetch(_attachment_page_item())


@pytest.mark.parametrize(
    ("since", "yielded"),
    [
        (datetime(2026, 3, 1, 9, 30, 10, 250_999, tzinfo=timezone.utc), True),  # the page's own millisecond
        (datetime(2026, 3, 1, 9, 30, 10, 250_000, tzinfo=timezone.utc), True),  # exactly the page's time
        (datetime(2026, 3, 1, 9, 30, 10, 251_000, tzinfo=timezone.utc), False),  # the next millisecond
    ],
    ids=["same-millisecond", "same-instant", "next-millisecond"],
)
def test_discovery_compares_since_at_the_millisecond_of_version_when(since, yielded):
    gene = ConfluenceGene(config={"base_url": "https://wiki.example.com"}, source_id="src-confluence")
    gene._base_url = "https://wiki.example.com"
    gene._discovered_page_ids = set()

    item = gene._parse_page(_page("p-1", "Page", "2026-03-01T09:30:10.250Z"), since, set())

    assert (item is not None) is yielded
