"""The server renders a PDF from its material alone (ADR 0058, design §10.5)."""

from __future__ import annotations

import asyncio
import struct
import zlib

import pytest
from bs4 import BeautifulSoup

from memforge.genes import confluence_pdf
from memforge.genes.toolkit import pdf as pdf_toolkit
from memforge.genes.toolkit.pdf import PdfImage, PdfMaterial, default_pdf_renderer, render_pdf_material

COMPLETE_PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


def _png() -> bytes:
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(b"\x00\xff\x00\x00"))
        + chunk(b"IEND", b"")
    )


SVG_THAT_NAMES_OTHER_URLS = (
    b'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="10" height="10">'
    b'<image href="https://wiki.example.test/wiki/download/attachments/123/secret.png" width="10" height="10"/>'
    b'<image xlink:href="file:///etc/passwd" width="10" height="10"/>'
    b"</svg>"
)


def _material() -> PdfMaterial:
    return PdfMaterial(
        title="Architecture",
        html=(
            '<p><img src="/wiki/download/attachments/123/chart.png"></p>'
            '<p><img src="/wiki/download/attachments/123/diagram.svg"></p>'
            '<p><img src="/wiki/download/attachments/123/never-arrived.png"></p>'
        ),
        images={
            "/wiki/download/attachments/123/chart.png": PdfImage(_png(), "image/png"),
            "/wiki/download/attachments/123/diagram.svg": PdfImage(SVG_THAT_NAMES_OTHER_URLS, "image/svg+xml"),
        },
    )


@pytest.mark.asyncio
async def test_rendering_serves_only_the_material_and_blanks_what_it_lacks():
    requested: list[tuple[str, bool]] = []
    captured: dict[str, str] = {}

    async def render_html(document: str, fetch) -> bytes:
        captured["document"] = document
        for url in (
            "memforge-material:/image/0",
            "memforge-material:/image/1",
            "https://wiki.example.test/wiki/download/attachments/123/secret.png",
            "file:///etc/passwd",
        ):
            try:
                image = fetch(url)
            except PermissionError:
                requested.append((url, False))
            else:
                requested.append((url, True))
                assert image.body == (_png() if url.endswith("/0") else SVG_THAT_NAMES_OTHER_URLS)
        return COMPLETE_PDF

    assert await render_pdf_material(_material(), render_html=render_html) == COMPLETE_PDF

    assert requested == [
        ("memforge-material:/image/0", True),
        ("memforge-material:/image/1", True),
        ("https://wiki.example.test/wiki/download/attachments/123/secret.png", False),
        ("file:///etc/passwd", False),
    ]
    sources = [img["src"] for img in BeautifulSoup(captured["document"], "html.parser").find_all("img")]
    assert sources == ["memforge-material:/image/0", "memforge-material:/image/1", ""]


@pytest.mark.asyncio
async def test_weasyprint_renders_the_material_without_reaching_anything_else(monkeypatch):
    renderer = default_pdf_renderer()
    if renderer is None:
        pytest.skip("WeasyPrint renderer is not available in this environment")

    import weasyprint.urls

    def no_network(url, *args, **kwargs):
        raise AssertionError(f"rendering reached {url}")

    monkeypatch.setattr(weasyprint.urls, "default_url_fetcher", no_network)
    requested: list[tuple[str, bool]] = []

    async def recording(document: str, fetch) -> bytes | None:
        def recorded_fetch(url: str):
            try:
                response = fetch(url)
            except Exception:
                requested.append((url, False))
                raise
            requested.append((url, True))
            return response

        return await renderer(document, recorded_fetch)

    pdf = await render_pdf_material(_material(), render_html=recording)

    assert pdf is not None and pdf.startswith(b"%PDF-")
    assert ("memforge-material:/image/0", True) in requested
    assert ("memforge-material:/image/1", True) in requested
    refused = [url for url, served in requested if not served]
    assert all(not url.startswith(pdf_toolkit.MATERIAL_URL_PREFIX) for url in refused)
    assert all(url.startswith(pdf_toolkit.MATERIAL_URL_PREFIX) for url, served in requested if served)


@pytest.mark.asyncio
async def test_a_render_that_outlasts_the_render_budget_produces_no_pdf(monkeypatch):
    monkeypatch.setattr("memforge.genes.toolkit.pdf.PDF_RENDER_BUDGET_SECONDS", 0.01)

    async def slow(_document: str, _fetch) -> bytes:
        await asyncio.sleep(1)
        return COMPLETE_PDF

    assert await render_pdf_material(_material(), render_html=slow) is None


@pytest.mark.asyncio
async def test_the_render_budget_is_not_the_image_budget(monkeypatch):
    monkeypatch.setattr("memforge.genes.confluence_pdf.PDF_IMAGE_BUDGET_SECONDS", 0.0)
    assert confluence_pdf.PDF_IMAGE_BUDGET_SECONDS == 0.0

    async def takes_a_moment(_document: str, _fetch) -> bytes:
        await asyncio.sleep(0.05)
        return COMPLETE_PDF

    assert await render_pdf_material(_material(), render_html=takes_a_moment) == COMPLETE_PDF


@pytest.mark.asyncio
@pytest.mark.parametrize("rendered", [b"%PDF-1.4 cut short", b"not a pdf"])
async def test_an_incomplete_render_produces_no_pdf(rendered):
    async def render_html(_document: str, _fetch) -> bytes:
        return rendered

    assert await render_pdf_material(_material(), render_html=render_html) is None


@pytest.mark.asyncio
async def test_a_failing_render_produces_no_pdf():
    async def render_html(_document: str, _fetch) -> bytes:
        raise RuntimeError("renderer crashed")

    assert await render_pdf_material(_material(), render_html=render_html) is None


@pytest.mark.asyncio
async def test_without_a_renderer_no_pdf_is_produced(monkeypatch):
    monkeypatch.setattr("memforge.genes.toolkit.pdf.default_pdf_renderer", lambda: None)

    assert await render_pdf_material(_material()) is None


def test_the_material_travels_as_its_payload():
    material = _material()

    assert PdfMaterial.from_payload(material.to_payload()) == material
