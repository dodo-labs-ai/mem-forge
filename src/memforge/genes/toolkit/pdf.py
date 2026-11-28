"""A document's PDF export: the material acquisition reads, and the PDF the server renders from it.

A type that exports PDFs splits the export along the acquisition boundary
(ADR 0058):

- Its Gene's ``fetch_pdf_material`` reads the page's export HTML and the
  images it names into a :class:`PdfMaterial`. It runs where acquisition
  runs, on the server or on the owner's device, and is bounded by the type's
  image budget.
- :func:`render_pdf_material` renders that material to a PDF on the server,
  within ``PDF_RENDER_BUDGET_SECONDS``. Rendering reaches nothing but the
  material: the renderer's URL fetcher serves only the material's own images
  and refuses every other request, including the URLs an SVG image names, so
  a resource the material lacks renders blank, as an image that did not
  arrive does.

Both sides read the export HTML through :func:`exportable_soup`, which drops
active content, and :func:`preferred_image_source`, which names the URL an
image is read from; so the images acquisition reads are the ones the renderer
places.
"""

from __future__ import annotations

import asyncio
import base64
import html as html_lib
import logging
import os
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

__all__ = [
    "MAX_PDF_BYTES",
    "PDF_RENDER_BUDGET_SECONDS",
    "PdfImage",
    "PdfMaterial",
    "MaterialFetcher",
    "RenderHtml",
    "default_pdf_renderer",
    "exportable_soup",
    "preferred_image_source",
    "render_pdf_material",
]

MAX_PDF_BYTES = 100 * 1024 * 1024
PDF_RENDER_BUDGET_SECONDS = 60.0
"""How long rendering one PDF may take; a render that outlasts it produces no PDF."""

MATERIAL_URL_PREFIX = "memforge-material:/image/"
"""The URL the rendered HTML gives the material's n-th image; the renderer's fetcher serves only these."""


@dataclass(frozen=True, slots=True)
class PdfImage:
    """One image of a page's export, as acquisition read it."""

    body: bytes
    content_type: str


@dataclass(frozen=True, slots=True)
class PdfMaterial:
    """What a PDF export renders from: the page title, its export HTML and its images.

    ``images`` holds each image acquisition read, keyed by the URL the HTML
    names for it (:func:`preferred_image_source`). An image the HTML names
    that is missing here renders blank.
    """

    title: str
    html: str
    images: Mapping[str, PdfImage] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "images", MappingProxyType(dict(self.images)))

    def to_payload(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "html": self.html,
            "images": {
                url: {"content_type": image.content_type, "body": base64.b64encode(image.body).decode("ascii")}
                for url, image in sorted(self.images.items())
            },
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PdfMaterial:
        return cls(
            title=str(payload["title"]),
            html=str(payload["html"]),
            images={
                str(url): PdfImage(body=base64.b64decode(str(image["body"])), content_type=str(image["content_type"]))
                for url, image in dict(payload.get("images") or {}).items()
            },
        )


MaterialFetcher = Callable[[str], PdfImage]
"""The one source of a render's resources: the material image at a URL, or ``PermissionError`` for any other URL."""

RenderHtml = Callable[[str, MaterialFetcher], Awaitable[bytes | None]]
"""Render one HTML document to PDF bytes, reading every resource it names through the given fetcher only."""


_BLOCKED_TAGS = [
    "script",
    "iframe",
    "object",
    "embed",
    "link",
    "style",
    "meta",
    "base",
    "picture",
    "source",
    "video",
    "audio",
    "track",
    "bgsound",
    "svg",
]
_BLOCKED_ATTRIBUTES = {
    "style",
    "srcset",
    "poster",
    "background",
    "xlink:href",
    "data",
    "formaction",
    "action",
    "ping",
    "srcdoc",
}


def exportable_soup(html: str) -> BeautifulSoup:
    """The export HTML with active content and every reference but an image's source and a link removed."""

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_BLOCKED_TAGS):
        tag.decompose()
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            attr_name = attr.lower()
            if attr_name.startswith("on") or attr_name in _BLOCKED_ATTRIBUTES:
                del tag.attrs[attr]
            elif attr_name == "src" and tag.name != "img":
                del tag.attrs[attr]
            elif attr_name == "href" and tag.name != "a":
                del tag.attrs[attr]
    return soup


def preferred_image_source(img) -> str | None:
    """The URL an image is read from: its lazy-load source over a ``data:`` placeholder in ``src``."""

    src = img.get("src")
    data_src = img.get("data-image-src")
    if data_src and not str(data_src).startswith("data:"):
        return str(data_src)
    if src:
        return str(src)
    return None


async def render_pdf_material(material: PdfMaterial, *, render_html: RenderHtml | None = None) -> bytes | None:
    """Render one page's PDF from its material alone, or ``None`` when no PDF results.

    Each image the material holds is placed under its own material URL; every
    other image source is blanked. ``render_html`` defaults to WeasyPrint. A
    render that outlasts ``PDF_RENDER_BUDGET_SECONDS``, fails, or yields an
    incomplete or oversized PDF produces none.
    """

    renderer = render_html if render_html is not None else default_pdf_renderer()
    if renderer is None:
        logger.warning("PDF renderer not found; skipping PDF export for %s", material.title)
        return None
    document, served = _material_document(material)

    def fetch(url: str) -> PdfImage:
        image = served.get(url)
        if image is None:
            raise PermissionError(f"the PDF material holds no resource at {url}")
        return image

    try:
        pdf = await asyncio.wait_for(renderer(document, fetch), timeout=PDF_RENDER_BUDGET_SECONDS)
    except TimeoutError:
        logger.warning("PDF rendering timed out for %s", material.title)
        return None
    except Exception as exc:
        logger.warning("PDF rendering failed for %s: %s", material.title, exc)
        return None
    if pdf is None:
        return None
    if len(pdf) > MAX_PDF_BYTES:
        logger.warning("PDF rendering exceeded the size limit for %s", material.title)
        return None
    if not _looks_like_complete_pdf(pdf):
        logger.warning("PDF rendering produced an incomplete PDF for %s", material.title)
        return None
    return pdf


def _material_document(material: PdfMaterial) -> tuple[str, dict[str, PdfImage]]:
    """The HTML document to render, and the images its material URLs serve."""

    soup = exportable_soup(material.html)
    served: dict[str, PdfImage] = {}
    material_urls: dict[str, str] = {}
    for img in soup.find_all("img"):
        source = preferred_image_source(img)
        image = material.images.get(source or "")
        if img.get("data-image-src"):
            del img["data-image-src"]
        if image is None:
            img["src"] = ""
            continue
        url = material_urls.get(source or "")
        if url is None:
            url = f"{MATERIAL_URL_PREFIX}{len(material_urls)}"
            material_urls[source or ""] = url
            served[url] = image
        img["src"] = url
    return _wrap_html(material.title, str(soup)), served


def default_pdf_renderer() -> RenderHtml | None:
    """WeasyPrint, or ``None`` when its system libraries are not installed."""

    _prepare_weasyprint_library_path()
    try:
        from weasyprint import HTML
        from weasyprint.urls import URLFetcherResponse
    except Exception as exc:
        logger.warning("WeasyPrint renderer not available: %s", exc)
        return None

    async def render(document: str, fetch: MaterialFetcher) -> bytes:
        def url_fetcher(url: str) -> URLFetcherResponse:
            image = fetch(url)
            return URLFetcherResponse(url, body=image.body, headers={"Content-Type": image.content_type})

        return await asyncio.to_thread(
            lambda: HTML(string=document, base_url=MATERIAL_URL_PREFIX, url_fetcher=url_fetcher).write_pdf()
        )

    return render


def _prepare_weasyprint_library_path() -> None:
    if sys.platform != "darwin":
        return
    candidates = ["/opt/homebrew/lib", "/usr/local/lib"]
    existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    paths = [path for path in existing.split(":") if path]
    for candidate in candidates:
        if os.path.exists(candidate) and candidate not in paths:
            paths.append(candidate)
    if paths:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(paths)


def _looks_like_complete_pdf(pdf_bytes: bytes) -> bool:
    return pdf_bytes.startswith(b"%PDF-") and b"%%EOF" in pdf_bytes[-2048:]


def _wrap_html(title: str, body: str) -> str:
    escaped_title = html_lib.escape(title, quote=True)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{escaped_title}</title>
<style>
@page {{ size: A4; margin: 16mm 14mm; }}
body {{ font-family: Arial, Helvetica, sans-serif; font-size: 11px; line-height: 1.4; color: #172b4d; }}
h1, h2, h3 {{ page-break-after: avoid; color: #172b4d; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0; page-break-inside: auto; }}
th, td {{ border: 1px solid #dfe1e6; padding: 4px 6px; vertical-align: top; }}
img {{ max-width: 100%; height: auto; page-break-inside: avoid; }}
pre, code {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
</style>
</head>
<body>
<h1>{escaped_title}</h1>
{body}
</body>
</html>"""
