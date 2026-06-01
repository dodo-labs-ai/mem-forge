"""Local Confluence HTML-to-PDF export for PAT-authenticated pages."""

from __future__ import annotations

import asyncio
import hashlib
import html as html_lib
import logging
import os
import sys
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from memforge.genes.atlassian_auth import get_with_rate_limit_retry

logger = logging.getLogger(__name__)

RenderPdf = Callable[[Path], Awaitable[bytes | None]]
MAX_IMAGE_ASSETS = 200
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 25 * 1024 * 1024
MAX_PDF_BYTES = 100 * 1024 * 1024
IMAGE_FETCH_TIMEOUT_SECONDS = 10.0
IMAGE_STREAM_CHUNK_BYTES = 64 * 1024
IMAGE_DOWNLOAD_CONCURRENCY = 4
PDF_EXPORT_TIMEOUT_SECONDS = 120.0


async def export_confluence_page_pdf(
    *,
    client,
    base_url: str,
    api_prefix: str,
    page_id: str,
    title: str,
    render_pdf: RenderPdf | None = None,
    work_dir: Path | None = None,
    limiter: Any | None = None,
) -> bytes | None:
    """Render a Confluence page's REST export HTML to a local PDF."""
    renderer = render_pdf if render_pdf is not None else default_pdf_renderer()
    if renderer is None:
        logger.warning("PDF renderer not found; skipping PDF export for %s", title)
        return None

    request_url = f"{api_prefix}/rest/api/content/{page_id}"
    if limiter is not None:
        resp = await get_with_rate_limit_retry(
            client,
            request_url,
            product_name="Confluence",
            params={"expand": "body.export_view,body.view"},
            limiter=limiter,
        )
    else:
        resp = await client.get(
            request_url,
            params={"expand": "body.export_view,body.view"},
        )
    resp.raise_for_status()
    data = resp.json()
    html = (
        data.get("body", {}).get("export_view", {}).get("value")
        or data.get("body", {}).get("view", {}).get("value")
        or ""
    )
    if not html.strip():
        logger.warning("Confluence page %s has no exportable HTML", page_id)
        return None

    if work_dir is None:
        with tempfile.TemporaryDirectory(prefix="memforge-confluence-pdf-") as tmp:
            return await _render_with_timeout(
                client=client,
                html=html,
                title=title,
                base_url=base_url,
                api_prefix=api_prefix,
                renderer=renderer,
                work_dir=Path(tmp),
            )

    work_dir.mkdir(parents=True, exist_ok=True)
    return await _render_with_timeout(
        client=client,
        html=html,
        title=title,
        base_url=base_url,
        api_prefix=api_prefix,
        renderer=renderer,
        work_dir=work_dir,
    )


async def _render_with_timeout(
    *,
    client,
    html: str,
    title: str,
    base_url: str,
    api_prefix: str,
    renderer: RenderPdf,
    work_dir: Path,
) -> bytes | None:
    try:
        return await asyncio.wait_for(
            _render_with_assets(
                client=client,
                html=html,
                title=title,
                base_url=base_url,
                api_prefix=api_prefix,
                renderer=renderer,
                work_dir=work_dir,
            ),
            timeout=PDF_EXPORT_TIMEOUT_SECONDS,
        )
    except TimeoutError:
        logger.warning("Confluence PDF export timed out for %s", title)
        return None


def default_pdf_renderer() -> RenderPdf | None:
    return default_weasyprint_renderer()


def default_weasyprint_renderer() -> RenderPdf | None:
    _prepare_weasyprint_library_path()
    try:
        from weasyprint import HTML
    except Exception as exc:
        logger.warning("WeasyPrint renderer not available: %s", exc)
        return None

    async def render(html_path: Path) -> bytes | None:
        def render_sync() -> bytes:
            return HTML(filename=str(html_path), base_url=html_path.parent.as_uri()).write_pdf()

        try:
            pdf_bytes = await asyncio.to_thread(render_sync)
        except Exception as exc:
            logger.warning("WeasyPrint PDF rendering failed for %s: %s", html_path, exc)
            return None
        if len(pdf_bytes) > MAX_PDF_BYTES:
            logger.warning("WeasyPrint PDF rendering exceeded size limit for %s", html_path)
            return None
        if _looks_like_complete_pdf(pdf_bytes):
            return pdf_bytes
        logger.warning("WeasyPrint PDF rendering produced an incomplete PDF for %s", html_path)
        return None

    return render


def _prepare_weasyprint_library_path() -> None:
    if sys.platform != "darwin":
        return
    candidates = ["/opt/homebrew/lib", "/usr/local/lib"]
    existing = os.environ.get("DYLD_FALLBACK_LIBRARY_PATH", "")
    paths = [path for path in existing.split(":") if path]
    for candidate in candidates:
        if Path(candidate).exists() and candidate not in paths:
            paths.append(candidate)
    if paths:
        os.environ["DYLD_FALLBACK_LIBRARY_PATH"] = ":".join(paths)


def _looks_like_complete_pdf(pdf_bytes: bytes) -> bool:
    return pdf_bytes.startswith(b"%PDF-") and b"%%EOF" in pdf_bytes[-2048:]


async def _render_with_assets(
    *,
    client,
    html: str,
    title: str,
    base_url: str,
    api_prefix: str,
    renderer: RenderPdf,
    work_dir: Path,
) -> bytes | None:
    assets_dir = work_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    html = await _rewrite_image_sources(
        client=client,
        html=html,
        base_url=base_url,
        api_prefix=api_prefix,
        assets_dir=assets_dir,
    )
    html_path = work_dir / "page.html"
    html_path.write_text(_wrap_html(title, html), encoding="utf-8")
    return await renderer(html_path)


async def _rewrite_image_sources(
    *,
    client,
    html: str,
    base_url: str,
    api_prefix: str,
    assets_dir: Path,
) -> str:
    soup = BeautifulSoup(html, "html.parser")
    _remove_active_content(soup)
    image_sources: list[str] = []
    for img in soup.find_all("img"):
        src = _preferred_image_source(img)
        if src and src not in image_sources:
            image_sources.append(src)

    if len(image_sources) > MAX_IMAGE_ASSETS:
        logger.warning("Skipping Confluence image after %d asset limit", MAX_IMAGE_ASSETS)
    image_sources = image_sources[:MAX_IMAGE_ASSETS]

    budget = _AssetBudget(MAX_TOTAL_IMAGE_BYTES)
    semaphore = asyncio.Semaphore(IMAGE_DOWNLOAD_CONCURRENCY)

    async def download(src: str) -> tuple[str, str, tuple[bytes, str] | None]:
        url = _same_origin_https_asset_url(src, base_url, api_prefix)
        if url is None:
            return src, "", None
        async with semaphore:
            image = await _download_image_asset(client=client, url=url, budget=budget)
        return src, url, image

    replacements: dict[str, str] = {}
    downloads = await asyncio.gather(*(download(src) for src in image_sources))
    for src, url, image in downloads:
        if image is None:
            continue
        content, content_type = image
        asset_path = assets_dir / _asset_filename(url, content_type)
        asset_path.write_bytes(content)
        replacements[src] = asset_path.as_uri()

    for img in soup.find_all("img"):
        src = _preferred_image_source(img)
        replacement = replacements.get(src or "")
        if not replacement:
            img["src"] = ""
            if img.get("srcset"):
                del img["srcset"]
            if img.get("data-image-src"):
                del img["data-image-src"]
            continue
        img["src"] = replacement
        if img.get("srcset"):
            del img["srcset"]
        if img.get("data-image-src"):
            img["data-image-src"] = replacement

    return str(soup)


async def _download_image_asset(client, url: str, budget: "_AssetBudget") -> tuple[bytes, str] | None:
    reserved_bytes = 0
    committed = False
    content = bytearray()
    content_type = ""
    try:
        async with client.stream(
            "GET",
            url,
            headers={"Accept": "*/*"},
            timeout=IMAGE_FETCH_TIMEOUT_SECONDS,
        ) as resp:
            resp.raise_for_status()
            declared_size = _content_length(resp.headers.get("content-length"))
            if declared_size is not None and declared_size > MAX_IMAGE_BYTES:
                logger.warning("Skipping Confluence image %s because it exceeds the PDF asset budget", url)
                return None

            content = bytearray()
            async for chunk in resp.aiter_bytes(chunk_size=IMAGE_STREAM_CHUNK_BYTES):
                if len(content) + len(chunk) > MAX_IMAGE_BYTES:
                    logger.warning("Skipping Confluence image %s because it exceeds the PDF asset budget", url)
                    return None
                if not await budget.reserve(len(chunk)):
                    logger.warning("Skipping Confluence image %s because the PDF asset budget is exhausted", url)
                    return None
                reserved_bytes += len(chunk)
                content.extend(chunk)
            content_type = resp.headers.get("content-type", "")
        committed = True
        return bytes(content), content_type
    except Exception as exc:
        logger.warning("Failed to download Confluence image %s: %s", url, exc)
        return None
    finally:
        if reserved_bytes and not committed:
            await budget.release(reserved_bytes)


class _AssetBudget:
    def __init__(self, total_bytes: int):
        self._remaining = total_bytes
        self._lock = asyncio.Lock()

    async def reserve(self, size: int) -> bool:
        async with self._lock:
            if size > self._remaining:
                return False
            self._remaining -= size
            return True

    async def release(self, size: int) -> None:
        async with self._lock:
            self._remaining += size


def _content_length(value: str | None) -> int | None:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _remove_active_content(soup: BeautifulSoup) -> None:
    blocked_tags = [
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
    blocked_attrs = {
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
    for tag in soup.find_all(blocked_tags):
        tag.decompose()
    for tag in soup.find_all(True):
        for attr in list(tag.attrs):
            attr_name = attr.lower()
            if attr_name.startswith("on") or attr_name in blocked_attrs:
                del tag.attrs[attr]
            elif attr_name == "src" and tag.name != "img":
                del tag.attrs[attr]
            elif attr_name == "href" and tag.name != "a":
                del tag.attrs[attr]


def _same_origin_https_asset_url(src: str, base_url: str, api_prefix: str) -> str | None:
    parsed_src = urlparse(src)
    if parsed_src.scheme and parsed_src.scheme.lower() not in {"http", "https"}:
        return None

    url = urljoin(f"{base_url}{api_prefix}/", src)
    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)
    url_scheme = parsed_url.scheme.lower()
    base_scheme = parsed_base.scheme.lower()
    if url_scheme != "https" or url_scheme != base_scheme:
        return None
    if parsed_url.netloc != parsed_base.netloc:
        return None
    return url


def _preferred_image_source(img) -> str | None:
    src = img.get("src")
    data_src = img.get("data-image-src")
    if data_src and not str(data_src).startswith("data:"):
        return str(data_src)
    if src:
        return str(src)
    return None


def _asset_filename(url: str, content_type: str) -> str:
    path = urlparse(url).path.lower()
    extension = ".bin"
    if "png" in content_type or path.endswith(".png"):
        extension = ".png"
    elif "jpeg" in content_type or "jpg" in content_type or path.endswith((".jpg", ".jpeg")):
        extension = ".jpg"
    elif "gif" in content_type or path.endswith(".gif"):
        extension = ".gif"
    elif "svg" in content_type or path.endswith(".svg"):
        extension = ".svg"
    return f"{hashlib.sha256(url.encode('utf-8')).hexdigest()[:16]}{extension}"


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
