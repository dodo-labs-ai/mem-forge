"""Acquire a Confluence page's PDF material: its export HTML and the same-origin images it names.

The page is read like any other Confluence request, through the Gene's
``ConnectorHttpClient`` with its limiter and retries; an error reading it
propagates. Its images are then read within the image budget of
``PDF_IMAGE_BUDGET_SECONDS``, and the server renders the material
(:mod:`memforge.genes.toolkit.pdf`):

- A download starts only when its ``IMAGE_FETCH_TIMEOUT_SECONDS`` fits before
  the image deadline, and a retry only when its wait and that timeout fit.
- At the image deadline every unfinished download is cancelled, which also
  bounds a body that trickles in under its per-read timeout.
- An image that does not arrive is left out of the material, and the page
  renders without it.

Image downloads go past the Atlassian origin limiter: a page may carry up to
``MAX_IMAGE_ASSETS`` images, and pacing them at the origin's interval (2 s for
Confluence) would spend the image budget on 30 images. They keep their own
short retry (``IMAGE_RETRY``).

The start and retry checks read the client's ``clock``, so tests can run them
on a fake one. The cut at the image deadline waits on the event loop's own
clock; both are monotonic in production.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import urljoin, urlparse

from memforge.genes.toolkit.http import ConnectorHttpClient, RetryPolicy
from memforge.genes.toolkit.pdf import PdfImage, PdfMaterial, exportable_soup, preferred_image_source

logger = logging.getLogger(__name__)

MAX_IMAGE_ASSETS = 200
MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 25 * 1024 * 1024
IMAGE_FETCH_TIMEOUT_SECONDS = 10.0
IMAGE_STREAM_CHUNK_BYTES = 64 * 1024
IMAGE_DOWNLOAD_CONCURRENCY = 4
PDF_IMAGE_BUDGET_SECONDS = 60.0
"""How long reading one page's images may take, from when its export HTML arrived."""
# An image is worth less than the page it illustrates: a few quick attempts,
# and a long Retry-After costs an image rather than the image budget.
IMAGE_RETRY = RetryPolicy(attempts=3, backoff_seconds=1.0, max_wait_seconds=10.0)


async def fetch_confluence_pdf_material(
    *,
    client: ConnectorHttpClient,
    base_url: str,
    api_prefix: str,
    page_id: str,
    title: str,
) -> PdfMaterial | None:
    """The page's export HTML and the images it names, or ``None`` when the page has no exportable HTML."""

    resp = await client.get(
        f"{api_prefix}/rest/api/content/{page_id}",
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
    images = await _read_images(
        client=client,
        html=html,
        base_url=base_url,
        api_prefix=api_prefix,
        deadline=client.clock() + PDF_IMAGE_BUDGET_SECONDS,
    )
    return PdfMaterial(title=title, html=html, images=images)


async def _read_images(
    *,
    client: ConnectorHttpClient,
    html: str,
    base_url: str,
    api_prefix: str,
    deadline: float,
) -> dict[str, PdfImage]:
    """Each same-origin HTTPS image the exportable HTML names that arrives by ``deadline``, by the URL it names."""

    image_sources: list[str] = []
    for img in exportable_soup(html).find_all("img"):
        src = preferred_image_source(img)
        if src and src not in image_sources:
            image_sources.append(src)

    if len(image_sources) > MAX_IMAGE_ASSETS:
        logger.warning("Skipping Confluence image after %d asset limit", MAX_IMAGE_ASSETS)
    image_sources = image_sources[:MAX_IMAGE_ASSETS]

    budget = _AssetBudget(MAX_TOTAL_IMAGE_BYTES)
    semaphore = asyncio.Semaphore(IMAGE_DOWNLOAD_CONCURRENCY)

    async def download(src: str) -> tuple[str, PdfImage | None]:
        url = _same_origin_https_asset_url(src, base_url, api_prefix)
        if url is None:
            return src, None
        async with semaphore:
            if client.clock() + IMAGE_FETCH_TIMEOUT_SECONDS > deadline:
                return src, None
            image = await _download_image_asset(client=client, url=url, budget=budget, deadline=deadline)
        return src, image

    downloads = await _finished_by(
        [asyncio.ensure_future(download(src)) for src in image_sources], deadline - client.clock()
    )
    images = {src: image for src, image in downloads if image is not None}
    if len(images) < len(image_sources):
        logger.warning(
            "Reading Confluence page without %d of its %d images", len(image_sources) - len(images), len(image_sources)
        )
    return images


async def _download_image_asset(
    client: ConnectorHttpClient, url: str, budget: "_AssetBudget", deadline: float
) -> PdfImage | None:
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
            limited=False,
            retry=IMAGE_RETRY,
            deadline=deadline,
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
        return PdfImage(body=bytes(content), content_type=content_type or "application/octet-stream")
    except Exception as exc:
        logger.warning("Failed to download Confluence image %s: %s", url, exc)
        return None
    finally:
        if reserved_bytes and not committed:
            await budget.release(reserved_bytes)


async def _finished_by(tasks: list[asyncio.Future], seconds: float) -> list:
    """The results of the tasks done within ``seconds``.

    The rest are cancelled, also when the read itself is cancelled while
    waiting, so no download outlives it.
    """
    if not tasks:
        return []
    try:
        done, _pending = await asyncio.wait(tasks, timeout=max(seconds, 0.0))
    finally:
        unfinished = [task for task in tasks if not task.done()]
        for task in unfinished:
            task.cancel()
        await asyncio.gather(*unfinished, return_exceptions=True)
    return [task.result() for task in tasks if task in done]


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
