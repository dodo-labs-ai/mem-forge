from __future__ import annotations

import asyncio
import heapq
import json
import time

import httpx
import pytest
from bs4 import BeautifulSoup

from memforge.genes import confluence_pdf
from memforge.genes.atlassian_auth import AtlassianRequestLimiter
from memforge.genes.confluence_pdf import fetch_confluence_pdf_material
from memforge.genes.toolkit.http import ConnectorHttpClient
from memforge.genes.toolkit.pdf import PdfImage, PdfMaterial, render_pdf_material

COMPLETE_PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"


class JsonResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200
        self.headers = {"content-type": "application/json"}
        self.content = json.dumps(payload).encode("utf-8")

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class BytesResponse:
    def __init__(self, content: bytes, content_type: str = "image/png", headers: dict[str, str] | None = None):
        self.content = content
        self.status_code = 200
        self.headers = {"content-type": content_type, **(headers or {})}

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self, chunk_size: int | None = None):
        size = chunk_size or len(self.content)
        for index in range(0, len(self.content), size):
            yield self.content[index : index + size]


class RecordingClient:
    clock = staticmethod(time.monotonic)

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        if url == "/wiki/rest/api/content/123":
            return JsonResponse(
                {"body": {"export_view": {"value": '<p><img src="/wiki/download/attachments/123/chart.png"></p>'}}}
            )
        if url == "https://wiki.example.test/wiki/download/attachments/123/chart.png":
            return BytesResponse(b"png-bytes")
        if url == "https://wiki.example.test/wiki/download/attachments/123/lazy.png":
            return BytesResponse(b"lazy-png-bytes")
        raise AssertionError(f"unexpected URL: {url}")

    def stream(self, method: str, url: str, **kwargs):
        if method != "GET":
            raise AssertionError(f"unexpected method: {method}")
        return ResponseStream(self, url, kwargs)


class ResponseStream:
    def __init__(self, client: RecordingClient, url: str, kwargs: dict):
        self.client = client
        self.url = url
        self.kwargs = kwargs
        self.response = None

    async def __aenter__(self):
        self.response = await self.client.get(self.url, **self.kwargs)
        return self.response

    async def __aexit__(self, _exc_type, _exc, _tb):
        return False


async def _material(client, *, title: str = "Architecture") -> PdfMaterial | None:
    return await fetch_confluence_pdf_material(
        client=client,
        base_url="https://wiki.example.test",
        api_prefix="/wiki",
        page_id="123",
        title=title,
    )


async def _rendered_document(material: PdfMaterial) -> str:
    """The HTML document the server's renderer is given for the material."""

    captured: dict[str, str] = {}

    async def render_html(document: str, _fetch) -> bytes:
        captured["document"] = document
        return COMPLETE_PDF

    assert await render_pdf_material(material, render_html=render_html) == COMPLETE_PDF
    return captured["document"]


def _page_with(html: str):
    async def get(url: str, **_kwargs):
        if url == "/wiki/rest/api/content/123":
            return JsonResponse({"body": {"export_view": {"value": html}}})
        raise AssertionError(f"unexpected URL: {url}")

    return get


@pytest.mark.asyncio
async def test_the_material_holds_the_export_html_and_its_same_origin_images():
    client = RecordingClient()

    material = await _material(client)

    assert material == PdfMaterial(
        title="Architecture",
        html='<p><img src="/wiki/download/attachments/123/chart.png"></p>',
        images={"/wiki/download/attachments/123/chart.png": PdfImage(b"png-bytes", "image/png")},
    )
    assert client.calls[0] == (
        "/wiki/rest/api/content/123",
        {"params": {"expand": "body.export_view,body.view"}},
    )
    assert client.calls[1][0] == "https://wiki.example.test/wiki/download/attachments/123/chart.png"
    document = await _rendered_document(material)
    assert "/wiki/download/attachments/123/chart.png" not in document
    assert BeautifulSoup(document, "html.parser").find("img")["src"] == "memforge-material:/image/0"


@pytest.mark.asyncio
async def test_a_lazy_image_is_read_from_its_lazy_source():
    client = RecordingClient()
    client.get_page = client.get

    async def get(url: str, **kwargs):
        if url == "/wiki/rest/api/content/123":
            client.calls.append((url, kwargs))
            return JsonResponse(
                {
                    "body": {
                        "export_view": {
                            "value": '<p><img data-image-src="/wiki/download/attachments/123/lazy.png" alt="Chart"></p>'
                        }
                    }
                }
            )
        return await client.get_page(url, **kwargs)

    client.get = get

    material = await _material(client)

    assert dict(material.images) == {
        "/wiki/download/attachments/123/lazy.png": PdfImage(b"lazy-png-bytes", "image/png")
    }
    img = BeautifulSoup(await _rendered_document(material), "html.parser").find("img")
    assert img["src"] == "memforge-material:/image/0"
    assert img.get("data-image-src") is None


@pytest.mark.asyncio
async def test_a_lazy_source_wins_over_a_data_placeholder():
    client = RecordingClient()
    client.get_page = client.get

    async def get(url: str, **kwargs):
        if url == "/wiki/rest/api/content/123":
            return JsonResponse(
                {
                    "body": {
                        "export_view": {
                            "value": (
                                '<p><img src="data:image/gif;base64,placeholder" '
                                'data-image-src="/wiki/download/attachments/123/lazy.png" alt="Chart"></p>'
                            )
                        }
                    }
                }
            )
        return await client.get_page(url, **kwargs)

    client.get = get

    material = await _material(client)

    assert list(material.images) == ["/wiki/download/attachments/123/lazy.png"]
    document = await _rendered_document(material)
    assert BeautifulSoup(document, "html.parser").find("img")["src"] == "memforge-material:/image/0"
    assert "data:image/gif;base64,placeholder" not in document


@pytest.mark.asyncio
async def test_active_content_and_foreign_images_are_neither_read_nor_rendered():
    client = RecordingClient()
    client.get = _page_with(
        '<p><img src="file:///etc/passwd" srcset="file:///etc/shadow 1x">'
        '<img src="http://internal.service/secret.png">'
        '<img src="data:image/png;base64,placeholder">'
        '<iframe src="file:///etc/passwd"></iframe>'
        '<object data="http://internal.service/object"></object>'
        '<style>@import url("http://internal.service/style.css");</style>'
        '<picture><source srcset="http://internal.service/picture.png"></picture>'
        '<video poster="http://internal.service/poster.png"></video>'
        '<svg><image href="http://internal.service/svg.png"/></svg>'
        '<span style="background:url(http://internal.service/bg.png)" '
        "onclick=\"fetch('http://internal.service/click')\">text</span>"
        '<input type="image" src="http://internal.service/input.png">'
        '<bgsound src="http://internal.service/sound.wav">'
        '<script>fetch("http://internal.service/script")</script></p>'
    )

    material = await _material(client)

    assert dict(material.images) == {}
    document = await _rendered_document(material)
    for reference in (
        "file:///etc/passwd",
        "file:///etc/shadow",
        "http://internal.service/secret.png",
        "http://internal.service/object",
        "http://internal.service/script",
        "http://internal.service/style.css",
        "http://internal.service/picture.png",
        "http://internal.service/poster.png",
        "http://internal.service/svg.png",
        "http://internal.service/bg.png",
        "http://internal.service/click",
        "http://internal.service/input.png",
        "http://internal.service/sound.wav",
        "data:image/png;base64,placeholder",
    ):
        assert reference not in document
    soup = BeautifulSoup(document, "html.parser")
    assert [img.get("src") for img in soup.find_all("img")] == ["", "", ""]
    assert not soup.body.find_all(
        ["iframe", "object", "script", "style", "picture", "source", "video", "svg", "bgsound"]
    )
    assert soup.find("span").attrs == {}
    assert soup.find("input").get("src") is None


@pytest.mark.asyncio
async def test_a_plain_http_image_on_the_same_host_is_not_read():
    client = RecordingClient()
    client.get = _page_with('<p><img src="http://wiki.example.test/wiki/download/attachments/123/chart.png"></p>')

    material = await _material(client)

    assert dict(material.images) == {}
    document = await _rendered_document(material)
    assert "http://wiki.example.test" not in document
    assert BeautifulSoup(document, "html.parser").find("img")["src"] == ""


@pytest.mark.asyncio
async def test_an_oversized_image_stops_streaming_and_is_left_out(monkeypatch):
    monkeypatch.setattr("memforge.genes.confluence_pdf.MAX_IMAGE_BYTES", 5)
    client = RecordingClient()
    chunks_yielded = 0
    client.get = _page_with('<p><img src="/wiki/download/attachments/123/huge.png"></p>')

    class OversizedImageResponse:
        headers = {"content-type": "image/png"}

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self, chunk_size: int | None = None):
            nonlocal chunks_yielded
            for chunk in (b"1234", b"5678", b"done"):
                chunks_yielded += 1
                yield chunk

    class OversizedImageStream:
        async def __aenter__(self):
            return OversizedImageResponse()

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

    def stream(method: str, url: str, **_kwargs):
        assert method == "GET"
        assert url == "https://wiki.example.test/wiki/download/attachments/123/huge.png"
        return OversizedImageStream()

    client.stream = stream

    material = await _material(client)

    assert chunks_yielded == 2
    assert dict(material.images) == {}
    assert BeautifulSoup(await _rendered_document(material), "html.parser").find("img")["src"] == ""


@pytest.mark.asyncio
async def test_images_are_read_with_bounded_concurrency(monkeypatch):
    monkeypatch.setattr("memforge.genes.confluence_pdf.IMAGE_DOWNLOAD_CONCURRENCY", 2)
    client = RecordingClient()
    active_streams = 0
    max_active_streams = 0
    client.get = _page_with(
        "".join(f'<img src="/wiki/download/attachments/123/image-{index}.png">' for index in range(4))
    )

    class SlowImageResponse:
        headers = {"content-type": "image/png"}

        def __init__(self, url: str):
            self.url = url

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self, chunk_size: int | None = None):
            await asyncio.sleep(0.01)
            yield self.url.encode("utf-8")

    class SlowImageStream:
        def __init__(self, url: str):
            self.url = url

        async def __aenter__(self):
            nonlocal active_streams, max_active_streams
            active_streams += 1
            max_active_streams = max(max_active_streams, active_streams)
            return SlowImageResponse(self.url)

        async def __aexit__(self, _exc_type, _exc, _tb):
            nonlocal active_streams
            active_streams -= 1
            return False

    def stream(method: str, url: str, **_kwargs):
        assert method == "GET"
        return SlowImageStream(url)

    client.stream = stream

    material = await _material(client)

    assert max_active_streams == 2
    assert len(material.images) == 4
    assert (await _rendered_document(material)).count("memforge-material:/image/") == 4


@pytest.mark.asyncio
async def test_the_rendered_page_escapes_its_title():
    material = await _material(RecordingClient(), title="Architecture </h1><script>alert(1)</script>&")

    document = await _rendered_document(material)

    assert "Architecture &lt;/h1&gt;&lt;script&gt;alert(1)&lt;/script&gt;&amp;" in document
    assert "</h1><script>alert(1)</script>" not in document


@pytest.mark.asyncio
async def test_a_page_without_exportable_html_has_no_material():
    client = RecordingClient()
    client.get = _page_with("   ")

    assert await _material(client) is None


@pytest.mark.asyncio
async def test_an_error_reading_the_export_html_propagates():
    client = RecordingClient()

    async def refuse(url: str, **_kwargs):
        raise httpx.ConnectError("refused")

    client.get = refuse

    with pytest.raises(httpx.ConnectError):
        await _material(client)


# ---------------------------------------------------------------------------
# Image downloads on the Gene's ConnectorHttpClient
# ---------------------------------------------------------------------------


class _FakeClock:
    """A monotonic clock that only the client's sleep advances."""

    def __init__(self) -> None:
        self.now = 5000.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


def _image_page(image_count: int) -> dict:
    images = "".join(f'<img src="/wiki/download/attachments/123/image-{index}.png">' for index in range(image_count))
    return {"body": {"export_view": {"value": images}}}


async def _material_over(client: ConnectorHttpClient) -> PdfMaterial | None:
    return await fetch_confluence_pdf_material(
        client=client,
        base_url="https://wiki.example.test",
        api_prefix="/wiki",
        page_id="123",
        title="Architecture",
    )


@pytest.mark.asyncio
async def test_a_page_with_many_images_reads_them_without_pacing_them_at_the_origin():
    image_count = 64
    clock = _FakeClock()
    limiter = AtlassianRequestLimiter(min_interval_seconds=2.0, clock=clock)
    limited_runs = 0
    real_run = limiter.run

    async def counting_run(operation, **options):
        nonlocal limited_runs
        limited_runs += 1
        return await real_run(operation, **options)

    limiter.run = counting_run

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/wiki/rest/api/content/123":
            return httpx.Response(200, json=_image_page(image_count))
        return httpx.Response(200, content=request.url.path.encode(), headers={"content-type": "image/png"})

    started_at = clock.now
    async with ConnectorHttpClient(
        product_name="Confluence",
        base_url="https://wiki.example.test",
        timeout=5.0,
        follow_redirects=True,
        limiter=limiter,
        sleep=clock.sleep,
        transport=httpx.MockTransport(handle),
    ) as client:
        material = await _material_over(client)

    assert len(material.images) == image_count
    # Only the page read went through the 2 s origin limiter; paced the same
    # way, 64 images would need 128 s, more than the whole image budget.
    assert limited_runs == 1
    assert image_count * 2.0 > confluence_pdf.PDF_IMAGE_BUDGET_SECONDS
    assert clock.now - started_at < confluence_pdf.PDF_IMAGE_BUDGET_SECONDS


@pytest.mark.asyncio
async def test_an_image_download_is_retried_with_its_own_short_policy():
    image_answers = [httpx.Response(503), httpx.ConnectError("reset"), httpx.Response(200, content=b"png")]
    image_requests = 0
    waits: list[float] = []

    def handle(request: httpx.Request) -> httpx.Response:
        nonlocal image_requests
        if request.url.path == "/wiki/rest/api/content/123":
            return httpx.Response(200, json=_image_page(1))
        image_requests += 1
        answer = image_answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer

    async def record_wait(seconds: float) -> None:
        waits.append(seconds)

    async with ConnectorHttpClient(
        product_name="Confluence",
        base_url="https://wiki.example.test",
        timeout=5.0,
        follow_redirects=True,
        sleep=record_wait,
        transport=httpx.MockTransport(handle),
    ) as client:
        material = await _material_over(client)

    assert [image.body for image in material.images.values()] == [b"png"]
    assert image_requests == confluence_pdf.IMAGE_RETRY.attempts == 3
    assert waits == [confluence_pdf.IMAGE_RETRY.backoff_seconds, confluence_pdf.IMAGE_RETRY.transport_wait_seconds]


class _VirtualClock:
    """Virtual time for concurrent tasks: ``sleep`` returns once every earlier sleeper has woken, at once in real time.

    When no task has run for a moment, time jumps to the earliest wake-up and
    wakes every sleeper due then, so four downloads that each hang for 10 s
    end together 10 s later, as they would in real time.
    """

    def __init__(self) -> None:
        self.now = 5000.0
        self._sleepers: list[tuple[float, int, asyncio.Future]] = []
        self._order = 0
        self._advance_handle: asyncio.TimerHandle | None = None

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        loop = asyncio.get_running_loop()
        woken = loop.create_future()
        heapq.heappush(self._sleepers, (self.now + max(seconds, 0.0), self._order, woken))
        self._order += 1
        self._arm(loop)
        await woken

    def _arm(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._advance_handle is None:
            self._advance_handle = loop.call_later(0.002, self._advance, loop)

    def _advance(self, loop: asyncio.AbstractEventLoop) -> None:
        self._advance_handle = None
        if not self._sleepers:
            return
        self.now = max(self.now, self._sleepers[0][0])
        while self._sleepers and self._sleepers[0][0] <= self.now:
            _wake, _order, woken = heapq.heappop(self._sleepers)
            if not woken.done():
                woken.set_result(None)
        if self._sleepers:
            self._arm(loop)


async def _read_on_virtual_clock(answer_image) -> tuple[PdfMaterial | None, float, int]:
    """Read a page of 64 images whose downloads ``answer_image(request, clock)`` answers.

    Returns the material, the time the images took from when the export HTML
    arrived, and the number of image requests.
    """
    clock = _VirtualClock()
    image_requests = 0
    html_read_at: list[float] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal image_requests
        if request.url.path == "/wiki/rest/api/content/123":
            html_read_at.append(clock())
            return httpx.Response(200, json=_image_page(64))
        image_requests += 1
        return await answer_image(request, clock)

    async with ConnectorHttpClient(
        product_name="Confluence",
        base_url="https://wiki.example.test",
        timeout=5.0,
        follow_redirects=True,
        sleep=clock.sleep,
        clock=clock,
        transport=httpx.MockTransport(handle),
    ) as client:
        material = await _material_over(client)
    return material, clock() - html_read_at[0], image_requests


@pytest.mark.asyncio
async def test_images_that_keep_asking_to_wait_stay_within_the_image_budget():
    async def rate_limited(_request: httpx.Request, _clock: _VirtualClock) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "10"})

    material, took, image_requests = await _read_on_virtual_clock(rate_limited)

    assert dict(material.images) == {}
    assert took <= confluence_pdf.PDF_IMAGE_BUDGET_SECONDS
    # Some images were tried and retried; the rest were never started.
    assert 0 < image_requests < 64 * confluence_pdf.IMAGE_RETRY.attempts


@pytest.mark.asyncio
async def test_images_that_hang_until_their_timeout_stay_within_the_image_budget():
    async def hanging(request: httpx.Request, clock: _VirtualClock) -> httpx.Response:
        await clock.sleep(confluence_pdf.IMAGE_FETCH_TIMEOUT_SECONDS)
        raise httpx.ReadTimeout("timed out", request=request)

    material, took, image_requests = await _read_on_virtual_clock(hanging)

    assert dict(material.images) == {}
    assert took <= confluence_pdf.PDF_IMAGE_BUDGET_SECONDS
    assert 0 < image_requests < 64


@pytest.mark.asyncio
async def test_a_download_still_running_at_the_image_deadline_is_cancelled(monkeypatch):
    monkeypatch.setattr("memforge.genes.confluence_pdf.PDF_IMAGE_BUDGET_SECONDS", 0.1)
    monkeypatch.setattr("memforge.genes.confluence_pdf.IMAGE_FETCH_TIMEOUT_SECONDS", 0.01)

    class TrickleForever(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                yield b"x"
                await asyncio.sleep(0.005)

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/wiki/rest/api/content/123":
            return httpx.Response(200, json=_image_page(1))
        return httpx.Response(200, stream=TrickleForever(), headers={"content-type": "image/png"})

    started_at = time.monotonic()
    async with ConnectorHttpClient(
        product_name="Confluence",
        base_url="https://wiki.example.test",
        timeout=5.0,
        follow_redirects=True,
        transport=httpx.MockTransport(handle),
    ) as client:
        material = await _material_over(client)

    assert dict(material.images) == {}
    # The cut came at the image deadline, 0.1 s in.
    assert time.monotonic() - started_at < 0.3


@pytest.mark.asyncio
async def test_cancelling_the_read_cancels_its_image_downloads():
    requested: list[str] = []
    first_downloads = asyncio.Event()

    async def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/wiki/rest/api/content/123":
            return httpx.Response(200, json=_image_page(8))
        requested.append(request.url.path)
        first_downloads.set()
        await asyncio.sleep(0.3)
        return httpx.Response(200, content=b"png", headers={"content-type": "image/png"})

    async with ConnectorHttpClient(
        product_name="Confluence",
        base_url="https://wiki.example.test",
        timeout=5.0,
        follow_redirects=True,
        transport=httpx.MockTransport(handle),
    ) as client:
        read = asyncio.ensure_future(_material_over(client))
        await first_downloads.wait()
        read.cancel()
        with pytest.raises(asyncio.CancelledError):
            await read
        requested_at_cancel = len(requested)
        await asyncio.sleep(0.5)

    assert requested_at_cancel < 8
    assert len(requested) == requested_at_cancel
