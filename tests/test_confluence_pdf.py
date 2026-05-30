from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

from meminception.genes.confluence_pdf import default_chrome_renderer, export_confluence_page_pdf

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
            yield self.content[index:index + size]


class RecordingClient:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        if url == "/wiki/rest/api/content/123":
            return JsonResponse({
                "body": {
                    "export_view": {
                        "value": '<p><img src="/wiki/download/attachments/123/chart.png"></p>'
                    }
                }
            })
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


@pytest.mark.asyncio
async def test_export_confluence_page_pdf_downloads_assets_and_rewrites_html(tmp_path):
    client = RecordingClient()
    captured_html: dict[str, str] = {}

    async def render_pdf(html_path: Path) -> bytes:
        captured_html["body"] = html_path.read_text(encoding="utf-8")
        return b"%PDF-local"

    pdf = await export_confluence_page_pdf(
        client=client,
        base_url="https://wiki.example.test",
        api_prefix="/wiki",
        page_id="123",
        title="Architecture",
        render_pdf=render_pdf,
        work_dir=tmp_path,
    )

    assert pdf == b"%PDF-local"
    assert "file://" in captured_html["body"]
    assert "/wiki/download/attachments/123/chart.png" not in captured_html["body"]
    assert client.calls[0] == (
        "/wiki/rest/api/content/123",
        {"params": {"expand": "body.export_view,body.view"}},
    )
    assert client.calls[1][0] == "https://wiki.example.test/wiki/download/attachments/123/chart.png"


@pytest.mark.asyncio
async def test_export_confluence_page_pdf_rewrites_lazy_image_sources(tmp_path):
    client = RecordingClient()
    captured_html: dict[str, str] = {}

    async def get_with_lazy_image(url: str, **kwargs):
        client.calls.append((url, kwargs))
        if url == "/wiki/rest/api/content/123":
            return JsonResponse({
                "body": {
                    "export_view": {
                        "value": '<p><img data-image-src="/wiki/download/attachments/123/lazy.png" alt="Chart"></p>'
                    }
                }
            })
        if url == "https://wiki.example.test/wiki/download/attachments/123/lazy.png":
            return BytesResponse(b"lazy-png-bytes")
        raise AssertionError(f"unexpected URL: {url}")

    client.get = get_with_lazy_image

    async def render_pdf(html_path: Path) -> bytes:
        captured_html["body"] = html_path.read_text(encoding="utf-8")
        return b"%PDF-local"

    pdf = await export_confluence_page_pdf(
        client=client,
        base_url="https://wiki.example.test",
        api_prefix="/wiki",
        page_id="123",
        title="Architecture",
        render_pdf=render_pdf,
        work_dir=tmp_path,
    )

    assert pdf == b"%PDF-local"
    soup = BeautifulSoup(captured_html["body"], "html.parser")
    assert soup.find("img")["src"].startswith("file://")
    assert "/wiki/download/attachments/123/lazy.png" not in captured_html["body"]


@pytest.mark.asyncio
async def test_export_confluence_page_pdf_prefers_attachment_over_data_placeholder(tmp_path):
    client = RecordingClient()
    captured_html: dict[str, str] = {}

    async def get_with_placeholder_image(url: str, **kwargs):
        client.calls.append((url, kwargs))
        if url == "/wiki/rest/api/content/123":
            return JsonResponse({
                "body": {
                    "export_view": {
                        "value": (
                            '<p><img src="data:image/gif;base64,placeholder" '
                            'data-image-src="/wiki/download/attachments/123/lazy.png" alt="Chart"></p>'
                        )
                    }
                }
            })
        if url == "https://wiki.example.test/wiki/download/attachments/123/lazy.png":
            return BytesResponse(b"lazy-png-bytes")
        raise AssertionError(f"unexpected URL: {url}")

    client.get = get_with_placeholder_image

    async def render_pdf(html_path: Path) -> bytes:
        captured_html["body"] = html_path.read_text(encoding="utf-8")
        return b"%PDF-local"

    pdf = await export_confluence_page_pdf(
        client=client,
        base_url="https://wiki.example.test",
        api_prefix="/wiki",
        page_id="123",
        title="Architecture",
        render_pdf=render_pdf,
        work_dir=tmp_path,
    )

    assert pdf == b"%PDF-local"
    soup = BeautifulSoup(captured_html["body"], "html.parser")
    img = soup.find("img")
    assert img["src"].startswith("file://")
    assert img["data-image-src"].startswith("file://")
    assert "data:image/gif;base64,placeholder" not in captured_html["body"]


@pytest.mark.asyncio
async def test_export_confluence_page_pdf_blanks_unrewritten_image_sources(tmp_path):
    client = RecordingClient()
    captured_html: dict[str, str] = {}

    async def get_with_unsafe_images(url: str, **kwargs):
        client.calls.append((url, kwargs))
        if url == "/wiki/rest/api/content/123":
            return JsonResponse({
                "body": {
                    "export_view": {
                        "value": (
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
                            'onclick="fetch(\'http://internal.service/click\')">text</span>'
                            '<input type="image" src="http://internal.service/input.png">'
                            '<bgsound src="http://internal.service/sound.wav">'
                            '<script>fetch("http://internal.service/script")</script></p>'
                        )
                    }
                }
            })
        raise AssertionError(f"unexpected URL: {url}")

    client.get = get_with_unsafe_images

    async def render_pdf(html_path: Path) -> bytes:
        captured_html["body"] = html_path.read_text(encoding="utf-8")
        return b"%PDF-local"

    pdf = await export_confluence_page_pdf(
        client=client,
        base_url="https://wiki.example.test",
        api_prefix="/wiki",
        page_id="123",
        title="Architecture",
        render_pdf=render_pdf,
        work_dir=tmp_path,
    )

    assert pdf == b"%PDF-local"
    assert len(client.calls) == 1
    assert "file:///etc/passwd" not in captured_html["body"]
    assert "file:///etc/shadow" not in captured_html["body"]
    assert "http://internal.service/secret.png" not in captured_html["body"]
    assert "http://internal.service/object" not in captured_html["body"]
    assert "http://internal.service/script" not in captured_html["body"]
    assert "http://internal.service/style.css" not in captured_html["body"]
    assert "http://internal.service/picture.png" not in captured_html["body"]
    assert "http://internal.service/poster.png" not in captured_html["body"]
    assert "http://internal.service/svg.png" not in captured_html["body"]
    assert "http://internal.service/bg.png" not in captured_html["body"]
    assert "http://internal.service/click" not in captured_html["body"]
    assert "http://internal.service/input.png" not in captured_html["body"]
    assert "http://internal.service/sound.wav" not in captured_html["body"]
    assert "data:image/png;base64,placeholder" not in captured_html["body"]
    soup = BeautifulSoup(captured_html["body"], "html.parser")
    assert [img.get("src") for img in soup.find_all("img")] == ["", "", ""]
    assert not soup.body.find_all(["iframe", "object", "script", "style", "picture", "source", "video", "svg", "bgsound"])
    assert soup.find("span").attrs == {}
    assert soup.find("input").get("src") is None


@pytest.mark.asyncio
async def test_export_confluence_page_pdf_rejects_plain_http_same_host_asset(tmp_path):
    client = RecordingClient()
    captured_html: dict[str, str] = {}

    async def get_with_plain_http_image(url: str, **kwargs):
        client.calls.append((url, kwargs))
        if url == "/wiki/rest/api/content/123":
            return JsonResponse({
                "body": {
                    "export_view": {
                        "value": '<p><img src="http://wiki.example.test/wiki/download/attachments/123/chart.png"></p>'
                    }
                }
            })
        raise AssertionError(f"unexpected URL: {url}")

    client.get = get_with_plain_http_image

    async def render_pdf(html_path: Path) -> bytes:
        captured_html["body"] = html_path.read_text(encoding="utf-8")
        return b"%PDF-local"

    pdf = await export_confluence_page_pdf(
        client=client,
        base_url="https://wiki.example.test",
        api_prefix="/wiki",
        page_id="123",
        title="Architecture",
        render_pdf=render_pdf,
        work_dir=tmp_path,
    )

    assert pdf == b"%PDF-local"
    assert len(client.calls) == 1
    assert "http://wiki.example.test" not in captured_html["body"]
    assert BeautifulSoup(captured_html["body"], "html.parser").find("img")["src"] == ""


@pytest.mark.asyncio
async def test_export_confluence_page_pdf_stops_streaming_oversized_images(tmp_path, monkeypatch):
    monkeypatch.setattr("meminception.genes.confluence_pdf.MAX_IMAGE_BYTES", 5)
    client = RecordingClient()
    captured_html: dict[str, str] = {}
    chunks_yielded = 0

    async def get_page_only(url: str, **kwargs):
        client.calls.append((url, kwargs))
        if url == "/wiki/rest/api/content/123":
            return JsonResponse({
                "body": {
                    "export_view": {
                        "value": '<p><img src="/wiki/download/attachments/123/huge.png"></p>'
                    }
                }
            })
        raise AssertionError(f"unexpected buffered URL: {url}")

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
            client.calls.append((
                "https://wiki.example.test/wiki/download/attachments/123/huge.png",
                {"stream": True},
            ))
            return OversizedImageResponse()

        async def __aexit__(self, _exc_type, _exc, _tb):
            return False

    def stream(method: str, url: str, **_kwargs):
        assert method == "GET"
        assert url == "https://wiki.example.test/wiki/download/attachments/123/huge.png"
        return OversizedImageStream()

    client.get = get_page_only
    client.stream = stream

    async def render_pdf(html_path: Path) -> bytes:
        captured_html["body"] = html_path.read_text(encoding="utf-8")
        return b"%PDF-local"

    pdf = await export_confluence_page_pdf(
        client=client,
        base_url="https://wiki.example.test",
        api_prefix="/wiki",
        page_id="123",
        title="Architecture",
        render_pdf=render_pdf,
        work_dir=tmp_path,
    )

    assert pdf == b"%PDF-local"
    assert chunks_yielded == 2
    assert BeautifulSoup(captured_html["body"], "html.parser").find("img")["src"] == ""
    assert not list((tmp_path / "assets").iterdir())


@pytest.mark.asyncio
async def test_export_confluence_page_pdf_downloads_images_with_bounded_concurrency(tmp_path, monkeypatch):
    monkeypatch.setattr("meminception.genes.confluence_pdf.IMAGE_DOWNLOAD_CONCURRENCY", 2)
    client = RecordingClient()
    captured_html: dict[str, str] = {}
    active_streams = 0
    max_active_streams = 0

    async def get_page_only(url: str, **kwargs):
        client.calls.append((url, kwargs))
        if url == "/wiki/rest/api/content/123":
            return JsonResponse({
                "body": {
                    "export_view": {
                        "value": "".join(
                            f'<img src="/wiki/download/attachments/123/image-{index}.png">'
                            for index in range(4)
                        )
                    }
                }
            })
        raise AssertionError(f"unexpected buffered URL: {url}")

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

    client.get = get_page_only
    client.stream = stream

    async def render_pdf(html_path: Path) -> bytes:
        captured_html["body"] = html_path.read_text(encoding="utf-8")
        return b"%PDF-local"

    pdf = await export_confluence_page_pdf(
        client=client,
        base_url="https://wiki.example.test",
        api_prefix="/wiki",
        page_id="123",
        title="Architecture",
        render_pdf=render_pdf,
        work_dir=tmp_path,
    )

    assert pdf == b"%PDF-local"
    assert max_active_streams == 2
    assert captured_html["body"].count("file://") == 4


@pytest.mark.asyncio
async def test_export_confluence_page_pdf_escapes_page_title(tmp_path):
    client = RecordingClient()
    captured_html: dict[str, str] = {}

    async def render_pdf(html_path: Path) -> bytes:
        captured_html["body"] = html_path.read_text(encoding="utf-8")
        return b"%PDF-local"

    await export_confluence_page_pdf(
        client=client,
        base_url="https://wiki.example.test",
        api_prefix="/wiki",
        page_id="123",
        title="Architecture </h1><script>alert(1)</script>&",
        render_pdf=render_pdf,
        work_dir=tmp_path,
    )

    assert "Architecture &lt;/h1&gt;&lt;script&gt;alert(1)&lt;/script&gt;&amp;" in captured_html["body"]
    assert "</h1><script>alert(1)</script>" not in captured_html["body"]


@pytest.mark.asyncio
async def test_export_confluence_page_pdf_returns_none_when_renderer_missing(tmp_path, monkeypatch):
    monkeypatch.setattr("meminception.genes.confluence_pdf._find_chrome", lambda: None)
    client = RecordingClient()

    pdf = await export_confluence_page_pdf(
        client=client,
        base_url="https://wiki.example.test",
        api_prefix="/wiki",
        page_id="123",
        title="Architecture",
        render_pdf=None,
        work_dir=tmp_path,
    )

    assert pdf is None


@pytest.mark.asyncio
async def test_export_confluence_page_pdf_times_out_slow_render(tmp_path, monkeypatch):
    monkeypatch.setattr("meminception.genes.confluence_pdf.PDF_EXPORT_TIMEOUT_SECONDS", 0.01)
    client = RecordingClient()

    async def render_pdf(_html_path: Path) -> bytes:
        await asyncio.sleep(1)
        return b"%PDF-local"

    pdf = await export_confluence_page_pdf(
        client=client,
        base_url="https://wiki.example.test",
        api_prefix="/wiki",
        page_id="123",
        title="Architecture",
        render_pdf=render_pdf,
        work_dir=tmp_path,
    )

    assert pdf is None


@pytest.mark.asyncio
async def test_default_chrome_renderer_keeps_sandbox_enabled_by_default(monkeypatch, tmp_path):
    captured_args: dict[str, tuple[str, ...]] = {}

    class FakeProcess:
        returncode = 0

        async def wait(self):
            return 0

        def kill(self):
            return None

    async def fake_create_subprocess_exec(*args, **_kwargs):
        captured_args["args"] = args
        for arg in args:
            if str(arg).startswith("--print-to-pdf="):
                Path(str(arg).split("=", 1)[1]).write_bytes(COMPLETE_PDF)
        return FakeProcess()

    monkeypatch.setattr("meminception.genes.confluence_pdf._find_chrome", lambda: "/usr/bin/chrome")
    monkeypatch.setattr("meminception.genes.confluence_pdf.asyncio.create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.delenv("MEMINCEPTION_CHROME_NO_SANDBOX", raising=False)
    html_path = tmp_path / "page.html"
    html_path.write_text("<html></html>", encoding="utf-8")

    renderer = default_chrome_renderer()
    assert renderer is not None
    assert await renderer(html_path) == COMPLETE_PDF
    assert "--no-sandbox" not in captured_args["args"]
    assert any(str(arg).startswith("--user-data-dir=") for arg in captured_args["args"])
    assert "--no-first-run" in captured_args["args"]
    assert "--disable-sync" in captured_args["args"]
    assert "--disable-default-apps" in captured_args["args"]
    assert "--headless=new" in captured_args["args"]
    assert "--allow-file-access-from-files" in captured_args["args"]
    assert "--blink-settings=scriptEnabled=false" not in captured_args["args"]


@pytest.mark.asyncio
async def test_default_chrome_renderer_no_sandbox_requires_explicit_opt_in(monkeypatch, tmp_path):
    captured_args: dict[str, tuple[str, ...]] = {}

    class FakeProcess:
        returncode = 0

        async def wait(self):
            return 0

        def kill(self):
            return None

    async def fake_create_subprocess_exec(*args, **_kwargs):
        captured_args["args"] = args
        for arg in args:
            if str(arg).startswith("--print-to-pdf="):
                Path(str(arg).split("=", 1)[1]).write_bytes(COMPLETE_PDF)
        return FakeProcess()

    monkeypatch.setattr("meminception.genes.confluence_pdf._find_chrome", lambda: "/usr/bin/chrome")
    monkeypatch.setattr("meminception.genes.confluence_pdf.asyncio.create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setenv("MEMINCEPTION_CHROME_NO_SANDBOX", "1")
    html_path = tmp_path / "page.html"
    html_path.write_text("<html></html>", encoding="utf-8")

    renderer = default_chrome_renderer()
    assert renderer is not None
    assert await renderer(html_path) == COMPLETE_PDF
    assert "--no-sandbox" in captured_args["args"]


@pytest.mark.asyncio
async def test_default_chrome_renderer_waits_for_complete_pdf_before_accepting(monkeypatch, tmp_path):
    wait_started = asyncio.Event()
    can_exit = asyncio.Event()
    killed = False

    class FakeProcess:
        returncode: int | None = None

        async def wait(self):
            wait_started.set()
            await can_exit.wait()
            self.returncode = 0
            return 0

        def kill(self):
            nonlocal killed
            killed = True
            self.returncode = -9
            can_exit.set()

    async def fake_create_subprocess_exec(*args, **_kwargs):
        pdf_path = None
        for arg in args:
            if str(arg).startswith("--print-to-pdf="):
                pdf_path = Path(str(arg).split("=", 1)[1])
                break
        assert pdf_path is not None
        pdf_path.write_bytes(b"%PDF-partial")

        async def finish_pdf():
            await wait_started.wait()
            await asyncio.sleep(1.1)
            pdf_path.write_bytes(COMPLETE_PDF)
            can_exit.set()

        asyncio.create_task(finish_pdf())
        return FakeProcess()

    monkeypatch.setattr("meminception.genes.confluence_pdf._find_chrome", lambda: "/usr/bin/chrome")
    monkeypatch.setattr("meminception.genes.confluence_pdf.asyncio.create_subprocess_exec", fake_create_subprocess_exec)
    html_path = tmp_path / "page.html"
    html_path.write_text("<html></html>", encoding="utf-8")

    renderer = default_chrome_renderer()
    assert renderer is not None
    assert await renderer(html_path) == COMPLETE_PDF
    assert killed is False


@pytest.mark.asyncio
async def test_default_chrome_renderer_kills_process_when_cancelled(monkeypatch, tmp_path):
    wait_started = asyncio.Event()
    killed = False

    class FakeProcess:
        returncode: int | None = None

        async def wait(self):
            wait_started.set()
            while self.returncode is None:
                await asyncio.sleep(0.01)
            return self.returncode

        def kill(self):
            nonlocal killed
            killed = True
            self.returncode = -9

    async def fake_create_subprocess_exec(*_args, **_kwargs):
        return FakeProcess()

    monkeypatch.setattr("meminception.genes.confluence_pdf._find_chrome", lambda: "/usr/bin/chrome")
    monkeypatch.setattr("meminception.genes.confluence_pdf.asyncio.create_subprocess_exec", fake_create_subprocess_exec)
    html_path = tmp_path / "page.html"
    html_path.write_text("<html></html>", encoding="utf-8")

    renderer = default_chrome_renderer()
    assert renderer is not None
    task = asyncio.create_task(renderer(html_path))
    await wait_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert killed is True
