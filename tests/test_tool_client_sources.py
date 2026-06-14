"""Contract tests for the source management methods on ToolClient.

These verify the HTTP method, path, and body the client sends, since those are
the parts most likely to drift from the admin API. The transport itself
(`_http_json`) is exercised by the agent-tool read paths elsewhere.
"""

from __future__ import annotations

import json
from typing import Any

import memforge.tool_client as tool_client
from memforge.tool_client import ToolClient


class _RecordingClient(ToolClient):
    """ToolClient that captures the _http_json call instead of making a request."""

    def __init__(self, response: dict[str, Any]) -> None:
        super().__init__(api_url="https://memforge.example.test", api_token="tok")
        self._response = response
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []

    def _http_json(self, method, path, body):  # type: ignore[override]
        self.calls.append((method, path, body))
        return self._response


def test_create_source_posts_type_name_and_config():
    client = _RecordingClient({"id": "src-abcd1234", "name": "Notes", "type": "local_markdown"})

    result = client.create_source(
        source_type="local_markdown",
        name="Notes",
        config={"vault_id": "engineering", "display_label": "Engineering notes"},
    )

    assert result["id"] == "src-abcd1234"
    assert client.calls == [
        (
            "POST",
            "/api/sources",
            {
                "type": "local_markdown",
                "name": "Notes",
                "config": {"vault_id": "engineering", "display_label": "Engineering notes"},
            },
        )
    ]


def test_list_sources_gets_sources_collection():
    client = _RecordingClient({"data": [{"id": "src-1", "type": "local_markdown"}]})

    result = client.list_sources()

    assert result["data"][0]["id"] == "src-1"
    assert client.calls == [("GET", "/api/sources", None)]


def test_tool_client_forwards_search_to_hosted_workspace(monkeypatch):
    captured = {}

    class FakeResponse:
        headers = {"content-type": "application/json"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b'{"results":[]}'

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["body"] = request.data
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")
    monkeypatch.setattr(tool_client, "build_opener", lambda *_handlers: FakeOpener())
    client = ToolClient(api_url="https://memforge.example.test", api_token="tok")

    result = client.search(query="artifact cache")

    assert result == {"results": []}
    assert captured["url"] == "https://memforge.example.test/api/workspaces/mount_tai/api/memories/search"
    assert captured["authorization"] == "Bearer tok"
    assert json.loads(captured["body"].decode())["query"] == "artifact cache"


def test_tool_client_fetches_resource_through_hosted_workspace(monkeypatch):
    captured = {}

    class FakeResponse:
        headers = {"content-type": "text/markdown"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size=-1):
            return b"# Source"

    class FakeOpener:
        def open(self, request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            return FakeResponse()

    monkeypatch.setenv("MEMFORGE_WORKSPACE_ID", "mount_tai")
    monkeypatch.setattr(tool_client, "build_opener", lambda *_handlers: FakeOpener())
    client = ToolClient(api_url="https://memforge.example.test", api_token="tok")

    result = client.get_resource(url="/api/documents/doc-1/content")

    assert result["text"] == "# Source"
    assert captured["url"] == "https://memforge.example.test/api/workspaces/mount_tai/api/documents/doc-1/content"
    assert captured["authorization"] == "Bearer tok"
