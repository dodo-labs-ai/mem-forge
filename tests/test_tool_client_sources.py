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


def test_get_source_schedule_uses_source_schedule_endpoint():
    client = _RecordingClient({"enabled": True, "interval_minutes": 60})

    result = client.get_source_schedule("src-1")

    assert result["enabled"] is True
    assert client.calls == [("GET", "/api/sources/src-1/schedule", None)]


def test_update_source_schedule_puts_source_schedule_payload():
    client = _RecordingClient({"ok": True})

    result = client.update_source_schedule(
        source_id="src-1",
        enabled=True,
        interval_minutes=60,
    )

    assert result["ok"] is True
    assert client.calls == [
        (
            "PUT",
            "/api/sources/src-1/schedule",
            {"enabled": True, "interval_minutes": 60},
        )
    ]


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


def test_tool_client_forwards_structured_memory_search_facets():
    client = _RecordingClient({"results": []})

    client.search(
        query="scheduler fixes last week",
        source_filter={
            "source_ids": ["src-codex-session"],
            "clients": ["codex"],
            "repo_identifiers": ["github.tools.sap/hcm/memforge-cloud"],
        },
        include_private=True,
        active_repo_identifier="github.tools.sap/hcm/memforge-cloud",
        status="active",
    )

    assert client.calls == [
        (
            "POST",
            "/api/memories/search",
            {
                "query": "scheduler fixes last week",
                "top_k": 10,
                "include_superseded": False,
                "source_filter": {
                    "source_ids": ["src-codex-session"],
                    "clients": ["codex"],
                    "repo_identifiers": ["github.tools.sap/hcm/memforge-cloud"],
                },
                "include_private": True,
                "active_repo_identifier": "github.tools.sap/hcm/memforge-cloud",
                "status": "active",
            },
        )
    ]


def test_tool_client_get_memory_uses_personalized_detail_route():
    client = _RecordingClient({"id": "mem-private"})

    result = client.get_memory("mem-private")

    assert result["id"] == "mem-private"
    assert client.calls == [
        ("GET", "/api/memories/mem-private?include_private=true", None),
    ]


def test_tool_client_create_memory_posts_user_memory_payload():
    client = _RecordingClient({"memory_id": "mem-new", "status": "inserted"})

    result = client.create_memory(
        content="Use readable confirmation previews before memory mutations.",
        provenance="User asked to remember this after reviewing the MemForge MCP UX.",
        memory_type="convention",
        tags=["ux", "mcp"],
        client="codex",
        repo_identifier="github.com/shno-labs/mem-forge",
    )

    assert result["memory_id"] == "mem-new"
    assert client.calls == [
        (
            "POST",
            "/api/memories/create",
            {
                "content": "Use readable confirmation previews before memory mutations.",
                "provenance": "User asked to remember this after reviewing the MemForge MCP UX.",
                "memory_type": "convention",
                "tags": ["ux", "mcp"],
                "client": "codex",
                "repo_identifier": "github.com/shno-labs/mem-forge",
            },
        )
    ]


def test_tool_client_retire_memory_posts_lifecycle_guard():
    client = _RecordingClient({"memory_id": "mem-1", "status": "retired"})

    result = client.retire_memory(
        "mem-1",
        reason="User says this is stale",
        expected_content_hash="hash-1",
    )

    assert result["status"] == "retired"
    assert client.calls == [
        (
            "POST",
            "/api/memories/mem-1/retire",
            {"reason": "User says this is stale", "expected_content_hash": "hash-1"},
        )
    ]


def test_tool_client_replace_memory_posts_lifecycle_guard():
    client = _RecordingClient({"memory_id": "mem-1", "replacement_memory_id": "mem-2"})

    result = client.replace_memory(
        "mem-1",
        replacement_content="Corrected memory",
        reason="User corrected it",
        expected_content_hash="hash-1",
        replacement_kind="revision",
    )

    assert result["replacement_memory_id"] == "mem-2"
    assert client.calls == [
        (
            "POST",
            "/api/memories/mem-1/replace",
            {
                "replacement_content": "Corrected memory",
                "reason": "User corrected it",
                "expected_content_hash": "hash-1",
                "replacement_kind": "revision",
            },
        )
    ]


def test_tool_client_memory_review_methods_use_review_endpoints():
    client = _RecordingClient({"ok": True})

    client.list_memory_reviews(status="open", limit=5, offset=2)
    client.get_memory_review("rev-1")
    client.resolve_memory_review("rev-1", decision="reject", note="Not durable", reviewer="alice")

    assert client.calls == [
        ("GET", "/api/memory-reviews?status=open&limit=5&offset=2", None),
        ("GET", "/api/memory-reviews/rev-1", None),
        ("POST", "/api/memory-reviews/rev-1/reject", {"note": "Not durable", "reviewer": "alice"}),
    ]


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
