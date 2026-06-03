"""Contract tests for the source management methods on ToolClient.

These verify the HTTP method, path, and body the client sends, since those are
the parts most likely to drift from the admin API. The transport itself
(`_http_json`) is exercised by the agent-tool read paths elsewhere.
"""

from __future__ import annotations

from typing import Any

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
