import json

from click.testing import CliRunner

import memforge.main as main
from memforge.main import cli


class FakeToolClient:
    calls: list[tuple[str, dict]] = []
    response: dict = {}

    def __init__(self, *, api_url: str, api_token: str | None = None):
        self.api_url = api_url
        self.api_token = api_token

    @classmethod
    def reset(cls, response: dict) -> None:
        cls.calls = []
        cls.response = response

    def search(self, **kwargs):
        self.calls.append(("search", {"api_url": self.api_url, "api_token": self.api_token, **kwargs}))
        return self.response

    def get_memory(self, memory_id: str):
        self.calls.append(("get_memory", {"api_url": self.api_url, "api_token": self.api_token, "memory_id": memory_id}))
        return self.response

    def get_resource(self, **kwargs):
        self.calls.append(("get_resource", {"api_url": self.api_url, "api_token": self.api_token, **kwargs}))
        return self.response


def test_search_cli_forwards_mcp_shape(monkeypatch):
    FakeToolClient.reset(
        {
            "results": [
                {
                    "memory_id": "mem-docker",
                    "summary": "Docker-hosted MemForge serves artifacts through provenance URLs.",
                }
            ]
        }
    )
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        [
            "search",
            "docker artifact provenance",
            "--top-k",
            "3",
            "--type",
            "fact",
            "--source",
            "src-docs",
            "--include-superseded",
        ],
        env={"MEMFORGE_API_URL": "https://memforge.example.test", "MEMFORGE_API_TOKEN": "token-1"},
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["results"][0]["memory_id"] == "mem-docker"
    assert FakeToolClient.calls == [
        (
            "search",
            {
                "api_url": "https://memforge.example.test",
                "api_token": "token-1",
                "query": "docker artifact provenance",
                "top_k": 3,
                "memory_types": ["fact"],
                "sources": ["src-docs"],
                "include_superseded": True,
            },
        )
    ]


def test_get_memory_cli_fetches_memory_detail(monkeypatch):
    FakeToolClient.reset({"memory_id": "mem-123", "content": "A memory"})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(cli, ["get-memory", "mem-123"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["memory_id"] == "mem-123"
    assert FakeToolClient.calls[0][0] == "get_memory"
    assert FakeToolClient.calls[0][1]["memory_id"] == "mem-123"


def test_get_resource_cli_supports_file_mode(monkeypatch):
    FakeToolClient.reset(
        {
            "doc_id": "doc-456",
            "kind": "pdf",
            "mode": "file",
            "local_path": "/tmp/memforge/doc-456.pdf",
            "cleanup": "temporary-cache",
        }
    )
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        ["get-resource", "/api/documents/doc-456/pdf", "--mode", "file", "--max-bytes", "1024"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["local_path"] == "/tmp/memforge/doc-456.pdf"
    assert FakeToolClient.calls == [
        (
            "get_resource",
            {
                "api_url": "http://127.0.0.1:8765",
                "api_token": None,
                "url": "/api/documents/doc-456/pdf",
                "mode": "file",
                "max_chars": 120000,
                "max_bytes": 1024,
            },
        )
    ]


def test_agent_tool_cli_returns_nonzero_for_tool_error(monkeypatch):
    FakeToolClient.reset({"error": "MemForge API unavailable", "detail": "connection refused"})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(cli, ["get-memory", "mem-123"])

    assert result.exit_code == 1
    assert json.loads(result.output)["error"] == "MemForge API unavailable"
