import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

import memforge.main as main
from memforge.main import cli


class FakeToolClient:
    calls: list[tuple[str, dict]] = []
    response: dict = {}
    list_response: dict = {"data": []}
    create_response: dict = {"id": "src-created"}

    def __init__(
        self,
        *,
        api_url: str,
        api_token: str | None = None,
        workspace_id: str | None = None,
        timeout_seconds: float = 60.0,
    ):
        self.api_url = api_url
        self.api_token = api_token
        self.workspace_id = (workspace_id or "").strip()
        self.timeout_seconds = timeout_seconds

    @classmethod
    def reset(cls, response: dict, *, list_response: dict | None = None, create_response: dict | None = None) -> None:
        cls.calls = []
        cls.response = response
        cls.list_response = {"data": []} if list_response is None else list_response
        cls.create_response = {"id": "src-created"} if create_response is None else create_response

    def list_sources(self):
        self.calls.append(("list_sources", {"api_url": self.api_url, "api_token": self.api_token}))
        return self.list_response

    def list_searchable_sources(self):
        self.calls.append(("list_searchable_sources", {"api_url": self.api_url, "api_token": self.api_token}))
        return self.list_response

    def create_source(self, **kwargs):
        self.calls.append(("create_source", {"api_url": self.api_url, "api_token": self.api_token, **kwargs}))
        return self.create_response

    def get_source_schedule(self, source_id: str):
        self.calls.append((
            "get_source_schedule",
            {"api_url": self.api_url, "api_token": self.api_token, "source_id": source_id},
        ))
        return self.response

    def update_source_schedule(self, **kwargs):
        self.calls.append(("update_source_schedule", {"api_url": self.api_url, "api_token": self.api_token, **kwargs}))
        return self.response

    def search(self, **kwargs):
        self.calls.append(("search", {"api_url": self.api_url, "api_token": self.api_token, **kwargs}))
        return self.response

    def get_memory(self, memory_id: str):
        self.calls.append(("get_memory", {"api_url": self.api_url, "api_token": self.api_token, "memory_id": memory_id}))
        return self.response

    def get_resource(self, **kwargs):
        self.calls.append(("get_resource", {"api_url": self.api_url, "api_token": self.api_token, **kwargs}))
        return self.response

    def start_source_sync(self, **kwargs):
        self.calls.append((
            "start_source_sync",
            {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
        ))
        return self.response

    def push_local_markdown_document(self, **kwargs):
        self.calls.append((
            "push_local_markdown_document",
            {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
        ))
        return self.response

    def push_github_repo_document(self, **kwargs):
        self.calls.append((
            "push_github_repo_document",
            {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
        ))
        return self.response

    def push_jira_document(self, **kwargs):
        self.calls.append((
            "push_jira_document",
            {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
        ))
        return self.response

    def health(self):
        self.calls.append(("health", {"api_url": self.api_url, "api_token": self.api_token}))
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
                "include_superseded": True,
            },
        )
    ]


def test_search_cli_forwards_exact_source_ids(monkeypatch):
    FakeToolClient.reset({"results": []})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        [
            "search",
            "--source-id",
            "src-mounttai",
            "--start-date",
            "2026-06-20",
            "--end-date",
            "2026-06-26",
        ],
        env={"MEMFORGE_API_URL": "https://memforge.example.test", "MEMFORGE_API_TOKEN": "token-1"},
    )

    assert result.exit_code == 0, result.output
    assert FakeToolClient.calls == [
        (
            "search",
            {
                "api_url": "https://memforge.example.test",
                "api_token": "token-1",
                "query": "",
                "top_k": 10,
                "include_superseded": False,
                "source_filter": {"source_ids": ["src-mounttai"]},
                "time_range": {
                    "date_type": "source_updated_at",
                    "start_date": "2026-06-20",
                    "end_date": "2026-06-26",
                },
            },
        )
    ]


def test_search_cli_rejects_legacy_source_name_filter(monkeypatch):
    FakeToolClient.reset({"results": []})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        ["search", "jira defects", "--source", "Matterhorn Defects"],
        env={"MEMFORGE_API_URL": "https://memforge.example.test"},
    )

    assert result.exit_code != 0
    assert "No such option '--source'" in result.output
    assert FakeToolClient.calls == []


def test_get_memory_cli_fetches_memory_detail(monkeypatch):
    FakeToolClient.reset({"memory_id": "mem-123", "content": "A memory"})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(cli, ["get-memory", "mem-123"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["memory_id"] == "mem-123"
    assert FakeToolClient.calls[0][0] == "get_memory"
    assert FakeToolClient.calls[0][1]["memory_id"] == "mem-123"


def test_get_resource_cli_supports_file_mode(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(tmp_path / "cli.toml"))
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


def test_memory_group_keeps_read_tools_api_backed(monkeypatch):
    FakeToolClient.reset({"memory_id": "mem-456", "content": "A memory"})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(cli, ["memory", "get", "mem-456"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["memory_id"] == "mem-456"
    assert FakeToolClient.calls[0][0] == "get_memory"


def test_sources_list_cli_reads_configured_sources_from_active_api_target(monkeypatch):
    FakeToolClient.reset(
        {},
        list_response={
            "data": [
                {
                    "id": "src-paused",
                    "source_id": "src-paused",
                    "name": "MountTai Defects",
                    "type": "jira",
                    "status": "paused",
                    "doc_count": 68,
                    "memory_count": 199,
                    "last_synced_at": "2026-06-27T00:00:00Z",
                }
            ]
        },
    )
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        ["sources", "list"],
        env={"MEMFORGE_API_URL": "https://memforge.example.test", "MEMFORGE_API_TOKEN": "token-1"},
    )

    assert result.exit_code == 0, result.output
    assert "Configured Sources" in result.output
    assert "src-paused" in result.output
    assert "paused" in result.output
    assert FakeToolClient.calls == [
        (
            "list_sources",
            {"api_url": "https://memforge.example.test", "api_token": "token-1"},
        )
    ]


def test_sources_searchable_cli_reads_searchable_sources_from_active_api_target(monkeypatch):
    FakeToolClient.reset(
        {},
        list_response={
            "data": [
                {
                    "source_id": "src-mounttai",
                    "name": "MountTai Defects",
                    "type": "jira",
                    "status": "active",
                    "doc_count": 68,
                    "memory_count": 199,
                    "last_synced_at": "2026-06-27T00:00:00Z",
                }
            ]
        },
    )
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        ["sources", "searchable"],
        env={"MEMFORGE_API_URL": "https://memforge.example.test", "MEMFORGE_API_TOKEN": "token-1"},
    )

    assert result.exit_code == 0, result.output
    assert "Searchable Sources" in result.output
    assert "src-mounttai" in result.output
    assert FakeToolClient.calls == [
        (
            "list_searchable_sources",
            {"api_url": "https://memforge.example.test", "api_token": "token-1"},
        )
    ]


def test_sources_schedule_cli_updates_active_api_target(monkeypatch):
    FakeToolClient.reset({"ok": True, "source_id": "src-1"})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        ["sources", "schedule", "src-1", "--every-minutes", "60"],
        env={"MEMFORGE_API_URL": "https://memforge.example.test", "MEMFORGE_API_TOKEN": "token-1"},
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["ok"] is True
    assert FakeToolClient.calls == [
        (
            "update_source_schedule",
            {
                "api_url": "https://memforge.example.test",
                "api_token": "token-1",
                "source_id": "src-1",
                "enabled": True,
                "interval_minutes": 60,
            },
        )
    ]


def test_sources_schedule_show_cli_reads_active_api_target(monkeypatch):
    FakeToolClient.reset({"enabled": True, "interval_minutes": 60})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        ["sources", "schedule-show", "src-1"],
        env={"MEMFORGE_API_URL": "https://memforge.example.test", "MEMFORGE_API_TOKEN": "token-1"},
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["interval_minutes"] == 60
    assert FakeToolClient.calls == [
        (
            "get_source_schedule",
            {
                "api_url": "https://memforge.example.test",
                "api_token": "token-1",
                "source_id": "src-1",
            },
        )
    ]


def test_target_profile_sets_default_api_url(monkeypatch, tmp_path: Path):
    cli_config = tmp_path / "cli.toml"
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(cli_config))
    FakeToolClient.reset({"status": "ok"})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    add_result = CliRunner().invoke(
        cli,
        ["target", "add", "sap.prod", "--api-url", "https://memforge.example.test", "--token-env", "SAP_TOKEN"],
    )
    check_result = CliRunner().invoke(cli, ["target", "check"], env={"SAP_TOKEN": "secret-token"})

    assert add_result.exit_code == 0, add_result.output
    assert check_result.exit_code == 0, check_result.output
    assert FakeToolClient.calls == [
        ("health", {"api_url": "https://memforge.example.test", "api_token": "secret-token"})
    ]


def test_target_profile_does_not_mix_global_token(monkeypatch, tmp_path: Path):
    cli_config = tmp_path / "cli.toml"
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(cli_config))
    FakeToolClient.reset({"status": "ok"})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    add_result = CliRunner().invoke(
        cli,
        ["target", "add", "sap", "--api-url", "https://memforge.example.test", "--token-env", "SAP_TOKEN"],
    )
    check_result = CliRunner().invoke(
        cli,
        ["target", "check"],
        env={"MEMFORGE_API_TOKEN": "wrong-token", "SAP_TOKEN": "target-token"},
    )

    assert add_result.exit_code == 0, add_result.output
    assert check_result.exit_code == 0, check_result.output
    assert FakeToolClient.calls == [
        ("health", {"api_url": "https://memforge.example.test", "api_token": "target-token"})
    ]


def test_env_api_url_does_not_use_active_target_token(monkeypatch, tmp_path: Path):
    cli_config = tmp_path / "cli.toml"
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(cli_config))
    FakeToolClient.reset({"status": "ok"})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    add_result = CliRunner().invoke(
        cli,
        ["target", "add", "sap", "--api-url", "https://memforge.example.test", "--token-env", "SAP_TOKEN"],
    )
    check_result = CliRunner().invoke(
        cli,
        ["target", "check"],
        env={"MEMFORGE_API_URL": "https://override.example.test", "SAP_TOKEN": "target-token"},
    )

    assert add_result.exit_code == 0, add_result.output
    assert check_result.exit_code == 0, check_result.output
    assert FakeToolClient.calls == [
        ("health", {"api_url": "https://override.example.test", "api_token": None})
    ]


class _StubClient:
    def __init__(self, **responses):
        self._responses = responses
        self.calls = []

    def get_jira_session(self, base_url):
        self.calls.append(("get_jira_session", base_url))
        return self._responses.get("get_jira_session", {})

    def list_jira_origins(self):
        self.calls.append(("list_jira_origins", None))
        return self._responses.get("list_jira_origins", {"origins": []})

    def forget_jira_session(self, base_url):
        self.calls.append(("forget_jira_session", base_url))
        return self._responses.get("forget_jira_session", {"ok": True, "forgotten": True})

    def upload_jira_session(self, **kwargs):
        self.calls.append(("upload_jira_session", kwargs))
        return self._responses.get("upload_jira_session", {})


def test_adapter_jira_status_reports_stored_session(monkeypatch):
    stub = _StubClient(get_jira_session={
        "provider": "jira", "origin": "https://jira.tools.sap", "status": "active",
        "principal_name": "Rose H", "browser": "chrome",
    })
    monkeypatch.setattr(main, "_tool_client", lambda ctx: stub)
    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "status", "--base-url", "https://jira.tools.sap"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "active"
    assert payload["principal_name"] == "Rose H"
    assert stub.calls == [("get_jira_session", "https://jira.tools.sap")]


def test_adapter_jira_status_missing_session_is_not_an_error(monkeypatch):
    stub = _StubClient(get_jira_session={"provider": "jira", "origin": "https://jira.tools.sap", "status": "missing"})
    monkeypatch.setattr(main, "_tool_client", lambda ctx: stub)
    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "status", "--base-url", "https://jira.tools.sap"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "missing"


def test_adapter_jira_refresh_captures_and_uploads(monkeypatch):
    from memforge.auth import jira_capture

    captured = {}

    async def fake_capture(base_url, *, browser=None):
        captured["base_url"] = base_url
        captured["browser"] = browser
        return jira_capture.JiraCaptureResult(
            origin=base_url, cookie_header="SESSION=x", browser=browser, principal={"accountId": "u1"},
        )

    stub = _StubClient(upload_jira_session={"provider": "jira", "origin": "https://jira.tools.sap", "status": "active"})
    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    monkeypatch.setattr(main, "_tool_client", lambda ctx: stub)

    result = CliRunner().invoke(
        cli, ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.tools.sap", "--browser", "chrome"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "active"
    assert captured == {"base_url": "https://jira.tools.sap", "browser": "chrome"}
    assert stub.calls[0][0] == "upload_jira_session"
    assert stub.calls[0][1]["cookie_header"] == "SESSION=x"
    assert stub.calls[0][1]["base_url"] == "https://jira.tools.sap"


def test_adapter_jira_refresh_no_session_returns_json_error(monkeypatch):
    from memforge.auth import jira_capture
    from memforge.auth.jira_auth import JiraAuthSessionMissingError

    async def fake_capture(base_url, *, browser=None):
        raise JiraAuthSessionMissingError("No active Jira browser session cookies were found")

    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    monkeypatch.setattr(main, "_tool_client", lambda ctx: _StubClient())

    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.tools.sap"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "no_session"
    assert "No active Jira" in payload["detail"]


def test_adapter_jira_refresh_capture_error_returns_json_error(monkeypatch):
    from memforge.auth import jira_capture

    async def fake_capture(base_url, *, browser=None):
        raise ValueError("Unsupported browser for Jira session extraction: netscape")

    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    monkeypatch.setattr(main, "_tool_client", lambda ctx: _StubClient())

    result = CliRunner().invoke(
        cli, ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.tools.sap", "--browser", "netscape"],
    )
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "auth_failed"
    assert "Unsupported browser" in payload["detail"]


def test_adapter_jira_refresh_principal_change_returns_json_error(monkeypatch):
    from memforge.auth import jira_capture

    async def fake_capture(base_url, *, browser=None):
        return jira_capture.JiraCaptureResult(
            origin=base_url, cookie_header="SESSION=x", browser=None, principal={"accountId": "u1"},
        )

    body = json.dumps({"detail": {
        "message": "changed", "origin": "https://jira.tools.sap",
        "old_principal_id": "old-user", "new_principal_id": "new-user",
    }})
    stub = _StubClient(upload_jira_session={
        "error": "MemForge API request failed", "status_code": 409, "detail": body,
    })
    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    monkeypatch.setattr(main, "_tool_client", lambda ctx: stub)

    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.tools.sap"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "principal_changed"
    assert payload["origin"] == "https://jira.tools.sap"
    assert payload["old_principal_id"] == "old-user"
    assert payload["new_principal_id"] == "new-user"


def test_adapter_jira_list_and_forget(monkeypatch):
    stub = _StubClient(
        list_jira_origins={"origins": [{"origin": "https://jira.tools.sap", "status": "active"}]},
        forget_jira_session={"ok": True, "origin": "https://jira.tools.sap", "forgotten": True},
    )
    monkeypatch.setattr(main, "_tool_client", lambda ctx: stub)
    list_result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "list"])
    assert list_result.exit_code == 0, list_result.output
    assert json.loads(list_result.output)["origins"][0]["origin"] == "https://jira.tools.sap"
    forget_result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "forget", "--base-url", "https://jira.tools.sap"])
    assert forget_result.exit_code == 0, forget_result.output
    assert json.loads(forget_result.output)["forgotten"] is True


def test_legacy_auth_jira_is_removed():
    help_result = CliRunner().invoke(cli, ["auth", "--help"])
    result = CliRunner().invoke(cli, ["auth", "jira", "--base-url", "https://jira.tools.sap"])

    assert help_result.exit_code == 0, help_result.output
    assert "jira" not in help_result.output
    assert result.exit_code != 0
    assert "No such command 'jira'" in result.output


def test_adapter_list_includes_markdown_kb_capability():
    result = CliRunner().invoke(cli, ["adapter", "list"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {"type": "kb", "kind": "markdown"} in payload["data"]


def test_adapter_kb_add_and_list_profiles(monkeypatch, tmp_path: Path):
    adapter_config = tmp_path / "adapter.toml"
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(adapter_config))

    add_result = CliRunner().invoke(
        cli,
        [
            "adapter",
            "kb",
            "add",
            "work.vault",
            "--root",
            str(vault_root),
            "--vault-id",
            "work-vault",
            "--include",
            "**/*.md",
            "--exclude",
            ".obsidian/**",
        ],
    )
    list_result = CliRunner().invoke(cli, ["adapter", "kb", "list"])

    assert add_result.exit_code == 0, add_result.output
    assert list_result.exit_code == 0, list_result.output
    payload = json.loads(list_result.output)
    assert payload["profiles"]["work.vault"]["root"] == str(vault_root)
    assert payload["profiles"]["work.vault"]["vault_id"] == "work-vault"


def test_adapter_kb_scan_reports_counts_without_a_profile(monkeypatch, tmp_path: Path):
    vault_root = tmp_path / "vault"
    (vault_root / "notes").mkdir(parents=True)
    (vault_root / ".obsidian").mkdir()
    (vault_root / "notes" / "a.md").write_text("# A\n\nbody", encoding="utf-8")
    (vault_root / "notes" / "b.md").write_text("# B\n\nbody", encoding="utf-8")
    (vault_root / ".obsidian" / "workspace.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))

    result = CliRunner().invoke(cli, ["adapter", "kb", "scan", "--root", str(vault_root), "--limit", "5"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["counts"]["included"] == 2
    paths = sorted(item["relative_path"] for item in payload["items"])
    assert paths == ["notes/a.md", "notes/b.md"]


def test_adapter_kb_scan_rejects_missing_root(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    result = CliRunner().invoke(cli, ["adapter", "kb", "scan", "--root", str(tmp_path / "nope")])
    assert result.exit_code != 0


def test_adapter_kb_remove_deletes_profile(monkeypatch, tmp_path: Path):
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))

    CliRunner().invoke(cli, ["adapter", "kb", "add", "work", "--root", str(vault_root)])
    remove_result = CliRunner().invoke(cli, ["adapter", "kb", "remove", "work"])
    list_result = CliRunner().invoke(cli, ["adapter", "kb", "list"])

    assert remove_result.exit_code == 0, remove_result.output
    assert json.loads(remove_result.output) == {"ok": True, "removed": "work"}
    assert json.loads(list_result.output)["profiles"] == {}


def test_adapter_kb_remove_unknown_profile_errors(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    result = CliRunner().invoke(cli, ["adapter", "kb", "remove", "nope"])
    assert result.exit_code != 0


def test_adapter_kb_add_create_source_creates_and_stores_source_id(monkeypatch, tmp_path: Path):
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({}, list_response={"data": []},
                         create_response={"id": "src-new123", "type": "local_markdown"})

    add_result = CliRunner().invoke(
        cli,
        ["adapter", "kb", "add", "work", "--root", str(vault_root), "--vault-id", "engineering",
         "--display-label", "Eng notes", "--create-source"],
    )
    list_result = CliRunner().invoke(cli, ["adapter", "kb", "list"])

    assert add_result.exit_code == 0, add_result.output
    payload = json.loads(add_result.output)
    assert payload["source_id"] == "src-new123"
    assert payload["source_reused"] is False
    assert json.loads(list_result.output)["profiles"]["work"]["source_id"] == "src-new123"

    create_calls = [c for c in FakeToolClient.calls if c[0] == "create_source"]
    assert len(create_calls) == 1
    assert create_calls[0][1]["source_type"] == "local_markdown"
    assert create_calls[0][1]["config"]["vault_id"] == "engineering"
    assert create_calls[0][1]["config"]["display_label"] == "Eng notes"


def test_adapter_kb_add_create_source_reuses_existing_by_vault_id(monkeypatch, tmp_path: Path):
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset(
        {},
        list_response={"data": [
            {"id": "src-other", "type": "jira", "config": {"vault_id": "engineering"}},
            {"id": "src-exist", "type": "local_markdown", "config": {"vault_id": "engineering"}},
        ]},
        create_response={"id": "should-not-be-used"},
    )

    add_result = CliRunner().invoke(
        cli,
        ["adapter", "kb", "add", "work", "--root", str(vault_root), "--vault-id", "engineering", "--create-source"],
    )

    assert add_result.exit_code == 0, add_result.output
    payload = json.loads(add_result.output)
    assert payload["source_id"] == "src-exist"
    assert payload["source_reused"] is True
    assert not [c for c in FakeToolClient.calls if c[0] == "create_source"]


def test_adapter_kb_add_create_source_reports_link_error_but_saves_profile(monkeypatch, tmp_path: Path):
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({}, list_response={"error": "MemForge API unavailable", "detail": "connection refused"})

    add_result = CliRunner().invoke(
        cli,
        ["adapter", "kb", "add", "work", "--root", str(vault_root), "--vault-id", "engineering", "--create-source"],
    )
    list_result = CliRunner().invoke(cli, ["adapter", "kb", "list"])

    assert add_result.exit_code == 0, add_result.output
    payload = json.loads(add_result.output)
    assert "source_id" not in payload
    assert payload["source_link_error"] == "MemForge API unavailable"
    assert json.loads(list_result.output)["profiles"]["work"]["root"] == str(vault_root)


def test_adapter_kb_push_uses_profile_source_id_when_omitted(monkeypatch, tmp_path: Path):
    vault_root = tmp_path / "vault"
    (vault_root / "notes").mkdir(parents=True)
    (vault_root / "notes" / "a.md").write_text("# A\n\nbody", encoding="utf-8")
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "d1", "document_hash": "h1"},
                         list_response={"data": []}, create_response={"id": "src-stored"})

    CliRunner().invoke(cli, ["adapter", "kb", "add", "work", "--root", str(vault_root),
                             "--vault-id", "engineering", "--create-source"])
    push_result = CliRunner().invoke(cli, ["adapter", "kb", "push", "work"])

    assert push_result.exit_code == 0, push_result.output
    payload = json.loads(push_result.output)
    assert payload["source_id"] == "src-stored"
    push_calls = [c for c in FakeToolClient.calls if c[0] == "push_local_markdown_document"]
    assert push_calls and push_calls[0][1]["source_id"] == "src-stored"


def test_adapter_kb_push_explicit_source_id_overrides_profile(monkeypatch, tmp_path: Path):
    vault_root = tmp_path / "vault"
    (vault_root / "notes").mkdir(parents=True)
    (vault_root / "notes" / "a.md").write_text("# A\n\nbody", encoding="utf-8")
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "d1", "document_hash": "h1"},
                         list_response={"data": []}, create_response={"id": "src-stored"})

    CliRunner().invoke(cli, ["adapter", "kb", "add", "work", "--root", str(vault_root),
                             "--vault-id", "engineering", "--create-source"])
    push_result = CliRunner().invoke(cli, ["adapter", "kb", "push", "work", "--source-id", "src-explicit"])

    assert push_result.exit_code == 0, push_result.output
    assert json.loads(push_result.output)["source_id"] == "src-explicit"


def test_adapter_kb_push_without_any_source_id_errors(monkeypatch, tmp_path: Path):
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({})

    CliRunner().invoke(cli, ["adapter", "kb", "add", "work", "--root", str(vault_root)])
    push_result = CliRunner().invoke(cli, ["adapter", "kb", "push", "work"])

    assert push_result.exit_code != 0


def test_adapter_kb_add_without_create_source_makes_no_api_calls(monkeypatch, tmp_path: Path):
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({})

    add_result = CliRunner().invoke(
        cli, ["adapter", "kb", "add", "work", "--root", str(vault_root)],
    )

    assert add_result.exit_code == 0, add_result.output
    assert FakeToolClient.calls == []
    assert "source_id" not in json.loads(add_result.output)


def test_adapter_kb_preview_scans_markdown_profile(monkeypatch, tmp_path: Path):
    adapter_config = tmp_path / "adapter.toml"
    vault_root = tmp_path / "vault"
    (vault_root / "notes").mkdir(parents=True)
    (vault_root / ".obsidian").mkdir()
    (vault_root / "notes" / "cutoff.md").write_text("# Cutoff\n\nA durable note.", encoding="utf-8")
    (vault_root / ".obsidian" / "workspace.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(adapter_config))

    add_result = CliRunner().invoke(
        cli,
        ["adapter", "kb", "add", "work", "--root", str(vault_root), "--exclude", ".obsidian/**"],
    )
    preview_result = CliRunner().invoke(cli, ["adapter", "kb", "preview", "work", "--limit", "5"])

    assert add_result.exit_code == 0, add_result.output
    assert preview_result.exit_code == 0, preview_result.output
    payload = json.loads(preview_result.output)
    assert payload["profile"] == "work"
    assert payload["counts"]["included"] == 1
    assert payload["items"][0]["relative_path"] == "notes/cutoff.md"
    assert payload["items"][0]["content_type"] == "text/markdown"


def test_adapter_kb_custom_excludes_keep_default_safety_excludes(monkeypatch, tmp_path: Path):
    adapter_config = tmp_path / "adapter.toml"
    vault_root = tmp_path / "vault"
    (vault_root / ".obsidian").mkdir(parents=True)
    (vault_root / "archive").mkdir()
    (vault_root / "notes").mkdir()
    (vault_root / ".obsidian" / "private.md").write_text("# Private\n", encoding="utf-8")
    (vault_root / "archive" / "old.md").write_text("# Old\n", encoding="utf-8")
    (vault_root / "notes" / "live.md").write_text("# Live\n", encoding="utf-8")
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(adapter_config))

    add_result = CliRunner().invoke(
        cli,
        ["adapter", "kb", "add", "work", "--root", str(vault_root), "--exclude", "archive/**"],
    )
    preview_result = CliRunner().invoke(cli, ["adapter", "kb", "preview", "work", "--limit", "10"])

    assert add_result.exit_code == 0, add_result.output
    assert preview_result.exit_code == 0, preview_result.output
    payload = json.loads(preview_result.output)
    assert payload["counts"]["included"] == 1
    assert [item["relative_path"] for item in payload["items"]] == ["notes/live.md"]


def test_adapter_kb_preview_rejects_missing_root(monkeypatch, tmp_path: Path):
    adapter_config = tmp_path / "adapter.toml"
    adapter_config.write_text('[kb."bad"]\nvault_id = "bad"\ninclude = ["*.md"]\n', encoding="utf-8")
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(adapter_config))

    result = CliRunner().invoke(cli, ["adapter", "kb", "preview", "bad"])

    assert result.exit_code != 0
    assert "root is required" in result.output


def test_adapter_kb_preview_does_not_follow_symlink_escape(monkeypatch, tmp_path: Path):
    adapter_config = tmp_path / "adapter.toml"
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("# Outside\n", encoding="utf-8")
    try:
        (vault_root / "leak.md").symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink unavailable: {exc}")
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(adapter_config))

    add_result = CliRunner().invoke(cli, ["adapter", "kb", "add", "work", "--root", str(vault_root)])
    preview_result = CliRunner().invoke(cli, ["adapter", "kb", "preview", "work", "--limit", "10"])

    assert add_result.exit_code == 0, add_result.output
    assert preview_result.exit_code == 0, preview_result.output
    payload = json.loads(preview_result.output)
    assert payload["counts"]["included"] == 0
    assert payload["counts"]["ignored"] == 1


def test_adapter_kb_push_forwards_documents_to_service(monkeypatch, tmp_path: Path):
    adapter_config = tmp_path / "adapter.toml"
    vault_root = tmp_path / "vault"
    (vault_root / "decisions").mkdir(parents=True)
    (vault_root / "decisions" / "cutoff.md").write_text("# Cutoff\n\nTuesday.", encoding="utf-8")
    (vault_root / "decisions" / "release.md").write_text("# Release\n\nThursday.", encoding="utf-8")
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(adapter_config))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "local-md-fake", "document_hash": "abc123"})

    add_result = CliRunner().invoke(
        cli,
        ["adapter", "kb", "add", "work", "--root", str(vault_root), "--vault-id", "engineering"],
    )
    push_result = CliRunner().invoke(
        cli,
        ["adapter", "kb", "push", "work", "--source-id", "src-abcd1234", "--submitted-by", "cli-test"],
    )

    assert add_result.exit_code == 0, add_result.output
    assert push_result.exit_code == 0, push_result.output
    payload = json.loads(push_result.output)
    assert payload["counts"]["pushed"] == 2
    assert payload["counts"]["failed"] == 0
    assert payload["source_id"] == "src-abcd1234"
    assert payload["vault_id"] == "engineering"
    relative_paths = sorted(item["relative_path"] for item in payload["pushed"])
    assert relative_paths == ["decisions/cutoff.md", "decisions/release.md"]

    push_calls = [call for call in FakeToolClient.calls if call[0] == "push_local_markdown_document"]
    assert len(push_calls) == 2
    first_kwargs = push_calls[0][1]
    assert first_kwargs["source_id"] == "src-abcd1234"
    assert first_kwargs["vault_id"] == "engineering"
    assert first_kwargs["submitted_by"] == "cli-test"
    assert first_kwargs["process_now"] is False
    # The CLI now sends raw file text tagged with a content type; conversion is server-side.
    assert first_kwargs["markdown_body"] == "# Cutoff\n\nTuesday."
    assert first_kwargs["content_type"] == "text/markdown"


def test_adapter_kb_push_process_now_triggers_on_last_document(monkeypatch, tmp_path: Path):
    adapter_config = tmp_path / "adapter.toml"
    vault_root = tmp_path / "vault"
    (vault_root / "notes").mkdir(parents=True)
    (vault_root / "notes" / "first.md").write_text("# First\n", encoding="utf-8")
    (vault_root / "notes" / "second.md").write_text("# Second\n", encoding="utf-8")
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(adapter_config))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "local-md-fake", "document_hash": "abc123"})

    add_result = CliRunner().invoke(
        cli,
        ["adapter", "kb", "add", "work", "--root", str(vault_root), "--vault-id", "engineering"],
    )
    push_result = CliRunner().invoke(
        cli,
        ["adapter", "kb", "push", "work", "--source-id", "src-abcd1234", "--process-now"],
    )

    assert add_result.exit_code == 0, add_result.output
    assert push_result.exit_code == 0, push_result.output
    push_calls = [call for call in FakeToolClient.calls if call[0] == "push_local_markdown_document"]
    assert [call[1]["process_now"] for call in push_calls] == [False, True]


def test_adapter_kb_push_reports_service_errors(monkeypatch, tmp_path: Path):
    adapter_config = tmp_path / "adapter.toml"
    vault_root = tmp_path / "vault"
    vault_root.mkdir()
    (vault_root / "doc.md").write_text("# Doc\n", encoding="utf-8")
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(adapter_config))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"error": "MemForge API request failed", "status_code": 400, "detail": "vault mismatch"})

    add_result = CliRunner().invoke(
        cli, ["adapter", "kb", "add", "work", "--root", str(vault_root), "--vault-id", "engineering"]
    )
    push_result = CliRunner().invoke(
        cli, ["adapter", "kb", "push", "work", "--source-id", "src-bad"]
    )

    assert add_result.exit_code == 0, add_result.output
    assert push_result.exit_code != 0
    payload = json.loads(push_result.output)
    assert payload["counts"]["failed"] == 1
    assert payload["failed"][0]["status_code"] == 400
    assert "error" in payload


def test_adapter_github_add_and_preview_uses_local_gh_tree(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))

    def fake_run(cmd, *, capture_output, text, env=None, check=False):
        assert cmd[:2] == ["gh", "api"]
        assert env["GH_HOST"] == "github.wdf.sap.corp"
        assert cmd[2] == "repos/nextgenpayroll-matterhorn/architecture/git/trees/main?recursive=1"

        class Result:
            returncode = 0
            stdout = json.dumps(
                {
                    "tree": [
                        {"path": "Payroll Processing/README.md", "type": "blob", "sha": "md-sha", "size": 10},
                        {"path": "Payroll Processing/images/diagram.png", "type": "blob", "sha": "png-sha", "size": 20},
                        {"path": "Flexible Payroll/README.md", "type": "blob", "sha": "other-sha", "size": 30},
                    ]
                }
            )
            stderr = ""

        return Result()

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    add_result = CliRunner().invoke(
        cli,
        [
            "adapter",
            "github",
            "add",
            "matterhorn",
            "--repo-url",
            "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
            "--ref",
            "main",
            "--include-path",
            "Payroll Processing/",
            "--include-extension",
            "md",
        ],
    )
    preview_result = CliRunner().invoke(cli, ["adapter", "github", "preview", "matterhorn", "--limit", "5"])

    assert add_result.exit_code == 0, add_result.output
    assert preview_result.exit_code == 0, preview_result.output
    payload = json.loads(preview_result.output)
    assert payload["repo_url"] == "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture"
    assert payload["counts"]["included"] == 1
    assert payload["counts"]["ignored"] == 2
    assert payload["extension_counts"] == {"md": 1, "png": 1}
    assert payload["items"][0]["relative_path"] == "Payroll Processing/README.md"


def test_adapter_github_add_help_uses_access_copy_not_internal_mode():
    result = CliRunner().invoke(cli, ["adapter", "github", "add", "--help"])
    normalized_output = " ".join(result.output.split())

    assert result.exit_code == 0, result.output
    assert "Internal network / VPN repositories" in normalized_output
    assert "local_push" not in result.output
    assert "cloud_pull" not in result.output


def test_adapter_github_remove_deletes_profile(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))

    add_result = CliRunner().invoke(
        cli,
        [
            "adapter",
            "github",
            "add",
            "matterhorn",
            "--repo-url",
            "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
        ],
    )
    remove_result = CliRunner().invoke(cli, ["adapter", "github", "remove", "matterhorn"])
    list_result = CliRunner().invoke(cli, ["adapter", "github", "list"])

    assert add_result.exit_code == 0, add_result.output
    assert remove_result.exit_code == 0, remove_result.output
    assert json.loads(remove_result.output) == {"ok": True, "removed": "matterhorn"}
    assert json.loads(list_result.output)["profiles"] == {}


def test_adapter_github_remove_unknown_profile_errors(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))

    result = CliRunner().invoke(cli, ["adapter", "github", "remove", "nope"])

    assert result.exit_code != 0


def _init_github_local_clone(
    tmp_path: Path,
    *,
    remote_url: str = "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture.git",
) -> Path:
    repo = tmp_path / "architecture"
    (repo / "Payroll Processing V2" / "images").mkdir(parents=True)
    (repo / "Flexible Payroll").mkdir(parents=True)
    (repo / "Payroll Processing V2" / "README.md").write_text("# Payroll Processing V2\n\nBody", encoding="utf-8")
    (repo / "Payroll Processing V2" / "Überblick.md").write_text("# Überblick\n", encoding="utf-8")
    (repo / "Payroll Processing V2" / "images" / "Flow.puml").write_text("@startuml\n@enduml\n", encoding="utf-8")
    (repo / "Payroll Processing V2" / "images" / "ignored.png").write_bytes(b"png")
    (repo / "Flexible Payroll" / "README.md").write_text("# Flexible Payroll\n", encoding="utf-8")
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "remote", "add", "origin", remote_url],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=MemForge Test",
            "-c",
            "user.email=memforge@example.test",
            "commit",
            "-m",
            "seed architecture docs",
        ],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return repo


def test_adapter_github_local_clone_accepts_credentialed_https_remote(monkeypatch, tmp_path: Path):
    repo = _init_github_local_clone(
        tmp_path,
        remote_url="https://x-access-token:SECRET_TOKEN@github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture.git",
    )
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))

    result = CliRunner().invoke(
        cli,
        [
            "adapter",
            "github",
            "add",
            "matterhorn",
            "--repo-url",
            "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
            "--repo-path",
            str(repo),
            "--ref",
            "main",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "SECRET_TOKEN" not in result.output


def test_adapter_github_local_clone_mismatch_redacts_credentialed_remote(monkeypatch, tmp_path: Path):
    repo = _init_github_local_clone(
        tmp_path,
        remote_url="https://x-access-token:SECRET_TOKEN@github.wdf.sap.corp/other-org/other-repo.git",
    )
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))

    result = CliRunner().invoke(
        cli,
        [
            "adapter",
            "github",
            "add",
            "matterhorn",
            "--repo-url",
            "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
            "--repo-path",
            str(repo),
            "--ref",
            "main",
        ],
    )

    assert result.exit_code != 0
    assert "does not match configured repo_url" in result.output
    assert "SECRET_TOKEN" not in result.output
    assert "x-access-token" not in result.output


def test_adapter_github_preview_can_use_local_clone_without_gh_api(monkeypatch, tmp_path: Path):
    repo = _init_github_local_clone(tmp_path)
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    original_run = main.subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["gh", "api"]:
            raise AssertionError("repo_path preview must not call gh api")
        return original_run(cmd, *args, **kwargs)

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    add_result = CliRunner().invoke(
        cli,
        [
            "adapter",
            "github",
            "add",
            "matterhorn",
            "--repo-url",
            "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
            "--repo-path",
            str(repo),
            "--ref",
            "main",
            "--include-path",
            "Payroll Processing V2",
            "--include-extension",
            "md",
            "--include-extension",
            "puml",
        ],
    )
    preview_result = CliRunner().invoke(cli, ["adapter", "github", "preview", "matterhorn", "--limit", "10"])

    assert add_result.exit_code == 0, add_result.output
    assert preview_result.exit_code == 0, preview_result.output
    payload = json.loads(preview_result.output)
    assert payload["repo_url"] == "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture"
    assert payload["repo_path"] == str(repo.resolve())
    assert payload["counts"] == {"included": 3, "ignored": 2}
    assert payload["extension_counts"] == {"md": 2, "png": 1, "puml": 1}
    assert [item["relative_path"] for item in payload["items"]] == [
        "Payroll Processing V2/README.md",
        "Payroll Processing V2/images/Flow.puml",
        "Payroll Processing V2/Überblick.md",
    ]


def test_adapter_github_push_can_read_local_clone_without_gh_api(monkeypatch, tmp_path: Path):
    repo = _init_github_local_clone(tmp_path)
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "github-repo-doc", "document_hash": "hash"})
    original_run = main.subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["gh", "api"]:
            raise AssertionError("repo_path push must not call gh api")
        return original_run(cmd, *args, **kwargs)

    monkeypatch.setattr(main.subprocess, "run", fake_run)
    expected_blob_sha = original_run(
        ["git", "-C", str(repo), "rev-parse", "main:Payroll Processing V2/README.md"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    add_result = CliRunner().invoke(
        cli,
        [
            "adapter",
            "github",
            "add",
            "matterhorn",
            "--repo-url",
            "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
            "--repo-path",
            str(repo),
            "--ref",
            "main",
            "--include-path",
            "Payroll Processing V2",
            "--include-extension",
            "md",
        ],
    )
    push_result = CliRunner().invoke(
        cli, ["adapter", "github", "push", "matterhorn", "--source-id", "src-gh", "--limit", "1"]
    )

    assert add_result.exit_code == 0, add_result.output
    assert push_result.exit_code == 0, push_result.output
    payload = json.loads(push_result.output)
    assert payload["counts"]["pushed"] == 1
    push_calls = [call for call in FakeToolClient.calls if call[0] == "push_github_repo_document"]
    assert len(push_calls) == 1
    kwargs = push_calls[0][1]
    assert kwargs["repo_url"] == "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture"
    assert kwargs["relative_path"] == "Payroll Processing V2/README.md"
    assert kwargs["blob_sha"] == expected_blob_sha
    assert kwargs["markdown_body"] == "# Payroll Processing V2\n\nBody"


def test_local_agent_cloud_github_preview_uses_job_payload_without_profile(monkeypatch, tmp_path: Path):
    repo = _init_github_local_clone(tmp_path)
    original_run = main.subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["gh", "api"]:
            raise AssertionError("cloud local-agent preview should use the job repo_path, not a local profile or gh api")
        return original_run(cmd, *args, **kwargs)

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-preview",
            "operation": "github_repo_preview_tree",
            "source_id": "src-gh",
            "payload": {
                "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                "repo_path": str(repo),
                "ref": "main",
                "include_paths": ["Payroll Processing V2"],
                "include_extensions": ["md"],
                "limit": 10,
            },
        },
        FakeToolClient(api_url="https://memforge.example.test", api_token="tok"),
    )

    assert payload["operation"] == "github_repo_preview_tree"
    assert payload["source_id"] == "src-gh"
    assert payload["counts"] == {"included": 2, "ignored": 3}
    assert [item["relative_path"] for item in payload["items"]] == [
        "Payroll Processing V2/README.md",
        "Payroll Processing V2/Überblick.md",
    ]


def test_local_agent_cloud_github_sync_pushes_job_source_and_snapshot(monkeypatch, tmp_path: Path):
    repo = _init_github_local_clone(tmp_path)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "github-repo-doc", "document_hash": "hash"})
    original_run = main.subprocess.run

    def fake_run(cmd, *args, **kwargs):
        if cmd[:2] == ["gh", "api"]:
            raise AssertionError("cloud local-agent sync should use the job repo_path, not gh api")
        return original_run(cmd, *args, **kwargs)

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-sync",
            "workspace_id": "ws-from-cloud",
            "operation": "github_repo_sync",
            "source_id": "src-from-cloud",
            "payload": {
                "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                "repo_path": str(repo),
                "ref": "main",
                "include_paths": ["Payroll Processing V2"],
                "include_extensions": ["md"],
                "limit": 1,
                "process_now": True,
            },
        },
        FakeToolClient(api_url="https://memforge.example.test", api_token="tok"),
    )

    assert payload["operation"] == "github_repo_sync"
    assert payload["source_id"] == "src-from-cloud"
    assert payload["counts"] == {"selected": 1, "pushed": 1, "failed": 0}
    push_calls = [call for call in FakeToolClient.calls if call[0] == "push_github_repo_document"]
    assert len(push_calls) == 1
    kwargs = push_calls[0][1]
    assert kwargs["workspace_id"] == "ws-from-cloud"
    assert kwargs["source_id"] == "src-from-cloud"
    assert kwargs["repo_url"] == "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture"
    assert kwargs["relative_path"] == "Payroll Processing V2/README.md"
    assert kwargs["process_now"] is True
    assert kwargs["submitted_by"] == "memforge-local-agent"


def test_local_agent_cloud_github_sync_requires_job_workspace(monkeypatch, tmp_path: Path):
    repo = _init_github_local_clone(tmp_path)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "github-repo-doc", "document_hash": "hash"})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-sync",
            "operation": "github_repo_sync",
            "source_id": "src-from-cloud",
            "payload": {
                "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                "repo_path": str(repo),
                "ref": "main",
                "include_paths": ["Payroll Processing V2"],
                "include_extensions": ["md"],
                "limit": 1,
            },
        },
        FakeToolClient(api_url="https://memforge.example.test", api_token="tok"),
    )

    assert payload == {
        "operation": "github_repo_sync",
        "source_id": "src-from-cloud",
        "error": "workspace_id is required",
    }
    assert FakeToolClient.calls == []


def test_local_agent_cloud_local_markdown_preview_uses_job_payload(tmp_path: Path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "Decision.md").write_text("# Decision\n\nUse the daemon.", encoding="utf-8")
    (root / "scratch.bin").write_bytes(b"\x00\x01")

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-local-preview",
            "operation": "local_markdown_preview_tree",
            "payload": {
                "root": str(root),
                "vault_id": "engineering",
                "include": ["*.md", "**/*.md"],
                "exclude": [],
                "limit": 20,
            },
        },
        FakeToolClient(api_url="https://memforge.example.test", api_token="tok"),
    )

    assert payload["operation"] == "local_markdown_preview_tree"
    assert payload["vault_id"] == "engineering"
    assert payload["counts"]["included"] == 1
    assert [item["relative_path"] for item in payload["items"]] == ["Decision.md"]


def test_local_agent_cloud_local_markdown_pick_root_uses_local_picker(monkeypatch, tmp_path: Path):
    selected = tmp_path / "notes"
    selected.mkdir()
    calls: list[dict] = []

    def fake_pick_folder(*, title: str | None = None, initial_directory: str | None = None) -> str:
        calls.append({"title": title, "initial_directory": initial_directory})
        return str(selected)

    monkeypatch.setattr(main, "pick_folder", fake_pick_folder)

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-pick-root",
            "operation": "local_markdown_pick_root",
            "payload": {
                "title": "Choose folder to sync",
                "initial_directory": str(tmp_path),
            },
        },
        FakeToolClient(api_url="https://memforge.example.test", api_token="tok"),
    )

    assert payload == {
        "operation": "local_markdown_pick_root",
        "root": str(selected),
    }
    assert calls == [{"title": "Choose folder to sync", "initial_directory": str(tmp_path)}]


def test_local_agent_cloud_local_markdown_pick_root_reports_cancellation(monkeypatch):
    def fake_pick_folder(*, title: str | None = None, initial_directory: str | None = None) -> str:
        raise main.FolderPickerCancelled("folder selection cancelled")

    monkeypatch.setattr(main, "pick_folder", fake_pick_folder)

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-pick-root",
            "operation": "local_markdown_pick_root",
            "payload": {"title": "Choose folder to sync"},
        },
        FakeToolClient(api_url="https://memforge.example.test", api_token="tok"),
    )

    assert payload == {
        "operation": "local_markdown_pick_root",
        "cancelled": True,
    }


def test_local_agent_cloud_local_markdown_sync_pushes_workspace_source(monkeypatch, tmp_path: Path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "Decision.md").write_text("# Decision\n\nUse the daemon.", encoding="utf-8")
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "local-doc", "document_hash": "hash"})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-local-sync",
            "workspace_id": "ws-from-cloud",
            "operation": "local_markdown_sync",
            "source_id": "src-local",
            "payload": {
                "root": str(root),
                "vault_id": "engineering",
                "include": ["*.md", "**/*.md"],
                "exclude": [],
                "limit": 1,
                "process_now": True,
            },
        },
        FakeToolClient(api_url="https://memforge.example.test", api_token="tok"),
    )

    assert payload["operation"] == "local_markdown_sync"
    assert payload["source_id"] == "src-local"
    assert payload["counts"]["pushed"] == 1
    push_calls = [call for call in FakeToolClient.calls if call[0] == "push_local_markdown_document"]
    assert len(push_calls) == 1
    kwargs = push_calls[0][1]
    assert kwargs["workspace_id"] == "ws-from-cloud"
    assert kwargs["source_id"] == "src-local"
    assert kwargs["vault_id"] == "engineering"
    assert kwargs["relative_path"] == "Decision.md"
    assert kwargs["process_now"] is True


def test_local_agent_cloud_jira_sync_uses_gene_and_pushes_packages(monkeypatch):
    from datetime import datetime, timezone

    from memforge.auth import jira_capture
    from memforge.models import ContentItem, NormalizedContent, RawContent

    class FakeJiraGene:
        def __init__(self, config, source_id):
            self.config = config
            self.source_id = source_id
            self._client = None

        async def authenticate(self):
            assert self.config["auth_mode"] == "browser_cookie"
            assert self.config["jira_cookie"] == "JSESSIONID=local"
            assert self.config["sync_mode"] == "cloud"
            assert "local_agent_documents_dir" not in self.config
            assert "pat" not in self.config

        async def discover(self, since):
            assert since is None
            yield ContentItem(
                item_id="jira-PAY-1",
                title="Create daemon source support",
                source_url="https://jira.example.test/browse/PAY-1",
                last_modified=datetime(2026, 7, 7, tzinfo=timezone.utc),
                content_type="application/json",
                space_or_project="PAY",
                version="v1",
                author="Andrew",
                labels=["jira"],
                extra={"issue_key": "PAY-1"},
            )

        async def fetch(self, item):
            return RawContent(item=item, body=b"{}", content_type="application/json")

        async def normalize(self, raw):
            return NormalizedContent(
                item=raw.item,
                markdown_body="# PAY-1\n\nCreate daemon source support.",
                source_semantics={"issue_key": "PAY-1", "status": "Open"},
            )

    import memforge.genes.jira_gene as jira_gene

    async def fake_capture(base_url, *, browser=None, tls_config=None):
        assert base_url == "https://jira.example.test"
        assert browser == "chrome"
        assert tls_config["auth_mode"] == "browser_cookie"
        assert tls_config["sync_mode"] == "cloud"
        assert "local_agent_documents_dir" not in tls_config
        assert "pat" not in tls_config
        return jira_capture.JiraCaptureResult(
            origin="https://jira.example.test",
            cookie_header="JSESSIONID=local",
            browser="chrome",
            principal={"name": "tester"},
        )

    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    monkeypatch.setattr(jira_gene, "JiraGene", FakeJiraGene)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "jira-doc", "document_hash": "hash"})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-jira-sync",
            "workspace_id": "ws-from-cloud",
            "operation": "jira_sync",
            "source_id": "src-jira",
            "payload": {
                "base_url": "https://jira.example.test",
                "auth_mode": "browser_cookie",
                "sync_mode": "local_agent",
                "local_agent_documents_dir": "/srv/memforge/inbox/src-jira",
                "pat": "must-not-enter-daemon-runtime-config",
                "projects": ["PAY"],
                "limit": 1,
                "process_now": True,
            },
        },
        FakeToolClient(api_url="https://memforge.example.test", api_token="tok"),
        browser="chrome",
    )

    assert payload["operation"] == "jira_sync"
    assert payload["counts"] == {"selected": 1, "pushed": 1, "failed": 0}
    push_calls = [call for call in FakeToolClient.calls if call[0] == "push_jira_document"]
    assert len(push_calls) == 1
    kwargs = push_calls[0][1]
    assert kwargs["workspace_id"] == "ws-from-cloud"
    assert kwargs["source_id"] == "src-jira"
    assert kwargs["base_url"] == "https://jira.example.test"
    assert kwargs["issue_key"] == "PAY-1"
    assert kwargs["source_semantics"]["status"] == "Open"
    assert kwargs["process_now"] is False
    sync_calls = [call for call in FakeToolClient.calls if call[0] == "start_source_sync"]
    assert len(sync_calls) == 1
    assert sync_calls[0][1]["source_id"] == "src-jira"


def test_local_agent_cloud_jira_sync_rejects_pat_payload():
    FakeToolClient.reset({"doc_id": "jira-doc", "document_hash": "hash"})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-jira-sync",
            "workspace_id": "ws-from-cloud",
            "operation": "jira_sync",
            "source_id": "src-jira",
            "payload": {
                "base_url": "https://jira.example.test",
                "auth_mode": "pat",
                "pat": "should-not-travel",
                "projects": ["PAY"],
            },
        },
        FakeToolClient(api_url="https://memforge.example.test", api_token="tok"),
    )

    assert payload["operation"] == "jira_sync"
    assert payload["error_type"] == "ClickException"
    assert "browser-session" in payload["error"]
    assert not [call for call in FakeToolClient.calls if call[0] == "push_jira_document"]


def test_local_agent_cloud_jira_sync_starts_source_sync_when_last_push_fails(monkeypatch):
    from datetime import datetime, timezone

    from memforge.auth import jira_capture
    from memforge.models import ContentItem, NormalizedContent, RawContent

    class FakeJiraGene:
        def __init__(self, config, source_id):
            self.config = config
            self.source_id = source_id
            self._client = None

        async def authenticate(self):
            assert self.config["sync_mode"] == "cloud"

        async def discover(self, since):
            assert since is None
            for key in ("PAY-1", "PAY-2"):
                yield ContentItem(
                    item_id=f"jira-{key}",
                    title=f"{key} title",
                    source_url=f"https://jira.example.test/browse/{key}",
                    last_modified=datetime(2026, 7, 7, tzinfo=timezone.utc),
                    content_type="application/json",
                    space_or_project="PAY",
                    version="v1",
                    author="Andrew",
                    labels=["jira"],
                    extra={"issue_key": key},
                )

        async def fetch(self, item):
            return RawContent(item=item, body=b"{}", content_type="application/json")

        async def normalize(self, raw):
            key = raw.item.extra["issue_key"]
            return NormalizedContent(
                item=raw.item,
                markdown_body=f"# {key}\n\nLocal daemon package.",
                source_semantics={"issue_key": key, "status": "Open"},
            )

    async def fake_capture(base_url, *, browser=None, tls_config=None):
        return jira_capture.JiraCaptureResult(
            origin=base_url,
            cookie_header="JSESSIONID=local",
            browser=browser,
            principal={"name": "tester"},
        )

    def fake_push_jira_document(self, **kwargs):
        self.calls.append((
            "push_jira_document",
            {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
        ))
        if kwargs["issue_key"] == "PAY-2":
            return {"error": "push failed", "status_code": 500}
        return {"doc_id": f"jira-{kwargs['issue_key'].lower()}", "document_hash": "hash"}

    import memforge.genes.jira_gene as jira_gene

    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    monkeypatch.setattr(jira_gene, "JiraGene", FakeJiraGene)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    monkeypatch.setattr(FakeToolClient, "push_jira_document", fake_push_jira_document)
    FakeToolClient.reset({})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-jira-sync",
            "workspace_id": "ws-from-cloud",
            "operation": "jira_sync",
            "source_id": "src-jira",
            "payload": {
                "base_url": "https://jira.example.test",
                "auth_mode": "browser_cookie",
                "sync_mode": "local_agent",
                "local_agent_documents_dir": "/srv/memforge/inbox/src-jira",
                "projects": ["PAY"],
                "process_now": True,
            },
        },
        FakeToolClient(api_url="https://memforge.example.test", api_token="tok"),
        browser="chrome",
    )

    assert payload["counts"] == {"selected": 2, "pushed": 1, "failed": 1}
    push_calls = [call[1] for call in FakeToolClient.calls if call[0] == "push_jira_document"]
    assert [(call["issue_key"], call["process_now"]) for call in push_calls] == [
        ("PAY-1", False),
        ("PAY-2", False),
    ]
    sync_calls = [call[1] for call in FakeToolClient.calls if call[0] == "start_source_sync"]
    assert len(sync_calls) == 1
    assert sync_calls[0]["source_id"] == "src-jira"


def test_adapter_github_preview_rejects_truncated_tree(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))

    def fake_run(cmd, *, capture_output, text, env=None, check=False):
        assert cmd[:2] == ["gh", "api"]

        class Result:
            returncode = 0
            stdout = json.dumps({"tree": [], "truncated": True})
            stderr = ""

        return Result()

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    add_result = CliRunner().invoke(
        cli,
        [
            "adapter",
            "github",
            "add",
            "matterhorn",
            "--repo-url",
            "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
            "--ref",
            "main",
            "--include-extension",
            "md",
        ],
    )
    preview_result = CliRunner().invoke(cli, ["adapter", "github", "preview", "matterhorn"])

    assert add_result.exit_code == 0, add_result.output
    assert preview_result.exit_code != 0
    assert "truncated" in preview_result.output
    assert "Internal network / VPN" in preview_result.output
    assert "local_push" not in preview_result.output
    assert "cloud_pull" not in preview_result.output


def test_adapter_github_push_rejects_malformed_base64_content(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "github-repo-doc", "document_hash": "hash"})

    def fake_run(cmd, *, capture_output, text, env=None, check=False):
        assert cmd[:2] == ["gh", "api"]

        class Result:
            returncode = 0
            stderr = ""

        result = Result()
        if cmd[2].endswith("/git/trees/main?recursive=1"):
            result.stdout = json.dumps(
                {"tree": [{"path": "docs/broken.md", "type": "blob", "sha": "broken-sha", "size": 10}]}
            )
            return result
        if cmd[2].endswith("/contents/docs/broken.md?ref=main"):
            result.stdout = json.dumps({"encoding": "base64", "content": "!!!!", "size": 10})
            return result
        raise AssertionError(f"unexpected gh api call: {cmd}")

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    add_result = CliRunner().invoke(
        cli,
        [
            "adapter",
            "github",
            "add",
            "repo",
            "--repo-url",
            "https://github.com/example/repo",
            "--ref",
            "main",
            "--include-path",
            "docs/",
            "--include-extension",
            "md",
        ],
    )
    push_result = CliRunner().invoke(cli, ["adapter", "github", "push", "repo", "--source-id", "src-gh"])

    assert add_result.exit_code == 0, add_result.output
    assert push_result.exit_code != 0
    payload = json.loads(push_result.output)
    assert payload["counts"] == {"selected": 1, "pushed": 0, "failed": 1}
    assert "base64" in payload["failed"][0]["error"]


def test_adapter_github_push_uses_profile_source_id_and_pushes_content(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "github-repo-doc", "document_hash": "hash"})

    def fake_run(cmd, *, capture_output, text, env=None, check=False):
        assert cmd[:2] == ["gh", "api"]

        class Result:
            returncode = 0
            stderr = ""

        result = Result()
        if cmd[2].endswith("/git/trees/main?recursive=1"):
            result.stdout = json.dumps(
                {
                    "tree": [
                        {
                            "path": "Payroll Processing/README.md",
                            "type": "blob",
                            "sha": "md-sha",
                            "size": 10,
                        }
                    ]
                }
            )
            return result
        if cmd[2].endswith("/contents/Payroll%20Processing/README.md?ref=main"):
            import base64

            result.stdout = json.dumps({"content": base64.b64encode(b"# Payroll Processing\n\nBody").decode()})
            return result
        raise AssertionError(f"unexpected gh api call: {cmd}")

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    CliRunner().invoke(
        cli,
        [
            "adapter",
            "github",
            "add",
            "matterhorn",
            "--repo-url",
            "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
            "--ref",
            "main",
            "--include-path",
            "Payroll Processing/",
            "--include-extension",
            "md",
        ],
    )
    push_result = CliRunner().invoke(
        cli,
        ["adapter", "github", "push", "matterhorn", "--source-id", "src-gh", "--submitted-by", "cli-test"],
    )

    assert push_result.exit_code == 0, push_result.output
    payload = json.loads(push_result.output)
    assert payload["counts"]["pushed"] == 1
    push_calls = [call for call in FakeToolClient.calls if call[0] == "push_github_repo_document"]
    assert len(push_calls) == 1
    kwargs = push_calls[0][1]
    assert kwargs["source_id"] == "src-gh"
    assert kwargs["repo_url"] == "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture"
    assert kwargs["repo_ref"] == "main"
    assert kwargs["relative_path"] == "Payroll Processing/README.md"
    assert kwargs["blob_sha"] == "md-sha"
    assert kwargs["markdown_body"] == "# Payroll Processing\n\nBody"
    assert kwargs["submitted_by"] == "cli-test"


def test_adapter_github_push_limit_selects_first_matching_files(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "github-repo-doc", "document_hash": "hash"})

    def fake_run(cmd, *, capture_output, text, env=None, check=False):
        assert cmd[:2] == ["gh", "api"]

        class Result:
            returncode = 0
            stderr = ""

        result = Result()
        if cmd[2].endswith("/git/trees/main?recursive=1"):
            result.stdout = json.dumps(
                {
                    "tree": [
                        {"path": "docs/first.md", "type": "blob", "sha": "first-sha", "size": 10},
                        {"path": "docs/second.md", "type": "blob", "sha": "second-sha", "size": 10},
                    ]
                }
            )
            return result
        if cmd[2].endswith("/contents/docs/first.md?ref=main"):
            import base64

            result.stdout = json.dumps({"content": base64.b64encode(b"# First\n").decode()})
            return result
        raise AssertionError(f"unexpected gh api call: {cmd}")

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    add_result = CliRunner().invoke(
        cli,
        [
            "adapter",
            "github",
            "add",
            "repo",
            "--repo-url",
            "https://github.com/example/repo",
            "--ref",
            "main",
            "--include-path",
            "docs/",
            "--include-extension",
            "md",
        ],
    )
    push_result = CliRunner().invoke(
        cli,
        ["adapter", "github", "push", "repo", "--source-id", "src-gh", "--limit", "1"],
    )

    assert add_result.exit_code == 0, add_result.output
    assert push_result.exit_code == 0, push_result.output
    payload = json.loads(push_result.output)
    assert payload["counts"]["pushed"] == 1
    push_calls = [call for call in FakeToolClient.calls if call[0] == "push_github_repo_document"]
    assert [call[1]["relative_path"] for call in push_calls] == ["docs/first.md"]


def test_adapter_github_push_process_now_uses_last_successful_push(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "github-repo-doc", "document_hash": "hash"})

    def fake_run(cmd, *, capture_output, text, env=None, check=False):
        assert cmd[:2] == ["gh", "api"]

        class Result:
            returncode = 0
            stderr = ""

        result = Result()
        if cmd[2].endswith("/git/trees/main?recursive=1"):
            result.stdout = json.dumps(
                {
                    "tree": [
                        {"path": "docs/first.md", "type": "blob", "sha": "first-sha", "size": 10},
                        {"path": "docs/second.md", "type": "blob", "sha": "second-sha", "size": 10},
                    ]
                }
            )
            return result
        if cmd[2].endswith("/contents/docs/first.md?ref=main"):
            import base64

            result.stdout = json.dumps({"content": base64.b64encode(b"# First\n").decode()})
            return result
        if cmd[2].endswith("/contents/docs/second.md?ref=main"):
            result.returncode = 1
            result.stdout = ""
            result.stderr = "not found"
            return result
        raise AssertionError(f"unexpected gh api call: {cmd}")

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    add_result = CliRunner().invoke(
        cli,
        [
            "adapter",
            "github",
            "add",
            "repo",
            "--repo-url",
            "https://github.com/example/repo",
            "--ref",
            "main",
            "--include-path",
            "docs/",
            "--include-extension",
            "md",
        ],
    )
    push_result = CliRunner().invoke(
        cli,
        ["adapter", "github", "push", "repo", "--source-id", "src-gh", "--process-now"],
    )

    assert add_result.exit_code == 0, add_result.output
    assert push_result.exit_code != 0
    payload = json.loads(push_result.output)
    assert payload["counts"] == {"selected": 2, "pushed": 1, "failed": 1}
    push_calls = [call for call in FakeToolClient.calls if call[0] == "push_github_repo_document"]
    assert [call[1]["relative_path"] for call in push_calls] == ["docs/first.md"]
    assert [call[1]["process_now"] for call in push_calls] == [True]


def test_adapter_jira_watch_command_is_registered():
    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "watch", "--help"])
    assert result.exit_code == 0, result.output
    assert "--interval-seconds" in result.output
