import base64
import json
import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

import memforge.main as main
from memforge.api_target import MemForgeTarget, build_target
from memforge.local_agent import folder_picker
from memforge.main import cli


@pytest.fixture(autouse=True)
def _isolate_cli_target_configuration(monkeypatch, tmp_path: Path):
    for name in ("MEMFORGE_API_URL", "MEMFORGE_WORKSPACE_ID", "MEMFORGE_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(tmp_path / "isolated-cli.toml"))
    from memforge.auth.teams_auth import TeamsAuthenticator

    monkeypatch.setattr(TeamsAuthenticator, "_load_keychain_token_data", staticmethod(lambda: None))
    monkeypatch.setattr(TeamsAuthenticator, "_save_keychain_token_data", staticmethod(lambda _data: False))


class FakeToolClient:
    calls: list[tuple[str, dict]] = []
    response: dict = {}
    list_response: dict = {"data": []}
    create_response: dict = {"id": "src-created"}
    projection_inventory_response: dict = {"units": []}

    def __init__(
        self,
        *,
        target: MemForgeTarget,
        api_token: str | None = None,
        timeout_seconds: float = 60.0,
    ):
        self.target = target
        self.api_url = target.origin
        self.api_token = api_token
        self.workspace_id = target.workspace_id or ""
        self.timeout_seconds = timeout_seconds

    @classmethod
    def reset(cls, response: dict, *, list_response: dict | None = None, create_response: dict | None = None) -> None:
        cls.calls = []
        cls.response = response
        cls.list_response = {"data": []} if list_response is None else list_response
        cls.create_response = {"id": "src-created"} if create_response is None else create_response
        cls.projection_inventory_response = {"units": []}

    def list_sources(self):
        self.calls.append(("list_sources", {"api_url": self.api_url, "api_token": self.api_token}))
        return self.list_response

    def list_searchable_sources(self):
        self.calls.append(("list_searchable_sources", {"api_url": self.api_url, "api_token": self.api_token}))
        return self.list_response

    def get_source_projection_inventory(self, source_id: str, **filters):
        self.calls.append(
            (
                "get_source_projection_inventory",
                {
                    "source_id": source_id,
                    "workspace_id": self.workspace_id,
                    **filters,
                },
            )
        )
        return self.projection_inventory_response

    def create_source(self, **kwargs):
        self.calls.append(("create_source", {"api_url": self.api_url, "api_token": self.api_token, **kwargs}))
        return self.create_response

    def get_source_schedule(self, source_id: str):
        self.calls.append(
            (
                "get_source_schedule",
                {"api_url": self.api_url, "api_token": self.api_token, "source_id": source_id},
            )
        )
        return self.response

    def update_source_schedule(self, **kwargs):
        self.calls.append(("update_source_schedule", {"api_url": self.api_url, "api_token": self.api_token, **kwargs}))
        return self.response

    def search(self, **kwargs):
        self.calls.append(("search", {"api_url": self.api_url, "api_token": self.api_token, **kwargs}))
        return self.response

    def get_memory(self, memory_id: str):
        self.calls.append(
            ("get_memory", {"api_url": self.api_url, "api_token": self.api_token, "memory_id": memory_id})
        )
        return self.response

    def get_resource(self, **kwargs):
        self.calls.append(("get_resource", {"api_url": self.api_url, "api_token": self.api_token, **kwargs}))
        return self.response

    def start_source_sync(self, **kwargs):
        self.calls.append(
            (
                "start_source_sync",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        return self.response

    def start_source_processing(self, **kwargs):
        self.calls.append(
            (
                "start_source_processing",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        return self.response

    def push_local_markdown_document(self, **kwargs):
        self.calls.append(
            (
                "push_local_markdown_document",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        return self.response

    def push_github_repo_document(self, **kwargs):
        self.calls.append(
            (
                "push_github_repo_document",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        return self.response

    def push_jira_package(self, **kwargs):
        self.calls.append(
            (
                "push_jira_package",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        return self.response

    def upload_jira_session(self, **kwargs):
        self.calls.append(
            (
                "upload_jira_session",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        return self.response

    def push_teams_window_package(self, **kwargs):
        self.calls.append(
            (
                "push_teams_window_package",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        return self.response

    def health(self):
        self.calls.append(("health", {"api_url": self.api_url, "api_token": self.api_token}))
        return self.response

    def for_workspace(self, workspace_id: str):
        return type(self)(
            target=build_target(origin=self.api_url, workspace_id=workspace_id),
            api_token=self.api_token,
            timeout_seconds=self.timeout_seconds,
        )


def _cloud_test_client() -> FakeToolClient:
    return FakeToolClient(
        target=build_target(
            origin="https://memforge-dev.cfapps.eu12.hana.ondemand.com",
            workspace_id="ws-from-cloud",
        ),
        api_token="tok",
    )


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
        env={
            "MEMFORGE_API_URL": "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
            "MEMFORGE_WORKSPACE_ID": "ws-from-cloud",
            "MEMFORGE_API_TOKEN": "token-1",
        },
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["results"][0]["memory_id"] == "mem-docker"
    assert FakeToolClient.calls == [
        (
            "search",
            {
                "api_url": "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
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
        env={
            "MEMFORGE_API_URL": "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
            "MEMFORGE_WORKSPACE_ID": "ws-from-cloud",
            "MEMFORGE_API_TOKEN": "token-1",
        },
    )

    assert result.exit_code == 0, result.output
    assert FakeToolClient.calls == [
        (
            "search",
            {
                "api_url": "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
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
        env={"MEMFORGE_API_URL": "https://memforge-dev.cfapps.eu12.hana.ondemand.com"},
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
        env={
            "MEMFORGE_API_URL": "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
            "MEMFORGE_WORKSPACE_ID": "ws-from-cloud",
            "MEMFORGE_API_TOKEN": "token-1",
        },
    )

    assert result.exit_code == 0, result.output
    assert "Configured Sources" in result.output
    assert "src-paused" in result.output
    assert "paused" in result.output
    assert FakeToolClient.calls == [
        (
            "list_sources",
            {"api_url": "https://memforge-dev.cfapps.eu12.hana.ondemand.com", "api_token": "token-1"},
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
        env={
            "MEMFORGE_API_URL": "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
            "MEMFORGE_WORKSPACE_ID": "ws-from-cloud",
            "MEMFORGE_API_TOKEN": "token-1",
        },
    )

    assert result.exit_code == 0, result.output
    assert "Searchable Sources" in result.output
    assert "src-mounttai" in result.output
    assert FakeToolClient.calls == [
        (
            "list_searchable_sources",
            {"api_url": "https://memforge-dev.cfapps.eu12.hana.ondemand.com", "api_token": "token-1"},
        )
    ]


def test_sources_schedule_cli_updates_active_api_target(monkeypatch):
    FakeToolClient.reset({"ok": True, "source_id": "src-1"})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        ["sources", "schedule", "src-1", "--every-minutes", "60"],
        env={
            "MEMFORGE_API_URL": "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
            "MEMFORGE_WORKSPACE_ID": "ws-from-cloud",
            "MEMFORGE_API_TOKEN": "token-1",
        },
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["ok"] is True
    assert FakeToolClient.calls == [
        (
            "update_source_schedule",
            {
                "api_url": "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
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
        env={
            "MEMFORGE_API_URL": "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
            "MEMFORGE_WORKSPACE_ID": "ws-from-cloud",
            "MEMFORGE_API_TOKEN": "token-1",
        },
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["interval_minutes"] == 60
    assert FakeToolClient.calls == [
        (
            "get_source_schedule",
            {
                "api_url": "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
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
        [
            "target",
            "add",
            "sap.prod",
            "--api-url",
            "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
            "--workspace-id",
            "mount_tai",
            "--token-env",
            "SAP_TOKEN",
        ],
    )
    check_result = CliRunner().invoke(cli, ["target", "check"], env={"SAP_TOKEN": "secret-token"})

    assert add_result.exit_code == 0, add_result.output
    assert check_result.exit_code == 0, check_result.output
    assert FakeToolClient.calls == [
        ("health", {"api_url": "https://memforge-dev.cfapps.eu12.hana.ondemand.com", "api_token": "secret-token"})
    ]
    assert "edition = " not in cli_config.read_text(encoding="utf-8")
    assert 'workspace_id = "mount_tai"' in cli_config.read_text(encoding="utf-8")


def test_target_profile_does_not_mix_global_token(monkeypatch, tmp_path: Path):
    cli_config = tmp_path / "cli.toml"
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(cli_config))
    FakeToolClient.reset({"status": "ok"})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    add_result = CliRunner().invoke(
        cli,
        [
            "target",
            "add",
            "sap",
            "--api-url",
            "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
            "--workspace-id",
            "mount_tai",
            "--token-env",
            "SAP_TOKEN",
        ],
    )
    check_result = CliRunner().invoke(
        cli,
        ["target", "check"],
        env={"MEMFORGE_API_TOKEN": "wrong-token", "SAP_TOKEN": "target-token"},
    )

    assert add_result.exit_code == 0, add_result.output
    assert check_result.exit_code == 0, check_result.output
    assert FakeToolClient.calls == [
        ("health", {"api_url": "https://memforge-dev.cfapps.eu12.hana.ondemand.com", "api_token": "target-token"})
    ]


def test_env_api_url_does_not_use_active_target_token(monkeypatch, tmp_path: Path):
    cli_config = tmp_path / "cli.toml"
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(cli_config))
    FakeToolClient.reset({"status": "ok"})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    add_result = CliRunner().invoke(
        cli,
        [
            "target",
            "add",
            "sap",
            "--api-url",
            "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
            "--workspace-id",
            "mount_tai",
            "--token-env",
            "SAP_TOKEN",
        ],
    )
    check_result = CliRunner().invoke(
        cli,
        ["target", "check"],
        env={
            "MEMFORGE_API_URL": "https://override.example.test",
            "SAP_TOKEN": "target-token",
        },
    )

    assert add_result.exit_code == 0, add_result.output
    assert check_result.exit_code == 0, check_result.output
    assert FakeToolClient.calls == [("health", {"api_url": "https://override.example.test", "api_token": None})]


def test_target_add_rejects_invalid_tagged_union_before_writing(monkeypatch, tmp_path: Path):
    cli_config = tmp_path / "cli.toml"
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(cli_config))

    result = CliRunner().invoke(
        cli,
        ["target", "add", "invalid", "--api-url", "https://memforge-dev.cfapps.eu12.hana.ondemand.com"],
    )

    assert result.exit_code != 0
    assert "cloud_workspace_required" in result.output
    assert not cli_config.exists()


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
    stub = _StubClient(
        get_jira_session={
            "provider": "jira",
            "origin": "https://jira.tools.sap",
            "status": "active",
            "principal_name": "Rose H",
            "browser": "chrome",
        }
    )
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

    async def fake_capture(base_url, *, browser=None, interactive=False):
        assert interactive is True
        captured["base_url"] = base_url
        captured["browser"] = browser
        return jira_capture.JiraCaptureResult(
            origin=base_url,
            cookie_header="SESSION=x",
            browser=browser,
            principal={"accountId": "u1"},
        )

    stub = _StubClient(upload_jira_session={"provider": "jira", "origin": "https://jira.tools.sap", "status": "active"})
    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    monkeypatch.setattr(main, "_tool_client", lambda ctx: stub)

    result = CliRunner().invoke(
        cli,
        ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.tools.sap", "--browser", "chrome"],
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

    async def fake_capture(base_url, *, browser=None, interactive=False):
        assert interactive is True
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

    async def fake_capture(base_url, *, browser=None, interactive=False):
        assert interactive is True
        raise ValueError("Unsupported browser for Jira session extraction: netscape")

    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    monkeypatch.setattr(main, "_tool_client", lambda ctx: _StubClient())

    result = CliRunner().invoke(
        cli,
        ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.tools.sap", "--browser", "netscape"],
    )
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "auth_failed"
    assert "Unsupported browser" in payload["detail"]


def test_adapter_jira_refresh_principal_change_returns_json_error(monkeypatch):
    from memforge.auth import jira_capture

    async def fake_capture(base_url, *, browser=None, interactive=False):
        assert interactive is True
        return jira_capture.JiraCaptureResult(
            origin=base_url,
            cookie_header="SESSION=x",
            browser=None,
            principal={"accountId": "u1"},
        )

    body = json.dumps(
        {
            "detail": {
                "message": "changed",
                "origin": "https://jira.tools.sap",
                "old_principal_id": "old-user",
                "new_principal_id": "new-user",
            }
        }
    )
    stub = _StubClient(
        upload_jira_session={
            "error": "MemForge API request failed",
            "status_code": 409,
            "detail": body,
        }
    )
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
    forgotten_locally = []

    monkeypatch.setattr(
        "memforge.auth.jira_browser_session.JiraBrowserSession.forget",
        lambda self, *, origin: forgotten_locally.append(origin),
    )
    stub = _StubClient(
        list_jira_origins={"origins": [{"origin": "https://jira.tools.sap", "status": "active"}]},
        forget_jira_session={"ok": True, "origin": "https://jira.tools.sap", "forgotten": True},
    )
    monkeypatch.setattr(main, "_tool_client", lambda ctx: stub)
    list_result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "list"])
    assert list_result.exit_code == 0, list_result.output
    assert json.loads(list_result.output)["origins"][0]["origin"] == "https://jira.tools.sap"
    forget_result = CliRunner().invoke(
        cli, ["adapter", "auth", "jira", "forget", "--base-url", "https://jira.tools.sap"]
    )
    assert forget_result.exit_code == 0, forget_result.output
    assert json.loads(forget_result.output)["forgotten"] is True
    assert forgotten_locally == ["https://jira.tools.sap"]


def test_legacy_auth_jira_is_removed():
    help_result = CliRunner().invoke(cli, ["auth", "--help"])
    result = CliRunner().invoke(cli, ["auth", "jira", "--base-url", "https://jira.tools.sap"])

    assert help_result.exit_code == 0, help_result.output
    assert "jira" not in help_result.output
    assert result.exit_code != 0
    assert "No such command 'jira'" in result.output


@pytest.mark.parametrize(
    ("keychain_available", "expected", "unexpected"),
    [
        (True, "Teams session saved to the OS keychain", "could not be saved"),
        (False, "could not be saved", "local compatibility cache"),
    ],
)
def test_auth_teams_reports_actual_keychain_persistence(
    monkeypatch,
    keychain_available: bool,
    expected: str,
    unexpected: str,
):
    from memforge.auth.teams_auth import TeamsAuthenticator

    def fake_authenticate(self, region="emea", **_options):
        self.keychain_session_available = keychain_available
        return {"tokens": {}}

    monkeypatch.setattr(TeamsAuthenticator, "authenticate", fake_authenticate)

    result = CliRunner().invoke(cli, ["auth", "teams"])

    assert result.exit_code == 0, result.output
    assert expected in result.output
    assert unexpected not in result.output


def test_adapter_list_includes_markdown_kb_capability():
    result = CliRunner().invoke(cli, ["adapter", "list"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {"type": "kb", "kind": "markdown"} in payload["data"]


def _fake_github_remote_run(cmd, *args, **kwargs):
    assert cmd[:2] == ["gh", "api"]
    assert kwargs["env"]["GH_HOST"] == "github.wdf.sap.corp"
    endpoint = cmd[2]
    if "/git/trees/" in endpoint:
        payload = {
            "tree": [
                {"path": "Payroll Processing V2/README.md", "type": "blob", "sha": "readme", "size": 30},
                {"path": "Payroll Processing V2/Überblick.md", "type": "blob", "sha": "overview", "size": 20},
                {"path": "Payroll Processing V2/images/Flow.puml", "type": "blob", "sha": "flow", "size": 10},
                {"path": "Payroll Processing V2/images/ignored.png", "type": "blob", "sha": "png", "size": 5},
                {"path": "Flexible Payroll/README.md", "type": "blob", "sha": "flex", "size": 25},
            ]
        }
    elif "/contents/" in endpoint:
        raw = b"# Payroll Processing V2\n\nBody"
        payload = {"content": base64.b64encode(raw).decode(), "encoding": "base64", "size": len(raw)}
    else:
        raise AssertionError(f"unexpected gh endpoint: {endpoint}")
    return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")


def test_local_agent_cloud_github_preview_tree_ignores_saved_scope(monkeypatch):
    monkeypatch.setattr(main.subprocess, "run", _fake_github_remote_run)

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-preview",
            "operation": "github_repo_preview_tree",
            "source_id": "src-gh",
            "payload": {
                "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                "ref": "main",
                "include_paths": ["Payroll Processing V2"],
                "exclude_paths": ["Payroll Processing V2/Überblick.md"],
                "include_extensions": ["md"],
                "limit": 10,
            },
        },
        _cloud_test_client(),
    )

    assert payload["operation"] == "github_repo_preview_tree"
    assert payload["source_id"] == "src-gh"
    assert payload["counts"] == {"included": 3, "ignored": 2}
    assert [item["relative_path"] for item in payload["items"]] == [
        "Payroll Processing V2/README.md",
        "Payroll Processing V2/Überblick.md",
        "Flexible Payroll/README.md",
    ]


def test_local_agent_cloud_github_sync_pushes_remote_gh_scope(monkeypatch):
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "github-repo-doc", "document_hash": "hash"})
    monkeypatch.setattr(main.subprocess, "run", _fake_github_remote_run)

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-sync",
            "attempt_count": 1,
            "workspace_id": "ws-from-job-payload",
            "operation": "github_repo_sync",
            "source_id": "src-from-cloud",
            "payload": {
                "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                "ref": "main",
                "include_paths": ["Payroll Processing V2"],
                "exclude_paths": ["Payroll Processing V2/Überblick.md"],
                "include_extensions": ["md"],
                "limit": 1,
            },
        },
        _cloud_test_client(),
    )

    assert payload["operation"] == "github_repo_sync"
    assert payload["source_id"] == "src-from-cloud"
    assert payload["counts"] == {"selected": 1, "pushed": 1, "failed": 0}
    push_calls = [call for call in FakeToolClient.calls if call[0] == "push_github_repo_document"]
    assert len(push_calls) == 1
    kwargs = push_calls[0][1]
    assert kwargs["workspace_id"] == "ws-from-job-payload"
    assert kwargs["source_id"] == "src-from-cloud"
    assert kwargs["repo_url"] == "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture"
    assert kwargs["relative_path"] == "Payroll Processing V2/README.md"
    assert "process_now" not in kwargs
    assert kwargs["submitted_by"] == "memforge-local-agent"
    assert kwargs["sync_snapshot_id"] == "laj-sync:attempt:1"
    process_calls = [call for call in FakeToolClient.calls if call[0] == "start_source_processing"]
    assert len(process_calls) == 1
    assert process_calls[0][1]["source_id"] == "src-from-cloud"
    assert process_calls[0][1]["sync_snapshot_id"] == "laj-sync:attempt:1"


def test_local_agent_cloud_github_sync_requires_job_workspace(monkeypatch):
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "github-repo-doc", "document_hash": "hash"})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-sync",
            "operation": "github_repo_sync",
            "source_id": "src-from-cloud",
            "payload": {
                "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                "ref": "main",
                "include_paths": ["Payroll Processing V2"],
                "include_extensions": ["md"],
                "limit": 1,
            },
        },
        _cloud_test_client(),
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
        _cloud_test_client(),
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
        _cloud_test_client(),
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
        _cloud_test_client(),
    )

    assert payload == {
        "operation": "local_markdown_pick_root",
        "cancelled": True,
    }


def test_native_folder_picker_times_out_as_cancellation(monkeypatch):
    calls: list[dict] = []

    def fake_system() -> str:
        return "Darwin"

    def fake_run(args, **kwargs):
        calls.append({"args": args, **kwargs})
        raise subprocess.TimeoutExpired(cmd=args, timeout=kwargs["timeout"])

    monkeypatch.setattr(folder_picker.platform, "system", fake_system)
    monkeypatch.setattr(folder_picker.subprocess, "run", fake_run)

    with pytest.raises(folder_picker.FolderPickerCancelled, match="timed out"):
        folder_picker.pick_folder(timeout_seconds=1.5)

    assert calls[0]["timeout"] == 1.5


def test_native_folder_picker_non_macos_error_points_to_manual_path(monkeypatch):
    monkeypatch.setattr(folder_picker.platform, "system", lambda: "Linux")

    with pytest.raises(folder_picker.FolderPickerUnavailable, match="type the folder path instead"):
        folder_picker.pick_folder()


def test_local_agent_cloud_local_markdown_sync_pushes_workspace_source(monkeypatch, tmp_path: Path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "Decision.md").write_text("# Decision\n\nUse the daemon.", encoding="utf-8")
    (root / "Runbook.md").write_text("# Runbook\n\nRestart safely.", encoding="utf-8")
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "local-doc", "document_hash": "hash"})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-local-sync",
            "attempt_count": 2,
            "workspace_id": "ws-from-cloud",
            "operation": "local_markdown_sync",
            "source_id": "src-local",
            "payload": {
                "root": str(root),
                "vault_id": "engineering",
                "include": ["*.md", "**/*.md"],
                "exclude": [],
                "limit": 1,
            },
        },
        _cloud_test_client(),
    )

    assert payload["operation"] == "local_markdown_sync"
    assert payload["source_id"] == "src-local"
    assert payload["counts"]["pushed"] == 2
    push_calls = [call for call in FakeToolClient.calls if call[0] == "push_local_markdown_document"]
    assert len(push_calls) == 2
    kwargs = push_calls[0][1]
    assert kwargs["workspace_id"] == "ws-from-cloud"
    assert kwargs["source_id"] == "src-local"
    assert kwargs["vault_id"] == "engineering"
    assert kwargs["relative_path"] == "Decision.md"
    assert kwargs["sync_snapshot_id"] == "laj-local-sync:attempt:2"
    assert "process_now" not in kwargs
    process_calls = [call for call in FakeToolClient.calls if call[0] == "start_source_processing"]
    assert len(process_calls) == 1
    assert process_calls[0][1]["source_id"] == "src-local"
    assert process_calls[0][1]["sync_snapshot_id"] == "laj-local-sync:attempt:2"


def test_local_agent_cloud_local_markdown_sync_publishes_empty_snapshot(monkeypatch, tmp_path: Path):
    root = tmp_path / "empty-notes"
    root.mkdir()
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"ok": True})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-empty-snapshot",
            "attempt_count": 1,
            "workspace_id": "ws-from-cloud",
            "operation": "local_markdown_sync",
            "source_id": "src-local",
            "payload": {
                "root": str(root),
                "vault_id": "engineering",
                "include": ["*.md", "**/*.md"],
                "exclude": [],
            },
        },
        _cloud_test_client(),
    )

    assert payload["counts"]["pushed"] == 0
    process_calls = [call[1] for call in FakeToolClient.calls if call[0] == "start_source_processing"]
    assert process_calls == [
        {
            "api_url": "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
            "api_token": "tok",
            "workspace_id": "ws-from-cloud",
            "source_id": "src-local",
            "force_full_sync": False,
            "sync_snapshot_id": "laj-empty-snapshot:attempt:1",
            "local_agent_job_id": "laj-empty-snapshot",
            "local_agent_attempt_count": 1,
        }
    ]


def test_local_agent_cloud_jira_sync_uses_gene_and_pushes_packages(monkeypatch):
    from datetime import datetime, timezone

    from memforge.auth import jira_capture
    from memforge.models import ContentItem, RawContent

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
            return RawContent(
                item=item,
                body=json.dumps({"key": "PAY-1", "fields": {"summary": "Create daemon source support"}}).encode(),
                content_type="application/json",
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
            "attempt_count": 3,
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
            },
        },
        _cloud_test_client(),
        browser="chrome",
    )

    assert payload["operation"] == "jira_sync"
    assert payload["counts"] == {"selected": 1, "pushed": 1, "failed": 0}
    upload_calls = [call for call in FakeToolClient.calls if call[0] == "upload_jira_session"]
    assert upload_calls == [
        (
            "upload_jira_session",
            {
                "api_url": "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
                "api_token": "tok",
                "workspace_id": "ws-from-cloud",
                "base_url": "https://jira.example.test",
                "cookie_header": "JSESSIONID=local",
                "browser": "chrome",
            },
        )
    ]
    push_calls = [call for call in FakeToolClient.calls if call[0] == "push_jira_package"]
    assert len(push_calls) == 1
    kwargs = push_calls[0][1]
    assert kwargs["workspace_id"] == "ws-from-cloud"
    assert kwargs["source_id"] == "src-jira"
    assert kwargs["base_url"] == "https://jira.example.test"
    assert kwargs["issue_key"] == "PAY-1"
    assert kwargs["raw_payload"]["fields"]["summary"] == "Create daemon source support"
    assert kwargs["sync_snapshot_id"] == "laj-jira-sync:attempt:3"
    assert "markdown_body" not in kwargs
    assert "process_now" not in kwargs
    process_calls = [call for call in FakeToolClient.calls if call[0] == "start_source_processing"]
    assert len(process_calls) == 1
    assert process_calls[0][1]["source_id"] == "src-jira"
    assert process_calls[0][1]["sync_snapshot_id"] == "laj-jira-sync:attempt:3"


def test_local_agent_cloud_jira_sync_rejects_pat_payload():
    FakeToolClient.reset({"doc_id": "jira-doc", "document_hash": "hash"})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-jira-sync",
            "attempt_count": 1,
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
        _cloud_test_client(),
    )

    assert payload["operation"] == "jira_sync"
    assert payload["error_type"] == "ClickException"
    assert "browser-session" in payload["error"]
    assert not [call for call in FakeToolClient.calls if call[0] == "push_jira_package"]


def test_local_agent_cloud_jira_sync_stops_on_principal_change(monkeypatch):
    from memforge.auth import jira_capture

    async def fake_capture(base_url, *, browser=None, tls_config=None):
        return jira_capture.JiraCaptureResult(
            origin=base_url,
            cookie_header="JSESSIONID=different-user",
            browser=browser,
            principal={"name": "different-user"},
        )

    conflict = {
        "error": "MemForge API request failed",
        "status_code": 409,
        "detail": json.dumps(
            {
                "detail": {
                    "origin": "https://jira.example.test",
                    "old_principal_id": "old-user",
                    "new_principal_id": "different-user",
                }
            }
        ),
    }
    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset(conflict)

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-jira-principal-change",
            "attempt_count": 1,
            "workspace_id": "ws-from-cloud",
            "operation": "jira_sync",
            "source_id": "src-jira",
            "payload": {
                "base_url": "https://jira.example.test",
                "auth_mode": "browser_cookie",
                "projects": ["PAY"],
            },
        },
        _cloud_test_client(),
        browser="chrome",
    )

    assert payload == {
        "operation": "jira_sync",
        "source_id": "src-jira",
        "error": "principal_changed",
        "origin": "https://jira.example.test",
        "old_principal_id": "old-user",
        "new_principal_id": "different-user",
        "error_type": "JiraPrincipalChangedError",
        "retryable": False,
    }
    assert not [call for call in FakeToolClient.calls if call[0] == "push_jira_package"]


def test_local_agent_cloud_jira_sync_does_not_publish_partial_snapshot(monkeypatch):
    from datetime import datetime, timezone

    from memforge.auth import jira_capture
    from memforge.models import ContentItem, RawContent

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
            key = item.extra["issue_key"]
            return RawContent(
                item=item,
                body=json.dumps({"key": key, "fields": {"summary": f"{key} title"}}).encode(),
                content_type="application/json",
            )

    async def fake_capture(base_url, *, browser=None, tls_config=None):
        return jira_capture.JiraCaptureResult(
            origin=base_url,
            cookie_header="JSESSIONID=local",
            browser=browser,
            principal={"name": "tester"},
        )

    def fake_push_jira_package(self, **kwargs):
        self.calls.append(
            (
                "push_jira_package",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        if kwargs["issue_key"] == "PAY-2":
            return {"error": "push failed", "status_code": 500}
        return {"doc_id": f"jira-{kwargs['issue_key'].lower()}", "document_hash": "hash"}

    import memforge.genes.jira_gene as jira_gene

    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    monkeypatch.setattr(jira_gene, "JiraGene", FakeJiraGene)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    monkeypatch.setattr(FakeToolClient, "push_jira_package", fake_push_jira_package)
    FakeToolClient.reset({})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-jira-sync",
            "attempt_count": 1,
            "workspace_id": "ws-from-cloud",
            "operation": "jira_sync",
            "source_id": "src-jira",
            "payload": {
                "base_url": "https://jira.example.test",
                "auth_mode": "browser_cookie",
                "sync_mode": "local_agent",
                "local_agent_documents_dir": "/srv/memforge/inbox/src-jira",
                "projects": ["PAY"],
            },
        },
        _cloud_test_client(),
        browser="chrome",
    )

    assert payload["counts"] == {"selected": 2, "pushed": 1, "failed": 1}
    assert payload["retryable"] is True
    push_calls = [call[1] for call in FakeToolClient.calls if call[0] == "push_jira_package"]
    assert [call["issue_key"] for call in push_calls] == ["PAY-1", "PAY-2"]
    assert all("process_now" not in call for call in push_calls)
    process_calls = [call[1] for call in FakeToolClient.calls if call[0] == "start_source_processing"]
    assert process_calls == []


def test_local_agent_cloud_teams_auth_captures_session(monkeypatch):
    captured_regions: list[str] = []
    wait_options: list[tuple[int, float]] = []

    class FakeTeamsAuthenticator:
        def authenticate(
            self,
            *,
            region: str = "emea",
            wait_seconds: int = 0,
            poll_interval_seconds: float = 2.0,
            rejected_token_hashes: set[str] | None = None,
        ) -> dict[str, object]:
            captured_regions.append(region)
            wait_options.append((wait_seconds, poll_interval_seconds))
            return {
                "region": region,
                "tokens": {
                    "https://ic3.teams.office.com": {
                        "expiresAt": 4_102_444_800,
                        "scopes": "Chat.Read",
                    }
                },
            }

    import memforge.auth.teams_auth as teams_auth

    monkeypatch.setattr(teams_auth, "TeamsAuthenticator", FakeTeamsAuthenticator)
    FakeToolClient.reset({})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-teams-auth",
            "operation": "teams_auth",
            "source_type": "teams",
            "payload": {"region": "amer"},
        },
        _cloud_test_client(),
    )

    assert payload == {
        "operation": "teams_auth",
        "authenticated": True,
        "region": "amer",
        "token_count": 1,
    }
    assert captured_regions == ["amer"]
    assert wait_options == [(90, 2.0)]
    assert FakeToolClient.calls == []


def test_local_agent_cloud_teams_auth_check_uses_daemon_token_status(monkeypatch):
    import memforge.local_agent.teams_browse as teams_browse

    monkeypatch.setattr(
        teams_browse,
        "teams_auth_status",
        lambda: {"authenticated": True, "expires_in_minutes": 42, "error": None},
    )
    FakeToolClient.reset({})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-teams-auth-check",
            "operation": "teams_auth_check",
            "source_type": "teams",
            "payload": {"region": "apac"},
        },
        _cloud_test_client(),
    )

    assert payload == {
        "operation": "teams_auth_check",
        "region": "apac",
        "authenticated": True,
        "expires_in_minutes": 42,
        "error": None,
    }
    assert FakeToolClient.calls == []


def test_local_agent_cloud_teams_browse_returns_picker_data(monkeypatch):
    import memforge.local_agent.teams_browse as teams_browse

    async def fake_browse(*, region: str = "emea"):
        assert region == "amer"
        return {
            "favorites": [{"id": "19:fav@thread.tacv2", "topic": "Engineering / General"}],
            "teams": [
                {
                    "id": "team-1",
                    "displayName": "Engineering",
                    "channels": [{"id": "19:channel@thread.tacv2", "displayName": "Architecture"}],
                }
            ],
            "group_chats": [{"id": "19:group@thread.v2", "topic": "Planning", "lastActivity": None}],
            "individual_chats": [{"id": "19:dm@thread.v2", "topic": "Ada", "lastActivity": None}],
        }

    monkeypatch.setattr(teams_browse, "browse_teams_conversations", fake_browse)
    FakeToolClient.reset({})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-teams-browse",
            "operation": "teams_browse",
            "source_type": "teams",
            "payload": {"region": "amer"},
        },
        _cloud_test_client(),
    )

    assert payload["operation"] == "teams_browse"
    assert payload["region"] == "amer"
    assert payload["teams"][0]["channels"][0]["id"] == "19:channel@thread.tacv2"
    assert payload["group_chats"][0]["topic"] == "Planning"
    assert FakeToolClient.calls == []


def test_local_agent_cloud_teams_browse_reauths_after_stale_cached_session(monkeypatch):
    import memforge.local_agent.teams_browse as teams_browse
    import memforge.auth.teams_auth as teams_auth
    from memforge.genes.teams_gene import AuthenticationError

    browse_calls: list[int] = []
    auth_calls: list[tuple[int, float]] = []

    async def fake_browse(*, region: str = "emea"):
        assert region == "emea"
        browse_calls.append(1)
        if len(browse_calls) == 1:
            raise AuthenticationError("Teams session expired. Connect Teams from the source wizard.")
        return {
            "favorites": [],
            "teams": [],
            "group_chats": [{"id": "19:group@thread.v2", "topic": "Planning", "lastActivity": None}],
            "individual_chats": [],
        }

    class FakeTeamsAuthenticator:
        def authenticate(
            self,
            *,
            region: str = "emea",
            wait_seconds: int = 0,
            poll_interval_seconds: float = 2.0,
            rejected_token_hashes: set[str] | None = None,
        ) -> dict[str, object]:
            auth_calls.append((wait_seconds, poll_interval_seconds))
            return {
                "region": region,
                "tokens": {
                    "https://ic3.teams.office.com": {
                        "token": "fresh",
                        "expiresAt": 4_102_444_800,
                    }
                },
            }

    monkeypatch.setattr(teams_browse, "browse_teams_conversations", fake_browse)
    monkeypatch.setattr(teams_auth, "TeamsAuthenticator", FakeTeamsAuthenticator)
    FakeToolClient.reset({})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-teams-browse",
            "operation": "teams_browse",
            "source_type": "teams",
            "payload": {"region": "emea"},
        },
        _cloud_test_client(),
    )

    assert payload["operation"] == "teams_browse"
    assert payload["group_chats"][0]["topic"] == "Planning"
    assert len(browse_calls) == 2
    assert auth_calls == [(90, 2.0)]


def test_local_agent_cloud_teams_sync_pushes_window_packages(monkeypatch, tmp_path: Path):
    from memforge.local_agent.teams_audit import validate_teams_audit_run

    async def fake_collect(job, *, source_id, limit, report_progress=None):
        assert source_id == "src-teams"
        assert limit == 2
        assert job["payload"]["conversation_gap_minutes"] == 60
        return [
            {
                "conversation_id": "19:conversation@thread.tacv2",
                "root_message_id": "1783500000000",
                "window_id": "teams-thread:src-teams:19:conversation@thread.tacv2:1783500000000",
                "window_type": "thread",
                "revision_hash": "sha256:revision-1",
                "title": "Thread window",
                "source_url": "https://teams.microsoft.com/l/message/19:conversation@thread.tacv2/1783500000001",
                "last_modified": "2026-07-08T09:24:57.5870000Z",
                "raw_payload": {
                    "conversation_type": "channel",
                    "messages": [{"id": "1783500000000", "content": "Thread window", "from": "Alice"}],
                },
                "raw_hash": "sha256:raw-1",
                "message_count": 2,
            },
            {
                "conversation_id": "19:conversation@thread.tacv2",
                "root_message_id": "1783503600000",
                "window_id": "teams-block:src-teams:19:conversation@thread.tacv2:1783503600000",
                "window_type": "time_block",
                "revision_hash": "sha256:revision-2",
                "title": "Group chat block",
                "source_url": "https://teams.microsoft.com/l/message/19:conversation@thread.tacv2/1783503600000",
                "last_modified": "2026-07-08T10:24:57.5870000Z",
                "raw_payload": {
                    "conversation_type": "group_chat",
                    "messages": [{"id": "1783503600000", "content": "Group chat block", "from": "Ada"}],
                },
                "raw_hash": "sha256:raw-2",
                "message_count": 1,
            },
        ]

    monkeypatch.setattr(main, "_collect_teams_documents_from_cloud_job", fake_collect, raising=False)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)

    def fake_push_teams_window_package(self, **kwargs):
        self.calls.append(
            (
                "push_teams_window_package",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        return {
            "doc_id": "teams-doc",
            "document_hash": "hash",
        }

    monkeypatch.setattr(FakeToolClient, "push_teams_window_package", fake_push_teams_window_package)
    FakeToolClient.reset({})
    audit_log_path = tmp_path / "teams-audit.jsonl"
    ledger_state_path = tmp_path / "teams-ledger.json"

    progress: list[dict] = []
    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-teams-sync",
            "workspace_id": "ws-from-cloud",
            "operation": "teams_sync",
            "attempt_count": 1,
            "source_id": "src-teams",
            "payload": {
                "conversation_ids": ["19:conversation@thread.tacv2"],
                "conversation_gap_minutes": 60,
                "limit": 2,
                "audit_log_path": str(audit_log_path),
                "ledger_state_path": str(ledger_state_path),
            },
        },
        _cloud_test_client(),
        report_progress=progress.append,
    )

    assert payload["operation"] == "teams_sync"
    assert payload["counts"] == {
        "selected": 2,
        "pushed": 2,
        "failed": 0,
        "skipped_existing": 0,
        "polls": 0,
    }
    assert payload["messages"] == 3
    assert payload["conversations"] == 1
    assert payload["sync_started"] is True
    assert [item["phase"] for item in progress] == [
        "connecting",
        "uploading",
        "uploading",
        "uploading",
    ]
    assert progress[1] == {
        "schema_version": 1,
        "phase": "uploading",
        "progress": {"completed": 0, "total": 3, "unit": "message"},
        "source_time_range": {
            "start": "2026-07-08T09:24:57.5870000Z",
            "end": "2026-07-08T09:24:57.5870000Z",
        },
        "counts": {"failed": 0},
    }
    push_calls = [call for call in FakeToolClient.calls if call[0] == "push_teams_window_package"]
    assert len(push_calls) == 2
    first = push_calls[0][1]
    assert first["workspace_id"] == "ws-from-cloud"
    assert first["source_id"] == "src-teams"
    assert first["conversation_id"] == "19:conversation@thread.tacv2"
    assert first["root_message_id"] == "1783500000000"
    assert first["window_id"] == "teams-thread:src-teams:19:conversation@thread.tacv2:1783500000000"
    assert first["window_type"] == "thread"
    assert first["revision_hash"] == "sha256:revision-1"
    assert first["raw_payload"]["messages"][0]["content"] == "Thread window"
    assert "markdown_body" not in first
    assert "last_modified" not in first
    assert "process_now" not in first
    assert "process_now" not in push_calls[1][1]
    process_calls = [call for call in FakeToolClient.calls if call[0] == "start_source_processing"]
    assert len(process_calls) == 1
    assert process_calls[0][1]["source_id"] == "src-teams"

    audit_rows = [json.loads(line) for line in audit_log_path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in audit_rows] == [
        "teams_window_projection",
        "teams_memory_patch",
        "teams_window_projection",
        "teams_memory_patch",
        "teams_sync_run",
    ]
    assert validate_teams_audit_run(audit_rows) == []
    assert all("raw_conversation_id" not in row for row in audit_rows)
    assert all("raw_root_message_id" not in row for row in audit_rows)
    assert {row["patch_status"] for row in audit_rows if row["event"] == "teams_memory_patch"} == {"pushed"}
    assert audit_rows[-1]["selected_windows"] == 2
    assert audit_rows[-1]["pushed_windows"] == 2
    assert audit_rows[-1]["sync_started"] is True
    assert audit_rows[-1]["sync_error"] is None


def test_local_agent_cloud_teams_sync_reauths_after_stale_session(monkeypatch, tmp_path: Path):
    import memforge.auth.teams_auth as teams_auth
    from memforge.genes.teams_gene import AuthenticationError

    collect_calls: list[int] = []
    auth_calls: list[tuple[int, float]] = []

    async def fake_collect(job, *, source_id, limit, report_progress=None):
        assert source_id == "src-teams"
        collect_calls.append(1)
        if len(collect_calls) == 1:
            raise AuthenticationError("Teams session expired. Connect Teams from the source wizard.")
        return {
            "documents": [
                {
                    "conversation_id": "19:conversation@thread.tacv2",
                    "root_message_id": "1783500000000",
                    "window_id": "teams-thread:v1:opaque-window",
                    "window_type": "thread",
                    "revision_hash": "sha256:revision-1",
                    "title": "Thread window",
                    "source_url": "https://teams.microsoft.com/l/message/19:conversation@thread.tacv2/1783500000001",
                    "last_modified": "2026-07-08T09:24:57.5870000Z",
                    "raw_payload": {
                        "conversation_type": "channel",
                        "messages": [{"id": "1783500000000", "content": "Thread window", "from": "Alice"}],
                    },
                    "raw_hash": "sha256:raw-1",
                    "message_count": 1,
                }
            ],
            "poll_audits": [
                {
                    "raw_conversation_id": "19:conversation@thread.tacv2",
                    "pagination_complete": True,
                    "access_probe_status": "ok",
                    "stop_reason": "no_backward_link",
                }
            ],
        }

    class FakeTeamsAuthenticator:
        def authenticate(
            self,
            *,
            region: str = "emea",
            wait_seconds: int = 0,
            poll_interval_seconds: float = 2.0,
            rejected_token_hashes: set[str] | None = None,
        ) -> dict[str, object]:
            auth_calls.append((wait_seconds, poll_interval_seconds))
            return {
                "region": region,
                "tokens": {
                    "https://ic3.teams.office.com": {
                        "token": "fresh",
                        "expiresAt": 4_102_444_800,
                    }
                },
            }

    monkeypatch.setattr(main, "_collect_teams_documents_from_cloud_job", fake_collect, raising=False)
    monkeypatch.setattr(teams_auth, "TeamsAuthenticator", FakeTeamsAuthenticator)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "teams-doc", "document_hash": "hash"})

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-teams-sync",
            "workspace_id": "ws-from-cloud",
            "operation": "teams_sync",
            "attempt_count": 1,
            "source_id": "src-teams",
            "payload": {
                "conversation_ids": "19:conversation@thread.tacv2",
                "audit_log_path": str(tmp_path / "teams-audit.jsonl"),
                "ledger_state_path": str(tmp_path / "teams-ledger.json"),
            },
        },
        _cloud_test_client(),
    )

    assert payload["operation"] == "teams_sync"
    assert payload["counts"]["pushed"] == 1
    assert len(collect_calls) == 2
    assert auth_calls == [(90, 2.0)]


def test_local_agent_cloud_teams_sync_reports_push_failure_without_generic_source_sync(monkeypatch, tmp_path: Path):
    async def fake_collect(job, *, source_id, limit, report_progress=None):
        return [
            {
                "conversation_id": "19:conversation@thread.tacv2",
                "root_message_id": "1783500000000",
                "window_id": "teams-thread:v1:opaque-window",
                "window_type": "thread",
                "revision_hash": "sha256:revision-1",
                "title": "Thread window",
                "source_url": "https://teams.microsoft.com/l/message/19:conversation@thread.tacv2/1783500000001",
                "last_modified": "2026-07-08T09:24:57.5870000Z",
                "raw_payload": {
                    "conversation_type": "channel",
                    "messages": [{"id": "1783500000000", "content": "Thread window", "from": "Alice"}],
                },
                "raw_hash": "sha256:raw-1",
                "message_count": 1,
            }
        ]

    def failing_push_teams_window_package(self, **kwargs):
        self.calls.append(
            (
                "push_teams_window_package",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        return {"error": "MemForge API unavailable", "detail": "Server disconnected without sending a response."}

    monkeypatch.setattr(main, "_collect_teams_documents_from_cloud_job", fake_collect, raising=False)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    monkeypatch.setattr(FakeToolClient, "push_teams_window_package", failing_push_teams_window_package)
    FakeToolClient.reset({})
    audit_log_path = tmp_path / "teams-audit.jsonl"
    ledger_state_path = tmp_path / "teams-ledger.json"

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-teams-sync",
            "workspace_id": "ws-from-cloud",
            "operation": "teams_sync",
            "attempt_count": 1,
            "source_id": "src-teams",
            "payload": {
                "conversation_ids": ["19:conversation@thread.tacv2"],
                "audit_log_path": str(audit_log_path),
                "ledger_state_path": str(ledger_state_path),
            },
        },
        _cloud_test_client(),
    )

    assert payload["counts"] == {"selected": 1, "pushed": 0, "failed": 1, "skipped_existing": 0, "polls": 0}
    assert payload["sync_started"] is False
    assert payload["error"] == "one or more Teams windows failed to push"
    assert payload["retryable"] is True
    assert not [call for call in FakeToolClient.calls if call[0] == "start_source_processing"]

    audit_rows = [json.loads(line) for line in audit_log_path.read_text(encoding="utf-8").splitlines()]
    assert audit_rows[-1]["event"] == "teams_sync_run"
    assert audit_rows[-1]["status"] == "completed_with_error"
    assert audit_rows[-1]["sync_started"] is False
    assert audit_rows[-1]["failed_windows"] == 1


def test_local_agent_cloud_teams_sync_stops_after_lease_is_rejected(monkeypatch, tmp_path: Path):
    async def fake_collect(job, *, source_id, limit, report_progress=None):
        return [
            {
                "conversation_id": "19:conversation@thread.tacv2",
                "root_message_id": f"178350000000{index}",
                "window_id": f"teams-thread:v1:window-{index}",
                "window_type": "thread",
                "revision_hash": f"sha256:revision-{index}",
                "title": f"Thread window {index}",
                "source_url": f"https://teams.example.test/window-{index}",
                "last_modified": "2026-07-08T09:24:57Z",
                "raw_payload": {"messages": [{"id": str(index), "content": "Window"}]},
                "raw_hash": f"sha256:raw-{index}",
                "message_count": 1,
            }
            for index in range(2)
        ]

    def rejected_push(self, **kwargs):
        self.calls.append(("push_teams_window_package", kwargs))
        return {
            "error": "MemForge API request failed",
            "status_code": 409,
            "detail": '{"detail":"local_agent_lease_not_current"}',
        }

    monkeypatch.setattr(main, "_collect_teams_documents_from_cloud_job", fake_collect, raising=False)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    monkeypatch.setattr(FakeToolClient, "push_teams_window_package", rejected_push)
    FakeToolClient.reset({})

    result = main._run_cloud_teams_sync_job(
        {
            "job_id": "laj-teams-stale-lease",
            "workspace_id": "ws-from-cloud",
            "operation": "teams_sync",
            "attempt_count": 1,
            "source_id": "src-teams",
            "payload": {
                "conversation_ids": ["19:conversation@thread.tacv2"],
                "audit_log_path": str(tmp_path / "teams-audit.jsonl"),
            },
        },
        _cloud_test_client(),
    )

    pushes = [call for call in FakeToolClient.calls if call[0] == "push_teams_window_package"]
    assert len(pushes) == 1
    assert result["counts"]["selected"] == 1
    assert result["error_type"] == "LocalAgentLeaseLost"
    assert result["retryable"] is False
    assert result["sync_started"] is False


def test_local_agent_cloud_teams_sync_pushes_current_attempt_scope_attestation_for_empty_target(
    monkeypatch,
    tmp_path: Path,
):
    conversation_id = "19:empty@thread.v2"
    target_scope = {"conversation_ids": [conversation_id]}

    async def fake_collect(job, *, source_id, limit, report_progress=None):
        return {
            "documents": [],
            "poll_audits": [
                {
                    "raw_conversation_id": conversation_id,
                    "pagination_complete": True,
                    "access_probe_status": "ok",
                    "stop_reason": "no_backward_link",
                }
            ],
        }

    monkeypatch.setattr(
        main,
        "_collect_teams_documents_from_cloud_job",
        fake_collect,
        raising=False,
    )
    FakeToolClient.reset({"doc_id": "scope-doc", "document_hash": "scope-hash"})
    result = main._run_cloud_teams_sync_job(
        {
            "job_id": "laj-teams-empty-scope",
            "workspace_id": "ws-from-cloud",
            "operation": "teams_sync",
            "attempt_count": 2,
            "source_id": "src-teams",
            "payload": {
                "conversation_ids": [conversation_id],
                "audit_log_path": str(tmp_path / "teams-audit.jsonl"),
                "projection_scope_transition": {
                    "id": "transition-empty-scope",
                    "previous_scope": {"conversation_ids": ["19:old@thread.v2"]},
                    "target_scope": target_scope,
                },
            },
        },
        _cloud_test_client(),
    )

    push = next(call for call in FakeToolClient.calls if call[0] == "push_teams_window_package")
    raw_payload = push[1]["raw_payload"]
    assert raw_payload["_scope_attestation"] is True
    assert raw_payload["transition_id"] == "transition-empty-scope"
    assert raw_payload["target_conversation_ids"] == [conversation_id]
    assert raw_payload["collection_attempt_id"] == "laj-teams-empty-scope:attempt:2"
    assert raw_payload["poll"]["stop_reason"] == "no_backward_link"
    assert result["counts"]["pushed"] == 1
    assert result["sync_started"] is True


@pytest.mark.parametrize(
    "inventory_response",
    [
        {"error": "inventory unavailable"},
        {
            "units": [
                {
                    "source_unit_id": "unit-stale",
                    "unit_type": "teams_window",
                    "provider_key": "window-stale",
                    "locator": {"conversation_id": "19:conversation@thread.tacv2"},
                }
            ]
        },
    ],
)
def test_local_agent_cloud_teams_sync_fails_closed_when_projection_inventory_is_unavailable(
    monkeypatch,
    tmp_path: Path,
    inventory_response: dict,
):
    async def fake_collect(job, *, source_id, limit, report_progress=None):
        return {
            "documents": [
                {
                    "conversation_id": "19:conversation@thread.tacv2",
                    "root_message_id": "1783500000000",
                    "window_id": "teams-thread:v1:opaque-window",
                    "window_type": "thread",
                    "revision_hash": "sha256:revision-1",
                    "title": "Thread window",
                    "source_url": "https://teams.example.test/window",
                    "last_modified": "2026-07-08T09:24:57Z",
                    "raw_payload": {
                        "messages": [
                            {
                                "id": "1783500000000",
                                "content": "Thread window",
                            }
                        ]
                    },
                    "raw_hash": "sha256:raw-1",
                    "message_count": 1,
                }
            ],
            "poll_audits": [
                {
                    "raw_conversation_id": "19:conversation@thread.tacv2",
                    "pagination_complete": True,
                    "access_probe_status": "ok",
                    "stop_reason": "no_backward_link",
                }
            ],
        }

    monkeypatch.setattr(
        main,
        "_collect_teams_documents_from_cloud_job",
        fake_collect,
        raising=False,
    )
    FakeToolClient.reset({})
    FakeToolClient.projection_inventory_response = inventory_response

    result = main._run_cloud_teams_sync_job(
        {
            "job_id": "laj-teams-inventory-failure",
            "workspace_id": "ws-from-cloud",
            "operation": "teams_sync",
            "attempt_count": 1,
            "source_id": "src-teams",
            "payload": {
                "conversation_ids": ["19:conversation@thread.tacv2"],
                "audit_log_path": str(tmp_path / "teams-audit.jsonl"),
            },
        },
        _cloud_test_client(),
    )

    assert result["error_type"] == "TeamsProjectionInventoryError"
    assert result["retryable"] is True
    assert not [
        call for call in FakeToolClient.calls if call[0] in {"push_teams_window_package", "start_source_processing"}
    ]


def test_local_agent_cloud_teams_sync_replays_revision_until_server_projection_is_confirmed(
    monkeypatch,
    tmp_path: Path,
):
    from memforge.local_agent.teams_audit import validate_teams_audit_run

    async def fake_collect(job, *, source_id, limit, report_progress=None):
        return [
            {
                "conversation_id": "19:conversation@thread.tacv2",
                "root_message_id": "1783500000000",
                "window_id": "teams-thread:v1:opaque-window",
                "window_type": "thread",
                "revision_hash": "sha256:revision-1",
                "title": "Thread window",
                "source_url": "https://teams.microsoft.com/l/message/19:conversation@thread.tacv2/1783500000001",
                "last_modified": "2026-07-08T09:24:57.5870000Z",
                "raw_payload": {
                    "conversation_type": "channel",
                    "messages": [{"id": "1783500000000", "content": "Thread window", "from": "Alice"}],
                },
                "raw_hash": "sha256:raw-1",
                "message_count": 2,
            }
        ]

    monkeypatch.setattr(main, "_collect_teams_documents_from_cloud_job", fake_collect, raising=False)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    audit_log_path = tmp_path / "teams-audit.jsonl"
    ledger_state_path = tmp_path / "teams-ledger.json"
    job = {
        "job_id": "laj-teams-sync",
        "workspace_id": "ws-from-cloud",
        "operation": "teams_sync",
        "attempt_count": 1,
        "source_id": "src-teams",
        "payload": {
            "conversation_ids": ["19:conversation@thread.tacv2"],
            "audit_log_path": str(audit_log_path),
            "ledger_state_path": str(ledger_state_path),
        },
    }

    FakeToolClient.reset({"doc_id": "teams-doc", "document_hash": "hash"})
    first = main._run_cloud_local_agent_job(
        job,
        _cloud_test_client(),
    )
    assert first["counts"] == {"selected": 1, "pushed": 1, "failed": 0, "skipped_existing": 0, "polls": 0}
    assert len([call for call in FakeToolClient.calls if call[0] == "push_teams_window_package"]) == 1

    FakeToolClient.reset({"doc_id": "teams-doc", "document_hash": "hash"})
    retry_progress: list[dict] = []
    second = main._run_cloud_local_agent_job(
        {**job, "job_id": "laj-teams-sync-retry"},
        _cloud_test_client(),
        report_progress=retry_progress.append,
    )

    assert second["counts"] == {"selected": 1, "pushed": 1, "failed": 0, "skipped_existing": 0, "polls": 0}
    assert len([call for call in FakeToolClient.calls if call[0] == "push_teams_window_package"]) == 1
    assert len([call for call in FakeToolClient.calls if call[0] == "start_source_processing"]) == 1
    assert retry_progress[-1]["progress"] == {
        "completed": 2,
        "total": 2,
        "unit": "message",
    }

    audit_rows = [json.loads(line) for line in audit_log_path.read_text(encoding="utf-8").splitlines()]
    second_run = [row for row in audit_rows if row["run_id"] == "laj-teams-sync-retry"]
    assert validate_teams_audit_run(second_run) == []
    assert second_run[0]["event"] == "teams_window_projection"
    assert second_run[0]["receipt_status"] == "new"
    assert "receipt_skip_reason" not in second_run[0]
    assert second_run[-1]["skipped_existing_windows"] == 0


def test_local_agent_cloud_teams_sync_retries_window_when_processing_was_not_accepted(
    monkeypatch,
    tmp_path: Path,
):
    async def fake_collect(job, *, source_id, limit, report_progress=None):
        return [
            {
                "conversation_id": "19:conversation@thread.tacv2",
                "root_message_id": "1783500000000",
                "window_id": "teams-thread:v1:retry-window",
                "window_type": "thread",
                "revision_hash": "sha256:revision-retry",
                "title": "Retry window",
                "source_url": "https://teams.microsoft.com/l/message/retry",
                "raw_payload": {"messages": [{"id": "1783500000000", "content": "Retry me"}]},
                "raw_hash": "sha256:raw-retry",
                "message_count": 1,
            }
        ]

    processing_results = [
        {"error": "service restarting"},
        {"run_id": "run-accepted"},
    ]

    def start_source_processing(self, **kwargs):
        self.calls.append(("start_source_processing", kwargs))
        return processing_results.pop(0)

    monkeypatch.setattr(main, "_collect_teams_documents_from_cloud_job", fake_collect, raising=False)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    monkeypatch.setattr(FakeToolClient, "start_source_processing", start_source_processing)
    audit_log_path = tmp_path / "teams-audit.jsonl"
    ledger_state_path = tmp_path / "teams-ledger.json"
    job = {
        "job_id": "laj-teams-retry",
        "workspace_id": "ws-from-cloud",
        "operation": "teams_sync",
        "attempt_count": 1,
        "source_id": "src-teams",
        "payload": {
            "conversation_ids": ["19:conversation@thread.tacv2"],
            "audit_log_path": str(audit_log_path),
            "ledger_state_path": str(ledger_state_path),
        },
    }

    FakeToolClient.reset({"doc_id": "teams-doc", "document_hash": "hash"})
    first = main._run_cloud_local_agent_job(
        job,
        _cloud_test_client(),
    )
    assert first["error"] == "source processing failed to start"
    assert first["retryable"] is True

    FakeToolClient.reset({"doc_id": "teams-doc", "document_hash": "hash"})
    second = main._run_cloud_local_agent_job(
        {**job, "job_id": "laj-teams-retry-2"},
        _cloud_test_client(),
    )

    assert second["sync_started"] is True
    assert second["counts"]["pushed"] == 1
    assert second["counts"]["skipped_existing"] == 0
    assert len([call for call in FakeToolClient.calls if call[0] == "push_teams_window_package"]) == 1


def test_local_agent_cloud_teams_sync_writes_conversation_poll_audit(monkeypatch, tmp_path: Path):
    from memforge.local_agent.teams_audit import validate_teams_audit_run

    async def fake_collect(job, *, source_id, limit, report_progress=None):
        assert source_id == "src-teams"
        return {
            "documents": [
                {
                    "conversation_id": "19:conversation@thread.tacv2",
                    "root_message_id": "1783500000000",
                    "window_id": "teams-thread:v1:opaque-window",
                    "window_type": "thread",
                    "revision_hash": "sha256:revision-1",
                    "title": "Thread window",
                    "source_url": "https://teams.microsoft.com/l/message/19:conversation@thread.tacv2/1783500000001",
                    "last_modified": "2026-07-08T09:24:57.5870000Z",
                    "raw_payload": {
                        "conversation_type": "channel",
                        "messages": [{"id": "1783500000000", "content": "Thread window", "from": "Alice"}],
                    },
                    "raw_hash": "sha256:raw-1",
                    "message_count": 2,
                }
            ],
            "poll_audits": [
                {
                    "raw_conversation_id": "19:conversation@thread.tacv2",
                    "thread_type": "chat",
                    "product_thread_type": "Chat",
                    "pagination_complete": True,
                    "access_probe_status": "ok",
                    "covered_created_from": "2026-07-08T09:00:00+00:00",
                    "covered_created_to": "2026-07-08T10:00:00+00:00",
                    "raw_messages_seen": 2,
                    "unique_message_keys_seen": 2,
                    "duplicate_raw_messages": 0,
                    "upsert_new": 1,
                    "upsert_updated": 0,
                    "upsert_unchanged": 1,
                    "explicit_delete_markers": 0,
                    "missing_once_candidates": 0,
                    "metadata_sync_state": "opaque-sync-state",
                    "metadata_backward_link": "opaque-backward-link",
                }
            ],
        }

    monkeypatch.setattr(main, "_collect_teams_documents_from_cloud_job", fake_collect, raising=False)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "teams-doc", "document_hash": "hash"})
    audit_log_path = tmp_path / "teams-audit.jsonl"
    ledger_state_path = tmp_path / "teams-ledger.json"

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-teams-sync",
            "workspace_id": "ws-from-cloud",
            "operation": "teams_sync",
            "attempt_count": 1,
            "source_id": "src-teams",
            "payload": {
                "conversation_ids": ["19:conversation@thread.tacv2"],
                "audit_log_path": str(audit_log_path),
                "ledger_state_path": str(ledger_state_path),
            },
        },
        _cloud_test_client(),
    )

    assert payload["counts"]["polls"] == 1
    audit_rows = [json.loads(line) for line in audit_log_path.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in audit_rows] == [
        "teams_conversation_poll",
        "teams_window_projection",
        "teams_memory_patch",
        "teams_sync_run",
    ]
    assert validate_teams_audit_run(audit_rows) == []
    poll = audit_rows[0]
    assert poll["raw_conversation_id_hash"].startswith("sha256:")
    assert "raw_conversation_id" not in poll
    assert "metadata_sync_state_hash" in poll
    assert "metadata_backward_link_hash" in poll
    assert "opaque-sync-state" not in audit_log_path.read_text(encoding="utf-8")


def test_collect_teams_documents_from_cloud_job_uses_gene_window_shape(monkeypatch):
    from datetime import datetime, timezone

    from memforge.local_agent.teams_ledger import decode_teams_window_id
    from memforge.models import ContentItem, RawContent

    closed = {"value": False}

    class FakeTeamsClient:
        async def close(self):
            closed["value"] = True

    class FakeTeamsGene:
        def __init__(self, config, source_id):
            self.config = config
            self.source_id = source_id
            self._client = FakeTeamsClient()

        async def authenticate(self):
            assert self.source_id == "src-teams"
            assert self.config["conversation_ids"] == ["19:conversation@thread.tacv2"]
            assert "group_chats" not in self.config
            assert self.config["conversation_gap_minutes"] == 60
            assert self.config["ledger_state_path"].endswith("teams-ledger-state.json")
            assert "local_agent_documents_dir" not in self.config
            assert "local_agent_package_manifest" not in self.config
            assert "audit_log_path" not in self.config

        async def discover(self, since):
            assert since is None
            yield ContentItem(
                item_id="teams-19:conversation@thread.tacv2#1783500000000",
                title="Group: planning -- Jul 8",
                source_url="https://teams.microsoft.com/l/message/19:conversation@thread.tacv2/1783500000000",
                last_modified=datetime(2026, 7, 8, 9, 24, 57, tzinfo=timezone.utc),
                content_type="application/json",
                space_or_project="planning",
                version="",
                author="Andrew",
                labels=["group_chat", "planning"],
                extra={
                    "conversation_id": "19:conversation@thread.tacv2",
                    "root_message_id": "1783500000000",
                    "message_count": 3,
                    "is_thread": False,
                },
            )

        async def fetch(self, item):
            return RawContent(
                item=item,
                body=json.dumps(
                    {
                        "conversation_type": "group_chat",
                        "messages": [
                            {"id": "1783500000000", "content": "Window body.", "from": "Andrew"},
                            {"id": "1783500000001", "content": "Second", "from": "Ada"},
                            {"id": "1783500000002", "content": "Third", "from": "Grace"},
                        ],
                    }
                ).encode(),
                content_type="application/json",
            )

        def get_poll_audits(self):
            return [
                {
                    "raw_conversation_id": "19:conversation@thread.tacv2",
                    "field_contract_version": "teams_chatsvc_rest_v1",
                    "pagination_complete": True,
                    "access_probe_status": "ok",
                    "stop_reason": "no_backward_link",
                    "page_count": 1,
                    "covered_created_from": "2026-07-08T09:00:00+00:00",
                    "covered_created_to": "2026-07-08T09:24:57+00:00",
                    "raw_messages_seen": 4,
                    "unique_message_keys_seen": 3,
                    "selected_message_keys_seen": 3,
                    "duplicate_raw_messages": 1,
                    "parse_filtered_messages": 1,
                    "upsert_new": 3,
                    "upsert_updated": 0,
                    "upsert_unchanged": 0,
                    "explicit_delete_markers": 0,
                    "missing_once_candidates": 0,
                }
            ]

    import memforge.genes.teams_gene as teams_gene

    monkeypatch.setattr(teams_gene, "TeamsGene", FakeTeamsGene)

    collection = main.asyncio.run(
        main._collect_teams_documents_from_cloud_job(
            {
                "payload": {
                    "group_chats": ["19:conversation@thread.tacv2"],
                    "conversation_gap_minutes": 60,
                    "local_agent_documents_dir": "/srv/memforge/inbox/src-teams",
                    "local_agent_package_manifest": [
                        {
                            "doc_id": "teams-src-teams-old",
                            "package_path": "/srv/memforge/inbox/src-teams/teams-src-teams-old.json",
                        }
                    ],
                    "audit_log_path": "/tmp/teams-audit.jsonl",
                },
            },
            source_id="src-teams",
            limit=1,
        )
    )

    assert closed["value"] is True
    assert sorted(collection) == ["documents", "poll_audits"]
    documents = collection["documents"]
    poll_audits = collection["poll_audits"]
    assert len(documents) == 1
    assert len(poll_audits) == 1
    assert poll_audits[0]["raw_conversation_id"] == "19:conversation@thread.tacv2"
    assert poll_audits[0]["pagination_complete"] is True
    assert poll_audits[0]["access_probe_status"] == "ok"
    assert poll_audits[0]["page_count"] == 1
    assert poll_audits[0]["raw_messages_seen"] == 4
    assert poll_audits[0]["unique_message_keys_seen"] == 3
    assert poll_audits[0]["duplicate_raw_messages"] == 1
    assert poll_audits[0]["upsert_new"] == 3
    doc = documents[0]
    assert doc["conversation_id"] == "19:conversation@thread.tacv2"
    assert doc["root_message_id"] == "1783500000000"
    assert doc["window_id"].startswith("teams-block:v1:")
    assert "19:conversation@thread.tacv2" not in doc["window_id"]
    assert decode_teams_window_id(doc["window_id"]) == {
        "source_id": "src-teams",
        "conversation_id": "19:conversation@thread.tacv2",
        "root_or_anchor_message_id": "1783500000000",
        "window_type": "time_block",
    }
    assert doc["window_type"] == "time_block"
    assert len(doc["raw_payload"]["messages"]) == 3
    assert doc["raw_payload"]["_authoritative_snapshot"] is True
    assert doc["message_count"] == 3
    assert doc["last_modified"] == "2026-07-08T09:24:57+00:00"
    assert doc["raw_hash"]
    assert doc["revision_hash"]


def test_teams_window_completeness_requires_successful_full_conversation_poll():
    assert main._teams_complete_poll_conversation_ids(
        [
            {
                "raw_conversation_id": "conversation-complete",
                "pagination_complete": True,
                "access_probe_status": "ok",
                "stop_reason": "no_backward_link",
            },
            {
                "raw_conversation_id": "conversation-truncated",
                "pagination_complete": False,
                "access_probe_status": "ok",
                "stop_reason": "cutoff_reached",
            },
            {
                "raw_conversation_id": "conversation-denied",
                "pagination_complete": True,
                "access_probe_status": "forbidden",
                "stop_reason": "no_backward_link",
            },
        ]
    ) == {"conversation-complete"}


def test_teams_coverage_time_rejects_naive_provider_evidence():
    assert main._parse_teams_coverage_time("2026-07-16T09:00:00") is None
    assert main._parse_teams_coverage_time("2026-07-16T09:00:00+00:00") is not None


def test_server_inventory_emits_window_tombstone_only_after_complete_poll():
    from memforge.local_agent.teams_ledger import build_teams_window_id

    conversation_id = "19:conversation@thread.tacv2"
    window_id = build_teams_window_id(
        source_id="src-teams",
        conversation_id=conversation_id,
        root_or_anchor_message_id="m1",
        window_type="time_block",
    )
    documents = main._reconcile_teams_documents_with_server_inventory(
        documents=[],
        poll_audits=[
            {
                "raw_conversation_id": conversation_id,
                "pagination_complete": True,
                "access_probe_status": "ok",
                "stop_reason": "no_backward_link",
            }
        ],
        inventory_units=[
            {
                "provider_key": window_id,
                "locator": {
                    "conversation_id": conversation_id,
                    "window_id": window_id,
                    "observed_from": "2026-07-08T09:00:00+00:00",
                    "observed_to": "2026-07-08T09:30:00+00:00",
                    "url": "https://teams.example.test/window/m1",
                },
            }
        ],
        configured_conversation_ids={conversation_id},
        destructive_enabled=True,
    )

    assert len(documents) == 1
    tombstone = documents[0]
    assert tombstone["window_id"] == window_id
    assert tombstone["message_count"] == 0
    assert tombstone["raw_payload"] == {
        "conversation_id": conversation_id,
        "window_id": window_id,
        "messages": [],
        "_authoritative_snapshot": True,
        "_tombstone": True,
        "tombstone_reason": "not_returned_by_complete_conversation_poll",
    }


def test_server_inventory_uses_bounded_poll_only_for_contained_window():
    from memforge.local_agent.teams_ledger import build_teams_window_id

    conversation_id = "19:conversation@thread.tacv2"

    def inventory(window_name: str, observed_from: str, observed_to: str):
        window_id = build_teams_window_id(
            source_id="src-teams",
            conversation_id=conversation_id,
            root_or_anchor_message_id=window_name,
            window_type="time_block",
        )
        return {
            "provider_key": window_id,
            "locator": {
                "conversation_id": conversation_id,
                "window_id": window_id,
                "observed_from": observed_from,
                "observed_to": observed_to,
            },
        }

    documents = main._reconcile_teams_documents_with_server_inventory(
        documents=[],
        poll_audits=[
            {
                "raw_conversation_id": conversation_id,
                "pagination_complete": False,
                "access_probe_status": "ok",
                "stop_reason": "cutoff_reached",
                "absence_covered_from": "2026-07-01T00:00:00+00:00",
                "absence_covered_to": "2026-07-16T00:00:00+00:00",
            }
        ],
        inventory_units=[
            inventory(
                "recent-deleted",
                "2026-07-08T09:00:00+00:00",
                "2026-07-08T09:30:00+00:00",
            ),
            inventory(
                "older-retained",
                "2026-06-08T09:00:00+00:00",
                "2026-06-08T09:30:00+00:00",
            ),
        ],
        configured_conversation_ids={conversation_id},
        destructive_enabled=True,
    )

    assert len(documents) == 2
    assert documents[0]["root_message_id"] == "recent-deleted"
    assert documents[0]["raw_payload"]["tombstone_reason"] == "not_returned_by_bounded_conversation_poll"
    assert documents[1]["root_message_id"] == "older-retained"
    assert documents[1]["raw_payload"]["tombstone_reason"] == "outside_configured_time_scope"


def test_server_inventory_never_tombstones_after_invalid_message_page():
    from memforge.local_agent.teams_ledger import build_teams_window_id

    conversation_id = "19:conversation@thread.tacv2"
    window_id = build_teams_window_id(
        source_id="src-teams",
        conversation_id=conversation_id,
        root_or_anchor_message_id="still-unknown",
        window_type="time_block",
    )

    documents = main._reconcile_teams_documents_with_server_inventory(
        documents=[],
        poll_audits=[
            {
                "raw_conversation_id": conversation_id,
                "pagination_complete": False,
                "access_probe_status": "ok",
                "stop_reason": "invalid_message_page_schema",
            }
        ],
        inventory_units=[
            {
                "provider_key": window_id,
                "locator": {
                    "conversation_id": conversation_id,
                    "window_id": window_id,
                    "observed_from": "2026-07-08T09:00:00+00:00",
                    "observed_to": "2026-07-08T09:30:00+00:00",
                },
            }
        ],
        configured_conversation_ids={conversation_id},
        destructive_enabled=True,
    )

    assert documents == []


def test_server_inventory_tombstones_conversation_removed_from_scope():
    from memforge.local_agent.teams_ledger import build_teams_window_id

    removed_conversation = "19:removed@thread.tacv2"
    window_id = build_teams_window_id(
        source_id="src-teams",
        conversation_id=removed_conversation,
        root_or_anchor_message_id="removed-root",
        window_type="thread",
    )

    documents = main._reconcile_teams_documents_with_server_inventory(
        documents=[],
        poll_audits=[],
        inventory_units=[
            {
                "provider_key": window_id,
                "locator": {
                    "conversation_id": removed_conversation,
                    "window_id": window_id,
                },
            }
        ],
        configured_conversation_ids={"19:retained@thread.tacv2"},
        destructive_enabled=True,
    )

    assert len(documents) == 1
    assert documents[0]["raw_payload"]["tombstone_reason"] == "conversation_removed_from_projection_scope"


def test_server_inventory_pagination_reconciles_every_unit_once():
    from memforge.local_agent.teams_ledger import build_teams_window_id

    conversation_id = "19:conversation@thread.tacv2"
    windows = [
        build_teams_window_id(
            source_id="src-teams",
            conversation_id=conversation_id,
            root_or_anchor_message_id=root,
            window_type="time_block",
        )
        for root in ("root-a", "root-b")
    ]

    class PagingClient:
        def __init__(self):
            self.cursors = []

        def get_source_projection_inventory(self, source_id, **filters):
            assert source_id == "src-teams"
            self.cursors.append(filters.get("cursor"))
            index = 0 if filters.get("cursor") is None else 1
            window_id = windows[index]
            return {
                "units": [
                    {
                        "source_unit_id": f"unit-{index}",
                        "unit_type": "teams_window",
                        "provider_key": window_id,
                        "locator": {
                            "conversation_id": conversation_id,
                            "window_id": window_id,
                            "observed_from": "2026-07-08T09:00:00+00:00",
                            "observed_to": "2026-07-08T09:30:00+00:00",
                        },
                    }
                ],
                "next_cursor": "unit-0" if index == 0 else None,
            }

    client = PagingClient()
    tombstones = list(
        main._iter_teams_inventory_tombstones(
            client=client,
            source_id="src-teams",
            current_documents=[],
            poll_audits=[
                {
                    "raw_conversation_id": conversation_id,
                    "pagination_complete": True,
                    "access_probe_status": "ok",
                    "stop_reason": "no_backward_link",
                }
            ],
            configured_conversation_ids={conversation_id},
            scope_transition=None,
        )
    )

    assert [item["window_id"] for item in tombstones] == windows
    assert client.cursors == [None, "unit-0"]


def test_bounded_inventory_plan_separates_recent_reconciliation_from_scope_cleanup():
    conversation_id = "19:conversation@thread.tacv2"
    audit = {
        "raw_conversation_id": conversation_id,
        "pagination_complete": False,
        "access_probe_status": "ok",
        "stop_reason": "cutoff_reached",
        "absence_covered_from": "2026-07-01T00:00:00+00:00",
        "absence_covered_to": "2026-07-16T00:00:00+00:00",
    }

    ordinary = main._teams_inventory_query_plans(
        poll_audits=[audit],
        configured_conversation_ids={conversation_id},
        scope_transition=None,
    )
    max_age_transition = main._teams_inventory_query_plans(
        poll_audits=[audit],
        configured_conversation_ids={conversation_id},
        scope_transition={
            "previous_scope": {
                "conversation_ids": [conversation_id],
                "max_age_days": 365,
            },
            "target_scope": {
                "conversation_ids": [conversation_id],
                "max_age_days": 30,
            },
        },
    )

    expected = [
        {
            "conversation_id": conversation_id,
            "observed_from_lte": "2026-07-16T00:00:00+00:00",
            "observed_to_gte": "2026-07-01T00:00:00+00:00",
        },
        {
            "conversation_id": conversation_id,
            "observed_to_lt": "2026-07-01T00:00:00+00:00",
        },
    ]
    assert ordinary == expected
    assert max_age_transition == expected


def test_collect_teams_documents_from_cloud_job_maps_cloud_conversation_ids_to_direct_rest_config(monkeypatch):
    class FakeTeamsClient:
        async def close(self):
            pass

    class FakeTeamsGene:
        def __init__(self, config, source_id):
            assert config["conversation_ids"] == ["19:conversation@thread.tacv2"]
            assert "group_chats" not in config
            assert "channels" not in config
            assert "individual_chats" not in config
            self._client = FakeTeamsClient()

        async def authenticate(self):
            pass

        async def discover(self, since):
            if False:
                yield

    import memforge.genes.teams_gene as teams_gene

    monkeypatch.setattr(teams_gene, "TeamsGene", FakeTeamsGene)

    collection = main.asyncio.run(
        main._collect_teams_documents_from_cloud_job(
            {"payload": {"conversation_ids": ["19:conversation@thread.tacv2"]}},
            source_id="src-teams",
            limit=0,
        )
    )

    assert collection == {"documents": [], "poll_audits": []}


def test_collect_teams_documents_from_cloud_job_rejects_name_based_teams_config():
    result = main.asyncio.run(
        _capture_collect_teams_documents_error(
            {
                "payload": {
                    "channels": ["Engineering/architecture"],
                    "group_chats": ["Planning Chat"],
                    "individual_chats": ["Alice"],
                },
            }
        )
    )

    assert result == "teams_sync_requires_direct_conversation_ids"


async def _capture_collect_teams_documents_error(job):
    try:
        await main._collect_teams_documents_from_cloud_job(job, source_id="src-teams", limit=0)
    except ValueError as exc:
        return str(exc)
    return None


def test_adapter_jira_watch_command_is_registered():
    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "watch", "--help"])
    assert result.exit_code == 0, result.output
    assert "--interval-seconds" in result.output
