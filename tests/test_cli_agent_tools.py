import base64
import hashlib
import json
import subprocess
import sys
from tempfile import TemporaryFile
from pathlib import Path

import pytest
from click.testing import CliRunner

import memforge.main as main
from memforge.api_target import MemForgeTarget, build_target
from memforge.genes.jira_gene import JiraGene
from memforge.local_agent import folder_picker
from memforge.local_agent.setup_results import (
    CredentialCheckResult,
    LocalMarkdownPickRootResult,
    TeamsAuthCheckResult,
    TeamsAuthResult,
)
from memforge.main import cli


@pytest.fixture(autouse=True)
def _isolate_cli_target_configuration(monkeypatch, tmp_path: Path):
    for name in ("MEMFORGE_API_URL", "MEMFORGE_EDITION", "MEMFORGE_WORKSPACE_ID", "MEMFORGE_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(tmp_path / "isolated-cli.toml"))
    monkeypatch.setenv("MEMFORGE_WORKSPACE_BINDINGS_FILE", str(tmp_path / "isolated-workspace-bindings.json"))
    monkeypatch.setattr(
        main,
        "discover_target",
        lambda origin: build_target(origin=origin, edition="cloud"),
    )
    from memforge.auth.teams_auth import TeamsAuthenticator

    monkeypatch.setattr(TeamsAuthenticator, "_load_keychain_token_data", staticmethod(lambda: None))
    monkeypatch.setattr(TeamsAuthenticator, "_save_keychain_token_data", staticmethod(lambda _data: False))


class FakeToolClient:
    calls: list[tuple[str, dict]] = []
    response: dict = {}
    list_response: dict = {"data": []}
    snapshot_plan_response: dict | None = None
    def __init__(
        self,
        *,
        target: MemForgeTarget,
        api_token: str | None = None,
        workspace_id: str | None = None,
        timeout_seconds: float = 60.0,
    ):
        self.target = target
        self.api_url = target.origin
        self.api_token = api_token
        self.workspace_id = workspace_id or ""
        self.timeout_seconds = timeout_seconds

    @classmethod
    def reset(cls, response: dict, *, list_response: dict | None = None) -> None:
        cls.calls = []
        cls.response = response
        cls.list_response = {"data": []} if list_response is None else list_response
        cls.snapshot_plan_response = None
    def list_sources(self):
        self.calls.append(("list_sources", {"api_url": self.api_url, "api_token": self.api_token}))
        return self.list_response
    def list_searchable_sources(self):
        self.calls.append(("list_searchable_sources", {"api_url": self.api_url, "api_token": self.api_token}))
        return self.list_response

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
    def reprocess_source_documents(self, **kwargs):
        self.calls.append(("reprocess_source_documents", {"api_url": self.api_url, **kwargs}))
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
    def start_source_processing(self, **kwargs):
        self.calls.append(
            (
                "start_source_processing",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        if self.response.get("error") or self.response.get("run_id"):
            return self.response
        return {"run_id": "run-test"}
    def push_local_markdown_document(self, **kwargs):
        self.calls.append(
            (
                "push_local_markdown_document",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        return self.response
    def push_source_artifact(self, **kwargs):
        self.calls.append(
            (
                "push_source_artifact",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        return {
            "input_sha256": "a" * 64,
            "content_sha256": hashlib.sha256(kwargs["content"]).hexdigest(),
            "size_bytes": len(kwargs["content"]),
        }
    def prepare_local_source_snapshot(self, **kwargs):
        self.calls.append(
            (
                "prepare_local_source_snapshot",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        if self.snapshot_plan_response is not None:
            return self.snapshot_plan_response
        return {
            "required_doc_ids": [str(item["doc_id"]) for item in kwargs.get("items", [])],
            "reused_count": 0,
        }
    def upload_jira_session(self, **kwargs):
        self.calls.append(
            (
                "upload_jira_session",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        return self.response
    def upload_browser_session(self, **kwargs):
        self.calls.append(
            (
                "upload_browser_session",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        return self.response

    def health(self):
        self.calls.append(("health", {"api_url": self.api_url, "api_token": self.api_token}))
        return self.response
    def for_workspace(self, workspace_id: str):
        return type(self)(
            target=build_target(origin=self.api_url, edition="cloud"),
            api_token=self.api_token,
            workspace_id=workspace_id,
            timeout_seconds=self.timeout_seconds,
        )


def _cloud_test_client() -> FakeToolClient:
    return FakeToolClient(
        target=build_target(
            origin="https://memforge-dev.example.com",
            edition="cloud",
        ),
        api_token="tok",
        workspace_id="ws-from-cloud",
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
            "--intent",
            "general_hybrid",
            "--type",
            "fact",
            "--include-superseded",
        ],
        env={
            "MEMFORGE_API_URL": "https://memforge-dev.example.com",
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
                "api_url": "https://memforge-dev.example.com",
                "api_token": "token-1",
                "query": "docker artifact provenance",
                "top_k": 3,
                "intent": "general_hybrid",
                "memory_types": ["fact"],
                "include_private": True,
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
            "src-alpha",
            "--start-date",
            "2026-06-20",
            "--end-date",
            "2026-06-26",
        ],
        env={
            "MEMFORGE_API_URL": "https://memforge-dev.example.com",
            "MEMFORGE_WORKSPACE_ID": "ws-from-cloud",
            "MEMFORGE_API_TOKEN": "token-1",
        },
    )

    assert result.exit_code == 0, result.output
    assert FakeToolClient.calls == [
        (
            "search",
            {
                "api_url": "https://memforge-dev.example.com",
                "api_token": "token-1",
                "query": "",
                "top_k": 20,
                "include_private": True,
                "include_superseded": False,
                "source_filter": {"source_ids": ["src-alpha"]},
                "time_range": {
                    "date_type": "source_updated_at",
                    "start_date": "2026-06-20",
                    "end_date": "2026-06-26",
                },
            },
        )
    ]


@pytest.mark.parametrize("command", ["search", "memory search"])
def test_search_cli_includes_the_callers_own_private_memories(monkeypatch, command):
    FakeToolClient.reset({"results": []})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        [*command.split(), "release process"],
        env={"MEMFORGE_API_URL": "https://memforge-dev.example.com"},
    )

    assert result.exit_code == 0, result.output
    [(name, kwargs)] = FakeToolClient.calls
    assert name == "search"
    assert kwargs["include_private"] is True


def test_search_cli_rejects_legacy_source_name_filter(monkeypatch):
    FakeToolClient.reset({"results": []})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        ["search", "jira defects", "--source", "Gamma Defects"],
        env={"MEMFORGE_API_URL": "https://memforge-dev.example.com"},
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
        ["get-resource", "/api/v1/documents/doc-456/pdf", "--mode", "file", "--max-bytes", "1024"],
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
                "url": "/api/v1/documents/doc-456/pdf",
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
                    "name": "Alpha Defects",
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
            "MEMFORGE_API_URL": "https://memforge-dev.example.com",
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
            {"api_url": "https://memforge-dev.example.com", "api_token": "token-1"},
        )
    ]


def test_sources_searchable_cli_reads_searchable_sources_from_active_api_target(monkeypatch):
    FakeToolClient.reset(
        {},
        list_response={
            "data": [
                {
                    "source_id": "src-alpha",
                    "name": "Alpha Defects",
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
            "MEMFORGE_API_URL": "https://memforge-dev.example.com",
            "MEMFORGE_WORKSPACE_ID": "ws-from-cloud",
            "MEMFORGE_API_TOKEN": "token-1",
        },
    )

    assert result.exit_code == 0, result.output
    assert "Searchable Sources" in result.output
    assert "src-alpha" in result.output
    assert FakeToolClient.calls == [
        (
            "list_searchable_sources",
            {"api_url": "https://memforge-dev.example.com", "api_token": "token-1"},
        )
    ]


def test_sources_schedule_cli_updates_active_api_target(monkeypatch):
    FakeToolClient.reset({"ok": True, "source_id": "src-1"})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        ["sources", "schedule", "src-1", "--every-minutes", "60"],
        env={
            "MEMFORGE_API_URL": "https://memforge-dev.example.com",
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
                "api_url": "https://memforge-dev.example.com",
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
            "MEMFORGE_API_URL": "https://memforge-dev.example.com",
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
                "api_url": "https://memforge-dev.example.com",
                "api_token": "token-1",
                "source_id": "src-1",
            },
        )
    ]


@pytest.mark.parametrize("dry_run", [False, True])
def test_sources_reprocess_cli_names_documents_on_the_active_api_target(monkeypatch, dry_run):
    FakeToolClient.reset({"ok": True, "estimated_model_calls": 7})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        ["sources", "reprocess", "src-1", "--document", "jira-PAY-1", "--document", "jira-PAY-2",
         *(["--dry-run"] if dry_run else [])],
        env={"MEMFORGE_API_URL": "https://memforge-dev.example.com", "MEMFORGE_API_TOKEN": "t"},
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["estimated_model_calls"] == 7
    assert FakeToolClient.calls == [
        (
            "reprocess_source_documents",
            {
                "api_url": "https://memforge-dev.example.com",
                "source_id": "src-1",
                "document_ids": ["jira-PAY-1", "jira-PAY-2"],
                "all_documents": False,
                "dry_run": dry_run,
            },
        )
    ]


def test_sources_reprocess_cli_takes_every_document_with_all(monkeypatch):
    FakeToolClient.reset({"ok": True})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        ["sources", "reprocess", "src-1", "--all"],
        env={"MEMFORGE_API_URL": "https://memforge-dev.example.com", "MEMFORGE_API_TOKEN": "t"},
    )

    assert result.exit_code == 0, result.output
    assert FakeToolClient.calls[0][1]["document_ids"] == []
    assert FakeToolClient.calls[0][1]["all_documents"] is True


@pytest.mark.parametrize(
    "arguments",
    [[], ["--all", "--document", "jira-PAY-1"]],
    ids=["neither", "both"],
)
def test_sources_reprocess_cli_takes_documents_or_all(arguments):
    result = CliRunner().invoke(cli, ["sources", "reprocess", "src-1", *arguments])

    assert result.exit_code != 0
    assert "--document" in result.output and "--all" in result.output


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
            "acme.prod",
            "--api-url",
            "https://memforge-dev.example.com",
            "--token-env",
            "ACME_TOKEN",
        ],
    )
    check_result = CliRunner().invoke(cli, ["target", "check"], env={"ACME_TOKEN": "secret-token"})

    assert add_result.exit_code == 0, add_result.output
    assert check_result.exit_code == 0, check_result.output
    assert FakeToolClient.calls == [
        ("health", {"api_url": "https://memforge-dev.example.com", "api_token": "secret-token"})
    ]
    assert 'edition = "cloud"' in cli_config.read_text(encoding="utf-8")
    assert "workspace_id" not in cli_config.read_text(encoding="utf-8")


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
            "acme",
            "--api-url",
            "https://memforge-dev.example.com",
            "--token-env",
            "ACME_TOKEN",
        ],
    )
    check_result = CliRunner().invoke(
        cli,
        ["target", "check"],
        env={"MEMFORGE_API_TOKEN": "wrong-token", "ACME_TOKEN": "target-token"},
    )

    assert add_result.exit_code == 0, add_result.output
    assert check_result.exit_code == 0, check_result.output
    assert FakeToolClient.calls == [
        ("health", {"api_url": "https://memforge-dev.example.com", "api_token": "target-token"})
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
            "acme",
            "--api-url",
            "https://memforge-dev.example.com",
            "--token-env",
            "ACME_TOKEN",
        ],
    )
    check_result = CliRunner().invoke(
        cli,
        ["target", "check"],
        env={
            "MEMFORGE_API_URL": "https://override.example.test",
            "ACME_TOKEN": "target-token",
        },
    )

    assert add_result.exit_code == 0, add_result.output
    assert check_result.exit_code == 0, check_result.output
    assert FakeToolClient.calls == [("health", {"api_url": "https://override.example.test", "api_token": None})]


def test_target_add_accepts_cloud_origin_without_client_workspace_state(monkeypatch, tmp_path: Path):
    cli_config = tmp_path / "cli.toml"
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(cli_config))

    result = CliRunner().invoke(
        cli,
        ["target", "add", "invalid", "--api-url", "https://memforge-dev.example.com"],
    )

    assert result.exit_code == 0, result.output
    assert "workspace_id" not in cli_config.read_text(encoding="utf-8")


CLOUD_TEST_ORIGIN = "https://memforge-dev.example.com"


class _WorkspaceRecordingClient(FakeToolClient):
    """Record the workspace each CLI command selects for its client."""

    selected: list[str] = []
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        type(self).selected.append(self.workspace_id)


def _bind_directory(tmp_path: Path, directory: Path, workspace_id: str) -> None:
    (tmp_path / "isolated-workspace-bindings.json").write_text(
        json.dumps(
            {
                "version": 1,
                "targets": {CLOUD_TEST_ORIGIN: {"directory_bindings": {str(directory): workspace_id}}},
            }
        ),
        encoding="utf-8",
    )


def _cloud_env() -> dict[str, str]:
    return {"MEMFORGE_API_URL": CLOUD_TEST_ORIGIN, "MEMFORGE_API_TOKEN": "token-1"}


def test_sources_list_uses_explicit_workspace_id(monkeypatch):
    FakeToolClient.reset({}, list_response={"data": []})
    _WorkspaceRecordingClient.selected = []
    monkeypatch.setattr(main, "ToolClient", _WorkspaceRecordingClient)

    result = CliRunner().invoke(cli, ["sources", "list", "--workspace-id", "team_beta"], env=_cloud_env())

    assert result.exit_code == 0, result.output
    assert _WorkspaceRecordingClient.selected == ["team_beta"]


def test_sources_list_lists_workspace_ids_when_selection_is_required(monkeypatch, tmp_path: Path):
    FakeToolClient.reset(
        {},
        list_response={
            "error": "MemForge API request failed",
            "status_code": 409,
            "code": "workspace_selection_required",
            "detail": "Select a workspace for this request.",
            "workspace_ids": ["team_alpha", "team_beta"],
        },
    )
    _WorkspaceRecordingClient.selected = []
    monkeypatch.setattr(main, "ToolClient", _WorkspaceRecordingClient)
    # A current-directory binding only selects the workspace for project commands.
    _bind_directory(tmp_path, tmp_path, "team_alpha")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["sources", "list"], env=_cloud_env())

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["workspace_ids"] == ["team_alpha", "team_beta"]
    assert payload["hint"] == "Pass --workspace-id with one of: team_alpha, team_beta."
    assert _WorkspaceRecordingClient.selected == [""]


@pytest.mark.parametrize(
    "command",
    [
        ["search", "release process"],
        ["memory", "search", "release process"],
        ["get-memory", "mem-1"],
        ["memory", "get", "mem-1"],
        ["get-resource", "/api/v1/documents/doc-1/content"],
        ["memory", "resource", "/api/v1/documents/doc-1/content"],
    ],
)
def test_project_commands_use_current_directory_binding(monkeypatch, tmp_path: Path, command):
    FakeToolClient.reset({"results": []})
    _WorkspaceRecordingClient.selected = []
    monkeypatch.setattr(main, "ToolClient", _WorkspaceRecordingClient)
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    _bind_directory(tmp_path, project, "team_alpha")
    monkeypatch.chdir(project / "src")

    bound = CliRunner().invoke(cli, command, env=_cloud_env())
    explicit = CliRunner().invoke(cli, [*command, "--workspace-id", "team_beta"], env=_cloud_env())

    assert bound.exit_code == 0, bound.output
    assert explicit.exit_code == 0, explicit.output
    assert _WorkspaceRecordingClient.selected == ["team_alpha", "team_beta"]


def test_project_command_without_binding_sends_no_workspace(monkeypatch, tmp_path: Path):
    FakeToolClient.reset({"results": []})
    _WorkspaceRecordingClient.selected = []
    monkeypatch.setattr(main, "ToolClient", _WorkspaceRecordingClient)
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["search", "release process"], env=_cloud_env())

    assert result.exit_code == 0, result.output
    assert _WorkspaceRecordingClient.selected == [""]


def test_project_command_reports_unusable_binding(monkeypatch, tmp_path: Path):
    FakeToolClient.reset({"results": []})
    monkeypatch.setattr(main, "ToolClient", _WorkspaceRecordingClient)
    (tmp_path / "isolated-workspace-bindings.json").write_text("{not json", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["search", "release process"], env=_cloud_env())

    assert result.exit_code == 1
    assert "workspace_bindings_invalid" in result.output
    assert "--workspace-id" in result.output


class _StubClient:
    """Workspace-scoped CLI client double; ``workspaces`` maps IDs to scoped doubles."""
    def __init__(self, *, workspaces=None, **responses):
        self._responses = responses
        self._workspaces = workspaces or {}
        self.calls = []
    def list_workspaces(self):
        self.calls.append(("list_workspaces", None))
        return self._responses.get(
            "list_workspaces",
            {"workspaces": [{"workspace_id": "ws-a", "selectable": True}]},
        )
    def for_workspace(self, workspace_id):
        self.calls.append(("for_workspace", workspace_id))
        return self._workspaces.get(workspace_id, self)
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


def _jira_principal_changed_response():
    return {
        "error": "MemForge API request failed",
        "status_code": 409,
        "code": "jira_principal_changed",
        "detail": {
            "code": "jira_principal_changed",
            "message": "changed",
            "origin": "https://jira.example.com",
            "old_principal_id": "old-user",
            "new_principal_id": "new-user",
        },
    }


def _workspace_selection_response():
    return {
        "error": "MemForge API request failed",
        "status_code": 409,
        "code": "workspace_selection_required",
        "detail": "Select a workspace for this request.",
        "workspace_ids": ["team_alpha", "team_beta"],
    }


def _fake_jira_capture(monkeypatch, captured=None):
    from memforge.auth import jira_capture

    async def fake_capture(base_url, *, browser=None, interactive=False):
        assert interactive is True
        if captured is not None:
            captured.update(base_url=base_url, browser=browser)
        return jira_capture.JiraCaptureResult(
            origin=base_url,
            cookie_header="SESSION=x",
            browser=browser,
            principal={"accountId": "u1"},
        )

    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)


def test_adapter_jira_status_reports_stored_session(monkeypatch):
    stub = _StubClient(
        get_jira_session={
            "provider": "jira",
            "origin": "https://jira.example.com",
            "status": "active",
            "principal_name": "Rose H",
            "browser": "chrome",
        }
    )
    selected = []
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: selected.append(workspace_id) or stub)
    result = CliRunner().invoke(
        cli,
        ["adapter", "auth", "jira", "status", "--base-url", "https://jira.example.com", "--workspace-id", "ws-a"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "active"
    assert payload["principal_name"] == "Rose H"
    assert stub.calls == [("get_jira_session", "https://jira.example.com")]
    assert selected == ["ws-a"]


def test_adapter_jira_status_missing_session_is_not_an_error(monkeypatch):
    stub = _StubClient(get_jira_session={"provider": "jira", "origin": "https://jira.example.com", "status": "missing"})
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: stub)
    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "status", "--base-url", "https://jira.example.com"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "missing"


def test_adapter_jira_status_reports_workspace_selection_error(monkeypatch):
    stub = _StubClient(get_jira_session=_workspace_selection_response())
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: stub)

    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "status", "--base-url", "https://jira.example.com"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["code"] == "workspace_selection_required"
    assert payload["workspace_ids"] == ["team_alpha", "team_beta"]
    assert payload["hint"] == "Pass --workspace-id with one of: team_alpha, team_beta."
    assert "principal_changed" not in result.output


def test_adapter_jira_refresh_captures_and_uploads(monkeypatch):
    captured = {}
    _fake_jira_capture(monkeypatch, captured)
    stub = _StubClient(upload_jira_session={"provider": "jira", "origin": "https://jira.example.com", "status": "active"})
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: stub)

    result = CliRunner().invoke(
        cli,
        ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.example.com", "--browser", "chrome"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "origin": "https://jira.example.com",
        "results": [
            {
                "workspace_id": "ws-a",
                "status": "uploaded",
                "session": {"provider": "jira", "origin": "https://jira.example.com", "status": "active"},
            }
        ],
    }
    assert captured == {"base_url": "https://jira.example.com", "browser": "chrome"}
    uploads = [call for call in stub.calls if call[0] == "upload_jira_session"]
    assert [call[0] for call in stub.calls[:2]] == ["list_workspaces", "for_workspace"]
    assert uploads[0][1]["cookie_header"] == "SESSION=x"
    assert uploads[0][1]["base_url"] == "https://jira.example.com"


def test_adapter_jira_refresh_uploads_only_to_explicit_workspace(monkeypatch):
    _fake_jira_capture(monkeypatch)
    stub = _StubClient(upload_jira_session={"status": "active"})
    selected = []
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: selected.append(workspace_id) or stub)

    result = CliRunner().invoke(
        cli,
        ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.example.com", "--workspace-id", "ws-b"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["results"][0]["workspace_id"] == "ws-b"
    assert selected == ["ws-b"]
    assert [call[0] for call in stub.calls] == ["upload_jira_session"]


def test_adapter_jira_refresh_fans_out_to_workspaces_with_jira_source(monkeypatch):
    _fake_jira_capture(monkeypatch)
    configured = {"origins": [{"origin": "https://jira.example.com", "configured": True}]}
    accepted = _StubClient(list_jira_origins=configured, upload_jira_session={"status": "active"})
    changed = _StubClient(list_jira_origins=configured, upload_jira_session=_jira_principal_changed_response())
    unrelated = _StubClient(list_jira_origins={"origins": [{"origin": "https://jira.other", "configured": True}]})
    account = _StubClient(
        list_workspaces={
            "workspaces": [
                {"workspace_id": "team_alpha", "selectable": True},
                {"workspace_id": "team_beta", "selectable": True},
                {"workspace_id": "sandbox", "selectable": True},
                {"workspace_id": "suspended", "selectable": False},
            ]
        },
        workspaces={"team_alpha": accepted, "team_beta": changed, "sandbox": unrelated},
    )
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: account)

    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.example.com"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "jira_session_upload_incomplete"
    assert payload["results"] == [
        {"workspace_id": "team_alpha", "status": "uploaded", "session": {"status": "active"}},
        {
            "workspace_id": "team_beta",
            "status": "principal_changed",
            "old_principal_id": "old-user",
            "new_principal_id": "new-user",
        },
    ]
    assert not any(call[0] == "upload_jira_session" for call in unrelated.calls)
    assert ("for_workspace", "suspended") not in account.calls


def test_adapter_jira_refresh_requires_workspace_when_no_workspace_has_the_origin(monkeypatch):
    _fake_jira_capture(monkeypatch)
    account = _StubClient(
        list_workspaces={
            "workspaces": [
                {"workspace_id": "team_alpha", "selectable": True},
                {"workspace_id": "team_beta", "selectable": True},
            ]
        },
    )
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: account)

    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.example.com"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "jira_source_not_found"
    assert payload["workspace_ids"] == ["team_alpha", "team_beta"]
    assert "--workspace-id" in payload["detail"]
    assert not any(call[0] == "upload_jira_session" for call in account.calls)


def test_adapter_jira_refresh_reports_non_principal_conflict_as_failure(monkeypatch):
    _fake_jira_capture(monkeypatch)
    stub = _StubClient(upload_jira_session=_workspace_selection_response())
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: stub)

    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.example.com"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["results"] == [
        {
            "workspace_id": "ws-a",
            "status": "failed",
            "status_code": 409,
            "code": "workspace_selection_required",
            "detail": "Select a workspace for this request.",
        }
    ]
    assert "principal_changed" not in result.output


def test_adapter_jira_refresh_no_session_returns_json_error(monkeypatch):
    from memforge.auth import jira_capture
    from memforge.auth.jira_auth import JiraAuthSessionMissingError

    async def fake_capture(base_url, *, browser=None, interactive=False):
        assert interactive is True
        raise JiraAuthSessionMissingError("No active Jira browser session cookies were found")

    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: _StubClient())

    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.example.com"])
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
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: _StubClient())

    result = CliRunner().invoke(
        cli,
        ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.example.com", "--browser", "netscape"],
    )
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "auth_failed"
    assert "Unsupported browser" in payload["detail"]


def test_adapter_jira_refresh_principal_change_returns_json_error(monkeypatch):
    _fake_jira_capture(monkeypatch)
    stub = _StubClient(upload_jira_session=_jira_principal_changed_response())
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: stub)

    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.example.com"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["origin"] == "https://jira.example.com"
    assert payload["results"] == [
        {
            "workspace_id": "ws-a",
            "status": "principal_changed",
            "old_principal_id": "old-user",
            "new_principal_id": "new-user",
        }
    ]


def test_adapter_jira_list_and_forget(monkeypatch):
    forgotten_locally = []

    monkeypatch.setattr(
        "memforge.auth.jira_browser_session.JiraBrowserSession.forget",
        lambda self, *, origin: forgotten_locally.append(origin),
    )
    stub = _StubClient(
        list_jira_origins={"origins": [{"origin": "https://jira.example.com", "status": "active"}]},
        forget_jira_session={"ok": True, "origin": "https://jira.example.com", "forgotten": True},
    )
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: stub)
    list_result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "list"])
    assert list_result.exit_code == 0, list_result.output
    assert json.loads(list_result.output)["origins"][0]["origin"] == "https://jira.example.com"
    forget_result = CliRunner().invoke(
        cli, ["adapter", "auth", "jira", "forget", "--base-url", "https://jira.example.com"]
    )
    assert forget_result.exit_code == 0, forget_result.output
    assert json.loads(forget_result.output)["forgotten"] is True
    assert forgotten_locally == ["https://jira.example.com"]


def test_legacy_auth_jira_is_removed():
    help_result = CliRunner().invoke(cli, ["auth", "--help"])
    result = CliRunner().invoke(cli, ["auth", "jira", "--base-url", "https://jira.example.com"])

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


def _git_blob_sha(body):
    return hashlib.sha1(b"blob " + str(len(body)).encode() + b"\0" + body).hexdigest()


@pytest.fixture(autouse=True)
def _github_raw_process_transport(monkeypatch):
    """Expose existing provider fixtures through a real bounded stdout pipe."""
    popen = subprocess.Popen
    def raw_popen(cmd, **kwargs):
        if cmd[:2] != ["gh", "api"]:
            return popen(cmd, **kwargs)
        assert "Accept: application/vnd.github.raw+json" in cmd
        result = main.subprocess.run(cmd, env=kwargs["env"])
        payload = json.loads(result.stdout)
        raw = base64.b64decode(payload["content"])
        with TemporaryFile() as body:
            body.write(raw)
            body.seek(0)
            return popen(
                [sys.executable, "-S", "-c", "import shutil,sys; shutil.copyfileobj(sys.stdin.buffer, sys.stdout.buffer)"],
                stdin=body, **kwargs,
            )

    monkeypatch.setattr(main.subprocess, "Popen", raw_popen)


GITHUB_FILE_COMMIT_DATE = "2026-05-01T10:00:00Z"


def _github_file_commits_listing():
    """The commits listing GitHub returns for one file's latest commit."""
    return [{"sha": "file-commit", "commit": {"committer": {"date": GITHUB_FILE_COMMIT_DATE}}}]


def _fake_github_remote_run(cmd, *args, **kwargs):
    assert cmd[:2] == ["gh", "api"]
    assert kwargs["env"]["GH_HOST"] == "github.example.com"
    endpoint = cmd[2]
    if endpoint.endswith("/commits/main"):
        payload = {"sha": "commit-main", "commit": {"tree": {"sha": "tree-main"}}}
    elif "/commits?" in endpoint:
        payload = _github_file_commits_listing()
    elif "/git/trees/tree-main" in endpoint:
        payload = {
            "truncated": False,
            "tree": [
                {"path": "Payroll Processing V2/README.md", "type": "blob", "mode": "100644", "sha": _git_blob_sha(b"# Payroll Processing V2\n\nBody"), "size": len(b"# Payroll Processing V2\n\nBody")},
                {"path": "Payroll Processing V2/Überblick.md", "type": "blob", "mode": "100644", "sha": _git_blob_sha(b"# Payroll Processing V2\n\nBody"), "size": len(b"# Payroll Processing V2\n\nBody")},
                {"path": "Payroll Processing V2/images/Flow.puml", "type": "blob", "mode": "100644", "sha": _git_blob_sha(b"# Payroll Processing V2\n\nBody"), "size": len(b"# Payroll Processing V2\n\nBody")},
                {"path": "Payroll Processing V2/images/ignored.png", "type": "blob", "mode": "100644", "sha": "png", "size": 5},
                {"path": "Flexible Payroll/README.md", "type": "blob", "mode": "100644", "sha": _git_blob_sha(b"# Payroll Processing V2\n\nBody"), "size": len(b"# Payroll Processing V2\n\nBody")},
            ]
        }
    elif "/git/blobs/" in endpoint:
        blob_sha = endpoint.rsplit("/", 1)[-1]
        raw = b"# Payroll Processing V2\n\nBody"
        payload = {
            "sha": blob_sha,
            "content": base64.b64encode(raw).decode(),
            "encoding": "base64",
            "size": len(raw),
        }
    else:
        raise AssertionError(f"unexpected gh endpoint: {endpoint}")
    return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")


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

    assert payload == LocalMarkdownPickRootResult(operation="local_markdown_pick_root", root=str(selected)).model_dump()
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

    assert payload == LocalMarkdownPickRootResult(operation="local_markdown_pick_root", cancelled=True).model_dump()


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
    assert payload["source_sync_run_id"] == "run-test"
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
            "api_url": "https://memforge-dev.example.com",
            "api_token": "tok",
            "workspace_id": "ws-from-cloud",
            "source_id": "src-local",
            "force_full_sync": False,
            "sync_snapshot_id": "laj-empty-snapshot:attempt:1",
            "local_agent_job_id": "laj-empty-snapshot",
            "local_agent_attempt_count": 1,
        }
    ]


def test_local_agent_cloud_local_markdown_noop_plans_complete_manifest_without_uploading_bodies(
    monkeypatch,
    tmp_path: Path,
):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "Decision.md").write_text("# Decision\n\nUse the daemon.", encoding="utf-8")
    (root / "Runbook.md").write_text("# Runbook\n\nRestart safely.", encoding="utf-8")
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({})
    FakeToolClient.snapshot_plan_response = {"required_doc_ids": [], "reused_count": 2}

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-local-noop",
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

    [plan_call] = [call for call in FakeToolClient.calls if call[0] == "prepare_local_source_snapshot"]
    assert plan_call[1]["coverage"] == "complete_snapshot"
    assert len(plan_call[1]["items"]) == 2
    assert {item["change_kind"] for item in plan_call[1]["items"]} == {"upsert"}
    assert not [call for call in FakeToolClient.calls if call[0] == "push_local_markdown_document"]
    assert payload["counts"]["reused"] == 2
    assert payload["counts"]["pushed"] == 0
    assert payload["metrics"]["full_bodies_uploaded"] == 0


def test_local_agent_cloud_local_markdown_incomplete_scan_fails_before_manifest_or_processing(
    monkeypatch,
    tmp_path: Path,
):
    root = tmp_path / "notes"
    root.mkdir()
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({})
    def incomplete_scan(_root, *, include, exclude, counts):
        del include, exclude
        counts["unreadable"] += 1
        return iter(())

    monkeypatch.setattr(main, "_scan_kb_profile", incomplete_scan)

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-local-incomplete",
            "attempt_count": 1,
            "workspace_id": "ws-from-cloud",
            "operation": "local_markdown_sync",
            "source_id": "src-local",
            "payload": {"root": str(root), "vault_id": "engineering"},
        },
        _cloud_test_client(),
    )

    assert payload["error"] == "local collection was incomplete"
    assert payload["retryable"] is True
    assert not [
        call
        for call in FakeToolClient.calls
        if call[0]
        in {"prepare_local_source_snapshot", "push_local_markdown_document", "start_source_processing"}
    ]


def test_local_markdown_complete_scan_rejects_concurrent_file_change(tmp_path: Path):
    root = tmp_path / "notes"
    root.mkdir()
    document = root / "Decision.md"
    document.write_text("# Decision\n\nFirst version.", encoding="utf-8")
    counts = {"included": 0, "ignored": 0, "too_large": 0, "invalid_utf8": 0, "unreadable": 0}
    entries = list(
        main._scan_kb_profile(
            root,
            include=["*.md", "**/*.md"],
            exclude=[],
            counts=counts,
        )
    )
    document.write_text("# Decision\n\nChanged during collection.", encoding="utf-8")

    with pytest.raises(ValueError, match="changed during collection"):
        main._attest_kb_scan_stable(
            root,
            include=["*.md", "**/*.md"],
            exclude=[],
            entries=entries,
        )


def test_local_markdown_complete_scan_rejects_concurrent_membership_change(tmp_path: Path):
    root = tmp_path / "notes"
    root.mkdir()
    (root / "Decision.md").write_text("# Decision", encoding="utf-8")
    counts = {"included": 0, "ignored": 0, "too_large": 0, "invalid_utf8": 0, "unreadable": 0}
    entries = list(
        main._scan_kb_profile(
            root,
            include=["*.md", "**/*.md"],
            exclude=[],
            counts=counts,
        )
    )
    (root / "Runbook.md").write_text("# Runbook", encoding="utf-8")

    with pytest.raises(ValueError, match="membership changed during collection"):
        main._attest_kb_scan_stable(
            root,
            include=["*.md", "**/*.md"],
            exclude=[],
            entries=entries,
        )


def _jira_sync_job(**payload) -> dict:
    return {
        "job_id": "laj-jira-sync",
        "attempt_count": 1,
        "workspace_id": "ws-from-cloud",
        "operation": "jira_sync",
        "source_id": "src-jira",
        "payload": {
            "base_url": "https://jira.example.test",
            "auth_mode": "browser_cookie",
            "sync_mode": "local_agent",
            "projects": ["PAY"],
            **payload,
        },
    }


def _confirming_jira_gene(
    events: list[str],
    *,
    confirm_error: Exception | None = None,
    rebuild_error: Exception | None = None,
    authenticate_error: Exception | None = None,
):
    """A Jira Gene whose inventory lists PAY-1, and which confirms what the server asks about."""
    from datetime import datetime, timezone

    from memforge.models import AbsenceAnswer, ContentItem

    class ConfirmingJiraGene:
        config_schema = JiraGene.config_schema
        configs: list[dict] = []

        def __init__(self, config, source_id):
            self.config = config
            self.source_id = source_id
            ConfirmingJiraGene.configs.append(dict(config))

        async def close(self):
            events.append("closed")

        async def list_scope(self):
            from memforge.genes.jira_gene import JiraQueryRejected
            from memforge.models import ScopeListing, ScopeListingKind

            jql = self.config["jql"]
            if jql == "project = GONE":
                raise JiraQueryRejected("Jira rejected the JQL (HTTP 400): project 'GONE' does not exist")
            if jql == "project = SLOW":
                raise RuntimeError("Jira timed out")
            return ScopeListing(kind=ScopeListingKind.QUERY, doc_ids=frozenset({"jira-PAY-1", "jira-PAY-9"}))

        async def authenticate(self):
            if authenticate_error is not None:
                raise authenticate_error

        async def discover_inventory(self):
            yield ContentItem(
                item_id="jira-PAY-1",
                title="PAY-1 title",
                source_url="https://jira.example.test/browse/PAY-1",
                last_modified=datetime(2026, 7, 7, tzinfo=timezone.utc),
                content_type="application/json",
                version="revision-PAY-1",
                extra={"issue_key": "PAY-1"},
            )

        async def fetch(self, item):
            raise AssertionError("an unchanged issue is not hydrated")

        @classmethod
        def confirmation_item(cls, document_id, provider_key):
            if rebuild_error is not None:
                raise rebuild_error
            return ContentItem(
                item_id=document_id,
                title=provider_key,
                source_url="",
                last_modified=datetime(1970, 1, 1, tzinfo=timezone.utc),
                extra={"issue_ref": provider_key},
            )

        async def confirm_absent(self, items):
            events.append("confirmed " + ", ".join(item.extra["issue_ref"] for item in items))
            if confirm_error is not None:
                raise confirm_error
            return {
                item.item_id: AbsenceAnswer.ABSENT if item.extra["issue_ref"] == "100007" else AbsenceAnswer.EXISTS
                for item in items
            }

    return ConfirmingJiraGene


def _run_confirming_jira_job(monkeypatch, gene_cls, **payload) -> dict:
    from memforge.auth import jira_capture
    import memforge.genes.jira_gene as jira_gene

    async def fake_capture(base_url, *, browser=None, tls_config=None):
        return jira_capture.JiraCaptureResult(
            origin=base_url, cookie_header="JSESSIONID=local", browser=browser, principal={"name": "tester"}
        )

    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    monkeypatch.setattr(jira_gene, "JiraGene", gene_cls)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.snapshot_plan_response = {
        "required_doc_ids": [],
        "reused_count": 1,
        "manifest_sha256": "d" * 64,
        "confirm": [
            {"doc_id": "jira-src-pay-7", "provider_key": "100007"},
            {"doc_id": "jira-src-pay-8", "provider_key": "PAY-8"},
        ],
    }
    return main._run_cloud_local_agent_job(
        {
            "job_id": "laj-jira-confirm",
            "attempt_count": 2,
            "workspace_id": "ws-from-cloud",
            "operation": "jira_sync",
            "source_id": "src-jira",
            "payload": {
                "base_url": "https://jira.example.test",
                "auth_mode": "browser_cookie",
                "projects": ["PAY"],
                **payload,
            },
        },
        _cloud_test_client(),
        browser="chrome",
    )


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

    assert (
        payload
        == TeamsAuthResult(operation="teams_auth", authenticated=True, region="amer", token_count=1).model_dump()
    )
    assert captured_regions == ["amer"]
    assert wait_options == [(90, 2.0)]
    assert FakeToolClient.calls == []


def test_a_server_sources_credential_check_opens_the_browser_and_stores_the_session_on_the_server(monkeypatch):
    from memforge.auth import jira_capture

    captured: dict[str, object] = {}

    async def fake_capture(
        base_url,
        *,
        browser=None,
        tls_config=None,
        interactive=False,
    ):
        captured.update(
            {
                "base_url": base_url,
                "browser": browser,
                "tls_config": tls_config,
                "interactive": interactive,
            }
        )
        return jira_capture.JiraCaptureResult(
            origin=base_url,
            cookie_header="SESSION=interactive",
            browser="MemForge Chrome profile",
            principal={"accountId": "jira-user"},
        )

    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    FakeToolClient.reset(
        {
            "provider": "jira",
            "origin": "https://jira.example.test",
            "status": "active",
        }
    )

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-credential-check",
            "workspace_id": "team_alpha",
            "operation": "credential_check",
            "source_type": "jira",
            "source_id": "src-jira",
            "payload": {
                "source_id": "src-jira",
                "source_type": "jira",
                "executor": "server",
                "config": {"base_url": "https://jira.example.test", "auth_mode": "browser_cookie"},
            },
        },
        _cloud_test_client(),
        browser="chrome",
    )

    assert (
        payload
        == CredentialCheckResult(
            operation="credential_check", source_id="src-jira", authenticated=True, origin="https://jira.example.test"
        ).model_dump()
    )
    assert captured["base_url"] == "https://jira.example.test"
    assert captured["browser"] == "chrome"
    assert captured["interactive"] is True
    assert captured["tls_config"] == {"base_url": "https://jira.example.test", "auth_mode": "browser_cookie"}
    upload = next(call for name, call in FakeToolClient.calls if name == "upload_browser_session")
    assert upload["provider"] == "jira"
    assert upload["workspace_id"] == "team_alpha"
    assert upload["cookie_header"] == "SESSION=interactive"


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

    assert (
        payload
        == TeamsAuthCheckResult(
            operation="teams_auth_check", region="apac", authenticated=True, expires_in_minutes=42
        ).model_dump()
    )
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
    from memforge.genes.teams_client import AuthenticationError

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


def test_adapter_jira_watch_command_is_registered():
    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "watch", "--help"])
    assert result.exit_code == 0, result.output
    assert "--interval-seconds" in result.output


