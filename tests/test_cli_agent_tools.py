import base64
import hashlib
import json
import subprocess
import sys
from tempfile import TemporaryFile
from pathlib import Path

import click
import pytest
from click.testing import CliRunner

import memforge.main as main
from memforge.api_target import MemForgeTarget, build_target
from memforge.genes.jira_gene import JiraGene
from memforge.local_agent import folder_picker
from memforge.local_agent.runner import CloudJobLeaseLost
from memforge.local_agent.setup_results import (
    JiraAuthResult,
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
    confirmations_response: dict | None = None
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
        cls.confirmations_response = None
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
    def push_github_repo_document(self, **kwargs):
        self.calls.append(
            (
                "push_github_repo_document",
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
    def submit_local_source_confirmations(self, **kwargs):
        self.calls.append(
            (
                "submit_local_source_confirmations",
                {"api_url": self.api_url, "api_token": self.api_token, "workspace_id": self.workspace_id, **kwargs},
            )
        )
        if self.confirmations_response is not None:
            return self.confirmations_response
        return {"answered_count": len(kwargs.get("answers") or []), "failed": kwargs.get("failure_cause") is not None}
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
            origin="https://memforge-dev.cfapps.eu12.hana.ondemand.com",
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
                "top_k": 20,
                "include_private": True,
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


@pytest.mark.parametrize("command", ["search", "memory search"])
def test_search_cli_includes_the_callers_own_private_memories(monkeypatch, command):
    FakeToolClient.reset({"results": []})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        [*command.split(), "release process"],
        env={"MEMFORGE_API_URL": "https://memforge-dev.cfapps.eu12.hana.ondemand.com"},
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


@pytest.mark.parametrize("dry_run", [False, True])
def test_sources_reprocess_cli_names_documents_on_the_active_api_target(monkeypatch, dry_run):
    FakeToolClient.reset({"ok": True, "estimated_model_calls": 7})
    monkeypatch.setattr(main, "ToolClient", FakeToolClient, raising=False)

    result = CliRunner().invoke(
        cli,
        ["sources", "reprocess", "src-1", "--document", "jira-PAY-1", "--document", "jira-PAY-2",
         *(["--dry-run"] if dry_run else [])],
        env={"MEMFORGE_API_URL": "https://memforge-dev.cfapps.eu12.hana.ondemand.com", "MEMFORGE_API_TOKEN": "t"},
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["estimated_model_calls"] == 7
    assert FakeToolClient.calls == [
        (
            "reprocess_source_documents",
            {
                "api_url": "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
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
        env={"MEMFORGE_API_URL": "https://memforge-dev.cfapps.eu12.hana.ondemand.com", "MEMFORGE_API_TOKEN": "t"},
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
            "sap.prod",
            "--api-url",
            "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
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
            "sap",
            "--api-url",
            "https://memforge-dev.cfapps.eu12.hana.ondemand.com",
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


def test_target_add_accepts_cloud_origin_without_client_workspace_state(monkeypatch, tmp_path: Path):
    cli_config = tmp_path / "cli.toml"
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(cli_config))

    result = CliRunner().invoke(
        cli,
        ["target", "add", "invalid", "--api-url", "https://memforge-dev.cfapps.eu12.hana.ondemand.com"],
    )

    assert result.exit_code == 0, result.output
    assert "workspace_id" not in cli_config.read_text(encoding="utf-8")


CLOUD_TEST_ORIGIN = "https://memforge-dev.cfapps.eu12.hana.ondemand.com"


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

    result = CliRunner().invoke(cli, ["sources", "list", "--workspace-id", "payroll_agent"], env=_cloud_env())

    assert result.exit_code == 0, result.output
    assert _WorkspaceRecordingClient.selected == ["payroll_agent"]


def test_sources_list_lists_workspace_ids_when_selection_is_required(monkeypatch, tmp_path: Path):
    FakeToolClient.reset(
        {},
        list_response={
            "error": "MemForge API request failed",
            "status_code": 409,
            "code": "workspace_selection_required",
            "detail": "Select a workspace for this request.",
            "workspace_ids": ["mount_tai", "payroll_agent"],
        },
    )
    _WorkspaceRecordingClient.selected = []
    monkeypatch.setattr(main, "ToolClient", _WorkspaceRecordingClient)
    # A current-directory binding only selects the workspace for project commands.
    _bind_directory(tmp_path, tmp_path, "mount_tai")
    monkeypatch.chdir(tmp_path)

    result = CliRunner().invoke(cli, ["sources", "list"], env=_cloud_env())

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["workspace_ids"] == ["mount_tai", "payroll_agent"]
    assert payload["hint"] == "Pass --workspace-id with one of: mount_tai, payroll_agent."
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
    _bind_directory(tmp_path, project, "mount_tai")
    monkeypatch.chdir(project / "src")

    bound = CliRunner().invoke(cli, command, env=_cloud_env())
    explicit = CliRunner().invoke(cli, [*command, "--workspace-id", "payroll_agent"], env=_cloud_env())

    assert bound.exit_code == 0, bound.output
    assert explicit.exit_code == 0, explicit.output
    assert _WorkspaceRecordingClient.selected == ["mount_tai", "payroll_agent"]


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
            "origin": "https://jira.tools.sap",
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
        "workspace_ids": ["mount_tai", "payroll_agent"],
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
            "origin": "https://jira.tools.sap",
            "status": "active",
            "principal_name": "Rose H",
            "browser": "chrome",
        }
    )
    selected = []
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: selected.append(workspace_id) or stub)
    result = CliRunner().invoke(
        cli,
        ["adapter", "auth", "jira", "status", "--base-url", "https://jira.tools.sap", "--workspace-id", "ws-a"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "active"
    assert payload["principal_name"] == "Rose H"
    assert stub.calls == [("get_jira_session", "https://jira.tools.sap")]
    assert selected == ["ws-a"]


def test_adapter_jira_status_missing_session_is_not_an_error(monkeypatch):
    stub = _StubClient(get_jira_session={"provider": "jira", "origin": "https://jira.tools.sap", "status": "missing"})
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: stub)
    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "status", "--base-url", "https://jira.tools.sap"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["status"] == "missing"


def test_adapter_jira_status_reports_workspace_selection_error(monkeypatch):
    stub = _StubClient(get_jira_session=_workspace_selection_response())
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: stub)

    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "status", "--base-url", "https://jira.tools.sap"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["code"] == "workspace_selection_required"
    assert payload["workspace_ids"] == ["mount_tai", "payroll_agent"]
    assert payload["hint"] == "Pass --workspace-id with one of: mount_tai, payroll_agent."
    assert "principal_changed" not in result.output


def test_adapter_jira_refresh_captures_and_uploads(monkeypatch):
    captured = {}
    _fake_jira_capture(monkeypatch, captured)
    stub = _StubClient(upload_jira_session={"provider": "jira", "origin": "https://jira.tools.sap", "status": "active"})
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: stub)

    result = CliRunner().invoke(
        cli,
        ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.tools.sap", "--browser", "chrome"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload == {
        "origin": "https://jira.tools.sap",
        "results": [
            {
                "workspace_id": "ws-a",
                "status": "uploaded",
                "session": {"provider": "jira", "origin": "https://jira.tools.sap", "status": "active"},
            }
        ],
    }
    assert captured == {"base_url": "https://jira.tools.sap", "browser": "chrome"}
    uploads = [call for call in stub.calls if call[0] == "upload_jira_session"]
    assert [call[0] for call in stub.calls[:2]] == ["list_workspaces", "for_workspace"]
    assert uploads[0][1]["cookie_header"] == "SESSION=x"
    assert uploads[0][1]["base_url"] == "https://jira.tools.sap"


def test_adapter_jira_refresh_uploads_only_to_explicit_workspace(monkeypatch):
    _fake_jira_capture(monkeypatch)
    stub = _StubClient(upload_jira_session={"status": "active"})
    selected = []
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: selected.append(workspace_id) or stub)

    result = CliRunner().invoke(
        cli,
        ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.tools.sap", "--workspace-id", "ws-b"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["results"][0]["workspace_id"] == "ws-b"
    assert selected == ["ws-b"]
    assert [call[0] for call in stub.calls] == ["upload_jira_session"]


def test_adapter_jira_refresh_fans_out_to_workspaces_with_jira_source(monkeypatch):
    _fake_jira_capture(monkeypatch)
    configured = {"origins": [{"origin": "https://jira.tools.sap", "configured": True}]}
    accepted = _StubClient(list_jira_origins=configured, upload_jira_session={"status": "active"})
    changed = _StubClient(list_jira_origins=configured, upload_jira_session=_jira_principal_changed_response())
    unrelated = _StubClient(list_jira_origins={"origins": [{"origin": "https://jira.other", "configured": True}]})
    account = _StubClient(
        list_workspaces={
            "workspaces": [
                {"workspace_id": "mount_tai", "selectable": True},
                {"workspace_id": "payroll_agent", "selectable": True},
                {"workspace_id": "sandbox", "selectable": True},
                {"workspace_id": "suspended", "selectable": False},
            ]
        },
        workspaces={"mount_tai": accepted, "payroll_agent": changed, "sandbox": unrelated},
    )
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: account)

    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.tools.sap"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "jira_session_upload_incomplete"
    assert payload["results"] == [
        {"workspace_id": "mount_tai", "status": "uploaded", "session": {"status": "active"}},
        {
            "workspace_id": "payroll_agent",
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
                {"workspace_id": "mount_tai", "selectable": True},
                {"workspace_id": "payroll_agent", "selectable": True},
            ]
        },
    )
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: account)

    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.tools.sap"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "jira_source_not_found"
    assert payload["workspace_ids"] == ["mount_tai", "payroll_agent"]
    assert "--workspace-id" in payload["detail"]
    assert not any(call[0] == "upload_jira_session" for call in account.calls)


def test_adapter_jira_refresh_reports_non_principal_conflict_as_failure(monkeypatch):
    _fake_jira_capture(monkeypatch)
    stub = _StubClient(upload_jira_session=_workspace_selection_response())
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: stub)

    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.tools.sap"])

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
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: _StubClient())

    result = CliRunner().invoke(
        cli,
        ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.tools.sap", "--browser", "netscape"],
    )
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["error"] == "auth_failed"
    assert "Unsupported browser" in payload["detail"]


def test_adapter_jira_refresh_principal_change_returns_json_error(monkeypatch):
    _fake_jira_capture(monkeypatch)
    stub = _StubClient(upload_jira_session=_jira_principal_changed_response())
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: stub)

    result = CliRunner().invoke(cli, ["adapter", "auth", "jira", "refresh", "--base-url", "https://jira.tools.sap"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["origin"] == "https://jira.tools.sap"
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
        list_jira_origins={"origins": [{"origin": "https://jira.tools.sap", "status": "active"}]},
        forget_jira_session={"ok": True, "origin": "https://jira.tools.sap", "forgotten": True},
    )
    monkeypatch.setattr(main, "_tool_client", lambda ctx, workspace_id: stub)
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
    assert kwargs["env"]["GH_HOST"] == "github.wdf.sap.corp"
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

    progress: list[dict] = []
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
        report_progress=progress.append,
    )

    assert payload["operation"] == "github_repo_sync"
    assert payload["source_id"] == "src-from-cloud"
    assert payload["source_sync_run_id"] == "run-test"
    assert payload["counts"] == {
        "selected": 1,
        "reused": 0,
        "fetched": 1,
        "pushed": 1,
        "failed": 0,
    }
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
    assert [(item["phase"], item.get("progress", {}).get("completed")) for item in progress] == [
        ("discovering", 1),
        ("fetching", 0),
        ("fetching", 1),
        ("uploading", 0),
        ("uploading", 1),
    ]


def test_local_agent_cloud_github_sync_streams_selected_image_as_artifact(monkeypatch):
    image_bytes = b"\x89PNG\r\n\x1a\nlocal-agent-image"
    def fake_github_image_run(cmd, *args, **kwargs):
        assert cmd[:2] == ["gh", "api"]
        endpoint = cmd[2]
        if endpoint.endswith("/commits/main"):
            payload = {"sha": "commit-main", "commit": {"tree": {"sha": "tree-main"}}}
        elif "/commits?" in endpoint:
            payload = _github_file_commits_listing()
        elif "/git/trees/tree-main" in endpoint:
            payload = {
                "truncated": False,
                "tree": [
                    {
                        "path": "Payroll Processing V2/images/architecture.png",
                        "type": "blob",
                        "mode": "100644",
                        "sha": "image-sha",
                        "size": len(image_bytes),
                    }
                ],
            }
        elif endpoint.endswith("/git/blobs/image-sha"):
            payload = {
                "sha": "image-sha",
                "content": base64.b64encode(image_bytes).decode(),
                "encoding": "base64",
                "size": len(image_bytes),
            }
        else:
            raise AssertionError(f"unexpected gh endpoint: {endpoint}")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "github-image-doc", "document_hash": "hash"})
    monkeypatch.setattr(main.subprocess, "run", fake_github_image_run)

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-image-sync",
            "attempt_count": 1,
            "workspace_id": "ws-from-job-payload",
            "operation": "github_repo_sync",
            "source_id": "src-from-cloud",
            "payload": {
                "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                "ref": "main",
                "include_paths": ["Payroll Processing V2/images"],
                "include_extensions": ["png"],
                "limit": 1,
            },
        },
        _cloud_test_client(),
    )

    assert payload["counts"]["failed"] == 0
    [artifact_call] = [call for call in FakeToolClient.calls if call[0] == "push_source_artifact"]
    assert artifact_call[1]["content"] == image_bytes
    assert artifact_call[1]["media_type"] == "image/png"
    [package_call] = [call for call in FakeToolClient.calls if call[0] == "push_github_repo_document"]
    assert package_call[1]["markdown_body"] == ""
    assert package_call[1]["artifact_source_unit_key"] == ("Payroll Processing V2/images/architecture.png")
    assert package_call[1]["artifact_input_hashes"] == ["a" * 64]


def test_local_agent_cloud_github_rejects_oversized_image_before_blob_download(monkeypatch):
    def fake_github_oversized_run(cmd, *args, **kwargs):
        assert cmd[:2] == ["gh", "api"]
        endpoint = cmd[2]
        if endpoint.endswith("/commits/main"):
            payload = {"sha": "commit-main", "commit": {"tree": {"sha": "tree-main"}}}
        elif "/commits?" in endpoint:
            payload = _github_file_commits_listing()
        elif "/git/trees/tree-main" in endpoint:
            payload = {
                "truncated": False,
                "tree": [
                    {
                        "path": "docs/oversized.png",
                        "type": "blob",
                        "mode": "100644",
                        "sha": "oversized-image-sha",
                        "size": main.MAX_SOURCE_ARTIFACT_STORAGE_BYTES + 1,
                    }
                ],
            }
        else:
            raise AssertionError(f"oversized blob must not be downloaded: {endpoint}")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({})
    monkeypatch.setattr(main.subprocess, "run", fake_github_oversized_run)

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-oversized-image-sync",
            "attempt_count": 1,
            "workspace_id": "ws-from-job-payload",
            "operation": "github_repo_sync",
            "source_id": "src-from-cloud",
            "payload": {
                "repo_url": "https://github.wdf.sap.corp/org/repo",
                "ref": "main",
                "include_extensions": ["png"],
                "limit": 1,
            },
        },
        _cloud_test_client(),
    )

    assert payload["counts"]["failed"] == 1
    assert payload["failed"][0]["error"] == "Source Artifact exceeds storage limit"
    assert not [call for call in FakeToolClient.calls if call[0] == "push_source_artifact"]
    assert not [call for call in FakeToolClient.calls if call[0] == "push_github_repo_document"]


def test_local_agent_cloud_github_noop_uses_manifest_without_reading_or_uploading_bodies(monkeypatch):
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({})
    FakeToolClient.snapshot_plan_response = {"required_doc_ids": [], "reused_count": 555}
    entries = [
        {
            "relative_path": f"docs/file-{index:03d}.md",
            "blob_sha": f"blob-{index:03d}",
            "file_mode": "100644",
            "content_type": "text/markdown",
        }
        for index in range(555)
    ]
    monkeypatch.setattr(
        main,
        "_preview_github_profile",
        lambda *_args, **_kwargs: {"items": entries},
    )
    monkeypatch.setattr(
        main,
        "_github_blob",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unchanged provider bodies must not be read")
        ),
    )

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-noop",
            "attempt_count": 1,
            "workspace_id": "ws-from-job-payload",
            "operation": "github_repo_sync",
            "source_id": "src-from-cloud",
            "payload": {
                "repo_url": "https://github.wdf.sap.corp/example/public-howtos",
                "ref": "main",
                "include_extensions": ["md"],
            },
        },
        _cloud_test_client(),
    )

    assert payload["counts"] == {
        "selected": 555,
        "reused": 555,
        "fetched": 0,
        "pushed": 0,
        "failed": 0,
    }
    assert not [call for call in FakeToolClient.calls if call[0] == "push_github_repo_document"]
    [plan_call] = [call for call in FakeToolClient.calls if call[0] == "prepare_local_source_snapshot"]
    assert len(plan_call[1]["items"]) == 555
    assert plan_call[1]["coverage"] == "complete_snapshot"
    assert {item["change_kind"] for item in plan_call[1]["items"]} == {"upsert"}
    assert payload["metrics"]["manifest_items"] == 555
    assert payload["metrics"]["full_bodies_read"] == 0
    assert payload["metrics"]["full_bodies_uploaded"] == 0
    assert payload["metrics"]["bytes_uploaded"] == 0
    assert payload["metrics"]["fallback_reason"] is None


def test_local_agent_cloud_github_sync_pins_tree_and_body_to_one_commit(monkeypatch):
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "github-repo-doc", "document_hash": "hash"})
    endpoints: list[str] = []
    def pinned_github_run(cmd, *args, **kwargs):
        assert cmd[:2] == ["gh", "api"]
        endpoint = cmd[2]
        endpoints.append(endpoint)
        if endpoint.endswith("/commits/main"):
            payload = {"sha": "commit-1", "commit": {"tree": {"sha": "tree-1"}}}
        elif "/commits?" in endpoint:
            payload = _github_file_commits_listing()
        elif "/git/trees/tree-1?recursive=1" in endpoint:
            payload = {
                "truncated": False,
                "tree": [
                    {"path": "README.md", "type": "blob", "mode": "100644", "sha": _git_blob_sha(b"# Pinned body"), "size": len(b"# Pinned body")}
                ],
            }
        elif endpoint.endswith("/git/blobs/" + _git_blob_sha(b"# Pinned body")):
            raw = b"# Pinned body"
            payload = {
                "sha": _git_blob_sha(b"# Pinned body"),
                "content": base64.b64encode(raw).decode(),
                "encoding": "base64",
                "size": len(raw),
            }
        else:
            raise AssertionError(f"unexpected gh endpoint: {endpoint}")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(main.subprocess, "run", pinned_github_run)

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-pinned",
            "attempt_count": 1,
            "workspace_id": "workspace-a",
            "operation": "github_repo_sync",
            "source_id": "src-github",
            "payload": {
                "repo_url": "https://github.wdf.sap.corp/example/public-howtos",
                "ref": "main",
                "include_extensions": ["md"],
            },
        },
        _cloud_test_client(),
    )

    assert payload["counts"]["pushed"] == 1
    assert endpoints == [
        "repos/example/public-howtos/commits/main",
        "repos/example/public-howtos/git/trees/tree-1?recursive=1",
        "repos/example/public-howtos/git/blobs/" + _git_blob_sha(b"# Pinned body"),
        "repos/example/public-howtos/commits?sha=commit-1&path=README.md&per_page=1",
    ]
    [push_call] = [call for call in FakeToolClient.calls if call[0] == "push_github_repo_document"]
    assert push_call[1]["blob_sha"] == _git_blob_sha(b"# Pinned body")
    # The file's source time is its latest commit at the pinned collection commit.
    assert push_call[1]["source_updated_at"] == "2026-05-01T10:00:00+00:00"


@pytest.mark.parametrize("failure", [None, "decode", "upload", "lease"])
def test_local_agent_github_uploads_each_body_before_fetching_the_next(monkeypatch, failure):
    events = []
    manifest = []

    class TransferClient(FakeToolClient):
        def prepare_local_source_snapshot(self, **kwargs):
            manifest.extend(kwargs["items"])
            events.append("manifest")
            return super().prepare_local_source_snapshot(**kwargs)

        def push_github_repo_document(self, **kwargs):
            events.append("upload:" + kwargs["relative_path"])
            if kwargs["relative_path"] == "1.md":
                if failure == "upload":
                    return {"error": "temporary upload failure", "status_code": 503}
                if failure == "lease":
                    return {
                        "error": "lease rejected",
                        "status_code": 409,
                        "detail": "local_agent_lease_not_current",
                    }
            return {"doc_id": kwargs["relative_path"], "document_hash": "receipt"}

        def start_source_processing(self, **kwargs):
            events.append("process")
            return super().start_source_processing(**kwargs)

    bodies = [b"T0xt", b"\xff" if failure == "decode" else b"T1xt", b"T2xt"]
    def github_run(cmd, *args, **kwargs):
        endpoint = cmd[2]
        if endpoint.endswith("/commits/main"):
            data = {"sha": "commit-1", "commit": {"tree": {"sha": "tree-1"}}}
        elif "/commits?" in endpoint:
            data = _github_file_commits_listing()
        elif "/git/trees/tree-1" in endpoint:
            data = {
                "truncated": False,
                "tree": [{"path": f"{n}.md", "type": "blob", "mode": "100644", "sha": _git_blob_sha(bodies[n]), "size": len(bodies[n])} for n in range(3)],
            }
        else:
            number = str([_git_blob_sha(body) for body in bodies].index(endpoint.rsplit("/", 1)[1]))
            events.append("read:" + number + ".md")
            raw = bodies[int(number)]
            data = {
                "sha": _git_blob_sha(raw),
                "content": base64.b64encode(raw).decode(),
                "encoding": "base64",
                "size": len(raw),
            }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(data), stderr="")

    monkeypatch.setattr(main, "ToolClient", TransferClient)
    FakeToolClient.reset({})
    monkeypatch.setattr(main.subprocess, "run", github_run)
    job = {
        "job_id": "laj-bounded-bodies",
        "attempt_count": 1,
        "workspace_id": "workspace-a",
        "operation": "github_repo_sync",
        "source_id": "src-github",
        "payload": {"repo_url": "https://github.example.test/org/repo", "ref": "main", "include_extensions": ["md"]},
    }
    client = TransferClient(target=_cloud_test_client().target)
    if failure == "lease":
        with pytest.raises(CloudJobLeaseLost):
            main._run_cloud_local_agent_job(job, client)
        assert events == ["manifest", "read:0.md", "upload:0.md", "read:1.md", "upload:1.md"]
        return

    payload = main._run_cloud_local_agent_job(job, client)
    expected = ["manifest", "read:0.md", "upload:0.md", "read:1.md"]
    if failure != "decode":
        expected.append("upload:1.md")
    expected += ["read:2.md", "upload:2.md"]
    if failure is None:
        expected.append("process")
    assert events == expected
    assert len(manifest) == 3
    assert payload["counts"] == {
        "selected": 3,
        "reused": 0,
        "fetched": 2 if failure == "decode" else 3,
        "pushed": 3 if failure is None else 2,
        "failed": 0 if failure is None else 1,
    }
    assert (
        payload["metrics"]["bytes_read"] == payload["metrics"]["bytes_uploaded"] == (8 if failure == "decode" else 12)
    )


def test_github_transfer_body_memory_does_not_grow_with_file_count(monkeypatch):
    import gc
    import tracemalloc

    file_count = 2
    body_bytes = 512 * 1024

    class DiscardingClient(FakeToolClient):
        def push_github_repo_document(self, **kwargs):
            assert len(kwargs["markdown_body"]) == body_bytes
            return {"doc_id": kwargs["relative_path"], "document_hash": "receipt"}
    def github_run(cmd, *args, **kwargs):
        endpoint = cmd[2]
        if endpoint.endswith("/commits/main"):
            data = {"sha": "commit-1", "commit": {"tree": {"sha": "tree-1"}}}
        elif "/commits?" in endpoint:
            data = _github_file_commits_listing()
        elif "/git/trees/tree-1" in endpoint:
            data = {
                "truncated": False,
                "tree": [
                    {"path": f"{n:02}.md", "type": "blob", "mode": "100644", "sha": _git_blob_sha(b"x" * body_bytes), "size": body_bytes}
                    for n in range(file_count)
                ],
            }
        else:
            data = {
                "sha": endpoint.rsplit("/", 1)[1],
                "content": base64.b64encode(b"x" * body_bytes).decode(),
                "encoding": "base64",
                "size": body_bytes,
            }
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(data), stderr="")

    monkeypatch.setattr(main.subprocess, "run", github_run)
    client = DiscardingClient(target=_cloud_test_client().target)
    job = {
        "job_id": "laj-memory",
        "attempt_count": 1,
        "workspace_id": "workspace-a",
        "operation": "github_repo_sync",
        "source_id": "src-github",
        "payload": {"repo_url": "https://github.example.test/org/repo", "ref": "main"},
    }
    # Warm imports before comparing the same actual collection path at two sizes.
    FakeToolClient.reset({})
    main._run_cloud_local_agent_job(job, client)
    peaks = []
    for file_count in (2, 12):
        FakeToolClient.reset({})
        gc.collect()
        tracemalloc.start()
        try:
            result = main._run_cloud_local_agent_job(job, client)
            peaks.append(tracemalloc.get_traced_memory()[1])
        finally:
            tracemalloc.stop()
        assert result["counts"]["pushed"] == file_count
    # Ten extra bodies must not be retained; leave slack for manifest/receipts.
    assert peaks[1] < peaks[0] + 2 * body_bytes, peaks


def test_local_agent_cloud_github_sync_rejects_unproven_tree_completeness(monkeypatch):
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({})
    def incomplete_tree_run(cmd, *args, **kwargs):
        endpoint = cmd[2]
        if endpoint.endswith("/commits/main"):
            payload = {"sha": "commit-1", "commit": {"tree": {"sha": "tree-1"}}}
        elif "/commits?" in endpoint:
            payload = _github_file_commits_listing()
        elif "/git/trees/tree-1?recursive=1" in endpoint:
            payload = {"tree": []}
        else:
            raise AssertionError(f"unexpected gh endpoint: {endpoint}")
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr(main.subprocess, "run", incomplete_tree_run)

    with pytest.raises(click.ClickException, match="did not prove complete"):
        main._run_cloud_local_agent_job(
            {
                "job_id": "laj-incomplete",
                "attempt_count": 1,
                "workspace_id": "workspace-a",
                "operation": "github_repo_sync",
                "source_id": "src-github",
                "payload": {
                    "repo_url": "https://github.wdf.sap.corp/example/public-howtos",
                    "ref": "main",
                },
            },
            _cloud_test_client(),
        )


def test_local_agent_cloud_github_sync_reports_failed_paths_for_retry_diagnostics(monkeypatch):
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "github-repo-doc", "document_hash": "hash"})
    monkeypatch.setattr(main.subprocess, "run", _fake_github_remote_run)
    def github_blob(_repo, _blob_sha, relative_path, *, size):
        if relative_path.endswith("Überblick.md"):
            raise click.ClickException("temporary GitHub API failure")
        return b"# Durable document\n\nBody"

    monkeypatch.setattr(main, "_github_text_blob", github_blob)

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-sync-failed-path",
            "attempt_count": 1,
            "workspace_id": "ws-from-job-payload",
            "operation": "github_repo_sync",
            "source_id": "src-from-cloud",
            "payload": {
                "repo_url": "https://github.wdf.sap.corp/example/public-howtos",
                "ref": "main",
                "include_extensions": ["md"],
                "limit": 2,
            },
        },
        _cloud_test_client(),
    )

    assert payload["counts"] == {
        "selected": 3,
        "reused": 0,
        "fetched": 2,
        "pushed": 2,
        "failed": 1,
    }
    assert payload["retryable"] is True
    assert payload["error"] == (
        "1 document(s) failed to push: Payroll Processing V2/Überblick.md: temporary GitHub API failure"
    )
    assert not [call for call in FakeToolClient.calls if call[0] == "start_source_processing"]


def test_local_agent_cloud_github_sync_stops_after_lease_rejection(monkeypatch):
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset(
        {
            "error": "MemForge API request failed",
            "status_code": 409,
            "detail": "local_agent_lease_not_current",
        }
    )
    monkeypatch.setattr(main.subprocess, "run", _fake_github_remote_run)

    with pytest.raises(CloudJobLeaseLost, match="local_agent_lease_not_current"):
        main._run_cloud_local_agent_job(
            {
                "job_id": "laj-stale",
                "attempt_count": 1,
                "workspace_id": "ws-from-job-payload",
                "operation": "github_repo_sync",
                "source_id": "src-from-cloud",
                "payload": {
                    "repo_url": "https://github.wdf.sap.corp/nextgenpayroll-matterhorn/architecture",
                    "ref": "main",
                    "include_paths": ["Payroll Processing V2"],
                    "include_extensions": ["md"],
                    "limit": 2,
                },
            },
            _cloud_test_client(),
        )

    push_calls = [call for call in FakeToolClient.calls if call[0] == "push_github_repo_document"]
    assert len(push_calls) == 1
    assert not [call for call in FakeToolClient.calls if call[0] == "start_source_processing"]


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


def test_local_agent_cloud_jira_sync_uses_gene_and_pushes_packages(monkeypatch):
    from contextlib import asynccontextmanager
    from datetime import datetime, timezone

    from memforge.auth import jira_capture
    from memforge.models import ContentItem, RawContent
    from memforge.source_artifacts import (
        RawSourceArtifact,
        SourceArtifactDownload,
    )

    image = b"\x89PNG\r\n\x1a\njira-comment-image"
    built: list = []

    class FakeJiraGene:
        config_schema = JiraGene.config_schema

        def __init__(self, config, source_id):
            self.config = config
            self.source_id = source_id
            self.closed = False
            built.append(self)

        async def close(self):
            self.closed = True

        async def authenticate(self):
            assert self.config["auth_mode"] == "browser_cookie"
            assert self.config["jira_cookie"] == "JSESSIONID=local"
            assert self.config["sync_mode"] == "cloud"
            assert "local_agent_documents_dir" not in self.config
            assert "pat" not in self.config

        async def discover_inventory(self):
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
                body=json.dumps(
                    {
                        "key": "PAY-1",
                        "fields": {
                            "summary": "Create daemon source support",
                            "updated": "v1",
                        },
                    }
                ).encode(),
                content_type="application/json",
                artifacts=(
                    RawSourceArtifact(
                        provider_key="attachment-1",
                        parent_observation_type="comment",
                        parent_provider_key="comment-1",
                        provider_revision="attachment-revision-1",
                        filename="screenshot.png",
                        media_type="image/png",
                        declared_size_bytes=len(image),
                        locator={"request_path": "/attachment/1"},
                    ),
                ),
            )

        @asynccontextmanager
        async def open_source_artifact(self, artifact):
            async def chunks():
                yield image

            yield SourceArtifactDownload(
                chunks=chunks(),
                media_type="image/png",
                content_length=len(image),
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
    assert payload["counts"] == {"selected": 1, "reused": 0, "pushed": 1, "failed": 0}
    assert payload["source_sync_run_id"] == "run-test"
    assert [gene.closed for gene in built] == [True]
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
    [artifact_call] = [call for call in FakeToolClient.calls if call[0] == "push_source_artifact"]
    assert artifact_call[1]["source_unit_key"] == "PAY-1"
    assert artifact_call[1]["provider_key"] == "attachment-1"
    assert artifact_call[1]["parent_observation_type"] == "comment"
    assert artifact_call[1]["parent_provider_key"] == "comment-1"
    assert artifact_call[1]["content"] == image
    assert kwargs["artifact_source_unit_key"] == "PAY-1"
    assert kwargs["artifact_input_hashes"] == ["a" * 64]
    assert "markdown_body" not in kwargs
    assert "process_now" not in kwargs
    process_calls = [call for call in FakeToolClient.calls if call[0] == "start_source_processing"]
    assert len(process_calls) == 1
    assert process_calls[0][1]["source_id"] == "src-jira"
    assert process_calls[0][1]["sync_snapshot_id"] == "laj-jira-sync:attempt:3"


def test_local_agent_cloud_jira_sync_rejects_issue_changed_after_inventory(monkeypatch):
    import asyncio
    from datetime import datetime, timezone

    from memforge.models import ContentItem, RawContent

    built: list = []

    class ConcurrentUpdateJiraGene:
        config_schema = JiraGene.config_schema

        def __init__(self, config, source_id):
            self.config = config
            self.source_id = source_id
            self.closed = False
            built.append(self)

        async def close(self):
            self.closed = True

        async def authenticate(self):
            return None

        async def discover_inventory(self):
            yield ContentItem(
                item_id="jira-PAY-1",
                title="PAY-1 title",
                source_url="https://jira.example.test/browse/PAY-1",
                last_modified=datetime(2026, 7, 7, tzinfo=timezone.utc),
                content_type="application/json",
                version="inventory-v1",
                extra={"issue_key": "PAY-1"},
            )

        async def fetch(self, item):
            return RawContent(
                item=item,
                body=json.dumps(
                    {
                        "key": "PAY-1",
                        "fields": {
                            "summary": "Changed after inventory",
                            "updated": "hydrated-v2",
                        },
                    }
                ).encode(),
                content_type="application/json",
            )

    import memforge.genes.jira_gene as jira_gene

    monkeypatch.setattr(jira_gene, "JiraGene", ConcurrentUpdateJiraGene)
    FakeToolClient.reset({})

    with pytest.raises(RuntimeError, match="changed during materialization; retry inventory"):
        asyncio.run(
            main._collect_jira_documents_from_cloud_job(
                {
                    "payload": {
                        "base_url": "https://jira.example.test",
                        "auth_mode": "browser_cookie",
                        "projects": ["PAY"],
                    }
                },
                source_id="src-jira",
                limit=0,
                jira_cookie="JSESSIONID=local",
                client=_cloud_test_client(),
                sync_snapshot_id="job-jira:attempt:1",
                local_agent_job_id="job-jira",
                local_agent_attempt_count=1,
            )
        )
    assert not [call for call in FakeToolClient.calls if call[0] == "push_jira_package"]
    assert [gene.closed for gene in built] == [True]


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


def test_local_agent_cloud_jira_sync_refuses_an_odd_boolean_before_capturing_a_session(monkeypatch):
    from memforge.auth import jira_capture

    captures: list[str] = []

    async def fake_capture(base_url, *, browser=None, tls_config=None):
        captures.append(base_url)
        raise AssertionError("a configuration no run can use must not capture a browser session")

    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    FakeToolClient.reset({"doc_id": "jira-doc", "document_hash": "hash"})

    payload = main._run_cloud_local_agent_job(
        _jira_sync_job(include_comments="sometimes"), _cloud_test_client(), browser="chrome"
    )

    assert payload["error_type"] == "SourceConfigurationError"
    assert "include_comments" in payload["error"]
    assert payload["retryable"] is False
    assert captures == []
    assert not [call for call in FakeToolClient.calls if call[0] == "upload_jira_session"]


def test_local_agent_cloud_jira_sync_does_not_retry_a_rejected_jql(monkeypatch):
    from memforge.auth import jira_capture
    import memforge.genes.jira_gene as jira_gene
    from memforge.genes.jira_gene import JiraQueryRejected

    async def fake_capture(base_url, *, browser=None, tls_config=None):
        return jira_capture.JiraCaptureResult(
            origin=base_url, cookie_header="JSESSIONID=local", browser=browser, principal={"name": "tester"}
        )

    class RejectingJiraGene:
        config_schema = JiraGene.config_schema

        def __init__(self, config, source_id):
            self.closed = False

        async def authenticate(self):
            return None

        async def discover_inventory(self):
            raise JiraQueryRejected("Jira rejected the JQL (HTTP 400): project 'GONE' does not exist")
            yield

        async def close(self):
            self.closed = True

    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    monkeypatch.setattr(jira_gene, "JiraGene", RejectingJiraGene)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({"doc_id": "jira-doc", "document_hash": "hash"})

    payload = main._run_cloud_local_agent_job(_jira_sync_job(), _cloud_test_client(), browser="chrome")

    assert payload["error_type"] == "JiraQueryRejected"
    assert payload["retryable"] is False


def test_local_agent_cloud_jira_noop_uses_inventory_without_hydrating_issues(monkeypatch):
    from datetime import datetime, timezone

    from memforge.auth import jira_capture
    from memforge.models import ContentItem

    class FakeJiraGene:
        config_schema = JiraGene.config_schema

        def __init__(self, config, source_id):
            self.config = config
            self.source_id = source_id
            self.closed = False

        async def close(self):
            self.closed = True

        async def authenticate(self):
            return None

        async def discover_inventory(self):
            for key in ("PAY-1", "PAY-2"):
                yield ContentItem(
                    item_id=f"jira-{key}",
                    title=f"{key} title",
                    source_url=f"https://jira.example.test/browse/{key}",
                    last_modified=datetime(2026, 7, 7, tzinfo=timezone.utc),
                    content_type="application/json",
                    version=f"revision-{key}",
                    extra={"issue_key": key},
                )

        async def fetch(self, item):
            raise AssertionError(f"unchanged issue must not be hydrated: {item.item_id}")

    async def fake_capture(base_url, *, browser=None, tls_config=None):
        return jira_capture.JiraCaptureResult(
            origin=base_url,
            cookie_header="JSESSIONID=local",
            browser=browser,
            principal={"name": "tester"},
        )

    import memforge.genes.jira_gene as jira_gene

    monkeypatch.setattr(jira_capture, "capture_and_prevalidate", fake_capture)
    monkeypatch.setattr(jira_gene, "JiraGene", FakeJiraGene)
    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    FakeToolClient.reset({})
    FakeToolClient.snapshot_plan_response = {"required_doc_ids": [], "reused_count": 2}

    payload = main._run_cloud_local_agent_job(
        {
            "job_id": "laj-jira-noop",
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

    [plan_call] = [call for call in FakeToolClient.calls if call[0] == "prepare_local_source_snapshot"]
    assert plan_call[1]["coverage"] == "complete_snapshot"
    assert len(plan_call[1]["items"]) == 2
    assert not [call for call in FakeToolClient.calls if call[0] == "push_jira_package"]
    assert payload["counts"] == {"selected": 2, "reused": 2, "pushed": 0, "failed": 0}


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


def _call_names() -> list[str]:
    return [
        name
        for name, _kwargs in FakeToolClient.calls
        if name in {"prepare_local_source_snapshot", "submit_local_source_confirmations", "start_source_processing"}
    ]


def test_local_agent_cloud_jira_lists_each_excluded_scope_with_its_own_jql(monkeypatch):
    from memforge.local_agent.document_identity import build_jira_doc_id

    events: list[str] = []
    FakeToolClient.reset({})
    gene_cls = _confirming_jira_gene(events)
    scope = {"base_url": "https://jira.example.test", "auth_mode": "browser_cookie", "query_mode": "advanced"}

    payload = _run_confirming_jira_job(
        monkeypatch,
        gene_cls,
        projection_scope_transition={"id": "t1", "previous_scope": {}, "target_scope": {}},
        excluded_scopes=[
            {"scope_key": "scope-old", "config": {**scope, "jql": "project in (PAY, OPS)"}},
            {"scope_key": "scope-rejected", "config": {**scope, "jql": "project = GONE"}},
            {"scope_key": "scope-unreachable", "config": {**scope, "jql": "project = SLOW"}},
        ],
    )

    assert payload.get("error") is None, payload
    [(_name, planned)] = [call for call in FakeToolClient.calls if call[0] == "prepare_local_source_snapshot"]
    # Each issue under this Source's device Document id; a rejected JQL is
    # reported; a scope that could not be reached is left out.
    # PAY-1 is in the collection and in the excluded scope under the same Document id.
    assert build_jira_doc_id(source_id="src-jira", issue_key="PAY-1") in {item["doc_id"] for item in planned["items"]}
    assert planned["excluded_listings"] == [
        {
            "scope_key": "scope-old",
            "doc_ids": sorted(build_jira_doc_id(source_id="src-jira", issue_key=key) for key in ("PAY-1", "PAY-9")),
        },
        {
            "scope_key": "scope-rejected",
            "rejection": "JiraQueryRejected: Jira rejected the JQL (HTTP 400): project 'GONE' does not exist",
        },
    ]
    assert [config.get("jql") for config in gene_cls.configs] == [
        None,
        "project in (PAY, OPS)",
        "project = GONE",
        "project = SLOW",
    ]
    assert all(
        "excluded_scopes" not in config and "projection_scope_transition" not in config for config in gene_cls.configs
    )
    # Every session, the collection's and each excluded scope's, is closed.
    assert events.count("closed") == 4


def test_local_agent_cloud_jira_reports_an_excluded_scope_over_the_manifest_bound_as_rejected(monkeypatch):
    events: list[str] = []
    FakeToolClient.reset({})
    # The fake scope lists two issues; a manifest here holds one.
    monkeypatch.setattr(main, "SNAPSHOT_MANIFEST_MAX_ITEMS", 1)
    scope = {"base_url": "https://jira.example.test", "auth_mode": "browser_cookie", "query_mode": "advanced"}

    payload = _run_confirming_jira_job(
        monkeypatch,
        _confirming_jira_gene(events),
        excluded_scopes=[{"scope_key": "scope-old", "config": {**scope, "jql": "project in (PAY, OPS)"}}],
    )

    assert payload.get("error") is None, payload
    [(_name, planned)] = [call for call in FakeToolClient.calls if call[0] == "prepare_local_source_snapshot"]
    assert planned["excluded_listings"] == [
        {"scope_key": "scope-old", "rejection": "The excluded scope lists 2 issues, more than the 1 a manifest holds"}
    ]


def test_local_agent_cloud_jira_confirms_what_its_manifest_left_out_before_processing(monkeypatch):
    events: list[str] = []
    FakeToolClient.reset({})

    payload = _run_confirming_jira_job(monkeypatch, _confirming_jira_gene(events))

    assert payload.get("error") is None, payload
    assert _call_names() == [
        "prepare_local_source_snapshot",
        "submit_local_source_confirmations",
        "start_source_processing",
    ]
    [(_name, submitted)] = [call for call in FakeToolClient.calls if call[0] == "submit_local_source_confirmations"]
    assert submitted["manifest_sha256"] == "d" * 64
    assert submitted["sync_snapshot_id"] == "laj-jira-confirm:attempt:2"
    assert (submitted["local_agent_job_id"], submitted["local_agent_attempt_count"]) == ("laj-jira-confirm", 2)
    assert submitted["answers"] == [
        {"doc_id": "jira-src-pay-7", "provider_key": "100007", "answer": "absent"},
        {"doc_id": "jira-src-pay-8", "provider_key": "PAY-8", "answer": "exists"},
    ]
    assert submitted["failure_cause"] is None
    # The Gene confirms with its own session, and closes it only afterwards.
    assert events == ["confirmed 100007, PAY-8", "closed"]


def test_local_agent_cloud_jira_reports_a_confirmation_it_could_not_make(monkeypatch):
    events: list[str] = []
    FakeToolClient.reset({})

    payload = _run_confirming_jira_job(
        monkeypatch, _confirming_jira_gene(events, confirm_error=RuntimeError("HTTP 401 from Jira"))
    )

    assert payload.get("error") is None, payload
    [(_name, submitted)] = [call for call in FakeToolClient.calls if call[0] == "submit_local_source_confirmations"]
    assert submitted["answers"] == []
    assert submitted["failure_cause"] == "RuntimeError: HTTP 401 from Jira"
    assert _call_names()[-1] == "start_source_processing"
    assert events == ["confirmed 100007, PAY-8", "closed"]


def test_local_agent_cloud_jira_reports_an_issue_it_cannot_rebuild_as_a_failed_confirmation(monkeypatch):
    events: list[str] = []
    FakeToolClient.reset({})

    payload = _run_confirming_jira_job(
        monkeypatch,
        _confirming_jira_gene(
            events, rebuild_error=ValueError("Jira issue reference '1#x' is neither an issue id nor an issue key")
        ),
    )

    assert payload.get("error") is None, payload
    [(_name, submitted)] = [call for call in FakeToolClient.calls if call[0] == "submit_local_source_confirmations"]
    assert submitted["answers"] == []
    assert submitted["failure_cause"] == (
        "ValueError: Jira issue reference '1#x' is neither an issue id nor an issue key"
    )
    assert _call_names()[-1] == "start_source_processing"
    assert events == ["closed"]


def test_local_agent_cloud_jira_closes_its_session_when_authentication_fails(monkeypatch):
    events: list[str] = []
    FakeToolClient.reset({})

    payload = _run_confirming_jira_job(
        monkeypatch, _confirming_jira_gene(events, authenticate_error=RuntimeError("session rejected"))
    )

    assert payload["error"] == "session rejected"
    assert _call_names() == []
    assert events == ["closed"]


def test_local_agent_cloud_jira_does_not_process_when_its_answers_are_refused(monkeypatch):
    events: list[str] = []
    FakeToolClient.reset({})
    FakeToolClient.confirmations_response = {
        "error": "MemForge API request failed",
        "status_code": 409,
        "detail": "snapshot confirmations answer another manifest digest",
    }

    payload = _run_confirming_jira_job(monkeypatch, _confirming_jira_gene(events))

    assert payload["error"] == "snapshot confirmations answer another manifest digest"
    assert payload["retryable"] is True
    assert "start_source_processing" not in _call_names()
    assert events == ["confirmed 100007, PAY-8", "closed"]


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
        "code": "jira_principal_changed",
        "detail": {
            "code": "jira_principal_changed",
            "origin": "https://jira.example.test",
            "old_principal_id": "old-user",
            "new_principal_id": "different-user",
        },
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
        config_schema = JiraGene.config_schema

        def __init__(self, config, source_id):
            self.config = config
            self.source_id = source_id
            self.closed = False

        async def close(self):
            self.closed = True

        async def authenticate(self):
            assert self.config["sync_mode"] == "cloud"

        async def discover_inventory(self):
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
                body=json.dumps(
                    {
                        "key": key,
                        "fields": {"summary": f"{key} title", "updated": "v1"},
                    }
                ).encode(),
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

    assert payload["counts"] == {"selected": 2, "reused": 0, "pushed": 1, "failed": 1}
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

    assert (
        payload
        == TeamsAuthResult(operation="teams_auth", authenticated=True, region="amer", token_count=1).model_dump()
    )
    assert captured_regions == ["amer"]
    assert wait_options == [(90, 2.0)]
    assert FakeToolClient.calls == []


def test_local_agent_cloud_jira_auth_opens_interactive_browser_and_uploads_session(monkeypatch):
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
            "job_id": "laj-jira-auth",
            "workspace_id": "mount_tai",
            "operation": "jira_auth",
            "source_type": "jira",
            "source_id": "src-jira",
            "payload": {
                "base_url": "https://jira.example.test",
                "auth_mode": "browser_cookie",
            },
        },
        _cloud_test_client(),
        browser="chrome",
    )

    assert (
        payload
        == JiraAuthResult(
            operation="jira_auth", source_id="src-jira", authenticated=True, origin="https://jira.example.test"
        ).model_dump()
    )
    assert captured["base_url"] == "https://jira.example.test"
    assert captured["browser"] == "chrome"
    assert captured["interactive"] is True
    assert captured["tls_config"] == {
        "base_url": "https://jira.example.test",
        "auth_mode": "browser_cookie",
        "sync_mode": "cloud",
    }
    upload = next(call for name, call in FakeToolClient.calls if name == "upload_jira_session")
    assert upload["workspace_id"] == "mount_tai"
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
