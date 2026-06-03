"""Tests for the bare `memforge` interactive dispatch and surrounding contract."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

import memforge.main as main
from memforge.main import cli


def test_bare_memforge_dispatches_to_interactive(monkeypatch):
    """Running `memforge` with no subcommand calls the Clack dispatcher."""
    calls: list[tuple] = []

    def fake_dispatch() -> int:
        calls.append(("dispatch",))
        return 0

    monkeypatch.setattr(main, "_dispatch_interactive", fake_dispatch)

    result = CliRunner().invoke(cli, [])

    assert result.exit_code == 0, result.output
    assert calls == [("dispatch",)]


def test_bare_memforge_propagates_dispatch_exit_code(monkeypatch):
    monkeypatch.setattr(main, "_dispatch_interactive", lambda: 7)

    result = CliRunner().invoke(cli, [])

    assert result.exit_code == 7


def test_dispatch_interactive_falls_back_to_help_when_disabled(monkeypatch):
    """The flag the Node wrapper sets on subprocess calls must short-circuit."""
    monkeypatch.setenv(main.INTERACTIVE_DISABLE_ENV, "1")

    runner = CliRunner()
    result = runner.invoke(cli, [])

    assert result.exit_code == 0
    assert "MemForge" in result.output


def test_scriptable_subcommands_remain_unaffected(monkeypatch):
    """Running an actual subcommand must not trigger the interactive dispatcher."""
    monkeypatch.setattr(main, "_dispatch_interactive", lambda: 99)

    result = CliRunner().invoke(cli, ["adapter", "list"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert {"type": "kb", "kind": "markdown"} in payload["data"]


def test_old_auth_jira_command_is_removed():
    """`memforge auth jira` must not exist; the canonical path is `adapter auth jira`."""
    result = CliRunner().invoke(cli, ["auth", "jira", "--help"])
    assert result.exit_code != 0

    adapter_help = CliRunner().invoke(cli, ["adapter", "auth", "jira", "--help"])
    assert adapter_help.exit_code == 0, adapter_help.output


def test_clack_package_is_wired_in_repository():
    """The Node Clack package must exist and declare `@clack/prompts`."""
    repo_root = Path(__file__).resolve().parents[1]
    package_json = repo_root / "cli" / "package.json"
    assert package_json.exists(), "cli/package.json must exist"
    data = json.loads(package_json.read_text())
    assert data.get("type") == "module"
    assert data.get("dependencies", {}).get("@clack/prompts"), (
        "cli/package.json must declare @clack/prompts"
    )
    assert (repo_root / "cli" / "index.mjs").exists(), "cli/index.mjs must exist"


def test_interactive_script_path_resolves_to_repo_cli(monkeypatch):
    monkeypatch.delenv(main.INTERACTIVE_SCRIPT_ENV, raising=False)
    resolved = main._interactive_script_path()
    assert resolved is not None
    assert resolved.name == "index.mjs"
    assert resolved.parent.name == "cli"


def test_run_interactive_script_reports_when_node_missing(monkeypatch):
    """Without Node on PATH the launcher prints a clear setup hint, no fallback to Click help."""
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv(main.INTERACTIVE_SCRIPT_ENV, str(repo_root / "cli" / "index.mjs"))
    monkeypatch.setattr(main.shutil, "which", lambda _name: None)

    rc = main._run_interactive_script()

    assert rc == 2


def test_run_interactive_script_reports_when_dependencies_missing(monkeypatch, tmp_path):
    """If the cli package was not `npm install`ed, the launcher must surface that."""
    fake_script = tmp_path / "cli" / "index.mjs"
    fake_script.parent.mkdir(parents=True)
    fake_script.write_text("// placeholder", encoding="utf-8")
    monkeypatch.setenv(main.INTERACTIVE_SCRIPT_ENV, str(fake_script))
    monkeypatch.setattr(main.shutil, "which", lambda _name: "/usr/bin/node")

    rc = main._run_interactive_script()

    assert rc == 2
