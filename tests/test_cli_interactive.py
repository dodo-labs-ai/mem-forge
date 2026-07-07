"""Tests for the bare `memforge` interactive dispatch and surrounding contract."""

from __future__ import annotations

import json
import subprocess
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


def test_interactive_resources_are_packaged_with_python_distribution():
    resources = main._interactive_resource_dir()

    assert (resources / "index.mjs").is_file()
    assert (resources / "package.json").is_file()
    assert (resources / "package-lock.json").is_file()


def test_node_cli_and_packaged_interactive_script_stay_in_sync():
    repo_root = Path(__file__).resolve().parents[1]
    node_entrypoint = repo_root / "cli" / "index.mjs"
    packaged_entrypoint = main._interactive_resource_dir() / "index.mjs"

    assert packaged_entrypoint.read_text() == node_entrypoint.read_text()


def test_interactive_script_path_resolves_to_packaged_resource(monkeypatch):
    monkeypatch.delenv(main.INTERACTIVE_SCRIPT_ENV, raising=False)
    resolved = main._interactive_script_path(main._interactive_resource_dir())
    assert resolved is not None
    assert resolved.name == "index.mjs"
    assert resolved.parent.name == "interactive_cli"


def test_run_interactive_script_reports_when_node_missing(monkeypatch):
    """Without Node on PATH the launcher prints a clear setup hint, no fallback to Click help."""
    monkeypatch.setattr(main.shutil, "which", lambda _name: None)

    rc = main._run_interactive_script()

    assert rc == 2


def test_run_interactive_script_honors_script_override(monkeypatch, tmp_path):
    fake_script = tmp_path / "index.mjs"
    fake_script.write_text("process.exit(0);", encoding="utf-8")
    monkeypatch.setenv(main.INTERACTIVE_SCRIPT_ENV, str(fake_script))
    monkeypatch.setattr(main.shutil, "which", lambda name: "/usr/bin/node" if name == "node" else None)
    calls: list[tuple[str, ...]] = []

    def fake_run(args, *, cwd=None, env=None, check=False):
        calls.append(tuple(args))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    rc = main._run_interactive_script()

    assert rc == 0
    assert calls == [("/usr/bin/node", str(fake_script))]


def test_run_interactive_script_installs_dependencies_into_cache(monkeypatch, tmp_path):
    """First bare run prepares the packaged Node UI without mutating the source tree."""
    cache_root = tmp_path / "cache"
    calls: list[tuple[tuple[str, ...], Path | None]] = []

    monkeypatch.setattr(
        main.shutil,
        "which",
        lambda name: {"node": "/usr/bin/node", "npm": "/usr/bin/npm"}.get(name),
    )
    monkeypatch.setenv(main.INTERACTIVE_CACHE_ENV, str(cache_root))

    def fake_run(args, *, cwd=None, env=None, check=False):
        calls.append((tuple(args), cwd))
        if args[:3] == ["/usr/bin/npm", "ci", "--omit=dev"]:
            prompts_dir = Path(cwd) / "node_modules" / "@clack" / "prompts"
            prompts_dir.mkdir(parents=True)
            return subprocess.CompletedProcess(args, 0)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    rc = main._run_interactive_script()

    assert rc == 0
    assert [call[0][:4] for call in calls] == [
        ("/usr/bin/npm", "ci", "--omit=dev", "--no-audit"),
        ("/usr/bin/node", str(calls[1][0][1])),
    ]
    workspace = calls[0][1]
    assert workspace is not None
    assert workspace.is_relative_to(cache_root)
    assert (workspace / "index.mjs").exists()
    assert (workspace / "package.json").exists()
    assert (workspace / "package-lock.json").exists()
    assert calls[1][0][1] == str(workspace / "index.mjs")


def test_run_interactive_script_reuses_installed_cache(monkeypatch, tmp_path):
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(
        main.shutil,
        "which",
        lambda name: {"node": "/usr/bin/node", "npm": "/usr/bin/npm"}.get(name),
    )
    monkeypatch.setenv(main.INTERACTIVE_CACHE_ENV, str(cache_root))
    monkeypatch.setattr(
        main,
        "_install_interactive_dependencies",
        lambda workspace: (workspace / main.INTERACTIVE_DEPENDENCY_SENTINEL).mkdir(parents=True),
    )

    workspace = main._prepare_interactive_workspace()

    calls: list[tuple[str, ...]] = []

    def fake_run(args, *, cwd=None, env=None, check=False):
        calls.append(tuple(args))
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    rc = main._run_interactive_script()

    assert rc == 0
    assert calls == [("/usr/bin/node", str(workspace / "index.mjs"))]
