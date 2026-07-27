from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest


ROOT = Path(__file__).resolve().parents[1]
CODEX_PLUGIN = ROOT / "integrations" / "codex" / "memforge-memory"
CLAUDE_PLUGIN = ROOT / "integrations" / "claude-code" / "memforge-memory"
SKILL_RELATIVE_PATH = Path("skills") / "memforge-setup" / "SKILL.md"
HELPER_RELATIVE_PATH = Path("skills") / "memforge-setup" / "scripts" / "workspace_config.py"
HELPER = CODEX_PLUGIN / HELPER_RELATIVE_PATH
CLOUD_ORIGIN = "https://memforge.example.hana.ondemand.com"


def _environment(home: Path, **values: str) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "CODEX_HOME",
        "MEMFORGE_API_URL",
        "MEMFORGE_API_TOKEN",
        "MEMFORGE_CLAUDE_SETTINGS",
        "MEMFORGE_CODEX_CONFIG",
        "MEMFORGE_WORKSPACE_ID",
    ):
        environment.pop(name, None)
    environment["HOME"] = str(home)
    environment.update(values)
    return environment


def _run_helper(
    home: Path,
    *arguments: str,
    environment_values: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(HELPER), *arguments],
        env=_environment(home, **(environment_values or {})),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )


def _init_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)


def _json_output(result: subprocess.CompletedProcess[str]) -> dict:
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_setup_skill_is_discoverable_and_packaged_identically():
    codex_manifest = json.loads((CODEX_PLUGIN / ".codex-plugin" / "plugin.json").read_text())
    codex_skill = (CODEX_PLUGIN / SKILL_RELATIVE_PATH).read_text(encoding="utf-8")
    claude_skill = (CLAUDE_PLUGIN / SKILL_RELATIVE_PATH).read_text(encoding="utf-8")
    codex_helper = (CODEX_PLUGIN / HELPER_RELATIVE_PATH).read_text(encoding="utf-8")
    claude_helper = (CLAUDE_PLUGIN / HELPER_RELATIVE_PATH).read_text(encoding="utf-8")

    assert codex_manifest["skills"] == "./skills/"
    assert codex_skill == claude_skill
    assert codex_helper == claude_helper
    assert codex_skill.startswith("---\nname: memforge-setup\n")
    for required_text in (
        "Configure or update the global default.",
        "Configure or update the current repository override.",
        "Inspect the effective workspace selection.",
        "Remove the current repository override.",
        "Ask for explicit confirmation",
        "Never put `MEMFORGE_API_TOKEN`",
        "Never create, replace, or duplicate a",
        "`~/.codex/config.toml`",
        "`~/.claude/settings.json`",
    ):
        assert required_text in codex_skill


def test_repository_setup_preview_apply_inspect_and_remove(tmp_path):
    home = tmp_path / "home"
    repository = tmp_path / "repository"
    home.mkdir()
    _init_repository(repository)
    environment = {
        "MEMFORGE_API_URL": CLOUD_ORIGIN,
        "MEMFORGE_WORKSPACE_ID": "global_workspace",
    }
    config_path = repository / ".memforge" / "config.toml"
    exclude_path = repository / ".git" / "info" / "exclude"
    original_exclude = exclude_path.read_text(encoding="utf-8")

    preview = _json_output(
        _run_helper(
            home,
            "set-repo",
            "--client",
            "codex",
            "--repo",
            str(repository),
            "--workspace",
            "repository_workspace",
            environment_values=environment,
        )
    )

    assert preview["mode"] == "preview"
    assert preview["workspace"] == "repository_workspace"
    assert preview["target"]["workspace_id"] == "repository_workspace"
    assert preview["new_session_required"] is False
    assert not config_path.exists()
    assert exclude_path.read_text(encoding="utf-8") == original_exclude

    applied = _json_output(
        _run_helper(
            home,
            "set-repo",
            "--client",
            "codex",
            "--repo",
            str(repository),
            "--workspace",
            "repository_workspace",
            "--apply",
            environment_values=environment,
        )
    )

    assert applied["mode"] == "applied"
    repository_document = tomllib.loads(config_path.read_text(encoding="utf-8"))
    assert repository_document == {"memforge": {"workspace_id": "repository_workspace"}}
    assert "TOKEN" not in config_path.read_text(encoding="utf-8").upper()
    assert subprocess.run(
        ["git", "-C", str(repository), "check-ignore", "-q", ".memforge/config.toml"],
        check=False,
    ).returncode == 0

    inspected = _json_output(
        _run_helper(
            home,
            "inspect",
            "--client",
            "codex",
            "--repo",
            str(repository),
            environment_values=environment,
        )
    )
    assert inspected["effective_source"] == "repository"
    assert inspected["effective_workspace"] == "repository_workspace"
    assert inspected["credentials_reported"] is False
    assert inspected["mcp_configuration_changed"] is False

    removal_preview = _json_output(
        _run_helper(
            home,
            "remove-repo",
            "--client",
            "codex",
            "--repo",
            str(repository),
            environment_values=environment,
        )
    )
    assert removal_preview["fallback_source"] == "process"
    assert removal_preview["fallback_workspace"] == "global_workspace"
    assert config_path.exists()

    removed = _json_output(
        _run_helper(
            home,
            "remove-repo",
            "--client",
            "codex",
            "--repo",
            str(repository),
            "--apply",
            environment_values=environment,
        )
    )
    assert removed["mode"] == "applied"
    assert not config_path.exists()
    assert "/.memforge/config.toml" in exclude_path.read_text(encoding="utf-8")


@pytest.mark.parametrize("client", ["codex", "claude-code"])
def test_global_setup_is_client_specific_and_requires_apply(tmp_path, client):
    home = tmp_path / "home"
    home.mkdir()
    codex_config = home / ".codex" / "config.toml"
    claude_settings = home / ".claude" / "settings.json"
    codex_config.parent.mkdir()
    claude_settings.parent.mkdir()
    codex_config.write_text(
        f'[memforge]\nMEMFORGE_API_URL = "{CLOUD_ORIGIN}"\nMEMFORGE_WORKSPACE_ID = "old_codex"\n',
        encoding="utf-8",
    )
    claude_settings.write_text(
        json.dumps(
            {
                "theme": "dark",
                "env": {
                    "MEMFORGE_API_URL": CLOUD_ORIGIN,
                    "MEMFORGE_WORKSPACE_ID": "old_claude",
                },
            }
        ),
        encoding="utf-8",
    )
    before_codex = codex_config.read_text(encoding="utf-8")
    before_claude = claude_settings.read_text(encoding="utf-8")

    preview = _json_output(
        _run_helper(
            home,
            "set-global",
            "--client",
            client,
            "--workspace",
            "new_workspace",
        )
    )
    assert preview["mode"] == "preview"
    assert preview["new_session_required"] is True
    assert codex_config.read_text(encoding="utf-8") == before_codex
    assert claude_settings.read_text(encoding="utf-8") == before_claude

    _json_output(
        _run_helper(
            home,
            "set-global",
            "--client",
            client,
            "--workspace",
            "new_workspace",
            "--apply",
        )
    )

    codex_workspace = tomllib.loads(codex_config.read_text(encoding="utf-8"))["memforge"][
        "MEMFORGE_WORKSPACE_ID"
    ]
    claude_document = json.loads(claude_settings.read_text(encoding="utf-8"))
    claude_workspace = claude_document["env"]["MEMFORGE_WORKSPACE_ID"]
    assert claude_document["theme"] == "dark"
    if client == "codex":
        assert codex_workspace == "new_workspace"
        assert claude_workspace == "old_claude"
    else:
        assert codex_workspace == "old_codex"
        assert claude_workspace == "new_workspace"


def test_repository_setup_rejects_oss_workspace_and_credentials(tmp_path):
    home = tmp_path / "home"
    repository = tmp_path / "repository"
    home.mkdir()
    _init_repository(repository)
    config_path = repository / ".memforge" / "config.toml"

    invalid_target = _run_helper(
        home,
        "set-repo",
        "--client",
        "codex",
        "--repo",
        str(repository),
        "--workspace",
        "cloud_workspace",
        "--apply",
        environment_values={"MEMFORGE_API_URL": "https://self.example"},
    )
    assert invalid_target.returncode == 2
    assert "workspace_not_supported_for_oss" in invalid_target.stderr
    assert not config_path.exists()

    config_path.parent.mkdir()
    original = '[memforge]\nworkspace_id = "old"\nMEMFORGE_API_TOKEN = "secret"\n'
    config_path.write_text(original, encoding="utf-8")
    credential_config = _run_helper(
        home,
        "set-repo",
        "--client",
        "codex",
        "--repo",
        str(repository),
        "--workspace",
        "new",
        "--apply",
        environment_values={"MEMFORGE_API_URL": CLOUD_ORIGIN},
    )
    assert credential_config.returncode == 2
    assert "credential-like keys" in credential_config.stderr
    assert config_path.read_text(encoding="utf-8") == original
