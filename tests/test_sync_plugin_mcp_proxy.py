from __future__ import annotations

import importlib.util
from pathlib import Path
import stat
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "sync_plugin_mcp_proxy.py"


def _load_sync_module():
    spec = importlib.util.spec_from_file_location("sync_plugin_mcp_proxy_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_check_reports_stale_copy_without_mutating_it(tmp_path):
    sync_module = _load_sync_module()
    canonical = tmp_path / "plugin_mcp_proxy.py"
    generated = tmp_path / "memforge_mcp.py"
    canonical.write_text("canonical\n")
    generated.write_text("stale\n")

    stale = sync_module.synchronize_plugin_copies(
        canonical,
        (generated,),
        check=True,
    )

    assert stale == (generated,)
    assert generated.read_text() == "stale\n"


def test_sync_replaces_stale_copy_and_matches_canonical_mode(tmp_path):
    sync_module = _load_sync_module()
    canonical = tmp_path / "plugin_mcp_proxy.py"
    generated = tmp_path / "memforge_mcp.py"
    canonical.write_text("canonical\n")
    canonical.chmod(0o755)
    generated.write_text("stale\n")
    generated.chmod(0o600)

    changed = sync_module.synchronize_plugin_copies(
        canonical,
        (generated,),
        check=False,
    )

    assert changed == (generated,)
    assert generated.read_bytes() == canonical.read_bytes()
    assert stat.S_IMODE(generated.stat().st_mode) == stat.S_IMODE(canonical.stat().st_mode)


def test_packaged_plugin_config_rewrites_only_the_target_import_block():
    sync_module = _load_sync_module()
    canonical = (
        sync_module.CANONICAL_PLUGIN_CONFIG_TARGET_IMPORT
        + b"\nCONFIG_VALUE = 'preserved'\n"
    )

    packaged = sync_module.packaged_plugin_config_content(canonical)

    assert sync_module.CANONICAL_PLUGIN_CONFIG_TARGET_IMPORT not in packaged
    assert sync_module.PACKAGED_PLUGIN_CONFIG_TARGET_IMPORT in packaged
    assert packaged.endswith(b"\nCONFIG_VALUE = 'preserved'\n")


def test_repo_plugin_copies_pass_the_check_command():
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "already in sync" in result.stdout
    assert result.stderr == ""


def test_daemon_setup_skill_is_packaged_identically_for_both_clients():
    codex_skill = ROOT / "integrations" / "codex" / "memforge-memory" / "skills" / "memforge-daemon-setup"
    claude_skill = ROOT / "integrations" / "claude-code" / "memforge-memory" / "skills" / "memforge-daemon-setup"

    assert (codex_skill / "SKILL.md").read_bytes() == (claude_skill / "SKILL.md").read_bytes()
    assert (codex_skill / "agents" / "openai.yaml").read_bytes() == (
        claude_skill / "agents" / "openai.yaml"
    ).read_bytes()
