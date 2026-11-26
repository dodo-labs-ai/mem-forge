"""``memforge upgrade``: install exactly the server's MemForge version the way it was installed (ADR 0058)."""

from __future__ import annotations

import json
import subprocess

import pytest
from click.testing import CliRunner

import memforge.main as main
from memforge.local_agent import device_identity
from memforge.local_agent.upgrade import InstallMethod, UnknownInstallMethod, install_method
from memforge.release import installed_version
from tests.test_daemon_service import FakeServiceManager

SERVER_VERSION = "9.9.9b1"


def _pipx_installation(root):
    venv = root / "pipx" / "venvs" / "memforge-ai"
    (venv / "bin").mkdir(parents=True)
    (venv / "pipx_metadata.json").write_text("{}", encoding="utf-8")
    (venv / "bin" / "memforge").write_text("#!/bin/sh\n", encoding="utf-8")
    (root / "bin").mkdir()
    (root / "bin" / "memforge").symlink_to(venv / "bin" / "memforge")
    return root / "bin" / "memforge"


def test_the_install_method_is_read_from_where_the_executable_lives(tmp_path) -> None:
    executable = _pipx_installation(tmp_path)
    checkout = tmp_path / "checkout" / ".venv" / "bin" / "memforge"
    checkout.parent.mkdir(parents=True)
    checkout.write_text("#!/bin/sh\n", encoding="utf-8")

    method = install_method(executable)

    assert method == InstallMethod(name="pipx", package="memforge-ai")
    assert method.install_command(SERVER_VERSION) == ["pipx", "install", "--force", f"memforge-ai=={SERVER_VERSION}"]
    with pytest.raises(UnknownInstallMethod):
        install_method(checkout)


class _Server:
    """The server the daemon serves: it names its version, and the device's report once the daemon restarts."""

    version = SERVER_VERSION
    reported_version = SERVER_VERSION
    restarted = False

    def __init__(self, *, target, api_token):
        pass

    def get_local_agent_status(self):
        devices = []
        if type(self).restarted:
            device = device_identity.DeviceIdentityStore(device_identity.device_identity_path()).ensure()
            devices = [
                {
                    "device_id": device.device_id,
                    "name": device.name,
                    "version": type(self).reported_version,
                    "status": "online",
                    "last_seen_at": "2026-11-26T09:00:00+00:00",
                }
            ]
        return {
            "status": "online",
            "devices": devices,
            "sources": [],
            "server": {"version": type(self).version, "acquisition_contracts": {}},
        }


@pytest.fixture
def upgrade(monkeypatch, tmp_path):
    manager = FakeServiceManager()
    replace = manager.replace

    def restart(**kwargs):
        _Server.restarted = True
        return replace(**kwargs)

    manager.replace = restart
    installs: list[list[str]] = []
    _Server.version, _Server.reported_version, _Server.restarted = SERVER_VERSION, SERVER_VERSION, False
    monkeypatch.setattr(main, "_daemon_service_manager", lambda: manager)
    monkeypatch.setattr(main, "ToolClient", _Server)
    monkeypatch.setattr(main, "_current_cli_executable", lambda: str(_pipx_installation(tmp_path)))
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda command, check=False: installs.append(command) or subprocess.CompletedProcess(command, 0),
    )

    def run():
        return CliRunner().invoke(main.cli, ["upgrade"], env={"MEMFORGE_CLI_CONFIG": str(tmp_path / "cli.toml")})

    return run, manager, installs


def test_upgrade_installs_the_servers_version_and_restarts_the_daemon_through_its_replacement(upgrade) -> None:
    run, manager, installs = upgrade

    result = run()

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {
        "ok": True,
        "version": SERVER_VERSION,
        "upgraded": True,
        "install_method": "pipx",
    }
    assert installs == [["pipx", "install", "--force", f"memforge-ai=={SERVER_VERSION}"]]
    assert manager.installs[0]["api_url"] == "https://stored.example"
    assert manager.replacements[0].committed is True


def test_upgrade_does_nothing_on_a_device_that_runs_the_servers_version(upgrade) -> None:
    run, manager, installs = upgrade
    _Server.version = installed_version()

    result = run()

    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["upgraded"] is False
    assert installs == [] and manager.replacements == []


def test_upgrade_restores_the_service_when_the_daemon_does_not_report_the_new_version(upgrade, monkeypatch) -> None:
    run, manager, installs = upgrade
    _Server.reported_version = "0.0.1"
    monkeypatch.setattr(
        main,
        "_wait_for_local_agent_online",
        lambda client, **kwargs: {**client.get_local_agent_status()["devices"][0], "fresh": True},
    )

    result = run()

    assert result.exit_code == 1
    assert f"did not report MemForge {SERVER_VERSION}" in result.output
    assert f"MemForge {installed_version()} was installed again" in result.output
    assert installs == [
        ["pipx", "install", "--force", f"memforge-ai=={SERVER_VERSION}"],
        ["pipx", "install", "--force", f"memforge-ai=={installed_version()}"],
    ]
    assert manager.replacements[0].rolled_back is True


def test_upgrade_names_the_installed_version_when_the_previous_one_cannot_be_installed_again(
    upgrade, monkeypatch
) -> None:
    run, manager, installs = upgrade
    _Server.reported_version = "0.0.1"
    monkeypatch.setattr(
        main,
        "_wait_for_local_agent_online",
        lambda client, **kwargs: {**client.get_local_agent_status()["devices"][0], "fresh": True},
    )
    monkeypatch.setattr(
        main.subprocess,
        "run",
        lambda command, check=False: installs.append(command)
        or subprocess.CompletedProcess(command, 0 if len(installs) == 1 else 1),
    )

    result = run()

    assert result.exit_code == 1
    assert f"MemForge {SERVER_VERSION} stays installed" in result.output
    assert manager.replacements[0].rolled_back is True


def test_upgrade_names_the_installed_version_when_the_service_cannot_be_restored(upgrade, monkeypatch) -> None:
    from memforge.local_agent.service import DaemonServiceError

    run, manager, installs = upgrade
    _Server.reported_version = "0.0.1"
    monkeypatch.setattr(
        main,
        "_wait_for_local_agent_online",
        lambda client, **kwargs: {**client.get_local_agent_status()["devices"][0], "fresh": True},
    )
    replace = manager.replace

    def unrestorable(**kwargs):
        replacement = replace(**kwargs)

        def rollback():
            raise DaemonServiceError("launchctl bootstrap failed")

        replacement.rollback = rollback
        return replacement

    manager.replace = unrestorable

    result = run()

    assert result.exit_code == 1
    assert f"MemForge {installed_version()} was installed again" in result.output
    assert "its service could not be restored: launchctl bootstrap failed" in result.output
