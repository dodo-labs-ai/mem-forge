from __future__ import annotations

import json
import os
import plistlib
import subprocess

from click.testing import CliRunner
import pytest

import memforge.main as main
from memforge.local_agent.service import (
    KEYRING_SERVICE,
    DaemonCredentialStore,
    DaemonServiceManager,
    DaemonServiceError,
    DaemonServicePaths,
    LaunchdUserService,
    SystemdUserService,
    service_adapter,
)
from memforge.main import cli


class FakeKeyring:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.values.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.values[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        self.values.pop((service, username), None)


class FakeLaunchctl:
    def __init__(self) -> None:
        self.loaded = False
        self.commands: list[list[str]] = []

    def __call__(self, args, **kwargs) -> subprocess.CompletedProcess[str]:
        command = list(args)
        self.commands.append(command)
        if command[:2] == ["launchctl", "print"]:
            if self.loaded:
                return subprocess.CompletedProcess(command, 0, "state = running\npid = 4321\n", "")
            return subprocess.CompletedProcess(command, 113, "", "service not found")
        if command[:2] == ["launchctl", "bootstrap"]:
            self.loaded = True
        elif command[:2] == ["launchctl", "bootout"]:
            self.loaded = False
        return subprocess.CompletedProcess(command, 0, "", "")


class FailingLaunchctl(FakeLaunchctl):
    def __init__(self) -> None:
        super().__init__()
        self.loaded = True
        self.fail_next_bootstrap = True

    def __call__(self, args, **kwargs) -> subprocess.CompletedProcess[str]:
        command = list(args)
        if command[:2] == ["launchctl", "bootstrap"] and self.fail_next_bootstrap:
            self.commands.append(command)
            self.fail_next_bootstrap = False
            raise subprocess.CalledProcessError(5, command, stderr="bootstrap failed")
        return super().__call__(args, **kwargs)


class FakeSystemctl:
    def __init__(self) -> None:
        self.active = False
        self.enabled = False
        self.commands: list[list[str]] = []

    def __call__(self, args, **kwargs) -> subprocess.CompletedProcess[str]:
        command = list(args)
        self.commands.append(command)
        if command[:3] == ["systemctl", "--user", "is-active"]:
            return subprocess.CompletedProcess(
                command, 0 if self.active else 3, "active\n" if self.active else "inactive\n", ""
            )
        if command[:3] == ["systemctl", "--user", "is-enabled"]:
            return subprocess.CompletedProcess(
                command, 0 if self.enabled else 1, "enabled\n" if self.enabled else "disabled\n", ""
            )
        if command[:4] == ["systemctl", "--user", "enable", "--now"]:
            self.enabled = True
            self.active = True
        elif command[:4] == ["systemctl", "--user", "disable", "--now"]:
            self.enabled = False
            self.active = False
        elif command[:3] == ["systemctl", "--user", "start"]:
            self.active = True
        elif command[:3] == ["systemctl", "--user", "stop"]:
            self.active = False
        elif command[:3] == ["systemctl", "--user", "restart"]:
            self.active = True
        return subprocess.CompletedProcess(command, 0, "", "")


def test_daemon_credentials_keep_token_out_of_config(tmp_path):
    keyring = FakeKeyring()
    config_path = tmp_path / "daemon-service.json"
    store = DaemonCredentialStore(config_path, keyring_store=keyring)

    target = store.save(api_url="https://memforge.example/", api_token="secret-token")

    saved = config_path.read_text(encoding="utf-8")
    assert "secret-token" not in saved
    assert json.loads(saved) == {
        "api_url": "https://memforge.example",
        "keyring_account": target.keyring_account,
        "version": 1,
    }
    assert config_path.stat().st_mode & 0o777 == 0o600
    assert keyring.values[(KEYRING_SERVICE, target.keyring_account)] == "secret-token"
    assert store.environment() == {
        "MEMFORGE_API_URL": "https://memforge.example",
        "MEMFORGE_API_TOKEN": "secret-token",
    }

    store.delete()

    assert not config_path.exists()
    assert keyring.values == {}


def test_failed_service_replacement_restores_previous_credential(tmp_path):
    keyring = FakeKeyring()
    store = DaemonCredentialStore(tmp_path / "daemon-service.json", keyring_store=keyring)
    store.save(api_url="https://old.example", api_token="old-token")

    class FailingAdapter:
        platform = "fake"
        unit_path = tmp_path / "fake.service"

        def install(self, command):
            raise RuntimeError("install failed")

    manager = DaemonServiceManager(adapter=FailingAdapter(), credentials=store)

    with pytest.raises(RuntimeError, match="install failed"):
        manager.install(
            executable="/opt/memforge/bin/memforge",
            api_url="https://new.example",
            api_token="new-token",
        )

    assert store.environment() == {
        "MEMFORGE_API_URL": "https://old.example",
        "MEMFORGE_API_TOKEN": "old-token",
    }
    assert "new-token" not in keyring.values.values()


def test_service_adapter_rejects_unsupported_platform_before_writing_state(tmp_path):
    with pytest.raises(DaemonServiceError, match="supports macOS launchd and Linux systemd"):
        service_adapter(platform_name="win32", home=tmp_path)

    assert list(tmp_path.rglob("*")) == []


def test_launchd_adapter_installs_login_service_without_shell_or_secrets(tmp_path):
    runner = FakeLaunchctl()
    paths = DaemonServicePaths.for_home(tmp_path)
    adapter = LaunchdUserService(home=tmp_path, uid=501, paths=paths, runner=runner)

    adapter.install(["/Users/test/.local/bin/memforge", "daemon", "_run"])

    payload = plistlib.loads(adapter.unit_path.read_bytes())
    assert payload["ProgramArguments"] == ["/Users/test/.local/bin/memforge", "daemon", "_run"]
    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] is True
    assert "EnvironmentVariables" not in payload
    assert adapter.unit_path.stat().st_mode & 0o777 == 0o600
    assert paths.stdout_log.stat().st_mode & 0o777 == 0o600
    assert paths.stderr_log.stat().st_mode & 0o777 == 0o600
    assert ["launchctl", "bootstrap", "gui/501", str(adapter.unit_path)] in runner.commands
    assert adapter.status() == {
        "platform": "launchd",
        "installed": True,
        "loaded": True,
        "running": True,
        "state": "running",
        "pid": 4321,
        "unit_path": str(adapter.unit_path),
    }

    adapter.restart()
    adapter.uninstall()

    assert ["launchctl", "kickstart", "-k", "gui/501/com.memforge.local-daemon"] in runner.commands
    assert not adapter.unit_path.exists()


def test_launchd_adapter_restores_previous_service_when_replacement_fails(tmp_path):
    runner = FailingLaunchctl()
    paths = DaemonServicePaths.for_home(tmp_path)
    adapter = LaunchdUserService(home=tmp_path, uid=501, paths=paths, runner=runner)
    adapter.unit_path.parent.mkdir(parents=True)
    adapter.unit_path.write_bytes(b"previous plist")

    with pytest.raises(DaemonServiceError, match="bootstrap failed"):
        adapter.install(["/Users/test/.local/bin/memforge", "daemon", "_run"])

    assert adapter.unit_path.read_bytes() == b"previous plist"
    assert runner.loaded is True


def test_systemd_adapter_installs_and_enables_user_service(tmp_path):
    runner = FakeSystemctl()
    paths = DaemonServicePaths.for_home(tmp_path)
    adapter = SystemdUserService(home=tmp_path, paths=paths, runner=runner)

    adapter.install(["/home/test/.local/bin/memforge", "daemon", "_run"])

    unit = adapter.unit_path.read_text(encoding="utf-8")
    assert 'ExecStart="/home/test/.local/bin/memforge" "daemon" "_run"' in unit
    assert "Restart=on-failure" in unit
    assert "Environment=" not in unit
    assert ["systemctl", "--user", "enable", "--now", "memforge-local-daemon.service"] in runner.commands
    assert adapter.status()["running"] is True

    adapter.stop()
    assert adapter.status()["state"] == "inactive"
    adapter.start()
    adapter.uninstall()

    assert not adapter.unit_path.exists()
    assert ["systemctl", "--user", "daemon-reload"] == runner.commands[-1]


def test_systemd_unit_quotes_arguments_and_escapes_specifiers():
    unit = SystemdUserService._unit(["/home/test user/100%/memforge", "daemon", "_run"])

    assert 'ExecStart="/home/test user/100%%/memforge" "daemon" "_run"' in unit


class FakeServiceManager:
    def __init__(self) -> None:
        self.installs: list[dict[str, str | None]] = []
        self.actions: list[str] = []

    def install(self, *, executable: str, api_url: str, api_token: str | None):
        self.installs.append({"executable": executable, "api_url": api_url, "api_token": api_token})
        return {"platform": "launchd", "installed": True, "running": True}

    def status(self):
        self.actions.append("status")
        return {"platform": "launchd", "installed": True, "loaded": True, "running": True}

    def start(self):
        self.actions.append("start")
        return {"running": True}

    def stop(self):
        self.actions.append("stop")
        return {"running": False}

    def restart(self):
        self.actions.append("restart")
        return {"running": True}

    def uninstall(self):
        self.actions.append("uninstall")
        return {"installed": False, "running": False}

    def environment(self):
        return {"MEMFORGE_API_URL": "https://stored.example", "MEMFORGE_API_TOKEN": "stored-token"}

    def logs(self, *, lines: int, follow: bool):
        self.actions.append(f"logs:{lines}:{follow}")
        return 0


def test_daemon_cli_installs_active_cloud_target(monkeypatch, tmp_path):
    manager = FakeServiceManager()
    monkeypatch.setattr(main, "_daemon_service_manager", lambda: manager)
    monkeypatch.setattr(main, "_current_cli_executable", lambda: "/opt/memforge/bin/memforge")

    result = CliRunner().invoke(
        cli,
        ["daemon", "install"],
        env={
            "MEMFORGE_CLI_CONFIG": str(tmp_path / "cli.toml"),
            "MEMFORGE_API_URL": "https://cloud.example",
            "MEMFORGE_API_TOKEN": "cloud-token",
        },
    )

    assert result.exit_code == 0, result.output
    assert manager.installs == [
        {
            "executable": "/opt/memforge/bin/memforge",
            "api_url": "https://cloud.example",
            "api_token": "cloud-token",
        }
    ]
    assert json.loads(result.output)["running"] is True


def test_daemon_install_refuses_to_compete_with_unmanaged_runner(monkeypatch, tmp_path):
    manager = FakeServiceManager()
    manager.status = lambda: {"platform": "launchd", "installed": False, "loaded": False, "running": False}
    state_path = tmp_path / "state.json"
    state_path.write_text(
        json.dumps({"version": 1, "tasks": {}, "daemon": {"pid": os.getpid()}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(main, "_daemon_service_manager", lambda: manager)

    result = CliRunner().invoke(
        cli,
        ["daemon", "install"],
        env={
            "MEMFORGE_CLI_CONFIG": str(tmp_path / "cli.toml"),
            "MEMFORGE_LOCAL_AGENT_STATE": str(state_path),
            "MEMFORGE_LOCAL_AGENT_LOCK": str(tmp_path / "daemon.lock"),
            "MEMFORGE_API_URL": "https://cloud.example.hana.ondemand.com",
            "MEMFORGE_API_TOKEN": "cloud-token",
        },
    )

    assert result.exit_code == 1
    assert "foreground or unmanaged MemForge daemon is already running" in result.output
    assert manager.installs == []


def test_active_target_reuses_exact_daemon_keyring_credential(monkeypatch, tmp_path):
    config_path = tmp_path / "cli.toml"
    config_path.write_text(
        'active = "dev"\n\n[targets.dev]\n'
        'api_url = "https://cloud.example.hana.ondemand.com"\n'
        'token_env = "MEMFORGE_API_TOKEN"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMFORGE_CLI_CONFIG", str(config_path))
    monkeypatch.delenv("MEMFORGE_API_URL", raising=False)
    monkeypatch.delenv("MEMFORGE_API_TOKEN", raising=False)
    monkeypatch.setattr(
        main,
        "_daemon_keyring_api_token",
        lambda api_url: "stored-token" if api_url == "https://cloud.example.hana.ondemand.com" else None,
    )

    resolved = main._resolve_api_target(object())

    assert resolved.api_token == "stored-token"
    assert resolved.active_target == "dev"


def test_daemon_service_run_loads_stored_environment(monkeypatch, tmp_path):
    manager = FakeServiceManager()
    received: dict[str, object] = {}
    monkeypatch.setattr(main, "_daemon_service_manager", lambda: manager)
    monkeypatch.setattr(
        main,
        "_run_local_agent_daemon",
        lambda ctx, **kwargs: received.update(
            {
                **kwargs,
                "api_url": os.environ.get("MEMFORGE_API_URL"),
                "api_token": os.environ.get("MEMFORGE_API_TOKEN"),
            }
        ),
    )

    result = CliRunner().invoke(
        cli,
        ["daemon", "_run"],
        env={"MEMFORGE_CLI_CONFIG": str(tmp_path / "cli.toml")},
    )

    assert result.exit_code == 0, result.output
    assert received == {
        "browser": None,
        "poll_interval_seconds": 10,
        "cloud_job_wait_seconds": 25,
        "api_url": "https://stored.example",
        "api_token": "stored-token",
    }


def test_setup_configures_target_installs_service_and_verifies_heartbeat(monkeypatch, tmp_path):
    manager = FakeServiceManager()
    monkeypatch.delenv("MEMFORGE_API_URL", raising=False)
    monkeypatch.delenv("MEMFORGE_API_TOKEN", raising=False)
    monkeypatch.setattr(main, "_daemon_service_manager", lambda: manager)
    monkeypatch.setattr(main, "_current_cli_executable", lambda: "/opt/memforge/bin/memforge")

    class FakeToolClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def health(self):
            return {"status": "ok"}

        def get_local_agent_status(self):
            return {"status": "online", "last_seen_at": "2026-08-26T00:00:00+00:00"}

    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    config_path = tmp_path / "cli.toml"
    result = CliRunner().invoke(
        cli,
        ["setup", "--api-url", "https://cloud.example.hana.ondemand.com", "--target-name", "dev"],
        input="cloud-token\ncloud-token\n",
        env={
            "MEMFORGE_CLI_CONFIG": str(config_path),
            "MEMFORGE_API_URL": "https://wrong.example.hana.ondemand.com",
        },
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output[result.output.index("{") :])
    assert payload["ok"] is True
    assert payload["daemon"]["status"] == "online"
    assert manager.installs[0]["api_token"] == "cloud-token"
    assert manager.installs[0]["api_url"] == "https://cloud.example.hana.ondemand.com"
    assert 'active = "dev"' in config_path.read_text(encoding="utf-8")


def test_daemon_help_exposes_user_service_lifecycle():
    root_help = CliRunner().invoke(cli, ["--help"])
    result = CliRunner().invoke(cli, ["daemon", "--help"])

    assert root_help.exit_code == 0, root_help.output
    assert "daemon" in root_help.output
    assert "setup" in root_help.output
    assert result.exit_code == 0, result.output
    for command in ("install", "start", "stop", "restart", "status", "logs", "uninstall"):
        assert command in result.output
    assert "_run" not in result.output
