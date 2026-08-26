from __future__ import annotations

import json
import os
import plistlib
import subprocess

from click.testing import CliRunner
import pytest

import memforge.main as main
import memforge.local_agent.service as daemon_service_module
from memforge.local_agent.service import (
    KEYRING_SERVICE,
    DaemonCredentialStore,
    DaemonLaunchSpec,
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
        elif command[:3] == ["systemctl", "--user", "enable"]:
            self.enabled = True
        elif command[:3] == ["systemctl", "--user", "disable"]:
            self.enabled = False
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


def test_credential_save_reports_config_and_keyring_rollback_failure(monkeypatch, tmp_path):
    class FailingRestoreKeyring(FakeKeyring):
        fail_restore = False

        def set_password(self, service: str, username: str, password: str) -> None:
            if self.fail_restore and password == "old-token":
                raise RuntimeError("keyring restore failed")
            super().set_password(service, username, password)

    keyring = FailingRestoreKeyring()
    store = DaemonCredentialStore(tmp_path / "daemon-service.json", keyring_store=keyring)
    target = store.save(api_url="https://same.example", api_token="old-token")
    keyring.fail_restore = True

    def fail_config_write(*args, **kwargs):
        raise OSError("config disk full")

    monkeypatch.setattr(daemon_service_module, "_atomic_write", fail_config_write)

    with pytest.raises(DaemonServiceError, match="config.*credential rollback failed.*keyring restore failed"):
        store.save(api_url="https://same.example", api_token="new-token")

    assert keyring.values[(KEYRING_SERVICE, target.keyring_account)] == "new-token"


def test_failed_service_replacement_restores_previous_credential(tmp_path):
    keyring = FakeKeyring()
    store = DaemonCredentialStore(tmp_path / "daemon-service.json", keyring_store=keyring)
    store.save(api_url="https://old.example", api_token="old-token")

    class FailingAdapter:
        platform = "fake"
        unit_path = tmp_path / "fake.service"

        def snapshot(self):
            return None

        def apply(self, launch):
            raise RuntimeError("install failed")

        def restore(self, snapshot):
            return None

    manager = DaemonServiceManager(adapter=FailingAdapter(), credentials=store)

    with pytest.raises(DaemonServiceError, match="replacement failed: install failed"):
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


def test_failed_service_replacement_reports_keyring_rollback_failure(tmp_path):
    class UndeletableKeyring(FakeKeyring):
        def delete_password(self, service: str, username: str) -> None:
            raise RuntimeError("keyring delete failed")

    class FailingAdapter:
        platform = "fake"
        unit_path = tmp_path / "fake.service"

        def snapshot(self):
            return None

        def apply(self, launch):
            raise RuntimeError("install failed")

        def restore(self, snapshot):
            raise AssertionError("service restore must not run before credentials are safe")

    manager = DaemonServiceManager(
        adapter=FailingAdapter(),
        credentials=DaemonCredentialStore(
            tmp_path / "daemon-service.json",
            keyring_store=UndeletableKeyring(),
        ),
    )

    with pytest.raises(DaemonServiceError, match="rollback failed.*keyring"):
        manager.install(
            executable="/opt/memforge/bin/memforge",
            api_url="https://new.example",
            api_token="new-token",
        )


def test_service_adapter_rejects_unsupported_platform_before_writing_state(tmp_path):
    with pytest.raises(DaemonServiceError, match="supports macOS launchd and Linux systemd"):
        service_adapter(platform_name="win32", home=tmp_path)

    assert list(tmp_path.rglob("*")) == []


def test_launchd_adapter_installs_login_service_without_shell_or_secrets(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", "/opt/homebrew/bin:/usr/bin:/bin")
    runner = FakeLaunchctl()
    paths = DaemonServicePaths.for_home(tmp_path)
    adapter = LaunchdUserService(home=tmp_path, uid=501, paths=paths, runner=runner)

    adapter.apply(
        DaemonLaunchSpec.for_executable(
            "/Users/test/.local/bin/memforge",
            inherited_path=os.environ["PATH"],
        )
    )

    payload = plistlib.loads(adapter.unit_path.read_bytes())
    assert payload["ProgramArguments"] == ["/Users/test/.local/bin/memforge", "daemon", "_run"]
    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] is True
    assert payload["EnvironmentVariables"]["PATH"].split(":") == [
        "/Users/test/.local/bin",
        "/opt/homebrew/bin",
        "/usr/bin",
        "/bin",
        "/usr/local/bin",
        "/usr/sbin",
        "/sbin",
    ]
    assert "MEMFORGE_API_TOKEN" not in payload["EnvironmentVariables"]
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
        manager = DaemonServiceManager(
            adapter=adapter,
            credentials=DaemonCredentialStore(tmp_path / "daemon-service.json", keyring_store=FakeKeyring()),
        )
        manager.install(
            executable="/Users/test/.local/bin/memforge",
            api_url="http://127.0.0.1:8765",
            api_token=None,
        )

    assert adapter.unit_path.read_bytes() == b"previous plist"
    assert runner.loaded is True


def test_launchd_adapter_reports_failed_rollback(tmp_path):
    class UnrestorableLaunchctl(FakeLaunchctl):
        def __init__(self) -> None:
            super().__init__()
            self.loaded = True
            self.bootstrap_count = 0

        def __call__(self, args, **kwargs) -> subprocess.CompletedProcess[str]:
            command = list(args)
            if command[:2] == ["launchctl", "bootstrap"]:
                self.commands.append(command)
                self.bootstrap_count += 1
                if self.bootstrap_count == 1:
                    raise subprocess.CalledProcessError(5, command, stderr="new bootstrap failed")
                return subprocess.CompletedProcess(command, 5, "", "old bootstrap failed")
            return super().__call__(args, **kwargs)

    runner = UnrestorableLaunchctl()
    paths = DaemonServicePaths.for_home(tmp_path)
    adapter = LaunchdUserService(home=tmp_path, uid=501, paths=paths, runner=runner)
    adapter.unit_path.parent.mkdir(parents=True)
    adapter.unit_path.write_bytes(b"previous plist")

    with pytest.raises(DaemonServiceError, match="rollback.*old bootstrap failed"):
        manager = DaemonServiceManager(
            adapter=adapter,
            credentials=DaemonCredentialStore(tmp_path / "daemon-service.json", keyring_store=FakeKeyring()),
        )
        manager.install(
            executable="/Users/test/.local/bin/memforge",
            api_url="http://127.0.0.1:8765",
            api_token=None,
        )


def test_verified_replacement_receipt_restores_previous_service_and_credential(tmp_path):
    runner = FakeLaunchctl()
    runner.loaded = True
    paths = DaemonServicePaths.for_home(tmp_path)
    adapter = LaunchdUserService(home=tmp_path, uid=501, paths=paths, runner=runner)
    adapter.unit_path.parent.mkdir(parents=True)
    adapter.unit_path.write_bytes(b"previous plist")
    keyring = FakeKeyring()
    credentials = DaemonCredentialStore(tmp_path / "daemon-service.json", keyring_store=keyring)
    credentials.save(api_url="https://old.example", api_token="old-token")
    manager = DaemonServiceManager(adapter=adapter, credentials=credentials)

    replacement = manager.replace(
        executable="/Users/test/.local/bin/memforge",
        api_url="https://new.example",
        api_token="new-token",
    )
    assert credentials.environment()["MEMFORGE_API_TOKEN"] == "new-token"
    assert adapter.unit_path.read_bytes() != b"previous plist"

    replacement.rollback()

    assert credentials.environment() == {
        "MEMFORGE_API_URL": "https://old.example",
        "MEMFORGE_API_TOKEN": "old-token",
    }
    assert adapter.unit_path.read_bytes() == b"previous plist"
    assert runner.loaded is True
    assert "new-token" not in keyring.values.values()


def test_install_rolls_back_when_old_keyring_credential_cannot_be_committed(tmp_path):
    class FailOldDeleteOnceKeyring(FakeKeyring):
        def __init__(self) -> None:
            super().__init__()
            self.fail_username: str | None = None

        def delete_password(self, service: str, username: str) -> None:
            if username == self.fail_username:
                self.fail_username = None
                raise RuntimeError("old keyring cleanup failed")
            super().delete_password(service, username)

    runner = FakeLaunchctl()
    runner.loaded = True
    paths = DaemonServicePaths.for_home(tmp_path)
    adapter = LaunchdUserService(home=tmp_path, uid=501, paths=paths, runner=runner)
    adapter.unit_path.parent.mkdir(parents=True)
    adapter.unit_path.write_bytes(b"previous plist")
    keyring = FailOldDeleteOnceKeyring()
    credentials = DaemonCredentialStore(tmp_path / "daemon-service.json", keyring_store=keyring)
    old_target = credentials.save(api_url="https://old.example", api_token="old-token")
    keyring.fail_username = old_target.keyring_account
    manager = DaemonServiceManager(adapter=adapter, credentials=credentials)

    with pytest.raises(DaemonServiceError, match="commit failed.*keyring"):
        manager.install(
            executable="/Users/test/.local/bin/memforge",
            api_url="https://new.example",
            api_token="new-token",
        )

    assert credentials.environment()["MEMFORGE_API_TOKEN"] == "old-token"
    assert adapter.unit_path.read_bytes() == b"previous plist"
    assert runner.loaded is True
    assert "new-token" not in keyring.values.values()


def test_systemd_adapter_installs_and_enables_user_service(tmp_path):
    runner = FakeSystemctl()
    paths = DaemonServicePaths.for_home(tmp_path)
    adapter = SystemdUserService(home=tmp_path, paths=paths, runner=runner)

    adapter.apply(DaemonLaunchSpec.for_executable("/home/test/.local/bin/memforge", inherited_path="/usr/bin:/bin"))

    unit = adapter.unit_path.read_text(encoding="utf-8")
    assert 'ExecStart="/home/test/.local/bin/memforge" "daemon" "_run"' in unit
    assert "Restart=on-failure" in unit
    assert 'Environment="PATH=/home/test/.local/bin:/usr/bin:/bin:/usr/local/bin:/usr/sbin:/sbin"' in unit
    assert "MEMFORGE_API_TOKEN" not in unit
    assert ["systemctl", "--user", "enable", "memforge-local-daemon.service"] in runner.commands
    assert ["systemctl", "--user", "start", "memforge-local-daemon.service"] in runner.commands
    assert adapter.status()["running"] is True

    adapter.stop()
    assert adapter.status()["state"] == "inactive"
    adapter.start()
    adapter.uninstall()

    assert not adapter.unit_path.exists()
    assert ["systemctl", "--user", "daemon-reload"] == runner.commands[-1]


def test_systemd_adapter_restarts_an_active_service_after_replacement(tmp_path):
    runner = FakeSystemctl()
    runner.enabled = True
    runner.active = True
    paths = DaemonServicePaths.for_home(tmp_path)
    adapter = SystemdUserService(home=tmp_path, paths=paths, runner=runner)

    adapter.apply(DaemonLaunchSpec.for_executable("/home/test/.local/bin/memforge", inherited_path="/usr/bin:/bin"))

    assert ["systemctl", "--user", "restart", "memforge-local-daemon.service"] in runner.commands


def test_systemd_adapter_reports_failed_rollback(tmp_path):
    class UnrestorableSystemctl(FakeSystemctl):
        def __init__(self) -> None:
            super().__init__()
            self.enabled = True
            self.active = True
            self.reload_count = 0

        def __call__(self, args, **kwargs) -> subprocess.CompletedProcess[str]:
            command = list(args)
            if command[:3] == ["systemctl", "--user", "daemon-reload"]:
                self.commands.append(command)
                self.reload_count += 1
                detail = "new reload failed" if self.reload_count == 1 else "old reload failed"
                raise subprocess.CalledProcessError(5, command, stderr=detail)
            return super().__call__(args, **kwargs)

    runner = UnrestorableSystemctl()
    paths = DaemonServicePaths.for_home(tmp_path)
    adapter = SystemdUserService(home=tmp_path, paths=paths, runner=runner)
    adapter.unit_path.parent.mkdir(parents=True)
    adapter.unit_path.write_text("previous unit", encoding="utf-8")
    manager = DaemonServiceManager(
        adapter=adapter,
        credentials=DaemonCredentialStore(tmp_path / "daemon-service.json", keyring_store=FakeKeyring()),
    )

    with pytest.raises(DaemonServiceError, match="rollback.*old reload failed"):
        manager.install(
            executable="/home/test/.local/bin/memforge",
            api_url="http://127.0.0.1:8765",
            api_token=None,
        )


def test_systemd_unit_quotes_arguments_and_escapes_specifiers():
    launch = DaemonLaunchSpec.for_executable(
        "/home/test user/100%/memforge",
        inherited_path="/usr/bin:/bin",
    )
    unit = SystemdUserService._unit(launch)

    assert 'ExecStart="/home/test user/100%%/memforge" "daemon" "_run"' in unit

    with pytest.raises(DaemonServiceError, match="control characters"):
        DaemonLaunchSpec.for_executable("/home/test\nuser/memforge", inherited_path="/usr/bin:/bin")


class FakeReplacement:
    def __init__(self, *, fail_commit: bool = False) -> None:
        self.committed = False
        self.rolled_back = False
        self.fail_commit = fail_commit

    def commit(self):
        if self.fail_commit:
            raise DaemonServiceError("keyring cleanup failed")
        self.committed = True

    def rollback(self):
        self.rolled_back = True


class FakeServiceManager:
    def __init__(self) -> None:
        self.installs: list[dict[str, str | None]] = []
        self.actions: list[str] = []
        self.replacements: list[FakeReplacement] = []
        self.fail_commit = False

    def replace(self, *, executable: str, api_url: str, api_token: str | None):
        self.actions.append("replace")
        self.installs.append({"executable": executable, "api_url": api_url, "api_token": api_token})
        replacement = FakeReplacement(fail_commit=self.fail_commit)
        self.replacements.append(replacement)
        return replacement

    def install(self, *, executable: str, api_url: str, api_token: str | None):
        replacement = self.replace(executable=executable, api_url=api_url, api_token=api_token)
        replacement.commit()
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
            self.statuses = iter(
                [
                    {"status": "online", "last_seen_at": "2026-08-25T23:59:00+00:00"},
                    {"status": "online", "last_seen_at": "2026-08-26T00:00:00+00:00"},
                ]
            )

        def health(self):
            return {"status": "ok"}

        def get_local_agent_status(self):
            return next(self.statuses)

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
    assert payload["daemon"]["last_seen_at"] == "2026-08-26T00:00:00+00:00"
    assert manager.installs[0]["api_token"] == "cloud-token"
    assert manager.installs[0]["api_url"] == "https://cloud.example.hana.ondemand.com"
    assert 'active = "dev"' in config_path.read_text(encoding="utf-8")


def test_setup_captures_heartbeat_baseline_after_replacing_old_daemon(monkeypatch, tmp_path):
    events: list[str] = []
    manager = FakeServiceManager()
    original_replace = manager.replace

    def replace(**kwargs):
        events.append("replace")
        return original_replace(**kwargs)

    manager.replace = replace
    monkeypatch.setattr(main, "_daemon_service_manager", lambda: manager)
    monkeypatch.setattr(main, "_current_cli_executable", lambda: "/opt/memforge/bin/memforge")

    class FakeToolClient:
        def __init__(self, **kwargs):
            self.statuses = iter(
                [
                    {"status": "online", "last_seen_at": "2026-08-26T00:00:00+00:00"},
                    {"status": "online", "last_seen_at": "2026-08-26T00:00:25+00:00"},
                ]
            )

        def health(self):
            return {"status": "ok"}

        def get_local_agent_status(self):
            events.append("heartbeat")
            return next(self.statuses)

    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    result = CliRunner().invoke(
        cli,
        ["setup", "--api-url", "http://127.0.0.1:8765"],
        env={"MEMFORGE_CLI_CONFIG": str(tmp_path / "cli.toml")},
    )

    assert result.exit_code == 0, result.output
    assert events == ["replace", "heartbeat", "heartbeat"]


def test_setup_guides_a_new_cloud_user_without_preconfigured_target(monkeypatch, tmp_path):
    manager = FakeServiceManager()
    monkeypatch.delenv("MEMFORGE_API_URL", raising=False)
    monkeypatch.delenv("MEMFORGE_API_TOKEN", raising=False)
    monkeypatch.setattr(main, "_daemon_service_manager", lambda: manager)
    monkeypatch.setattr(main, "_current_cli_executable", lambda: "/opt/memforge/bin/memforge")

    class FakeToolClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.statuses = iter(
                [
                    {"status": "offline", "last_seen_at": None},
                    {"status": "online", "last_seen_at": "2026-08-26T00:00:00+00:00"},
                ]
            )

        def health(self):
            return {"status": "ok"}

        def get_local_agent_status(self):
            return next(self.statuses)

    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    config_path = tmp_path / "cli.toml"
    result = CliRunner().invoke(
        cli,
        ["setup"],
        input="https://cloud.example.hana.ondemand.com\ncloud-token\ncloud-token\n",
        env={"MEMFORGE_CLI_CONFIG": str(config_path)},
    )

    assert result.exit_code == 0, result.output
    assert manager.installs[0]["api_url"] == "https://cloud.example.hana.ondemand.com"
    assert manager.installs[0]["api_token"] == "cloud-token"
    assert 'api_url = "https://cloud.example.hana.ondemand.com"' in config_path.read_text(encoding="utf-8")


def test_setup_preserves_previous_target_when_health_validation_fails(monkeypatch, tmp_path):
    config_path = tmp_path / "cli.toml"
    original = (
        'active = "old"\n\n[targets.old]\n'
        'api_url = "https://old.example.hana.ondemand.com"\n'
        'token_env = "MEMFORGE_API_TOKEN"\n'
    )
    config_path.write_text(original, encoding="utf-8")
    manager = FakeServiceManager()
    monkeypatch.setattr(main, "_daemon_service_manager", lambda: manager)

    class UnhealthyToolClient:
        def __init__(self, **kwargs):
            pass

        def health(self):
            return {"error": "MemForge API unavailable"}

    monkeypatch.setattr(main, "ToolClient", UnhealthyToolClient)
    result = CliRunner().invoke(
        cli,
        ["setup", "--api-url", "https://new.example.hana.ondemand.com"],
        input="new-token\nnew-token\n",
        env={"MEMFORGE_CLI_CONFIG": str(config_path)},
    )

    assert result.exit_code == 1
    assert config_path.read_text(encoding="utf-8") == original
    assert manager.installs == []


def test_setup_rolls_back_service_and_target_when_fresh_heartbeat_never_arrives(monkeypatch, tmp_path):
    config_path = tmp_path / "cli.toml"
    original = (
        'active = "old"\n\n[targets.old]\n'
        'api_url = "https://old.example.hana.ondemand.com"\n'
        'token_env = "MEMFORGE_API_TOKEN"\n'
    )
    config_path.write_text(original, encoding="utf-8")
    manager = FakeServiceManager()
    monkeypatch.setattr(main, "_daemon_service_manager", lambda: manager)
    monkeypatch.setattr(main, "_current_cli_executable", lambda: "/opt/memforge/bin/memforge")
    monkeypatch.setattr(
        main,
        "_wait_for_local_agent_online",
        lambda *args, **kwargs: {
            "status": "online",
            "last_seen_at": "2026-08-25T23:59:00+00:00",
            "fresh": False,
        },
    )

    class FakeToolClient:
        def __init__(self, **kwargs):
            pass

        def health(self):
            return {"status": "ok"}

        def get_local_agent_status(self):
            return {"status": "online", "last_seen_at": "2026-08-25T23:59:00+00:00"}

    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    result = CliRunner().invoke(
        cli,
        ["setup", "--api-url", "https://new.example.hana.ondemand.com"],
        input="new-token\nnew-token\n",
        env={"MEMFORGE_CLI_CONFIG": str(config_path)},
    )

    assert result.exit_code == 1
    assert manager.replacements[0].rolled_back is True
    assert manager.replacements[0].committed is False
    assert config_path.read_text(encoding="utf-8") == original


def test_setup_rolls_back_service_when_target_commit_fails(monkeypatch, tmp_path):
    manager = FakeServiceManager()
    monkeypatch.setattr(main, "_daemon_service_manager", lambda: manager)
    monkeypatch.setattr(main, "_current_cli_executable", lambda: "/opt/memforge/bin/memforge")
    monkeypatch.setattr(
        main,
        "_wait_for_local_agent_online",
        lambda *args, **kwargs: {
            "status": "online",
            "last_seen_at": "2026-08-26T00:00:00+00:00",
            "fresh": True,
        },
    )

    def fail_target_commit(**kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(main, "_set_cli_target", fail_target_commit)

    class FakeToolClient:
        def __init__(self, **kwargs):
            pass

        def health(self):
            return {"status": "ok"}

        def get_local_agent_status(self):
            return {"status": "offline", "last_seen_at": None}

    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    result = CliRunner().invoke(
        cli,
        ["setup", "--api-url", "http://127.0.0.1:8765"],
        env={"MEMFORGE_CLI_CONFIG": str(tmp_path / "cli.toml")},
    )

    assert result.exit_code == 1
    assert "Could not save the CLI target: disk full" in result.output
    assert manager.replacements[0].rolled_back is True
    assert manager.replacements[0].committed is False


def test_setup_restores_cli_target_and_service_when_replacement_commit_fails(monkeypatch, tmp_path):
    config_path = tmp_path / "cli.toml"
    original = (
        'active = "old"\n\n[targets.old]\n'
        'api_url = "https://old.example.hana.ondemand.com"\n'
        'token_env = "MEMFORGE_API_TOKEN"\n'
    )
    config_path.write_text(original, encoding="utf-8")
    manager = FakeServiceManager()
    manager.fail_commit = True
    monkeypatch.setattr(main, "_daemon_service_manager", lambda: manager)
    monkeypatch.setattr(main, "_current_cli_executable", lambda: "/opt/memforge/bin/memforge")
    monkeypatch.setattr(
        main,
        "_wait_for_local_agent_online",
        lambda *args, **kwargs: {
            "status": "online",
            "last_seen_at": "2026-08-26T00:00:00+00:00",
            "fresh": True,
        },
    )

    class FakeToolClient:
        def __init__(self, **kwargs):
            pass

        def health(self):
            return {"status": "ok"}

        def get_local_agent_status(self):
            return {"status": "offline", "last_seen_at": None}

    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    result = CliRunner().invoke(
        cli,
        ["setup", "--api-url", "https://new.example.hana.ondemand.com"],
        input="new-token\nnew-token\n",
        env={"MEMFORGE_CLI_CONFIG": str(config_path)},
    )

    assert result.exit_code == 1
    assert "Could not commit daemon setup" in result.output
    assert manager.replacements[0].rolled_back is True
    assert config_path.read_text(encoding="utf-8") == original


def test_daemon_status_uses_the_daemons_stored_target_for_server_connection(monkeypatch, tmp_path):
    manager = FakeServiceManager()
    captured: dict[str, object] = {}
    monkeypatch.setattr(main, "_daemon_service_manager", lambda: manager)
    monkeypatch.setattr(
        main,
        "_local_agent_status_payload",
        lambda ctx, verbose: {"status": "running", "daemon": {"pid": 123}},
    )

    class FakeToolClient:
        def __init__(self, *, target, api_token):
            captured.update({"api_url": target.origin, "api_token": api_token})

        def get_local_agent_status(self):
            return {
                "status": "online",
                "last_seen_at": "2026-08-26T00:00:00+00:00",
                "stale_after_seconds": 90,
            }

    monkeypatch.setattr(main, "ToolClient", FakeToolClient)
    result = CliRunner().invoke(
        cli,
        ["daemon", "status"],
        env={"MEMFORGE_CLI_CONFIG": str(tmp_path / "cli.toml")},
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "healthy"
    assert payload["connection"]["status"] == "online"
    assert captured == {"api_url": "https://stored.example", "api_token": "stored-token"}


def test_daemon_check_exits_nonzero_when_server_heartbeat_is_offline(monkeypatch, tmp_path):
    manager = FakeServiceManager()
    monkeypatch.setattr(main, "_daemon_service_manager", lambda: manager)
    monkeypatch.setattr(
        main,
        "_local_agent_status_payload",
        lambda ctx, verbose: {"status": "running", "daemon": {"pid": 123}},
    )

    class OfflineToolClient:
        def __init__(self, **kwargs):
            pass

        def get_local_agent_status(self):
            return {"status": "offline", "last_seen_at": None, "stale_after_seconds": 90}

    monkeypatch.setattr(main, "ToolClient", OfflineToolClient)
    result = CliRunner().invoke(
        cli,
        ["daemon", "check"],
        env={"MEMFORGE_CLI_CONFIG": str(tmp_path / "cli.toml")},
    )

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["status"] == "unhealthy"
    assert payload["connection"]["status"] == "offline"
    assert payload["error"] == "MemForge daemon health check failed."


def test_daemon_help_exposes_user_service_lifecycle():
    root_help = CliRunner().invoke(cli, ["--help"])
    result = CliRunner().invoke(cli, ["daemon", "--help"])
    setup_help = CliRunner().invoke(cli, ["setup", "--help"])

    assert root_help.exit_code == 0, root_help.output
    assert "daemon" in root_help.output
    assert "setup" in root_help.output
    assert result.exit_code == 0, result.output
    for command in ("install", "start", "stop", "restart", "status", "check", "logs", "uninstall"):
        assert command in result.output
    assert "_run" not in result.output
    assert "--no-wait" not in setup_help.output
