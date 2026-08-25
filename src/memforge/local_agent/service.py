"""Install and manage the local agent as an operating-system user service."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import subprocess
import sys
from typing import Any, Callable, Protocol, Sequence


LAUNCHD_LABEL = "com.memforge.local-daemon"
SYSTEMD_UNIT_NAME = "memforge-local-daemon.service"
KEYRING_SERVICE = "memforge-local-daemon"
SERVICE_CONFIG_VERSION = 1


class DaemonServiceError(RuntimeError):
    """Raised when the daemon user service cannot be configured or controlled."""


@dataclass(frozen=True)
class DaemonServicePaths:
    """User-owned files used by the daemon service."""

    config: Path
    stdout_log: Path
    stderr_log: Path

    @classmethod
    def for_home(cls, home: Path) -> DaemonServicePaths:
        state_dir = home / ".memforge"
        configured = os.getenv("MEMFORGE_DAEMON_SERVICE_CONFIG", "").strip()
        return cls(
            config=Path(configured).expanduser() if configured else state_dir / "daemon-service.json",
            stdout_log=state_dir / "local-agent-daemon.stdout.log",
            stderr_log=state_dir / "local-agent-daemon.stderr.log",
        )


@dataclass(frozen=True)
class StoredDaemonTarget:
    """Non-secret target configuration plus an optional keyring reference."""

    api_url: str
    keyring_account: str | None


@dataclass(frozen=True)
class DaemonCredentialSnapshot:
    """Restorable credential state used while replacing a service definition."""

    target: StoredDaemonTarget
    api_token: str | None


class SecretStore(Protocol):
    def get_password(self, service: str, username: str) -> str | None: ...

    def set_password(self, service: str, username: str, password: str) -> None: ...

    def delete_password(self, service: str, username: str) -> None: ...


def _system_keyring() -> SecretStore:
    try:
        import keyring
    except ImportError as exc:  # pragma: no cover - declared runtime dependency
        raise DaemonServiceError("The keyring package is required to install the daemon service.") from exc
    return keyring


def _atomic_write(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(data)
        temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class DaemonCredentialStore:
    """Keep target metadata on disk and its bearer token in the OS keyring."""

    def __init__(self, path: Path, *, keyring_store: SecretStore | None = None) -> None:
        self.path = path
        self._keyring = keyring_store

    @property
    def keyring(self) -> SecretStore:
        if self._keyring is None:
            self._keyring = _system_keyring()
        return self._keyring

    @staticmethod
    def _account(api_url: str) -> str:
        return hashlib.sha256(api_url.encode("utf-8")).hexdigest()

    def save(self, *, api_url: str, api_token: str | None) -> StoredDaemonTarget:
        normalized_url = api_url.rstrip("/")
        account = self._account(normalized_url) if api_token else None
        previous = self.load() if self.path.exists() else None
        previous_token = None
        if previous is not None and account and previous.keyring_account == account:
            try:
                previous_token = self.keyring.get_password(KEYRING_SERVICE, account)
            except Exception as exc:
                raise DaemonServiceError(
                    "Could not read the existing daemon API token from the operating-system keyring."
                ) from exc
        if account is not None:
            try:
                self.keyring.set_password(KEYRING_SERVICE, account, api_token)
            except Exception as exc:
                raise DaemonServiceError(
                    "Could not save the MemForge API token in the operating-system keyring."
                ) from exc
        payload = {
            "version": SERVICE_CONFIG_VERSION,
            "api_url": normalized_url,
            "keyring_account": account,
        }
        try:
            _atomic_write(
                self.path,
                json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n",
            )
        except Exception as exc:
            if account is not None:
                try:
                    if previous_token is not None:
                        self.keyring.set_password(KEYRING_SERVICE, account, previous_token)
                    elif previous is None or previous.keyring_account != account:
                        self.keyring.delete_password(KEYRING_SERVICE, account)
                except Exception:
                    pass
            raise DaemonServiceError(f"Could not write daemon service config: {self.path}") from exc
        return StoredDaemonTarget(api_url=normalized_url, keyring_account=account)

    def load(self) -> StoredDaemonTarget:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise DaemonServiceError("The daemon service is not configured. Run `memforge daemon install`.") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise DaemonServiceError(f"Could not read daemon service config: {self.path}") from exc
        if not isinstance(payload, dict) or payload.get("version") != SERVICE_CONFIG_VERSION:
            raise DaemonServiceError(f"Unsupported daemon service config: {self.path}")
        api_url = str(payload.get("api_url") or "").strip().rstrip("/")
        account = payload.get("keyring_account")
        if not api_url or (account is not None and not isinstance(account, str)):
            raise DaemonServiceError(f"Invalid daemon service config: {self.path}")
        return StoredDaemonTarget(api_url=api_url, keyring_account=account)

    def environment(self) -> dict[str, str]:
        target = self.load()
        environment = {"MEMFORGE_API_URL": target.api_url}
        if target.keyring_account:
            try:
                token = self.keyring.get_password(KEYRING_SERVICE, target.keyring_account)
            except Exception as exc:
                raise DaemonServiceError(
                    "Could not read the MemForge API token from the operating-system keyring."
                ) from exc
            if not token:
                raise DaemonServiceError(
                    "The daemon API token is missing from the operating-system keyring. "
                    "Run `memforge daemon install` again."
                )
            environment["MEMFORGE_API_TOKEN"] = token
        return environment

    def snapshot(self) -> DaemonCredentialSnapshot | None:
        if not self.path.exists():
            return None
        target = self.load()
        token = None
        if target.keyring_account:
            try:
                token = self.keyring.get_password(KEYRING_SERVICE, target.keyring_account)
            except Exception as exc:
                raise DaemonServiceError(
                    "Could not read the existing daemon API token from the operating-system keyring."
                ) from exc
            if not token:
                raise DaemonServiceError("The existing daemon API token is missing from the operating-system keyring.")
        return DaemonCredentialSnapshot(target=target, api_token=token)

    def delete_secret(self, target: StoredDaemonTarget, *, strict: bool = False) -> None:
        if not target.keyring_account:
            return
        try:
            self.keyring.delete_password(KEYRING_SERVICE, target.keyring_account)
        except Exception as exc:
            if exc.__class__.__name__ == "PasswordDeleteError":
                return
            if strict:
                raise DaemonServiceError(
                    "Could not delete the daemon API token from the operating-system keyring."
                ) from exc

    def restore(self, snapshot: DaemonCredentialSnapshot | None) -> None:
        current = self.load() if self.path.exists() else None
        if snapshot is None:
            self.delete(strict=False)
            return
        self.save(api_url=snapshot.target.api_url, api_token=snapshot.api_token)
        if current is not None and current.keyring_account != snapshot.target.keyring_account:
            self.delete_secret(current)

    def delete(self, *, strict: bool = True) -> None:
        try:
            target = self.load()
        except DaemonServiceError:
            self.path.unlink(missing_ok=True)
            return
        if target.keyring_account:
            self.delete_secret(target, strict=strict)
        self.path.unlink(missing_ok=True)


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _default_command_runner(args: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), **kwargs)  # noqa: S603 - arguments are constructed without a shell


class UserServiceAdapter(Protocol):
    platform: str
    unit_path: Path

    def install(self, command: Sequence[str]) -> None: ...

    def uninstall(self) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def restart(self) -> None: ...

    def status(self) -> dict[str, Any]: ...

    def logs(self, *, lines: int, follow: bool) -> int: ...


class _CommandAdapter:
    def __init__(self, *, runner: CommandRunner) -> None:
        self._runner = runner

    def _run(
        self,
        args: Sequence[str],
        *,
        check: bool = False,
        capture_output: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        try:
            return self._runner(
                list(args),
                check=check,
                capture_output=capture_output,
                text=True,
            )
        except FileNotFoundError as exc:
            raise DaemonServiceError(f"Required service manager command is unavailable: {args[0]}") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or str(exc)).strip()
            raise DaemonServiceError(detail or f"Service manager command failed: {' '.join(args)}") from exc


class LaunchdUserService(_CommandAdapter):
    """Manage a per-login macOS LaunchAgent."""

    platform = "launchd"

    def __init__(
        self,
        *,
        home: Path,
        uid: int,
        paths: DaemonServicePaths,
        runner: CommandRunner = _default_command_runner,
    ) -> None:
        super().__init__(runner=runner)
        self.unit_path = home / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"
        self.paths = paths
        self.domain = f"gui/{uid}"
        self.service_target = f"{self.domain}/{LAUNCHD_LABEL}"

    def _loaded(self) -> bool:
        return self._run(["launchctl", "print", self.service_target]).returncode == 0

    def install(self, command: Sequence[str]) -> None:
        self.paths.stdout_log.parent.mkdir(parents=True, exist_ok=True)
        for log_path in (self.paths.stdout_log, self.paths.stderr_log):
            log_path.touch(exist_ok=True)
            log_path.chmod(0o600)
        previous = self.unit_path.read_bytes() if self.unit_path.exists() else None
        was_loaded = self._loaded()
        payload = {
            "Label": LAUNCHD_LABEL,
            "ProgramArguments": list(command),
            "RunAtLoad": True,
            "KeepAlive": True,
            "ProcessType": "Background",
            "ThrottleInterval": 10,
            "StandardOutPath": str(self.paths.stdout_log),
            "StandardErrorPath": str(self.paths.stderr_log),
        }
        if was_loaded:
            self._run(["launchctl", "bootout", self.service_target], check=True)
        try:
            _atomic_write(self.unit_path, plistlib.dumps(payload, fmt=plistlib.FMT_XML))
            self._run(["launchctl", "bootstrap", self.domain, str(self.unit_path)], check=True)
        except Exception:
            self.unit_path.unlink(missing_ok=True)
            if previous is not None:
                _atomic_write(self.unit_path, previous)
                if was_loaded:
                    self._run(["launchctl", "bootstrap", self.domain, str(self.unit_path)])
            raise

    def uninstall(self) -> None:
        if self._loaded():
            self._run(["launchctl", "bootout", self.service_target], check=True)
        self.unit_path.unlink(missing_ok=True)

    def start(self) -> None:
        if not self.unit_path.exists():
            raise DaemonServiceError("The daemon service is not installed. Run `memforge daemon install`.")
        if self._loaded():
            self._run(["launchctl", "kickstart", self.service_target], check=True)
        else:
            self._run(["launchctl", "bootstrap", self.domain, str(self.unit_path)], check=True)

    def stop(self) -> None:
        if self._loaded():
            self._run(["launchctl", "bootout", self.service_target], check=True)

    def restart(self) -> None:
        if not self.unit_path.exists():
            raise DaemonServiceError("The daemon service is not installed. Run `memforge daemon install`.")
        if self._loaded():
            self._run(["launchctl", "kickstart", "-k", self.service_target], check=True)
        else:
            self._run(["launchctl", "bootstrap", self.domain, str(self.unit_path)], check=True)

    def status(self) -> dict[str, Any]:
        result = self._run(["launchctl", "print", self.service_target])
        output = result.stdout or ""
        state_match = re.search(r"^\s*state\s*=\s*(\S+)", output, re.MULTILINE)
        pid_match = re.search(r"^\s*pid\s*=\s*(\d+)", output, re.MULTILINE)
        loaded = result.returncode == 0
        return {
            "platform": self.platform,
            "installed": self.unit_path.exists(),
            "loaded": loaded,
            "running": loaded and bool(state_match and state_match.group(1) == "running"),
            "state": state_match.group(1) if state_match else ("unloaded" if not loaded else "unknown"),
            "pid": int(pid_match.group(1)) if pid_match else None,
            "unit_path": str(self.unit_path),
        }

    def logs(self, *, lines: int, follow: bool) -> int:
        args = ["tail", "-n", str(lines)]
        if follow:
            args.append("-f")
        args.extend([str(self.paths.stdout_log), str(self.paths.stderr_log)])
        return self._run(args, capture_output=False).returncode


class SystemdUserService(_CommandAdapter):
    """Manage a systemd user service on Linux."""

    platform = "systemd"

    def __init__(
        self,
        *,
        home: Path,
        paths: DaemonServicePaths,
        runner: CommandRunner = _default_command_runner,
    ) -> None:
        super().__init__(runner=runner)
        self.unit_path = home / ".config" / "systemd" / "user" / SYSTEMD_UNIT_NAME
        self.paths = paths

    @staticmethod
    def _unit(command: Sequence[str]) -> str:
        def quote(argument: str) -> str:
            escaped = argument.replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%")
            return f'"{escaped}"'

        exec_start = " ".join(quote(part) for part in command)
        return (
            "[Unit]\n"
            "Description=MemForge local agent daemon\n"
            "Wants=network-online.target\n"
            "After=network-online.target\n\n"
            "[Service]\n"
            f"ExecStart={exec_start}\n"
            "Restart=on-failure\n"
            "RestartSec=10\n\n"
            "[Install]\n"
            "WantedBy=default.target\n"
        )

    def install(self, command: Sequence[str]) -> None:
        previous = self.unit_path.read_bytes() if self.unit_path.exists() else None
        was_enabled = self._run(["systemctl", "--user", "is-enabled", SYSTEMD_UNIT_NAME]).returncode == 0
        was_active = self._run(["systemctl", "--user", "is-active", SYSTEMD_UNIT_NAME]).returncode == 0
        try:
            _atomic_write(self.unit_path, self._unit(command).encode("utf-8"))
            self._run(["systemctl", "--user", "daemon-reload"], check=True)
            self._run(["systemctl", "--user", "enable", "--now", SYSTEMD_UNIT_NAME], check=True)
        except Exception:
            self.unit_path.unlink(missing_ok=True)
            if previous is not None:
                _atomic_write(self.unit_path, previous)
            self._run(["systemctl", "--user", "daemon-reload"])
            if previous is not None and was_enabled:
                args = ["systemctl", "--user", "enable"]
                if was_active:
                    args.append("--now")
                args.append(SYSTEMD_UNIT_NAME)
                self._run(args)
            raise

    def uninstall(self) -> None:
        self._run(
            ["systemctl", "--user", "disable", "--now", SYSTEMD_UNIT_NAME],
            check=self.unit_path.exists(),
        )
        self.unit_path.unlink(missing_ok=True)
        self._run(["systemctl", "--user", "daemon-reload"], check=True)

    def start(self) -> None:
        if not self.unit_path.exists():
            raise DaemonServiceError("The daemon service is not installed. Run `memforge daemon install`.")
        self._run(["systemctl", "--user", "start", SYSTEMD_UNIT_NAME], check=True)

    def stop(self) -> None:
        self._run(["systemctl", "--user", "stop", SYSTEMD_UNIT_NAME], check=True)

    def restart(self) -> None:
        if not self.unit_path.exists():
            raise DaemonServiceError("The daemon service is not installed. Run `memforge daemon install`.")
        self._run(["systemctl", "--user", "restart", SYSTEMD_UNIT_NAME], check=True)

    def status(self) -> dict[str, Any]:
        active = self._run(["systemctl", "--user", "is-active", SYSTEMD_UNIT_NAME])
        enabled = self._run(["systemctl", "--user", "is-enabled", SYSTEMD_UNIT_NAME])
        state = (active.stdout or "").strip() or "inactive"
        return {
            "platform": self.platform,
            "installed": self.unit_path.exists(),
            "loaded": enabled.returncode == 0,
            "running": active.returncode == 0 and state == "active",
            "state": state,
            "pid": None,
            "unit_path": str(self.unit_path),
        }

    def logs(self, *, lines: int, follow: bool) -> int:
        args = ["journalctl", "--user", "--unit", SYSTEMD_UNIT_NAME, "--lines", str(lines)]
        if follow:
            args.append("--follow")
        return self._run(args, capture_output=False).returncode


def service_adapter(
    *,
    platform_name: str | None = None,
    home: Path | None = None,
    uid: int | None = None,
    runner: CommandRunner = _default_command_runner,
) -> UserServiceAdapter:
    """Return the native user-service adapter for this operating system."""

    selected_platform = platform_name or sys.platform
    selected_home = home or Path.home()
    paths = DaemonServicePaths.for_home(selected_home)
    if selected_platform == "darwin":
        return LaunchdUserService(
            home=selected_home,
            uid=os.getuid() if uid is None else uid,
            paths=paths,
            runner=runner,
        )
    if selected_platform.startswith("linux"):
        return SystemdUserService(home=selected_home, paths=paths, runner=runner)
    raise DaemonServiceError(
        "Automatic daemon service installation currently supports macOS launchd and Linux systemd."
    )


class DaemonServiceManager:
    """Own credentials and native user-service lifecycle behind one interface."""

    def __init__(self, *, adapter: UserServiceAdapter, credentials: DaemonCredentialStore) -> None:
        self.adapter = adapter
        self.credentials = credentials

    @classmethod
    def current(cls) -> DaemonServiceManager:
        home = Path.home()
        paths = DaemonServicePaths.for_home(home)
        return cls(
            adapter=service_adapter(home=home),
            credentials=DaemonCredentialStore(paths.config),
        )

    def install(self, *, executable: str, api_url: str, api_token: str | None) -> dict[str, Any]:
        previous = self.credentials.snapshot()
        current = self.credentials.save(api_url=api_url, api_token=api_token)
        try:
            self.adapter.install([executable, "daemon", "_run"])
        except Exception:
            self.credentials.restore(previous)
            raise
        if previous is not None and previous.target.keyring_account != current.keyring_account:
            self.credentials.delete_secret(previous.target)
        return self.status()

    def uninstall(self) -> dict[str, Any]:
        self.adapter.uninstall()
        self.credentials.delete()
        return self.status()

    def start(self) -> dict[str, Any]:
        self.adapter.start()
        return self.status()

    def stop(self) -> dict[str, Any]:
        self.adapter.stop()
        return self.status()

    def restart(self) -> dict[str, Any]:
        self.adapter.restart()
        return self.status()

    def status(self) -> dict[str, Any]:
        payload = self.adapter.status()
        payload["configured"] = self.credentials.path.exists()
        return payload

    def environment(self) -> dict[str, str]:
        return self.credentials.environment()

    def logs(self, *, lines: int, follow: bool) -> int:
        return self.adapter.logs(lines=lines, follow=follow)
