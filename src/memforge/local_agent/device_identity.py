"""Who this daemon installation is: its device id and the name its owner sees (ADR 0058).

``memforge setup`` gives each installation one device id and a name, the
host name unless the owner names it, and keeps them in the daemon's
configuration directory beside the service configuration. A daemon run on an
installation that was never set up gets its identity the first time it
starts. The id never changes; setup may rename the device.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path

from memforge.local_agent.device_report import DEVICE_NAME_MAX_CHARS

__all__ = ["DeviceIdentity", "DeviceIdentityStore", "checked_device_name", "device_identity_path"]

DEVICE_IDENTITY_FILENAME = "device.json"
DEVICE_IDENTITY_VERSION = 1


def device_identity_path() -> Path:
    """Where this installation keeps its device identity: beside the daemon service configuration."""

    from memforge.local_agent.service import DaemonServicePaths

    return DaemonServicePaths.for_home(Path.home()).config.with_name(DEVICE_IDENTITY_FILENAME)


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    device_id: str
    name: str


def checked_device_name(name: str | None) -> str:
    """The device name the owner gave, stripped; empty when none. Raises ``ValueError`` when it is too long."""

    wanted = (name or "").strip()
    if len(wanted) > DEVICE_NAME_MAX_CHARS:
        raise ValueError(f"A device name has at most {DEVICE_NAME_MAX_CHARS} characters")
    return wanted


def _host_name() -> str:
    name = socket.gethostname().strip().removesuffix(".local") or "This computer"
    return name[:DEVICE_NAME_MAX_CHARS]


class DeviceIdentityStore:
    """Reads and keeps one installation's device identity."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> DeviceIdentity | None:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read the device identity: {self.path}") from exc
        device_id = payload.get("device_id") if isinstance(payload, dict) else None
        name = payload.get("name") if isinstance(payload, dict) else None
        if payload.get("version") != DEVICE_IDENTITY_VERSION or not isinstance(device_id, str) or not device_id:
            raise ValueError(f"Invalid device identity: {self.path}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid device identity: {self.path}")
        return DeviceIdentity(device_id=device_id, name=name)

    def ensure(self, name: str | None = None) -> DeviceIdentity:
        """The installation's identity, created on first use; ``name`` renames the device."""

        current = self.load()
        wanted = checked_device_name(name)
        if current is not None and (not wanted or wanted == current.name):
            return current
        identity = DeviceIdentity(
            device_id=current.device_id if current is not None else f"dev-{uuid.uuid4().hex}",
            name=wanted or (current.name if current is not None else _host_name()),
        )
        self._write(identity)
        return identity

    def _write(self, identity: DeviceIdentity) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            temporary.write_text(
                json.dumps(
                    {"version": DEVICE_IDENTITY_VERSION, "device_id": identity.device_id, "name": identity.name},
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.chmod(0o600)
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
