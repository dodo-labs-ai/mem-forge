"""How ``memforge upgrade`` installs exactly the server's MemForge version (ADR 0058).

A device acquires under the acquisition contracts of the version it runs, so
it takes a type's jobs only while that version's contract equals the
server's. ``memforge upgrade`` installs the server's version, behind or ahead
alike, the way this installation was installed. It reads that way from where
its own executable lives: a pipx installation keeps the executable in a
virtual environment that pipx describes with ``pipx_metadata.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

__all__ = ["InstallMethod", "UnknownInstallMethod", "install_method"]

PIPX_METADATA_FILENAME = "pipx_metadata.json"


class UnknownInstallMethod(ValueError):
    """This installation was installed in a way ``memforge upgrade`` cannot repeat."""


@dataclass(frozen=True, slots=True)
class InstallMethod:
    """One way MemForge was installed, and the package it installed."""

    name: str
    package: str

    def install_command(self, version: str) -> list[str]:
        """The command that installs exactly ``version`` in place of the current one."""

        return ["pipx", "install", "--force", f"{self.package}=={version}"]


def install_method(executable: Path) -> InstallMethod:
    """How the installation that owns ``executable`` was installed.

    Raises :class:`UnknownInstallMethod` for an installation outside pipx,
    such as a development checkout, which its owner updates themselves.
    """

    resolved = executable.resolve()
    for directory in resolved.parents:
        if (directory / PIPX_METADATA_FILENAME).is_file() and directory.parent.name == "venvs":
            return InstallMethod(name="pipx", package=directory.name)
    raise UnknownInstallMethod(f"{executable} is not part of a pipx installation")
