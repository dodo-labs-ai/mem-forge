"""Read the packages a device submitted for a Source, the same way in every connector.

A device-collected Source (local Markdown, GitHub repository local push, Jira
local agent) receives its items as packages: one JSON document per item, which
the device submits to the server. A run finds those packages in one of two
places:

- **the manifest**: ``config["local_agent_package_manifest"]``, a list of
  entries the server builds from the run's input snapshot. Each entry points
  at its stored package through ``package_uri``. When the key is present, even
  as an empty list, it is the run's whole set of packages;
- **the inbox**: a folder of ``*.json`` package files, read only when the
  server supplied no manifest.

:func:`device_packages` reads whichever applies and gives every package one
time, so a connector only turns each package into a ``ContentItem``::

    from memforge.genes.toolkit.local_packages import device_packages

    for package in device_packages(self.config, package_kind="local_markdown_document", inbox=self._documents_dir):
        if not package.changed_after(since):
            continue
        yield ContentItem(
            item_id=package.text("doc_id"),
            title=package.text("title", package.text("doc_id")),
            last_modified=package.last_modified,
            extra=dict(package.locator),
            ...
        )
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memforge.genes.local_adapter_packages import has_package_manifest, package_manifest

__all__ = ["DevicePackage", "device_packages", "package_time"]

logger = logging.getLogger(__name__)

_MANIFEST_LOCATOR_KEYS = ("package_uri", "package_path", "package_sha256", "input_sha256")


@dataclass(frozen=True)
class DevicePackage:
    """One submitted package, as a connector reads it during discovery.

    - ``fields`` is the manifest entry or, from the inbox, the package file's
      JSON object. Both name the item the same way (``doc_id``, ``title``,
      ``source_url``, ``version``, ``submitted_by`` and the connector's own
      keys).
    - ``last_modified`` is the package's time in UTC (see :func:`package_time`).
    - ``locator`` says where ``fetch`` reads the package body. Put it in the
      item's ``extra`` unchanged: ``package_uri``, ``package_path``,
      ``package_sha256`` and ``input_sha256`` for a manifest entry, and
      ``package_path`` for an inbox file.
    """

    fields: Mapping[str, Any]
    last_modified: datetime
    locator: Mapping[str, Any]

    def text(self, key: str, default: str = "") -> str:
        """A field as a string, or ``default`` when it is missing or empty::

        package.text("title", "PAY-1")   # "Payroll run fails" or "PAY-1"
        """
        return str(self.fields.get(key) or default)

    def changed_after(self, since: datetime | None) -> bool:
        """Whether an incremental run from ``since`` reads this package; every package when ``since`` is ``None``."""
        return since is None or self.last_modified > since


def package_time(fields: Mapping[str, Any]) -> datetime:
    """The time of one package, in UTC.

    The package's ``last_modified`` is used when it names one. A package
    without it is stamped with its ``submitted_at``, the time the device
    submitted it, which the server records on every package and manifest
    entry. A time without a timezone offset is read as UTC::

        package_time({"last_modified": "2026-07-07T11:30:00+02:00"})   # 2026-07-07 09:30 UTC
        package_time({"submitted_at": "2026-07-07T09:30:00Z"})         # 2026-07-07 09:30 UTC
        package_time({"submitted_at": "2026-07-07T09:30:00"})          # 2026-07-07 09:30 UTC

    The submission time stays the same each time a run reads the same
    package, and a package the device submits again carries a new one. So an
    unchanged package reads as the same provider revision run after run, and
    a resubmitted package as a new revision.

    Raises ``ValueError`` when the package names neither time, or names one
    that is not an ISO 8601 time.
    """
    value = fields.get("last_modified") or fields.get("submitted_at")
    if not value:
        raise ValueError(
            f"device package {fields.get('doc_id') or '?'} has neither last_modified nor submitted_at"
        )
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).strip())
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def device_packages(
    config: Mapping[str, Any],
    *,
    package_kind: str,
    inbox: Callable[[], Path],
) -> list[DevicePackage]:
    """Every package a run reads, ordered by time and then ``doc_id``.

    With a manifest in ``config``, its entries are the packages; an entry
    without a ``package_uri`` names no stored package and is left out. Without
    one, ``inbox()`` is called for the folder, and every ``*.json`` file in it
    or below it whose ``package_kind`` is ``package_kind`` is a package. A file
    that cannot be read or is not JSON is logged and left out; a file of
    another kind is not this Source's package.

    Raises ``ValueError`` for a package without a usable time (see
    :func:`package_time`), whatever ``inbox()`` raises when the Source has no
    inbox, and nothing for an empty manifest or folder.
    """
    if has_package_manifest(config):
        packages = [
            DevicePackage(
                fields=entry,
                last_modified=package_time(entry),
                locator={key: entry.get(key) for key in _MANIFEST_LOCATOR_KEYS},
            )
            for entry in package_manifest(config)
            if str(entry.get("package_uri") or "").strip()
        ]
    else:
        packages = [
            DevicePackage(
                fields=package,
                last_modified=package_time(package),
                locator={"package_path": str(path)},
            )
            for path, package in _inbox_packages(inbox(), package_kind)
        ]
    return sorted(packages, key=lambda package: (package.last_modified, package.text("doc_id")))


def _inbox_packages(folder: Path, package_kind: str) -> list[tuple[Path, Mapping[str, Any]]]:
    found: list[tuple[Path, Mapping[str, Any]]] = []
    for path in sorted(folder.rglob("*.json")):
        try:
            package = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("Skipping unreadable %s package: %s", package_kind, path)
            continue
        if isinstance(package, dict) and package.get("package_kind") == package_kind:
            found.append((path, package))
    return found
