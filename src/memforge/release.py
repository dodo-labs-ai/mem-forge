"""The release a MemForge process runs.

A release is the code that computes and records work: the installed
``memforge-ai`` version, joined with the deployment revision when the deployer
configures one (``MEMFORGE_DEPLOYMENT_REVISION``, or the application version
Cloud Foundry publishes in ``VCAP_APPLICATION``). Every deployment of a
revision is therefore its own release.

Work records the release that produced it: a failed Source item is retried
once by each later release, and derivations and their cached model work are
never resumed or replayed by another release.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from functools import lru_cache
from importlib import metadata

PACKAGE_NAME = "memforge-ai"
# The version of a process that runs from a source tree without the package installed.
UNINSTALLED_VERSION = "0.0.0"

_REVISION_CHARACTERS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.:-")
_MAX_REVISION_CHARS = 255


@lru_cache(maxsize=1)
def installed_version() -> str:
    """The version of the installed ``memforge-ai`` distribution."""

    try:
        return metadata.version(PACKAGE_NAME)
    except metadata.PackageNotFoundError:
        return UNINSTALLED_VERSION


def current_deployment_revision() -> str | None:
    """The bounded deployment revision the deployer configured, if any."""

    configured = os.environ.get("MEMFORGE_DEPLOYMENT_REVISION", "").strip()
    if configured:
        return _safe_revision(configured)
    raw_vcap = os.environ.get("VCAP_APPLICATION", "").strip()
    if not raw_vcap:
        return None
    try:
        payload = json.loads(raw_vcap)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, Mapping):
        return None
    for key in ("application_version", "version"):
        value = payload.get(key)
        if isinstance(value, str) and (revision := _safe_revision(value)) is not None:
            return revision
    return None


def current_release() -> str:
    """The release this process runs: its version, and its deployment revision when one is configured."""

    revision = current_deployment_revision()
    version = installed_version()
    return f"{version}+{revision}" if revision is not None else version


def _safe_revision(value: str) -> str | None:
    if not value or len(value) > _MAX_REVISION_CHARS or any(ch not in _REVISION_CHARACTERS for ch in value):
        return None
    return value


__all__ = [
    "PACKAGE_NAME",
    "current_deployment_revision",
    "current_release",
    "installed_version",
]
