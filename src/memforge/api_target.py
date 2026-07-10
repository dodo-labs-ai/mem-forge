"""Explicit immutable routing targets for OSS and Cloud API clients."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from urllib.parse import quote, urlsplit


_LOCAL_OSS_ORIGIN = "http://127.0.0.1:8765"


class Edition(StrEnum):
    OSS = "oss"
    CLOUD = "cloud"


class TargetConfigurationError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class MemForgeTarget:
    edition: Edition
    origin: str
    workspace_id: str | None

    @property
    def workspace_api_base(self) -> str:
        if self.edition is Edition.OSS:
            return f"{self.origin}/api"
        workspace = quote(self.workspace_id or "", safe="")
        return f"{self.origin}/api/workspaces/{workspace}/api"

    def resource_url(self, relative_path: str) -> str:
        if not relative_path.startswith("/") or relative_path.startswith("/api/"):
            raise ValueError("resource_path_must_be_relative_to_api_base")
        return f"{self.workspace_api_base}{relative_path}"


def build_target(
    *,
    edition: str | Edition | None,
    origin: str | None,
    workspace_id: str | None,
) -> MemForgeTarget:
    """Build one canonical target from the explicit edition-tagged configuration."""
    edition_value = _normalized_optional(edition)
    origin_value = _normalized_optional(origin)
    workspace_value = _normalized_optional(workspace_id)

    if edition_value is None and origin_value is None and workspace_value is None:
        return MemForgeTarget(Edition.OSS, _LOCAL_OSS_ORIGIN, None)
    if edition_value is None:
        raise TargetConfigurationError("memforge_edition_required")

    try:
        target_edition = Edition(edition_value)
    except ValueError as exc:
        raise TargetConfigurationError("invalid_memforge_edition") from exc

    if target_edition is Edition.OSS:
        if workspace_value is not None:
            raise TargetConfigurationError("workspace_not_supported_for_oss")
        if origin_value is None:
            raise TargetConfigurationError("memforge_origin_required")
    else:
        if origin_value is None:
            raise TargetConfigurationError("cloud_api_url_required")
        if workspace_value is None:
            raise TargetConfigurationError("cloud_workspace_required")

    canonical_origin = _canonical_origin(origin_value)
    return MemForgeTarget(target_edition, canonical_origin, workspace_value)


def _normalized_optional(value: str | None) -> str | None:
    normalized = value.strip() if value is not None else ""
    return normalized or None


def _canonical_origin(origin: str) -> str:
    try:
        parsed = urlsplit(origin)
    except ValueError as exc:
        raise TargetConfigurationError("memforge_origin_required") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise TargetConfigurationError("memforge_origin_required")
    return origin.rstrip("/")
