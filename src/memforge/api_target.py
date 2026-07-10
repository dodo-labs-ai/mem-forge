"""Explicit immutable routing targets for OSS and Cloud API clients."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from urllib.parse import quote, urlsplit


_LOCAL_OSS_ORIGIN = "http://127.0.0.1:8765"
_CLOUD_HOST_SUFFIX = "hana.ondemand.com"


class Edition(str, Enum):
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
    origin: str | None,
    workspace_id: str | None,
) -> MemForgeTarget:
    """Build one canonical target, deriving Cloud only from the origin hostname."""
    origin_value = _normalized_optional(origin)
    workspace_value = _normalized_optional(workspace_id)

    if origin_value is None and workspace_value is None:
        return MemForgeTarget(Edition.OSS, _LOCAL_OSS_ORIGIN, None)

    canonical_origin = _canonical_origin(origin_value)
    target_edition = Edition.CLOUD if _is_cloud_origin(canonical_origin) else Edition.OSS
    if target_edition is Edition.CLOUD and workspace_value is None:
        raise TargetConfigurationError("cloud_workspace_required")
    if target_edition is Edition.OSS and workspace_value is not None:
        raise TargetConfigurationError("workspace_not_supported_for_oss")
    return MemForgeTarget(target_edition, canonical_origin, workspace_value)


def _normalized_optional(value: str | None) -> str | None:
    normalized = value.strip() if value is not None else ""
    return normalized or None


def _canonical_origin(origin: str | None) -> str:
    if origin is None:
        raise TargetConfigurationError("memforge_origin_required")
    try:
        parsed = urlsplit(origin)
        parsed.port
    except ValueError as exc:
        raise TargetConfigurationError("memforge_origin_required") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise TargetConfigurationError("memforge_origin_required")
    return origin.rstrip("/")


def _is_cloud_origin(origin: str) -> bool:
    hostname = (urlsplit(origin).hostname or "").lower().rstrip(".")
    return hostname == _CLOUD_HOST_SUFFIX or hostname.endswith(f".{_CLOUD_HOST_SUFFIX}")
