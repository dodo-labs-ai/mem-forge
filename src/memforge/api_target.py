"""Explicit immutable routing targets for OSS and Cloud API clients."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


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

    @property
    def api_base(self) -> str:
        return f"{self.origin}/api/v1"

    def resource_url(
        self,
        relative_path: str,
        *,
        workspace_id: str | None = None,
    ) -> str:
        if not relative_path.startswith("/") or relative_path.startswith("/api/"):
            raise ValueError("resource_path_must_be_relative_to_api_base")
        parsed = urlsplit(relative_path)
        query = parse_qsl(parsed.query, keep_blank_values=True)
        if any(key == "workspace_id" for key, _value in query):
            raise ValueError("workspace_id_must_use_explicit_parameter")
        normalized_workspace_id = _normalized_optional(workspace_id)
        if normalized_workspace_id is not None:
            query.append(("workspace_id", normalized_workspace_id))
        resource_path = urlunsplit(("", "", parsed.path, urlencode(query), ""))
        return f"{self.api_base}{resource_path}"


def build_target(
    *,
    origin: str | None,
) -> MemForgeTarget:
    """Build one canonical v1 target, deriving edition only from the hostname."""
    origin_value = _normalized_optional(origin)

    if origin_value is None:
        return MemForgeTarget(Edition.OSS, _LOCAL_OSS_ORIGIN)

    canonical_origin = _canonical_origin(origin_value)
    target_edition = Edition.CLOUD if _is_cloud_origin(canonical_origin) else Edition.OSS
    return MemForgeTarget(target_edition, canonical_origin)


def build_host_target(*, origin: str | None) -> MemForgeTarget:
    """Build a host-level target for APIs that are not workspace-routed."""
    return build_target(origin=origin)


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
