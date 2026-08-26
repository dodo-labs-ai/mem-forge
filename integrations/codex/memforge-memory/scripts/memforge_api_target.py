"""Explicit immutable routing targets for OSS and Cloud API clients."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


_LOCAL_OSS_ORIGIN = "http://127.0.0.1:8765"
CAPABILITY_PATH = "/.well-known/memforge"
CAPABILITY_PROTOCOL = "memforge"
CAPABILITY_PROTOCOL_VERSION = 1


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
    api_base_path: str
    health_path: str
    authentication_required: bool
    authentication_scheme: str

    @property
    def api_base(self) -> str:
        return f"{self.origin}{self.api_base_path}"

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
    edition: Edition | str | None = None,
) -> MemForgeTarget:
    """Build one canonical target from an explicit or local-default edition."""
    origin_value = _normalized_optional(origin)

    if origin_value is None:
        if edition not in {None, Edition.OSS, Edition.OSS.value}:
            raise TargetConfigurationError("local_target_must_use_oss_edition")
        return _target_for_edition(_LOCAL_OSS_ORIGIN, Edition.OSS)

    normalized_origin = canonical_origin(origin_value)
    if edition is None:
        raise TargetConfigurationError("memforge_edition_required")
    try:
        target_edition = Edition(edition)
    except ValueError as exc:
        raise TargetConfigurationError("memforge_edition_invalid") from exc
    return _target_for_edition(normalized_origin, target_edition)


def build_host_target(
    *,
    origin: str | None,
    edition: Edition | str | None = None,
) -> MemForgeTarget:
    """Build a host-level target for APIs that are not workspace-routed."""
    return build_target(origin=origin, edition=edition)


def capability_document(edition: Edition) -> dict[str, Any]:
    """Return the public protocol-v1 capability document for one edition."""
    target = _target_for_edition("http://capability.invalid", edition)
    return {
        "protocol": CAPABILITY_PROTOCOL,
        "protocol_version": CAPABILITY_PROTOCOL_VERSION,
        "edition": edition.value,
        "api_base": target.api_base_path,
        "health_path": target.health_path,
        "authentication": {
            "required": target.authentication_required,
            "scheme": target.authentication_scheme,
        },
    }


def target_from_capability_document(
    *,
    origin: str,
    document: Mapping[str, Any],
) -> MemForgeTarget:
    """Validate one protocol-v1 discovery document and bind it to its origin."""
    expected_keys = {
        "protocol",
        "protocol_version",
        "edition",
        "api_base",
        "health_path",
        "authentication",
    }
    if set(document) != expected_keys:
        raise TargetConfigurationError("memforge_capability_schema_invalid")
    if document.get("protocol") != CAPABILITY_PROTOCOL:
        raise TargetConfigurationError("memforge_capability_protocol_invalid")
    version = document.get("protocol_version")
    if type(version) is not int or version != CAPABILITY_PROTOCOL_VERSION:
        raise TargetConfigurationError("memforge_capability_version_unsupported")
    try:
        edition = Edition(document.get("edition"))
    except (TypeError, ValueError) as exc:
        raise TargetConfigurationError("memforge_capability_edition_invalid") from exc
    authentication = document.get("authentication")
    if not isinstance(authentication, Mapping) or set(authentication) != {"required", "scheme"}:
        raise TargetConfigurationError("memforge_capability_auth_invalid")
    required = authentication.get("required")
    scheme = authentication.get("scheme")
    expected_scheme = "bearer" if required is True else "none"
    if type(required) is not bool or scheme != expected_scheme:
        raise TargetConfigurationError("memforge_capability_auth_invalid")
    api_base_path = _capability_path(document.get("api_base"), field="api_base")
    health_path = _capability_path(document.get("health_path"), field="health_path")
    expected = _target_for_edition(canonical_origin(origin), edition)
    if (
        api_base_path != expected.api_base_path
        or health_path != expected.health_path
        or required != expected.authentication_required
    ):
        raise TargetConfigurationError("memforge_capability_contract_invalid")
    return expected


def _target_for_edition(origin: str, edition: Edition) -> MemForgeTarget:
    if edition is Edition.CLOUD:
        return MemForgeTarget(
            edition=edition,
            origin=origin,
            api_base_path="/api/v1",
            health_path="/healthz",
            authentication_required=True,
            authentication_scheme="bearer",
        )
    return MemForgeTarget(
        edition=edition,
        origin=origin,
        api_base_path="/api/v1",
        health_path="/api/v1/health",
        authentication_required=False,
        authentication_scheme="none",
    )


def _capability_path(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or value.startswith("//"):
        raise TargetConfigurationError(f"memforge_capability_{field}_invalid")
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise TargetConfigurationError(f"memforge_capability_{field}_invalid")
    return value


def _normalized_optional(value: str | None) -> str | None:
    normalized = value.strip() if value is not None else ""
    return normalized or None


def canonical_origin(origin: str | None) -> str:
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
