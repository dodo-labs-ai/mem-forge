"""Discover and validate origin-scoped MemForge service capabilities."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, Request, build_opener

from memforge_api_target import (
    CAPABILITY_PATH,
    MemForgeTarget,
    TargetConfigurationError,
    canonical_origin,
    target_from_capability_document,
)


MAX_CAPABILITY_BYTES = 16_384
CapabilityFetcher = Callable[[str], Mapping[str, Any]]


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


def discover_target(
    origin: str,
    *,
    fetcher: CapabilityFetcher | None = None,
    timeout_seconds: float = 10.0,
) -> MemForgeTarget:
    """Discover one origin's edition and protocol-v1 routing contract."""
    normalized_origin = canonical_origin(origin)
    capability_url = f"{normalized_origin}{CAPABILITY_PATH}"
    try:
        document = (
            fetcher(capability_url)
            if fetcher is not None
            else _fetch_capability_document(capability_url, timeout_seconds=timeout_seconds)
        )
        return target_from_capability_document(origin=normalized_origin, document=document)
    except TargetConfigurationError:
        raise
    except Exception as exc:
        raise TargetConfigurationError("memforge_capability_unavailable") from exc


def _fetch_capability_document(url: str, *, timeout_seconds: float) -> Mapping[str, Any]:
    request = Request(
        url,
        headers={"Accept": "application/json"},
        method="GET",
    )
    try:
        with build_opener(_NoRedirectHandler).open(request, timeout=timeout_seconds) as response:
            content_type = str(response.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()
            if content_type != "application/json":
                raise TargetConfigurationError("memforge_capability_content_type_invalid")
            raw = response.read(MAX_CAPABILITY_BYTES + 1)
    except HTTPError as exc:
        raise TargetConfigurationError("memforge_capability_unavailable") from exc
    except (OSError, URLError) as exc:
        raise TargetConfigurationError("memforge_capability_unavailable") from exc
    if len(raw) > MAX_CAPABILITY_BYTES:
        raise TargetConfigurationError("memforge_capability_too_large")
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TargetConfigurationError("memforge_capability_json_invalid") from exc
    if not isinstance(document, Mapping):
        raise TargetConfigurationError("memforge_capability_schema_invalid")
    return document
