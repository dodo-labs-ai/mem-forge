"""Provider-neutral LiteLLM model helpers."""

from __future__ import annotations

from urllib.parse import urlsplit

__all__ = ["endpoint_origin", "is_litellm_provider_model", "litellm_optional_kwargs"]

_DEFAULT_PORTS = {"http": 80, "https": 443}


def is_litellm_provider_model(model: str | None) -> bool:
    """Return true when a model is already in LiteLLM provider/model form."""
    value = (model or "").strip()
    return bool(value and "/" in value and "://" not in value)


def litellm_optional_kwargs(
    *,
    api_base: str | None,
    api_key: str | None,
) -> dict[str, str]:
    """Return only explicit LiteLLM credential kwargs.

    Some LiteLLM providers resolve credentials from environment variables
    or service bindings. Passing empty ``api_key`` / ``api_base`` values
    can override that provider-native resolution, so callers include these
    kwargs only when the user configured concrete values.
    """
    kwargs: dict[str, str] = {}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    return kwargs


def endpoint_origin(url: str | None) -> tuple[str, str, int] | None:
    """Return the normalized ``(scheme, host, port)`` an endpoint URL addresses.

    Two URLs with the same origin reach the same server, whatever their paths.
    Scheme and host compare case-insensitively and a missing port means the
    scheme's default. A value that is not an absolute HTTP(S) URL has no origin.
    """
    try:
        parsed = urlsplit((url or "").strip())
        port = parsed.port
    except ValueError:
        return None
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").rstrip(".")
    if scheme not in _DEFAULT_PORTS or not host:
        return None
    return scheme, host, port if port is not None else _DEFAULT_PORTS[scheme]
