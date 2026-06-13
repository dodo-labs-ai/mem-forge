"""Provider-neutral LiteLLM model helpers."""

from __future__ import annotations

__all__ = ["is_litellm_provider_model", "litellm_optional_kwargs"]


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
