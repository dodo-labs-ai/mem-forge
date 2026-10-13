"""Model provider descriptors and the invocation a model request is made from.

A provider descriptor states what MemForge knows about one way of reaching a
model: which kinds of model it serves, where its endpoint is, how it
authenticates, how a model id becomes a LiteLLM route, and whether its endpoint
can list models. Settings name a descriptor; nothing else about a provider is
configured.

A ``ResolvedInvocation`` is everything the LiteLLM SDK is given for one
configured model: the route, the endpoint, the credential and a short list of
permitted extra parameters. The generation client and the Embedding Runtime are
built from one and pass the SDK nothing else about where a request goes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, get_args
from urllib.parse import urlsplit

import litellm

__all__ = [
    "AUTH_STRATEGIES",
    "KEYLESS_API_KEY",
    "MODEL_KINDS",
    "PROVIDER_DESCRIPTORS",
    "AuthStrategy",
    "ModelConfigError",
    "ModelKind",
    "ProviderDescriptor",
    "ResolvedInvocation",
    "endpoint_carries_credentials",
    "endpoint_origin",
    "litellm_optional_kwargs",
    "names_litellm_route",
    "origin_label",
    "provider_descriptor",
]

ModelKind = Literal["generation", "embedding"]
AuthStrategy = Literal["api_key", "none", "environment"]

MODEL_KINDS: tuple[ModelKind, ...] = get_args(ModelKind)
AUTH_STRATEGIES: tuple[AuthStrategy, ...] = get_args(AuthStrategy)

# The SDK builds no request for an OpenAI-compatible route without a key, and
# reads one from the process environment when it is given none. An endpoint
# that needs no key is sent this value, so a key from the environment is never
# sent to an endpoint the operator did not pair it with.
KEYLESS_API_KEY = "no-api-key"

_DEFAULT_PORTS = {"http": 80, "https": 443}

# Every extra parameter an invocation may carry to the SDK, with the type its
# value must have.
_EXTRA_PARAMETER_TYPES: Mapping[str, type] = MappingProxyType(
    {
        "dimensions": int,
        "organization": str,
    }
)


class ModelConfigError(ValueError):
    """Model settings that cannot be used, named by a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class ResolvedInvocation:
    """What the SDK is given for every request to one configured model."""

    # The LiteLLM ``provider/model`` route.
    route: str
    api_base: str | None = None
    api_key: str | None = field(default=None, repr=False)
    extra_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in self.extra_params.items():
            expected = _EXTRA_PARAMETER_TYPES.get(name)
            if expected is None or isinstance(value, bool) or not isinstance(value, expected):
                raise ModelConfigError(
                    "parameter_not_supported",
                    f"{name!r} is not a parameter a model request may carry.",
                )
        object.__setattr__(self, "extra_params", MappingProxyType(dict(self.extra_params)))

    def request_kwargs(self) -> dict[str, Any]:
        """The endpoint, credential and extra parameters, as SDK keyword arguments.

        A value that is not set is left out, so the SDK's own lookup applies to
        it and to nothing that is set here.

        On its ``openai/`` route the SDK sends ``dimensions`` only with the
        model names it knows to take it. That route also carries every model
        of an OpenAI-compatible endpoint, whatever its name, so a dimension
        set here is named as one to send and the endpoint answers for itself.
        """
        kwargs: dict[str, Any] = {
            **litellm_optional_kwargs(api_base=self.api_base, api_key=self.api_key),
            **self.extra_params,
        }
        if "dimensions" in self.extra_params and self.route.startswith("openai/"):
            kwargs["allowed_openai_params"] = ["dimensions"]
        return kwargs


@dataclass(frozen=True)
class ProviderDescriptor:
    """One way of reaching a model, as Settings offers it."""

    id: str
    label: str
    kinds: tuple[ModelKind, ...]
    # Accepted strategies; the first is the one Settings proposes.
    auth_strategies: tuple[AuthStrategy, ...]
    # The LiteLLM provider segment a model id is prefixed with, per kind. A
    # descriptor with no prefix takes a full ``provider/model`` route as its
    # model id.
    route_prefixes: Mapping[ModelKind, str] = field(default_factory=dict)
    default_endpoint: str | None = None
    endpoint_required: bool = False
    # Whether the endpoint answers a model listing request.
    lists_models: bool = False
    # The header a model listing request carries the API key in.
    key_header: Literal["authorization", "x-api-key"] = "authorization"
    # Non-secret settings the provider accepts besides endpoint and model.
    options: tuple[str, ...] = ()

    @property
    def model_is_route(self) -> bool:
        """Whether the model id is itself a LiteLLM ``provider/model`` route."""
        return not self.route_prefixes

    def compile_route(self, kind: ModelKind, model_id: str) -> str:
        """Return the LiteLLM route for ``model_id`` used as a model of ``kind``."""
        if kind not in self.kinds:
            raise ModelConfigError(
                "kind_not_supported",
                f"{self.label} does not serve {kind} models.",
            )
        model = model_id.strip()
        if not model:
            raise ModelConfigError("model_required", "Enter a model.")
        if self.model_is_route:
            provider, separator, provider_model = model.partition("/")
            if not separator or not provider or not provider_model or "://" in model:
                raise ModelConfigError(
                    "model_route_invalid",
                    "Enter the model as a LiteLLM route, provider/model.",
                )
            return model
        return f"{self.route_prefixes[kind]}/{model}"


PROVIDER_DESCRIPTORS: tuple[ProviderDescriptor, ...] = (
    ProviderDescriptor(
        id="openai",
        label="OpenAI",
        kinds=("generation", "embedding"),
        auth_strategies=("api_key", "environment"),
        route_prefixes={"generation": "openai", "embedding": "openai"},
        default_endpoint="https://api.openai.com/v1",
        lists_models=True,
        options=("organization",),
    ),
    ProviderDescriptor(
        id="anthropic",
        label="Anthropic",
        kinds=("generation",),
        auth_strategies=("api_key", "environment"),
        route_prefixes={"generation": "anthropic"},
        default_endpoint="https://api.anthropic.com",
        lists_models=True,
        key_header="x-api-key",
    ),
    ProviderDescriptor(
        id="openrouter",
        label="OpenRouter",
        kinds=("generation", "embedding"),
        auth_strategies=("api_key", "environment"),
        route_prefixes={"generation": "openrouter", "embedding": "openrouter"},
        default_endpoint="https://openrouter.ai/api/v1",
        lists_models=True,
    ),
    ProviderDescriptor(
        id="ollama",
        label="Ollama",
        kinds=("generation", "embedding"),
        auth_strategies=("none", "api_key"),
        route_prefixes={"generation": "ollama_chat", "embedding": "ollama"},
        default_endpoint="http://localhost:11434",
        lists_models=True,
    ),
    ProviderDescriptor(
        id="openai_compatible",
        label="OpenAI-compatible endpoint",
        kinds=("generation", "embedding"),
        auth_strategies=("api_key", "none"),
        route_prefixes={"generation": "openai", "embedding": "openai"},
        endpoint_required=True,
        lists_models=True,
    ),
    ProviderDescriptor(
        id="litellm",
        label="LiteLLM route (advanced)",
        kinds=("generation", "embedding"),
        auth_strategies=("api_key", "environment"),
    ),
)

_DESCRIPTORS_BY_ID: Mapping[str, ProviderDescriptor] = MappingProxyType(
    {descriptor.id: descriptor for descriptor in PROVIDER_DESCRIPTORS}
)


def provider_descriptor(provider_id: str) -> ProviderDescriptor:
    """Return the descriptor named ``provider_id``."""
    descriptor = _DESCRIPTORS_BY_ID.get(provider_id)
    if descriptor is None:
        raise ModelConfigError("unknown_provider", f"{provider_id!r} is not a known provider.")
    return descriptor


def names_litellm_route(model: str | None) -> bool:
    """Whether ``model`` is a ``provider/model`` route whose provider LiteLLM knows.

    A model id may contain a slash itself, as ``BAAI/bge-m3`` does; only a
    first segment that names a provider makes it a route.
    """
    provider, separator, provider_model = (model or "").strip().partition("/")
    if not separator or not provider_model:
        return False
    return provider in litellm.provider_list or any(
        registered.get("provider") == provider for registered in litellm.custom_provider_map
    )


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


def endpoint_carries_credentials(url: str | None) -> bool:
    """Whether an endpoint URL holds a user name or a password.

    An endpoint is stored and shown as it is written. A credential belongs in
    the API key, which is stored encrypted and never shown.
    """
    try:
        return "@" in urlsplit((url or "").strip()).netloc
    except ValueError:
        return False


def origin_label(origin: tuple[str, str, int]) -> str:
    """Write an origin as ``scheme://host`` with its port when it is not the default."""
    scheme, host, port = origin
    if ":" in host:
        host = f"[{host}]"
    return f"{scheme}://{host}" if port == _DEFAULT_PORTS[scheme] else f"{scheme}://{host}:{port}"
