"""Model provider descriptors and the invocation a model request is made from.

A provider descriptor states what MemForge knows about one way of reaching a
model: which kinds of model it serves, where its endpoint is, how it
authenticates, how a model id becomes a LiteLLM route, and how its endpoint
lists the models of each kind and what each listed entry declares
(:mod:`memforge.llm.model_listings`). Settings name a descriptor; nothing else
about a provider is configured.

A ``ResolvedInvocation`` is everything the LiteLLM SDK is given for one
configured model: the route, the endpoint, the credential and a short list of
permitted extra parameters. The generation client and the Embedding Runtime are
built from one and pass the SDK nothing else about where a request goes.

A descriptor also says how a request asks its model to answer without reasoning
first. Closed-choice calls size their output for the answer alone, and a model
that reasons by default would spend that output before it answers.

Whether a structured request asks the provider to hold the reply to its JSON
schema is decided by what the provider declares: a provider that enforces a
schema for every model it serves says so through its descriptor, and through
the LiteLLM routes that reach it; a model listing that declares it for each
model says so for the model chosen from it. A model that nothing declares
anything for is left to LiteLLM's model map.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, get_args
from urllib.parse import urlsplit, urlunsplit

import litellm

from memforge.llm.model_listings import (
    AnthropicListing,
    ModelListing,
    OllamaListing,
    OpenAiCompatibleListing,
    OpenAiListing,
    OpenRouterListing,
)

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
    "route_enforces_response_schema",
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

# The LiteLLM providers whose server holds every generation reply to the JSON
# schema a request gives. Ollama takes the schema as the ``format`` of a chat
# or a generate request for every model
# (https://docs.ollama.com/capabilities/structured-outputs), and LiteLLM's
# ``ollama_chat/`` and ``ollama/`` routes write a ``json_schema`` response
# format there.
_SCHEMA_ENFORCING_ROUTE_PROVIDERS = frozenset({"ollama_chat", "ollama"})

# The request paths of an OpenAI-style API, written after its base URL.
_OPENAI_REQUEST_PATHS = (
    "/chat/completions",
    "/completions",
    "/responses",
    "/embeddings",
    "/embeddings/models",
    "/models",
)

# Every extra parameter an invocation may carry to the SDK, with the type its
# value must have.
_EXTRA_PARAMETER_TYPES: Mapping[str, type] = MappingProxyType(
    {
        "dimensions": int,
        "organization": str,
    }
)


def _sends_nothing(route: str) -> dict[str, Any]:
    """Ask for nothing: the model answers directly unless asked to reason, or no
    parameter is known to be safe for every route the descriptor serves."""
    return {}


def _openrouter_reasoning_off(route: str) -> dict[str, Any]:
    """OpenRouter's own ``reasoning`` request field, which LiteLLM writes into
    the request body from ``extra_body``."""
    return {"extra_body": {"reasoning": {"enabled": False}}}


def _openai_lowest_reasoning_effort(route: str) -> dict[str, Any]:
    """The lowest ``reasoning_effort`` LiteLLM accepts for the model, if it takes one.

    LiteLLM's model map says which models reason and which effort levels each
    takes beyond ``low``; a model it does not know is sent nothing.
    """
    model = route.partition("/")[2]
    try:
        supported = litellm.get_supported_openai_params(model=model, custom_llm_provider="openai") or ()
        if "reasoning_effort" not in supported or not litellm.supports_reasoning(model=route):
            return {}
        info = litellm.get_model_info(route)
    except Exception:
        return {}
    for level in ("none", "minimal"):
        if info.get(f"supports_{level}_reasoning_effort"):
            return {"reasoning_effort": level}
    if info.get("supports_low_reasoning_effort") is False:
        return {}
    return {"reasoning_effort": "low"}


def _ollama_thinking_off(route: str) -> dict[str, Any]:
    """Ollama's ``think`` request field, which LiteLLM passes through as given.

    A gpt-oss model cannot stop thinking and takes a level instead, as LiteLLM's
    own mapping of ``reasoning_effort`` for Ollama does.
    """
    if route.partition("/")[2].startswith("gpt-oss"):
        return {"think": "low"}
    return {"think": False}


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
    # The descriptor the invocation was compiled from. An invocation named by
    # its route alone has none, and its requests carry nothing provider-specific.
    provider: str | None = None
    # Whether the provider holds the model's reply to the JSON schema a
    # structured request gives, as the provider declares it; ``None`` when it
    # declares nothing, and LiteLLM's model map decides.
    native_schema: bool | None = None

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
    # How the endpoint lists its models and what each entry declares; every
    # descriptor states one, ``None`` when the endpoint lists none.
    listing: ModelListing | None = field(kw_only=True)
    # Non-secret settings the provider accepts besides endpoint and model.
    options: tuple[str, ...] = ()
    # The request paths the SDK and the model listing write after the base URL.
    # A base URL pasted with one of them at its end is read without it.
    request_paths: tuple[str, ...] = ()
    # The SDK keyword arguments that ask the generation model at a route to
    # answer without reasoning first, as LiteLLM passes them to this provider.
    reasoning_off: Callable[[str], dict[str, Any]] = _sends_nothing
    # The kinds whose requests go to the provider's OpenAI-compatible API, with
    # the path of that API after the base URL. They are sent on LiteLLM's
    # ``openai/`` route, which carries the API key and the timeout of every
    # request, while the route of the model id still names the model.
    openai_api_paths: Mapping[ModelKind, str] = field(default_factory=dict)
    # Whether the provider holds the reply of every generation model it serves
    # to the JSON schema a request gives.
    enforces_response_schema: bool = False

    @property
    def model_is_route(self) -> bool:
        """Whether the model id is itself a LiteLLM ``provider/model`` route."""
        return not self.route_prefixes

    @property
    def lists_models(self) -> bool:
        """Whether the endpoint answers a model listing request."""
        return self.listing is not None

    @property
    def lists_token_limits(self) -> bool:
        """Whether the model listing can give the token limits of a listed model."""
        return self.listing is not None and self.listing.gives_token_limits

    def base_endpoint(self, endpoint: str) -> str:
        """Return ``endpoint`` as the base URL requests are written after.

        Surrounding whitespace and trailing slashes are dropped, and so is a
        request path of the provider at the end of the URL path, as an address
        copied from API documentation carries one. A query or a fragment is
        left as it is.
        """
        value = endpoint.strip().rstrip("/")
        try:
            address = urlsplit(value)
        except ValueError:
            return value
        if address.query or address.fragment:
            return value
        path = address.path
        lowered = path.lower()
        for request_path in sorted(self.request_paths, key=len, reverse=True):
            if lowered.endswith(request_path):
                path = path[: -len(request_path)].rstrip("/")
                break
        return urlunsplit((address.scheme, address.netloc, path, "", ""))

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

    def request_target(self, kind: ModelKind, route: str, endpoint: str | None) -> tuple[str, str | None]:
        """Return the LiteLLM route and the base URL a request of ``kind`` is sent with.

        ``route`` and ``endpoint`` are the model's own, as ``compile_route`` and
        ``base_endpoint`` give them. A kind the provider serves through its
        OpenAI-compatible API is sent there under the same model name.
        """
        api_path = self.openai_api_paths.get(kind)
        base = endpoint or self.default_endpoint
        if api_path is None or base is None:
            return route, endpoint
        return f"openai/{route.partition('/')[2]}", f"{base}{api_path}"


PROVIDER_DESCRIPTORS: tuple[ProviderDescriptor, ...] = (
    ProviderDescriptor(
        id="openai",
        label="OpenAI",
        kinds=("generation", "embedding"),
        auth_strategies=("api_key", "environment"),
        route_prefixes={"generation": "openai", "embedding": "openai"},
        default_endpoint="https://api.openai.com/v1",
        listing=OpenAiListing(),
        options=("organization",),
        request_paths=_OPENAI_REQUEST_PATHS,
        reasoning_off=_openai_lowest_reasoning_effort,
    ),
    ProviderDescriptor(
        id="anthropic",
        label="Anthropic",
        kinds=("generation",),
        auth_strategies=("api_key", "environment"),
        route_prefixes={"generation": "anthropic"},
        default_endpoint="https://api.anthropic.com",
        listing=AnthropicListing(),
        # LiteLLM writes ``/v1/messages`` after the base URL itself.
        request_paths=("/v1", "/v1/messages", "/v1/models"),
        # An Anthropic model thinks only when a request asks it to.
        reasoning_off=_sends_nothing,
    ),
    ProviderDescriptor(
        id="openrouter",
        label="OpenRouter",
        kinds=("generation", "embedding"),
        auth_strategies=("api_key", "environment"),
        route_prefixes={"generation": "openrouter", "embedding": "openrouter"},
        default_endpoint="https://openrouter.ai/api/v1",
        listing=OpenRouterListing(),
        request_paths=_OPENAI_REQUEST_PATHS,
        reasoning_off=_openrouter_reasoning_off,
    ),
    ProviderDescriptor(
        id="ollama",
        label="Ollama",
        kinds=("generation", "embedding"),
        auth_strategies=("none", "api_key"),
        route_prefixes={"generation": "ollama_chat", "embedding": "ollama"},
        default_endpoint="http://localhost:11434",
        listing=OllamaListing(),
        # LiteLLM writes Ollama's own ``/api`` paths after the server address,
        # so the address of its OpenAI-compatible ``/v1`` API is read without it.
        request_paths=(
            "/api/tags",
            "/api/show",
            "/api/chat",
            "/api/generate",
            "/api/embed",
            "/api/embeddings",
            "/v1",
            *(f"/v1{path}" for path in _OPENAI_REQUEST_PATHS),
        ),
        reasoning_off=_ollama_thinking_off,
        # LiteLLM's ``ollama/`` route posts an embedding request with neither
        # the API key nor the timeout
        # (https://github.com/BerriAI/litellm/issues/47261), and Ollama's
        # OpenAI-compatible API takes the same models under the same names.
        openai_api_paths={"embedding": "/v1"},
        # Ollama holds a chat reply to the JSON schema given as its ``format``
        # for every model (https://docs.ollama.com/capabilities/structured-outputs),
        # and LiteLLM writes a ``json_schema`` response format there.
        enforces_response_schema=True,
    ),
    ProviderDescriptor(
        id="openai_compatible",
        label="OpenAI-compatible endpoint",
        kinds=("generation", "embedding"),
        auth_strategies=("api_key", "none"),
        route_prefixes={"generation": "openai", "embedding": "openai"},
        endpoint_required=True,
        listing=OpenAiCompatibleListing(),
        request_paths=_OPENAI_REQUEST_PATHS,
        # Any server may answer here, and no reasoning parameter is one every
        # such server accepts.
        reasoning_off=_sends_nothing,
    ),
    ProviderDescriptor(
        id="litellm",
        label="Other provider (LiteLLM route)",
        kinds=("generation", "embedding"),
        auth_strategies=("api_key", "environment"),
        # A route may name any LiteLLM provider, gateways among them: there is
        # no one listing to ask, and no reasoning parameter is one every such
        # provider accepts.
        listing=None,
        reasoning_off=_sends_nothing,
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


def route_enforces_response_schema(route: str | None) -> bool:
    """Whether the provider a LiteLLM ``provider/model`` route reaches holds every reply to the request's schema."""
    provider, separator, _model = (route or "").strip().partition("/")
    return bool(separator) and provider in _SCHEMA_ENFORCING_ROUTE_PROVIDERS


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
