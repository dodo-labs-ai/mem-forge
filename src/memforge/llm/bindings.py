"""Model bindings: which model MemForge uses for generation and for embedding.

A binding is the whole configuration of one kind of model: a provider
descriptor, an endpoint, an auth strategy, a model id and the route it compiles
to. Each kind has at most one active binding, which the runtime uses, and at
most one candidate, which Settings edits and a capability check promotes.

A binding holds no API key. It holds a reference: to a key stored encrypted in
the database, or to the ``MEMFORGE_*`` environment variable that carries one.
The key is resolved only when the binding is compiled into the invocation a
request is made from.

Bindings are resolved for one database at a time. In an installation whose
model settings are writable, the stored active binding is used and the
``MEMFORGE_*`` model variables describe a kind that has none stored. In a
deployment that manages its model settings, every resolution compiles the
binding from the environment and the database is neither read nor written.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, Protocol
from urllib.parse import urlsplit, urlunsplit

from memforge.llm.embedding_profile import EmbeddingIdentity, EmbeddingProfile
from memforge.llm.providers import (
    KEYLESS_API_KEY,
    AuthStrategy,
    ModelConfigError,
    ModelKind,
    ProviderDescriptor,
    ResolvedInvocation,
    endpoint_carries_credentials,
    endpoint_origin,
    names_litellm_route,
    origin_label,
    provider_descriptor,
)
from memforge.source_secrets import SecretConfigurationError, decrypt_secret, encrypt_secret

if TYPE_CHECKING:
    from memforge.config import AppConfig, LlmConfig

logger = logging.getLogger(__name__)

__all__ = [
    "BindingSlot",
    "BindingState",
    "ConfigSource",
    "KeySource",
    "ModelBinding",
    "ModelBindingStore",
    "ValidationResult",
    "binding_from_record",
    "binding_from_settings",
    "binding_record",
    "build_binding",
    "compile_binding",
    "embedding_identity",
    "embedding_profile",
    "environment_binding",
    "environment_secret_ref",
    "installed_litellm_version",
    "key_last4",
    "key_source",
    "new_stored_secret",
    "parameters_for_kind",
    "resolve_active_binding",
    "resolve_active_invocation",
    "resolve_credential",
    "resolve_invocation",
    "state_after_change",
    "state_after_check",
    "stored_secret_id",
    "stored_secret_ref",
    "token_limits_for_kind",
    "translate_stored_settings",
    "usable_invocation",
    "validation_record",
]

BindingState = Literal["not_validated", "ready", "needs_revalidation"]
BindingSlot = Literal["active", "candidate"]
# ``settings``: stored in the database. ``environment``: read from the
# ``MEMFORGE_*`` variables for a kind with nothing stored. ``deployment``: read
# from the environment of a deployment that manages its model settings.
ConfigSource = Literal["settings", "environment", "deployment"]
# ``saved``: entered in Settings and stored encrypted. ``environment``: carried
# by a ``MEMFORGE_*`` variable.
KeySource = Literal["saved", "environment"]

_STORED_SECRET_PREFIX = "secret:"
_ENVIRONMENT_SECRET_PREFIX = "env:"
_ENVIRONMENT_KEY_VARIABLES: Mapping[ModelKind, str] = MappingProxyType(
    {
        "generation": "MEMFORGE_ENRICHMENT_API_KEY",
        "embedding": "MEMFORGE_EMBEDDING_API_KEY",
    }
)
# The request parameters a binding of each kind may set.
_PARAMETERS_BY_KIND: Mapping[ModelKind, tuple[str, ...]] = MappingProxyType(
    {
        "generation": (),
        "embedding": ("dimensions",),
    }
)
# The token limits a binding of each kind may state for its model.
_TOKEN_LIMITS_BY_KIND: Mapping[ModelKind, tuple[str, ...]] = MappingProxyType(
    {
        "generation": ("context_window_tokens", "max_output_tokens"),
        "embedding": (),
    }
)
_OPENAI_HOST = "api.openai.com"
# Four characters of a shorter key would be too much of it.
_SHOWN_KEY_MIN_LENGTH = 12


@dataclass(frozen=True)
class ValidationResult:
    """The outcome of the latest capability check of one binding.

    It records when the check ran, the digest of the binding it ran against and
    what was observed. It never holds a prompt or a response.
    """

    checked_at: str
    digest: str
    facts: Mapping[str, Any] = field(default_factory=dict)
    error_code: str | None = None

    @property
    def passed(self) -> bool:
        return self.error_code is None


@dataclass(frozen=True)
class ModelBinding:
    """The configuration of one kind of model."""

    kind: ModelKind
    provider: str
    # The base URL requests are sent to; ``None`` leaves it to the route.
    endpoint: str | None
    # The normalized origin of the endpoint, which a saved key is bound to.
    origin: str
    auth_strategy: AuthStrategy
    model_id: str
    # The LiteLLM route the model id compiles to.
    route: str
    options: Mapping[str, Any]
    parameters: Mapping[str, Any]
    secret_ref: str | None
    config_source: ConfigSource
    state: BindingState = "not_validated"
    validation: ValidationResult | None = None
    # When the stored binding was last saved. It identifies that version of the
    # binding; a binding that is not stored has none.
    saved_at: str | None = None
    # The context window and the largest output of the model, in tokens, for a
    # model whose limits LiteLLM does not know.
    token_limits: Mapping[str, int] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        """Identify everything a passed capability check vouches for.

        A check that passed for one digest says nothing about a binding with
        another: the endpoint, provider, route, auth strategy, model, requested
        dimensions, token limits, the key the binding refers to or the
        installed LiteLLM version has changed.
        """
        material = {
            "provider": self.provider,
            "endpoint": self.endpoint,
            "origin": self.origin,
            "route": self.route,
            "auth_strategy": self.auth_strategy,
            "model_id": self.model_id,
            "dimensions": self.parameters.get("dimensions"),
            "token_limits": dict(self.token_limits),
            "secret_ref": self.secret_ref,
            "litellm": installed_litellm_version(),
        }
        encoded = json.dumps(material, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def installed_litellm_version() -> str:
    return importlib.metadata.version("litellm")


def parameters_for_kind(kind: ModelKind) -> tuple[str, ...]:
    """The request parameters a binding of ``kind`` may set."""
    return _PARAMETERS_BY_KIND[kind]


def token_limits_for_kind(kind: ModelKind) -> tuple[str, ...]:
    """The token limits a binding of ``kind`` may state for its model."""
    return _TOKEN_LIMITS_BY_KIND[kind]


def embedding_identity(binding: ModelBinding) -> EmbeddingIdentity:
    """The settings of an embedding binding that decide its vector space."""
    return EmbeddingIdentity(
        provider=binding.provider,
        route=binding.route,
        endpoint=_identity_endpoint(binding.endpoint),
        model_id=binding.model_id,
        requested_dimensions=binding.parameters.get("dimensions"),
    )


def embedding_profile(binding: ModelBinding | None) -> EmbeddingProfile | None:
    """The Embedding Profile of an embedding binding its latest check vouches for.

    The profile needs the vector length the model answered with, so a binding
    has one only while a check that passed is on record for the binding as it
    stands.
    """
    if binding is None or binding.kind != "embedding":
        return None
    validation = binding.validation
    if validation is None or not validation.passed or validation.digest != binding.digest:
        return None
    dimension = validation.facts.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        return None
    return embedding_identity(binding).profile(dimension)


def state_after_change(state: BindingState, *, previous_digest: str, digest: str) -> BindingState:
    """The state of a binding once its settings are saved again.

    A change to what a passed check vouches for leaves a ready binding to be
    checked again. Any other binding keeps its state.
    """
    if state == "ready" and previous_digest != digest:
        return "needs_revalidation"
    return state


def state_after_check(state: BindingState, *, passed: bool) -> BindingState:
    """The state of a binding once a capability check has run against it."""
    if passed:
        return "ready"
    return "needs_revalidation" if state == "ready" else state


def build_binding(
    *,
    kind: ModelKind,
    provider: str,
    model_id: str,
    config_source: ConfigSource,
    endpoint: str | None = None,
    auth_strategy: AuthStrategy | None = None,
    options: Mapping[str, Any] | None = None,
    parameters: Mapping[str, Any] | None = None,
    token_limits: Mapping[str, Any] | None = None,
    secret_ref: str | None = None,
) -> ModelBinding:
    """Build a binding of ``kind`` from settings, or say which setting is unusable."""
    descriptor = provider_descriptor(provider)
    route = descriptor.compile_route(kind, model_id)
    strategy = auth_strategy or descriptor.auth_strategies[0]
    if strategy not in descriptor.auth_strategies:
        raise ModelConfigError(
            "auth_strategy_not_supported",
            f"{descriptor.label} does not accept the {strategy} auth strategy.",
        )
    normalized_endpoint = _normalized_endpoint(descriptor, endpoint)
    return ModelBinding(
        kind=kind,
        provider=descriptor.id,
        endpoint=normalized_endpoint,
        origin=_binding_origin(normalized_endpoint, route),
        auth_strategy=strategy,
        model_id=model_id.strip(),
        route=route,
        options=_validated_options(descriptor, options),
        parameters=_validated_parameters(kind, parameters),
        secret_ref=secret_ref if strategy == "api_key" else None,
        config_source=config_source,
        token_limits=_validated_token_limits(kind, token_limits),
    )


def compile_binding(binding: ModelBinding, credential: str | None) -> ResolvedInvocation:
    """Compile a binding and its resolved credential into what the SDK is given."""
    if binding.auth_strategy == "api_key":
        if not credential:
            raise ModelConfigError(
                "credential_unavailable",
                f"The API key of the {binding.kind} model is not available.",
            )
        api_key: str | None = credential
    elif binding.auth_strategy == "none":
        api_key = KEYLESS_API_KEY
    else:
        api_key = None
    return ResolvedInvocation(
        route=binding.route,
        api_base=binding.endpoint,
        api_key=api_key,
        extra_params={**binding.options, **binding.parameters},
        provider=binding.provider,
    )


def binding_from_settings(
    kind: ModelKind,
    *,
    model: str | None,
    base_url: str | None,
    secret_ref: str | None,
    config_source: ConfigSource,
) -> ModelBinding | None:
    """Read a model name, a base URL and an optional key reference as a binding.

    This is how the ``MEMFORGE_*`` model variables name a provider. A
    generation model written ``provider/model`` is a LiteLLM route and any
    other name is an Anthropic model. An embedding model whose first segment is
    a provider LiteLLM knows is a route and any other name is a model id of the
    OpenAI-compatible endpoint at the base URL. Returns ``None`` when the values
    do not describe a usable model: no model, a model that needs a key without
    one, an embedding model id with no endpoint, or a setting no binding
    accepts.
    """
    try:
        return _binding_from_settings(
            kind,
            model=model,
            base_url=base_url,
            secret_ref=secret_ref,
            config_source=config_source,
        )
    except ModelConfigError as exc:
        logger.warning("The %s model settings are not usable (%s)", kind, exc.code)
        return None


def _binding_from_settings(
    kind: ModelKind,
    *,
    model: str | None,
    base_url: str | None,
    secret_ref: str | None,
    config_source: ConfigSource,
) -> ModelBinding | None:
    """Read the values as a binding, or raise for a setting no binding accepts."""
    name = (model or "").strip()
    endpoint = (base_url or "").strip() or None
    if not name:
        return None
    if kind == "generation":
        is_route = "/" in name and "://" not in name
    else:
        is_route = names_litellm_route(name)
    if is_route:
        provider = "litellm"
        strategy: AuthStrategy = "api_key" if secret_ref else "environment"
    elif kind == "generation":
        provider, strategy = "anthropic", "api_key"
    elif endpoint is None:
        return None
    elif (endpoint_origin(endpoint) or ("", "", 0))[1] == _OPENAI_HOST:
        provider, strategy = "openai", "api_key"
    else:
        provider = "openai_compatible"
        strategy = "api_key" if secret_ref else "none"
    if strategy == "api_key" and not secret_ref:
        return None
    return build_binding(
        kind=kind,
        provider=provider,
        model_id=name,
        endpoint=endpoint,
        auth_strategy=strategy,
        secret_ref=secret_ref,
        config_source=config_source,
    )


def environment_secret_ref(kind: ModelKind) -> str:
    """The reference to the key the environment carries for ``kind``."""
    return f"{_ENVIRONMENT_SECRET_PREFIX}{_ENVIRONMENT_KEY_VARIABLES[kind]}"


def stored_secret_ref(secret_id: str) -> str:
    """The reference to the stored key ``secret_id``."""
    return f"{_STORED_SECRET_PREFIX}{secret_id}"


def stored_secret_id(secret_ref: str | None) -> str | None:
    """The id of the stored key ``secret_ref`` refers to, when it refers to one."""
    if secret_ref and secret_ref.startswith(_STORED_SECRET_PREFIX):
        return secret_ref[len(_STORED_SECRET_PREFIX) :]
    return None


def key_source(binding: ModelBinding) -> KeySource | None:
    """Where the API key of ``binding`` is kept, or ``None`` when it uses no key."""
    if binding.auth_strategy != "api_key" or not binding.secret_ref:
        return None
    return "saved" if stored_secret_id(binding.secret_ref) else "environment"


def environment_binding(
    config: "AppConfig",
    kind: ModelKind,
    *,
    config_source: ConfigSource,
) -> ModelBinding | None:
    """The binding the ``MEMFORGE_*`` model variables describe for ``kind``."""
    model, base_url, api_key = _environment_settings(config.llm, kind)
    return binding_from_settings(
        kind,
        model=model,
        base_url=base_url,
        secret_ref=environment_secret_ref(kind) if api_key.strip() else None,
        config_source=config_source,
    )


def translate_stored_settings(
    kind: ModelKind,
    *,
    model: str | None,
    base_url: str | None,
    api_key: str | None,
    defaults: "LlmConfig",
) -> tuple[ModelBinding, dict[str, str] | None] | None:
    """Read stored model, base URL and key columns of one kind as a binding.

    ``defaults`` are the model settings of the running configuration. A column
    left empty takes its value from them. With no stored key the binding
    refers to the key the environment carries for the kind, whether or not one
    is set now: the model stays the stored one, and it can be called once the
    key is there. Returns the binding and, for a stored key, the encrypted
    secret to store with it.

    Returns ``None`` when nothing is stored for the kind, and also, with a
    warning in the log, when what is stored cannot be written as a binding.
    Raises ``ModelConfigError`` when a stored key cannot be encrypted.
    """
    stored_model = (model or "").strip()
    stored_base_url = (base_url or "").strip()
    stored_key = (api_key or "").strip()
    if not (stored_model or stored_base_url or stored_key):
        return None
    default_model, default_base_url, default_key = _environment_settings(defaults, kind)
    secret_id = uuid.uuid4().hex if stored_key else None
    if secret_id is not None:
        secret_ref: str | None = stored_secret_ref(secret_id)
    elif default_key.strip():
        secret_ref = environment_secret_ref(kind)
    else:
        secret_ref = None
    settings: dict[str, Any] = {
        "model": stored_model or default_model,
        "base_url": stored_base_url or default_base_url,
        "config_source": "settings",
    }
    try:
        binding = _binding_from_settings(kind, secret_ref=secret_ref, **settings)
        if binding is None and secret_ref is None:
            # The model needs a key and none is set anywhere yet.
            binding = _binding_from_settings(kind, secret_ref=environment_secret_ref(kind), **settings)
        if binding is None:
            raise ModelConfigError(
                "settings_incomplete",
                f"The stored {kind} settings name no model or no endpoint.",
            )
    except ModelConfigError as exc:
        logger.warning(
            "The stored %s model settings cannot be kept as a binding (%s). "
            "Enter the %s model in Settings again.",
            kind,
            exc.code,
            kind,
        )
        return None
    if secret_id is None:
        return binding, None
    return binding, _encrypted_secret(secret_id, stored_key, origin=binding.origin)


def binding_record(binding: ModelBinding) -> dict[str, Any]:
    """The stored form of a binding."""
    validation = binding.validation
    return {
        "kind": binding.kind,
        "provider": binding.provider,
        "endpoint": binding.endpoint,
        "origin": binding.origin,
        "auth_strategy": binding.auth_strategy,
        "options": dict(binding.options),
        "secret_ref": binding.secret_ref,
        "model_id": binding.model_id,
        "route": binding.route,
        "parameters": dict(binding.parameters),
        "token_limits": dict(binding.token_limits),
        "config_source": binding.config_source,
        "state": binding.state,
        "validation": None if validation is None else validation_record(validation),
    }


def validation_record(validation: ValidationResult) -> dict[str, Any]:
    """The stored form of a validation result."""
    return {
        "checked_at": validation.checked_at,
        "digest": validation.digest,
        "facts": dict(validation.facts),
        "error_code": validation.error_code,
    }


def binding_from_record(record: Mapping[str, Any]) -> ModelBinding:
    """Read a stored binding, in the state it is in under the installed LiteLLM."""
    stored_validation = record.get("validation")
    validation = (
        None
        if stored_validation is None
        else ValidationResult(
            checked_at=stored_validation["checked_at"],
            digest=stored_validation["digest"],
            facts=MappingProxyType(dict(stored_validation.get("facts") or {})),
            error_code=stored_validation.get("error_code"),
        )
    )
    binding = ModelBinding(
        kind=record["kind"],
        provider=record["provider"],
        endpoint=record.get("endpoint"),
        origin=record["origin"],
        auth_strategy=record["auth_strategy"],
        model_id=record["model_id"],
        route=record["route"],
        options=MappingProxyType(dict(record.get("options") or {})),
        parameters=MappingProxyType(dict(record.get("parameters") or {})),
        secret_ref=record.get("secret_ref"),
        config_source=record["config_source"],
        state=record["state"],
        validation=validation,
        saved_at=record.get("updated_at"),
        token_limits=MappingProxyType(dict(record.get("token_limits") or {})),
    )
    if validation is None:
        return binding
    return replace(
        binding,
        state=state_after_change(
            binding.state,
            previous_digest=validation.digest,
            digest=binding.digest,
        ),
    )


class ModelBindingStore(Protocol):
    """The storage operations model bindings need, as the database provides them."""

    async def get_model_binding(self, kind: str, slot: str) -> dict[str, Any] | None: ...

    async def get_model_secret(self, secret_id: str) -> dict[str, Any] | None: ...


async def resolve_active_binding(
    db: ModelBindingStore,
    config: "AppConfig",
    kind: ModelKind,
) -> ModelBinding | None:
    """Return the binding the runtime uses for ``kind`` with this database."""
    if not config.server.llm_config_writable:
        return environment_binding(config, kind, config_source="deployment")
    stored = await db.get_model_binding(kind, "active")
    if stored is not None:
        return binding_from_record(stored)
    return environment_binding(config, kind, config_source="environment")


async def resolve_credential(
    db: ModelBindingStore,
    config: "AppConfig",
    binding: ModelBinding,
) -> str | None:
    """Return the API key a binding refers to, or ``None`` when it uses none.

    A stored key is released only to a binding with the origin it was saved for.
    """
    if binding.auth_strategy != "api_key":
        return None
    reference = binding.secret_ref or ""
    if reference.startswith(_ENVIRONMENT_SECRET_PREFIX):
        variable = reference[len(_ENVIRONMENT_SECRET_PREFIX) :]
        for kind, known in _ENVIRONMENT_KEY_VARIABLES.items():
            if known == variable:
                value = _environment_settings(config.llm, kind)[2].strip()
                if value:
                    return value
        raise ModelConfigError(
            "credential_unavailable",
            f"{variable} is not set, and the {binding.kind} model takes its API key from it.",
        )
    secret_id = stored_secret_id(reference)
    secret = await db.get_model_secret(secret_id) if secret_id else None
    if secret is None:
        raise ModelConfigError(
            "credential_unavailable",
            f"No API key is saved for the {binding.kind} model.",
        )
    if secret["origin"] != binding.origin:
        raise ModelConfigError(
            "api_key_origin_changed",
            f"The saved API key belongs to {secret['origin']}, not to {binding.origin}.",
        )
    try:
        return decrypt_secret(secret["ciphertext"])
    except SecretConfigurationError as exc:
        raise ModelConfigError(
            "credential_unavailable",
            f"The saved API key of the {binding.kind} model could not be decrypted. Enter it again.",
        ) from exc


async def resolve_invocation(
    db: ModelBindingStore,
    config: "AppConfig",
    binding: ModelBinding,
) -> ResolvedInvocation:
    """Compile ``binding`` with the credential it refers to."""
    return compile_binding(binding, await resolve_credential(db, config, binding))


async def resolve_active_invocation(
    db: ModelBindingStore,
    config: "AppConfig",
    kind: ModelKind,
) -> ResolvedInvocation | None:
    """Return what the SDK is given for the active binding of ``kind``.

    ``None`` means no model of this kind can be called: none is configured, or
    the configured one has no usable credential.
    """
    return await usable_invocation(db, config, await resolve_active_binding(db, config, kind))


async def usable_invocation(
    db: ModelBindingStore,
    config: "AppConfig",
    binding: ModelBinding | None,
) -> ResolvedInvocation | None:
    """Compile ``binding`` for use, or return ``None`` when it cannot be called."""
    if binding is None:
        return None
    try:
        return await resolve_invocation(db, config, binding)
    except ModelConfigError as exc:
        logger.warning("The active %s model cannot be used (%s)", binding.kind, exc.code)
        return None


def new_stored_secret(api_key: str, *, origin: str) -> tuple[str, dict[str, str]]:
    """Encrypt an API key for ``origin``.

    Returns the reference a binding holds and the secret to store with that
    binding.
    """
    secret_id = uuid.uuid4().hex
    return stored_secret_ref(secret_id), _encrypted_secret(secret_id, api_key, origin=origin)


def key_last4(api_key: str) -> str:
    """The last four characters of a key, or nothing for a key too short to show them."""
    return api_key[-4:] if len(api_key) >= _SHOWN_KEY_MIN_LENGTH else ""


def _encrypted_secret(secret_id: str, api_key: str, *, origin: str) -> dict[str, str]:
    try:
        ciphertext = encrypt_secret(api_key)
    except SecretConfigurationError as exc:
        raise ModelConfigError(
            "secret_storage_unavailable",
            "The API key cannot be stored because the secret key of this installation is not usable.",
        ) from exc
    return {
        "id": secret_id,
        "ciphertext": ciphertext,
        "origin": origin,
        "last4": key_last4(api_key),
    }


def _environment_settings(llm: "LlmConfig", kind: ModelKind) -> tuple[str, str, str]:
    if kind == "generation":
        return llm.enrichment_model, llm.enrichment_base_url, llm.enrichment_api_key
    return llm.embedding_model, llm.embedding_base_url, llm.embedding_api_key


def _normalized_endpoint(descriptor: ProviderDescriptor, endpoint: str | None) -> str | None:
    value = descriptor.base_endpoint(endpoint or "") or descriptor.default_endpoint
    if value is None:
        if descriptor.endpoint_required:
            raise ModelConfigError(
                "endpoint_required",
                f"Enter the base URL of the {descriptor.label}.",
            )
        return None
    if endpoint_origin(value) is None:
        raise ModelConfigError(
            "endpoint_invalid",
            "Enter a base URL starting with http:// or https://.",
        )
    if endpoint_carries_credentials(value):
        raise ModelConfigError(
            "endpoint_invalid",
            "Enter the base URL without a user name or password.",
        )
    return value


def _identity_endpoint(endpoint: str | None) -> str | None:
    """Write an endpoint the one way an Embedding Profile records it.

    Scheme and host are written in lower case, a port that is the scheme's
    default is left out and a trailing slash is dropped, so every spelling of
    one address gives one vector space. The path keeps its case.
    """
    origin = endpoint_origin(endpoint)
    if endpoint is None or origin is None:
        return endpoint
    address = urlsplit(endpoint.strip())
    return origin_label(origin) + urlunsplit(("", "", address.path.rstrip("/"), address.query, address.fragment))


def _binding_origin(endpoint: str | None, route: str) -> str:
    """What a key saved with a binding is bound to.

    A binding with an endpoint is bound to that endpoint's origin. One without
    leaves the endpoint to its route, so it is bound to the route's provider.
    """
    origin = endpoint_origin(endpoint)
    if origin is not None:
        return origin_label(origin)
    return f"provider:{route.partition('/')[0]}"


def _validated_options(
    descriptor: ProviderDescriptor,
    options: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    validated: dict[str, Any] = {}
    for name, value in (options or {}).items():
        if name not in descriptor.options:
            raise ModelConfigError(
                "option_not_supported",
                f"{descriptor.label} has no {name!r} option.",
            )
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if not isinstance(value, str):
            raise ModelConfigError("option_invalid", f"The {name!r} option must be text.")
        validated[name] = value.strip()
    return MappingProxyType(validated)


def _validated_parameters(
    kind: ModelKind,
    parameters: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    validated: dict[str, Any] = {}
    for name, value in (parameters or {}).items():
        if name not in _PARAMETERS_BY_KIND[kind]:
            raise ModelConfigError(
                "parameter_not_supported",
                f"A {kind} model has no {name!r} parameter.",
            )
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ModelConfigError(
                "parameter_invalid",
                f"The {name!r} parameter must be a whole number greater than zero.",
            )
        validated[name] = value
    return MappingProxyType(validated)


def _validated_token_limits(
    kind: ModelKind,
    token_limits: Mapping[str, Any] | None,
) -> Mapping[str, int]:
    validated: dict[str, int] = {}
    for name, value in (token_limits or {}).items():
        if name not in _TOKEN_LIMITS_BY_KIND[kind]:
            raise ModelConfigError(
                "token_limit_not_supported",
                f"A {kind} model has no {name!r} limit.",
            )
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ModelConfigError(
                "token_limit_invalid",
                f"The {name!r} limit must be a whole number of tokens greater than zero.",
            )
        validated[name] = value
    return MappingProxyType(validated)
