"""HTTP API for model settings: provider descriptors, bindings, validation and model discovery."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field

from memforge.llm.bindings import (
    BindingState,
    ConfigSource,
    ModelBinding,
    installed_litellm_version,
    parameters_for_kind,
)
from memforge.llm.providers import (
    PROVIDER_DESCRIPTORS,
    AuthStrategy,
    ModelConfigError,
    ModelKind,
    ProviderDescriptor,
    endpoint_carries_credentials,
    endpoint_origin,
    origin_label,
    provider_descriptor,
)
from memforge.model_settings import (
    BindingView,
    CandidateDraft,
    KindView,
    ModelSettings,
    ValidationOutcome,
)

__all__ = [
    "ModelHealthResponse",
    "ModelKindHealthResponse",
    "build_model_settings_router",
    "model_health_response",
]

DiscoveryStage = Literal["validation", "connect", "tls", "timeout", "auth", "http"]

# The HTTP status of each refusal that is not an invalid setting.
_ERROR_STATUS = {
    "model_settings_read_only": 405,
    "candidate_not_found": 409,
    "candidate_changed": 409,
    "model_not_configured": 409,
}


class _RequestNotRepeatedRoute(APIRoute):
    """A route that reports a malformed request without repeating any of it.

    A request to these routes may carry an API key. The answer to one that does
    not fit its schema names the field and the problem, and leaves out the
    value that was sent.
    """

    def get_route_handler(self) -> Callable[[Request], Any]:
        handle = super().get_route_handler()

        async def handle_without_repeating_the_request(request: Request) -> Response:
            try:
                return await handle(request)
            except RequestValidationError as exc:
                return JSONResponse(
                    status_code=422,
                    content={
                        "detail": [
                            {
                                "type": str(error.get("type", "")),
                                "loc": [
                                    part if isinstance(part, int) else str(part)
                                    for part in error.get("loc", ())
                                ],
                                "msg": str(error.get("msg", "")),
                            }
                            for error in exc.errors()
                        ]
                    },
                )

        return handle_without_repeating_the_request


class ModelProviderResponse(BaseModel):
    id: str
    label: str
    kinds: list[ModelKind] = Field(description="The kinds of model the provider serves.")
    auth_strategies: list[AuthStrategy] = Field(
        description="The accepted auth strategies; the first is the one to propose.",
    )
    default_endpoint: str | None = Field(
        description="The base URL used when none is entered; null when the provider has none.",
    )
    endpoint_required: bool = Field(description="Whether a base URL has to be entered.")
    model_is_route: bool = Field(
        description="Whether the model is entered as a full LiteLLM provider/model route.",
    )
    lists_models: bool = Field(description="Whether the endpoint can list its models.")
    options: list[str] = Field(description="The names of the non-secret options the provider accepts.")
    parameters: dict[ModelKind, list[str]] = Field(
        description="The request parameters a binding of each served kind may set.",
    )


class ModelProviderListResponse(BaseModel):
    providers: list[ModelProviderResponse]


class ModelApiKeyStatus(BaseModel):
    is_set: bool = Field(description="Whether a key is available to the binding.")
    source: Literal["saved", "environment"] | None = Field(
        default=None,
        description=(
            "'saved' for a key entered in Settings and stored encrypted, 'environment' "
            "for a key read from a MEMFORGE_* variable, null when the binding uses no key."
        ),
    )
    last4: str | None = Field(
        default=None,
        description="The last four characters of the key; null for a key too short to show them.",
    )


class ModelValidationResult(BaseModel):
    checked_at: str
    passed: bool
    error_code: str | None = None
    facts: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "What the check observed: latency_ms; for generation structured_output_mode "
            "('native_schema' or 'json_text'), schema_fallback, token_limits_known and "
            "provider_model; for embedding dimension and provider_model. provider_model is "
            "present only when the provider reported one."
        ),
    )
    current: bool = Field(
        description="Whether the check ran against the binding as it is now.",
    )


class ModelBindingResponse(BaseModel):
    kind: ModelKind
    provider: str
    endpoint: str | None
    origin: str = Field(description="The normalized origin a saved API key is bound to.")
    auth_strategy: AuthStrategy
    model_id: str
    route: str = Field(description="The LiteLLM route the model id compiles to.")
    options: dict[str, Any]
    parameters: dict[str, Any]
    config_source: ConfigSource
    state: BindingState
    validation: ModelValidationResult | None
    api_key: ModelApiKeyStatus
    saved_at: str | None = Field(
        description="When the stored binding was last saved; null for one read from the environment.",
    )


class ModelKindSettingsResponse(BaseModel):
    active: ModelBindingResponse | None = Field(description="The binding the runtime uses.")
    candidate: ModelBindingResponse | None = Field(
        description="The binding Settings is editing, not yet in use.",
    )


class ModelSettingsResponse(BaseModel):
    writable: bool = Field(
        description=(
            "False when the deployment environment manages model settings "
            "(MEMFORGE_LLM_CONFIG_WRITABLE), so every write is refused."
        ),
    )
    litellm_version: str
    generation: ModelKindSettingsResponse
    embedding: ModelKindSettingsResponse


class ModelCandidateRequest(BaseModel):
    provider: str = Field(default="", description="The id of a provider descriptor.")
    model_id: str = Field(default="", description="The model, as the provider descriptor reads it.")
    endpoint: str | None = None
    auth_strategy: AuthStrategy | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    api_key: str | None = Field(
        default=None,
        description="A new API key. Leave it out to keep the key saved for this endpoint.",
    )


class ModelValidationResponse(BaseModel):
    passed: bool
    promoted: bool = Field(description="Whether the checked candidate became the active binding.")
    error_code: str | None = None
    validation: ModelValidationResult
    settings: ModelKindSettingsResponse


class ModelOption(BaseModel):
    id: str
    label: str | None = None


class ModelDiscoveryRequest(BaseModel):
    provider: str = Field(default="", description="The id of a provider descriptor.")
    endpoint: str | None = None
    auth_strategy: AuthStrategy | None = None
    api_key: str | None = Field(
        default=None,
        description="Leave it out to use the key saved for this endpoint's origin.",
    )


class ModelDiscoveryResponse(BaseModel):
    ok: bool
    models_supported: bool = False
    models: list[ModelOption] = Field(default_factory=list)
    stage: DiscoveryStage | None = None
    error_code: str | None = None
    status: int | None = None
    message: str
    latency_ms: int | None = None
    suggested_endpoint: str | None = None


class ModelKindHealthResponse(BaseModel):
    configured: bool = Field(description="Whether a binding of this kind is in use.")
    state: BindingState | None = Field(
        default=None,
        description="The state of that binding; null when none is configured.",
    )


class ModelHealthResponse(BaseModel):
    generation: ModelKindHealthResponse
    embedding: ModelKindHealthResponse


async def model_health_response(settings: ModelSettings, db: Any) -> ModelHealthResponse:
    report = await settings.health(db)
    return ModelHealthResponse(
        generation=ModelKindHealthResponse(
            configured=report["generation"].configured,
            state=report["generation"].state,
        ),
        embedding=ModelKindHealthResponse(
            configured=report["embedding"].configured,
            state=report["embedding"].state,
        ),
    )


def build_model_settings_router(
    *,
    get_db: Callable[..., Any],
    get_model_settings: Callable[..., ModelSettings],
) -> APIRouter:
    router = APIRouter(
        prefix="/api/v1/model-settings",
        tags=["model-settings"],
        route_class=_RequestNotRepeatedRoute,
    )

    @router.get("/providers", response_model=ModelProviderListResponse)
    async def list_model_providers():
        """List the provider descriptors a binding can name."""
        return ModelProviderListResponse(
            providers=[_descriptor_response(descriptor) for descriptor in PROVIDER_DESCRIPTORS]
        )

    @router.get("", response_model=ModelSettingsResponse)
    async def read_model_settings(
        db: Any = Depends(get_db),
        settings: ModelSettings = Depends(get_model_settings),
    ):
        """Read the active and candidate bindings of both kinds. No key is returned."""
        return ModelSettingsResponse(
            writable=settings.writable,
            litellm_version=installed_litellm_version(),
            generation=_kind_response(await settings.view(db, "generation")),
            embedding=_kind_response(await settings.view(db, "embedding")),
        )

    @router.put("/{kind}/candidate", response_model=ModelKindSettingsResponse)
    async def save_model_candidate(
        kind: ModelKind,
        req: ModelCandidateRequest,
        db: Any = Depends(get_db),
        settings: ModelSettings = Depends(get_model_settings),
    ):
        """Save the candidate binding of a kind. The active binding is not changed."""
        try:
            view = await settings.save_candidate(
                db,
                kind,
                CandidateDraft(
                    provider=req.provider,
                    model_id=req.model_id,
                    endpoint=req.endpoint,
                    auth_strategy=req.auth_strategy,
                    options=req.options,
                    parameters=req.parameters,
                    api_key=req.api_key,
                ),
            )
        except ModelConfigError as exc:
            raise _http_error(exc) from exc
        return _kind_response(view)

    @router.delete("/{kind}/candidate", response_model=ModelKindSettingsResponse)
    async def discard_model_candidate(
        kind: ModelKind,
        db: Any = Depends(get_db),
        settings: ModelSettings = Depends(get_model_settings),
    ):
        """Discard the candidate binding of a kind."""
        try:
            view = await settings.discard_candidate(db, kind)
        except ModelConfigError as exc:
            raise _http_error(exc) from exc
        return _kind_response(view)

    @router.post("/{kind}/candidate/validate", response_model=ModelValidationResponse)
    async def validate_model_candidate(
        kind: ModelKind,
        request: Request,
        db: Any = Depends(get_db),
        settings: ModelSettings = Depends(get_model_settings),
    ):
        """Check the candidate with one real request and make it active when it passes."""
        try:
            outcome = await settings.validate_candidate(db, kind)
        except ModelConfigError as exc:
            raise _http_error(exc) from exc
        if outcome.promoted:
            _discard_runtimes_of_the_replaced_binding(request.app)
        return await _validation_response(settings, db, outcome)

    @router.post("/{kind}/active/validate", response_model=ModelValidationResponse)
    async def validate_active_model(
        kind: ModelKind,
        db: Any = Depends(get_db),
        settings: ModelSettings = Depends(get_model_settings),
    ):
        """Check the active binding again. It stays in use whatever the outcome."""
        outcome = await settings.validate_active(db, kind)
        if outcome is None:
            raise _http_error(
                ModelConfigError("model_not_configured", f"No {kind} model is configured.")
            )
        return await _validation_response(settings, db, outcome)

    @router.post("/{kind}/discover-models", response_model=ModelDiscoveryResponse)
    async def discover_models(
        kind: ModelKind,
        req: ModelDiscoveryRequest,
        db: Any = Depends(get_db),
        settings: ModelSettings = Depends(get_model_settings),
    ):
        """List the models an endpoint offers, for choosing a candidate's model."""
        try:
            descriptor = provider_descriptor(req.provider)
        except ModelConfigError as exc:
            raise _http_error(exc) from exc
        endpoint = (req.endpoint or "").strip() or descriptor.default_endpoint or ""
        if not descriptor.lists_models:
            return ModelDiscoveryResponse(
                ok=True,
                models_supported=False,
                message=f"{descriptor.label} has no model list. Enter the model yourself.",
            )
        origin = endpoint_origin(endpoint)
        if origin is None:
            return _discovery_error(
                endpoint=endpoint,
                stage="validation",
                error_code="endpoint_invalid",
                message="Enter a URL starting with http:// or https://.",
            )
        if endpoint_carries_credentials(endpoint):
            return ModelDiscoveryResponse(
                ok=False,
                stage="validation",
                error_code="endpoint_invalid",
                message="Enter the base URL without a user name or password.",
            )
        api_key = (req.api_key or "").strip() or None
        strategy = req.auth_strategy or descriptor.auth_strategies[0]
        if api_key is None and strategy == "api_key":
            api_key, key_required = await settings.discovery_key(db, kind, origin=origin_label(origin))
            if key_required:
                return _discovery_error(
                    endpoint=endpoint,
                    stage="auth",
                    error_code="api_key_origin_changed",
                    message=(
                        "The saved API key belongs to another endpoint. "
                        "Enter the API key for this endpoint, then try again."
                    ),
                )
        return await _discover_models(descriptor, endpoint=endpoint, api_key=api_key, strategy=strategy)

    return router


def _discard_runtimes_of_the_replaced_binding(app: Any) -> None:
    """Make the server build its model clients again, for the binding now active.

    The search engine and the embedded sync worker keep what they built: a
    runtime holding the model, the endpoint and the key of the binding a
    promotion replaces.
    """
    app.state.memory_search_engine = None
    sync_worker = getattr(app.state, "sync_worker", None)
    if sync_worker is not None:
        sync_worker.models_changed()


def _descriptor_response(descriptor: ProviderDescriptor) -> ModelProviderResponse:
    return ModelProviderResponse(
        id=descriptor.id,
        label=descriptor.label,
        kinds=list(descriptor.kinds),
        auth_strategies=list(descriptor.auth_strategies),
        default_endpoint=descriptor.default_endpoint,
        endpoint_required=descriptor.endpoint_required,
        model_is_route=descriptor.model_is_route,
        lists_models=descriptor.lists_models,
        options=list(descriptor.options),
        parameters={kind: list(parameters_for_kind(kind)) for kind in descriptor.kinds},
    )


def _binding_response(view: BindingView) -> ModelBindingResponse:
    binding: ModelBinding = view.binding
    validation = binding.validation
    return ModelBindingResponse(
        kind=binding.kind,
        provider=binding.provider,
        endpoint=binding.endpoint,
        origin=binding.origin,
        auth_strategy=binding.auth_strategy,
        model_id=binding.model_id,
        route=binding.route,
        options=dict(binding.options),
        parameters=dict(binding.parameters),
        config_source=binding.config_source,
        state=binding.state,
        validation=(
            None
            if validation is None
            else ModelValidationResult(
                checked_at=validation.checked_at,
                passed=validation.passed,
                error_code=validation.error_code,
                facts=dict(validation.facts),
                current=validation.digest == binding.digest,
            )
        ),
        api_key=ModelApiKeyStatus(
            is_set=view.key.is_set,
            source=view.key.source,
            last4=view.key.last4,
        ),
        saved_at=binding.saved_at,
    )


def _kind_response(view: KindView) -> ModelKindSettingsResponse:
    return ModelKindSettingsResponse(
        active=None if view.active is None else _binding_response(view.active),
        candidate=None if view.candidate is None else _binding_response(view.candidate),
    )


async def _validation_response(
    settings: ModelSettings,
    db: Any,
    outcome: ValidationOutcome,
) -> ModelValidationResponse:
    validation = outcome.validation
    return ModelValidationResponse(
        passed=validation.passed,
        promoted=outcome.promoted,
        error_code=validation.error_code,
        validation=ModelValidationResult(
            checked_at=validation.checked_at,
            passed=validation.passed,
            error_code=validation.error_code,
            facts=dict(validation.facts),
            current=True,
        ),
        settings=_kind_response(await settings.view(db, outcome.kind)),
    )


def _http_error(exc: ModelConfigError) -> HTTPException:
    return HTTPException(
        status_code=_ERROR_STATUS.get(exc.code, 422),
        detail={"error": exc.code, "message": exc.message},
    )


def _is_running_in_container() -> bool:
    return Path("/.dockerenv").exists()


def _suggest_host_endpoint(endpoint: str) -> str | None:
    parsed = urlsplit(endpoint)
    if parsed.hostname not in {"localhost", "127.0.0.1"}:
        return None
    if not _is_running_in_container():
        return None
    netloc = "host.docker.internal"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _model_list_urls(endpoint: str) -> list[str]:
    root = endpoint.strip().rstrip("/")
    urls = [f"{root}/models"]
    parsed = urlsplit(root)
    if not parsed.path.rstrip("/").endswith("/v1"):
        urls.append(f"{root}/v1/models")
    return list(dict.fromkeys(urls))


def _model_list_headers(descriptor: ProviderDescriptor, api_key: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if not api_key:
        return headers
    if descriptor.key_header == "x-api-key":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _extract_model_options(payload: Any) -> list[ModelOption]:
    if isinstance(payload, dict):
        raw_items = payload.get("data")
        if raw_items is None:
            raw_items = payload.get("models")
    else:
        raw_items = payload

    if not isinstance(raw_items, list):
        return []

    seen: set[str] = set()
    models: list[ModelOption] = []
    for item in raw_items:
        model_id: str | None = None
        label: str | None = None
        if isinstance(item, str):
            model_id = item
        elif isinstance(item, dict):
            raw_id = item.get("id") or item.get("name") or item.get("model")
            if raw_id is not None:
                model_id = str(raw_id)
            raw_label = item.get("display_name") or item.get("label") or item.get("name")
            if raw_label is not None and str(raw_label) != model_id:
                label = str(raw_label)
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        models.append(ModelOption(id=model_id, label=label))
    return models


def _discovery_error(
    *,
    endpoint: str,
    stage: DiscoveryStage,
    error_code: str,
    message: str,
    status: int | None = None,
    latency_ms: int | None = None,
) -> ModelDiscoveryResponse:
    return ModelDiscoveryResponse(
        ok=False,
        stage=stage,
        error_code=error_code,
        status=status,
        message=message,
        latency_ms=latency_ms,
        suggested_endpoint=_suggest_host_endpoint(endpoint),
    )


def _refused_listing(
    strategy: AuthStrategy,
    api_key: str | None,
    status: int,
) -> tuple[str, str]:
    """The error code and message for an endpoint that refused to list its models.

    A listing request carries an API key or nothing: the credentials the
    ``environment`` strategy stands for are attached by the SDK to model
    requests only. An endpoint that wants a key therefore cannot list its
    models under that strategy, and no key entered would change that.
    """
    if api_key:
        return "authentication_failed", f"Endpoint rejected the API key (HTTP {status})."
    if strategy == "environment":
        return (
            "listing_needs_api_key",
            "This endpoint lists its models only for an API key, and credentials from the "
            "server environment are not sent with this request. Enter the model yourself.",
        )
    return "api_key_required", "Add an API key, then try again."


async def _discover_models(
    descriptor: ProviderDescriptor,
    *,
    endpoint: str,
    api_key: str | None,
    strategy: AuthStrategy,
) -> ModelDiscoveryResponse:
    """List the models an endpoint offers, without ever leaving its origin.

    The request carries the API key, so it does not follow redirects: an answer
    that points to another origin fails the listing, and one that points
    elsewhere on the same origin only rules out the address that gave it.
    """
    value = endpoint.strip()
    host = urlsplit(value).hostname or "the endpoint"
    headers = _model_list_headers(descriptor, api_key)
    unsupported: tuple[int, int] | None = None
    redirected: tuple[int, int] | None = None
    non_json_latency_ms: int | None = None

    async with httpx.AsyncClient(timeout=5.0, follow_redirects=False) as client:
        for url in _model_list_urls(value):
            started = time.perf_counter()
            try:
                resp = await client.get(url, headers=headers)
            except httpx.TimeoutException:
                return _discovery_error(
                    endpoint=value,
                    stage="timeout",
                    error_code="timeout",
                    message="No response before the timeout.",
                )
            except httpx.ConnectError as exc:
                text = str(exc).lower()
                stage: DiscoveryStage = "tls" if "ssl" in text or "tls" in text else "connect"
                return _discovery_error(
                    endpoint=value,
                    stage=stage,
                    error_code="provider_unreachable",
                    message=f"Could not reach {host}.",
                )
            except httpx.HTTPError:
                return _discovery_error(
                    endpoint=value,
                    stage="connect",
                    error_code="provider_unreachable",
                    message=f"Could not reach {host}.",
                )

            latency_ms = int((time.perf_counter() - started) * 1000)
            if resp.has_redirect_location:
                target_origin = endpoint_origin(urljoin(url, resp.headers["location"]))
                if target_origin != endpoint_origin(url):
                    target = (
                        "an address that is not HTTP(S)"
                        if target_origin is None
                        else origin_label(target_origin)
                    )
                    return _discovery_error(
                        endpoint=value,
                        stage="http",
                        error_code="redirect_to_another_origin",
                        status=resp.status_code,
                        latency_ms=latency_ms,
                        message=(
                            f"Endpoint redirected to {target}, "
                            "another origin, and the redirect is not followed. "
                            "Enter the base URL of the endpoint that answers directly."
                        ),
                    )
                redirected = (resp.status_code, latency_ms)
                continue
            if resp.status_code in {404, 405, 501}:
                unsupported = (resp.status_code, latency_ms)
                continue
            if resp.status_code in {401, 403}:
                error_code, message = _refused_listing(strategy, api_key, resp.status_code)
                return _discovery_error(
                    endpoint=value,
                    stage="auth",
                    error_code=error_code,
                    status=resp.status_code,
                    latency_ms=latency_ms,
                    message=message,
                )
            if resp.status_code >= 400:
                return _discovery_error(
                    endpoint=value,
                    stage="http",
                    error_code="provider_error",
                    status=resp.status_code,
                    latency_ms=latency_ms,
                    message=f"Endpoint returned HTTP {resp.status_code}.",
                )

            try:
                data = resp.json()
            except ValueError:
                non_json_latency_ms = latency_ms
                continue

            models = _extract_model_options(data)
            return ModelDiscoveryResponse(
                ok=True,
                models_supported=True,
                models=models,
                message=(
                    f"Connected. Found {len(models)} model{'s' if len(models) != 1 else ''}."
                    if models
                    else "Connected, but this endpoint returned no models."
                ),
                latency_ms=latency_ms,
            )

    if non_json_latency_ms is not None:
        return ModelDiscoveryResponse(
            ok=True,
            models_supported=False,
            message="Connected, but this endpoint did not return a model list.",
            latency_ms=non_json_latency_ms,
        )

    if redirected is not None:
        status, latency_ms = redirected
        return _discovery_error(
            endpoint=value,
            stage="http",
            error_code="redirect_not_followed",
            status=status,
            latency_ms=latency_ms,
            message=(
                "Endpoint answered with a redirect, which is not followed. "
                "Enter the base URL of the endpoint that answers directly."
            ),
        )

    status, latency_ms = unsupported or (404, None)
    return ModelDiscoveryResponse(
        ok=True,
        models_supported=False,
        status=status,
        message="Connected, but this endpoint does not expose a model list.",
        latency_ms=latency_ms,
    )
