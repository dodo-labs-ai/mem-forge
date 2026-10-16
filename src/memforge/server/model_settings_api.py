"""HTTP API for model settings.

Provider descriptors, the bindings of both kinds, their validation, model
discovery, and the Memory vector index that belongs to the embedding model.
"""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Callable, Mapping
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
    token_limits_for_kind,
)
from memforge.llm.embedding_profile import EmbeddingProfile
from memforge.llm.providers import (
    PROVIDER_DESCRIPTORS,
    AuthStrategy,
    ModelConfigError,
    ModelKind,
    ModelListing,
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
    VectorIndexStatus,
    VectorIndexView,
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
    "reindex_in_progress": 409,
    "sync_in_progress": 409,
    "reindex_not_required": 409,
    "embedding_not_validated": 409,
    "reindex_not_available": 405,
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
    lists_token_limits: bool = Field(
        description=(
            "Whether the model listing can give the token limits of a listed model: its "
            "context window and its largest output."
        ),
    )
    options: list[str] = Field(description="The names of the non-secret options the provider accepts.")
    parameters: dict[ModelKind, list[str]] = Field(
        description="The request parameters a binding of each served kind may set.",
    )
    token_limits: dict[ModelKind, list[str]] = Field(
        description=(
            "The token limits a binding of each served kind may state for its model: "
            "context_window_tokens and max_output_tokens for a generation model."
        ),
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


class EmbeddingProfileResponse(BaseModel):
    """The identity of one vector space: an embedding model's settings and its vector length."""

    provider: str
    route: str
    endpoint: str | None
    model_id: str
    requested_dimensions: int | None = Field(
        description="The output size requests ask for; null takes the model's own.",
    )
    dimension: int = Field(description="The vector length the model answered with.")
    fingerprint: str = Field(
        description="A hash of the fields above. Vectors are shared only between equal fingerprints.",
    )
    thresholds_calibrated: bool = Field(
        description=(
            "Whether the similarity thresholds were tuned on this model's vectors. When false "
            "they still apply, as approximate values nobody has measured for this model."
        ),
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
    token_limits: dict[str, int] = Field(
        description=(
            "The token limits the binding states for its model: context_window_tokens and "
            "max_output_tokens. Empty when it states none."
        ),
    )
    config_source: ConfigSource
    state: BindingState
    validation: ModelValidationResult | None
    api_key: ModelApiKeyStatus
    saved_at: str | None = Field(
        description="When the stored binding was last saved; null for one read from the environment.",
    )
    profile: EmbeddingProfileResponse | None = Field(
        default=None,
        description=(
            "For an embedding binding whose latest check passed for the binding as it is: "
            "the Embedding Profile it embeds in. Null otherwise."
        ),
    )
    waits_for_reindex: bool = Field(
        default=False,
        description=(
            "True for an embedding candidate that passed its check and becomes active once "
            "the vector index is rebuilt for it."
        ),
    )


class ModelKindSettingsResponse(BaseModel):
    active: ModelBindingResponse | None = Field(description="The binding the runtime uses.")
    candidate: ModelBindingResponse | None = Field(
        description="The binding Settings is editing, not yet in use.",
    )


class VectorIndexRecordResponse(BaseModel):
    """One physical Memory vector index."""

    fingerprint: str = Field(description="The fingerprint of the Embedding Profile it was built with.")
    collection: str = Field(description="The name of the collection that holds its vectors.")
    dimension: int
    adopted: bool = Field(
        description=(
            "True when the vectors were in place before the index was recorded: the model "
            "they came from is assumed to be the profile's, not known."
        ),
    )
    vector_count: int = Field(description="The vectors the index holds now.")
    activated_at: str | None = Field(description="When the index came into use.")
    profile: EmbeddingProfileResponse | None = Field(
        description="The Embedding Profile the index was built with.",
    )


class VectorReindexResponse(BaseModel):
    """The latest rebuild of the vector index."""

    state: Literal["running", "succeeded", "failed"]
    target: Literal["active", "candidate"] | None = Field(
        description="The embedding binding the rebuild is for.",
    )
    fingerprint: str
    profile: EmbeddingProfileResponse | None
    memories_total: int | None = Field(
        description="The searchable Memories the rebuild covers; null until it has counted them.",
    )
    memories_indexed: int = Field(description="The Memories embedded into the new index so far.")
    started_at: str | None
    finished_at: str | None
    error_code: str | None = Field(
        default=None,
        description=(
            "Why a failed rebuild failed: a provider code of the embedding check "
            "(provider_unreachable, authentication_failed, rate_limited, timeout, "
            "embedding_response_invalid, provider_error, ...), embedding_dimension_mismatch, "
            "vector_count_mismatch, vector_invalid, memories_changed, candidate_changed, "
            "reindex_lease_lost, interrupted or index_error."
        ),
    )


class VectorIndexResponse(BaseModel):
    status: VectorIndexStatus = Field(
        description=(
            "'ready': the embedding model in use is the one the index was built with. "
            "'reindex_required': it is another one, or the stored vectors cannot be adopted; "
            "until the index is rebuilt vector search is off, a write that needs a new vector "
            "is refused and vector work already queued waits. "
            "'reindexing': a rebuild is running. 'awaiting_validation': no index is in use "
            "because the embedding model has not passed a check yet. 'not_configured': no "
            "embedding model. 'not_managed': the deployment keeps its index by its own means."
        ),
    )
    reindex_required: bool = Field(description="True exactly when status is 'reindex_required'.")
    serves_search: bool = Field(
        description=(
            "Whether search reads the index in use: there is one, and the embedding model in "
            "use is the one it was built with. A rebuild does not change this before it "
            "switches, so during a rebuild that started while a reindex was required it "
            "stays false."
        ),
    )
    reindex_target: Literal["active", "candidate"] | None = Field(
        description=(
            "The embedding binding a rebuild started now would be for: 'candidate' for a "
            "candidate that waits for the index, 'active' when the index in use is missing, "
            "was built with another model or was adopted. An active binding that has not "
            "passed a check as it stands is checked when the rebuild is started, which is "
            "refused with embedding_not_validated when the check fails. Null when there is "
            "nothing to rebuild for, and while a rebuild runs."
        ),
    )
    index: VectorIndexRecordResponse | None = Field(description="The index in use.")
    reindex: VectorReindexResponse | None = Field(
        description="The latest rebuild, running or ended; null when none has been started.",
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
    vector_index: VectorIndexResponse


class ModelCandidateRequest(BaseModel):
    provider: str = Field(default="", description="The id of a provider descriptor.")
    model_id: str = Field(default="", description="The model, as the provider descriptor reads it.")
    endpoint: str | None = None
    auth_strategy: AuthStrategy | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    token_limits: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "For a generation model LiteLLM has no limits for: context_window_tokens and "
            "max_output_tokens, whole numbers of tokens."
        ),
    )
    api_key: str | None = Field(
        default=None,
        description="A new API key. Leave it out to keep the key saved for this endpoint.",
    )


class ModelValidationResponse(BaseModel):
    passed: bool
    promoted: bool = Field(description="Whether the checked candidate became the active binding.")
    reindex_required: bool = Field(
        default=False,
        description=(
            "True when an embedding candidate passed and stays the candidate until the "
            "vector index is rebuilt for it."
        ),
    )
    error_code: str | None = None
    validation: ModelValidationResult
    settings: ModelKindSettingsResponse


class ModelOption(BaseModel):
    id: str
    label: str | None = None
    context_window_tokens: int | None = Field(
        default=None,
        description="The context window the listing gives for the model, in tokens; null when it gives none.",
    )
    max_output_tokens: int | None = Field(
        default=None,
        description="The largest output the listing gives for the model, in tokens; null when it gives none.",
    )


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
    models: list[ModelOption] = Field(
        default_factory=list,
        description=(
            "The listed models of the requested kind, or every listed model when "
            "kind_filtered is false."
        ),
    )
    kind_filtered: bool = Field(
        default=False,
        description=(
            "Whether models holds only models of the requested kind. False when the endpoint "
            "lists its models without saying which kind each one is."
        ),
    )
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


class EmbeddingModelHealthResponse(ModelKindHealthResponse):
    vector_index: VectorIndexStatus = Field(
        description="The state of the Memory vector index, as the model settings report it.",
    )
    reindex_required: bool = Field(
        description=(
            "True when the vector index has to be rebuilt for the embedding model in use. "
            "Until then search answers without its vector channel."
        ),
    )


class ModelHealthResponse(BaseModel):
    generation: ModelKindHealthResponse
    embedding: EmbeddingModelHealthResponse


async def model_health_response(settings: ModelSettings, db: Any) -> ModelHealthResponse:
    report = await settings.health(db)
    vector_index = report["embedding"].vector_index or "not_managed"
    return ModelHealthResponse(
        generation=ModelKindHealthResponse(
            configured=report["generation"].configured,
            state=report["generation"].state,
        ),
        embedding=EmbeddingModelHealthResponse(
            configured=report["embedding"].configured,
            state=report["embedding"].state,
            vector_index=vector_index,
            reindex_required=vector_index == "reindex_required",
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
            vector_index=_vector_index_response(await settings.vector_index(db, count_vectors=True)),
        )

    @router.get("/vector-index", response_model=VectorIndexResponse)
    async def read_vector_index(
        db: Any = Depends(get_db),
        settings: ModelSettings = Depends(get_model_settings),
    ):
        """Read the Memory vector index in use and the latest rebuild of it."""
        return _vector_index_response(await settings.vector_index(db, count_vectors=True))

    @router.post("/vector-index/reindex", response_model=VectorIndexResponse, status_code=202)
    async def start_vector_reindex(
        request: Request,
        db: Any = Depends(get_db),
        settings: ModelSettings = Depends(get_model_settings),
    ):
        """Start a rebuild of the vector index. It runs beside the server; read its state with GET."""
        app = request.app
        try:
            view = await settings.start_reindex(
                db,
                on_switch=lambda: _discard_runtimes_of_the_replaced_binding(app),
            )
        except ModelConfigError as exc:
            raise _http_error(exc) from exc
        return _vector_index_response(view)

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
                    token_limits=req.token_limits,
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
        endpoint = descriptor.base_endpoint(req.endpoint or "") or descriptor.default_endpoint or ""
        listing = descriptor.listing
        if listing is None:
            return ModelDiscoveryResponse(
                ok=True,
                models_supported=False,
                message=f"{descriptor.label} has no model list. Enter the model yourself.",
            )
        if kind not in listing.paths:
            return ModelDiscoveryResponse(
                ok=True,
                models_supported=False,
                message=f"{descriptor.label} serves no {kind} models.",
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
        return await _discover_models(
            descriptor,
            listing,
            kind,
            endpoint=endpoint,
            api_key=api_key,
            strategy=strategy,
        )

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
        lists_token_limits=descriptor.lists_token_limits,
        options=list(descriptor.options),
        parameters={kind: list(parameters_for_kind(kind)) for kind in descriptor.kinds},
        token_limits={kind: list(token_limits_for_kind(kind)) for kind in descriptor.kinds},
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
        token_limits=dict(binding.token_limits),
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
        profile=_profile_response(view.profile),
        waits_for_reindex=view.waits_for_reindex,
    )


def _profile_response(profile: EmbeddingProfile | None) -> EmbeddingProfileResponse | None:
    if profile is None:
        return None
    return EmbeddingProfileResponse(
        **profile.fields(),
        fingerprint=profile.fingerprint,
        thresholds_calibrated=profile.thresholds_calibrated,
    )


def _recorded_profile(record: Mapping[str, Any]) -> EmbeddingProfileResponse | None:
    fields = record.get("profile") or {}
    return _profile_response(EmbeddingProfile.from_fields(fields)) if fields else None


def _vector_index_response(view: VectorIndexView) -> VectorIndexResponse:
    in_use = view.in_use
    reindex = view.reindex
    return VectorIndexResponse(
        status=view.status,
        reindex_required=view.reindex_required,
        serves_search=view.serves_search,
        reindex_target=view.reindex_target,
        index=(
            None
            if in_use is None
            else VectorIndexRecordResponse(
                fingerprint=in_use["fingerprint"],
                collection=in_use["collection_name"],
                dimension=in_use["dimension"],
                adopted=in_use["adopted"],
                vector_count=in_use["vector_count"] if view.vector_count is None else view.vector_count,
                activated_at=in_use["activated_at"],
                profile=_recorded_profile(in_use),
            )
        ),
        reindex=(
            None
            if reindex is None
            else VectorReindexResponse(
                state=_reindex_state(reindex, running=view.reindex_running),
                target=reindex["binding_slot"],
                fingerprint=reindex["fingerprint"],
                profile=_recorded_profile(reindex),
                memories_total=reindex["memory_count"],
                memories_indexed=reindex["vector_count"],
                started_at=reindex["started_at"],
                finished_at=reindex["finished_at"],
                error_code=_reindex_error_code(reindex, running=view.reindex_running),
            )
        ),
    )


def _reindex_state(
    reindex: Mapping[str, Any],
    *,
    running: bool,
) -> Literal["running", "succeeded", "failed"]:
    """Read the state of a rebuild; one that stopped reporting has failed."""
    if reindex["state"] == "building":
        return "running" if running else "failed"
    return "failed" if reindex["state"] == "failed" else "succeeded"


def _reindex_error_code(reindex: Mapping[str, Any], *, running: bool) -> str | None:
    if reindex["state"] == "building" and not running:
        return "interrupted"
    return reindex["error_code"]


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
        reindex_required=outcome.reindex_required,
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


def _model_list_urls(endpoint: str, listing: ModelListing, kind: ModelKind) -> list[str]:
    root = endpoint.strip().rstrip("/")
    path = listing.paths[kind]
    urls = [f"{root}{path}"]
    if listing.tries_v1 and not urlsplit(root).path.rstrip("/").endswith("/v1"):
        urls.append(f"{root}/v1{path}")
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


# How long one model listing request may take, from sending it to the last
# byte of the answer. The HTTP client's own timeout bounds each wait on its
# own, which an endpoint that sends a byte now and then never runs out of.
_LISTING_TIMEOUT_S = 5.0

# Ollama families of models that only embed text; Ollama names a model's family
# by its architecture.
_EMBEDDING_FAMILIES = frozenset({"bert", "nomic-bert", "xlm-roberta"})
# A word in a model id that names a model which does not answer in text: an
# image, audio or moderation model of an OpenAI-style listing.
_NON_TEXT_MODEL = re.compile(
    r"(?<![a-z0-9])(dall-e|image|whisper|tts|transcribe|audio|realtime|moderation|sora)(?![a-z])"
)


def _listed_entries(payload: Any) -> list[Any]:
    if isinstance(payload, dict):
        raw_items = payload.get("data")
        if raw_items is None:
            raw_items = payload.get("models")
    else:
        raw_items = payload
    return raw_items if isinstance(raw_items, list) else []


def _token_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _first_token_count(*values: Any) -> int | None:
    for value in values:
        count = _token_count(value)
        if count is not None:
            return count
    return None


def _model_option(item: Any) -> ModelOption | None:
    """Read one listed model, with the token limits its entry gives.

    The context window is read from ``context_length`` (OpenRouter and others),
    ``context_window`` or ``max_model_len`` (vLLM); the largest output from
    ``top_provider.max_completion_tokens`` (OpenRouter) or
    ``max_completion_tokens``.
    """
    if isinstance(item, str):
        return ModelOption(id=item) if item else None
    if not isinstance(item, dict):
        return None
    raw_id = item.get("id") or item.get("name") or item.get("model")
    if raw_id is None or not str(raw_id):
        return None
    model_id = str(raw_id)
    raw_label = item.get("display_name") or item.get("label") or item.get("name")
    top_provider = item.get("top_provider")
    if not isinstance(top_provider, dict):
        top_provider = {}
    return ModelOption(
        id=model_id,
        label=None if raw_label is None or str(raw_label) == model_id else str(raw_label),
        context_window_tokens=_first_token_count(
            item.get("context_length"),
            item.get("context_window"),
            item.get("max_model_len"),
        ),
        max_output_tokens=_first_token_count(
            top_provider.get("max_completion_tokens"),
            item.get("max_completion_tokens"),
        ),
    )


def _declared_kinds(item: Any) -> frozenset[ModelKind] | None:
    """The kinds a listed entry declares through ``capabilities``, as Ollama writes them.

    ``completion`` is a generation model and ``embedding`` an embedding model.
    Returns ``None`` for an entry that declares no capabilities.
    """
    capabilities = item.get("capabilities") if isinstance(item, dict) else None
    if not isinstance(capabilities, list):
        return None
    kinds: set[ModelKind] = set()
    if "completion" in capabilities:
        kinds.add("generation")
    if "embedding" in capabilities:
        kinds.add("embedding")
    return frozenset(kinds)


def _entry_kinds(item: Any, model_id: str) -> frozenset[ModelKind]:
    """The kinds one entry of a listing that holds every model serves.

    What the entry declares decides. Otherwise an Ollama family of embedding
    models, or ``embed`` in the id, makes it an embedding model; an id that
    names an image, audio or moderation model makes it neither; any other
    model generates text.
    """
    declared = _declared_kinds(item)
    if declared is not None:
        return declared
    details = item.get("details") if isinstance(item, dict) else None
    if isinstance(details, dict):
        families = details.get("families")
        named = {details.get("family"), *(families if isinstance(families, list) else ())}
        if named & _EMBEDDING_FAMILIES:
            return frozenset({"embedding"})
    lowered = model_id.lower()
    if "embed" in lowered:
        return frozenset({"embedding"})
    if _NON_TEXT_MODEL.search(lowered):
        return frozenset()
    return frozenset({"generation"})


def _extract_model_options(payload: Any, listing: ModelListing, kind: ModelKind) -> tuple[list[ModelOption], int]:
    """Read the models of ``kind`` from a listing, with the number of models it listed.

    A listing of one kind holds only models of that kind, and a listing that
    does not tell the kinds apart gives every model it lists.
    """
    seen: set[str] = set()
    listed = 0
    models: list[ModelOption] = []
    for item in _listed_entries(payload):
        option = _model_option(item)
        if option is None or option.id in seen:
            continue
        seen.add(option.id)
        listed += 1
        if listing.kind_signal != "entry" or kind in _entry_kinds(item, option.id):
            models.append(option)
    return models, listed


def _listing_message(models: list[ModelOption], listed: int, kind: ModelKind, *, kind_filtered: bool) -> str:
    def counted(count: int, noun: str) -> str:
        return f"{count} {noun}{'s' if count != 1 else ''}"

    if listed == 0:
        return "Connected, but this endpoint returned no models."
    if not kind_filtered:
        return (
            f"Connected. Found {counted(listed, 'model')}; the endpoint does not say "
            f"which are {kind} models."
        )
    if not models:
        return (
            f"Connected, but this endpoint listed no {kind} models. "
            "Type the model id if you know it."
        )
    return f"Connected. Found {counted(len(models), f'{kind} model')}."


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
    listing: ModelListing,
    kind: ModelKind,
    *,
    endpoint: str,
    api_key: str | None,
    strategy: AuthStrategy,
) -> ModelDiscoveryResponse:
    """List the models of ``kind`` an endpoint offers, without ever leaving its origin.

    The request carries the API key, so it does not follow redirects: an answer
    that points to another origin fails the listing, and one that points
    elsewhere on the same origin only rules out the address that gave it. An
    address that has not answered in full within ``_LISTING_TIMEOUT_S`` fails
    the listing as a timeout.
    """
    value = endpoint.strip()
    host = urlsplit(value).hostname or "the endpoint"
    headers = _model_list_headers(descriptor, api_key)
    unsupported: tuple[int, int] | None = None
    redirected: tuple[int, int] | None = None
    non_json_latency_ms: int | None = None

    async with httpx.AsyncClient(timeout=_LISTING_TIMEOUT_S, follow_redirects=False) as client:
        for url in _model_list_urls(value, listing, kind):
            started = time.perf_counter()
            try:
                async with asyncio.timeout(_LISTING_TIMEOUT_S):
                    resp = await client.get(url, headers=headers)
            except (TimeoutError, httpx.TimeoutException):
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

            models, listed = _extract_model_options(data, listing, kind)
            kind_filtered = listing.kind_signal != "none"
            return ModelDiscoveryResponse(
                ok=True,
                models_supported=True,
                models=models,
                kind_filtered=kind_filtered,
                message=_listing_message(models, listed, kind, kind_filtered=kind_filtered),
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
