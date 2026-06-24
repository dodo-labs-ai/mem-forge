"""Admin REST API for MemForge.

FastAPI application providing management endpoints for memories, entities,
sources/genes, schedule, LLM configuration, and system health/stats.
"""

from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from memforge.config import AppConfig
from memforge.auth import browser_session
from memforge.auth.jira_auth import (
    JiraAuthSessionError,
    JiraAuthSessionService,
    JiraPrincipalChangedError,
    canonical_jira_origin,
    effective_jira_auth_mode,
)
from memforge.genes import GENE_REGISTRY, create_gene, list_available_genes
from memforge.genes.atlassian_auth import (
    release_atlassian_request_limiter,
    require_https_base_url,
    validate_tls_ca_bundle,
)
from memforge.memory.lifecycle import normalize_memory_status
from memforge.memory.review_service import (
    ReviewAlreadyResolved,
    ReviewError,
    ReviewKindUnsupported,
    ReviewNotFound,
    ReviewService,
    ReviewStaleConflict,
)
from memforge.memory.store import MemoryStore
from memforge.models import (
    ConfigField,
    ConfigFieldType,
    Memory,
    MemoryStatus,
    MemoryType,
    MemoryReview,
    Project,
    UNSORTED_PROJECT_KEY,
    canonicalize_entity_name,
)
from memforge.provenance import (
    DocumentArtifactStore,
    document_content_url,
    document_content_url_for_store,
    document_pdf_url,
    document_pdf_url_for_store,
    list_document_artifacts,
    select_document_artifact,
)
from memforge.retrieval.filters import MemorySourceFilter
from memforge.runtime import (
    DefaultRuntimeProvider,
    RuntimeHealthComponent,
    RuntimeProvider,
    SyncAlreadyRunningError,
    SyncService,
    SourcePausedError,
)
from memforge.scheduler import SyncScheduler
from memforge.source_secrets import (
    decrypt_secret,
    SecretConfigurationError,
    prepare_source_config_for_storage,
    redact_source_config,
    source_secret_fields,
)
from memforge.storage.document_store import LocalDocumentStore
from memforge.server.memory_admin_service import (
    list_memory_admin_page,
    pick_origin_source_type,
)
from memforge.server.source_admin_service import (
    can_manage_source,
    list_source_admin_rows,
    normalize_workspace_role,
)
from memforge.storage.admin_memory import MemoryAdminListFilters
from memforge.storage.admin_source import (
    SOURCE_SYNC_SCHEDULE_DEFAULT_INTERVAL_MINUTES,
    SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES,
    SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES,
)
from memforge.storage.database import Database

logger = logging.getLogger(__name__)

SEARCH_CLIENTS = ("claude-code", "codex")

SOURCE_ACTIVE_STATUS = "active"
SOURCE_PAUSED_STATUS = "paused"
# Source lifecycle is deliberately closed for now. A paused source keeps its
# configuration and existing memories but cannot accept new sync work.
SOURCE_STATUSES = {SOURCE_ACTIVE_STATUS, SOURCE_PAUSED_STATUS}


def _workspace_default_scope(request: Request, *, include_private: bool):
    """Build the workspace-default AccessScope for an admin-API HTTP read.

    The caller's identity is server-derived (``resolve_principal(request)``);
    only the active status is allowed; private rows surface only when the
    caller opts in via ``include_private``. Per-id and list readers do not
    receive a request-time `scope_mode`, so the scope mode is fixed at
    ``project-first``: cross-project rows stay visible and the ranker
    handles the affinity weighting.
    """
    from memforge.storage.adapters.context import AccessScope

    return AccessScope(
        user_id=resolve_request_principal(request),
        include_private=include_private,
        allowed_statuses=("active",),
        active_project=None,
        scope_mode="project-first",
    )


def resolve_request_principal(request: Request) -> str:
    """Resolve the request principal through the app-scoped resolver."""
    resolver = getattr(request.app.state, "principal_resolver", None)
    if resolver is None:
        from memforge.server.principal import resolve_principal

        resolver = resolve_principal
    return resolver(request)


def resolve_request_workspace_role(request: Request) -> str:
    """Resolve the caller's workspace role through the app-scoped resolver."""
    resolver = getattr(request.app.state, "workspace_role_resolver", None)
    if resolver is None:
        from memforge.server.principal import resolve_workspace_role

        resolver = resolve_workspace_role
    return normalize_workspace_role(resolver(request))


def _source_management_forbidden() -> HTTPException:
    return HTTPException(
        status_code=403,
        detail={
            "error": "source_management_forbidden",
            "message": "Only the source creator or a workspace admin can manage this source.",
        },
    )


def _require_source_management(request: Request, source: dict[str, Any]) -> None:
    if not can_manage_source(
        source,
        viewer_id=resolve_request_principal(request),
        viewer_role=resolve_request_workspace_role(request),
    ):
        raise _source_management_forbidden()


async def _filter_visible_ids(db: Database, ids, scope) -> set[str]:
    """Return the subset of ``ids`` the caller may see under ``scope``."""
    if not hasattr(db, "db") and hasattr(db, "filter_visible_ids"):
        return await db.filter_visible_ids(list(ids), scope)

    from memforge.storage.adapters.sqlite.relational import SqliteRelationalStore

    return await SqliteRelationalStore(db).filter_visible_ids(list(ids), scope)


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


# -- Health & Stats --


class ComponentHealth(BaseModel):
    status: str
    detail: str | None = None
    payload: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    database: ComponentHealth
    vector_store: ComponentHealth
    index_consistency: ComponentHealth | None = None
    audit_failures: ComponentHealth | None = None
    genes: dict[str, ComponentHealth] = {}


def _runtime_component_to_response(component: RuntimeHealthComponent) -> ComponentHealth:
    return ComponentHealth(
        status=component.status,
        detail=component.detail,
        payload=component.payload,
    )


class MemoryStatEntry(BaseModel):
    key: str
    count: int


class StatsResponse(BaseModel):
    total_memories: int
    memories_by_type: list[MemoryStatEntry]
    memories_by_status: list[MemoryStatEntry]
    total_entities: int
    total_sources: int


# -- Memories --


class MemorySourceDetail(BaseModel):
    doc_id: str
    source_type: str
    excerpt: str | None = None
    support_kind: str = "extracted"
    doc_title: str | None = None
    source_url: str | None = None
    content_url: str | None = None
    pdf_url: str | None = None
    added_at: str | None = None
    source_updated_at: str | None = None


class MemoryResponse(BaseModel):
    id: str
    memory_type: str
    content: str
    content_hash: str
    visibility: str
    owner_user_id: str | None = None
    project_key: str | None = None
    tags: list[str] = []
    confidence: float
    corroboration_count: int
    contradiction_count: int
    valid_from: str | None = None
    valid_until: str | None = None
    superseded_by: str | None = None
    status: str
    retirement_reason: str | None = None
    retired_at: str | None = None
    superseded_at: str | None = None
    replacement_reason: str | None = None
    replacement_kind: str | None = None
    extraction_context: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    # Source type for the leading glyph: the memory's extraction origin, else its
    # first attached source. None only when a memory has no remaining provenance.
    origin_source_type: str | None = None
    # The originating client for agent-session memories (e.g. 'codex' or
    # 'claude-code'); None for memories extracted from non-agent-session sources.
    origin_client: str | None = None


class MemoryDetailResponse(MemoryResponse):
    entity_refs: list[str] = []
    sources: list[MemorySourceDetail] = []


class MemoryListResponse(BaseModel):
    data: list[MemoryResponse]
    total: int
    limit: int
    offset: int


class MemoryStatsResponse(BaseModel):
    by_type: list[MemoryStatEntry]
    by_status: list[MemoryStatEntry]
    total: int


class MemoryUpdateRequest(BaseModel):
    content: str | None = None
    confidence: float | None = None
    status: str | None = None  # active, superseded, retired, decayed, pending_review


class SourceFacetFilterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_types: list[str] | None = None
    clients: list[str] | None = None
    repo_identifiers: list[str] | None = None

    @field_validator("source_types")
    @classmethod
    def _validate_source_types(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        allowed = set(GENE_REGISTRY)
        invalid = [item for item in value if item not in allowed]
        if invalid:
            raise ValueError("source_types must exactly match registered source types: " + ", ".join(sorted(allowed)))
        return value

    @field_validator("clients")
    @classmethod
    def _validate_clients(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        allowed = set(SEARCH_CLIENTS)
        invalid = [item for item in value if item not in allowed]
        if invalid:
            raise ValueError("clients must exactly match supported clients: " + ", ".join(SEARCH_CLIENTS))
        return value

    def to_source_filter(self) -> MemorySourceFilter:
        return MemorySourceFilter(
            source_types=tuple(self.source_types or ()),
            clients=tuple(self.clients or ()),
            repo_identifiers=tuple(self.repo_identifiers or ()),
        )


class MemorySearchRequest(BaseModel):
    query: str
    memory_types: list[str] | None = None
    sources: list[str] | None = None
    time_range: dict[str, Any] | None = None
    entities: list[str] | None = None
    include_superseded: bool = False
    top_k: int = Field(default=10, ge=1, le=50)
    active_project: str | None = None
    active_repo_identifier: str | None = None
    scope_mode: Literal["project", "project-first", "workspace"] = "project-first"
    include_private: bool = False
    status: str | None = None  # post-rank row-status filter; mirrors GET /api/memories
    source_filter: SourceFacetFilterRequest | None = None
    # NO user_id field. The caller's identity is server-derived from
    # resolve_principal(request); a body-supplied user_id is never used as
    # access authority.

    @field_validator("memory_types")
    @classmethod
    def _validate_memory_types(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        allowed = {item.value for item in MemoryType}
        invalid = [item for item in value if item not in allowed]
        if invalid:
            raise ValueError("memory_types must exactly match supported memory types: " + ", ".join(sorted(allowed)))
        return value

    @field_validator("status")
    @classmethod
    def _validate_status(cls, value: str | None) -> str | None:
        if value is None:
            return value
        allowed = {item.value for item in MemoryStatus}
        if value not in allowed:
            raise ValueError("status must exactly match supported memory statuses: " + ", ".join(sorted(allowed)))
        return value

    @model_validator(mode="after")
    def _coerce_scope_without_active_project(self) -> "MemorySearchRequest":
        """Project-aware ranking requires an active project. When the caller
        asks for a project-aware mode without one, fall through to the flat
        workspace ranking so the contract stays "default just works"."""
        if self.scope_mode in ("project", "project-first") and self.active_project is None:
            logger.info("Search request omitted active_project; falling back to flat workspace ranking.")
            self.scope_mode = "workspace"
        return self


# -- Entities --


class EntityAliasResponse(BaseModel):
    alias: str
    alias_normalized: str
    source: str
    created_at: str | None = None


class EntityResponse(BaseModel):
    id: int
    canonical_name: str
    tags: list[str] = []
    display_name: str
    created_at: str | None = None


class EntityDetailResponse(EntityResponse):
    aliases: list[EntityAliasResponse] = []
    linked_memory_count: int = 0


class EntityListResponse(BaseModel):
    data: list[EntityResponse]
    total: int


class MergeEntitiesRequest(BaseModel):
    source_id: int
    target_id: int


class AddAliasRequest(BaseModel):
    alias: str


# -- Sources / Genes --


class GeneMetadataResponse(BaseModel):
    name: str
    display_name: str
    description: str
    default_sync_interval_minutes: int
    auth_method: str
    data_shape: str


class ConfigFieldResponse(BaseModel):
    key: str
    label: str
    field_type: str
    required: bool = True
    placeholder: str = ""
    help_text: str = ""
    group: str = "general"
    order: int = 0
    default: str = ""
    options: list[str] = []
    advanced: bool = False


class ConfigGroupResponse(BaseModel):
    key: str
    label: str
    order: int = 0


class GeneConfigSchemaResponse(BaseModel):
    groups: list[ConfigGroupResponse] = []
    fields: list[ConfigFieldResponse] = []


class DiscoveryPreviewRequest(BaseModel):
    config: dict[str, Any]
    limit: int = Field(default=50, ge=1, le=200)


class DiscoveryPreviewItemResponse(BaseModel):
    item_id: str
    title: str
    source_url: str
    last_modified: str | None = None


class DiscoveryPreviewResponse(BaseModel):
    source_type: str
    count: int
    truncated: bool
    items: list[DiscoveryPreviewItemResponse]


# -- Teams browse --


class TeamsAuthCheckResponse(BaseModel):
    authenticated: bool
    expires_in_minutes: int | None = None
    error: str | None = None


class TeamsChannelResponse(BaseModel):
    id: str
    displayName: str


class TeamsTeamResponse(BaseModel):
    id: str
    displayName: str
    channels: list[TeamsChannelResponse] = []


class TeamsChatResponse(BaseModel):
    id: str
    topic: str
    lastActivity: str | None = None


class TeamsBrowseResponse(BaseModel):
    favorites: list[TeamsChatResponse] = []
    teams: list[TeamsTeamResponse] = []
    group_chats: list[TeamsChatResponse] = []
    individual_chats: list[TeamsChatResponse] = []


class JiraSessionUploadRequest(BaseModel):
    base_url: str
    cookie_header: str
    browser: str | None = None
    confirm_principal_change: bool = False


class JiraSessionExpireRequest(BaseModel):
    base_url: str
    error: str = "client reported the session expired"


class JiraSessionStatusResponse(BaseModel):
    provider: str = "jira"
    origin: str
    status: str
    principal_id: str | None = None
    principal_name: str | None = None
    principal_email: str | None = None
    browser: str | None = None
    captured_at: str | None = None
    validated_at: str | None = None
    last_error: str | None = None
    principal_changed: bool = False
    sources_reset: list[str] = []


class SourceSyncScheduleResponse(BaseModel):
    enabled: bool = False
    interval_minutes: int = SOURCE_SYNC_SCHEDULE_DEFAULT_INTERVAL_MINUTES
    next_run_at: str | None = None
    updated_at: str | None = None


class SourceSyncScheduleRequest(BaseModel):
    enabled: bool = False
    interval_minutes: int = Field(
        default=SOURCE_SYNC_SCHEDULE_DEFAULT_INTERVAL_MINUTES,
        ge=SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES,
        le=SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES,
    )


class SourceResponse(BaseModel):
    id: str
    type: str
    name: str
    config: dict
    status: str
    last_sync: str | None = None
    doc_count: int = 0
    created_at: str | None = None
    project_binding: dict | None = None
    sync_schedule: SourceSyncScheduleResponse | None = None


class SourceProjectResponse(BaseModel):
    project: str
    document_count: int
    memory_count: int
    last_observed_at: str | None = None


class SourceProjectsResponse(BaseModel):
    source_id: str
    projects: list[SourceProjectResponse]


class ResolvedProjectResponse(BaseModel):
    """Where memories from a source actually landed after the resolver ran."""

    project_key: str
    memory_count: int


class ResolvedProjectsResponse(BaseModel):
    source_id: str
    projects: list[ResolvedProjectResponse]


# Wire/storage translation: `kind` ("normal" | "shared") rides over the wire,
# `is_shared` lives in the column. Translation happens in `_project_to_response`
# (storage to wire) and inline in the create/update handlers (wire to storage).
class ProjectCreateRequest(BaseModel):
    name: str
    key: str | None = None
    kind: Literal["normal", "shared"] = "normal"


class ProjectUpdateRequest(BaseModel):
    name: str | None = None
    kind: Literal["normal", "shared"] | None = None


class ProjectResponse(BaseModel):
    id: str
    key: str
    name: str
    kind: Literal["normal", "shared"]
    created_at: str | None = None


class ProjectDeleteResponse(BaseModel):
    id: str
    rebucketed_count: int
    rebucketed_memory_ids: list[str]


class CreateSourceRequest(BaseModel):
    type: str
    name: str
    config: dict
    project_binding: dict | None = None
    sync_schedule: SourceSyncScheduleRequest | None = None


class SourceSyncRequest(BaseModel):
    force_full_sync: bool = False


class SourceSubscriptionRequest(BaseModel):
    enabled: bool


class UpdateSourceRequest(BaseModel):
    name: str | None = None
    config: dict | None = None
    status: str | None = None
    project_binding: dict | None = None
    sync_schedule: SourceSyncScheduleRequest | None = None


def _request_includes_field(model: BaseModel, field_name: str) -> bool:
    fields_set = getattr(model, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(model, "__fields_set__", set())
    return field_name in fields_set


def _validate_source_project_binding(binding: dict | None) -> None:
    if binding is None:
        return
    mode = binding.get("mode")
    if mode == "fixed":
        project_key = str(binding.get("project_key") or "").strip()
        if not project_key:
            raise ValueError("fixed project binding requires project_key")
        return
    if mode == "by_field":
        field = str(binding.get("field") or "").strip()
        default = str(binding.get("default") or "").strip()
        if not field or not default:
            raise ValueError("field project binding requires field and default")
        return
    raise ValueError("project binding mode must be fixed or by_field")


def _validate_source_status(status: str | None) -> None:
    if status is None:
        return
    if status not in SOURCE_STATUSES:
        raise HTTPException(
            status_code=400,
            detail=(f"Invalid source status: {status}. Expected one of {', '.join(sorted(SOURCE_STATUSES))}"),
        )


def _source_paused_http_error() -> HTTPException:
    return HTTPException(status_code=400, detail="Source is paused")


async def _raise_if_source_paused(db: Database, source_id: str) -> None:
    source = await db.get_source(source_id)
    if source and source.get("status") == SOURCE_PAUSED_STATUS:
        raise _source_paused_http_error()


class AgentSessionDocumentRequest(BaseModel):
    client: str
    session_id: str
    trigger: str
    workspace: str
    document_markdown: str
    repo: str | None = None
    branch: str | None = None
    commit_sha: str | None = None
    history_window_kind: str = "session"
    history_window_start: str | None = None
    history_window_end: str | None = None
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    submitted_at: str | None = None
    source_updated_at: str | None = None
    process_now: bool = True


class LocalAdapterDocumentRequest(BaseModel):
    """One file pushed by the local CLI adapter into a configured source.

    ``markdown_body`` carries the raw file text for the declared ``content_type``
    (Markdown, plain text, JSON, or HTML). The service converts it to markdown
    during sync; the CLI does no parsing.
    """

    vault_id: str
    relative_path: str
    markdown_body: str
    content_type: str = "text/markdown"
    title: str | None = None
    raw_hash: str | None = None
    submitted_by: str | None = None
    submitted_at: str | None = None
    process_now: bool = True


class AgentSessionWindowRequest(BaseModel):
    schema_version: str = "agent-session-window/v1"
    plugin_version: str | None = None
    client: str
    session_id: str
    trigger: str
    workspace: str
    events: list[dict[str, Any]] = Field(default_factory=list)
    history_window: dict[str, Any] = Field(default_factory=dict)
    transcript_markdown: str | None = None
    repo: str | None = None
    branch: str | None = None
    commit_sha: str | None = None
    receipt: dict[str, Any] = Field(default_factory=dict)
    retention: str = "none"
    submitted_at: str | None = None
    source_updated_at: str | None = None
    process_now: bool = False


class AgentHookReceiptRequest(BaseModel):
    client: str
    session_id: str
    hook: str
    workspace: str
    repo: str | None = None
    branch: str | None = None
    commit_sha: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    submitted_at: str | None = None


class AgentHookContextRequest(BaseModel):
    client: str
    hook: str
    workspace: str
    repo: str | None = None
    branch: str | None = None
    prompt: str | None = None
    touched_files: list[str] = Field(default_factory=list)
    max_memories: int = 5
    include_recent_changes: bool = True


# -- Schedule --


class ScheduleConfigResponse(BaseModel):
    enabled: bool = False
    frequency: str = "daily"
    time: str = "02:00"
    day_of_week: int = 0
    timezone: str = "UTC"


class ScheduleConfigRequest(BaseModel):
    enabled: bool = False
    frequency: str = "daily"
    time: str = "02:00"
    day_of_week: int = 0
    timezone: str = "UTC"


# -- LLM Config --


class LlmConfigResponse(BaseModel):
    enrichment_model: str | None = None
    enrichment_base_url: str | None = None
    enrichment_api_key: str | None = None
    enrichment_api_key_set: bool = False
    enrichment_api_key_last4: str | None = None
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None
    embedding_api_key_set: bool = False
    embedding_api_key_last4: str | None = None


class LlmConfigRequest(BaseModel):
    enrichment_model: str | None = None
    enrichment_base_url: str | None = None
    enrichment_api_key: str | None = None
    embedding_model: str | None = None
    embedding_base_url: str | None = None
    embedding_api_key: str | None = None


class LlmModelOption(BaseModel):
    id: str
    label: str | None = None


class LlmConfigProbeRequest(BaseModel):
    kind: Literal["enrichment", "embedding"]
    base_url: str
    api_key: str | None = None


class LlmConfigProbeResponse(BaseModel):
    ok: bool
    models_supported: bool = False
    models: list[LlmModelOption] = Field(default_factory=list)
    stage: Literal["validation", "connect", "tls", "timeout", "auth", "http"] | None = None
    status: int | None = None
    message: str
    latency_ms: int | None = None
    suggested_base_url: str | None = None


# -- Memory reviews --


class MemoryReviewMemorySummary(BaseModel):
    id: str
    memory_type: str
    content: str
    confidence: float
    corroboration_count: int
    status: str
    tags: list[str] = []
    entity_refs: list[str] = []
    sources: list[MemorySourceDetail] = []
    created_at: str | None = None
    updated_at: str | None = None


class MemoryReviewResponse(BaseModel):
    id: str
    kind: str
    status: str
    incumbent_memory_id: str
    challenger_memory_id: str
    reason: str | None = None
    review_note: str | None = None
    reviewer: str | None = None
    expected_incumbent_updated_at: str | None = None
    expected_challenger_updated_at: str | None = None
    created_at: str | None = None
    resolved_at: str | None = None
    is_stale: bool = False


class MemoryReviewListResponse(BaseModel):
    data: list[MemoryReviewResponse]
    total: int


class MemoryReviewDetailResponse(MemoryReviewResponse):
    incumbent: MemoryReviewMemorySummary | None = None
    challenger: MemoryReviewMemorySummary | None = None
    related_challengers: list[MemoryReviewMemorySummary] = Field(default_factory=list)


class MemoryReviewDecisionRequest(BaseModel):
    note: str | None = None
    reviewer: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_api_key(key: str | None) -> str | None:
    """Mask an API key, showing only the last 4 characters."""
    if not key:
        return None
    if len(key) <= 4:
        return "****"
    return "*" * (len(key) - 4) + key[-4:]


def _api_key_last4(key: str | None) -> str | None:
    if not key:
        return None
    return key[-4:]


def _is_running_in_container() -> bool:
    return Path("/.dockerenv").exists()


def _suggest_host_base_url(base_url: str) -> str | None:
    parsed = urlsplit(base_url)
    if parsed.hostname not in {"localhost", "127.0.0.1"}:
        return None
    if not _is_running_in_container():
        return None
    netloc = "host.docker.internal"
    if parsed.port:
        netloc = f"{netloc}:{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path, parsed.query, parsed.fragment))


def _model_list_urls(base_url: str) -> list[str]:
    root = base_url.strip().rstrip("/")
    urls = [f"{root}/models"]
    parsed = urlsplit(root)
    if not parsed.path.rstrip("/").endswith("/v1"):
        urls.append(f"{root}/v1/models")
    return list(dict.fromkeys(urls))


def _model_probe_headers(api_key: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    return headers


def _extract_model_options(payload: Any) -> list[LlmModelOption]:
    if isinstance(payload, dict):
        raw_items = payload.get("data")
        if raw_items is None:
            raw_items = payload.get("models")
    else:
        raw_items = payload

    if not isinstance(raw_items, list):
        return []

    seen: set[str] = set()
    models: list[LlmModelOption] = []
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
        models.append(LlmModelOption(id=model_id, label=label))
    return models


def _probe_error_response(
    *,
    base_url: str,
    stage: Literal["validation", "connect", "tls", "timeout", "auth", "http"],
    message: str,
    status: int | None = None,
    latency_ms: int | None = None,
) -> LlmConfigProbeResponse:
    return LlmConfigProbeResponse(
        ok=False,
        stage=stage,
        status=status,
        message=message,
        latency_ms=latency_ms,
        suggested_base_url=_suggest_host_base_url(base_url),
    )


async def _probe_llm_models(
    *,
    base_url: str,
    api_key: str | None,
) -> LlmConfigProbeResponse:
    value = base_url.strip()
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return _probe_error_response(
            base_url=value,
            stage="validation",
            message="Enter a URL starting with http:// or https://.",
        )

    headers = _model_probe_headers(api_key)
    unsupported: tuple[int, int] | None = None
    non_json_latency_ms: int | None = None

    async with httpx.AsyncClient(timeout=5.0, follow_redirects=True) as client:
        for url in _model_list_urls(value):
            started = time.perf_counter()
            try:
                resp = await client.get(url, headers=headers)
            except httpx.TimeoutException:
                return _probe_error_response(
                    base_url=value,
                    stage="timeout",
                    message="No response before the timeout.",
                )
            except httpx.ConnectError as exc:
                text = str(exc).lower()
                stage: Literal["connect", "tls"] = "tls" if "ssl" in text or "tls" in text else "connect"
                return _probe_error_response(
                    base_url=value,
                    stage=stage,
                    message=f"Could not reach {parsed.hostname or 'the endpoint'}.",
                )
            except httpx.HTTPError:
                return _probe_error_response(
                    base_url=value,
                    stage="connect",
                    message=f"Could not reach {parsed.hostname or 'the endpoint'}.",
                )

            latency_ms = int((time.perf_counter() - started) * 1000)
            if resp.status_code in {404, 405, 501}:
                unsupported = (resp.status_code, latency_ms)
                continue
            if resp.status_code in {401, 403}:
                return _probe_error_response(
                    base_url=value,
                    stage="auth",
                    status=resp.status_code,
                    latency_ms=latency_ms,
                    message=(
                        "Add an API key, then test again."
                        if not api_key
                        else f"Endpoint rejected the API key (HTTP {resp.status_code})."
                    ),
                )
            if resp.status_code >= 400:
                return _probe_error_response(
                    base_url=value,
                    stage="http",
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
            return LlmConfigProbeResponse(
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
        return LlmConfigProbeResponse(
            ok=True,
            models_supported=False,
            message="Connected, but this endpoint did not return a model list.",
            latency_ms=non_json_latency_ms,
        )

    status, latency_ms = unsupported or (404, None)
    return LlmConfigProbeResponse(
        ok=True,
        models_supported=False,
        status=status,
        message="Connected, but this endpoint does not expose a model list.",
        latency_ms=latency_ms,
    )


def _source_secret_fields(source_type: str) -> tuple[str, ...]:
    fields = set(source_secret_fields(source_type, GENE_REGISTRY))
    if source_type == "jira":
        fields.add("jira_cookie")
    return tuple(sorted(fields))


def _sync_scope_config(source_type: str, config: dict[str, Any]) -> dict[str, Any]:
    """Return config fields that affect the document set discovered by sync."""
    jira_auth_mode = _jira_auth_mode(config) if source_type == "jira" else None
    scope = dict(config)
    gene_cls = GENE_REGISTRY.get(source_type)
    if gene_cls:
        normalizer = getattr(gene_cls, "normalize_config", None)
        if callable(normalizer):
            normalizer(scope)
    for field in _source_secret_fields(source_type):
        scope.pop(field, None)
        scope.pop(f"{field}_encrypted", None)
        scope.pop(f"{field}_configured", None)
        scope.pop(f"{field}_hint", None)
        scope.pop(f"{field}_decrypt_failed", None)
    scope.pop("tls_ca_bundle", None)
    scope.pop("request_interval_ms", None)
    if source_type == "jira":
        scope["auth_mode"] = jira_auth_mode
    if source_type == "confluence":
        scope = _canonical_confluence_scope(scope)
    return scope


def _canonical_confluence_scope(scope: dict[str, Any]) -> dict[str, Any]:
    mode = str(scope.get("sync_mode") or "").strip().lower()
    mode = (
        mode
        if mode in {"page_tree", "space"}
        else ("page_tree" if str(scope.get("page_tree_root") or "").strip() else "space")
    )
    exclude_labels = _config_list_value(scope.get("exclude_labels"))
    canonical: dict[str, Any] = {"sync_mode": mode}
    api_prefix = str(scope.get("api_prefix") or "").strip()
    if api_prefix:
        canonical["api_prefix"] = api_prefix.rstrip("/")
    if exclude_labels:
        canonical["exclude_labels"] = exclude_labels
    if mode == "page_tree":
        canonical["page_tree_root"] = str(scope.get("page_tree_root") or "").strip()
        canonical["include_children"] = _config_bool(scope.get("include_children"), default=True)
    else:
        canonical["spaces"] = _config_list_value(scope.get("spaces"))
    return canonical


def _config_list_value(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return []


def _config_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "1", "yes", "on"}:
            return True
        if text in {"false", "0", "no", "off"}:
            return False
    return default


def _validate_source_config(
    source_type: str,
    config: dict[str, Any],
    existing_config: dict[str, Any] | None = None,
) -> None:
    if source_type == "github_pages":
        _validate_github_pages_config(config, existing_config=existing_config)
    validate_tls_ca_bundle(config)
    secret_fields = _source_secret_fields(source_type)
    gene_cls = GENE_REGISTRY.get(source_type)
    if gene_cls:
        normalizer = getattr(gene_cls, "normalize_config", None)
        if callable(normalizer):
            normalizer(config)
    has_secret_contract = bool(secret_fields) if gene_cls else _config_contains_secret(config, secret_fields)
    if has_secret_contract:
        product_name = gene_cls.metadata().display_name if gene_cls else source_type
        base_url = str(config.get("base_url") or "").strip()
        if not base_url:
            raise ValueError(f"{product_name} base_url is required before storing source secrets")
        require_https_base_url(base_url, product_name)

    if gene_cls:
        product_name = gene_cls.metadata().display_name
        _validate_required_source_fields(
            product_name,
            gene_cls.config_schema().fields,
            config,
            existing_config=existing_config,
        )
    if source_type == "confluence":
        _validate_confluence_config(config, existing_config=existing_config)
    if source_type == "jira":
        _validate_jira_auth_config(config, existing_config=existing_config)
        _validate_jira_scope_config(config)


def _validate_confluence_config(
    config: dict[str, Any],
    existing_config: dict[str, Any] | None = None,
) -> None:
    existing_config = existing_config or {}
    sync_mode = str(config.get("sync_mode") or existing_config.get("sync_mode") or "").strip().lower()
    if sync_mode not in {"", "page_tree", "space"}:
        raise ValueError("Confluence Sync Scope must be Page tree or Space")
    if not sync_mode:
        sync_mode = (
            "page_tree"
            if str(config.get("page_tree_root") or existing_config.get("page_tree_root") or "").strip()
            else "space"
        )

    if sync_mode == "page_tree":
        page_tree_root = str(config.get("page_tree_root") or existing_config.get("page_tree_root") or "").strip()
        if not page_tree_root:
            raise ValueError("Confluence Page Tree Root is required when syncing a page tree")
        return

    if not _config_list_has_value(config.get("spaces")) and not _config_list_has_value(existing_config.get("spaces")):
        raise ValueError("Confluence Spaces to Sync is required when syncing whole spaces")


def _config_list_has_value(value: Any) -> bool:
    if isinstance(value, list):
        return any(str(item).strip() for item in value)
    if isinstance(value, str):
        return any(part.strip() for part in value.split(","))
    return False


def _validate_github_pages_config(
    config: dict[str, Any],
    existing_config: dict[str, Any] | None = None,
) -> None:
    existing_config = existing_config or {}
    base_url = _ensure_github_pages_base_url(config, existing_config=existing_config)
    require_https_base_url(base_url, "GitHub Pages")

    auth_mode = str(config.get("auth_mode") or existing_config.get("auth_mode") or "github_pat").strip().lower()
    if auth_mode not in {"github_pat", "none"}:
        raise ValueError("GitHub Pages Authentication Method must be Personal access token or No authentication")
    if auth_mode == "github_pat" and not _source_secret_has_value("pat", config, existing_config):
        raise ValueError("GitHub Pages Personal Access Token is required for github_pat authentication")

    sync_mode = str(config.get("sync_mode") or "single_page").strip().lower()
    if sync_mode == "single_page":
        page_url = str(config.get("page_url") or "").strip()
        if not page_url:
            raise ValueError("GitHub Pages Page URL is required for single page sync")
        _validate_github_pages_scope_url(base_url, page_url, "Page URL")
        return
    if sync_mode == "subtree":
        root_url = str(config.get("root_url") or "").strip()
        if not root_url:
            raise ValueError("GitHub Pages Subtree Root URL is required for subtree sync")
        _validate_github_pages_scope_url(base_url, root_url, "Subtree Root URL")
        return
    if sync_mode == "explicit_list":
        pages = config.get("pages")
        page_urls = [
            str(page).strip()
            for page in (pages if isinstance(pages, list) else str(pages or "").split(","))
            if str(page).strip()
        ]
        if page_urls:
            for page_url in page_urls:
                _validate_github_pages_scope_url(base_url, page_url, "Explicit Page URL")
            return
        raise ValueError("GitHub Pages Explicit Page URLs are required for explicit list sync")
    raise ValueError("GitHub Pages Sync Mode must be Single page, Subtree, or Explicit list")


def _ensure_github_pages_base_url(
    config: dict[str, Any],
    existing_config: dict[str, Any] | None = None,
) -> str:
    existing_config = existing_config or {}
    base_url = str(config.get("base_url") or existing_config.get("base_url") or "").strip()
    if not base_url:
        scope_url = _github_pages_scope_url(config, existing_config)
        base_url = _github_pages_site_root_from_url(scope_url)
    if base_url:
        config["base_url"] = base_url
    return base_url


def _github_pages_scope_url(config: dict[str, Any], existing_config: dict[str, Any]) -> str:
    sync_mode = str(config.get("sync_mode") or existing_config.get("sync_mode") or "single_page").strip().lower()
    if sync_mode == "subtree":
        return str(config.get("root_url") or existing_config.get("root_url") or "").strip()
    if sync_mode == "explicit_list":
        pages = config.get("pages") or existing_config.get("pages")
        page_urls = [
            str(page).strip()
            for page in (pages if isinstance(pages, list) else str(pages or "").split(","))
            if str(page).strip()
        ]
        return page_urls[0] if page_urls else ""
    return str(config.get("page_url") or existing_config.get("page_url") or "").strip()


def _github_pages_site_root_from_url(url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    value = str(url or "").strip()
    if not value:
        return ""
    parts = urlsplit(value)
    path_parts = [part for part in parts.path.split("/") if part]
    if len(path_parts) >= 3 and path_parts[0] == "pages":
        root_path = "/" + "/".join(path_parts[:3])
        return urlunsplit((parts.scheme, parts.netloc, root_path, "", ""))
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def _validate_github_pages_scope_url(base_url: str, candidate_url: str, label: str) -> None:
    require_https_base_url(candidate_url, f"GitHub Pages {label}")
    from urllib.parse import urlsplit

    base = urlsplit(base_url)
    candidate = urlsplit(candidate_url)
    if candidate.netloc.lower() != base.netloc.lower():
        raise ValueError(f"GitHub Pages {label} must stay on the configured site origin")
    base_path = base.path.rstrip("/") + "/"
    candidate_path = candidate.path.rstrip("/") + "/"
    if not candidate_path.startswith(base_path):
        raise ValueError(f"GitHub Pages {label} must stay under the configured site path")


def _validate_jira_scope_config(config: dict[str, Any]) -> None:
    """Require the field that actually drives discovery for the chosen query mode.

    ``projects`` is optional at the schema level so advanced JQL queries (which
    embed their own project clause) do not have to duplicate it.
    """
    mode = str(config.get("query_mode") or "simple").strip().lower()
    if mode == "advanced":
        if not str(config.get("jql") or "").strip():
            raise ValueError("Jira JQL is required in advanced query mode")
        return
    projects = config.get("projects")
    has_projects = (isinstance(projects, list) and any(str(p).strip() for p in projects)) or (
        isinstance(projects, str) and any(p.strip() for p in projects.split(","))
    )
    if not has_projects:
        raise ValueError("Jira Projects to Sync is required in simple query mode")


def _validate_jira_auth_config(
    config: dict[str, Any],
    existing_config: dict[str, Any] | None = None,
) -> None:
    existing_config = existing_config or {}
    _reject_source_owned_jira_cookie(config)
    auth_mode = _jira_auth_mode(config, existing_config)
    if auth_mode not in {"browser_cookie", "pat"}:
        raise ValueError("Jira Authentication Method must be Browser session or Personal access token")

    if auth_mode == "browser_cookie":
        return

    if _source_secret_has_value("pat", config, existing_config):
        return
    raise ValueError("Jira Personal Access Token is required for pat authentication")


def _reject_source_owned_jira_cookie(config: dict[str, Any]) -> None:
    forbidden = {"jira_cookie", "jira_cookie_encrypted"}
    if any(key in config for key in forbidden):
        raise ValueError("Jira browser sessions are managed by shared auth sessions, not source config")


def _jira_auth_mode(
    config: dict[str, Any],
    existing_config: dict[str, Any] | None = None,
) -> str:
    existing_config = existing_config or {}
    configured_auth_mode = config.get("auth_mode") or existing_config.get("auth_mode")
    if configured_auth_mode is None and _source_secret_has_value("pat", config, existing_config):
        return "pat"
    return str(configured_auth_mode or "browser_cookie").strip().lower()


def _source_secret_has_value(
    field: str,
    config: dict[str, Any],
    existing_config: dict[str, Any],
) -> bool:
    value = config.get(field)
    if isinstance(value, str) and value.strip():
        return True
    return bool(
        config.get(f"{field}_encrypted")
        or existing_config.get(f"{field}_encrypted")
        or (isinstance(existing_config.get(field), str) and existing_config[field].strip())
    )


def _jira_auth_secret_changed(
    source_type: str,
    config: dict[str, Any],
    existing_config: dict[str, Any],
) -> bool:
    if source_type != "jira":
        return False
    auth_mode = _jira_auth_mode(config, existing_config)
    if auth_mode == "browser_cookie":
        return False
    field = "pat"
    incoming = config.get(field)
    if not isinstance(incoming, str) or not incoming.strip():
        return False
    existing_secret = _stored_secret_value(field, existing_config)
    return existing_secret != incoming.strip()


def _stored_secret_value(field: str, config: dict[str, Any]) -> str | None:
    encrypted = config.get(f"{field}_encrypted")
    if encrypted:
        try:
            return decrypt_secret(str(encrypted))
        except SecretConfigurationError:
            return None
    plaintext = config.get(field)
    if isinstance(plaintext, str) and plaintext.strip():
        return plaintext.strip()
    return None


def _drop_source_owned_jira_cookie(config: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(config)
    for key in (
        "jira_cookie",
        "jira_cookie_encrypted",
        "jira_cookie_configured",
        "jira_cookie_hint",
        "jira_cookie_decrypt_failed",
    ):
        cleaned.pop(key, None)
    if _jira_auth_mode(cleaned) == "browser_cookie":
        for key in (
            "pat",
            "pat_encrypted",
            "pat_configured",
            "pat_hint",
            "pat_decrypt_failed",
        ):
            cleaned.pop(key, None)
    return cleaned


def _populate_local_markdown_inbox(
    config: dict[str, Any],
    source_id: str,
    app_config: AppConfig,
) -> dict[str, Any]:
    """Fill the per-source inbox path so the gene can read pushed packages."""
    from memforge.local_adapter import default_local_adapter_inbox

    cleaned = dict(config)
    inbox = default_local_adapter_inbox(app_config, source_id)
    inbox.mkdir(parents=True, exist_ok=True)
    cleaned["documents_dir"] = str(inbox)
    return cleaned


async def _cancel_running_jira_browser_sources_for_origin(
    *,
    db: Database,
    sync_service: SyncService,
    base_url: str,
) -> None:
    origin = canonical_jira_origin(base_url)
    for source in await db.list_sources():
        if source.get("type") != "jira":
            continue
        config = source.get("config", {})
        if effective_jira_auth_mode(config) != "browser_cookie":
            continue
        try:
            if canonical_jira_origin(str(config.get("base_url") or "")) != origin:
                continue
        except ValueError:
            continue
        if sync_service.is_running(source["id"]):
            await sync_service.cancel_source(source["id"])


def _plaintext_session_upload_allowed() -> bool:
    """Whether the deployment trusts its network enough to accept a session cookie over plaintext.

    Behind a container the source IP is not a reliable "local" signal (Docker can
    present a private gateway or the host's own public address), so a trusted local
    or dev deployment opts in explicitly instead of relying on the client IP.
    """
    return os.getenv("MEMFORGE_ALLOW_PLAINTEXT_SESSION_UPLOAD", "").strip().lower() in {"1", "true", "yes", "on"}


def _require_secure_or_loopback(request: Request) -> None:
    """A Jira session cookie is a live credential: accept it over HTTPS, from loopback,
    or when the deployment explicitly trusts its network. Reject plaintext otherwise."""
    # x-forwarded-proto is meaningful only when a trusted TLS-terminating proxy sets it;
    # a direct plaintext client falls back to request.url.scheme ("http").
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    if proto == "https":
        return
    host = request.client.host if request.client else ""
    if host in {"127.0.0.1", "::1", "localhost", "testclient"}:
        return
    if _plaintext_session_upload_allowed():
        return
    raise HTTPException(
        status_code=400,
        detail=(
            "Refusing a Jira session cookie over plaintext from a non-local client. Use HTTPS, "
            "or set MEMFORGE_ALLOW_PLAINTEXT_SESSION_UPLOAD=1 for a trusted local or dev deployment "
            "(for example a server reached through Docker's localhost port mapping)."
        ),
    )


def _validate_required_source_fields(
    product_name: str,
    fields: list[ConfigField],
    config: dict[str, Any],
    existing_config: dict[str, Any] | None = None,
) -> None:
    existing_config = existing_config or {}
    for field in fields:
        if not field.required:
            continue
        if _source_field_has_value(field, config, existing_config):
            continue
        raise ValueError(f"{product_name} {field.label} is required")


def _source_field_has_value(
    field: ConfigField,
    config: dict[str, Any],
    existing_config: dict[str, Any],
) -> bool:
    value = config.get(field.key)
    if field.field_type == ConfigFieldType.SECRET:
        if isinstance(value, str) and value.strip():
            return True
        return bool(
            existing_config.get(f"{field.key}_encrypted")
            or (isinstance(existing_config.get(field.key), str) and existing_config[field.key].strip())
        )
    if field.field_type in {ConfigFieldType.TAG_LIST, ConfigFieldType.MULTI_SELECT}:
        if isinstance(value, list):
            return any(str(item).strip() for item in value)
        if isinstance(value, str):
            return any(item.strip() for item in value.split(","))
        return bool(field.default)
    if field.field_type == ConfigFieldType.BOOLEAN:
        return field.key in config or field.default != ""
    return str(value if value is not None else field.default).strip() != ""


def _config_contains_secret(config: dict[str, Any], secret_fields: tuple[str, ...]) -> bool:
    for field in secret_fields:
        if config.get(field) or config.get(f"{field}_encrypted") or config.get(f"{field}_configured"):
            return True
    return False


def _dt_iso(dt: datetime | None) -> str | None:
    """Convert a datetime to ISO string, or None."""
    if dt is None:
        return None
    value = dt
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _json_ready(value: Any) -> Any:
    """Convert dataclasses and datetimes into JSON-native values."""
    if is_dataclass(value) and not isinstance(value, type):
        return _json_ready(asdict(value))
    if isinstance(value, dict):
        return {k: _json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_ready(v) for v in value]
    if isinstance(value, datetime):
        return _dt_iso(value)
    return value


def _memory_source_detail(
    ms: Any,
    doc: Any | None,
    config: AppConfig | None = None,
    artifact_store: DocumentArtifactStore | None = None,
) -> MemorySourceDetail:
    content_url = None
    pdf_url = None
    if doc is not None and config is not None:
        content_url = document_content_url_for_store(doc, config, artifact_store)
        pdf_url = document_pdf_url_for_store(doc, config, artifact_store)
    elif doc is not None:
        content_url = document_content_url(doc, config)
        pdf_url = document_pdf_url(doc, config)
    return MemorySourceDetail(
        doc_id=ms.doc_id,
        source_type=ms.source_type,
        excerpt=ms.excerpt,
        support_kind=ms.support_kind,
        doc_title=doc.title if doc else None,
        source_url=doc.source_url if doc else None,
        content_url=content_url,
        pdf_url=pdf_url,
        added_at=_dt_iso(ms.added_at),
        source_updated_at=_dt_iso(ms.source_updated_at),
    )


def _pick_origin_source_type(pairs: list[tuple[str, str | None, str | None]]) -> tuple[str | None, str | None]:
    """Pick a memory's display source and client from its (source_type, support_kind, client) triples,
    ordered oldest-first: the extraction origin if any, else the first source.
    Returns (source_type, client). Both are None when there are no sources."""
    return pick_origin_source_type(pairs)


async def _origin_source_types(db: Database, memory_ids: list[str]) -> dict[str, tuple[str, str | None]]:
    """Map memory id -> (origin_source_type, origin_client) for a batch of memories in one query.
    Memories with no source are omitted."""
    grouped = await db.get_origin_source_pairs(memory_ids)
    origins: dict[str, tuple[str, str | None]] = {}
    for mid, pairs in grouped.items():
        source_type, client = _pick_origin_source_type(pairs)
        if source_type is not None:
            origins[mid] = (source_type, client)
    return origins


def _memory_to_response(
    mem: Memory,
    origin_source_type: str | None = None,
    origin_client: str | None = None,
) -> MemoryResponse:
    """Convert a Memory dataclass to a Pydantic response model."""
    return MemoryResponse(
        id=mem.id,
        memory_type=mem.memory_type,
        content=mem.content,
        content_hash=mem.content_hash,
        visibility=mem.visibility,
        owner_user_id=mem.owner_user_id,
        project_key=mem.project_key,
        tags=mem.tags,
        confidence=mem.confidence,
        corroboration_count=mem.corroboration_count,
        contradiction_count=mem.contradiction_count,
        valid_from=_dt_iso(mem.valid_from),
        valid_until=_dt_iso(mem.valid_until),
        superseded_by=mem.superseded_by,
        status=mem.status,
        retirement_reason=mem.retirement_reason,
        retired_at=_dt_iso(mem.retired_at),
        superseded_at=_dt_iso(mem.superseded_at),
        replacement_reason=mem.replacement_reason,
        replacement_kind=mem.replacement_kind,
        extraction_context=mem.extraction_context,
        created_at=_dt_iso(mem.created_at),
        updated_at=_dt_iso(mem.updated_at),
        origin_source_type=origin_source_type,
        origin_client=origin_client,
    )


def _is_review_stale(review: MemoryReview, incumbent: Memory | None, challenger: Memory | None) -> bool:
    """Detect drift between the review's pinned timestamps and current memories."""
    if review.status != "pending":
        return False
    if incumbent is None or challenger is None:
        return True
    actual_incumbent = incumbent.updated_at.isoformat() if incumbent.updated_at else None
    actual_challenger = challenger.updated_at.isoformat() if challenger.updated_at else None
    if review.expected_incumbent_updated_at is not None and review.expected_incumbent_updated_at != actual_incumbent:
        return True
    if review.expected_challenger_updated_at is not None and review.expected_challenger_updated_at != actual_challenger:
        return True
    return False


def _review_to_response(
    review: MemoryReview,
    *,
    incumbent: Memory | None = None,
    challenger: Memory | None = None,
) -> MemoryReviewResponse:
    return MemoryReviewResponse(
        id=review.id,
        kind=review.kind,
        status=review.status,
        incumbent_memory_id=review.incumbent_memory_id,
        challenger_memory_id=review.challenger_memory_id,
        reason=review.reason,
        review_note=review.review_note,
        reviewer=review.reviewer,
        expected_incumbent_updated_at=review.expected_incumbent_updated_at,
        expected_challenger_updated_at=review.expected_challenger_updated_at,
        created_at=_dt_iso(review.created_at),
        resolved_at=_dt_iso(review.resolved_at),
        is_stale=_is_review_stale(review, incumbent, challenger),
    )


async def _build_memory_summary(
    db: Database,
    memory: Memory,
    config: AppConfig | None = None,
    artifact_store: DocumentArtifactStore | None = None,
) -> MemoryReviewMemorySummary:
    """Hydrate a memory with provenance and entity refs for the review detail view."""
    raw_sources = await db.get_memory_sources(memory.id)
    sources: list[MemorySourceDetail] = []
    for ms in raw_sources:
        doc = await db.get_document(ms.doc_id)
        sources.append(_memory_source_detail(ms, doc, config, artifact_store))

    entity_names = await db.get_memory_entity_names(memory.id)

    return MemoryReviewMemorySummary(
        id=memory.id,
        memory_type=memory.memory_type,
        content=memory.content,
        confidence=memory.confidence,
        corroboration_count=memory.corroboration_count,
        status=memory.status,
        tags=memory.tags,
        entity_refs=entity_names,
        sources=sources,
        created_at=_dt_iso(memory.created_at),
        updated_at=_dt_iso(memory.updated_at),
    )


async def _build_memory_store(
    db: Database,
    config: AppConfig,
    runtime_provider: RuntimeProvider | None = None,
) -> MemoryStore:
    """Build a request-scoped memory store with effective embedding settings."""

    from memforge.memory.audit import AuditContext, MemoryAuditLogger
    from memforge.retrieval.document_index import DocumentVectorIndex
    from memforge.retrieval.embeddings import get_chroma_collection
    from memforge.runtime import get_effective_llm_config

    llm = await get_effective_llm_config(db, config)
    memory_collection = get_chroma_collection(
        chroma_path=config.storage.chroma_path,
        name="memories",
    )
    document_collection = get_chroma_collection(
        chroma_path=config.storage.chroma_path,
        name="documents",
    )
    embed_cfg = {
        "base_url": llm.embedding_base_url,
        "api_key": llm.embedding_api_key,
        "model": llm.embedding_model,
    }
    provider = runtime_provider or DefaultRuntimeProvider()
    adapters = provider.build_adapters(
        db,
        memory_collection,
        audit_logger=MemoryAuditLogger(db, default_context=AuditContext(actor_type="admin")),
    )
    return MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embed_cfg=embed_cfg,
        audit_logger=MemoryAuditLogger(db, default_context=AuditContext(actor_type="admin")),
        document_index=DocumentVectorIndex(document_collection),
    )


async def _build_project_adapters(
    db: Database,
    config: AppConfig,
    runtime_provider: RuntimeProvider | None = None,
):
    """Build the storage adapters for project CRUD requests.

    The relational handle owns project rows; the vector handle is bound to
    the same memories collection that the memory store rebuckets so the
    delete handler can update both sides in lockstep.
    """
    from memforge.memory.audit import AuditContext, MemoryAuditLogger
    from memforge.retrieval.embeddings import get_chroma_collection

    memory_collection = get_chroma_collection(
        chroma_path=config.storage.chroma_path,
        name="memories",
    )
    provider = runtime_provider or DefaultRuntimeProvider()
    return provider.build_adapters(
        db,
        memory_collection,
        audit_logger=MemoryAuditLogger(db, default_context=AuditContext(actor_type="admin")),
    )


# Allowed character class for derived project keys: ASCII letters and digits
# only, joined by single underscores. Anything else collapses to one
# underscore. The cap matches the size of typical workspace tags so derived
# keys remain human-readable.
_PROJECT_KEY_ALLOWED_PATTERN = r"[^A-Za-z0-9]+"
_PROJECT_KEY_FALLBACK = "PROJECT"
_PROJECT_KEY_MAX_LENGTH = 32


def _derive_project_key(name: str) -> str:
    """Derive a deterministic key from a project name.

    Uppercase A-Z, 0-9, single underscores, capped at
    `_PROJECT_KEY_MAX_LENGTH`. A name that derives to a reserved key
    (SHARED, UNSORTED) collides with the seeded row and is rejected by
    the create handler's UNIQUE-constraint path with HTTP 409.
    """
    import re

    cleaned = re.sub(_PROJECT_KEY_ALLOWED_PATTERN, "_", name).strip("_").upper()
    return (cleaned or _PROJECT_KEY_FALLBACK)[:_PROJECT_KEY_MAX_LENGTH]


def _project_to_response(project: Project) -> ProjectResponse:
    return ProjectResponse(
        id=project.id,
        key=project.key,
        name=project.name,
        kind="shared" if project.is_shared else "normal",
        created_at=_dt_iso(project.created_at),
    )


async def _build_review_service(
    db: Database,
    config: AppConfig,
    runtime_provider: RuntimeProvider | None = None,
) -> ReviewService:
    """Build a request-scoped review service.

    Embedding configuration comes from the same effective resolution that the
    sync runtime uses, so admin overrides flow through to approve-time
    re-embedding instead of going to a stale process default.
    """
    memory_store = await _build_memory_store(db, config, runtime_provider)
    return ReviewService(db=db, memory_store=memory_store)


async def _build_agent_session_window_client(db: Database, config: AppConfig):
    """Build the request-scoped LLM client for agent-session window packaging."""

    from memforge.llm.providers import is_litellm_provider_model
    from memforge.llm.structured import LiteLlmStructuredClient, StructuredLlmConfig
    from memforge.runtime import get_effective_llm_config

    llm = await get_effective_llm_config(db, config)
    if not llm.enrichment_api_key and not is_litellm_provider_model(llm.enrichment_model):
        return None
    return LiteLlmStructuredClient(
        StructuredLlmConfig(
            model=llm.enrichment_model,
            base_url=llm.enrichment_base_url or None,
            api_key=llm.enrichment_api_key or None,
            timeout_s=llm.request_timeout_s,
        )
    )


# ---------------------------------------------------------------------------
# Dependency injection
# ---------------------------------------------------------------------------


def get_db(request: Request) -> Database:
    """FastAPI dependency: retrieve the Database instance from app.state."""
    return request.app.state.db


def get_config(request: Request) -> AppConfig:
    """FastAPI dependency: retrieve the AppConfig instance from app.state."""
    return request.app.state.config


def get_document_store(request: Request) -> DocumentArtifactStore:
    """FastAPI dependency: retrieve the configured document artifact store."""
    return request.app.state.document_store


def get_sync_service(request: Request) -> SyncService:
    """FastAPI dependency: retrieve the app-scoped sync service."""
    return request.app.state.sync_service


def get_sync_scheduler(request: Request) -> SyncScheduler | None:
    """FastAPI dependency: retrieve the app-scoped scheduler."""
    return getattr(request.app.state, "sync_scheduler", None)


def get_runtime_provider(request: Request) -> RuntimeProvider:
    """FastAPI dependency: retrieve the app-scoped runtime provider."""
    return request.app.state.runtime_provider


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_admin_app(
    db: Database | None = None,
    config: AppConfig | None = None,
    runtime_provider: RuntimeProvider | None = None,
    principal_resolver: Callable[[Request], str] | None = None,
    workspace_role_resolver: Callable[[Request], str] | None = None,
    document_store: DocumentArtifactStore | None = None,
) -> FastAPI:
    """Create and configure the MemForge Admin API FastAPI application.

    Parameters
    ----------
    db:
        Connected Database instance for all storage operations.
    config:
        Application configuration (paths, LLM settings, server settings).

    Returns
    -------
    FastAPI
        Fully configured application ready to be served by uvicorn.
    """
    if config is None:
        raise ValueError("config is required")
    runtime_provider = runtime_provider or DefaultRuntimeProvider()

    owned_db: Database | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal owned_db

        if db is None:
            owned_db = Database(config.storage.db_path)
            await owned_db.connect()
            app.state.db = owned_db
        else:
            app.state.db = db

        app.state.config = config
        app.state.document_store = document_store or LocalDocumentStore(config.storage.docs_path)
        app.state.runtime_provider = runtime_provider
        app.state.principal_resolver = principal_resolver
        app.state.workspace_role_resolver = workspace_role_resolver
        app.state.sync_service = SyncService(app.state.db, config, runtime_provider=runtime_provider)
        app.state.sync_scheduler = SyncScheduler(app.state.db, app.state.sync_service)
        await app.state.sync_scheduler.start()

        try:
            yield
        finally:
            await app.state.sync_scheduler.shutdown()
            await app.state.sync_service.shutdown()
            if owned_db is not None:
                await owned_db.close()

    app = FastAPI(
        title="MemForge Admin API",
        version="0.1.0",
        description="Management API for the MemForge agent memory layer.",
        lifespan=lifespan,
    )
    if db is not None:
        app.state.db = db
        app.state.sync_service = SyncService(db, config, runtime_provider=runtime_provider)
        app.state.sync_scheduler = SyncScheduler(db, app.state.sync_service)
    app.state.config = config
    app.state.document_store = document_store or LocalDocumentStore(config.storage.docs_path)
    app.state.runtime_provider = runtime_provider
    app.state.principal_resolver = principal_resolver
    app.state.workspace_role_resolver = workspace_role_resolver

    # -- CORS --
    cors_origins = config.server.cors_origins.split(",") if config.server.cors_origins != "*" else ["*"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- JWT Authentication --
    _security = HTTPBearer(auto_error=False)

    async def verify_jwt(
        credentials: HTTPAuthorizationCredentials | None = Depends(_security),
    ) -> dict:
        """Verify JWT token from Authorization header. Returns decoded payload."""
        import jwt as pyjwt

        if not config.server.jwt_secret or config.server.jwt_secret == "dev-secret-change-me":
            # Dev mode: skip auth when no secret is configured
            return {"sub": "dev", "role": "admin"}

        if credentials is None:
            raise HTTPException(status_code=401, detail="Missing authorization header")

        try:
            payload = pyjwt.decode(
                credentials.credentials,
                config.server.jwt_secret,
                algorithms=["HS256"],
            )
            return payload
        except pyjwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except pyjwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

    # -- Auth endpoints --
    auth_router = APIRouter(prefix="/api/auth", tags=["auth"])

    class LoginRequest(BaseModel):
        username: str
        password: str

    class LoginResponse(BaseModel):
        token: str
        expires_in: int = 86400

    @auth_router.post("/login", response_model=LoginResponse)
    async def login(req: LoginRequest, db: Database = Depends(get_db)):
        """Authenticate and return a JWT token."""
        import bcrypt
        import jwt as pyjwt

        user = await db.get_user_by_username(req.username)
        if not user or not bcrypt.checkpw(req.password.encode(), user["password_hash"].encode()):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        payload = {
            "sub": req.username,
            "role": user.get("role", "admin"),
            "exp": datetime.now(timezone.utc).timestamp() + 86400,
        }
        token = pyjwt.encode(payload, config.server.jwt_secret, algorithm="HS256")
        return LoginResponse(token=token)

    @auth_router.get("/jira-session", response_model=JiraSessionStatusResponse)
    async def get_jira_session(base_url: str, db: Database = Depends(get_db)):
        """Return redacted Jira browser-session status for a Jira origin."""
        try:
            status = await JiraAuthSessionService(db).get_status(base_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JiraSessionStatusResponse(**status)

    @auth_router.post("/jira-session", response_model=JiraSessionStatusResponse)
    async def upload_jira_session(
        req: JiraSessionUploadRequest,
        request: Request,
        db: Database = Depends(get_db),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """Store a client-captured Jira session cookie. The server validates it."""
        _require_secure_or_loopback(request)
        try:
            if req.confirm_principal_change:
                await _cancel_running_jira_browser_sources_for_origin(
                    db=db,
                    sync_service=sync_service,
                    base_url=req.base_url,
                )
            result = await JiraAuthSessionService(db).store_uploaded_session(
                base_url=req.base_url,
                cookie_header=req.cookie_header,
                browser=req.browser,
                confirm_principal_change=req.confirm_principal_change,
            )
        except JiraPrincipalChangedError as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": str(exc),
                    "origin": exc.origin,
                    "old_principal_id": exc.old_principal_id,
                    "new_principal_id": exc.new_principal_id,
                },
            ) from exc
        except (JiraAuthSessionError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JiraSessionStatusResponse(**result)

    @auth_router.get("/jira-origins")
    async def list_jira_origins(db: Database = Depends(get_db)):
        """Known Jira origins: authenticated sessions plus configured sources."""
        origins = await browser_session.list_origins(db, "jira")
        return {"origins": origins}

    @auth_router.delete("/jira-session")
    async def forget_jira_session(base_url: str, db: Database = Depends(get_db)):
        """Delete the stored Jira session for an origin."""
        try:
            return await browser_session.forget(db, "jira", base_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @auth_router.post("/jira-session/expire")
    async def expire_jira_session(req: JiraSessionExpireRequest, db: Database = Depends(get_db)):
        """Mark a Jira session expired (the client found the browser session dead)."""
        try:
            await JiraAuthSessionService(db).mark_expired(req.base_url, req.error)
            return {"ok": True}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # -- Exception handlers --

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc: HTTPException):
        return _json_error(404, str(exc.detail) if hasattr(exc, "detail") else "Not found")

    @app.exception_handler(500)
    async def server_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled server error")
        return _json_error(500, "Internal server error")

    # -- Register routers --
    health_router = APIRouter(tags=["health"])
    document_router = APIRouter(prefix="/api/documents", tags=["documents"])
    memory_router = APIRouter(prefix="/api/memories", tags=["memories"])
    review_router = APIRouter(prefix="/api/memory-reviews", tags=["memory-reviews"])
    entity_router = APIRouter(prefix="/api/entities", tags=["entities"])
    gene_router = APIRouter(prefix="/api/genes", tags=["genes"])
    source_router = APIRouter(prefix="/api/sources", tags=["sources"])
    agent_session_router = APIRouter(prefix="/api/agent-sessions", tags=["agent-sessions"])
    hook_router = APIRouter(prefix="/api/hooks", tags=["hooks"])
    recent_change_router = APIRouter(prefix="/api/recent-changes", tags=["recent-changes"])
    schedule_router = APIRouter(prefix="/api/schedule", tags=["schedule"])
    llm_router = APIRouter(prefix="/api/llm-config", tags=["llm-config"])
    projects_router = APIRouter(prefix="/api/projects", tags=["projects"])

    async def get_search_engine(request: Request, db: Database, config: AppConfig):
        from memforge.memory.audit import AuditContext, MemoryAuditLogger

        engine = getattr(request.app.state, "memory_search_engine", None)
        if engine is None:
            runtime_provider = request.app.state.runtime_provider
            engine = await runtime_provider.build_search_engine(
                db,
                config,
                audit_logger=MemoryAuditLogger(db, default_context=AuditContext(actor_type="admin")),
            )
            request.app.state.memory_search_engine = engine
        return engine

    # ===================================================================
    # 1. Health & Stats
    # ===================================================================

    @health_router.get("/api/health", response_model=HealthResponse)
    async def health(request: Request, db: Database = Depends(get_db)):
        """System health check through the active runtime provider."""
        runtime_provider = request.app.state.runtime_provider
        report = await runtime_provider.check_health(db, config)
        return HealthResponse(
            status=report.status,
            database=_runtime_component_to_response(report.database),
            vector_store=_runtime_component_to_response(report.vector_store),
            index_consistency=(
                _runtime_component_to_response(report.index_consistency) if report.index_consistency else None
            ),
            audit_failures=(_runtime_component_to_response(report.audit_failures) if report.audit_failures else None),
            genes={name: _runtime_component_to_response(component) for name, component in report.genes.items()},
        )

    @health_router.get("/api/stats", response_model=StatsResponse)
    async def stats(db: Database = Depends(get_db)):
        """Overall system statistics: memory counts, entity counts, source counts."""
        # Memory counts by type
        type_counts: list[MemoryStatEntry] = []
        for mt in ["fact", "decision", "convention", "procedure"]:
            count = await db.count_memories(type=mt)
            type_counts.append(MemoryStatEntry(key=mt, count=count))

        # Memory counts by status
        status_counts: list[MemoryStatEntry] = []
        for st in ["active", "superseded", "retired", "pending_review"]:
            count = await db.count_memories(status=st)
            status_counts.append(MemoryStatEntry(key=st, count=count))

        total_memories = await db.count_memories()

        # Entity count
        entities = await db.get_all_entities()
        total_entities = len(entities)

        # Source count
        sources = await db.list_sources()
        total_sources = len(sources)

        return StatsResponse(
            total_memories=total_memories,
            memories_by_type=type_counts,
            memories_by_status=status_counts,
            total_entities=total_entities,
            total_sources=total_sources,
        )

    # ===================================================================
    # 1b. Source Document Artifacts
    # ===================================================================

    @document_router.get("/{doc_id}/artifacts")
    async def list_document_artifact_manifest(
        doc_id: str,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """List service-readable artifacts for a stored source document."""
        doc = await db.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")

        artifacts = list_document_artifacts(doc, config, artifact_store)
        return {
            "doc_id": doc.doc_id,
            "title": doc.title,
            "source_type": doc.source,
            "source_url": doc.source_url,
            "artifacts": {kind: artifact.metadata() for kind, artifact in artifacts.items()},
        }

    @document_router.api_route("/{doc_id}/artifacts/{kind}", methods=["GET", "HEAD"])
    async def get_document_artifact(
        doc_id: str,
        kind: str,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """Serve an explicit source artifact kind through the API."""
        doc = await db.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")

        artifact = select_document_artifact(doc, kind, config, artifact_store)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Document artifact not found")

        content = b"" if request.method == "HEAD" else artifact_store.read_artifact(artifact.uri)
        return Response(
            content=content,
            media_type=artifact.media_type,
            headers={"Content-Disposition": f'inline; filename="{artifact.filename}"'},
        )

    @document_router.api_route("/{doc_id}/content", methods=["GET", "HEAD"])
    async def get_document_content(
        doc_id: str,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """Serve normalized source content through the API for Docker/SaaS clients."""
        doc = await db.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")

        artifact = select_document_artifact(doc, "content", config, artifact_store)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Document content artifact not found")

        content = b"" if request.method == "HEAD" else artifact_store.read_artifact(artifact.uri)
        return Response(
            content=content,
            media_type=artifact.media_type,
            headers={"Content-Disposition": f'inline; filename="{artifact.filename}"'},
        )

    @document_router.api_route("/{doc_id}/pdf", methods=["GET", "HEAD"])
    async def get_document_pdf(
        doc_id: str,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """Serve a stored source PDF through the API for Docker/SaaS clients."""
        doc = await db.get_document(doc_id)
        if doc is None:
            raise HTTPException(status_code=404, detail="Document not found")

        artifact = select_document_artifact(doc, "pdf", config, artifact_store)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Document PDF artifact not found")

        content = b"" if request.method == "HEAD" else artifact_store.read_artifact(artifact.uri)
        return Response(
            content=content,
            media_type=artifact.media_type,
            headers={"Content-Disposition": f'inline; filename="{artifact.filename}"'},
        )

    # ===================================================================
    # 2. Memory Endpoints
    # ===================================================================

    @memory_router.get("/stats", response_model=MemoryStatsResponse)
    async def memory_stats(db: Database = Depends(get_db)):
        """Memory counts broken down by type and status."""
        by_type: list[MemoryStatEntry] = []
        for mt in ["fact", "decision", "convention", "procedure"]:
            count = await db.count_memories(type=mt)
            by_type.append(MemoryStatEntry(key=mt, count=count))

        by_status: list[MemoryStatEntry] = []
        for st in ["active", "superseded", "retired", "pending_review"]:
            count = await db.count_memories(status=st)
            by_status.append(MemoryStatEntry(key=st, count=count))

        total = await db.count_memories()

        return MemoryStatsResponse(by_type=by_type, by_status=by_status, total=total)

    @memory_router.post("/search")
    async def search_memories(
        req: MemorySearchRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
    ):
        """Service-owned memory search used by local agent MCP proxies."""
        from memforge.memory.lifecycle import allowed_search_statuses
        from memforge.storage.adapters.context import AccessScope

        try:
            engine = await get_search_engine(request, db, config)
            user_id = resolve_request_principal(request)
            # `project-first` and `workspace` modes weight cross-project hits
            # but keep them visible at the predicate. `project` mode is the
            # only mode that narrows the workspace branch, and it narrows to
            # `{active_project, SHARED}` inside the predicate itself.
            scope = AccessScope(
                user_id=user_id,
                include_private=req.include_private,
                allowed_statuses=allowed_search_statuses(req.include_superseded),
                active_project=req.active_project,
                scope_mode=req.scope_mode,
                active_repo_identifier=req.active_repo_identifier,
            )
            result = await engine.search(
                query=req.query,
                memory_types=req.memory_types,
                sources=req.sources,
                time_range=req.time_range,
                entities=req.entities,
                include_superseded=req.include_superseded,
                top_k=req.top_k,
                source_filter=(req.source_filter.to_source_filter() if req.source_filter is not None else None),
                request_scope=scope,
            )
            # Row-status filter is applied post-rank so the GET and POST
            # routes share the same status semantics without coupling the
            # ranker to lifecycle filtering.
            if req.status:
                kept = [r for r in result["results"] if getattr(r, "status", "active") == req.status]
                result["results"] = kept
                result["total_candidates"] = len(kept)
            return _json_ready(result)
        except Exception as e:
            logger.warning("Search failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=503,
                detail=f"Search unavailable: {e}",
            ) from e

    @memory_router.get("/contradictions")
    async def memory_contradictions(
        limit: int = 50,
        offset: int = 0,
        db: Database = Depends(get_db),
    ):
        """List memories that have contradiction_count > 0."""
        async with db.db.execute(
            """SELECT * FROM memories
               WHERE contradiction_count > 0 AND status = 'active'
               ORDER BY contradiction_count DESC
               LIMIT ? OFFSET ?""",
            (limit, offset),
        ) as cursor:
            rows = await cursor.fetchall()

        memories = [_memory_to_response(db._row_to_memory(row)) for row in rows]

        async with db.db.execute(
            "SELECT COUNT(*) FROM memories WHERE contradiction_count > 0 AND status = 'active'"
        ) as cursor:
            total_row = await cursor.fetchone()
            total = total_row[0] if total_row else 0

        return {"data": [m.model_dump() for m in memories], "total": total}

    @memory_router.get("/{memory_id}", response_model=MemoryDetailResponse)
    async def get_memory(
        memory_id: str,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """Get full memory detail including provenance (linked source documents)."""
        scope = _workspace_default_scope(request, include_private=True)
        mem = await db.get_memory(memory_id)
        if not mem:
            raise HTTPException(status_code=404, detail="Memory not found")
        # Default-deny: a row the caller cannot see by the access predicate is
        # indistinguishable from a missing row to that caller.
        visible = await _filter_visible_ids(db, [memory_id], scope)
        if memory_id not in visible:
            raise HTTPException(status_code=404, detail="Memory not found")

        # Fetch provenance: memory_sources with document titles and artifact URLs.
        raw_sources = await db.get_memory_sources(memory_id)
        source_details: list[MemorySourceDetail] = []
        for ms in raw_sources:
            doc = await db.get_document(ms.doc_id)
            source_details.append(_memory_source_detail(ms, doc, config, artifact_store))

        # Fetch linked entity names.
        entity_names = await db.get_memory_entity_names(memory_id)

        origin_info = (await _origin_source_types(db, [memory_id])).get(memory_id, (None, None))
        origin_source_type, origin_client = origin_info

        return MemoryDetailResponse(
            id=mem.id,
            memory_type=mem.memory_type,
            content=mem.content,
            content_hash=mem.content_hash,
            visibility=mem.visibility,
            owner_user_id=mem.owner_user_id,
            project_key=mem.project_key,
            tags=mem.tags,
            confidence=mem.confidence,
            corroboration_count=mem.corroboration_count,
            contradiction_count=mem.contradiction_count,
            valid_from=_dt_iso(mem.valid_from),
            valid_until=_dt_iso(mem.valid_until),
            superseded_by=mem.superseded_by,
            status=mem.status,
            retirement_reason=mem.retirement_reason,
            retired_at=_dt_iso(mem.retired_at),
            superseded_at=_dt_iso(mem.superseded_at),
            replacement_reason=mem.replacement_reason,
            replacement_kind=mem.replacement_kind,
            extraction_context=mem.extraction_context,
            created_at=_dt_iso(mem.created_at),
            updated_at=_dt_iso(mem.updated_at),
            entity_refs=entity_names,
            sources=source_details,
            origin_source_type=origin_source_type,
            origin_client=origin_client,
        )

    # -- Memory update (admin actions) --

    @memory_router.put("/{memory_id}")
    async def update_memory(
        memory_id: str,
        req: MemoryUpdateRequest = Body(...),
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        """Update a memory's content, confidence, or status (admin override)."""
        memory = await db.get_memory(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        if req.content is not None or req.confidence is not None:
            memory_store = await _build_memory_store(db, config, runtime_provider)
            await memory_store.update_memory(
                memory_id,
                new_content=req.content or memory.content,
                new_confidence=req.confidence,
                new_tags=memory.tags,
            )
        if req.status is not None:
            if req.status not in ("active", "superseded", "retired", "decayed", "pending_review"):
                raise HTTPException(status_code=400, detail=f"Invalid status: {req.status}")
            status = normalize_memory_status(req.status)
            memory_store = await _build_memory_store(db, config, runtime_provider)
            if status == "retired":
                await memory_store.retire_memory(memory_id, reason="admin_hidden")
            elif status == "pending_review":
                await memory_store.mark_pending_review(memory_id, reason="admin_hidden")
            elif status == "active":
                if memory.status != "active":
                    raise HTTPException(
                        status_code=400,
                        detail="Reactivating a hidden memory requires an explicit reindex workflow.",
                    )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Use the review workflow to supersede memories.",
                )

        return {"status": "updated", "memory_id": memory_id}

    @memory_router.delete("/{memory_id}")
    async def delete_memory(
        memory_id: str,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        """Soft-delete a memory (mark as retired and hide from search)."""
        memory = await db.get_memory(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")
        memory_store = await _build_memory_store(db, config, runtime_provider)
        await memory_store.retire_memory(memory_id, reason="admin_hidden")
        return {"status": "deleted", "memory_id": memory_id}

    @memory_router.delete("/{memory_id}/purge")
    async def purge_memory(
        memory_id: str,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        """Hard-purge a memory for privacy/compliance removal."""
        memory = await db.get_memory(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        memory_store = await _build_memory_store(db, config, runtime_provider)
        purged = await memory_store.purge_memory(memory_id)
        if not purged:
            raise HTTPException(status_code=404, detail="Memory not found")
        return {"status": "purged", "memory_id": memory_id}

    @memory_router.get("", response_model=MemoryListResponse)
    async def list_memories(
        request: Request,
        type: str | None = None,
        status: str | None = None,
        source: str | None = None,
        project: str | None = None,
        search: str | None = None,
        include_private: bool = False,
        limit: int = 50,
        offset: int = 0,
        db: Database = Depends(get_db),
    ):
        """List memories with pagination and filters.

        Filters: type (fact/decision/convention/procedure), status, source,
        project, free-text search. Supports limit/offset pagination.

        The access predicate gates every row: workspace rows are visible
        across every project (the ranker handles project relevance, not
        the predicate), and the caller's own private rows surface only
        when ``include_private=True``.
        """
        scope = _workspace_default_scope(request, include_private=include_private)
        page = await list_memory_admin_page(
            db,
            scope=scope,
            filters=MemoryAdminListFilters(
                memory_type=type,
                status=status,
                source=source,
                project=project,
                search=search,
            ),
            limit=limit,
            offset=offset,
        )
        return MemoryListResponse(
            data=[_memory_to_response(m, *page.origins.get(m.id, (None, None))) for m in page.memories],
            total=page.total,
            limit=limit,
            offset=offset,
        )

    # ===================================================================
    # 3. Entity Endpoints
    # ===================================================================

    @entity_router.get("", response_model=EntityListResponse)
    async def list_entities(
        tag: str | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Database = Depends(get_db),
    ):
        """List entities with optional tag filter and search."""
        entity_rows, total = await db.list_entities(
            tag=tag,
            search=search,
            limit=limit,
            offset=offset,
        )
        entities = [
            EntityResponse(
                id=ent.id,
                canonical_name=ent.canonical_name,
                tags=ent.tags,
                display_name=ent.display_name,
                created_at=_dt_iso(ent.created_at),
            )
            for ent in entity_rows
        ]

        return EntityListResponse(data=entities, total=total)

    @entity_router.get("/{entity_id}", response_model=EntityDetailResponse)
    async def get_entity(entity_id: int, db: Database = Depends(get_db)):
        """Get entity detail with aliases and linked memory count."""
        ent = await db.get_entity(entity_id)
        if ent is None:
            raise HTTPException(status_code=404, detail="Entity not found")

        aliases = await db.get_aliases_for_entity(entity_id)
        alias_responses = [
            EntityAliasResponse(
                alias=a.alias,
                alias_normalized=a.alias_normalized,
                source=a.source,
                created_at=_dt_iso(a.created_at),
            )
            for a in aliases
        ]

        linked_count = await db.count_memories_for_entity(entity_id)

        return EntityDetailResponse(
            id=ent.id,
            canonical_name=ent.canonical_name,
            tags=ent.tags,
            display_name=ent.display_name,
            created_at=_dt_iso(ent.created_at),
            aliases=alias_responses,
            linked_memory_count=linked_count,
        )

    @entity_router.post("/merge")
    async def merge_entities(
        req: MergeEntitiesRequest,
        db: Database = Depends(get_db),
    ):
        """Merge two entities: reassign all references from source to target.

        All memory_entities rows, aliases, and document references pointing to
        source_id are moved to target_id. The source entity is then deleted.
        """
        try:
            merged = await db.merge_entities(
                source_id=req.source_id,
                target_id=req.target_id,
            )
        except LookupError as exc:
            detail = str(exc)
            if detail not in {"Source entity not found", "Target entity not found"}:
                detail = "Entity not found"
            raise HTTPException(status_code=404, detail=detail)

        return {
            "ok": True,
            "merged": merged,
        }

    @entity_router.get("/{entity_id}/aliases")
    async def list_entity_aliases(
        entity_id: int,
        db: Database = Depends(get_db),
    ):
        """List all aliases for an entity."""
        if await db.get_entity(entity_id) is None:
            raise HTTPException(status_code=404, detail="Entity not found")

        aliases = await db.get_aliases_for_entity(entity_id)
        return {
            "data": [
                EntityAliasResponse(
                    alias=a.alias,
                    alias_normalized=a.alias_normalized,
                    source=a.source,
                    created_at=_dt_iso(a.created_at),
                ).model_dump()
                for a in aliases
            ]
        }

    @entity_router.post("/{entity_id}/aliases")
    async def add_entity_alias(
        entity_id: int,
        req: AddAliasRequest,
        db: Database = Depends(get_db),
    ):
        """Add a manual alias for an entity."""
        if await db.get_entity(entity_id) is None:
            raise HTTPException(status_code=404, detail="Entity not found")

        normalized = canonicalize_entity_name(req.alias)
        await db.insert_alias(
            alias=req.alias,
            alias_normalized=normalized,
            canonical_id=entity_id,
            source="admin_manual",
        )
        return {"ok": True, "alias": req.alias, "alias_normalized": normalized}

    @entity_router.delete("/{entity_id}/aliases/{alias}")
    async def remove_entity_alias(
        entity_id: int,
        alias: str,
        db: Database = Depends(get_db),
    ):
        """Remove an alias from an entity."""
        normalized = canonicalize_entity_name(alias)
        removed = await db.remove_entity_alias(
            entity_id=entity_id,
            alias_normalized=normalized,
        )
        if not removed:
            raise HTTPException(status_code=404, detail="Alias not found")

        return {"ok": True}

    # ===================================================================
    # 4. Gene / Source Endpoints
    # ===================================================================

    @gene_router.get("", response_model=list[GeneMetadataResponse])
    async def list_genes():
        """List all available gene types from the registry."""
        genes = list_available_genes()
        return [
            GeneMetadataResponse(
                name=g.name,
                display_name=g.display_name,
                description=g.description,
                default_sync_interval_minutes=g.default_sync_interval_minutes,
                auth_method=g.auth_method,
                data_shape=g.data_shape,
            )
            for g in genes
        ]

    @gene_router.get("/{name}/config-schema", response_model=GeneConfigSchemaResponse)
    async def get_gene_config_schema(name: str):
        """Get the configuration schema for a gene type (for dynamic UI rendering)."""
        if name not in GENE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Gene '{name}' not found")

        cls = GENE_REGISTRY[name]
        schema = cls.config_schema()

        return GeneConfigSchemaResponse(
            groups=[ConfigGroupResponse(key=g.key, label=g.label, order=g.order) for g in schema.groups],
            fields=[
                ConfigFieldResponse(
                    key=f.key,
                    label=f.label,
                    field_type=f.field_type.value,
                    required=f.required,
                    placeholder=f.placeholder,
                    help_text=f.help_text,
                    group=f.group,
                    order=f.order,
                    default=f.default,
                    options=f.options,
                    advanced=f.advanced,
                )
                for f in schema.fields
            ],
        )

    @gene_router.post("/{name}/preview-discovery", response_model=DiscoveryPreviewResponse)
    async def preview_gene_discovery(
        name: str,
        req: DiscoveryPreviewRequest,
        db: Database = Depends(get_db),
    ):
        """Preview the documents a source config would discover without saving it."""
        if name not in GENE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Gene '{name}' not found")

        try:
            _validate_source_config(name, req.config)
            preview_config = dict(req.config)
            preview_config["_memforge_preview_limit"] = req.limit + 1
            # Browser-session sources keep the cookie in the auth store, not the
            # source config. Inject it the same way a real sync does (no-op for
            # source types that do not use a browser session).
            await browser_session.inject_cookie_for_source(db, name, preview_config)
            gene = create_gene(name, preview_config, source_id=f"preview-{name}")
            await gene.authenticate()
            items: list[DiscoveryPreviewItemResponse] = []
            count = 0
            async for item in gene.discover(since=None):
                count += 1
                if len(items) < req.limit:
                    items.append(
                        DiscoveryPreviewItemResponse(
                            item_id=item.item_id,
                            title=item.title,
                            source_url=item.source_url,
                            last_modified=_dt_iso(item.last_modified),
                        )
                    )
                if count > req.limit:
                    break
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except browser_session.BrowserSessionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Discovery preview failed for gene %s", name)
            raise HTTPException(status_code=502, detail=f"Discovery preview failed: {exc}") from exc
        finally:
            client = locals().get("gene") and getattr(locals()["gene"], "_client", None)
            if client is not None:
                close = getattr(client, "aclose", None)
                if close:
                    await close()

        return DiscoveryPreviewResponse(
            source_type=name,
            count=count,
            truncated=count > len(items),
            items=items,
        )

    @gene_router.get("/teams/auth-check", response_model=TeamsAuthCheckResponse)
    async def teams_auth_check():
        """Check if Teams authentication tokens are available and valid."""
        from memforge.auth.teams_auth import TeamsAuthenticator

        tokens = TeamsAuthenticator.load_tokens()
        if not tokens:
            return TeamsAuthCheckResponse(
                authenticated=False,
                error="No Teams tokens found. Run: memforge auth teams",
            )

        validity = TeamsAuthenticator.check_token_expiry(tokens)
        if not any(validity.values()):
            return TeamsAuthCheckResponse(
                authenticated=False,
                error="All Teams tokens have expired. Run: memforge auth teams",
            )

        # Find the shortest time-to-expiry across valid tokens
        now = datetime.now(timezone.utc).timestamp()
        min_minutes = None
        for aud, is_valid in validity.items():
            if is_valid and isinstance(tokens.get(aud), dict):
                exp = tokens[aud].get("expiresAt", 0)
                if exp > 0:
                    remaining = int((exp - now) / 60)
                    if min_minutes is None or remaining < min_minutes:
                        min_minutes = remaining

        return TeamsAuthCheckResponse(
            authenticated=True,
            expires_in_minutes=min_minutes,
        )

    @gene_router.get("/teams/browse", response_model=TeamsBrowseResponse)
    async def teams_browse(region: str = "emea"):
        """Browse available Teams conversations for the picker UI."""
        from memforge.auth.teams_auth import TeamsAuthenticator
        from memforge.genes.teams_gene import _TeamsAPIClient

        tokens = TeamsAuthenticator.load_tokens()
        if not tokens:
            raise HTTPException(
                status_code=401,
                detail="No Teams tokens. Run: memforge auth teams",
            )

        client = _TeamsAPIClient(region=region)
        try:
            # Load chat token — only the Chat API is needed for browsing
            chat_token = TeamsAuthenticator.get_token_for_audience(
                tokens,
                "https://ic3.teams.office.com",
            )

            if not chat_token:
                raise HTTPException(status_code=401, detail="Missing Chat API token")

            import httpx

            client._chat_client = httpx.AsyncClient(
                base_url=client._chat_base,
                headers={"Authorization": f"Bearer {chat_token}"},
                timeout=30.0,
            )
            # Set graph_client to a dummy so _ensure_clients() is a no-op
            client._graph_client = httpx.AsyncClient(
                base_url="https://localhost",
                timeout=1.0,
            )

            # Fetch channels (grouped by team) + conversations concurrently
            raw_channel_teams, raw_convos = await asyncio.gather(
                client.list_channels(),
                client.list_conversations(),
                return_exceptions=True,
            )

            # Handle errors gracefully
            if isinstance(raw_channel_teams, Exception):
                logger.warning("Failed to fetch channels: %s", raw_channel_teams)
                raw_channel_teams = []
            if isinstance(raw_convos, Exception):
                logger.warning("Failed to fetch conversations: %s", raw_convos)
                raw_convos = []

            # Build teams result from Chat API channel data
            teams_result: list[TeamsTeamResponse] = [
                TeamsTeamResponse(
                    id=t["id"],
                    displayName=t["displayName"],
                    channels=[TeamsChannelResponse(id=ch["id"], displayName=ch["displayName"]) for ch in t["channels"]],
                )
                for t in raw_channel_teams
                if t.get("channels")  # skip teams with no named channels
            ]

            # Filter out system conversations (48:notifications, 48:mentions, etc.)
            user_convos = [c for c in raw_convos if c.get("id", "").startswith("19:")]

            # Resolve individual chat display names from lastMessage sender info.
            # Build a GUID→name map from all conversations' last message senders,
            # then use it to name DMs where the last sender was the current user.
            import base64 as _b64

            my_oid = ""
            try:
                parts = chat_token.split(".")
                payload = parts[1] + "=" * (4 - len(parts[1]) % 4)
                jwt_data = json.loads(_b64.b64decode(payload))
                my_oid = jwt_data.get("oid", "")
            except Exception:
                pass

            guid_to_name: dict[str, str] = {}
            if my_oid:
                for c in user_convos:
                    sender_id = c.get("lastMessageSenderId", "")
                    sender_name = c.get("lastMessageSender", "")
                    if sender_name and "orgid:" in sender_id:
                        guid = sender_id.split("orgid:")[-1].split("/")[0]
                        if guid and guid != my_oid:
                            guid_to_name[guid] = sender_name

            def _resolve_dm_name(conv: dict) -> str:
                """Get the other person's name for a 1:1 chat."""
                sender_id = conv.get("lastMessageSenderId", "")
                is_me = my_oid and my_oid in sender_id
                if not is_me:
                    return conv.get("lastMessageSender", "")
                # Last message was from me — look up other person from GUID map
                cid = conv["id"]
                id_part = cid.replace("19:", "").split("@")[0]
                guids = id_part.split("_")
                other_guid = next((g for g in guids if g != my_oid), None)
                return guid_to_name.get(other_guid, "") if other_guid else ""

            # Partition chats by type, sorted by recency
            group_chats = sorted(
                [
                    TeamsChatResponse(
                        id=c["id"],
                        topic=c.get("topic", "Untitled"),
                        lastActivity=c["lastActivity"].isoformat() if c.get("lastActivity") else None,
                    )
                    for c in user_convos
                    if c.get("type") == "group_chat"
                ],
                key=lambda x: x.lastActivity or "",
                reverse=True,
            )
            # Include individual chats with resolved display names
            individual_chats_raw = [c for c in user_convos if c.get("type") == "individual_chat"]
            individual_chats = sorted(
                [
                    TeamsChatResponse(
                        id=c["id"],
                        topic=name,
                        lastActivity=c["lastActivity"].isoformat() if c.get("lastActivity") else None,
                    )
                    for c in individual_chats_raw
                    if (name := _resolve_dm_name(c))
                ],
                key=lambda x: x.lastActivity or "",
                reverse=True,
            )

            # Deduplicate by ID (keep most recent)
            seen_ids: set[str] = set()
            deduped_individual: list[TeamsChatResponse] = []
            for chat in individual_chats:
                if chat.id not in seen_ids:
                    seen_ids.add(chat.id)
                    deduped_individual.append(chat)

            # Build favorites from teams/channels marked as favorite
            favorites: list[TeamsChatResponse] = []
            for t in raw_channel_teams:
                if t.get("favorite"):
                    # Favorited team — add the team name as a selectable item
                    # using the first channel's ID (General channel) as the identifier
                    fav_channels = [ch for ch in t["channels"] if ch.get("favorite")]
                    if fav_channels:
                        # Use the specific favorite channel
                        for ch in fav_channels:
                            favorites.append(
                                TeamsChatResponse(
                                    id=ch["id"],
                                    topic=f"{t['displayName']} / {ch['displayName']}",
                                )
                            )
                    else:
                        # Team is favorite but no specific channel — show team name
                        # Pick the first channel as representative
                        if t["channels"]:
                            favorites.append(
                                TeamsChatResponse(
                                    id=t["channels"][0]["id"],
                                    topic=t["displayName"],
                                )
                            )

            return TeamsBrowseResponse(
                favorites=favorites,
                teams=teams_result,
                group_chats=group_chats,
                individual_chats=deduped_individual,
            )
        finally:
            await client.close()

    @source_router.get("")
    async def list_sources(
        request: Request,
        db: Database = Depends(get_db),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """List all configured sources with per-source memory counts and sync status."""
        sources = await list_source_admin_rows(
            db,
            sync_service=sync_service,
            viewer_id=resolve_request_principal(request),
            viewer_role=resolve_request_workspace_role(request),
        )

        # Attach memory_count and sync status to each source
        from memforge.agent_sessions import (
            AGENT_SESSION_SOURCE_TYPE,
            agent_session_client_for_source_id,
        )

        jira_auth_service = JiraAuthSessionService(db)
        for s in sources:
            original_config = s.get("config", {})
            jira_auth_mode = _jira_auth_mode(original_config) if s["type"] == "jira" else None
            if s.get("capabilities", {}).get("can_configure"):
                s["config"] = redact_source_config(
                    original_config,
                    secret_fields=_source_secret_fields(s["type"]),
                    validate_encryption=True,
                )
            else:
                s["config"] = {}
            # Surface the originating client for agent-session sources so the UI
            # can pick a per-client brand mark without re-deriving from the id.
            if s["type"] == AGENT_SESSION_SOURCE_TYPE:
                s["client"] = agent_session_client_for_source_id(s["id"])
            else:
                s["client"] = None
            if s["type"] == "jira" and jira_auth_mode == "browser_cookie":
                try:
                    s["auth_session"] = await jira_auth_service.get_status(
                        str(s.get("config", {}).get("base_url") or "")
                    )
                except ValueError as exc:
                    s["auth_session"] = {
                        "provider": "jira",
                        "origin": "",
                        "status": "failed",
                        "last_error": str(exc),
                    }

        return {"data": sources}

    @source_router.get("/{source_id}/projects", response_model=SourceProjectsResponse)
    async def list_source_projects(
        source_id: str,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """List project buckets observed for one source."""
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        projects = [
            SourceProjectResponse(
                project=str(row["project"]),
                document_count=int(row["document_count"]),
                memory_count=int(row["memory_count"]),
                last_observed_at=row.get("last_observed_at"),
            )
            for row in await db.list_source_projects(
                source_id,
                include_private=True,
                owner_user_id=resolve_request_principal(request),
            )
        ]

        return SourceProjectsResponse(source_id=source_id, projects=projects)

    @source_router.get("/{source_id}/projects/resolved", response_model=ResolvedProjectsResponse)
    async def list_source_resolved_projects(
        source_id: str,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """List the resolved `project_key` distribution for one source.

        Reflects where the project resolver actually placed writes under
        the source's current `project_binding`. The sibling
        `/projects` endpoint reports the raw `documents.space_or_project`
        observed during sync, before resolution.
        """
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        rows = await db.list_resolved_projects_for_source(
            source_id,
            include_private=True,
            owner_user_id=resolve_request_principal(request),
        )
        projects = [ResolvedProjectResponse(project_key=key, memory_count=count) for key, count in rows]
        return ResolvedProjectsResponse(source_id=source_id, projects=projects)

    @source_router.put("/{source_id}/subscription")
    async def set_source_subscription(
        request: Request,
        source_id: str,
        req: SourceSubscriptionRequest,
        db: Database = Depends(get_db),
    ):
        """Set the caller's per-source subscription preference."""
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        await db.set_source_subscription(
            source_id,
            resolve_request_principal(request),
            req.enabled,
        )
        return {
            "ok": True,
            "source_id": source_id,
            "subscription": {"enabled": req.enabled},
        }

    # ===================================================================
    # 4c. Recent Changes
    # ===================================================================

    @recent_change_router.get("")
    async def recent_changes(
        since: str | None = None,
        source: str | None = None,
        include_memories: bool = True,
        db: Database = Depends(get_db),
    ):
        """Return recent source-document changes and optionally new or updated memories."""
        if since:
            since_dt = datetime.fromisoformat(since)
            if since_dt.tzinfo is None:
                since_dt = since_dt.replace(tzinfo=timezone.utc)
        else:
            since_dt = datetime.now(timezone.utc) - timedelta(days=7)
        since_iso = since_dt.isoformat()

        changelog: list[dict[str, Any]] = []
        try:
            query = "SELECT * FROM changelog WHERE detected_at >= ?"
            params: list[Any] = [since_iso]
            if source:
                query += " AND source = ?"
                params.append(source)
            query += " ORDER BY detected_at DESC LIMIT 100"

            async with db.db.execute(query, params) as cursor:
                async for row in cursor:
                    d = dict(row)
                    changelog.append(
                        {
                            "id": d["id"],
                            "doc_id": d["doc_id"],
                            "change_type": d["change_type"],
                            "title": d.get("title"),
                            "source": d.get("source"),
                            "previous_version": d.get("previous_version"),
                            "current_version": d.get("current_version"),
                            "ai_change_summary": d.get("ai_change_summary"),
                            "detected_at": d["detected_at"],
                        }
                    )
        except Exception as e:
            logger.warning("Failed to query changelog: %s", e)

        recent_memories: list[dict[str, Any]] = []
        if include_memories:
            try:
                mem_query = "SELECT * FROM memories WHERE updated_at >= ?"
                mem_params: list[Any] = [since_iso]
                if source:
                    mem_query = (
                        "SELECT DISTINCT m.* FROM memories m "
                        "JOIN memory_sources ms ON m.id = ms.memory_id "
                        "JOIN documents d ON ms.doc_id = d.doc_id "
                        "WHERE m.updated_at >= ? AND d.source = ?"
                    )
                    mem_params = [since_iso, source]
                    mem_query += " ORDER BY m.updated_at DESC LIMIT 50"
                else:
                    mem_query += " ORDER BY updated_at DESC LIMIT 50"

                async with db.db.execute(mem_query, mem_params) as cursor:
                    async for row in cursor:
                        d = dict(row)
                        recent_memories.append(
                            {
                                "id": d["id"],
                                "memory_type": d["memory_type"],
                                "content": d["content"],
                                "confidence": d["confidence"],
                                "status": d["status"],
                                "corroboration_count": d["corroboration_count"],
                                "updated_at": d.get("updated_at"),
                                "created_at": d.get("created_at"),
                            }
                        )
            except Exception as e:
                logger.warning("Failed to query recent memories: %s", e)

        result: dict[str, Any] = {
            "since": since_iso,
            "changelog_entries": changelog,
            "total_changes": len(changelog),
        }
        if include_memories:
            result["recent_memories"] = recent_memories
            result["total_memories"] = len(recent_memories)
        return result

    @source_router.post("")
    async def create_source(
        request: Request,
        req: CreateSourceRequest,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
    ):
        """Create a new source (gene instance) with the given type, name, and config."""
        # Validate gene type exists
        if req.type not in GENE_REGISTRY:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown gene type '{req.type}'. Available: {', '.join(sorted(GENE_REGISTRY))}",
            )

        source_id = f"src-{uuid.uuid4().hex[:8]}"
        try:
            _validate_source_config(req.type, req.config)
            source_config = prepare_source_config_for_storage(
                req.config,
                secret_fields=_source_secret_fields(req.type),
            )
            _validate_source_project_binding(req.project_binding)
            if req.type == "jira":
                source_config = _drop_source_owned_jira_cookie(source_config)
            if req.type == "local_markdown":
                source_config = _populate_local_markdown_inbox(source_config, source_id, config)
        except (SecretConfigurationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        await db.upsert_source(
            id=source_id,
            type=req.type,
            name=req.name,
            config_json=json.dumps(source_config),
            project_binding=req.project_binding,
            created_by_user_id=resolve_request_principal(request),
        )
        if req.sync_schedule is not None:
            await db.set_source_sync_schedule(
                source_id,
                enabled=req.sync_schedule.enabled,
                interval_minutes=req.sync_schedule.interval_minutes,
            )
        return {"id": source_id, "name": req.name, "type": req.type}

    @source_router.put("/{source_id}")
    async def update_source(
        request: Request,
        source_id: str,
        req: UpdateSourceRequest,
        db: Database = Depends(get_db),
        sync_service: SyncService = Depends(get_sync_service),
        config: AppConfig = Depends(get_config),
    ):
        """Update an existing source's configuration."""
        existing = await db.get_source(source_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_management(request, existing)

        _validate_source_status(req.status)
        name = req.name or existing["name"]
        if req.config is not None:
            try:
                _validate_source_config(existing["type"], req.config, existing_config=existing["config"])
                src_config = prepare_source_config_for_storage(
                    req.config,
                    existing_config=existing["config"],
                    secret_fields=_source_secret_fields(existing["type"]),
                )
                if existing["type"] == "jira":
                    src_config = _drop_source_owned_jira_cookie(src_config)
                if existing["type"] == "local_markdown":
                    src_config = _populate_local_markdown_inbox(src_config, source_id, config)
            except (SecretConfigurationError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        else:
            src_config = existing["config"]
        scope_changed = req.config is not None and (
            _sync_scope_config(existing["type"], src_config) != _sync_scope_config(existing["type"], existing["config"])
        )
        auth_secret_changed = req.config is not None and _jira_auth_secret_changed(
            existing["type"],
            req.config,
            existing["config"],
        )
        old_base_url = str(existing["config"].get("base_url") or "")
        new_base_url = str(src_config.get("base_url") or "")
        base_url_changed = bool(old_base_url) and old_base_url != new_base_url

        if scope_changed or auth_secret_changed or base_url_changed:
            await sync_service.cancel_source(source_id)

        if _request_includes_field(req, "project_binding"):
            try:
                _validate_source_project_binding(req.project_binding)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        await db.upsert_source(
            id=source_id,
            type=existing["type"],
            name=name,
            config_json=json.dumps(src_config),
            status=req.status if _request_includes_field(req, "status") else None,
            project_binding=(
                req.project_binding
                if _request_includes_field(req, "project_binding")
                else existing.get("project_binding")
            ),
        )
        if _request_includes_field(req, "sync_schedule"):
            schedule = req.sync_schedule or SourceSyncScheduleRequest()
            await db.set_source_sync_schedule(
                source_id,
                enabled=schedule.enabled,
                interval_minutes=schedule.interval_minutes,
            )
        if base_url_changed:
            release_atlassian_request_limiter(old_base_url, owner_id=source_id)
        if scope_changed or auth_secret_changed:
            await db.reset_source_sync_cursor(source_id)
        return {"ok": True, "id": source_id}

    @source_router.delete("/{source_id}")
    async def delete_source(
        request: Request,
        source_id: str,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        document_store: DocumentArtifactStore = Depends(get_document_store),
        sync_service: SyncService = Depends(get_sync_service),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        """Delete a source, its documents, and retire memories left without support."""
        existing = await db.get_source(source_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_management(request, existing)

        # Cancel running sync task if any
        await sync_service.cancel_source(source_id)
        release_atlassian_request_limiter(
            str(existing["config"].get("base_url") or ""),
            owner_id=source_id,
        )

        for doc in await db.list_documents(source=source_id, limit=100000):
            document_store.delete_document_files(source_name=existing["name"], title=doc.title)

        memory_store = await _build_memory_store(db, config, runtime_provider)
        await memory_store.delete_source_cascade(source_id)
        return {"ok": True, "deleted_source": source_id}

    @source_router.post("/{source_id}/sync")
    async def trigger_sync(
        request: Request,
        source_id: str,
        req: SourceSyncRequest | None = Body(default=None),
        db: Database = Depends(get_db),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """Trigger a manual sync for a source. Returns immediately; sync runs in background."""
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_management(request, source)

        try:
            await sync_service.start_source(
                source_id,
                force_full_sync=bool(req and req.force_full_sync),
            )
        except SyncAlreadyRunningError:
            raise HTTPException(status_code=409, detail="Sync already running for this source")
        except SourcePausedError:
            raise _source_paused_http_error()
        return {"ok": True, "message": "Sync started", "source_id": source_id}

    @source_router.post("/{source_id}/force-resync")
    async def trigger_force_resync(
        request: Request,
        source_id: str,
        db: Database = Depends(get_db),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """Reset incremental sync state and trigger a full source sync."""
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_management(request, source)

        if sync_service.is_running(source_id):
            raise HTTPException(status_code=409, detail="Sync already running for this source")

        await db.reset_source_sync_cursor(source_id)
        try:
            await sync_service.start_source(source_id, force_full_sync=True)
        except SyncAlreadyRunningError:
            raise HTTPException(status_code=409, detail="Sync already running for this source")
        except SourcePausedError:
            raise _source_paused_http_error()
        return {"ok": True, "message": "Force resync started", "source_id": source_id}

    @source_router.get("/{source_id}/schedule", response_model=SourceSyncScheduleResponse)
    async def get_source_sync_schedule(
        request: Request,
        source_id: str,
        db: Database = Depends(get_db),
    ):
        """Return the per-source automatic sync schedule."""
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_management(request, source)
        return SourceSyncScheduleResponse(**(source.get("sync_schedule") or {}))

    @source_router.put("/{source_id}/schedule")
    async def update_source_sync_schedule(
        request: Request,
        source_id: str,
        req: SourceSyncScheduleRequest,
        db: Database = Depends(get_db),
    ):
        """Update the per-source automatic sync schedule."""
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_management(request, source)
        await db.set_source_sync_schedule(
            source_id,
            enabled=req.enabled,
            interval_minutes=req.interval_minutes,
        )
        return {"ok": True, "source_id": source_id}

    @source_router.post("/{source_id}/adapter/documents")
    async def push_local_adapter_document(
        source_id: str,
        req: LocalAdapterDocumentRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """Receive one markdown document pushed by the local CLI adapter.

        The service owns the inbox layout and the package format. The CLI never
        writes into MemForge storage directly: it sends the normalized body and
        the service creates a stable doc id, atomically writes the package, and
        leaves the rest of ingestion to the source's sync pipeline.
        """
        from memforge.local_adapter import (
            LOCAL_MARKDOWN_SOURCE_TYPE,
            submit_local_markdown_document,
        )

        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        if source.get("type") != LOCAL_MARKDOWN_SOURCE_TYPE:
            raise HTTPException(
                status_code=400,
                detail=(f"source {source_id} is type {source.get('type')!r}, not 'local_markdown'"),
            )
        _require_source_management(request, source)
        if source.get("status") == SOURCE_PAUSED_STATUS:
            raise HTTPException(status_code=400, detail="Source is paused")

        try:
            result = await submit_local_markdown_document(
                db=db,
                config=config,
                source=source,
                vault_id=req.vault_id,
                relative_path=req.relative_path,
                markdown_body=req.markdown_body,
                content_type=req.content_type,
                title=req.title,
                raw_hash=req.raw_hash,
                submitted_by=req.submitted_by,
                submitted_at=req.submitted_at,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        sync_started = False
        if req.process_now:
            try:
                await sync_service.start_source(source_id)
                sync_started = True
            except SyncAlreadyRunningError:
                sync_started = True
            except SourcePausedError:
                raise _source_paused_http_error()

        return {**result, "sync_started": sync_started}

    # ===================================================================
    # 4b. Agent Session Document Intake
    # ===================================================================

    @agent_session_router.post("/documents")
    async def submit_agent_session_summary(
        req: AgentSessionDocumentRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """Submit a client-generated agent session summary document."""
        from memforge.agent_sessions import agent_session_source_id, submit_agent_session_document

        await _raise_if_source_paused(db, agent_session_source_id(req.client))

        try:
            result = await submit_agent_session_document(
                db=db,
                config=config,
                client=req.client,
                session_id=req.session_id,
                trigger=req.trigger,
                document_markdown=req.document_markdown,
                workspace=req.workspace,
                repo=req.repo,
                branch=req.branch,
                commit_sha=req.commit_sha,
                history_window_kind=req.history_window_kind,
                history_window_start=req.history_window_start,
                history_window_end=req.history_window_end,
                title=req.title,
                metadata=req.metadata,
                submitted_at=req.submitted_at,
                source_updated_at=req.source_updated_at,
                user_id=resolve_request_principal(request),
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        sync_started = False
        if req.process_now:
            try:
                await sync_service.start_source(result["source_id"])
                sync_started = True
            except SyncAlreadyRunningError:
                sync_started = True
            except SourcePausedError:
                raise _source_paused_http_error()

        return {
            **result,
            "sync_started": sync_started,
        }

    @agent_session_router.get("/completeness")
    async def agent_session_completeness(
        session_id: str | None = None,
        source_id: str | None = None,
        db: Database = Depends(get_db),
    ):
        """Return window outcome counts and the no_output fraction on demand."""
        return await db.summarize_agent_session_outcomes(session_id=session_id, source_id=source_id)

    @agent_session_router.post("/windows")
    async def submit_agent_session_window_summary(
        req: AgentSessionWindowRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        """Submit a client transcript window for private agent-knowledge patching."""
        from memforge.agent_sessions import agent_session_source_id, submit_agent_session_window

        if req.schema_version != "agent-session-window/v1":
            raise HTTPException(status_code=400, detail=f"unsupported schema_version: {req.schema_version}")

        await _raise_if_source_paused(db, agent_session_source_id(req.client))

        structured_client = getattr(request.app.state, "agent_session_window_client", None)
        if structured_client is None:
            structured_client = await _build_agent_session_window_client(db, config)
        memory_store = await _build_memory_store(db, config, runtime_provider)

        try:
            result = await submit_agent_session_window(
                db=db,
                config=config,
                memory_store=memory_store,
                structured_llm_client=structured_client,
                client=req.client,
                session_id=req.session_id,
                trigger=req.trigger,
                workspace=req.workspace,
                events=req.events,
                history_window=req.history_window,
                transcript_markdown=req.transcript_markdown,
                repo=req.repo,
                branch=req.branch,
                commit_sha=req.commit_sha,
                receipt=req.receipt,
                retention=req.retention,
                submitted_at=req.submitted_at,
                source_updated_at=req.source_updated_at,
                process_now=req.process_now,
                user_id=resolve_request_principal(request),
            )
        except ValueError as e:
            detail = str(e)
            if "LLM unavailable" in detail:
                raise HTTPException(status_code=503, detail=detail)
            raise HTTPException(status_code=400, detail=detail)

        return {
            **result,
            "sync_started": False,
            "sync_queued": False,
        }

    @hook_router.post("/receipts")
    async def record_agent_hook_receipt(
        req: AgentHookReceiptRequest,
        db: Database = Depends(get_db),
    ):
        """Record a coding-agent lifecycle hook without creating a source document."""
        from memforge.agent_sessions import submit_agent_hook_receipt as record_hook_receipt

        try:
            return await record_hook_receipt(
                db=db,
                client=req.client,
                session_id=req.session_id,
                hook=req.hook,
                workspace=req.workspace,
                repo=req.repo,
                branch=req.branch,
                commit_sha=req.commit_sha,
                metadata=req.metadata,
                submitted_at=req.submitted_at,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ===================================================================
    # 4c. Agent Hook Context
    # ===================================================================

    @hook_router.post("/context")
    async def build_hook_context(
        req: AgentHookContextRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
    ):
        """Return compact memory context for Codex/Claude lifecycle hooks."""
        from memforge.agent_hooks import (
            AgentHookContextRequest as HookContextRequest,
            build_agent_hook_context,
        )

        principal_user_id = resolve_request_principal(request)
        engine = await get_search_engine(request, db, config)

        hook_request = HookContextRequest(
            client=req.client,
            hook=req.hook,
            workspace=req.workspace,
            repo=req.repo,
            branch=req.branch,
            prompt=req.prompt,
            touched_files=req.touched_files,
            max_memories=req.max_memories,
            include_recent_changes=req.include_recent_changes,
        )
        return await build_agent_hook_context(
            db,
            hook_request,
            principal_user_id=principal_user_id,
            search_engine=engine,
        )

    # ===================================================================
    # 5. Schedule Endpoints
    # ===================================================================

    @schedule_router.get("")
    async def get_schedule(db: Database = Depends(get_db)):
        """Get the current sync schedule configuration."""
        sched = await db.get_schedule_config()
        return ScheduleConfigResponse(**sched)

    @schedule_router.put("")
    async def update_schedule(
        req: ScheduleConfigRequest,
        db: Database = Depends(get_db),
        sync_scheduler: SyncScheduler | None = Depends(get_sync_scheduler),
    ):
        """Update the sync schedule configuration."""
        await db.set_schedule_config(
            {
                "enabled": req.enabled,
                "frequency": req.frequency,
                "time": req.time,
                "day_of_week": req.day_of_week,
                "timezone": req.timezone,
            }
        )
        if sync_scheduler:
            await sync_scheduler.reload()
        return {"ok": True}

    # ===================================================================
    # 6. LLM Config Endpoints
    # ===================================================================

    @llm_router.get("")
    async def get_llm_config(db: Database = Depends(get_db)):
        """Get LLM configuration. API keys are masked in the response."""
        cfg = await db.get_llm_config()
        enrichment_key = cfg.get("enrichment_api_key")
        embedding_key = cfg.get("embedding_api_key")
        return LlmConfigResponse(
            enrichment_model=cfg.get("enrichment_model"),
            enrichment_base_url=cfg.get("enrichment_base_url"),
            enrichment_api_key=_mask_api_key(enrichment_key),
            enrichment_api_key_set=bool(enrichment_key),
            enrichment_api_key_last4=_api_key_last4(enrichment_key),
            embedding_model=cfg.get("embedding_model"),
            embedding_base_url=cfg.get("embedding_base_url"),
            embedding_api_key=_mask_api_key(embedding_key),
            embedding_api_key_set=bool(embedding_key),
            embedding_api_key_last4=_api_key_last4(embedding_key),
        )

    @llm_router.post("/probe")
    async def probe_llm_config(
        req: LlmConfigProbeRequest,
        db: Database = Depends(get_db),
    ):
        """Test an LLM endpoint and fetch model ids when the endpoint supports it."""
        current = await db.get_llm_config()
        api_key = req.api_key
        if api_key is None:
            api_key = current.get(f"{req.kind}_api_key")
        return await _probe_llm_models(base_url=req.base_url, api_key=api_key or None)

    @llm_router.put("")
    async def update_llm_config(
        req: LlmConfigRequest,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
    ):
        """Update LLM configuration."""
        if not config.server.llm_config_writable:
            raise HTTPException(
                status_code=405,
                detail="LLM settings are managed by the deployment environment",
            )

        # Fetch current config to preserve masked keys
        current = await db.get_llm_config()
        fields_set = req.model_fields_set

        def _resolve_value(field: str) -> str | None:
            if field not in fields_set:
                return current.get(field)
            new_val = getattr(req, field)
            if new_val is None:
                return current.get(field)
            return new_val.strip() or None

        def _resolve_key(field: str) -> str | None:
            if field not in fields_set:
                return current.get(field)
            new_val = getattr(req, field)
            if new_val is None:
                return current.get(field)
            if new_val == "":
                return None
            if new_val.startswith("*"):
                # Masked values represent an existing secret, not a replacement.
                return current.get(field)
            return new_val.strip() or None

        await db.set_llm_config(
            {
                "enrichment_model": _resolve_value("enrichment_model"),
                "enrichment_base_url": _resolve_value("enrichment_base_url"),
                "enrichment_api_key": _resolve_key("enrichment_api_key"),
                "embedding_model": _resolve_value("embedding_model"),
                "embedding_base_url": _resolve_value("embedding_base_url"),
                "embedding_api_key": _resolve_key("embedding_api_key"),
            }
        )
        return {"ok": True}

    # ===================================================================
    # Projects
    # ===================================================================

    @projects_router.get("", response_model=list[ProjectResponse])
    async def list_projects_route(
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        adapters = await _build_project_adapters(db, config, runtime_provider)
        rows = await adapters.relational.list_projects()
        return [_project_to_response(p) for p in rows]

    @projects_router.post("", response_model=ProjectResponse, status_code=201)
    async def create_project_route(
        req: ProjectCreateRequest,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        adapters = await _build_project_adapters(db, config, runtime_provider)
        key = (req.key or _derive_project_key(req.name)).strip()
        if not key:
            raise HTTPException(status_code=400, detail="project key cannot be empty")
        try:
            created = await adapters.relational.create_project(
                key=key,
                name=req.name,
                is_shared=(req.kind == "shared"),
            )
        except ValueError:
            raise HTTPException(
                status_code=409,
                detail=f"project key {key!r} already exists",
            )
        return _project_to_response(created)

    @projects_router.patch("/{project_id}", response_model=ProjectResponse)
    async def update_project_route(
        project_id: str,
        req: ProjectUpdateRequest,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        adapters = await _build_project_adapters(db, config, runtime_provider)
        existing = await adapters.relational.get_project(project_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="project not found")
        is_shared: bool | None = None
        if req.kind is not None:
            is_shared = req.kind == "shared"
        updated = await adapters.relational.update_project(
            project_id,
            name=req.name,
            is_shared=is_shared,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="project not found")
        return _project_to_response(updated)

    @projects_router.delete("/{project_id}", response_model=ProjectDeleteResponse)
    async def delete_project_route(
        project_id: str,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        adapters = await _build_project_adapters(db, config, runtime_provider)
        try:
            affected = await adapters.relational.list_project_memory_ids(project_id)
        except LookupError:
            raise HTTPException(status_code=404, detail="project not found")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        memory_store = await _build_memory_store(db, config, runtime_provider)
        # Vector metadata moves first so a failure here aborts the
        # transaction with both stores still pointing at the original
        # project. Only after the vector channel reports success do we
        # commit the relational rebucket and drop the project row.
        await memory_store.rebucket_project_memories(
            affected,
            UNSORTED_PROJECT_KEY,
        )
        await adapters.relational.commit_project_deletion(project_id, affected)
        return ProjectDeleteResponse(
            id=project_id,
            rebucketed_count=len(affected),
            rebucketed_memory_ids=affected,
        )

    # ===================================================================
    # Memory Reviews
    # ===================================================================

    @review_router.get("", response_model=MemoryReviewListResponse)
    async def list_memory_reviews(
        status: str | None = "open",
        kind: str | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Database = Depends(get_db),
    ):
        """List memory reviews. Defaults to open items that still need attention."""
        normalized_status = status if status and status != "all" else None
        reviews = await db.list_memory_reviews(
            status=normalized_status,
            kind=kind,
            limit=limit,
            offset=offset,
        )

        responses: list[MemoryReviewResponse] = []
        for review in reviews:
            incumbent = await db.get_memory(review.incumbent_memory_id)
            challenger = await db.get_memory(review.challenger_memory_id)
            responses.append(_review_to_response(review, incumbent=incumbent, challenger=challenger))

        total = await db.count_memory_reviews(status=normalized_status, kind=kind)
        return MemoryReviewListResponse(data=responses, total=total)

    @review_router.get("/{review_id}", response_model=MemoryReviewDetailResponse)
    async def get_memory_review(
        review_id: str,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        review = await db.get_memory_review(review_id)
        if review is None:
            raise HTTPException(status_code=404, detail="Review not found")

        incumbent = await db.get_memory(review.incumbent_memory_id)
        challenger = await db.get_memory(review.challenger_memory_id)
        base = _review_to_response(review, incumbent=incumbent, challenger=challenger)
        incumbent_summary = await _build_memory_summary(db, incumbent, config, artifact_store) if incumbent else None
        challenger_summary = await _build_memory_summary(db, challenger, config, artifact_store) if challenger else None
        related_challengers: list[MemoryReviewMemorySummary] = []
        for related in await db.list_memory_review_related_challengers(review.id):
            related_memory = await db.get_memory(related.challenger_memory_id)
            if related_memory is None:
                continue
            related_challengers.append(await _build_memory_summary(db, related_memory, config, artifact_store))
        return MemoryReviewDetailResponse(
            **base.model_dump(),
            incumbent=incumbent_summary,
            challenger=challenger_summary,
            related_challengers=related_challengers,
        )

    @review_router.post("/{review_id}/approve", response_model=MemoryReviewDetailResponse)
    async def approve_memory_review(
        review_id: str,
        req: MemoryReviewDecisionRequest = MemoryReviewDecisionRequest(),
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        service = await _build_review_service(db, config, runtime_provider)
        try:
            result = await service.approve(
                review_id,
                reviewer=req.reviewer,
                note=req.note,
            )
        except ReviewNotFound:
            raise HTTPException(status_code=404, detail="Review not found")
        except ReviewAlreadyResolved as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ReviewStaleConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "stale",
                    "message": str(exc),
                    "review_id": exc.review.id,
                },
            )
        except ReviewKindUnsupported as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except ReviewError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        review = result.review or await db.get_memory_review(review_id)
        assert review is not None
        base = _review_to_response(review, incumbent=result.incumbent, challenger=result.challenger)
        incumbent_summary = (
            await _build_memory_summary(db, result.incumbent, config, artifact_store) if result.incumbent else None
        )
        challenger_summary = (
            await _build_memory_summary(db, result.challenger, config, artifact_store) if result.challenger else None
        )
        return MemoryReviewDetailResponse(
            **base.model_dump(),
            incumbent=incumbent_summary,
            challenger=challenger_summary,
        )

    @review_router.post("/{review_id}/reject", response_model=MemoryReviewDetailResponse)
    async def reject_memory_review(
        review_id: str,
        req: MemoryReviewDecisionRequest,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        if not req.note or not req.note.strip():
            raise HTTPException(status_code=400, detail="A note is required to reject a review")

        service = await _build_review_service(db, config, runtime_provider)
        try:
            result = await service.reject(
                review_id,
                reviewer=req.reviewer,
                note=req.note,
            )
        except ReviewNotFound:
            raise HTTPException(status_code=404, detail="Review not found")
        except ReviewAlreadyResolved as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ReviewStaleConflict as exc:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "stale",
                    "message": str(exc),
                    "review_id": exc.review.id,
                },
            )
        except ReviewError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        review = result.review or await db.get_memory_review(review_id)
        assert review is not None
        base = _review_to_response(review, incumbent=result.incumbent, challenger=result.challenger)
        incumbent_summary = (
            await _build_memory_summary(db, result.incumbent, config, artifact_store) if result.incumbent else None
        )
        challenger_summary = (
            await _build_memory_summary(db, result.challenger, config, artifact_store) if result.challenger else None
        )
        return MemoryReviewDetailResponse(
            **base.model_dump(),
            incumbent=incumbent_summary,
            challenger=challenger_summary,
        )

    @review_router.post("/{review_id}/refresh", response_model=MemoryReviewDetailResponse)
    async def refresh_memory_review(
        review_id: str,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        service = await _build_review_service(db, config, runtime_provider)
        try:
            result = await service.refresh(review_id)
        except ReviewNotFound:
            raise HTTPException(status_code=404, detail="Review not found")
        except ReviewError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        review = result.review
        base = _review_to_response(review, incumbent=result.incumbent, challenger=result.challenger)
        incumbent_summary = (
            await _build_memory_summary(db, result.incumbent, config, artifact_store) if result.incumbent else None
        )
        challenger_summary = (
            await _build_memory_summary(db, result.challenger, config, artifact_store) if result.challenger else None
        )
        return MemoryReviewDetailResponse(
            **base.model_dump(),
            incumbent=incumbent_summary,
            challenger=challenger_summary,
        )

    # -- Include all routers --
    # Local tool — no auth required. All routers accessible directly.
    app.include_router(auth_router)
    app.include_router(health_router)
    app.include_router(document_router)
    app.include_router(memory_router)
    app.include_router(review_router)
    app.include_router(entity_router)
    app.include_router(gene_router)
    app.include_router(source_router)
    app.include_router(agent_session_router)
    app.include_router(hook_router)
    app.include_router(recent_change_router)
    app.include_router(schedule_router)
    app.include_router(llm_router)
    app.include_router(projects_router)

    # -- Detailed stats endpoint (observability) --

    @app.get("/api/stats/detailed")
    async def detailed_stats(db: Database = Depends(get_db)):
        """Detailed system stats for observability — entity resolution, cache, memory growth."""
        stats: dict[str, Any] = {}

        # Memory counts
        async with db.db.execute("SELECT COUNT(*) FROM memories WHERE status = 'active'") as cur:
            row = await cur.fetchone()
            stats["active_memories"] = row[0] if row else 0

        async with db.db.execute("SELECT COUNT(*) FROM entities") as cur:
            row = await cur.fetchone()
            stats["total_entities"] = row[0] if row else 0

        async with db.db.execute("SELECT COUNT(*) FROM entity_aliases") as cur:
            row = await cur.fetchone()
            stats["total_aliases"] = row[0] if row else 0

        # Alias sources breakdown
        alias_sources: dict[str, int] = {}
        async with db.db.execute("SELECT source, COUNT(*) FROM entity_aliases GROUP BY source") as cur:
            async for row in cur:
                alias_sources[row[0]] = row[1]
        stats["alias_sources"] = alias_sources

        # Memory growth (last 7 days)
        async with db.db.execute(
            "SELECT date(created_at) as day, COUNT(*) FROM memories "
            "WHERE created_at >= datetime('now', '-7 days') "
            "GROUP BY day ORDER BY day"
        ) as cur:
            growth = {}
            async for row in cur:
                growth[row[0]] = row[1]
        stats["memory_growth_7d"] = growth

        # Entity resolution stats (if available on app state)
        if hasattr(app.state, "entity_resolver_stats"):
            stats["entity_resolution"] = app.state.entity_resolver_stats

        return stats

    return app


# ---------------------------------------------------------------------------
# JSON error helper
# ---------------------------------------------------------------------------


def _json_error(status_code: int, message: str):
    """Return a JSONResponse with an error body."""
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=status_code,
        content={"error": message, "status_code": status_code},
    )
