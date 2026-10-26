"""Admin REST API for MemForge.

FastAPI application providing management endpoints for memories, entities,
sources/genes, schedule, model settings, and system health/stats.
"""

from __future__ import annotations

import asyncio
import copy
from dataclasses import asdict, dataclass, is_dataclass
import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
import tempfile
import time
import unicodedata
import uuid
from collections.abc import Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, NoReturn, cast
from urllib.parse import quote

from fastapi import APIRouter, BackgroundTasks, Body, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.routing import APIRoute
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from memforge.api_target import CAPABILITY_PATH, Edition, capability_document
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic.json_schema import models_json_schema

from memforge.config import DEFAULT_SEARCH_TOP_K, AppConfig
from memforge.auth import browser_session
from memforge.auth.jira_auth import (
    JiraAuthSessionError,
    JiraAuthSessionService,
    JIRA_PRINCIPAL_CHANGED_CODE,
    JiraPrincipalChangedError,
    canonical_jira_origin,
    effective_jira_auth_mode,
)
from memforge.evals.agent_evaluation import (
    AgentAssessmentAnnotatorKind,
    AgentAssessmentConfidence,
    AgentAssessmentLabel,
    AgentAssessmentStatus,
    AgentIssueLabel,
    OnlineEvaluationCoveragePolicy,
    SourceEvaluationStatus,
)
from memforge.conversations.collections import ArtifactLink, CollectedBody, StoredCollections
from memforge.conversations.intake import CollectionIntake, CollectionRejected
from memforge.conversations.manifest import ManifestFormatError, request_to_payload
from memforge.conversations.reader import SourceProjectionUnitReader
from memforge.conversations.records import MessageIdentity, canonical_record_json
from memforge.conversations.report import plan_report
from memforge.source_artifact_inputs import retain_artifact_input
from memforge.genes import (
    GENE_REGISTRY,
    DocumentGene,
    create_gene,
    list_available_genes,
    source_lists_scope,
    source_rediscovers_documents,
    source_type_supports_sync,
)
from memforge.genes.conversation_gene import CollectionAttempt, ConversationGene
from memforge.genes.family import CollectionFamilyKind
from memforge.genes.atlassian_auth import release_atlassian_request_limiter
from memforge.genes.base import SourceConfigurationError
from memforge.genes.github_repo_gene import GitHubRepoGene
from memforge.genes.toolkit.config import (
    FieldIssue,
    SourceConfigInvalid,
    require_https_base_url,
    validate_tls_ca_bundle,
)
from memforge.genes.toolkit.github_api import BROWSE_TREE_REQUEST_BUDGET, GitHubTreeBudgetExceeded
from memforge.github_repo_utils import github_extension_allowed, github_include_extensions
from memforge.memory.audit import AuditContext
from memforge.memory.cross_document_relation import (
    MAX_RELATIONS_PER_MEMORY,
    CrossDocumentRelationLabel,
    RelationDismissalConflict,
)
from memforge.memory.cross_source_conflict_reviews import ReviewLabelOverrideError
from memforge.memory.cross_source_review_conversion import (
    CrossSourceReviewConversionConflict,
    CrossSourceReviewConversionReceipt,
    apply_conversion,
    build_conversion_plan,
    delete_converted_reviews,
)
from memforge.memory.relation_discovery import (
    DEFAULT_RELATION_DISCOVERY_BUDGET,
    rerun_relation_discovery_failed_by_another_release,
)
from memforge.memory.relation_discovery_contract import (
    RelationDiscoveryWork,
    RelationDiscoveryWorkSelection,
    RelationDiscoveryWorkState,
)
from memforge.memory.cross_document_relation_reader import (
    dismiss_relation,
    dismissed_relation_views,
    list_relation_pairs,
    read_memory_relations,
    relation_notice,
)
from memforge.memory.lifecycle import normalize_memory_status
from memforge.memory.lifecycle_plan import (
    LifecycleMutationType,
    LifecyclePlan,
    LifecycleReview,
    LifecycleReviewStatus,
    lifecycle_review_candidate_memory_id,
)
from memforge.memory.lifecycle_service import (
    MaintenanceClosureEntry,
    MemoryLifecycleConflict,
    MemoryLifecycleNotFound,
    MemoryLifecycleService,
    is_source_backed,
)
from memforge.memory.review_service import (
    ReviewAlreadyResolved,
    ReviewError,
    ReviewKindUnsupported,
    ReviewNotFound,
    ReviewService,
    ReviewStaleConflict,
)
from memforge.memory.review_presentation import (
    ReviewPresentation,
    present_lifecycle_review,
    present_memory_review,
)
from memforge.memory.review_decision import (
    lifecycle_review_decision_fingerprint,
    memory_review_decision_fingerprint,
)
from memforge.model_settings import ModelSettings
from memforge.llm.embedding import EmbeddingIndexMismatch
from memforge.vector_index_errors import (
    REINDEX_IN_PROGRESS,
    REINDEX_IN_PROGRESS_MESSAGE,
    NoVectorIndex,
    ReindexInProgress,
    is_reindex_refusal,
)
from memforge.memory.store import MemoryStore
from memforge.models import (
    AbsenceCheckSkippedReason,
    ConfigCondition,
    ConfigGroupPurpose,
    ScopeListingKind,
    ConfigFieldType,
    Entity,
    Memory,
    MemoryRelationContext,
    MemorySourceRef,
    MemoryStatus,
    MemoryType,
    MemoryReview,
    Project,
    RESERVED_PROJECT_KEYS,
    ReviewKind,
    SourceSyncInput,
    SourceUnitInput,
    SourceExecutionKind,
    UNSORTED_PROJECT_KEY,
    VIRTUAL_DOCUMENT_SOURCE_IDS,
    Visibility,
    canonicalize_entity_name,
    comparison_string,
)
from memforge.sync_progress import (
    SyncProgressPhase,
    SyncProgressUnit,
    normalize_sync_progress_snapshot,
)
from memforge.source_artifacts import (
    MAX_SOURCE_ARTIFACT_STORAGE_BYTES,
    SOURCE_ARTIFACT_SPOOL_MEMORY_BYTES,
    SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES,
    normalize_source_artifact_media_type,
    parse_source_artifact_content_length,
)
from memforge.provenance import (
    DocumentArtifactStore,
    document_resource_path,
    list_stored_input_artifacts,
    select_stored_input_artifact,
    source_unit_content_url,
    source_unit_pdf_url,
    source_unit_resource_path,
)
from memforge.retrieval.filters import MemorySourceFilter, MemoryTimeRange
from memforge.retrieval.intents import RankedRetrievalIntent
from memforge.runtime import (
    DefaultRuntimeProvider,
    RuntimeHealthComponent,
    RuntimeProvider,
    SourceNotActiveError,
    SourceSyncBoundaryError,
    SourceSyncWorker,
    SourceSyncUnsupportedError,
    SyncService,
    SourcePausedError,
)
from memforge.pipeline.collection_family import collection_family
from memforge.pipeline.source_projection_adapters import DEFAULT_SOURCE_PROJECTION_ADAPTER
from memforge.pipeline.stored_document import reprocess_preview
from memforge.scheduler import SyncScheduler
from memforge.server.model_settings_api import (
    ModelHealthResponse,
    build_model_settings_router,
    listed_native_schema,
    model_health_response,
)
from memforge.source_secrets import (
    decrypt_source_config_for_runtime,
    SecretConfigurationError,
    prepare_source_config_for_storage,
    redact_source_config,
    source_secret_fields,
)
from memforge.source_projection import (
    ProjectionScopeTransition,
    ProjectionScopeTransitionStale,
    provider_item_key,
)
from memforge.source_projection_config import (
    canonical_scope_transition,
    excluded_scope_config,
    projection_scope_transition_id,
    source_access_context,
)
from memforge.source_types import (
    LocalAgentCollection,
    SourceConfigPolicy,
    load_reference,
    source_type_descriptor,
)
from memforge.source_types.form import check_config, effective_value
from memforge.source_types.jira import jira_auth_mode
from memforge.source_activity import SourceActivityConflict, SourceActivityKind, SourceSyncRunActive
from memforge.storage.document_store import LocalDocumentStore
from memforge.storage.recovery_points import RecoveryPoints, UpgradeError
from memforge.storage.source_cleanup import SourceArtifactCleanupService
from memforge.server.review_admin_service import (
    QueuedMemoryReview, QueuedLifecycleReview, ReviewQueue,
    is_memory_review_stale, read_open_review_ids, read_review_queue, review_access_scope, review_is_visible,
)
from memforge.server.memory_admin_service import (
    count_project_memories,
    list_memory_admin_page,
    pick_origin_source_type,
)
from memforge.server.source_admin_service import (
    ScopeChangeRemoval,
    can_manage_source,
    can_manage_workspace,
    image_documents_by_source,
    list_source_admin_rows,
    normalize_workspace_role,
    scope_change_removes,
)
from memforge.source_access import source_is_discoverable
from memforge.source_access_transition import (
    SourceAccessTransitionError,
    SourceAccessTransitionService,
)
from memforge.local_agent.readiness import (
    SourceConnectionReason,
    SourceConnectionState,
    connection_status_from_browser_session,
)
from memforge.local_agent.setup_results import SETUP_RESULT_MODELS
from memforge.local_agent.snapshot_manifest import (
    COMPLETE_SNAPSHOT,
    SNAPSHOT_MANIFEST_MAX_ITEMS,
    UPSERT,
    ExcludedScopeListing,
    SnapshotConfirmation,
    SnapshotConfirmationAnswer,
    SnapshotManifestItem,
    plan_snapshot_manifest,
    snapshot_manifest_sha256,
    validate_excluded_listings_for_job,
)
from memforge.local_agent.source_contract import (
    LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD,
    LOCAL_AGENT_SYNC_OPERATIONS,
    execution_owner_user_id,
    is_local_agent_backed_source,
    local_agent_collection_attempt_id,
    local_agent_semantic_input_sha256,
    local_agent_completion_status,
    local_agent_source_config_revision,
    local_agent_sync_job_payload,
    local_agent_sync_operation,
    local_agent_sync_snapshot_id,
    local_package_confirms_absence,
    local_package_requires_source_time,
    validate_local_agent_replay_package,
)
from memforge.storage.adapters.protocols import RelationalStore
from memforge.storage.admin_memory import MemoryAdminListFilters
from memforge.storage.admin_source import (
    SOURCE_SYNC_SCHEDULE_DEFAULT_INTERVAL_MINUTES,
    SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES,
    SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES,
    SourceListSortMode,
)
from memforge.storage.database import Database

logger = logging.getLogger(__name__)

# How long to wait before trying again to empty a write-ahead log a reader holds.
_KEY_ERASURE_RETRY_S = 30.0
# How long a search engine whose embedding model is not the vector index's is
# kept before it is built again for the model then in use.
_SEARCH_ENGINE_REBUILD_S = 30.0


async def _run_source_access_background(
    operation_id: str,
    action: Callable[[str], Awaitable[dict[str, Any]]],
) -> None:
    """Run a durable access command without leaking its recorded failure to ASGI."""

    try:
        await action(operation_id)
    except Exception:
        logger.exception("Background Source access operation failed: %s", operation_id)


SEARCH_CLIENTS = ("claude-code", "codex")

SOURCE_ACTIVE_STATUS = "active"
SOURCE_PAUSED_STATUS = "paused"
LOCAL_AGENT_STATUS_STALE_SECONDS = 90
LOCAL_AGENT_SETUP_OPERATION_SOURCE_TYPES = {
    "github_repo_preview_tree": "github_repo",
    "jira_auth": "jira",
    "local_markdown_pick_root": "local_markdown",
    "local_markdown_preview_tree": "local_markdown",
    "teams_auth": "teams",
    "teams_auth_check": "teams",
    "teams_browse": "teams",
}
# Source lifecycle is deliberately closed for now. A paused source keeps its
# configuration and existing memories but cannot accept new sync work.
SOURCE_STATUSES = {SOURCE_ACTIVE_STATUS, SOURCE_PAUSED_STATUS}
# What admin reads show unless the caller asks for another lifecycle status.
ACTIVE_MEMORY_STATUSES = (MemoryStatus.ACTIVE.value,)
MAX_REVIEW_PAGE_SIZE = 500
"""The most Reviews one page of the Review queue returns."""
INCLUDE_PRIVATE_DESCRIPTION = (
    "Include the caller's own private Memories. Other users' private Memories are never returned."
)
"""What ``include_private`` means on every Memory read that accepts it."""


def _workspace_default_scope(
    request: Request,
    *,
    include_private: bool,
    allowed_statuses: tuple[str, ...] = ACTIVE_MEMORY_STATUSES,
):
    """Build the workspace-default AccessScope for an admin-API HTTP read.

    The caller's identity is server-derived (``resolve_principal(request)``);
    active Memories are read unless the caller names other statuses; the
    caller's own private rows surface when ``include_private`` is set, and
    other users' private rows never do. Per-id
    and list readers do not receive a request-time `scope_mode`, so the scope
    mode is fixed at ``project-first``: cross-project rows stay visible and
    the ranker handles the affinity weighting.
    """
    from memforge.storage.adapters.context import AccessScope

    return AccessScope(
        user_id=resolve_request_principal(request),
        include_private=include_private,
        allowed_statuses=allowed_statuses,
        active_project=None,
        scope_mode="project-first",
    )


def _lifecycle_visibility_scope(request: Request):
    """Reviews and purges use caller access independently of lifecycle page filters."""
    return review_access_scope(resolve_request_principal(request))


def _inline_content_disposition(filename: str) -> str:
    """Return an RFC 8187 filename header with an ASCII-safe fallback."""

    basename = filename.replace("\\", "/").rsplit("/", 1)[-1]
    basename = "".join(
        character for character in basename if character not in {"\r", "\n", "\x7f"} and ord(character) >= 0x20
    )
    if not basename:
        basename = "artifact"
    suffix = Path(basename).suffix
    stem = basename[: -len(suffix)] if suffix else basename
    ascii_stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    ascii_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", ascii_stem).strip("_-")
    ascii_suffix = unicodedata.normalize("NFKD", suffix).encode("ascii", "ignore").decode()
    ascii_suffix = re.sub(r"[^A-Za-z0-9.]+", "", ascii_suffix)
    ascii_fallback = f"{ascii_stem or 'artifact'}{ascii_suffix}"
    encoded = quote(basename, safe="!#$&+-.^_`|~")
    return f"inline; filename=\"{ascii_fallback}\"; filename*=UTF-8''{encoded}"


def _verified_artifact_chunks(
    artifact_store: DocumentArtifactStore,
    uri: str,
    *,
    expected_size: int,
    expected_sha256: str,
):
    """Yield one stored Artifact while verifying its immutable revision."""

    digest = hashlib.sha256()
    observed_size = 0
    with artifact_store.open_artifact(uri) as body:
        while chunk := body.read(256 * 1024):
            observed_size += len(chunk)
            digest.update(chunk)
            yield chunk
    if observed_size != expected_size or digest.hexdigest() != expected_sha256:
        raise RuntimeError("Source Artifact integrity mismatch during streaming")


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


async def _run_embedded_service_workers(
    sync_worker: SourceSyncWorker,
    evaluation_worker: object | None,
    *,
    poll_seconds: float,
) -> None:
    """Prioritize source lifecycle work, then claim at most one offline run."""

    interval = max(0.1, poll_seconds)
    while True:
        source_run = await sync_worker.run_once()
        if source_run is not None:
            await sync_worker.process_bounded_maintenance_once()
            await asyncio.sleep(0)
            continue
        execution = None
        if evaluation_worker is not None:
            execution = await evaluation_worker.run_once()
        await asyncio.sleep(0 if execution is not None else interval)


def _source_management_forbidden() -> HTTPException:
    return HTTPException(
        status_code=403,
        detail={
            "error": "source_management_forbidden",
            "message": "Only the source owner or a workspace admin can manage this source.",
        },
    )


def _require_source_management(request: Request, source: dict[str, Any]) -> None:
    _require_source_discoverability(request, source)
    if not can_manage_source(
        source,
        viewer_id=resolve_request_principal(request),
        viewer_role=resolve_request_workspace_role(request),
    ):
        raise _source_management_forbidden()


@dataclass(frozen=True)
class _RequestCorrectionAuthority:
    actor_user_id: str
    workspace_role: str

    def can_manage_source(self, source: Mapping[str, Any]) -> bool:
        return can_manage_source(
            dict(source),
            viewer_id=self.actor_user_id,
            viewer_role=self.workspace_role,
        )

    def can_manage_workspace_memory(self) -> bool:
        return can_manage_workspace(self.workspace_role)


def _request_correction_authority(request: Request) -> _RequestCorrectionAuthority:
    return _RequestCorrectionAuthority(
        actor_user_id=resolve_request_principal(request),
        workspace_role=resolve_request_workspace_role(request),
    )


def _require_source_discoverability(request: Request, source: dict[str, Any]) -> None:
    if not source_is_discoverable(
        source,
        viewer_id=resolve_request_principal(request),
    ):
        raise HTTPException(status_code=404, detail="Source not found")


async def _require_visible_entity(
    request: Request,
    db: Database,
    entity_id: int,
    *,
    missing_detail: str = "Entity not found",
) -> tuple[Entity, int]:
    """Return the Entity and its visible linked Memory count, or 404 when the caller cannot discover it."""
    entity = await db.get_entity(entity_id)
    linked_count = await db.count_memories_for_entity(
        entity_id,
        scope=_workspace_default_scope(request, include_private=True),
    )
    if entity is None or linked_count == 0:
        raise HTTPException(status_code=404, detail=missing_detail)
    return entity, linked_count


def _require_entity_curation(request: Request) -> None:
    """Manual aliases and merges are workspace-wide, so only workspace administrators curate them."""
    if not can_manage_workspace(resolve_request_workspace_role(request)):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "entity_curation_forbidden",
                "message": "Only a workspace admin can change entity aliases or merge entities.",
            },
        )


def _require_local_agent_connection_management(
    request: Request,
    source: dict[str, Any],
) -> Mapping[str, Any]:
    if not is_local_agent_backed_source(source):
        return
    requester_id = resolve_request_principal(request)
    expected_owner_id = execution_owner_user_id(source)
    if expected_owner_id == requester_id:
        return
    logger.warning(
        "Rejected local source connection update from non-owner",
        extra={
            "source_id": str(source.get("id") or ""),
            "requester_user_id": requester_id,
            "execution_owner_user_id": expected_owner_id,
        },
    )
    raise HTTPException(
        status_code=403,
        detail={
            "error": "local_agent_source_connection_owner_forbidden",
            "message": "Only the local execution owner can change this source connection.",
        },
    )


def _require_source_sync_execution(
    request: Request,
    source: dict[str, Any],
) -> None:
    _require_source_discoverability(request, source)
    if not is_local_agent_backed_source(source):
        _require_source_management(request, source)
        return
    requester_id = resolve_request_principal(request)
    expected_owner_id = execution_owner_user_id(source)
    if expected_owner_id is None:
        raise HTTPException(
            status_code=409,
            detail="local_agent_sync_execution_owner_required",
        )
    if expected_owner_id == requester_id:
        return
    logger.warning(
        "Rejected local source execution from non-owner",
        extra={
            "source_id": str(source.get("id") or ""),
            "requester_user_id": requester_id,
            "execution_owner_user_id": expected_owner_id,
        },
    )
    raise HTTPException(
        status_code=403,
        detail="local_agent_sync_execution_owner_forbidden",
    )


def _require_source_sync_support(source: Mapping[str, Any]) -> None:
    if not source_type_supports_sync(str(source.get("type") or "")):
        raise HTTPException(status_code=409, detail="source_sync_not_supported")


async def _require_current_local_agent_lease(
    request: Request,
    db: Database,
    *,
    source: Mapping[str, Any],
    job_id: str | None,
    attempt_count: int | None,
) -> None:
    source_id = str(source.get("id") or "").strip()
    expected_config_revision = local_agent_source_config_revision(source)
    normalized_job_id = str(job_id or "").strip()
    if not normalized_job_id or attempt_count is None:
        raise HTTPException(status_code=409, detail="local_agent_lease_context_required")
    validator = getattr(request.app.state, "local_agent_lease_validator", None)
    job_payload: Mapping[str, Any] = {}
    if validator is not None:
        validation = await validator(
            request,
            source_id,
            normalized_job_id,
            int(attempt_count or 0),
            expected_config_revision,
        )
        valid = bool(validation)
        if isinstance(validation, Mapping):
            job_payload = validation
    else:
        job = await db.get_local_agent_job(normalized_job_id)
        if job:
            job_payload = job.get("payload") or {}
        leased_until = None
        if job and job.get("leased_until"):
            try:
                leased_until = datetime.fromisoformat(str(job["leased_until"]).replace("Z", "+00:00"))
            except ValueError:
                leased_until = None
        valid = bool(
            job
            and job.get("status") == "leased"
            and job.get("source_id") == source_id
            and job.get("lease_owner_user_id") == resolve_request_principal(request)
            and int(job.get("attempt_count") or 0) == attempt_count
            and (job.get("payload") or {}).get("source_config_revision") == expected_config_revision
            and leased_until is not None
            and leased_until > datetime.now(timezone.utc)
        )
    if not valid:
        raise HTTPException(status_code=409, detail="local_agent_lease_not_current")
    return {
        **job_payload,
        "source_config_revision": expected_config_revision,
    }


async def _filter_visible_ids(db: Database, ids, scope) -> set[str]:
    """Return the subset of ``ids`` the caller may see under ``scope``."""
    if not hasattr(db, "db") and hasattr(db, "filter_visible_ids"):
        return await db.filter_visible_ids(list(ids), scope)

    from memforge.storage.adapters.sqlite.relational import SqliteRelationalStore

    return await SqliteRelationalStore(db).filter_visible_ids(list(ids), scope)


async def _require_visible_relation_pair(
    db: Database,
    memory_id: str,
    counterpart_memory_id: str,
    scope,
) -> None:
    """Anyone who can see both Memories may act on their relation; others see no pair."""

    if memory_id == counterpart_memory_id:
        raise HTTPException(status_code=404, detail="Relation not found")
    visible = await _filter_visible_ids(db, [memory_id, counterpart_memory_id], scope)
    if {memory_id, counterpart_memory_id} - visible:
        raise HTTPException(status_code=404, detail="Relation not found")


async def _review_source_ids(
    db: Database,
    memories: list[Memory],
    *,
    cache: dict[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    source_ids: list[str] = []
    for memory in memories:
        memory_source_ids = cache.get(memory.id) if cache is not None else None
        if memory_source_ids is None:
            memory_source_ids = tuple(
                source.source_id
                for source in await db.get_memory_sources(memory.id)
                if source.source_id and source.source_id not in VIRTUAL_DOCUMENT_SOURCE_IDS
            )
            if cache is not None:
                cache[memory.id] = memory_source_ids
        source_ids.extend(memory_source_ids)
    return tuple(dict.fromkeys(source_ids))


async def _require_memory_review_visibility(
    request: Request,
    db: Database,
    review: MemoryReview,
) -> tuple[Memory, Memory, tuple[Memory, ...]]:
    incumbent = await db.get_memory(review.incumbent_memory_id)
    challenger = await db.get_memory(review.challenger_memory_id)
    if incumbent is None or challenger is None:
        raise HTTPException(status_code=404, detail="Review not found")
    related_memories: list[Memory] = []
    for related in await db.list_memory_review_related_challengers(review.id):
        memory = await db.get_memory(related.challenger_memory_id)
        if memory is None:
            raise HTTPException(status_code=409, detail="Review participant is unavailable")
        related_memories.append(memory)
    participant_ids = {incumbent.id, challenger.id, *(memory.id for memory in related_memories)}
    visible = await _filter_visible_ids(
        db,
        participant_ids,
        _lifecycle_visibility_scope(request),
    )
    sources = await _memory_review_sources(db, [incumbent, challenger, *related_memories])
    if not review_is_visible(tuple(participant_ids), visible, sources, scope=_lifecycle_visibility_scope(request)):
        raise HTTPException(status_code=404, detail="Review not found")
    return incumbent, challenger, tuple(related_memories)


async def _memory_review_sources(
    db: Database,
    participants: Iterable[Memory],
) -> list[dict[str, Any] | None]:
    """Each Source behind the Review's participants, or None where the Source no longer exists."""
    return [await db.get_source(source_id) for source_id in await _review_source_ids(db, list(participants))]


def _can_decide_review(request: Request, sources: Iterable[Mapping[str, Any] | None]) -> bool:
    """A decision changes Memories of every Source behind the Review, so the caller must manage each one."""
    viewer_id = resolve_request_principal(request)
    viewer_role = resolve_request_workspace_role(request)
    return all(
        source is not None and can_manage_source(dict(source), viewer_id=viewer_id, viewer_role=viewer_role)
        for source in sources
    )


def _require_review_decision_authority(request: Request, sources: list[dict[str, Any]]) -> None:
    """Enforce `_can_decide_review`; a Source the caller cannot see is reported as missing."""
    for source in sources:
        _require_source_discoverability(request, source)
    if not _can_decide_review(request, sources):
        raise _source_management_forbidden()


async def _require_memory_review_management(
    request: Request,
    db: Database,
    review: MemoryReview,
) -> tuple[Memory, Memory, tuple[Memory, ...]]:
    participants = await _require_memory_review_visibility(request, db, review)
    incumbent, challenger, related = participants
    sources = await _memory_review_sources(db, [incumbent, challenger, *related])
    available = [source for source in sources if source is not None]
    if len(available) != len(sources):
        raise HTTPException(status_code=409, detail="Review source is unavailable")
    _require_review_decision_authority(request, available)
    return participants


async def _require_lifecycle_review_visibility(
    request: Request,
    db: Database,
    review: Any,
) -> None:
    candidate_id, _ = _lifecycle_candidate_payload(review.staged_evidence)
    participant_ids = [review.incumbent_memory_id]
    if candidate_id is not None and await db.get_memory(candidate_id) is not None:
        participant_ids.append(candidate_id)
    visible = await _filter_visible_ids(
        db,
        participant_ids,
        _lifecycle_visibility_scope(request),
    )
    participants = await db.list_memories_by_ids(participant_ids)
    sources = await _memory_review_sources(db, participants)
    if not review_is_visible(participant_ids, visible, sources, scope=_lifecycle_visibility_scope(request)):
        raise HTTPException(status_code=404, detail="Review not found")


def _require_workspace_memory_administration(request: Request) -> None:
    if not can_manage_workspace(resolve_request_workspace_role(request)):
        raise HTTPException(status_code=403, detail="workspace_admin_authority_required")


def _operation_id(route: APIRoute) -> str:
    """Name an OpenAPI operation from the route name, path and first method in sorted order.

    A route that serves several methods (GET and HEAD) gets the same id on
    every run, so the published OpenAPI document is reproducible.
    """
    operation_id = re.sub(r"\W", "_", f"{route.name}{route.path_format}")
    return f"{operation_id}_{sorted(route.methods)[0].lower()}"


def _publish_setup_result_models(app: FastAPI) -> None:
    """List each local setup job's result model in the app's OpenAPI document.

    The host job route serves every operation, so its ``result`` stays a
    free-form object and FastAPI publishes no setup model of its own. Adding
    the models of ``SETUP_RESULT_MODELS`` as components lets the admin UI
    generate one result type per setup operation from the same document.
    """
    generate = app.openapi

    def openapi() -> dict[str, Any]:
        if app.openapi_schema is None:
            document = generate()
            _, definitions = models_json_schema(
                [(model, "serialization") for model in SETUP_RESULT_MODELS.values()],
                ref_template="#/components/schemas/{model}",
            )
            schemas = document.setdefault("components", {}).setdefault("schemas", {})
            for name, schema in definitions["$defs"].items():
                if schemas.setdefault(name, schema) != schema:
                    raise RuntimeError(f"setup result model {name} collides with another OpenAPI component")
        return app.openapi_schema

    app.openapi = openapi  # type: ignore[method-assign]


def _request_audit_context(request: Request) -> AuditContext:
    return AuditContext(actor_type="user", actor_id=resolve_request_principal(request))


def _require_maintenance_operator(request: Request) -> str:
    resolver = getattr(request.app.state, "maintenance_operator_resolver", None)
    if resolver is None:
        from memforge.server.principal import resolve_maintenance_operator

        resolver = resolve_maintenance_operator
    if not resolver(request):
        raise HTTPException(
            status_code=403,
            detail="maintenance_operator_authority_required",
        )
    return resolve_request_principal(request)


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
    failed_items: ComponentHealth | None = Field(
        default=None,
        description=(
            "Failed Documents across Sources, as the failure ledger records them. The payload counts "
            "failed_items and retrying_after_upgrade, the rows an earlier release recorded that this "
            "release is retrying."
        ),
    )
    models: ModelHealthResponse = Field(
        description=(
            "Whether a generation model and an embedding model are configured, and the "
            "state of each. No generation model is ready unless generation.state is 'ready'."
        ),
    )


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


class MemoryEvidenceDocumentDetail(BaseModel):
    doc_id: str
    title: str | None = None
    source_url: str | None = None
    content_url: str | None = None
    pdf_url: str | None = None
    source_updated_at: str | None = None


class MemoryEvidenceArtifactDetail(BaseModel):
    """The Artifact an Evidence item cites; its item holds the Observation, role and digest."""

    artifact_id: str
    parent_observation_id: str
    filename: str
    content_type: str
    size_bytes: int
    url: str


class MemoryEvidenceItemDetail(BaseModel):
    authority: Literal["revision_pinned", "application_document"]
    evidence_reference_id: str | None = None
    role: Literal["primary", "required", "context"]
    kind: Literal["text", "artifact"]
    support_contribution: bool
    observation_id: str | None = None
    observation_revision_id: str | None = None
    anchor_kind: str
    fragment_id: str | None = None
    range_start: int | None = None
    range_end: int | None = None
    excerpt: str | None = None
    raw_content_sha256: str | None = None
    text: str | None = None
    presentation_sha256: str | None = None
    text_view: dict[str, Any] | None = None
    interpretation_available: bool = True
    current: bool
    artifact: MemoryEvidenceArtifactDetail | None = None
    resource_url: str | None = None


class MemoryEvidenceGroupDetail(BaseModel):
    kind: Literal["evidence_unit", "document"]
    evidence_unit_id: str | None = None
    support_ids: list[str]
    source_id: str
    source_type: str
    source_unit_id: str | None = None
    source_unit_revision_id: str | None = None
    doc_id: str | None = None
    document: MemoryEvidenceDocumentDetail | None = None
    current: bool
    items: list[MemoryEvidenceItemDetail]
    resource_url: str | None = None


# The labels a stored relation can have; none is never stored.
RelationLabelLiteral = Literal["equivalent", "updates", "contradicts"]
# A dismissal note is a short explanation, not a document.
RELATION_DISMISSAL_NOTE_MAX_CHARS = 2000
# Runtime events and assessments read per online evaluation window; a window
# that reaches it reports partial counts.
ONLINE_EVALUATION_ROW_LIMIT = 1000


class MemorySourceRefDetail(BaseModel):
    source_id: str
    source_type: str
    name: str | None = None


class RelatedMemoryDetail(BaseModel):
    memory_id: str
    summary: str
    content_hash: str
    sources: list[MemorySourceRefDetail] = []
    evidence_time: str | None = None


class MemoryRelationDetail(BaseModel):
    """A current Cross-Document Relation read from this Memory."""

    label: RelationLabelLiteral
    role: Literal["newer", "older", "peer"]
    counterpart: RelatedMemoryDetail
    decided_by: Literal["classifier", "review"]


class DismissedRelationDetail(BaseModel):
    """The Relation Dismissals in force for one pair; undoing them shows the relations again."""

    counterpart: RelatedMemoryDetail
    labels: list[RelationLabelLiteral]
    dismissed_by: str
    dismissed_at: str
    note: str | None = None


class RelationPairDetail(BaseModel):
    label: RelationLabelLiteral
    newer_memory_id: str | None = None
    decided_by: Literal["classifier", "review"]
    decided_at: str
    memories: list[RelatedMemoryDetail]


class RelationListResponse(BaseModel):
    data: list[RelationPairDetail]
    total: int
    limit: int
    offset: int


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str
    expires_in: int = 86400


class RelationDismissalRequest(BaseModel):
    """Dismiss the relation the caller sees, bound to both contents it was shown for."""

    model_config = ConfigDict(extra="forbid")

    label: RelationLabelLiteral
    expected_content_hash: str = Field(min_length=1)
    counterpart_expected_content_hash: str = Field(min_length=1)
    note: str | None = Field(default=None, max_length=RELATION_DISMISSAL_NOTE_MAX_CHARS)


class RelationDismissalResponse(BaseModel):
    dismissal_id: str
    memory_low_id: str
    memory_high_id: str
    label: RelationLabelLiteral
    dismissed_by: str
    dismissed_at: str


class RelationRestoreResponse(BaseModel):
    restored_dismissal_ids: list[str]


class RelationDiscoveryWorkDetail(BaseModel):
    id: str
    memory_id: str
    source_id: str
    source_unit_id: str
    status: str
    attempts: int
    run_generation: int
    error: str | None = None
    error_code: str | None = None
    classifier_version: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None


class RelationDiscoveryWorkListResponse(BaseModel):
    data: list[RelationDiscoveryWorkDetail]
    total: int
    limit: int
    offset: int


class RelationDiscoveryWorkFilter(BaseModel):
    """Which finished discovery work an operator re-runs."""

    model_config = ConfigDict(extra="forbid")

    state: Literal["exhausted", "completed"]
    error_code: str | None = None
    updated_from: datetime | None = None
    updated_to: datetime | None = None
    classifier_version: str | None = None


class RelationDiscoveryRerunResponse(BaseModel):
    rerun_count: int


class CrossSourceReviewLabelsRequest(BaseModel):
    """Human relabels of decided Cross-Source Conflict Reviews, by Review id."""

    model_config = ConfigDict(extra="forbid")

    label_overrides: dict[str, CrossDocumentRelationLabel] = Field(default_factory=dict)


class RelationGroupLabelsRequest(BaseModel):
    """One challenger, every candidate discovery asked about for it in discovery's order, and labels for some.

    ``content_hashes`` gives each of these Memories' content hash when the
    group was labelled.
    """

    model_config = ConfigDict(extra="forbid")

    challenger_memory_id: str = Field(min_length=1)
    candidate_memory_ids: list[str] = Field(min_length=1)
    labels: dict[str, CrossDocumentRelationLabel] = Field(default_factory=dict)
    content_hashes: dict[str, str]


class RelationGroupCasesSeedRequest(BaseModel):
    """Labelled discovery groups of the request's workspace, named by Memory id only."""

    model_config = ConfigDict(extra="forbid")

    groups: list[RelationGroupLabelsRequest] = Field(min_length=1)
    labelled_by: str = Field(min_length=1)


class DerivationCaseRequest(BaseModel):
    """One synced Document to pin as a derivation case, by ``doc_id`` or by ``title`` within ``source_id``,
    with the rubric its extraction is checked against. ``derivation_id`` names the derivation to pin;
    without it, the Document's first derivation is pinned."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str | None = Field(default=None, min_length=1)
    source_id: str | None = Field(default=None, min_length=1)
    title: str | None = Field(default=None, min_length=1)
    derivation_id: str | None = Field(default=None, min_length=1)
    rubric: dict[str, object] = Field(min_length=1)
    population: Literal["representative_control", "failure_regression"] = "representative_control"


class DerivationCasesSeedRequest(BaseModel):
    """Documents of the request's workspace to pin, and the cohort they form."""

    model_config = ConfigDict(extra="forbid")

    cases: list[DerivationCaseRequest] = Field(min_length=1)
    role: Literal["sentinel", "development", "release_holdout"] = "sentinel"
    selection_policy_version: str = Field(min_length=1)


class SupportCaseRequest(BaseModel):
    """One synced Document whose applied update to pin as a Support case, by ``doc_id`` or by ``title``
    within ``source_id``. ``derivation_id`` names the update; without it, the Document's latest applied
    update whose Support the model judged is pinned. The rubric labels Memories ``supported`` or
    ``unsupported``: ``expected_support`` by Memory id, ``expected_support_terms`` by a term their claims hold."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str | None = Field(default=None, min_length=1)
    source_id: str | None = Field(default=None, min_length=1)
    title: str | None = Field(default=None, min_length=1)
    derivation_id: str | None = Field(default=None, min_length=1)
    rubric: dict[str, object] = Field(min_length=1)
    population: Literal["representative_control", "failure_regression"] = "representative_control"


class SupportCasesSeedRequest(BaseModel):
    """Documents of the request's workspace whose updates to pin, and the cohort they form."""

    model_config = ConfigDict(extra="forbid")

    cases: list[SupportCaseRequest] = Field(min_length=1)
    role: Literal["sentinel", "development", "release_holdout"] = "sentinel"
    selection_policy_version: str = Field(min_length=1)


class ReconciliationCaseRequest(BaseModel):
    """One hand-written reconciliation situation imported against a synced Document, by ``doc_id`` or by
    ``title`` within ``source_id``. ``input`` names the ``candidates`` with their Evidence, the
    ``incumbents`` and each incumbent's ``support`` result; the rubric's ``expected_incumbents`` labels
    incumbents ``kept``, ``revised`` or ``retired``."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str | None = Field(default=None, min_length=1)
    source_id: str | None = Field(default=None, min_length=1)
    title: str | None = Field(default=None, min_length=1)
    input: dict[str, object] = Field(min_length=1)
    rubric: dict[str, object] = Field(min_length=1)
    population: Literal["representative_control", "failure_regression"] = "representative_control"


class ReconciliationCasesSeedRequest(BaseModel):
    """Hand-written situations to import into the request's workspace, and the cohort they form."""

    model_config = ConfigDict(extra="forbid")

    cases: list[ReconciliationCaseRequest] = Field(min_length=1)
    role: Literal["sentinel", "development", "release_holdout"] = "sentinel"
    selection_policy_version: str = Field(min_length=1)


class CrossSourceReviewConversionReportResponse(BaseModel):
    report_id: str
    review_count: int
    # The relation each converted Review becomes, by Review id.
    relation_labels: dict[str, CrossDocumentRelationLabel]
    dismissal_review_ids: list[str]
    discarded_review_ids: list[str]
    rerun_review_ids: list[str]
    nothing_to_rerun_review_ids: list[str]
    exhausted_work_ids: list[str]
    planned: dict[str, int]
    applied: bool


class CrossSourceReviewConversionApplyRequest(CrossSourceReviewLabelsRequest):
    """Apply a report with the relabels it was made with."""

    report_id: str = Field(min_length=1)


class CrossSourceReviewConversionReceiptResponse(BaseModel):
    report_id: str
    applied_by: str
    applied_at: str
    planned: dict[str, int]
    written: dict[str, int]
    complete: bool


class CrossSourceReviewDeletionRequest(BaseModel):
    """Delete converted Reviews once their labeled evaluation cases are frozen."""

    model_config = ConfigDict(extra="forbid")

    report_id: str = Field(min_length=1)
    cohort_id: str = Field(min_length=1)


class CrossSourceReviewDeletionResponse(BaseModel):
    report_id: str
    deleted_review_count: int


class MemoryResponse(BaseModel):
    id: str
    memory_type: str
    content: str
    content_hash: str
    visibility: str
    owner_user_id: str | None = None
    project_key: str | None = None
    corroboration_count: int
    valid_from: str | None = None
    valid_until: str | None = None
    superseded_by: str | None = None
    status: str
    retirement_reason: str | None = None
    retired_at: str | None = None
    superseded_at: str | None = None
    replacement_reason: str | None = None
    replacement_kind: str | None = None
    created_at: str | None = None
    updated_at: str | None = None
    # Source type for the leading glyph: the memory's extraction origin, else its
    # first attached source. None only when a memory has no remaining provenance.
    origin_source_type: str | None = None
    # The originating agent client when provenance is submitted by a client
    # plugin (e.g. 'codex' or 'claude-code').
    origin_client: str | None = None
    # The Sources this Memory comes from that the caller may read.
    sources: list[MemorySourceRefDetail] = []
    # The Cross-Document Relations current for the caller.
    relations: list[MemoryRelationDetail] = []
    # The pending Review a Memory in ``pending_review`` waits for, so the
    # Memory opens straight into its decision.
    open_review_id: str | None = None


class MemoryDetailResponse(MemoryResponse):
    entity_refs: list[str] = []
    evidence: list[MemoryEvidenceGroupDetail] = []
    relation_notice: str | None = None
    dismissed_relations: list[DismissedRelationDetail] = []
    # Active Source Units support this Memory, so retiring it directly is
    # refused; ``is_source_backed`` defines the rule.
    source_backed: bool = False


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
    status: str | None = None  # active, superseded, retired, decayed, pending_review


class MemoryRetireRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: str = Field(min_length=1)
    expected_content_hash: str = Field(min_length=1)


class MemoryLifecycleResponse(BaseModel):
    memory_id: str
    status: str


class MemoryMaintenanceClosureItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memory_id: str = Field(min_length=1)
    expected_content_hash: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class MemoryMaintenanceClosureReportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entries: list[MemoryMaintenanceClosureItem] = Field(min_length=1, max_length=100)


class MemoryMaintenanceClosureApplyRequest(MemoryMaintenanceClosureReportRequest):
    expected_report_id: str = Field(min_length=1)


class MemoryMaintenanceClosureReportItemResponse(BaseModel):
    memory_id: str
    disposition: str


class MemoryMaintenanceClosureReportResponse(BaseModel):
    report_id: str
    ready_count: int
    entries: list[MemoryMaintenanceClosureReportItemResponse]


class MemoryMaintenanceClosureReceiptResponse(BaseModel):
    receipt_id: str
    report_id: str
    memory_id: str
    status: str
    authority: Literal["maintenance_operator"]
    operator_actor_id: str


class MemoryMaintenanceClosureApplyResponse(BaseModel):
    report_id: str
    receipts: list[MemoryMaintenanceClosureReceiptResponse]


class MemoryCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    memory_type: Literal["fact", "decision", "convention", "procedure"] = "fact"
    client: Literal["codex", "claude-code"] = "codex"
    repo_identifier: str | None = None
    idempotency_key: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _ignore_client_confidence(cls, value: object) -> object:
        """A Memory has no confidence; a ``confidence`` sent by released plugins is ignored."""

        if isinstance(value, Mapping) and "confidence" in value:
            return {key: item for key, item in value.items() if key != "confidence"}
        return value


class MemoryCreateResponse(BaseModel):
    memory_id: str
    status: str


class MemoryCorrectionProposalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    replacement_content: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    expected_content_hash: str = Field(min_length=1)
    replacement_kind: Literal["revision", "supersession"] = "supersession"


class MemoryCorrectionProposalResponse(BaseModel):
    memory_id: str
    replacement_memory_id: str
    outcome: Literal["applied", "review_created"]
    status: str
    replacement_kind: str
    review_id: str


class SourceFacetFilterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ids: list[str] | None = None
    clients: list[str] | None = None
    repo_identifiers: list[str] | None = None

    @field_validator("source_ids")
    @classmethod
    def _validate_source_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        if not value or any(not isinstance(item, str) or not item.strip() for item in value):
            raise ValueError("source_ids must be a non-empty list of source IDs")
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
            source_ids=tuple(self.source_ids or ()),
            clients=tuple(self.clients or ()),
            repo_identifiers=tuple(self.repo_identifiers or ()),
        )


_DATE_ONLY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class MemoryTimeRangeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date_type: Literal["source_updated_at", "memory_updated_at"] = "source_updated_at"
    start_date: date | None = None
    end_date: date | None = None

    @field_validator("start_date", "end_date", mode="before")
    @classmethod
    def _validate_date_only(cls, value: Any) -> Any:
        if value is None or (isinstance(value, date) and not isinstance(value, datetime)):
            return value
        if isinstance(value, str):
            if not _DATE_ONLY_RE.fullmatch(value):
                raise ValueError("date bounds must be YYYY-MM-DD dates, not datetimes")
            return value
        raise ValueError("date bounds must be YYYY-MM-DD strings")

    @model_validator(mode="after")
    def _validate_bounds(self) -> "MemoryTimeRangeRequest":
        if self.start_date is None and self.end_date is None:
            raise ValueError("time_range requires start_date or end_date")
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError("start_date must be on or before end_date")
        return self

    def to_time_range(self) -> MemoryTimeRange:
        after = (
            datetime(self.start_date.year, self.start_date.month, self.start_date.day, tzinfo=timezone.utc)
            if self.start_date is not None
            else None
        )
        before = (
            datetime(self.end_date.year, self.end_date.month, self.end_date.day, tzinfo=timezone.utc)
            + timedelta(days=1)
            if self.end_date is not None
            else None
        )
        return MemoryTimeRange(after=after, before=before, date_type=self.date_type)


class RecentMemoryTimeRangeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date_type: Literal["source_updated_at", "memory_updated_at"] = "source_updated_at"
    start_at: datetime
    end_at: datetime

    @field_validator("start_at", "end_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("recent-memory time bounds require an explicit UTC offset")
        return value

    @model_validator(mode="after")
    def _validate_half_open_window(self) -> "RecentMemoryTimeRangeRequest":
        if self.start_at >= self.end_at:
            raise ValueError("time_range.start_at must be before end_at")
        return self

    def to_time_range(self) -> MemoryTimeRange:
        return MemoryTimeRange(
            after=self.start_at.astimezone(timezone.utc),
            before=self.end_at.astimezone(timezone.utc),
            date_type=self.date_type,
        )


class RecentMemoryListRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_ids: list[str] | None = None
    time_range: RecentMemoryTimeRangeRequest
    memory_types: list[str] | None = None
    page_size: int = Field(default=50, ge=1, le=50)
    cursor: str | None = None
    include_private: bool = Field(default=True, description=INCLUDE_PRIVATE_DESCRIPTION)

    @field_validator("source_ids")
    @classmethod
    def _validate_source_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        if not value or any(not item.strip() for item in value):
            raise ValueError("source_ids must be a non-empty list")
        return list(dict.fromkeys(item.strip() for item in value))

    @field_validator("memory_types")
    @classmethod
    def _validate_memory_types(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        allowed = {item.value for item in MemoryType}
        if not value or any(item not in allowed for item in value):
            raise ValueError("memory_types must exactly match supported memory types")
        return list(dict.fromkeys(value))

    @field_validator("cursor")
    @classmethod
    def _validate_cursor(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError("cursor must not be empty")
        return normalized


class MemorySearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = ""
    intent: RankedRetrievalIntent | None = None
    memory_types: list[str] | None = None
    time_range: MemoryTimeRangeRequest | None = None
    entities: list[str] | None = None
    include_superseded: bool = False
    top_k: int = Field(default=DEFAULT_SEARCH_TOP_K, ge=1, le=50)
    offset: int = Field(default=0, ge=0)
    active_project: str | None = None
    active_repo_identifier: str | None = None
    scope_mode: Literal["project", "project-first", "workspace"] = "project-first"
    include_private: bool = Field(default=True, description=INCLUDE_PRIVATE_DESCRIPTION)
    status: str | None = None
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
    def _validate_query_or_deterministic_filter(self) -> "MemorySearchRequest":
        if self.query.strip():
            return self
        if self.intent is not None:
            raise ValueError("intent requires a ranked query")
        has_source_filter = self.source_filter is not None and bool(
            (self.source_filter.source_ids or [])
            or (self.source_filter.clients or [])
            or (self.source_filter.repo_identifiers or [])
        )
        has_time_range = self.time_range is not None
        if not has_source_filter and not has_time_range:
            raise ValueError("query may be omitted only when source_filter or time_range is provided")
        return self

    @model_validator(mode="after")
    def _coerce_scope_without_active_project(self) -> "MemorySearchRequest":
        """Project-aware ranking requires an active project. When the caller
        asks for a project-aware mode without one, fall through to the flat
        workspace ranking so the contract stays "default just works"."""
        if self.scope_mode in ("project", "project-first") and self.active_project is None:
            logger.info("Search request omitted active_project; falling back to flat workspace ranking.")
            self.scope_mode = "workspace"
        return self


class MemorySearchResultDetail(BaseModel):
    """One ranked Memory, as ``SearchResult`` (memforge.models) carries it."""

    memory_id: str
    memory_type: str
    summary: str
    relevance_score: float
    corroborated_by: int
    last_observed_at: str | None = None
    freshness: str
    relation_notice: str | None = None
    relations: list[MemoryRelationDetail] = []
    status: str
    repo_identifier: str | None = None
    follow_up: dict[str, str] | None = None
    retrieval_evidence: dict[str, Any] | None = None


class MemorySearchResponse(BaseModel):
    """The search engine's result page.

    The route leaves unset fields out, so a queryless listing and a ranked
    search each keep their own shape: only a listing reports ``total_count``,
    only a ranked search reports ``retrieval_intent``.
    """

    query_analysis: dict[str, Any]
    retrieval_intent: dict[str, str | None] | None = None
    results: list[MemorySearchResultDetail]
    total_candidates: int
    total_count: int | None = None
    candidate_count_kind: Literal["exact", "windowed"]
    ranking_window_size: int
    limit: int
    offset: int
    has_more: bool
    retrieval_time_ms: int


# -- Entities --


class EntityAliasResponse(BaseModel):
    alias: str
    alias_normalized: str
    source: str
    created_at: str | None = None


class EntityResponse(BaseModel):
    id: int
    canonical_name: str
    display_name: str
    created_at: str | None = None


class EntityDetailResponse(EntityResponse):
    aliases: list[EntityAliasResponse] = []
    linked_memory_count: int = 0
    can_curate: bool = False


class EntityListResponse(BaseModel):
    data: list[EntityResponse]
    total: int


class MergeEntitiesRequest(BaseModel):
    source_id: int
    target_id: int


class AddAliasRequest(BaseModel):
    alias: str


class MergedEntitiesResponse(BaseModel):
    """The two Entities a merge joined: the source is gone, the target holds its references."""

    source_id: int
    source_name: str
    target_id: int
    target_name: str


class EntityMergeResponse(BaseModel):
    ok: bool
    merged: MergedEntitiesResponse


class EntityAliasListResponse(BaseModel):
    data: list[EntityAliasResponse]


class EntityAliasAddedResponse(BaseModel):
    """The manual alias as given and the normalised form that matching reads."""

    ok: bool
    alias: str
    alias_normalized: str


class OkResponse(BaseModel):
    """A change that reports only that it was made."""

    ok: bool


# -- Sources / Genes --


class GeneMetadataResponse(BaseModel):
    name: str
    display_name: str
    description: str
    default_sync_interval_minutes: int
    auth_method: str
    data_shape: str
    execution_kinds: list[SourceExecutionKind]
    managed: bool = Field(description="Managed capture, created by the service; the type picker leaves it out.")


class ConfigConditionResponse(BaseModel):
    field: str
    values: list[str]


class ConfigOptionResponse(BaseModel):
    value: str
    label: str
    help_text: str = ""
    visible_when: list[ConfigConditionResponse] = []


class ConfigPanelResponse(BaseModel):
    name: str
    visible_when: list[ConfigConditionResponse] = []


class ConfigFieldResponse(BaseModel):
    key: str
    label: str
    field_type: str
    required: bool = Field(default=True, description="Required while the field is visible.")
    placeholder: str = ""
    help_text: str = ""
    group: str = "general"
    order: int = 0
    default: str = ""
    choices: list[ConfigOptionResponse] = []
    visible_when: list[ConfigConditionResponse] = []
    widget: str | None = None
    advanced: bool = False
    scope: bool = Field(default=False, description="Changing it changes which items the Source holds.")
    fixed_after_create: bool = Field(default=False, description="Chooses the executor; a stored Source keeps it.")
    # Deprecated (admin-ui-v1): V1 setup reads bare option values; removed with V1.
    options: list[str] = Field(default=[], deprecated=True, description="The values of `choices`.")


class ConfigGroupResponse(BaseModel):
    key: str
    label: str
    order: int = 0
    description: str = ""
    purpose: ConfigGroupPurpose = "content"
    panels: list[ConfigPanelResponse] = []


class GeneConfigSchemaResponse(BaseModel):
    groups: list[ConfigGroupResponse] = []
    fields: list[ConfigFieldResponse] = []
    project_field: str | None = None


class ConfigCheckRequest(BaseModel):
    config: dict[str, Any]
    source_id: str | None = None


class ConfigFactResponse(BaseModel):
    label: str
    value: str


class FieldIssueResponse(BaseModel):
    key: str | None
    message: str


class ConfigCheckResponse(BaseModel):
    derived: dict[str, Any] = Field(
        description="Declared non-secret fields whose effective value the save would change, as it would store them."
    )
    facts: list[ConfigFactResponse]
    issues: list[FieldIssueResponse]


class DiscoveryPreviewRequest(BaseModel):
    source_id: str | None = None
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


class GitHubRepoTreeRequest(BaseModel):
    source_id: str | None = None
    config: dict[str, Any]
    limit: int = Field(default=2_000, ge=1, le=10_000)


class GitHubRepoTreeItemResponse(BaseModel):
    path: str
    type: Literal["tree", "blob"]
    size: int | None = None


class GitHubRepoTreeResponse(BaseModel):
    source_type: Literal["github_repo"] = "github_repo"
    ref: str
    count: int
    truncated: bool
    items: list[GitHubRepoTreeItemResponse]


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


SourceAccessPolicyValue = Literal["private", "workspace"]
SourceViewerRole = Literal["owner", "workspace_admin", "member", "viewer"]
SourceSyncRunStatus = Literal["pending", "running", "success", "failed"]
SourceSyncStatus = Literal["pending", "running", "recovering", "success", "partial", "failed"]
LocalAgentJobStatus = Literal["queued", "leased", "succeeded", "failed"]


class SourceSyncProgressAmountResponse(BaseModel):
    completed: int
    total: int | None = None
    unit: SyncProgressUnit


class SourceSyncProgressTimeRangeResponse(BaseModel):
    start: str | None = None
    end: str | None = None


class SourceSyncProgressCountsResponse(BaseModel):
    changed: int | None = None
    failed: int | None = None
    memories_created: int | None = None


class SourceSyncProgressResponse(BaseModel):
    """The public sync progress snapshot (see ``normalize_sync_progress_snapshot``)."""

    schema_version: Literal[1]
    phase: SyncProgressPhase
    progress: SourceSyncProgressAmountResponse | None = None
    source_time_range: SourceSyncProgressTimeRangeResponse | None = None
    counts: SourceSyncProgressCountsResponse | None = None


class SourceSyncFailedDocResponse(BaseModel):
    doc_id: str
    title: str
    error: str


class SourceFailedItemResponse(BaseModel):
    """One failed Document of a Source, as the failure ledger records it."""

    document_id: str
    title: str | None
    stage: str = Field(description="The step that failed, such as fetch, normalize, projection or derivation.")
    error_code: str | None
    message: str
    retryable: bool = Field(description="Whether a scheduled sync of the same release retries the Document.")
    release: str = Field(description="The release whose code failed.")
    failures_in_release: int
    first_failed_at: str
    last_failed_at: str
    retrying_after_upgrade: bool = Field(
        description="An earlier release recorded the failure, and this release retries the Document once.",
    )


class SourceFailedItemsResponse(BaseModel):
    """The Source's failed Documents: retried by the failure ledger, not by holding back the sync cursor.

    ``items`` lists the most recently failed first and is bounded; ``count`` is
    the total. A sync of the Source retries every one of them.
    """

    count: int
    retrying_after_upgrade: int
    items: list[SourceFailedItemResponse]


class SourceSyncStatusResponse(BaseModel):
    """A Source's latest sync, read from one of three places.

    An active durable run carries the run fields (``run_id``, ``trigger``,
    ``force_full_sync``, ``created_at``, ``next_attempt_at``,
    ``recovery_count``, ``progress_revision``, ``progress_updated_at``, and
    ``stop_requested_at`` once a person asked to stop it). A run a person
    stopped ends ``failed`` with ``error_message`` ``cancelled_by_user``. A
    sync running in this process carries its live counters (``phase``,
    ``docs_*``, ``memories_*``, ``current_title``). The last recorded sync
    carries its totals, ``run_id``, ``failed_docs``,
    ``absence_check_skipped_reason``, which names the step that kept that run
    from removing anything although its scope listing was due,
    ``unlisted_present_count``, the Documents its scope listing left out that
    the provider confirmed it still has or hides (``null`` when the run
    confirmed none), and ``skipped_reading_group_count``, the parts of its
    Documents the generation model could not read. Fields outside the chosen
    shape are absent.
    """

    status: SourceSyncStatus
    started_at: str | None
    finished_at: str | None
    error_message: str | None
    progress: SourceSyncProgressResponse | None
    run_id: str | None = None
    trigger: str | None = None
    force_full_sync: bool | None = None
    created_at: str | None = None
    next_attempt_at: str | None = None
    recovery_count: int | None = None
    progress_revision: int | None = None
    progress_updated_at: str | None = None
    stop_requested_at: str | None = None
    phase: str | None = None
    current_title: str | None = None
    docs_processed: int | None = None
    docs_total: int | None = None
    docs_updated: int | None = None
    docs_failed: int | None = None
    docs_stored: int | None = None
    memories_extracted: int | None = None
    memories_stored: int | None = None
    failed_docs: list[SourceSyncFailedDocResponse] | None = None
    absence_check_skipped_reason: AbsenceCheckSkippedReason | None = None
    unlisted_present_count: int | None = None
    skipped_reading_group_count: int | None = None


class SourceAccessTransitionResponse(BaseModel):
    """One durable change of a Source's access policy across its whole history.

    The operation is ``queued`` until its worker starts, ``running`` while it
    re-projects the Source's Memories, and ends ``completed``. A ``failed``
    operation leaves the Source paused until its owner retries it or reverts
    it to ``previous_policy``, which ends it ``reverted``.
    ``processed_memories`` counts the Memories re-projected so far, out of
    ``total_memories``.
    """

    operation_id: str
    source_id: str
    previous_policy: SourceAccessPolicyValue
    target_policy: SourceAccessPolicyValue
    status: Literal["queued", "running", "failed", "completed", "reverted"]
    total_memories: int
    processed_memories: int
    error_code: str | None = None
    error_message: str | None = None
    created_at: str
    updated_at: str
    completed_at: str | None = None


class SourceOwnershipResponse(BaseModel):
    created_by_user_id: str | None
    owner_user_id: str
    execution_owner_user_id: str | None
    viewer_role: SourceViewerRole
    viewer_relationship: SourceViewerRole


class SourceCapabilitiesResponse(BaseModel):
    """The viewer's authority over one Source; clients render row actions from it."""

    can_subscribe: bool
    can_configure: bool
    can_configure_connection: bool
    can_sync: bool
    can_force_resync: bool
    can_delete: bool
    can_change_access: bool


class SourceExecutionResponse(BaseModel):
    kind: Literal["server", "local_agent"]
    operation: str | None
    immutable_config_fields: list[str]


class SourceSubscriptionResponse(BaseModel):
    enabled: bool


class SourceConnectionStatusResponse(BaseModel):
    state: SourceConnectionState
    reason: SourceConnectionReason | None = None


class SourceScopeTransitionResponse(BaseModel):
    """The scope change a Source is applying.

    A change of what a Source syncs opens a transition from the last applied
    scope to the new one. ``removes`` says when what the new scope leaves
    out is removed:

    - ``at_completion`` (a document Source): its syncs commit the items the
      new scope lists, and once every listed item has committed, the change
      completes and what the new scope leaves out is removed. Until then
      nothing is removed. ``members_committed`` of ``listed_count`` items
      have committed; the count is ``null`` until a sync lists the new scope
      in full. ``failing_items`` are the failed items not yet committed under
      the change; one the new scope lists keeps it open until a sync commits
      it, and one only the previous scope held is settled when the change
      completes.
    - ``next_sync`` (a conversation Source): the next sync removes the
      conversations the new selection leaves out, and the change completes
      once every selected conversation is synced. Nothing is listed, so
      ``listed_count`` stays ``null``.

    Saving ``restore_config`` with ``PUT /sources/{id}``, with this
    transition's ``id`` as ``expected_scope_transition_id``, returns to the
    previous scope: it replaces the change in progress, and nothing the
    previous scope synced is removed. It is ``null`` when the viewer cannot
    change the configuration, or the change already returns to the previous
    scope.
    """

    id: str
    status: Literal["pending", "running", "failed"] = Field(
        description=(
            "pending: no sync has worked on the change yet; running: a sync is working on it; "
            "failed: the latest sync left it open, as ``error`` says."
        ),
    )
    previous_scope: dict[str, Any] | None = Field(
        description="The canonical scope fields of the last applied scope; null when the viewer cannot read the configuration.",
    )
    target_scope: dict[str, Any] | None = Field(
        description=(
            "The canonical scope fields of the new scope; null when the viewer cannot read the configuration. "
            "Equal to ``previous_scope`` while the Source returns to its previous scope."
        ),
    )
    created_at: str
    updated_at: str
    members_committed: int
    listed_count: int | None
    excluded_scopes_due: bool = Field(
        description="Every listed item has committed, and the next sync lists what the previous scope held.",
    )
    error: str | None = Field(description="Why the latest sync left the change open; null before one did.")
    failing_items: SourceFailedItemsResponse
    restore_config: dict[str, Any] | None
    removes: ScopeChangeRemoval = Field(
        description=(
            "When the change removes what the new scope leaves out: at_completion, once every listed item "
            "has committed (a document Source); next_sync, the deselected conversations in the next sync "
            "(a conversation Source)."
        ),
    )


class SourceResponse(BaseModel):
    """One Source as the viewer sees it in the Source List."""

    id: str
    type: str
    name: str
    config: dict[str, Any] = Field(
        description="Redacted configuration; empty when the viewer cannot configure the Source.",
    )
    status: Literal["active", "paused"]
    owner_user_id: str
    access_policy: SourceAccessPolicyValue
    access_state: Literal["active", "changing", "orphaned_private"]
    access_transition: SourceAccessTransitionResponse | None
    last_sync: str | None = None
    doc_count: int
    memory_count: int
    created_at: str | None = None
    project_binding: dict[str, Any] | None = Field(
        default=None,
        description="Project routing rule; null leaves the Source's Memories unmapped.",
    )
    sync_schedule: SourceSyncScheduleResponse | None = None
    sync: SourceSyncStatusResponse | None
    failed_items: SourceFailedItemsResponse | None = Field(
        description="The Source's failed Documents; null for a Source that does not sync.",
    )
    scope_transition: SourceScopeTransitionResponse | None = Field(
        description="The scope change the Source is applying; null when none is in progress.",
    )
    client: str | None = Field(
        description="The agent client (for example codex or claude-code) of an agent-session Source; null otherwise.",
    )
    connection_status: SourceConnectionStatusResponse | None = Field(
        default=None,
        description="Readiness of a browser-session connection; present only for Sources that use one.",
    )
    ownership: SourceOwnershipResponse
    capabilities: SourceCapabilitiesResponse
    execution: SourceExecutionResponse
    subscription: SourceSubscriptionResponse
    enabled_for_me: bool
    pinned_for_me: bool


class SourceListResponse(BaseModel):
    data: list[SourceResponse]


class SourceSyncRunReceiptResponse(BaseModel):
    """Admission receipt for a server-side sync run."""

    ok: bool
    message: str
    source_id: str
    run_id: str
    status: SourceSyncRunStatus
    created_at: str
    coalesced: bool


class SourceSyncStopResponse(BaseModel):
    """What stopping a Source's sync reached.

    ``outcome`` is ``stopping`` while the running server run ``run_id`` has
    yet to see the request: its worker stops at its next heartbeat, within
    about 30 seconds, and the run then fails with ``cancelled_by_user``.
    ``stopped`` means everything found has ended: a pending run, and the local
    daemon jobs in ``local_agent_job_ids``. ``not_running`` means the Source
    had nothing to stop. Asking again is harmless.
    """

    source_id: str
    outcome: Literal["stopping", "stopped", "not_running"]
    run_id: str | None
    local_agent_job_ids: list[str]


class LocalCollectionReceiptResponse(BaseModel):
    """Admission receipt for a sync that the execution owner's local daemon collects."""

    ok: bool
    message: str
    source_id: str
    job_id: str
    status: Literal["queued"]
    coalesced: bool


class SourceSyncRunStateResponse(BaseModel):
    """Where one admitted sync run stands, read without advancing it."""

    run_id: str
    status: SourceSyncRunStatus


class SourceCreatedResponse(BaseModel):
    id: str
    name: str
    type: str
    access_policy: SourceAccessPolicyValue


class SourceUpdatedResponse(BaseModel):
    ok: bool
    id: str


class SourcePinResponse(BaseModel):
    """Whether the Source is now pinned in the caller's Source List."""

    source_id: str
    pinned: bool


class SourceSubscriptionChangeResponse(BaseModel):
    """The caller's subscription to the Source after the change."""

    ok: bool
    source_id: str
    subscription: SourceSubscriptionResponse


class SourceCurrentAccessTransitionResponse(BaseModel):
    """The Source's unfinished access operation, or ``null`` when there is none."""

    transition: SourceAccessTransitionResponse | None


class SourceAccessTransitionOperationResponse(BaseModel):
    transition: SourceAccessTransitionResponse


class SourceAccessTransitionQueuedResponse(BaseModel):
    """A retry or revert admitted for the operation's worker."""

    operation_id: str
    status: Literal["queued"]


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
# Deprecated (admin-ui-v1): ranking and access read only the reserved SHARED
# key, so `kind` changes a label and nothing else. It is removed with the V1
# admin UI, together with `is_shared` in storage.
# The schema flag marks the field in OpenAPI without a runtime warning on V1 calls.
PROJECT_KIND_SCHEMA = {"deprecated": True}


class ProjectCreateRequest(BaseModel):
    name: str
    key: str | None = None
    kind: Literal["normal", "shared"] | None = Field(default=None, json_schema_extra=PROJECT_KIND_SCHEMA)


class ProjectUpdateRequest(BaseModel):
    name: str | None = None
    kind: Literal["normal", "shared"] | None = Field(default=None, json_schema_extra=PROJECT_KIND_SCHEMA)


class ProjectResponse(BaseModel):
    id: str
    key: str
    name: str
    kind: Literal["normal", "shared"] = Field(json_schema_extra=PROJECT_KIND_SCHEMA)
    created_at: str | None = None


class ProjectListItemResponse(ProjectResponse):
    memory_count: int = Field(
        description=(
            "Active memories in this project that the caller can see, including the caller's private memories."
        )
    )


class ProjectListResponse(BaseModel):
    data: list[ProjectListItemResponse]
    can_manage: bool = Field(
        description=(
            "Whether the caller may create, rename and delete projects. Built-in projects are never renamed or deleted."
        )
    )


class ProjectDeletionImpactResponse(BaseModel):
    """What deleting the project changes, counted across the whole workspace."""

    memory_count: int = Field(description="Memories in the project, in every status, that move to UNSORTED.")
    source_count: int = Field(
        description=(
            "Sources that stop writing to the project. A fixed binding to it is removed; a field binding "
            "drops its mappings to it, and its default moves to UNSORTED if it pointed to the project."
        ),
    )


class ProjectDeleteResponse(BaseModel):
    id: str
    rebucketed_count: int
    rebucketed_memory_ids: list[str]
    released_source_count: int = Field(
        description=(
            "Sources that stopped writing to the project. A fixed binding to it was removed; a field binding "
            "dropped its mappings to it, and its default moved to UNSORTED if it pointed to the project."
        ),
    )


class CreateSourceRequest(BaseModel):
    type: str
    name: str
    access_policy: Literal["private", "workspace"]
    config: dict
    project_binding: dict | None = None
    sync_schedule: SourceSyncScheduleRequest | None = None


class SourceAccessTransitionRequest(BaseModel):
    target_policy: Literal["private", "workspace"]


class SourceSyncRetryTarget(BaseModel):
    execution_kind: Literal["source_sync_run"]
    execution_id: str = Field(min_length=1, max_length=255)


class SourceSyncRequest(BaseModel):
    force_full_sync: bool = False
    dry_run: bool = Field(
        default=False,
        description=(
            "For a conversation Source with stored Units, to verify a migration: collect, then report what the "
            "attempt would change without processing it."
        ),
    )
    sync_snapshot_id: str | None = None
    local_agent_job_id: str | None = None
    local_agent_attempt_count: int | None = Field(default=None, ge=1)
    retry_target: SourceSyncRetryTarget | None = None


class SourceReprocessRequest(BaseModel):
    document_ids: list[str] = Field(
        default_factory=list,
        description="The Documents to reprocess; leave it empty with all_documents.",
    )
    all_documents: bool = Field(
        default=False,
        description="Reprocess every current Document of the Source instead of document_ids.",
    )
    dry_run: bool = False


class ReprocessUnitPreviewResponse(BaseModel):
    """What reprocessing one Document would read; an unavailable one has only its reason."""

    document_id: str
    available: bool
    reason: str | None = None
    detail: str | None = None
    source_unit_id: str | None = None
    unit_revision_id: str | None = None
    artifact_count: int | None = None
    active_memory_count: int | None = None
    support_reading_count: int | None = None
    reading_group_count: int | None = None
    extraction_item_count: int | None = None
    reading_chars: int | None = None
    estimated_model_calls: int | None = None


class ReprocessPreviewResponse(BaseModel):
    """What a reprocess would read, answered to a dry run."""

    source_id: str
    units: list[ReprocessUnitPreviewResponse]
    available_unit_count: int
    unavailable_unit_count: int
    estimated_model_calls: int


class SourceReprocessQueuedResponse(BaseModel):
    """The reprocess run queued for the Documents."""

    ok: bool
    message: str
    source_id: str
    run_id: str
    status: str
    created_at: str
    document_ids: list[str]


class ImageDocumentsResponse(BaseModel):
    """The current Documents with an image a model can read, in the Sources the caller can reprocess."""

    document_count: int
    source_count: int


class UnqueuedSource(BaseModel):
    source_id: str
    reason: Literal[
        "source_busy",
        "source_paused",
        "source_not_active",
        "source_not_synced",
        "source_missing",
        "projection_scope_transition_open",
    ] = Field(
        description=(
            "Why the Source's reprocess was not queued: another run is pending or running, the "
            "Source is paused or no longer active, it has not completed a sync, it was deleted "
            "meanwhile, or its projection scope is changing."
        ),
    )


class ImageDocumentsReprocessResponse(BaseModel):
    """The reprocess runs queued for the Documents with an image, one per Source."""

    document_count: int = Field(description="The Documents whose reprocess was queued.")
    source_count: int = Field(description="The Sources a reprocess run was queued for.")
    unqueued: list[UnqueuedSource] = Field(default_factory=list)


class LocalAgentJobCreateRequest(BaseModel):
    workspace_id: str | None = None
    source_id: str = ""
    source_type: str
    operation: str
    payload: dict[str, Any] = Field(default_factory=dict)
    retry_job_id: str | None = Field(default=None, min_length=1, max_length=255)


class LocalAgentJobLeaseRequest(BaseModel):
    limit: int = Field(default=5, ge=1, le=20)
    lease_seconds: int = Field(default=60, ge=1, le=3600)
    wait_seconds: int = Field(default=0, ge=0, le=25)


class LocalAgentJobHeartbeatRequest(BaseModel):
    attempt_count: int = Field(ge=1)
    lease_seconds: int = Field(default=60, ge=1, le=3600)
    progress: dict[str, Any] | None = None


class LocalAgentJobCompleteRequest(BaseModel):
    status: Literal["succeeded", "failed"]
    attempt_count: int = Field(ge=1)
    result: dict[str, Any] = Field(default_factory=dict)
    error: str | None = Field(default=None, max_length=2000)


class LocalAgentJobCreateResponse(BaseModel):
    job_id: str
    status: LocalAgentJobStatus
    coalesced: bool
    created_at: str | None = Field(
        default=None,
        description="Present for sync jobs, which may coalesce into an existing job.",
    )


class LocalAgentJobResponse(BaseModel):
    job_id: str
    workspace_id: str
    source_id: str
    source_type: str
    operation: str
    status: LocalAgentJobStatus
    payload: dict[str, Any]
    execution_owner_user_id: str
    result: dict[str, Any] = Field(
        description=(
            "The daemon's report: sync progress while the job runs, then its completion result. "
            "Empty before the first report."
        ),
    )
    last_error: str | None
    next_attempt_at: str | None
    leased_until: str | None
    attempt_count: int
    created_at: str
    updated_at: str
    finished_at: str | None


class LocalAgentJobListResponse(BaseModel):
    data: list[LocalAgentJobResponse]


class LocalAgentDaemonStatusResponse(BaseModel):
    status: Literal["online", "offline"]
    last_seen_at: str | None
    checked_at: str
    stale_after_seconds: int


class SourceSubscriptionRequest(BaseModel):
    enabled: bool


class SourceListPreferenceRequest(BaseModel):
    sort_mode: SourceListSortMode


class SourceListPreferencesResponse(BaseModel):
    sort_mode: SourceListSortMode


class UpdateSourceRequest(BaseModel):
    name: str | None = None
    config: dict | None = None
    status: str | None = None
    project_binding: dict | None = None
    sync_schedule: SourceSyncScheduleRequest | None = None
    expected_scope_transition_id: str | None = Field(
        default=None,
        description=(
            "The scope change the caller saw in progress. The save answers 409 "
            "``projection_scope_transition_stale`` when another one, or none, is in progress."
        ),
    )


class SourceScopeChangePreviewRequest(BaseModel):
    config: dict[str, Any] = Field(description="The configuration the Source would be saved with.")


class SourceScopeChangeRefusalResponse(BaseModel):
    """Why saving the configuration would be refused (409)."""

    code: str = Field(
        description=(
            "scope_change_requires_listing, transition_requires_listing, "
            "source_provider_namespace_immutable or source_execution_mode_immutable."
        ),
    )
    mode: str | None = Field(default=None, description="The configured mode that cannot list its scope.")
    message: str | None = None


class SourceScopeChangePreviewResponse(BaseModel):
    """What saving a configuration would do to the Source's scope, decided as ``PUT /sources/{id}`` decides it.

    - ``unchanged``: the scope stays as it is.
    - ``opens_transition``: a scope change opens.
    - ``supersedes_transition``: the change replaces the scope change in progress with a change
      from the last applied scope to the new one.
    - ``restores_previous_scope``: the change returns to the last applied scope, which the
      configuration cannot list. The change in progress closes, none opens, and nothing is removed.
    - ``returns_to_previous_scope``: the change returns to the last applied scope and replaces the
      change in progress with a change back to it. Nothing that scope synced is removed; what only
      the replaced change added is removed as ``removes`` says.
    - ``refused``: saving answers 409, as ``refusal`` says.

    ``removes`` says when a change that opens removes what its new scope leaves out:
    ``at_completion`` for a document Source, where nothing is removed until a sync lists the new
    scope in full and every listed item has committed; ``next_sync`` for a conversation Source,
    whose next sync removes the conversations the new selection leaves out.

    ``open_transition`` says whether a scope change is in progress now.
    """

    outcome: Literal[
        "unchanged",
        "opens_transition",
        "supersedes_transition",
        "restores_previous_scope",
        "returns_to_previous_scope",
        "refused",
    ]
    open_transition: bool
    removes: ScopeChangeRemoval = Field(
        description=(
            "When a change that opens removes what the new scope leaves out: at_completion (a document Source) "
            "or next_sync (a conversation Source)."
        ),
    )
    refusal: SourceScopeChangeRefusalResponse | None = None


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


def _source_sync_running_http_error(source_id: str, run_id: str) -> HTTPException:
    """The answer to a Source change that waits for the Source's sync run to end."""

    return HTTPException(
        status_code=409,
        detail={
            "code": "source_sync_running",
            "run_id": run_id,
            "message": (
                "A sync of this Source is running. Stop it with "
                f"POST /api/v1/sources/{source_id}/sync/stop, wait for it to end, and save again."
            ),
        },
    )


async def _stop_source_sync(
    request: Request,
    db: Database,
    sync_service: SyncService,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Stop the collection jobs of the Source's local daemon, then its sync run.

    The jobs go first, so that a job that completes meanwhile hands over a run
    that the run's stop then reaches.
    """

    source_id = str(source["id"])
    job_ids: list[str] = []
    if local_agent_sync_operation(source["type"], source.get("config")) is not None:
        stopper = getattr(request.app.state, "local_agent_job_stopper", None)
        if stopper is not None:
            job_ids = await stopper(source_id)
        else:
            job_ids = await db.stop_local_agent_source_jobs(source_id=source_id)
    run = await sync_service.stop_source(source_id)
    if run is not None and run.status == "running":
        outcome = "stopping"
    elif run is not None or job_ids:
        outcome = "stopped"
    else:
        outcome = "not_running"
    return {
        "source_id": source_id,
        "outcome": outcome,
        "run_id": run.run_id if run is not None else None,
        "local_agent_job_ids": job_ids,
    }


def _source_paused_http_error() -> HTTPException:
    return HTTPException(status_code=400, detail="Source is paused")


class LocalSourcePackageRequest(BaseModel):
    """One raw local-source package pushed by the local daemon/adapter.

    File-like sources use ``markdown_body`` as raw file text for the declared
    ``content_type``. Structured sources such as Jira use ``raw_payload`` and
    defer canonical markdown rendering to the source gene.
    ``source_updated_at`` is a file-like source's own time for this content
    (an offset-aware ISO time); structured sources carry their times in
    ``raw_payload``.
    """

    vault_id: str | None = None
    relative_path: str | None = None
    markdown_body: str = ""
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    content_type: str = "text/markdown"
    base_url: str | None = None
    repo_url: str | None = None
    repo_ref: str | None = None
    blob_sha: str | None = None
    symlink_chain: list[dict[str, str]] = Field(default_factory=list)
    resolved_relative_path: str | None = None
    issue_key: str | None = None
    source_url: str | None = None
    title: str | None = None
    raw_hash: str | None = None
    source_updated_at: str | None = None
    provider_revision: str | None = None
    sync_snapshot_id: str | None = None
    local_agent_job_id: str | None = None
    local_agent_attempt_count: int | None = Field(default=None, ge=1)
    artifact_source_unit_key: str | None = None
    artifact_input_hashes: list[str] = Field(default_factory=list)
    submitted_by: str | None = None
    submitted_at: str | None = None


async def _resolve_local_source_artifact_inputs(
    db: Database,
    *,
    source_id: str,
    workspace_id: str,
    input_hashes: list[str],
    expected_source_unit_key: str,
) -> list[dict[str, Any]]:
    """Resolve only attested raw inputs owned by this Source Unit."""

    requested = tuple(dict.fromkeys(str(value).strip() for value in input_hashes if str(value).strip()))
    if not requested:
        return []
    if any(len(value) != 64 for value in requested):
        raise ValueError("local source Artifact input hash is invalid")
    inputs = await db.get_source_sync_inputs_by_raw_hashes(
        source_id=source_id,
        workspace_id=workspace_id,
        raw_sha256s=requested,
    )
    by_hash = {item.raw_sha256: item for item in inputs}
    if set(by_hash) != set(requested):
        raise ValueError("local source Artifact input was not found for this source")
    resolved: list[dict[str, Any]] = []
    for input_hash in requested:
        source_input = by_hash[input_hash]
        metadata = source_input.metadata.get("local_source_artifact")
        if not isinstance(metadata, Mapping):
            raise ValueError("local source Artifact input metadata is invalid")
        if str(metadata.get("source_unit_key") or "") != expected_source_unit_key:
            raise ValueError("local source Artifact input belongs to another Source Unit")
        resolved.append(
            {
                "provider_key": str(metadata.get("provider_key") or ""),
                "parent_observation_type": str(metadata.get("parent_observation_type") or ""),
                "parent_provider_key": str(metadata.get("parent_provider_key") or ""),
                "provider_revision": str(metadata.get("provider_revision") or ""),
                "filename": str(metadata.get("filename") or ""),
                "media_type": str(metadata.get("media_type") or ""),
                "size_bytes": int(metadata.get("size_bytes") or 0),
                "sha256": str(metadata.get("content_sha256") or ""),
                "uri": source_input.raw_uri,
            }
        )
    return resolved


class LocalSourceManifestItemRequest(BaseModel):
    """One item of a device manifest: the item as the device read it."""

    doc_id: str = Field(min_length=1, max_length=512)
    revision: str = Field(min_length=1, max_length=512)
    change_kind: Literal["upsert"]


LOCAL_SOURCE_MANIFEST_MAX_ITEMS = SNAPSHOT_MANIFEST_MAX_ITEMS


class LocalSourceExcludedListingRequest(BaseModel):
    """What the device listed in one scope its job's open transition excludes, or that the provider rejected it.

    ``scope_key`` is the key the job named the scope by. A listing holds the
    Document ids the scope lists, up to the manifest's item bound. A
    definitive rejection, such as a JQL the provider refuses, gives its cause
    in ``rejection`` and lists nothing.
    """

    scope_key: str = Field(min_length=1, max_length=128)
    doc_ids: list[str] = Field(default_factory=list, max_length=LOCAL_SOURCE_MANIFEST_MAX_ITEMS)
    rejection: str | None = Field(default=None, min_length=1, max_length=2_000)

    @model_validator(mode="after")
    def _listed_or_rejected(self) -> "LocalSourceExcludedListingRequest":
        if self.rejection is not None and self.doc_ids:
            raise ValueError("a rejected excluded scope lists nothing")
        return self


class LocalSourceManifestRequest(BaseModel):
    """One complete local collection, planned against the retained packages.

    A device manifest lists the complete item set of its scope, so it declares
    ``complete_snapshot`` coverage and every item an ``upsert``.
    ``package_contract_version`` is the package contract the local agent
    collects under (``LOCAL_PACKAGE_CONTRACT_VERSION``); an agent that predates
    the numbered contract sends none. ``excluded_listings`` answers the
    excluded scopes the job carried, one entry per scope at most.
    """

    items: list[LocalSourceManifestItemRequest] = Field(max_length=LOCAL_SOURCE_MANIFEST_MAX_ITEMS)
    coverage: Literal["complete_snapshot"]
    sync_snapshot_id: str
    local_agent_job_id: str
    local_agent_attempt_count: int = Field(ge=1)
    package_contract_version: int | None = Field(default=None, ge=1)
    excluded_listings: list[LocalSourceExcludedListingRequest] = Field(default_factory=list, max_length=100)


class LocalSourceConfirmationAnswerRequest(BaseModel):
    """The device's answer for one item its manifest response asked it to confirm."""

    doc_id: str = Field(min_length=1, max_length=512)
    provider_key: str = Field(min_length=1, max_length=512)
    answer: Literal["absent", "exists", "hidden"]


class LocalSourceConfirmationsRequest(BaseModel):
    """A device's answers for the items its query manifest left out, bound to that manifest.

    ``manifest_sha256`` is the digest the manifest response returned, and the
    job attempt is the one that submitted the manifest. The request answers
    every item the response named in ``confirm``, or, when the device could
    not confirm them, gives ``failure_cause`` and no answers.
    """

    sync_snapshot_id: str
    local_agent_job_id: str
    local_agent_attempt_count: int = Field(ge=1)
    manifest_sha256: str = Field(min_length=64, max_length=64)
    answers: list[LocalSourceConfirmationAnswerRequest] = Field(
        default_factory=list, max_length=LOCAL_SOURCE_MANIFEST_MAX_ITEMS
    )
    failure_cause: str | None = Field(default=None, min_length=1, max_length=2_000)


CONVERSATION_MANIFEST_MAX_MESSAGES = 200_000
CONVERSATION_RECORD_MAX_BYTES = 1_000_000


class ConversationCollectionAttemptRequest(BaseModel):
    """The local-agent job attempt an upload of a conversation collection belongs to."""

    local_agent_job_id: str
    local_agent_attempt_count: int = Field(ge=1)


class ConversationManifestRequest(ConversationCollectionAttemptRequest):
    manifest: dict[str, Any]

    @field_validator("manifest")
    @classmethod
    def _bounded(cls, manifest: dict[str, Any]) -> dict[str, Any]:
        messages = manifest.get("messages")
        if isinstance(messages, list) and len(messages) > CONVERSATION_MANIFEST_MAX_MESSAGES:
            raise ValueError(f"a Collection Manifest records at most {CONVERSATION_MANIFEST_MAX_MESSAGES} messages")
        return manifest


class ConversationArtifactLinkRequest(BaseModel):
    provider_key: str = Field(min_length=1, max_length=1024)
    provider_revision: str = Field(min_length=1, max_length=1024)
    input_sha256: str = Field(min_length=64, max_length=64)


class ConversationBodyRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=512)
    message_id: str = Field(min_length=1, max_length=512)
    record: dict[str, Any]
    artifacts: list[ConversationArtifactLinkRequest] = Field(default_factory=list, max_length=100)

    @field_validator("record")
    @classmethod
    def _bounded(cls, record: dict[str, Any]) -> dict[str, Any]:
        if len(canonical_record_json(record).encode("utf-8")) > CONVERSATION_RECORD_MAX_BYTES:
            raise ValueError(f"a message record holds at most {CONVERSATION_RECORD_MAX_BYTES} bytes")
        return record


class ConversationBodiesRequest(ConversationCollectionAttemptRequest):
    bodies: list[ConversationBodyRequest] = Field(max_length=2_000)


async def _require_ready_snapshot_manifest(
    db: Database,
    *,
    source: Mapping[str, Any],
    source_id: str,
    workspace_id: str,
    input_snapshot_id: str,
    req: Any,
    lease_payload: Mapping[str, Any],
) -> None:
    """Require the document snapshot manifest of this job attempt, materialized and answered.

    Every declared item must be materialized, and every item the manifest
    asked its device to confirm must be answered.
    """

    manifest_status = await db.get_source_sync_snapshot_manifest_status(
        source_id=source_id,
        workspace_id=workspace_id,
        snapshot_id=input_snapshot_id,
    )
    if manifest_status is None:
        raise HTTPException(status_code=409, detail="source_snapshot_manifest_required")
    if (
        manifest_status["local_agent_job_id"] != (req.local_agent_job_id if req else None)
        or manifest_status["local_agent_attempt_count"] != (req.local_agent_attempt_count if req else None)
        or manifest_status["source_config_revision"] != str(lease_payload["source_config_revision"])
    ):
        raise HTTPException(status_code=409, detail="source_snapshot_manifest_stale")
    if manifest_status["materialized_count"] != manifest_status["item_count"]:
        raise HTTPException(status_code=409, detail="source_snapshot_materialization_incomplete")
    if manifest_status["awaiting_confirmations"]:
        raise HTTPException(status_code=409, detail="source_snapshot_confirmations_pending")
    if manifest_status["coverage"] != COMPLETE_SNAPSHOT:
        raise HTTPException(status_code=409, detail="authoritative_snapshot_coverage_required")


def _is_document_source_type(source_type: str) -> bool:
    """Whether Sources of this type belong to the document collection family."""

    descriptor = source_type_descriptor(source_type)
    return descriptor is not None and descriptor.family is CollectionFamilyKind.DOCUMENT


async def _snapshot_confirm_set(
    db: Database,
    *,
    source: Mapping[str, Any],
    workspace_id: str,
    snapshot_id: str,
    listed_doc_ids: frozenset[str],
    excluded_listings: Sequence[ExcludedScopeListing],
    package_contract_version: int | None,
) -> tuple[SnapshotConfirmation, ...]:
    """The items a device manifest asks its device to confirm by identifier.

    Only a document Source whose device listing is a query asks, and only an
    agent that confirms is asked. It asks about every Document with a current
    Unit the manifest does not list, by the provider's key of the Unit's item
    (without the incarnation a Unit that came back carries), and every
    failed Document the manifest does not list that never became a Unit, by
    the provider key of its stored item. A replay of a stored manifest returns
    the set the manifest was issued, without asking again. When the manifest
    lists the excluded scopes of a transition and none was rejected, an item
    an excluded listing names is not asked about: completing the transition
    removes it on that listing.

    A set larger than a manifest is issued empty: the run then records
    ``absence_confirmation_missing`` and removes nothing.
    """

    source_type = str(source.get("type") or "")
    descriptor = source_type_descriptor(source_type)
    if (
        not local_package_confirms_absence(package_contract_version)
        or descriptor is None
        or descriptor.family is not CollectionFamilyKind.DOCUMENT
    ):
        return ()
    gene_cls = cast(type[DocumentGene], descriptor.gene_class())
    if gene_cls.snapshot_listing_kind() is not ScopeListingKind.QUERY:
        return ()
    source_id = str(source["id"])
    stored = await db.get_source_sync_snapshot_manifest_status(
        source_id=source_id, workspace_id=workspace_id, snapshot_id=snapshot_id
    )
    if stored is not None:
        issued = await db.list_source_sync_snapshot_confirmations(
            source_id=source_id, workspace_id=workspace_id, snapshot_id=snapshot_id
        )
        return tuple(SnapshotConfirmation(item.doc_id, item.provider_key) for item in issued)
    held = await db.list_current_document_provider_keys(source_id)
    if excluded_listings and all(listing.rejection is None for listing in excluded_listings):
        listed_doc_ids = listed_doc_ids.union(*(listing.doc_ids for listing in excluded_listings))
    unlisted_units = sorted(set(held) - listed_doc_ids)
    unlisted_failures = sorted(
        (
            failure
            for failure in await db.list_source_item_failures(source_id=source_id)
            if failure.document_id not in listed_doc_ids and failure.document_id not in held
        ),
        key=lambda failure: failure.document_id,
    )
    if len(unlisted_units) + len(unlisted_failures) > LOCAL_SOURCE_MANIFEST_MAX_ITEMS:
        logger.warning(
            "Source %s has %d unlisted Documents, more than a manifest holds; none is issued for confirmation",
            source_id,
            len(unlisted_units) + len(unlisted_failures),
        )
        return ()
    confirm = [SnapshotConfirmation(doc_id, provider_item_key(held[doc_id])) for doc_id in unlisted_units]
    confirm.extend(
        SnapshotConfirmation(failure.document_id, gene_cls.provider_key(failure.item))
        for failure in unlisted_failures
        if failure.item is not None
    )
    return tuple(sorted(confirm, key=lambda item: item.doc_id))


@dataclass(frozen=True)
class _ScopeChange:
    """What a scope change writes with the Source: the transition it opens and the one it supersedes.

    A change while no transition is open supersedes nothing. A return to the
    last applied scope of a configuration that cannot list supersedes the
    open transition and opens none.
    """

    transition: ProjectionScopeTransition | None
    superseded_transition_id: str | None = None


async def _plan_scope_change(
    db: Database,
    source: Mapping[str, Any],
    config: Mapping[str, Any],
    current_scope: dict[str, object],
    target_scope: dict[str, object],
) -> _ScopeChange:
    """Plan the transition of one scope change from the Source's last applied scope to ``target_scope``.

    The last applied scope is the open transition's previous scope, or the
    current scope when no transition is open. The new transition runs from
    it to the target and supersedes the open transition, which becomes its
    predecessor. For example, A→B open and a change to C opens A→C after
    A→B; a change back to A opens A→A. What only B added is removed when a
    document transition completes, and by the next sync of a conversation
    transition, which deselects every held conversation its target leaves
    out.

    A document change into a configuration that cannot list is refused (409
    ``scope_change_requires_listing``), because its transition could never
    complete. Returning to exactly the last applied scope is the exception:
    the open transition is superseded and none opens, since that scope has no
    listing to remove anything by. A conversation configuration always lists
    its selection, so neither applies to it.
    """

    source_id = str(source["id"])
    source_type = str(source["type"])
    open_transition = await db.get_open_projection_scope_transition(source_id)
    if open_transition is not None:
        open_transition = canonical_scope_transition(source_type, open_transition, current_scope)
    last_applied_scope = dict(open_transition.previous_scope) if open_transition is not None else current_scope
    superseded_transition_id = open_transition.id if open_transition is not None else None
    if _is_document_source_type(source_type) and not source_lists_scope(source_type, config):
        if open_transition is not None and target_scope == last_applied_scope:
            return _ScopeChange(transition=None, superseded_transition_id=superseded_transition_id)
        raise _scope_change_requires_listing_http_error(source_type, config)
    latest_transitions = await db.list_projection_scope_transitions(source_id, limit=1)
    return _ScopeChange(
        transition=ProjectionScopeTransition(
            id=projection_scope_transition_id(
                source_id,
                last_applied_scope,
                target_scope,
                previous_cycle_transition_id=latest_transitions[0].id if latest_transitions else None,
            ),
            source_id=source_id,
            previous_scope=last_applied_scope,
            target_scope=target_scope,
            predecessor_transition_id=superseded_transition_id,
            created_at=datetime.now(timezone.utc).isoformat(),
        ),
        superseded_transition_id=superseded_transition_id,
    )


def _scope_change_requires_listing_http_error(source_type: str, config: Mapping[str, Any]) -> HTTPException:
    """The answer to a document scope change into a configuration that cannot list its whole scope."""

    unlisted = cast(type[DocumentGene], GENE_REGISTRY[source_type]).unlisted_scope_mode(config)
    return HTTPException(
        status_code=409,
        detail={
            "code": "scope_change_requires_listing",
            "mode": unlisted.mode,
            "message": (
                f"A scope change needs a configuration that lists its whole scope, and the mode "
                f"'{unlisted.mode}' cannot, so the change could never complete. "
                f"{unlisted.suggestion} Or create a new Source for this scope."
            ),
        },
    )


async def _require_listing_while_a_transition_is_open(
    db: Database,
    source: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    """Refuse a change that keeps the scope but leaves a document Source unable to finish its open transition.

    A transition completes only by listing its new scope. A configuration
    change outside the scope, such as GitHub Pages turning off its token or
    its complete sitemap, can make a listing configuration one that cannot
    list; while a transition is open that answers 409
    ``transition_requires_listing``.
    """

    source_type = str(source["type"])
    if (
        not _is_document_source_type(source_type)
        or not source_lists_scope(source_type, source["config"])
        or source_lists_scope(source_type, config)
        or await db.get_open_projection_scope_transition(str(source["id"])) is None
    ):
        return
    unlisted = cast(type[DocumentGene], GENE_REGISTRY[source_type]).unlisted_scope_mode(config)
    raise HTTPException(
        status_code=409,
        detail={
            "code": "transition_requires_listing",
            "mode": unlisted.mode,
            "message": (
                f"A scope change in progress completes only by listing its new scope, and the mode "
                f"'{unlisted.mode}' cannot list it. {unlisted.suggestion} Keep a configuration that lists "
                f"until the change completes, or return to the previous scope."
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class _ConfigChange:
    """A requested configuration as the Source would store it, and the scope change saving it makes."""

    config: dict[str, Any]
    scope_changed: bool
    scope_change: _ScopeChange


async def _plan_config_change(
    db: Database,
    source: Mapping[str, Any],
    requested: Mapping[str, Any],
    app_config: AppConfig,
    *,
    preview: bool = False,
) -> _ConfigChange:
    """Prepare ``requested`` as the Source would store it, and plan the scope change saving it makes.

    Saving and previewing a configuration decide alike; a ``preview``
    writes nothing, not even the Source's local inbox. A configuration the
    source type rejects answers 400. A change the Source refuses answers
    409: another execution mode or provider namespace, or a scope change it
    cannot apply (:func:`_plan_scope_change`,
    :func:`_require_listing_while_a_transition_is_open`).
    """

    source_id = str(source["id"])
    source_type = str(source["type"])
    try:
        incoming_config = _complete_source_config(source_type, dict(requested), source_id, source["config"])
        _validate_source_config(source_type, incoming_config, existing_config=source["config"])
        config = prepare_source_config_for_storage(
            incoming_config,
            existing_config=source["config"],
            secret_fields=_source_secret_fields(source_type),
        )
        config = _store_source_config(source_type, config, source_id, app_config, create_inbox=not preview)
    except SourceConfigInvalid as exc:
        raise _config_invalid_response(exc) from exc
    except (SecretConfigurationError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if is_local_agent_backed_source(source) != is_local_agent_backed_source({"type": source_type, "config": config}):
        raise HTTPException(status_code=409, detail="source_execution_mode_immutable")
    if _provider_namespace_config(source_type, config) != _provider_namespace_config(source_type, source["config"]):
        raise HTTPException(status_code=409, detail="source_provider_namespace_immutable")
    current_scope = _sync_scope_config(source_type, source["config"])
    target_scope = _sync_scope_config(source_type, config)
    if target_scope == current_scope:
        await _require_listing_while_a_transition_is_open(db, source, config)
        return _ConfigChange(config=config, scope_changed=False, scope_change=_ScopeChange(transition=None))
    return _ConfigChange(
        config=config,
        scope_changed=True,
        scope_change=await _plan_scope_change(db, source, config, current_scope, target_scope),
    )


def _scope_change_outcome(change: _ConfigChange) -> str:
    """How a planned configuration change moves the Source's scope (``SourceScopeChangePreviewResponse``)."""

    if not change.scope_changed:
        return "unchanged"
    scope_change = change.scope_change
    if scope_change.superseded_transition_id is None:
        return "opens_transition"
    transition = scope_change.transition
    if transition is None:
        return "restores_previous_scope"
    if dict(transition.target_scope) == dict(transition.previous_scope):
        return "returns_to_previous_scope"
    return "supersedes_transition"


def _scope_change_refusal(detail: Any) -> dict[str, Any]:
    """The refusal a 409 of a configuration change carries, as its code, mode and message."""

    if isinstance(detail, Mapping):
        return {"code": str(detail.get("code")), "mode": detail.get("mode"), "message": detail.get("message")}
    return {"code": str(detail)}


def _restore_previous_scope_config(source: Mapping[str, Any], previous_scope: Mapping[str, object]) -> dict[str, Any] | None:
    """The configuration that returns a Source to the last applied scope of its open transition.

    It is the configuration the viewer reads, with its scope fields replaced
    by the previous scope's (:func:`excluded_scope_config`); secrets stay
    redacted and are kept when it is saved. Saving it replaces the change in
    progress. ``None`` when the viewer cannot change the configuration.
    """

    capabilities = source.get("capabilities") or {}
    if not (capabilities.get("can_configure") and capabilities.get("can_configure_connection")):
        return None
    return excluded_scope_config(str(source.get("type") or ""), source.get("config") or {}, previous_scope)


def _viewed_scope_transition(source: Mapping[str, Any], transition: dict[str, Any]) -> dict[str, Any]:
    """The open transition as the viewer may read it.

    Both scopes are configuration, so a viewer who cannot read the
    configuration reads neither; the status and the counts stay. A
    transition back to the previous scope, which a restore opens, offers
    no restore of its own.
    """

    if not (source.get("capabilities") or {}).get("can_configure"):
        return {**transition, "previous_scope": None, "target_scope": None, "restore_config": None}
    returning = transition["previous_scope"] == transition["target_scope"]
    return {
        **transition,
        "restore_config": None if returning else _restore_previous_scope_config(source, transition["previous_scope"]),
    }


def _conversation_gene(source: Mapping[str, Any]) -> ConversationGene | None:
    """The conversation Gene of a Source, or ``None`` for a Source of another family."""

    descriptor = source_type_descriptor(str(source.get("type") or ""))
    if descriptor is None or descriptor.family is not CollectionFamilyKind.CONVERSATION:
        return None
    return cast(
        ConversationGene,
        create_gene(str(source["type"]), dict(source.get("config") or {}), str(source["id"])),
    )


@dataclass(frozen=True, slots=True)
class _ConversationUpload:
    """One device collection attempt of a conversation Source, as its uploads address it."""

    gene: ConversationGene
    collections: StoredCollections
    intake: CollectionIntake
    collection_attempt_id: str
    config_revision: str


def _collection_intake(db: Database, collections: StoredCollections) -> CollectionIntake:
    return CollectionIntake(reader=SourceProjectionUnitReader(db), collections=collections)


async def _require_complete_collection(intake: CollectionIntake, source_id: str, collection_attempt_id: str) -> None:
    """Refuse to plan or process an attempt before its manifest and every body it lacks are uploaded."""

    try:
        missing = await intake.missing_bodies_of(source_id, collection_attempt_id)
    except CollectionRejected as exc:
        raise HTTPException(status_code=409, detail="conversation_manifest_required") from exc
    if missing:
        raise HTTPException(status_code=409, detail="conversation_bodies_incomplete")


AGENT_SESSION_CAPTURE_REMOVED_CODE = "agent_session_capture_removed"
AGENT_SESSION_CAPTURE_REMOVED_MESSAGE = (
    "MemForge no longer captures agent sessions automatically. Update the MemForge memory plugin."
)


class AgentSessionCaptureRemovedDetail(BaseModel):
    """Why an automatic agent-session upload is refused, and what the user does next."""

    code: Literal["agent_session_capture_removed"]
    message: str


class AgentSessionCaptureRemovedResponse(BaseModel):
    """The ``410 Gone`` body of the removed agent-session capture routes.

    Example::

        {"detail": {"code": "agent_session_capture_removed",
                    "message": "MemForge no longer captures agent sessions automatically. "
                               "Update the MemForge memory plugin."}}
    """

    detail: AgentSessionCaptureRemovedDetail


def _refuse_removed_agent_session_capture() -> NoReturn:
    raise HTTPException(
        status_code=410,
        detail={
            "code": AGENT_SESSION_CAPTURE_REMOVED_CODE,
            "message": AGENT_SESSION_CAPTURE_REMOVED_MESSAGE,
        },
    )


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


# -- Memory reviews --


class MemoryReviewMemorySummary(BaseModel):
    id: str
    memory_type: str
    content: str
    corroboration_count: int
    status: str
    entity_refs: list[str] = []
    evidence: list[MemoryEvidenceGroupDetail] = []
    created_at: str | None = None
    updated_at: str | None = None
    origin_source_type: str | None = None
    origin_client: str | None = None


class ReviewActionPresentationResponse(BaseModel):
    key: Literal[
        "use_latest_state",
        "keep_current_state",
    ]
    decision: Literal["approve", "reject"]
    label: str
    consequence: str
    requires_note: bool


class ReviewPresentationResponse(BaseModel):
    decision_label: Literal["Updated", "Support removed"]
    summary: str
    why_human: str
    current_label: str
    proposed_label: str
    proposed_empty_text: str
    actions: list[ReviewActionPresentationResponse]
    technical_reason: str | None = None


class MemoryReviewResponse(BaseModel):
    id: str
    kind: str
    status: str
    review_origin: Literal["memory", "lifecycle"] = "memory"
    source_id: str | None = None
    source_name: str | None = None
    incumbent_memory_id: str
    challenger_memory_id: str | None
    reason: str | None = None
    review_note: str | None = None
    reviewer: str | None = None
    expected_incumbent_updated_at: str | None = None
    expected_challenger_updated_at: str | None = None
    replacement_kind: str = "supersession"
    created_at: str | None = None
    resolved_at: str | None = None
    is_stale: bool = False
    decision_fingerprint: str
    presentation: ReviewPresentationResponse
    can_decide: bool = Field(
        description=(
            "Whether the caller has the authority to decide this Review: it manages every Source behind it. "
            "The status says whether a decision is still open."
        ),
    )


class MemoryReviewListItemResponse(MemoryReviewResponse):
    incumbent: MemoryReviewMemorySummary | None = None
    challenger: MemoryReviewMemorySummary | None = None


class MemoryReviewListResponse(BaseModel):
    data: list[MemoryReviewListItemResponse]
    total: int
    limit: int
    offset: int


class MemoryReviewDetailResponse(MemoryReviewResponse):
    incumbent: MemoryReviewMemorySummary | None = None
    challenger: MemoryReviewMemorySummary | None = None
    related_challengers: list[MemoryReviewMemorySummary] = Field(default_factory=list)


class MemoryReviewDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    expected_fingerprint: str = Field(min_length=1)
    note: str | None = None


class AgentEvaluationRunCreateRequest(BaseModel):
    """Pinned inputs for admitting one durable offline evaluation run."""

    model_config = ConfigDict(extra="forbid")

    cohort_id: str = Field(min_length=1)
    candidate_manifest: dict[str, object]
    evaluator_suite: str = Field(min_length=1)
    evaluator_version: str = Field(min_length=1)
    replicate_count: int = Field(default=1, ge=1)
    baseline_run_id: str | None = None
    semantic_judge: dict[str, object] | None = None


class AgentEvaluationScopeResponse(BaseModel):
    kind: Literal["workspace"]
    source_id: str | None
    source_type: str | None


class AgentEvaluationWindowResponse(BaseModel):
    from_: str = Field(alias="from")
    to: str
    days: int


class AgentEvaluationSummaryResponse(BaseModel):
    """Occurrence counts over the effective assessments in the window."""

    total_assessments: int
    runtime_event_count: int
    eligible_assessment_count: int
    missing_assessment_count: int
    action_issue_group_count: int
    review_issue_group_count: int
    source_count: int
    affected_source_count: int = Field(
        description="Sources with failures, checks needing review, or coverage gaps.",
    )
    label_counts: dict[str, int]
    criterion_counts: dict[str, int]
    status_counts: dict[str, int]
    truncated: bool = Field(
        description="True when the window reached ``row_limit``, so counts are partial.",
    )
    row_limit: int = Field(
        description="The most runtime events, and the most assessments, that one window reads.",
    )


class AgentEvaluationCoverageResponse(BaseModel):
    """How many expected checks have a matching completed assessment."""

    policy: OnlineEvaluationCoveragePolicy
    eligible_occurrences: int
    assessed_occurrences: int
    pending_occurrences: int
    coverage_rate: float
    oldest_pending_at: str | None
    evaluator_failure_occurrences: int


class AgentEvaluationCaseResponse(BaseModel):
    """One example of an issue group: identifiers and versions only, no content."""

    assessment_id: str
    event_id: str
    label: AgentIssueLabel
    criterion: str
    reason_code: str
    occurred_at: str
    occurrence_count: int
    source_id: str
    source_type: str
    doc_id: str
    source_unit_id: str
    target_unit_revision_id: str
    observation_id: str | None
    observation_revision_id: str | None
    projection_run_id: str
    operation_id: str | None
    execution_id: str | None
    derivation_id: str | None
    batch_id: str | None
    trace_id: str | None
    provider: str | None
    model: str | None
    contract_version: str | None
    extraction_contract_version: str | None
    deployment_revision: str | None


class AgentEvaluationIssueGroupResponse(BaseModel):
    """Failed or degraded checks that share a criterion, reason and evaluator version."""

    group_id: str
    label: AgentIssueLabel
    criterion: str
    reason_code: str
    evaluator_name: str
    evaluator_version: str
    occurrence_count: int
    distinct_event_count: int
    criterion_occurrence_count: int
    criterion_rate: float
    affected_source_ids: list[str]
    affected_source_count: int
    source_types: list[str]
    first_seen_at: str
    last_seen_at: str
    representative_cases: list[AgentEvaluationCaseResponse]


class AgentEvaluationSourceHealthResponse(BaseModel):
    source_id: str
    name: str
    type: str
    source_status: str
    evaluation_status: SourceEvaluationStatus
    action_issue_group_count: int
    review_issue_group_count: int
    fail_occurrences: int
    review_occurrences: int
    coverage: AgentEvaluationCoverageResponse
    last_event_at: str | None


class AgentAssessmentResponse(BaseModel):
    """One content-free assessment, as ``assessment_public_payload`` returns it."""

    assessment_id: str
    target_event_id: str | None
    criterion: str
    status: AgentAssessmentStatus
    label: AgentAssessmentLabel | None
    reason_code: str
    annotator_kind: AgentAssessmentAnnotatorKind
    evaluator_name: str
    evaluator_version: str
    created_at: str
    target_result_id: str | None
    target_candidate_id: str | None
    annotator_id: str | None
    content_policy_id: str | None
    input_fingerprint: str | None
    confidence: AgentAssessmentConfidence | None
    reused_from_assessment_id: str | None
    occurrence_count: int
    schema_version: str


class WorkspaceAgentEvaluationResponse(BaseModel):
    """The online evaluation of the Sources the viewer can discover, over one window."""

    scope: AgentEvaluationScopeResponse
    window: AgentEvaluationWindowResponse
    summary: AgentEvaluationSummaryResponse
    coverage: AgentEvaluationCoverageResponse
    issue_groups: list[AgentEvaluationIssueGroupResponse]
    available_source_types: list[str]
    sources: list[AgentEvaluationSourceHealthResponse]
    runtime_events: list[dict[str, Any]] = Field(
        description="The latest runtime events, as ``event_public_payload`` returns them.",
    )
    assessments: list[AgentAssessmentResponse] = Field(
        description="The latest effective assessments, newest first.",
    )


class AgentEvaluationLangfusePolicyRequest(BaseModel):
    """Authorize one Source for one preconfigured reviewer queue."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)
    queue_id: str = Field(min_length=1)


class AgentEvaluationLangfuseExportRequest(BaseModel):
    """Create one blinded reviewer task in an approved Langfuse queue."""

    model_config = ConfigDict(extra="forbid")

    result_id: str = Field(min_length=1)
    content_policy_id: str = Field(min_length=1)
    criterion: str = Field(min_length=1)
    rubric_version: str = Field(min_length=1)
    provider_reviewer_id: str = Field(min_length=1)
    queue_id: str = Field(min_length=1)
    score_config_id: str = Field(min_length=1)


class MemoryReviewManifestDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    review_id: str = Field(min_length=1)
    decision: Literal["approve", "reject"]
    expected_fingerprint: str = Field(min_length=1)
    note: str | None = None
    rationale: str | None = Field(default=None, max_length=2000)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    risk: Literal["low", "medium", "high"] = "medium"


class MemoryReviewDecisionManifestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decisions: list[MemoryReviewManifestDecision] = Field(min_length=1, max_length=50)
    validation_receipt: str | None = Field(default=None, min_length=1)

    @field_validator("decisions")
    @classmethod
    def _unique_review_ids(
        cls,
        decisions: list[MemoryReviewManifestDecision],
    ) -> list[MemoryReviewManifestDecision]:
        review_ids = [item.review_id for item in decisions]
        if len(set(review_ids)) != len(review_ids):
            raise ValueError("decision manifest contains duplicate review_id values")
        return decisions


class MemoryReviewDecisionResult(BaseModel):
    review_id: str
    decision: Literal["approve", "reject"]
    outcome: Literal[
        "ready",
        "applied",
        "already_applied",
        "stale",
        "forbidden",
        "not_found",
        "invalid",
        "failed",
    ]
    status: str | None = None
    message: str | None = None
    decision_label: str | None = None
    consequence: str | None = None


class MemoryReviewDecisionManifestResponse(BaseModel):
    mode: Literal["validate", "apply"]
    results: list[MemoryReviewDecisionResult]
    ready: int = 0
    applied: int = 0
    already_applied: int = 0
    stale: int = 0
    forbidden: int = 0
    failed: int = 0
    validation_receipt: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _local_agent_collection(source_type: object) -> LocalAgentCollection | None:
    descriptor = source_type_descriptor(source_type)
    return descriptor.local_agent if descriptor is not None else None


def _config_policy(source_type: str) -> SourceConfigPolicy:
    descriptor = source_type_descriptor(source_type)
    return descriptor.config if descriptor is not None else SourceConfigPolicy()


def _source_secret_fields(source_type: str) -> tuple[str, ...]:
    fields = set(source_secret_fields(source_type, GENE_REGISTRY))
    fields.update(_config_policy(source_type).extra_secret_fields)
    return tuple(sorted(fields))


def _complete_source_config(
    source_type: str,
    config: dict[str, Any],
    source_id: str,
    existing_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill the values a source type derives from the Source or its stored configuration.

    A refusal raises ``SourceConfigInvalid``, which the routes answer as 400 ``config_invalid``.
    """

    policy = _config_policy(source_type)
    if policy.complete is None:
        return config
    try:
        return policy.complete(dict(config), source_id, existing_config)
    except SourceConfigInvalid:
        raise
    except ValueError as exc:
        raise _unkeyed_refusal(exc) from exc


def _store_source_config(
    source_type: str,
    config: dict[str, Any],
    source_id: str,
    app_config: AppConfig,
    *,
    create_inbox: bool = True,
) -> dict[str, Any]:
    """Adjust a configuration whose secrets are encrypted before it is stored.

    A configuration that names the Source's local inbox creates it, unless
    ``create_inbox`` is false, as for a preview that stores nothing.
    """

    policy = _config_policy(source_type)
    if policy.store is None:
        return config

    def local_inbox() -> str:
        from memforge.local_adapter import default_local_adapter_inbox

        inbox = default_local_adapter_inbox(app_config, source_id)
        if create_inbox:
            inbox.mkdir(parents=True, exist_ok=True)
        return str(inbox)

    return policy.store(dict(config), local_inbox)


def _source_credentials_changed(
    source_type: str,
    config: dict[str, Any],
    existing_config: dict[str, Any],
) -> bool:
    policy = _config_policy(source_type)
    return policy.credentials_changed(config, existing_config) if policy.credentials_changed else False


def _sync_scope_config(source_type: str, config: dict[str, Any]) -> dict[str, Any]:
    """Return only config fields that affect projected membership."""

    from memforge.source_projection_config import canonical_projection_scope

    return canonical_projection_scope(source_type, config)


def _public_source_config(source_type: str, config: dict[str, Any]) -> dict[str, Any]:
    """The configuration a reader of the Source sees."""

    policy = _config_policy(source_type)
    return policy.public(dict(config)) if policy.public else config


def _provider_namespace_config(source_type: str, config: dict[str, Any]) -> dict[str, Any]:
    """Return immutable identity-bearing fields for one Configured Source."""

    from memforge.source_projection_config import canonical_provider_namespace

    return canonical_provider_namespace(source_type, config)


def _validate_source_config(
    source_type: str,
    config: dict[str, Any],
    existing_config: dict[str, Any] | None = None,
) -> None:
    """Check a configuration the way a save does, normalizing it in place.

    The steps run in this order (``SourceConfigPolicy``): the type's
    ``validate_input`` on the configuration as given, the TLS CA bundle check,
    the Gene's ``normalize_config``, the shared form check
    (:func:`~memforge.source_types.form.check_config`), the secret-origin
    guard, and the type's ``validate``. Any refusal raises
    ``SourceConfigInvalid``, a ``ValueError`` the routes answer as 400
    ``config_invalid``; a plain ``ValueError`` from a hook becomes one issue
    with no field.
    """
    existing_config = existing_config or {}
    policy = _config_policy(source_type)
    gene_cls = GENE_REGISTRY.get(source_type)
    try:
        if policy.validate_input is not None:
            policy.validate_input(config, existing_config)
        validate_tls_ca_bundle(config)
        if gene_cls:
            normalizer = getattr(gene_cls, "normalize_config", None)
            if callable(normalizer):
                normalizer(config)
            descriptor = source_type_descriptor(source_type)
            issues = check_config(gene_cls.config_schema(), config, existing_config, descriptor)
            if issues:
                raise SourceConfigInvalid(issues)
        _require_secret_origin(source_type, config)
        if policy.validate is not None:
            policy.validate(config, existing_config)
    except SourceConfigInvalid:
        raise
    except ValueError as exc:
        raise _unkeyed_refusal(exc) from exc


def _require_secret_origin(source_type: str, config: dict[str, Any]) -> None:
    """A type that stores a secret sends it only to an HTTPS ``base_url``.

    The refusal names ``base_url`` when the Gene declares that field, and no
    field when the Gene derives it.
    """
    secret_fields = _source_secret_fields(source_type)
    gene_cls = GENE_REGISTRY.get(source_type)
    has_secret_contract = bool(secret_fields) if gene_cls else _config_contains_secret(config, secret_fields)
    if not has_secret_contract:
        return
    product_name = gene_cls.metadata().display_name if gene_cls else source_type
    declares_base_url = gene_cls is not None and any(f.key == "base_url" for f in gene_cls.config_schema().fields)
    key = "base_url" if declares_base_url else None
    base_url = str(config.get("base_url") or "").strip()
    if not base_url:
        raise SourceConfigInvalid([FieldIssue(key, f"{product_name} base_url is required before storing source secrets")])
    try:
        require_https_base_url(base_url, product_name)
    except ValueError as exc:
        raise SourceConfigInvalid([FieldIssue(key, str(exc))]) from None


def _unkeyed_refusal(exc: ValueError) -> SourceConfigInvalid:
    """A plain ``ValueError`` from a configuration hook, as a refusal that names no field."""

    return SourceConfigInvalid([FieldIssue(None, str(exc))])


def _config_invalid_response(exc: SourceConfigInvalid) -> HTTPException:
    """The 400 every configuration refusal answers, keyed by field where the check knows it."""

    return HTTPException(
        status_code=400,
        detail={
            "code": "config_invalid",
            "message": str(exc),
            "fields": [{"key": issue.key, "message": issue.message} for issue in exc.issues],
        },
    )


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


def _config_contains_secret(config: dict[str, Any], secret_fields: tuple[str, ...]) -> bool:
    for field in secret_fields:
        if config.get(field) or config.get(f"{field}_encrypted") or config.get(f"{field}_configured"):
            return True
    return False


def _dt_iso(dt: date | datetime | None) -> str | None:
    """Convert a date or datetime to an ISO string, or None."""
    if dt is None:
        return None
    if isinstance(dt, date) and not isinstance(dt, datetime):
        return dt.isoformat()
    value = dt
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _github_repo_tree_items(
    entries: list[dict],
    *,
    limit: int,
    include_extensions: set[str],
) -> list[GitHubRepoTreeItemResponse]:
    folders: set[str] = set()
    blobs: dict[str, int | None] = {}
    for entry in entries:
        raw_path = str(entry.get("path") or "").strip().strip("/")
        if not raw_path:
            continue
        entry_type = str(entry.get("type") or "")
        if entry_type != "blob":
            continue
        if not github_extension_allowed(raw_path, include_extensions):
            continue
        try:
            size = int(entry["size"]) if entry.get("size") is not None else None
        except (TypeError, ValueError):
            size = None
        blobs[raw_path] = size
        parts = raw_path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            folders.add("/".join(parts[:index]))

    items = [GitHubRepoTreeItemResponse(path=path, type="tree") for path in sorted(folders)]
    items.extend(GitHubRepoTreeItemResponse(path=path, type="blob", size=size) for path, size in sorted(blobs.items()))
    return items[: limit + 1]


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


def _memory_evidence_document_detail(
    doc_id: str,
    doc: Any | None,
    source_row: Any | None,
    unit_input: SourceUnitInput | None,
    config: AppConfig | None = None,
    artifact_store: DocumentArtifactStore | None = None,
) -> MemoryEvidenceDocumentDetail:
    """Describe the Document of one Evidence group; its links read the supporting Unit's stored input."""
    content_url = source_unit_content_url(unit_input, config, artifact_store)
    pdf_url = source_unit_pdf_url(unit_input, config, artifact_store)
    return MemoryEvidenceDocumentDetail(
        doc_id=doc_id,
        title=doc.title if doc else None,
        source_url=doc.source_url if doc else None,
        content_url=content_url,
        pdf_url=pdf_url,
        source_updated_at=(
            _dt_iso(source_row.source_updated_at)
            if source_row is not None
            else None
        ),
    )


def _memory_evidence_unit_detail(
    group,
    *,
    document: MemoryEvidenceDocumentDetail | None,
    memory_id: str,
) -> MemoryEvidenceGroupDetail:
    resource_url = (
        "/api/v1/memories/" + quote(memory_id, safe="")
        + "/evidence/" + quote(group.evidence_unit_id, safe="") + "/resource"
    )
    items: list[MemoryEvidenceItemDetail] = []
    for item in group.items:
        artifact = None
        if item.kind.value == "artifact":
            metadata = dict(item.artifact_metadata)
            artifact = MemoryEvidenceArtifactDetail(
                artifact_id=str(metadata.get("artifact_id") or item.anchor.observation_id),
                parent_observation_id=str(metadata.get("parent_observation_id") or ""),
                filename=str(metadata.get("filename") or "artifact"),
                content_type=str(metadata.get("media_type") or "application/octet-stream"),
                size_bytes=int(metadata.get("size_bytes") or 0),
                url=(
                    "/api/v1/source-artifacts/"
                    + quote(item.anchor.observation_revision_id, safe="")
                ),
            )
        items.append(
            MemoryEvidenceItemDetail(
                authority="revision_pinned",
                evidence_reference_id=item.reference_id,
                role=item.role.value,
                kind=item.kind.value,
                support_contribution=item.grants_support,
                observation_id=item.anchor.observation_id,
                observation_revision_id=item.anchor.observation_revision_id,
                anchor_kind=item.anchor.kind.value,
                fragment_id=item.anchor.fragment_id,
                range_start=item.anchor.range_start,
                range_end=item.anchor.range_end,
                excerpt=item.excerpt,
                text=item.display_text if item.display_text is not None else item.excerpt,
                raw_content_sha256=item.raw_content_sha256,
                presentation_sha256=item.presentation_sha256,
                text_view=item.text_view.payload() if item.text_view else None,
                interpretation_available=item.interpretation_available,
                current=item.current,
                artifact=artifact,
                resource_url=(resource_url.removesuffix("/resource") + "/references/"
                              + quote(item.reference_id, safe="") + "/resource"),
            )
        )
    return MemoryEvidenceGroupDetail(
        kind="evidence_unit",
        evidence_unit_id=group.evidence_unit_id,
        support_ids=list(group.support_ids),
        source_id=group.source_id,
        source_type=group.source_type,
        source_unit_id=group.source_unit_id,
        source_unit_revision_id=group.source_unit_revision_id,
        doc_id=group.doc_id,
        document=document,
        current=group.current,
        items=items,
        resource_url=resource_url,
    )


def _application_document_evidence_detail(
    source_row: Any,
    *,
    document: MemoryEvidenceDocumentDetail,
) -> MemoryEvidenceGroupDetail:
    excerpt = source_row.excerpt
    digest = (
        hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        if isinstance(excerpt, str)
        else None
    )
    return MemoryEvidenceGroupDetail(
        kind="document",
        support_ids=[],
        source_id=str(source_row.source_id),
        source_type=source_row.source_type,
        doc_id=source_row.doc_id,
        document=document,
        current=True,
        items=[
            MemoryEvidenceItemDetail(
                authority="application_document",
                role="primary",
                kind="text",
                support_contribution=True,
                anchor_kind="document",
                excerpt=excerpt,
                raw_content_sha256=digest,
                presentation_sha256=digest,
                current=True,
            )
        ],
    )


async def _memory_evidence_details(
    db: Database,
    memory_id: str,
    *,
    request: Request,
    config: AppConfig | None,
    artifact_store: DocumentArtifactStore | None,
) -> list[MemoryEvidenceGroupDetail]:
    """Build the sole visibility-safe Memory provenance projection."""

    raw_sources = await db.get_memory_sources(memory_id)
    raw_source_by_key = {
        (str(source.source_id), source.doc_id): source for source in raw_sources
    }
    details: list[MemoryEvidenceGroupDetail] = []
    seen_source_keys: set[tuple[str, str]] = set()
    for group in await db.get_memory_evidence_units(memory_id):
        if not group.visible_to(resolve_request_principal(request)):
            continue
        source = await db.get_source(group.source_id)
        if source is None:
            continue
        try:
            _require_source_discoverability(request, source)
        except HTTPException:
            # An unreadable supporting item omits the complete Unit.
            continue
        source_row = (
            raw_source_by_key.get((group.source_id, group.doc_id))
            if group.doc_id is not None
            else None
        )
        document = None
        if group.doc_id is not None:
            seen_source_keys.add((group.source_id, group.doc_id))
            doc = await db.get_document(group.doc_id)
            unit_input = await db.get_source_unit_input(group.source_unit_id)
            if unit_input is not None and unit_input.unit_revision_id != group.source_unit_revision_id:
                # A latest-input document link is navigation, not proof of this
                # historical Evidence. Keep the pinned excerpt, never substitute
                # another revision as its supporting content/PDF.
                unit_input = None
            document = _memory_evidence_document_detail(
                group.doc_id,
                doc,
                source_row,
                unit_input,
                config,
                artifact_store,
            )
        details.append(
            _memory_evidence_unit_detail(group, document=document, memory_id=memory_id)
        )

    for source_row in raw_sources:
        if source_row.has_unit_support:
            # Unit-backed provenance is emitted only through the verified Unit,
            # including its complete supporting parts and visibility checks.
            continue
        source_id = str(source_row.source_id)
        key = (source_id, source_row.doc_id)
        if key in seen_source_keys:
            continue
        if source_id not in VIRTUAL_DOCUMENT_SOURCE_IDS:
            source = await db.get_source(source_id)
            if source is None:
                continue
            try:
                _require_source_discoverability(request, source)
            except HTTPException:
                continue
        doc = await db.get_document(source_row.doc_id)
        source_unit = (
            await db.find_source_unit_by_document_id(source_id, source_row.doc_id, current_only=True)
            if source_id not in VIRTUAL_DOCUMENT_SOURCE_IDS
            else None
        )
        document = _memory_evidence_document_detail(
            source_row.doc_id,
            doc,
            source_row,
            await db.get_source_unit_input(source_unit.id) if source_unit is not None else None,
            config,
            artifact_store,
        )
        details.append(
            _application_document_evidence_detail(
                source_row,
                document=document,
            )
        )
        seen_source_keys.add(key)
    return details


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
    *,
    sources: Sequence[MemorySourceRef] = (),
    relations: Sequence[MemoryRelationContext] = (),
    open_review_id: str | None = None,
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
        corroboration_count=mem.corroboration_count,
        valid_from=_dt_iso(mem.valid_from),
        valid_until=_dt_iso(mem.valid_until),
        superseded_by=mem.superseded_by,
        status=mem.status,
        retirement_reason=mem.retirement_reason,
        retired_at=_dt_iso(mem.retired_at),
        superseded_at=_dt_iso(mem.superseded_at),
        replacement_reason=mem.replacement_reason,
        replacement_kind=mem.replacement_kind,
        created_at=_dt_iso(mem.created_at),
        updated_at=_dt_iso(mem.updated_at),
        origin_source_type=origin_source_type,
        origin_client=origin_client,
        sources=[MemorySourceRefDetail(**asdict(source)) for source in sources],
        relations=[MemoryRelationDetail(**asdict(relation)) for relation in relations],
        open_review_id=open_review_id,
    )


def _review_to_response(
    review: MemoryReview,
    *,
    incumbent: Memory | None = None,
    challenger: Memory | None = None,
    related_challengers: tuple[Memory, ...] = (),
    can_decide: bool,
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
        replacement_kind=review.replacement_kind,
        created_at=_dt_iso(review.created_at),
        resolved_at=_dt_iso(review.resolved_at),
        is_stale=is_memory_review_stale(review, incumbent, challenger),
        decision_fingerprint=memory_review_decision_fingerprint(review, related_challengers),
        presentation=_presentation_response(
            present_memory_review(
                reason=review.reason,
                source_backed_correction=review.expected_support_set_hash is not None,
            )
        ),
        can_decide=can_decide,
    )


def _relation_discovery_selection(
    *,
    state: str,
    error_code: str | None,
    updated_from: datetime | None,
    updated_to: datetime | None,
    classifier_version: str | None,
) -> RelationDiscoveryWorkSelection:
    return RelationDiscoveryWorkSelection(
        state=RelationDiscoveryWorkState(state),
        max_attempts=DEFAULT_RELATION_DISCOVERY_BUDGET.max_attempts,
        error_code=error_code,
        updated_from=updated_from,
        updated_to=updated_to,
        classifier_version=classifier_version,
    )


def _relation_discovery_work_detail(work: RelationDiscoveryWork) -> RelationDiscoveryWorkDetail:
    return RelationDiscoveryWorkDetail(
        id=work.request.id,
        memory_id=work.request.memory_id,
        source_id=work.request.source_id,
        source_unit_id=work.request.source_unit_id,
        status=work.status.value,
        attempts=work.attempts,
        run_generation=work.run_generation,
        error=work.error,
        error_code=work.error_code,
        classifier_version=work.classifier_version,
        created_at=work.created_at,
        updated_at=work.updated_at,
        completed_at=work.completed_at,
    )


def _conversion_receipt_response(
    receipt: CrossSourceReviewConversionReceipt,
) -> CrossSourceReviewConversionReceiptResponse:
    return CrossSourceReviewConversionReceiptResponse(
        report_id=receipt.report_id,
        applied_by=receipt.applied_by,
        applied_at=receipt.applied_at,
        planned=dict(receipt.planned),
        written=dict(receipt.written),
        complete=receipt.complete,
    )


async def _get_supersede_review(db: Database, review_id: str) -> MemoryReview | None:
    """The workbench Review with this id; supersede is the only Memory Review kind."""

    review = await db.get_memory_review(review_id)
    return review if review is not None and review.kind == ReviewKind.SUPERSEDE.value else None


def _presentation_response(value: ReviewPresentation) -> ReviewPresentationResponse:
    return ReviewPresentationResponse.model_validate(value.to_dict())


async def _build_memory_summary(
    db: Database,
    memory: Memory,
    request: Request,
    config: AppConfig | None = None,
    artifact_store: DocumentArtifactStore | None = None,
) -> MemoryReviewMemorySummary:
    """Hydrate a memory with provenance and entity refs for the review detail view."""
    evidence = await _memory_evidence_details(
        db,
        memory.id,
        request=request,
        config=config,
        artifact_store=artifact_store,
    )
    entity_names = await db.get_memory_entity_names(memory.id)

    return MemoryReviewMemorySummary(
        id=memory.id,
        memory_type=memory.memory_type,
        content=memory.content,
        corroboration_count=memory.corroboration_count,
        status=memory.status,
        entity_refs=entity_names,
        evidence=evidence,
        created_at=_dt_iso(memory.created_at),
        updated_at=_dt_iso(memory.updated_at),
    )


async def _memory_review_detail_response(
    db: Database,
    review: MemoryReview,
    *,
    request: Request,
    incumbent: Memory | None,
    challenger: Memory | None,
    related_challengers: tuple[Memory, ...],
    config: AppConfig | None = None,
    artifact_store: DocumentArtifactStore | None = None,
) -> MemoryReviewDetailResponse:
    participants = [memory for memory in (incumbent, challenger, *related_challengers) if memory is not None]
    base = _review_to_response(
        review,
        incumbent=incumbent,
        challenger=challenger,
        related_challengers=related_challengers,
        can_decide=_can_decide_review(request, await _memory_review_sources(db, participants)),
    )

    async def summary(memory: Memory | None) -> MemoryReviewMemorySummary | None:
        return await _build_memory_summary(db, memory, request, config, artifact_store) if memory else None

    return MemoryReviewDetailResponse(
        **base.model_dump(),
        incumbent=await summary(incumbent),
        challenger=await summary(challenger),
        related_challengers=[
            await _build_memory_summary(db, related, request, config, artifact_store)
            for related in related_challengers
        ],
    )


def _build_memory_review_list_summary(
    memory: Memory,
    *,
    origin_source_type: str | None = None,
    origin_client: str | None = None,
) -> MemoryReviewMemorySummary:
    """Build the review-queue list snapshot without full provenance hydration."""
    return MemoryReviewMemorySummary(
        id=memory.id,
        memory_type=memory.memory_type,
        content=memory.content,
        corroboration_count=memory.corroboration_count,
        status=memory.status,
        created_at=_dt_iso(memory.created_at),
        updated_at=_dt_iso(memory.updated_at),
        origin_source_type=origin_source_type,
        origin_client=origin_client,
    )


def _lifecycle_candidate_payload(
    staged_evidence: Mapping[str, object],
) -> tuple[str | None, Mapping[str, object] | None]:
    """The candidate Memory a Lifecycle Review proposes: its id, and the content it would have."""
    candidate = staged_evidence.get("candidate")
    candidate_payload = candidate if isinstance(candidate, Mapping) else None
    proposed = staged_evidence.get("proposed_mutations")
    if isinstance(proposed, list):
        creation = next(
            (
                raw
                for raw in proposed
                if isinstance(raw, Mapping) and raw.get("mutation_type") == LifecycleMutationType.CREATE_MEMORY.value
            ),
            None,
        )
        payload = creation.get("payload") if creation is not None else None
        memory_payload = payload.get("memory") if isinstance(payload, Mapping) else None
        if isinstance(memory_payload, Mapping):
            candidate_payload = memory_payload
    return lifecycle_review_candidate_memory_id(staged_evidence), candidate_payload


async def _lifecycle_review_response(
    db: Database,
    review: Any,
    source: Mapping[str, object],
    *,
    request: Request,
    detail: bool,
    config: AppConfig | None = None,
    artifact_store: DocumentArtifactStore | None = None,
    memory_cache: Mapping[str, Memory] | None = None,
) -> MemoryReviewListItemResponse | MemoryReviewDetailResponse:
    incumbent = (
        memory_cache.get(review.incumbent_memory_id)
        if memory_cache is not None
        else await db.get_memory(review.incumbent_memory_id)
    )
    incumbent_summary = None
    if incumbent is not None:
        incumbent_summary = (
            await _build_memory_summary(
                db, incumbent, request, config, artifact_store
            )
            if detail
            else _build_memory_review_list_summary(incumbent)
        )

    candidate_id, candidate_payload = _lifecycle_candidate_payload(review.staged_evidence)
    candidate_memory = (
        memory_cache.get(candidate_id)
        if memory_cache is not None and candidate_id
        else await db.get_memory(candidate_id)
        if candidate_id
        else None
    )
    challenger_summary = None
    if candidate_memory is not None:
        challenger_summary = (
            await _build_memory_summary(
                db, candidate_memory, request, config, artifact_store
            )
            if detail
            else _build_memory_review_list_summary(candidate_memory)
        )
    elif candidate_payload is not None and isinstance(candidate_payload.get("content"), str):
        challenger_summary = MemoryReviewMemorySummary(
            id=candidate_id or f"candidate:{review.id}",
            memory_type=str(candidate_payload.get("memory_type") or "fact"),
            content=str(candidate_payload["content"]),
            corroboration_count=0,
            status="proposed",
            origin_source_type=str(source.get("type") or "") or None,
        )

    common = {
        "id": review.id,
        "kind": "lifecycle",
        "status": review.status.value,
        "review_origin": "lifecycle",
        "source_id": str(source.get("id") or "") or None,
        "source_name": str(source.get("name") or "") or None,
        "incumbent_memory_id": review.incumbent_memory_id,
        "challenger_memory_id": candidate_id,
        "reason": review.reason,
        "review_note": getattr(review, "review_note", None),
        "reviewer": getattr(review, "reviewer", None),
        "created_at": review.created_at,
        "resolved_at": review.resolved_at,
        "is_stale": review.status.value == "stale",
        "decision_fingerprint": lifecycle_review_decision_fingerprint(review),
        "presentation": _presentation_response(
            present_lifecycle_review(
                staged_evidence=review.staged_evidence,
                reason=review.reason,
            )
        ),
        "incumbent": incumbent_summary,
        "challenger": challenger_summary,
        "can_decide": _can_decide_review(request, [source]),
    }
    if detail:
        return MemoryReviewDetailResponse(**common, related_challengers=[])
    return MemoryReviewListItemResponse(**common)


async def _review_queue_page(
    request: Request,
    db: Database,
    queue: ReviewQueue,
    page: Sequence[QueuedMemoryReview | QueuedLifecycleReview],
) -> list[MemoryReviewListItemResponse]:
    """Read the Reviews on one page in full and shape them for the queue."""
    memory_page = [queued for queued in page if isinstance(queued, QueuedMemoryReview)]
    lifecycle_page = [queued for queued in page if isinstance(queued, QueuedLifecycleReview)]
    lifecycle_reviews = (
        {review.id: review for review in await db.list_lifecycle_reviews_by_ids([q.entry.id for q in lifecycle_page])}
        if lifecycle_page
        else {}
    )
    shown_ids = (
        *(
            memory_id
            for queued in memory_page
            for memory_id in (
                queued.review.incumbent_memory_id,
                queued.review.challenger_memory_id,
                *queued.related_challenger_ids,
            )
        ),
        *(queued.entry.incumbent_memory_id for queued in lifecycle_page),
    )
    unread_ids = tuple(dict.fromkeys(memory_id for memory_id in shown_ids if memory_id not in queue.memories))
    memories = {
        **queue.memories,
        **({memory.id: memory for memory in await db.list_memories_by_ids(unread_ids)} if unread_ids else {}),
    }
    origin_ids = tuple(
        dict.fromkeys(
            memory_id
            for queued in memory_page
            for memory_id in (queued.review.incumbent_memory_id, queued.review.challenger_memory_id)
        )
    )
    origins = await _origin_source_types(db, list(origin_ids)) if origin_ids else {}

    def summary(memory: Memory) -> MemoryReviewMemorySummary:
        origin_source_type, origin_client = origins.get(memory.id, (None, None))
        return _build_memory_review_list_summary(
            memory,
            origin_source_type=origin_source_type,
            origin_client=origin_client,
        )

    responses: list[MemoryReviewListItemResponse] = []
    for queued in page:
        if isinstance(queued, QueuedLifecycleReview):
            lifecycle_review = lifecycle_reviews.get(queued.entry.id)
            if lifecycle_review is not None:
                responses.append(
                    await _lifecycle_review_response(
                        db,
                        lifecycle_review,
                        queued.source,
                        request=request,
                        detail=False,
                        memory_cache=memories,
                    )
                )
            continue
        review = queued.review
        incumbent = memories.get(review.incumbent_memory_id)
        challenger = memories.get(review.challenger_memory_id)
        related_challengers = tuple(
            memories[memory_id] for memory_id in queued.related_challenger_ids if memory_id in memories
        )
        if incumbent is None or challenger is None or len(related_challengers) != len(queued.related_challenger_ids):
            continue
        base = _review_to_response(
            review,
            incumbent=incumbent,
            challenger=challenger,
            related_challengers=related_challengers,
            can_decide=_can_decide_review(request, queued.sources),
        )
        only_source = queued.sources[0] if len(queued.sources) == 1 else None
        responses.append(
            MemoryReviewListItemResponse(
                **{
                    **base.model_dump(),
                    "source_id": str(only_source["id"]) if only_source is not None else None,
                    "source_name": str(only_source["name"]) if only_source is not None else None,
                },
                incumbent=summary(incumbent),
                challenger=summary(challenger),
            )
        )
    return responses


async def _build_memory_store(
    db: Database,
    config: AppConfig,
    runtime_provider: RuntimeProvider | None = None,
    audit_context: AuditContext | None = None,
) -> MemoryStore:
    """Build a request-scoped memory store that embeds with the active embedding model."""

    from memforge.memory.audit import MemoryAuditLogger
    from memforge.runtime import build_active_embedding_runtime

    provider = runtime_provider or DefaultRuntimeProvider()
    default_audit_context = audit_context or AuditContext(actor_type="admin")
    audit_logger = MemoryAuditLogger(db, default_context=default_audit_context)
    adapters = provider.build_adapters(db, config, audit_logger=audit_logger)
    return MemoryStore(
        relational=adapters.relational,
        keyword=adapters.keyword,
        vector=adapters.vector,
        embedding=await build_active_embedding_runtime(db, config, provider),
        audit_logger=audit_logger,
    )


async def _build_project_adapters(
    db: Database,
    config: AppConfig,
    runtime_provider: RuntimeProvider | None = None,
):
    """Build the storage adapters for project CRUD requests.

    The relational handle owns project rows; the vector handle is the same
    one the runtime provider hands the memory store, so the delete handler
    can update both sides in lockstep.
    """
    from memforge.memory.audit import AuditContext, MemoryAuditLogger

    provider = runtime_provider or DefaultRuntimeProvider()
    return provider.build_adapters(
        db,
        config,
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
    (SHARED, UNSORTED) is rejected by the create handler with HTTP 409.
    """
    import re

    cleaned = re.sub(_PROJECT_KEY_ALLOWED_PATTERN, "_", name).strip("_").upper()
    return (cleaned or _PROJECT_KEY_FALLBACK)[:_PROJECT_KEY_MAX_LENGTH]


def _can_manage_projects(request: Request) -> bool:
    """Projects shape the whole workspace, so only its admins may change them."""
    return can_manage_workspace(resolve_request_workspace_role(request))


def _require_project_management(request: Request) -> None:
    if not _can_manage_projects(request):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "project_management_forbidden",
                "message": "Only a workspace admin can create, change, or delete projects.",
            },
        )


def _reserved_project_conflict(key: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={
            "error": "reserved_project",
            "message": f"{key} is a built-in project. It cannot be created, renamed, changed, or deleted.",
        },
    )


async def _require_mutable_project(relational: RelationalStore, project_id: str) -> Project:
    project = await relational.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="project not found")
    if project.key in RESERVED_PROJECT_KEYS:
        raise _reserved_project_conflict(project.key)
    return project


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

    Its memory store embeds with the active embedding model, as the sync
    runtime does, so approve-time re-embedding follows the model settings.
    """
    memory_store = await _build_memory_store(db, config, runtime_provider)
    return ReviewService(db=db, memory_store=memory_store)


async def _build_lifecycle_service(
    db: Database,
    config: AppConfig,
    runtime_provider: RuntimeProvider | None = None,
    audit_context: AuditContext | None = None,
) -> MemoryLifecycleService:
    memory_store = await _build_memory_store(db, config, runtime_provider, audit_context=audit_context)
    return MemoryLifecycleService(db=db, memory_store=memory_store)


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


def get_workspace_id(request: Request) -> str:
    """FastAPI dependency: retrieve the app-scoped durable run namespace."""
    return request.app.state.workspace_id


def get_runtime_provider(request: Request) -> RuntimeProvider:
    """FastAPI dependency: retrieve the app-scoped runtime provider."""
    return request.app.state.runtime_provider


def get_model_settings(request: Request) -> ModelSettings:
    """FastAPI dependency: retrieve the app-scoped model settings."""
    return request.app.state.model_settings


async def _validate_models_after_startup(app: FastAPI) -> None:
    """Check the active model bindings that have not passed a check as they stand.

    The vector index is brought into order first, which asks no model: a
    rebuild that a stopped server left unfinished is ended before anything
    waits for a provider. An embedding model that passes its check here gives
    an installation without an index its first one.
    """
    await app.state.model_settings.prepare_vector_index(app.state.db)
    await app.state.model_settings.validate_unready_active_bindings(app.state.db)


async def refuse_memory_writes_during_reindex(
    request: Request,
    db: Database = Depends(get_db),
) -> None:
    """FastAPI dependency: refuse a request that changes Memories while the vector index is rebuilt.

    A rebuild embeds every searchable Memory as it stands, so Memories are
    not created, corrected, retired or decided on until it has finished.
    """
    if await request.app.state.model_settings.reindex_running(db):
        raise HTTPException(
            status_code=409,
            detail={"error": REINDEX_IN_PROGRESS, "message": REINDEX_IN_PROGRESS_MESSAGE},
        )


async def _erase_plaintext_model_keys(db: Database) -> None:
    """Empty the write-ahead log of plain-text model keys once no reader holds it."""
    while not await db.erase_plaintext_keys():
        await asyncio.sleep(_KEY_ERASURE_RETRY_S)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _validate_local_source_replay_artifact_identity(
    *,
    source_type: str,
    source_id: str,
    body: bytes,
    expected_doc_id: str,
    expected_version: str,
    expected_input_sha256: str,
    expected_package_sha256: str,
) -> Mapping[str, Any]:
    """Validate one immutable package and its provider-derived document identity."""
    from memforge.local_agent.replay_adapter import get_local_source_replay_adapter

    try:
        package = validate_local_agent_replay_package(
            source_type,
            body,
            expected_doc_id=expected_doc_id,
            expected_version=expected_version,
            expected_input_sha256=expected_input_sha256,
            expected_package_sha256=expected_package_sha256,
        )
        derived_doc_id = get_local_source_replay_adapter(source_type).derive_document_id(
            source_id=source_id,
            package=package,
        )
    except ValueError as exc:
        raise ValueError("source_lifecycle_local_replay_artifact_invalid") from exc
    if derived_doc_id != expected_doc_id:
        raise ValueError("source_lifecycle_local_replay_artifact_invalid")
    return package


async def _attest_retained_local_source_input(
    *,
    db: Any,
    artifact_store: DocumentArtifactStore,
    source_type: str,
    source_id: str,
    retained_input: SourceSyncInput,
) -> tuple[SourceSyncInput, str]:
    """Validate and attest a legacy retained artifact without trusting a duplicate upload."""
    metadata = retained_input.metadata
    manifest_entry = metadata.get("manifest_entry")
    if not isinstance(manifest_entry, Mapping):
        raise ValueError("source_lifecycle_local_replay_artifact_invalid")
    top_level_hash = str(metadata.get("package_sha256") or "").strip()
    manifest_hash = str(manifest_entry.get("package_sha256") or "").strip()
    if top_level_hash and manifest_hash and top_level_hash != manifest_hash:
        raise ValueError(f"source sync input artifact attestation conflict: {retained_input.input_id}")
    package_hash = top_level_hash or manifest_hash
    if top_level_hash and manifest_hash:
        return retained_input, package_hash
    try:
        body = artifact_store.read_artifact(retained_input.raw_uri)
    except Exception as exc:
        raise ValueError("source_lifecycle_local_replay_artifact_invalid") from exc
    package_hash = hashlib.sha256(body).hexdigest()
    expected_doc_id = str(manifest_entry.get("doc_id") or metadata.get("doc_id") or "")
    _validate_local_source_replay_artifact_identity(
        source_type=source_type,
        source_id=source_id,
        body=body,
        expected_doc_id=expected_doc_id,
        expected_version=str(manifest_entry.get("version") or ""),
        expected_input_sha256=retained_input.raw_sha256,
        expected_package_sha256=package_hash,
    )
    attested = await db.attest_source_sync_input_artifact(
        source_id=source_id,
        input_id=retained_input.input_id,
        package_sha256=package_hash,
    )
    return attested, package_hash


def create_admin_app(
    db: Database | None = None,
    config: AppConfig | None = None,
    runtime_provider: RuntimeProvider | None = None,
    principal_resolver: Callable[[Request], str] | None = None,
    workspace_role_resolver: Callable[[Request], str] | None = None,
    maintenance_operator_resolver: Callable[[Request], bool] | None = None,
    document_store: DocumentArtifactStore | None = None,
    local_agent_lease_validator: Callable[[Request, str, str, int, str], Awaitable[bool]] | None = None,
    local_agent_job_enqueuer: Callable[..., Awaitable[tuple[str, bool]]] | None = None,
    local_agent_job_stopper: Callable[[str], Awaitable[list[str]]] | None = None,
    langfuse_annotation_adapter: object | None = None,
    workspace_id: str = "local",
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
    if maintenance_operator_resolver is None and (
        principal_resolver is not None or workspace_role_resolver is not None
    ):
        def deny_maintenance_operator(_request: Request) -> bool:
            return False

        maintenance_operator_resolver = deny_maintenance_operator

    owned_db: Database | None = None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        nonlocal owned_db

        if db is None:
            owned_db = Database(
                config.storage.db_path,
                document_store=document_store or LocalDocumentStore(config.storage.docs_path),
                recovery_points=RecoveryPoints.for_config(config),
            )
            try:
                await owned_db.connect()
            except UpgradeError as exc:
                logger.error("MemForge did not start: %s", exc)
                raise
            try:
                # A database made before model bindings holds its model
                # settings in one row; this configuration completes them.
                await owned_db.translate_llm_config(config.llm)
            except BaseException:
                await owned_db.close()
                raise
            app.state.db = owned_db
            app.state.key_erasure_task = asyncio.create_task(_erase_plaintext_model_keys(owned_db))
        else:
            app.state.db = db
            app.state.key_erasure_task = None

        app.state.config = config
        app.state.workspace_id = workspace_id
        app.state.document_store = document_store or LocalDocumentStore(config.storage.docs_path)
        app.state.runtime_provider = runtime_provider
        app.state.principal_resolver = principal_resolver
        app.state.workspace_role_resolver = workspace_role_resolver
        app.state.maintenance_operator_resolver = maintenance_operator_resolver
        app.state.local_agent_lease_validator = local_agent_lease_validator
        app.state.local_agent_job_enqueuer = local_agent_job_enqueuer
        app.state.local_agent_job_stopper = local_agent_job_stopper
        app.state.langfuse_annotation_adapter = langfuse_annotation_adapter
        app.state.sync_service = SyncService(
            app.state.db,
            config,
            runtime_provider=runtime_provider,
            workspace_id=workspace_id,
        )
        # Each Document and each relation discovery work item an earlier
        # release failed is retried once by this one.
        try:
            await app.state.sync_service.enqueue_release_recovery()
        except Exception:
            logger.exception("Could not queue the retry of the failed Documents an earlier release recorded")
        try:
            await rerun_relation_discovery_failed_by_another_release(app.state.db)
        except Exception:
            logger.exception("Could not re-run the relation discovery work an earlier release exhausted")
        # The checks make model requests, so they run beside the server and
        # never hold up its start.
        app.state.model_validation_task = asyncio.create_task(_validate_models_after_startup(app))
        app.state.sync_scheduler = None
        if config.sync.scheduler_enabled:
            app.state.sync_scheduler = SyncScheduler(
                app.state.db,
                app.state.sync_service,
                document_store=app.state.document_store,
            )
            await app.state.sync_scheduler.start()
        app.state.sync_worker = None
        app.state.evaluation_worker = None
        app.state.sync_worker_task = None
        if config.sync.worker_enabled:
            app.state.sync_worker = SourceSyncWorker(
                app.state.db,
                config,
                runtime_provider=runtime_provider,
                worker_id="embedded-admin-worker",
                workspace_id=workspace_id,
            )
            if config.agent_evaluation.worker_enabled:
                from memforge.evals.offline_runtime import (
                    build_offline_evaluation_for_run,
                )
                from memforge.evals.offline_worker import OfflineEvaluationWorker

                async def evaluation_factory(run):
                    return await build_offline_evaluation_for_run(
                        app.state.db,
                        config,
                        runtime_provider,
                        run,
                        document_store=app.state.document_store,
                    )

                app.state.evaluation_worker = OfflineEvaluationWorker(
                    app.state.db,
                    evaluation_factory=evaluation_factory,
                    worker_id="embedded-admin-evaluation-worker",
                    lease_seconds=config.agent_evaluation.worker_lease_seconds,
                )
            app.state.sync_worker_task = asyncio.create_task(
                _run_embedded_service_workers(
                    app.state.sync_worker,
                    app.state.evaluation_worker,
                    poll_seconds=min(
                        config.sync.worker_poll_seconds,
                        config.agent_evaluation.worker_poll_seconds,
                    ),
                )
            )

        try:
            yield
        finally:
            for background_task in (
                app.state.model_validation_task,
                app.state.key_erasure_task,
                getattr(app.state, "sync_worker_task", None),
            ):
                if background_task is not None:
                    background_task.cancel()
                    try:
                        await background_task
                    except asyncio.CancelledError:
                        pass
            await app.state.model_settings.shutdown()
            if app.state.sync_scheduler is not None:
                await app.state.sync_scheduler.shutdown()
            await app.state.sync_service.shutdown()
            annotation_adapter = getattr(
                app.state,
                "langfuse_annotation_adapter",
                None,
            )
            if annotation_adapter is not None and hasattr(annotation_adapter, "shutdown"):
                await asyncio.to_thread(annotation_adapter.shutdown)
            if owned_db is not None:
                await owned_db.close()

    app = FastAPI(
        title="MemForge Admin API",
        version="0.1.0",
        description="Management API for the MemForge agent memory layer.",
        lifespan=lifespan,
        generate_unique_id_function=_operation_id,
    )
    _publish_setup_result_models(app)
    if db is not None:
        app.state.db = db
        app.state.sync_service = SyncService(
            db,
            config,
            runtime_provider=runtime_provider,
            workspace_id=workspace_id,
        )
        app.state.sync_worker = None
        app.state.evaluation_worker = None
        app.state.sync_worker_task = None
    app.state.config = config
    app.state.workspace_id = workspace_id
    app.state.document_store = document_store or LocalDocumentStore(config.storage.docs_path)
    app.state.runtime_provider = runtime_provider
    app.state.model_settings = ModelSettings(config, runtime_provider, listed_native_schema=listed_native_schema)
    app.state.principal_resolver = principal_resolver
    app.state.workspace_role_resolver = workspace_role_resolver
    app.state.maintenance_operator_resolver = maintenance_operator_resolver
    app.state.local_agent_lease_validator = local_agent_lease_validator
    app.state.local_agent_job_enqueuer = local_agent_job_enqueuer
    app.state.local_agent_job_stopper = local_agent_job_stopper
    app.state.langfuse_annotation_adapter = langfuse_annotation_adapter

    @app.middleware("http")
    async def resolve_workspace_context(request: Request, call_next):
        """Resolve the singleton workspace for every v1 data-plane request."""
        workspace_control_plane_paths = {
            "/api/v1/workspaces",
            "/api/v1/me/default-workspace",
        }
        if request.url.path.startswith("/api/v1/") and request.url.path not in workspace_control_plane_paths:
            current_workspace_id = str(request.app.state.workspace_id)
            requested_values = request.query_params.getlist("workspace_id")
            if len(requested_values) > 1:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "invalid_workspace_selector",
                        "detail": "workspace_id may be supplied at most once.",
                    },
                )
            requested_workspace_id = requested_values[0].strip() if requested_values else None
            if requested_values and not requested_workspace_id:
                return JSONResponse(
                    status_code=400,
                    content={
                        "code": "invalid_workspace_selector",
                        "detail": "workspace_id must not be empty.",
                    },
                )
            if requested_workspace_id and requested_workspace_id != current_workspace_id:
                return JSONResponse(
                    status_code=404,
                    content={
                        "code": "workspace_not_found_or_inaccessible",
                        "detail": "Workspace not found or inaccessible.",
                    },
                )
            response = await call_next(request)
            response.headers["MemForge-Workspace"] = current_workspace_id
            return response
        return await call_next(request)

    # -- CORS --
    # The loopback bind does not hold against the operator's own browser, so
    # only the configured origins may read responses. A wildcard answers every
    # origin and therefore never carries credentials.
    cors_origins = [origin.strip() for origin in config.server.cors_origins.split(",") if origin.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials="*" not in cors_origins,
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
    auth_router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

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
                    "code": JIRA_PRINCIPAL_CHANGED_CODE,
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

    @app.exception_handler(ReindexInProgress)
    @app.exception_handler(sqlite3.IntegrityError)
    async def refused_during_reindex_handler(request: Request, exc: Exception):
        """Answer a write the database refused during a rebuild of the vector index with its code."""
        if not is_reindex_refusal(exc):
            raise exc
        return JSONResponse(
            status_code=409,
            content={"detail": {"error": REINDEX_IN_PROGRESS, "message": REINDEX_IN_PROGRESS_MESSAGE}},
        )

    @app.exception_handler(EmbeddingIndexMismatch)
    @app.exception_handler(NoVectorIndex)
    async def waits_for_vector_index_handler(request: Request, exc: Exception):
        """Answer a write whose vector has no index to go to; it can be repeated once there is one."""
        return JSONResponse(
            status_code=409,
            content={
                "detail": {
                    "error": exc.code,
                    "message": (
                        "The vector index has to be rebuilt for the embedding model in use "
                        "before this change can be made."
                    ),
                }
            },
        )

    # -- Register routers --
    health_router = APIRouter(tags=["health"])
    document_router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
    source_artifact_router = APIRouter(prefix="/api/v1/source-artifacts", tags=["source-artifacts"])
    source_unit_router = APIRouter(prefix="/api/v1/source-units", tags=["source-units"])
    memory_router = APIRouter(prefix="/api/v1/memories", tags=["memories"])
    relation_discovery_router = APIRouter(prefix="/api/v1/relation-discovery", tags=["relation-discovery"])
    review_router = APIRouter(prefix="/api/v1/memory-reviews", tags=["memory-reviews"])
    entity_router = APIRouter(prefix="/api/v1/entities", tags=["entities"])
    gene_router = APIRouter(prefix="/api/v1/genes", tags=["genes"])
    source_router = APIRouter(prefix="/api/v1/sources", tags=["sources"])
    evaluation_router = APIRouter(
        prefix="/api/v1/agent-evaluations",
        tags=["agent-evaluations"],
    )
    source_list_router = APIRouter(prefix="/api/v1/source-list", tags=["sources"])
    agent_session_router = APIRouter(prefix="/api/v1/agent-sessions", tags=["agent-sessions"])
    hook_router = APIRouter(prefix="/api/v1/hooks", tags=["hooks"])
    recent_change_router = APIRouter(prefix="/api/v1/recent-changes", tags=["recent-changes"])
    model_settings_router = build_model_settings_router(
        get_db=get_db,
        get_model_settings=get_model_settings,
    )
    projects_router = APIRouter(prefix="/api/v1/projects", tags=["projects"])
    local_agent_router = APIRouter(
        prefix="/api/cloud/local-agent",
        tags=["local-agent"],
    )

    async def get_search_engine(request: Request, db: Database, config: AppConfig):
        from memforge.memory.audit import AuditContext, MemoryAuditLogger

        state = request.app.state
        engine = getattr(state, "memory_search_engine", None)
        if (
            engine is not None
            and getattr(engine, "serves_another_index", False)
            and time.monotonic() - getattr(state, "memory_search_engine_built_at", 0.0) > _SEARCH_ENGINE_REBUILD_S
        ):
            # The vector index was built with another embedding model than
            # this engine's. That model may be the one in use by now.
            engine = None
        if engine is None:
            runtime_provider = state.runtime_provider
            engine = await runtime_provider.build_search_engine(
                db,
                config,
                audit_logger=MemoryAuditLogger(db, default_context=AuditContext(actor_type="admin")),
            )
            state.memory_search_engine = engine
            state.memory_search_engine_built_at = time.monotonic()
        return engine

    # ===================================================================
    # 1. Health & Stats
    # ===================================================================

    @health_router.get(CAPABILITY_PATH)
    async def capabilities() -> JSONResponse:
        """Declare the origin-scoped OSS protocol contract."""
        return JSONResponse(
            capability_document(Edition.OSS),
            headers={
                "Cache-Control": "public, max-age=300",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @health_router.get("/api/v1/workspaces")
    async def list_workspaces(request: Request) -> dict[str, object]:
        """Return the single workspace exposed by a self-hosted runtime."""
        current_workspace_id = str(request.app.state.workspace_id)
        return {
            "workspaces": [
                {
                    "workspace_id": current_workspace_id,
                    "name": "Local workspace",
                    "role": "owner",
                    "status": "active",
                    "selectable": True,
                }
            ]
        }

    @health_router.get("/api/v1/health", response_model=HealthResponse)
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
            failed_items=(_runtime_component_to_response(report.failed_items) if report.failed_items else None),
            models=await model_health_response(request.app.state.model_settings, db),
        )

    @health_router.get("/api/v1/stats", response_model=StatsResponse)
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

    async def _readable_document_inputs(request: Request, db: Database, doc_id: str) -> list[SourceUnitInput]:
        """The stored copies of the Document from Sources the caller can read, most recently recorded first."""
        if await db.get_document(doc_id) is None:
            raise HTTPException(status_code=404, detail="Document not found")
        principal = resolve_request_principal(request)
        readable: list[SourceUnitInput] = []
        for unit_input in await db.list_document_source_unit_inputs(doc_id):
            source = await db.get_source(unit_input.source_id)
            if source is not None and source_is_discoverable(source, viewer_id=principal):
                readable.append(unit_input)
        return readable

    async def _document_artifact_input(
        request: Request,
        db: Database,
        doc_id: str,
        kind: str,
        artifact_store: DocumentArtifactStore,
        *,
        missing_detail: str,
    ) -> SourceUnitInput:
        """The most recently recorded readable copy whose object of this kind is still stored."""
        for unit_input in await _readable_document_inputs(request, db, doc_id):
            if select_stored_input_artifact(
                unit_input, kind, config, artifact_store, resource_path=document_resource_path(doc_id),
            ) is not None:
                return unit_input
        raise HTTPException(status_code=404, detail=missing_detail)

    async def _readable_source_unit_input(request: Request, db: Database, source_unit_id: str) -> SourceUnitInput:
        """The stored input of the Unit's current revision, when the caller can read its Source."""
        unit_input = await db.get_source_unit_input(source_unit_id)
        source = await db.get_source(unit_input.source_id) if unit_input is not None else None
        if (
            unit_input is None
            or source is None
            or not source_is_discoverable(source, viewer_id=resolve_request_principal(request))
        ):
            raise HTTPException(status_code=404, detail="Source Unit content not found")
        return unit_input

    def _artifact_manifest(
        unit_input: SourceUnitInput,
        doc: Any | None,
        *,
        resource_path: str,
        artifact_store: DocumentArtifactStore,
    ) -> dict[str, Any]:
        artifacts = list_stored_input_artifacts(
            unit_input, config, artifact_store, resource_path=resource_path,
        )
        return {
            "doc_id": unit_input.document_id,
            "source_id": unit_input.source_id,
            "source_unit_id": unit_input.source_unit_id,
            "title": doc.title if doc is not None else unit_input.item.title,
            "source_url": doc.source_url if doc is not None else unit_input.item.source_url,
            "artifacts": {kind: artifact.metadata() for kind, artifact in artifacts.items()},
        }

    def _stored_input_artifact_response(
        request: Request,
        unit_input: SourceUnitInput,
        kind: str,
        *,
        resource_path: str,
        artifact_store: DocumentArtifactStore,
        missing_detail: str,
    ) -> Response:
        artifact = select_stored_input_artifact(
            unit_input, kind, config, artifact_store, resource_path=resource_path,
        )
        if artifact is None:
            raise HTTPException(status_code=404, detail=missing_detail)
        content = b"" if request.method == "HEAD" else artifact_store.read_artifact(artifact.uri)
        return Response(
            content=content,
            media_type=artifact.media_type,
            headers={"Content-Disposition": _inline_content_disposition(artifact.filename)},
        )

    @document_router.get("/{doc_id}/artifacts")
    async def list_document_artifact_manifest(
        doc_id: str,
        request: Request,
        db: Database = Depends(get_db),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """List service-readable artifacts of the newest readable copy of a Document that still has any."""
        resource_path = document_resource_path(doc_id)
        readable = await _readable_document_inputs(request, db, doc_id)
        if not readable:
            raise HTTPException(status_code=404, detail="Document artifact not found")
        unit_input = next(
            (
                candidate
                for candidate in readable
                if list_stored_input_artifacts(candidate, config, artifact_store, resource_path=resource_path)
            ),
            readable[0],
        )
        return _artifact_manifest(
            unit_input,
            await db.get_document(doc_id),
            resource_path=document_resource_path(doc_id),
            artifact_store=artifact_store,
        )

    @document_router.api_route("/{doc_id}/artifacts/{kind}", methods=["GET", "HEAD"])
    async def get_document_artifact(
        doc_id: str,
        kind: str,
        request: Request,
        db: Database = Depends(get_db),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """Serve an explicit artifact kind of the newest stored copy of a Document that has it."""
        unit_input = await _document_artifact_input(
            request, db, doc_id, kind, artifact_store, missing_detail="Document artifact not found",
        )
        return _stored_input_artifact_response(
            request, unit_input, kind,
            resource_path=document_resource_path(doc_id),
            artifact_store=artifact_store,
            missing_detail="Document artifact not found",
        )

    @source_artifact_router.api_route(
        "/{observation_revision_id}",
        methods=["GET", "HEAD"],
    )
    async def get_source_artifact(
        observation_revision_id: str,
        request: Request,
        db: Database = Depends(get_db),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """Serve one exact current Artifact revision under source access."""

        artifact = await db.get_source_artifact_revision(observation_revision_id)
        if artifact is None:
            raise HTTPException(status_code=404, detail="Source Artifact not found")
        source = await db.get_source(artifact.source_id)
        principal = resolve_request_principal(request)
        if (
            source is None
            or source.get("access_state") != "active"
            or (source.get("access_policy") == "private" and source.get("owner_user_id") != principal)
        ):
            raise HTTPException(status_code=404, detail="Source Artifact not found")
        stored = artifact_store.get_artifact(artifact.uri, artifact.media_type)
        if stored is None or stored.size_bytes is not None and stored.size_bytes != artifact.size_bytes:
            raise HTTPException(status_code=409, detail="Source Artifact integrity mismatch")
        headers = {
            "Content-Disposition": _inline_content_disposition(artifact.filename),
            "Content-Length": str(artifact.size_bytes),
            "ETag": f'"sha256:{artifact.sha256}"',
            "X-Content-SHA256": artifact.sha256,
        }
        if request.method == "HEAD":
            return Response(content=b"", media_type=artifact.media_type, headers=headers)
        return StreamingResponse(
            _verified_artifact_chunks(
                artifact_store,
                artifact.uri,
                expected_size=artifact.size_bytes,
                expected_sha256=artifact.sha256,
            ),
            media_type=artifact.media_type,
            headers=headers,
        )

    @document_router.api_route("/{doc_id}/content", methods=["GET", "HEAD"])
    async def get_document_content(
        doc_id: str,
        request: Request,
        db: Database = Depends(get_db),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """Serve normalized source content through the API for Docker/SaaS clients."""
        unit_input = await _document_artifact_input(
            request, db, doc_id, "content", artifact_store, missing_detail="Document content artifact not found",
        )
        return _stored_input_artifact_response(
            request, unit_input, "content",
            resource_path=document_resource_path(doc_id),
            artifact_store=artifact_store,
            missing_detail="Document content artifact not found",
        )

    @document_router.api_route("/{doc_id}/pdf", methods=["GET", "HEAD"])
    async def get_document_pdf(
        doc_id: str,
        request: Request,
        db: Database = Depends(get_db),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """Serve a stored source PDF through the API for Docker/SaaS clients."""
        unit_input = await _document_artifact_input(
            request, db, doc_id, "pdf", artifact_store, missing_detail="Document PDF artifact not found",
        )
        return _stored_input_artifact_response(
            request, unit_input, "pdf",
            resource_path=document_resource_path(doc_id),
            artifact_store=artifact_store,
            missing_detail="Document PDF artifact not found",
        )

    @source_unit_router.get("/{source_unit_id}/artifacts")
    async def list_source_unit_artifact_manifest(
        source_unit_id: str,
        request: Request,
        db: Database = Depends(get_db),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """List service-readable artifacts of the stored input of one Source Unit."""
        unit_input = await _readable_source_unit_input(request, db, source_unit_id)
        return _artifact_manifest(
            unit_input,
            await db.get_document(unit_input.document_id),
            resource_path=source_unit_resource_path(source_unit_id),
            artifact_store=artifact_store,
        )

    @source_unit_router.api_route("/{source_unit_id}/artifacts/{kind}", methods=["GET", "HEAD"])
    async def get_source_unit_artifact(
        source_unit_id: str,
        kind: str,
        request: Request,
        db: Database = Depends(get_db),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """Serve an explicit artifact kind of the stored input of one Source Unit."""
        unit_input = await _readable_source_unit_input(request, db, source_unit_id)
        return _stored_input_artifact_response(
            request, unit_input, kind,
            resource_path=source_unit_resource_path(source_unit_id),
            artifact_store=artifact_store,
            missing_detail="Source Unit artifact not found",
        )

    @source_unit_router.api_route("/{source_unit_id}/content", methods=["GET", "HEAD"])
    async def get_source_unit_content(
        source_unit_id: str,
        request: Request,
        db: Database = Depends(get_db),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """Serve the normalized content one Source stored for its Unit."""
        unit_input = await _readable_source_unit_input(request, db, source_unit_id)
        return _stored_input_artifact_response(
            request, unit_input, "content",
            resource_path=source_unit_resource_path(source_unit_id),
            artifact_store=artifact_store,
            missing_detail="Source Unit content artifact not found",
        )

    @source_unit_router.api_route("/{source_unit_id}/pdf", methods=["GET", "HEAD"])
    async def get_source_unit_pdf(
        source_unit_id: str,
        request: Request,
        db: Database = Depends(get_db),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """Serve the PDF one Source stored for its Unit."""
        unit_input = await _readable_source_unit_input(request, db, source_unit_id)
        return _stored_input_artifact_response(
            request, unit_input, "pdf",
            resource_path=source_unit_resource_path(source_unit_id),
            artifact_store=artifact_store,
            missing_detail="Source Unit PDF artifact not found",
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

    @memory_router.post("/recent")
    async def list_recent_memories(
        req: RecentMemoryListRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
    ):
        """List current Memories in one resolved half-open time window."""
        from memforge.storage.adapters.context import AccessScope

        try:
            engine = await get_search_engine(request, db, config)
            user_id = resolve_request_principal(request)
            if req.source_ids:
                available = await db.list_searchable_source_ids_for_user(req.source_ids, user_id)
                if any(source_id not in available for source_id in req.source_ids):
                    raise HTTPException(status_code=400, detail="unknown_or_unavailable_source_id")
            result = await engine.list_recent_memories(
                source_filter=(MemorySourceFilter(source_ids=tuple(req.source_ids)) if req.source_ids else None),
                time_range=req.time_range.to_time_range(),
                memory_types=req.memory_types,
                page_size=req.page_size,
                cursor=req.cursor,
                request_scope=AccessScope(
                    user_id=user_id,
                    include_private=req.include_private,
                    allowed_statuses=("active",),
                    active_project=None,
                    scope_mode="workspace",
                ),
            )
            return _json_ready(result)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.warning("Recent-memory listing failed: %s", exc, exc_info=True)
            raise HTTPException(status_code=503, detail=f"Recent-memory listing unavailable: {exc}") from exc

    @memory_router.post("/search", response_model=MemorySearchResponse, response_model_exclude_unset=True)
    async def search_memories(
        req: MemorySearchRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """Service-owned memory search used by local agent MCP proxies."""
        from memforge.memory.lifecycle import allowed_search_statuses
        from memforge.storage.adapters.context import AccessScope

        try:
            engine = await get_search_engine(request, db, config)
            user_id = resolve_request_principal(request)
            if req.source_filter is not None and req.source_filter.source_ids:
                available = await db.list_searchable_source_ids_for_user(
                    req.source_filter.source_ids,
                    user_id,
                )
                if any(source_id not in available for source_id in req.source_filter.source_ids):
                    raise HTTPException(status_code=400, detail="unknown_or_unavailable_source_id")
            allowed_statuses = (
                (normalize_memory_status(req.status),)
                if req.status
                else allowed_search_statuses(req.include_superseded)
            )
            # `project-first` and `workspace` modes weight cross-project hits
            # but keep them visible at the predicate. `project` mode is the
            # only mode that narrows the workspace branch, and it narrows to
            # `{active_project, SHARED}` inside the predicate itself.
            scope = AccessScope(
                user_id=user_id,
                include_private=req.include_private,
                allowed_statuses=allowed_statuses,
                active_project=req.active_project,
                scope_mode=req.scope_mode,
                active_repo_identifier=req.active_repo_identifier,
            )
            result = await engine.search(
                query=req.query,
                memory_types=req.memory_types,
                time_range=req.time_range.to_time_range() if req.time_range is not None else None,
                entities=req.entities,
                include_superseded=req.include_superseded,
                top_k=req.top_k,
                source_filter=(req.source_filter.to_source_filter() if req.source_filter is not None else None),
                request_scope=scope,
                offset=req.offset,
                intent=req.intent,
            )
            return _json_ready(result)
        except HTTPException:
            raise
        except Exception as e:
            logger.warning("Search failed: %s", e, exc_info=True)
            raise HTTPException(
                status_code=503,
                detail=f"Search unavailable: {e}",
            ) from e

    @memory_router.get("/relations", response_model=RelationListResponse)
    async def list_memory_relations(
        request: Request,
        label: RelationLabelLiteral | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        db: Database = Depends(get_db),
    ):
        """List the relations current for the caller, newest decision first.

        A view, not a work queue. ``label`` filters by the recorded label.
        """

        views, total = await list_relation_pairs(
            db,
            _workspace_default_scope(request, include_private=True),
            label=CrossDocumentRelationLabel(label) if label is not None else None,
            limit=limit,
            offset=offset,
        )
        return RelationListResponse(
            data=[RelationPairDetail(**asdict(view)) for view in views],
            total=total,
            limit=limit,
            offset=offset,
        )

    @memory_router.post(
        "/{memory_id}/relations/{counterpart_memory_id}/dismissal",
        response_model=RelationDismissalResponse,
    )
    async def dismiss_memory_relation(
        memory_id: str,
        counterpart_memory_id: str,
        body: RelationDismissalRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Hide one relation until either Memory changes; no lifecycle effect."""

        scope = _workspace_default_scope(request, include_private=True)
        try:
            dismissal = await dismiss_relation(
                db,
                memory_id=memory_id,
                counterpart_memory_id=counterpart_memory_id,
                label=CrossDocumentRelationLabel(body.label),
                expected_content_hash=body.expected_content_hash,
                counterpart_expected_content_hash=body.counterpart_expected_content_hash,
                actor=resolve_request_principal(request),
                scope=scope,
                note=body.note,
            )
        except RelationDismissalConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except LookupError as exc:
            raise HTTPException(status_code=404, detail="Relation not found") from exc
        return RelationDismissalResponse(
            dismissal_id=dismissal.id,
            memory_low_id=dismissal.memory_low_id,
            memory_high_id=dismissal.memory_high_id,
            label=dismissal.label.value,
            dismissed_by=dismissal.dismissed_by,
            dismissed_at=dismissal.dismissed_at,
        )

    @memory_router.delete(
        "/{memory_id}/relations/{counterpart_memory_id}/dismissal",
        response_model=RelationRestoreResponse,
    )
    async def restore_memory_relation(
        memory_id: str,
        counterpart_memory_id: str,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Undo the dismissals in force for one pair of Memories."""

        scope = _workspace_default_scope(request, include_private=True)
        await _require_visible_relation_pair(db, memory_id, counterpart_memory_id, scope)
        restored = await db.restore_cross_document_relation_dismissals(
            memory_id=memory_id,
            counterpart_memory_id=counterpart_memory_id,
            actor=resolve_request_principal(request),
        )
        if not restored:
            raise HTTPException(status_code=404, detail="Relation dismissal not found")
        return RelationRestoreResponse(restored_dismissal_ids=[item.id for item in restored])

    @memory_router.get("/{memory_id}/evidence/{evidence_unit_id}/resource")
    @memory_router.get("/{memory_id}/evidence/{evidence_unit_id}/references/{reference_id}/resource")
    async def get_memory_evidence_resource(
        memory_id: str, evidence_unit_id: str, request: Request,
        db: Database = Depends(get_db),
        reference_id: str | None = None,
    ):
        """Read verified stored citations and the complete immutable Unit source material."""
        if memory_id not in await _filter_visible_ids(db, [memory_id], _lifecycle_visibility_scope(request)):
            raise HTTPException(status_code=404, detail="Evidence not found")
        group = next((unit for unit in await db.get_memory_evidence_units(
                         memory_id, include_historical=True, context_reference_id=reference_id)
                      if unit.evidence_unit_id == evidence_unit_id), None)
        if group is None or not group.visible_to(resolve_request_principal(request)):
            raise HTTPException(status_code=404, detail="Evidence not found")
        if reference_id is not None and not any(item.reference_id == reference_id for item in group.items):
            raise HTTPException(status_code=404, detail="Evidence not found")
        source = await db.get_source(group.source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Evidence not found")
        _require_source_discoverability(request, source)
        try:
            pinned = (await db.get_source_unit_revision_projection(group.source_unit_id, group.source_unit_revision_id)
                      if group.source_unit_revision_id is not None else None)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={
                "code": "pinned_evidence_unavailable", "reason": "The historical Unit manifest is invalid.",
            }) from exc
        if (pinned is None or pinned.source_id != group.source_id
                or len(pinned.source_unit_revisions) != 1
                or pinned.source_unit_revisions[0].id != group.source_unit_revision_id
                or pinned.source_unit_revisions[0].source_unit_id != group.source_unit_id):
            raise HTTPException(status_code=409, detail={
                "code": "pinned_evidence_unavailable", "reason": "The exact historical Unit is unavailable.",
            })
        member_ids = {revision.id for revision in pinned.observation_revisions}
        if any(item.anchor.observation_revision_id not in member_ids
               for item in group.items if item.grants_support):
            raise HTTPException(status_code=409, detail={
                "code": "pinned_evidence_unavailable", "reason": "A supporting part is outside its Unit revision.",
            })
        material = {revision.id: revision for revision in pinned.observation_revisions}
        context_revision_ids = {item.anchor.observation_revision_id for item in group.items if not item.grants_support}
        material.update(await db.get_source_observation_revisions(context_revision_ids))
        if not context_revision_ids <= material.keys():
            raise HTTPException(status_code=409, detail={"code": "pinned_evidence_unavailable"})
        payload = {
            "memory_id": memory_id,
            "selected_reference_id": reference_id,
            "evidence": _memory_evidence_unit_detail(group, document=None, memory_id=memory_id).model_dump(),
            "source_material": [{
                "observation_id": revision.observation_id,
                "observation_revision_id": revision.id,
                "content": revision.content,
                "representation_profile": _json_ready(revision.evidence_profile),
                "support_unit_member": revision.id in member_ids,
            } for revision in sorted(material.values(), key=lambda item: item.id)],
        }
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return Response(content=body, media_type="application/json", headers={
            "Cache-Control": "private, no-store",
            "X-Content-SHA256": hashlib.sha256(body).hexdigest(),
            "Content-Disposition": _inline_content_disposition(f"evidence-{evidence_unit_id}.json"),
        })

    @memory_router.get("/{memory_id}", response_model=MemoryDetailResponse)
    async def get_memory(
        memory_id: str,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        """Get full memory detail including provenance (linked source documents).

        Every lifecycle status is readable, so a superseded, retired or
        pending Memory still opens from the list, a review or its replacement.
        """
        scope = _lifecycle_visibility_scope(request)
        mem = await db.get_memory(memory_id)
        if not mem:
            raise HTTPException(status_code=404, detail="Memory not found")
        # Default-deny: a row the caller cannot see by the access predicate is
        # indistinguishable from a missing row to that caller.
        visible = await _filter_visible_ids(db, [memory_id], scope)
        if memory_id not in visible:
            raise HTTPException(status_code=404, detail="Memory not found")

        evidence_groups = await _memory_evidence_details(
            db,
            memory_id,
            request=request,
            config=config,
            artifact_store=artifact_store,
        )

        # Fetch linked entity names.
        entity_names = await db.get_memory_entity_names(memory_id)

        origin_info = (await _origin_source_types(db, [memory_id])).get(memory_id, (None, None))
        origin_source_type, origin_client = origin_info
        sources = (await db.get_memory_source_refs_many([memory_id], scope)).get(memory_id, ())
        relations = (await read_memory_relations(db, [memory_id], scope)).get(memory_id, ())
        dismissed = await dismissed_relation_views(
            db,
            memory_id,
            await db.list_active_cross_document_relation_dismissals(
                memory_id,
                scope,
                limit=MAX_RELATIONS_PER_MEMORY,
            ),
            scope,
        )

        return MemoryDetailResponse(
            id=mem.id,
            memory_type=mem.memory_type,
            content=mem.content,
            content_hash=mem.content_hash,
            visibility=mem.visibility,
            owner_user_id=mem.owner_user_id,
            project_key=mem.project_key,
            corroboration_count=mem.corroboration_count,
            valid_from=_dt_iso(mem.valid_from),
            valid_until=_dt_iso(mem.valid_until),
            superseded_by=mem.superseded_by,
            status=mem.status,
            retirement_reason=mem.retirement_reason,
            retired_at=_dt_iso(mem.retired_at),
            superseded_at=_dt_iso(mem.superseded_at),
            replacement_reason=mem.replacement_reason,
            replacement_kind=mem.replacement_kind,
            created_at=_dt_iso(mem.created_at),
            updated_at=_dt_iso(mem.updated_at),
            entity_refs=entity_names,
            evidence=evidence_groups,
            relations=[MemoryRelationDetail(**asdict(item)) for item in relations],
            relation_notice=relation_notice(relations),
            dismissed_relations=[DismissedRelationDetail(**asdict(item)) for item in dismissed],
            origin_source_type=origin_source_type,
            origin_client=origin_client,
            sources=[MemorySourceRefDetail(**asdict(source)) for source in sources],
            source_backed=await is_source_backed(db, memory_id),
            open_review_id=(await read_open_review_ids(db, [mem], scope=_lifecycle_visibility_scope(request))).get(memory_id),
        )

    # -- Memory update (admin actions) --

    # Deprecated (admin-ui-v1): the V1 memory detail page retires through this
    # route with a fixed reason and no content hash, visibility check or actor.
    # The admin UI uses POST /memories/{id}/retire; this route is removed with
    # the V1 admin UI.
    @memory_router.put(
        "/{memory_id}",
        deprecated=True,
        dependencies=[Depends(refuse_memory_writes_during_reindex)],
    )
    async def update_memory(
        memory_id: str,
        req: MemoryUpdateRequest = Body(...),
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        """Update a memory's content or status (admin override)."""
        memory = await db.get_memory(memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail="Memory not found")

        if req.content:
            memory_store = await _build_memory_store(db, config, runtime_provider)
            try:
                await memory_store.update_memory(
                    memory_id,
                    new_content=req.content,
                )
            except ValueError as exc:
                if "active source support" in str(exc):
                    raise HTTPException(
                        status_code=409,
                        detail="source_backed_memory_requires_lifecycle_review",
                    ) from exc
                raise
        if req.status is not None:
            if req.status not in ("active", "superseded", "retired", "decayed", "pending_review"):
                raise HTTPException(status_code=400, detail=f"Invalid status: {req.status}")
            status = normalize_memory_status(req.status)
            memory_store = await _build_memory_store(db, config, runtime_provider)
            if status == "retired":
                try:
                    await memory_store.retire_memory(memory_id, reason="admin_hidden")
                except ValueError as exc:
                    if "active source support" in str(exc):
                        raise HTTPException(
                            status_code=409,
                            detail="source_backed_memory_requires_lifecycle_review",
                        ) from exc
                    raise
            elif status == "pending_review":
                try:
                    await memory_store.mark_pending_review(memory_id, reason="admin_hidden")
                except ValueError as exc:
                    if "active source support" in str(exc):
                        raise HTTPException(
                            status_code=409,
                            detail="source_backed_memory_requires_lifecycle_review",
                        ) from exc
                    raise
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

    @memory_router.post(
        "/create",
        response_model=MemoryCreateResponse,
        dependencies=[Depends(refuse_memory_writes_during_reindex)],
    )
    async def create_memory_route(
        req: MemoryCreateRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        service = await _build_lifecycle_service(
            db,
            config,
            runtime_provider,
            audit_context=_request_audit_context(request),
        )
        try:
            result = await service.create_memory(
                content=req.content,
                provenance=req.provenance,
                memory_type=req.memory_type,
                owner_user_id=resolve_request_principal(request),
                client=req.client,
                repo_identifier=req.repo_identifier,
                idempotency_key=req.idempotency_key,
            )
        except MemoryLifecycleConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return MemoryCreateResponse(memory_id=result.memory_id, status=result.status)

    @memory_router.post(
        "/maintenance-closures/report",
        response_model=MemoryMaintenanceClosureReportResponse,
    )
    async def report_memory_maintenance_closures(
        req: MemoryMaintenanceClosureReportRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        operator_actor_id = _require_maintenance_operator(request)
        service = await _build_lifecycle_service(
            db,
            config,
            runtime_provider,
            audit_context=AuditContext(
                actor_type="maintenance_operator",
                actor_id=operator_actor_id,
            ),
        )
        try:
            report = await service.report_maintenance_closures(
                tuple(MaintenanceClosureEntry(**item.model_dump()) for item in req.entries),
                operator_actor_id=operator_actor_id,
            )
        except MemoryLifecycleConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return MemoryMaintenanceClosureReportResponse(
            report_id=report.id,
            ready_count=report.ready_count,
            entries=[
                MemoryMaintenanceClosureReportItemResponse(
                    memory_id=item.memory_id,
                    disposition=item.disposition,
                )
                for item in report.entries
            ],
        )

    @memory_router.post(
        "/maintenance-closures/apply",
        response_model=MemoryMaintenanceClosureApplyResponse,
        dependencies=[Depends(refuse_memory_writes_during_reindex)],
    )
    async def apply_memory_maintenance_closures(
        req: MemoryMaintenanceClosureApplyRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        operator_actor_id = _require_maintenance_operator(request)
        service = await _build_lifecycle_service(
            db,
            config,
            runtime_provider,
            audit_context=AuditContext(
                actor_type="maintenance_operator",
                actor_id=operator_actor_id,
            ),
        )
        try:
            receipts = await service.apply_maintenance_closures(
                tuple(MaintenanceClosureEntry(**item.model_dump()) for item in req.entries),
                expected_report_id=req.expected_report_id,
                operator_actor_id=operator_actor_id,
            )
        except MemoryLifecycleConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return MemoryMaintenanceClosureApplyResponse(
            report_id=req.expected_report_id,
            receipts=[
                MemoryMaintenanceClosureReceiptResponse(
                    receipt_id=item.id,
                    report_id=item.report_id,
                    memory_id=item.memory_id,
                    status=item.status,
                    authority="maintenance_operator",
                    operator_actor_id=item.operator_actor_id,
                )
                for item in receipts
            ],
        )

    @memory_router.post(
        "/cross-source-review-conversion/report",
        response_model=CrossSourceReviewConversionReportResponse,
    )
    async def report_cross_source_review_conversion(
        request: Request,
        body: CrossSourceReviewLabelsRequest = Body(default_factory=CrossSourceReviewLabelsRequest),
        db: Database = Depends(get_db),
    ):
        """Classify every Cross-Source Conflict Review for conversion; writes nothing."""

        _require_maintenance_operator(request)
        try:
            plan = await build_conversion_plan(
                db,
                label_overrides=body.label_overrides,
                max_attempts=DEFAULT_RELATION_DISCOVERY_BUDGET.max_attempts,
            )
        except ReviewLabelOverrideError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return CrossSourceReviewConversionReportResponse(
            report_id=plan.report_id,
            review_count=len(plan.review_ids),
            relation_labels={item.review_id: item.label for item in plan.relations},
            dismissal_review_ids=[item.review_id for item in plan.dismissals],
            discarded_review_ids=list(plan.discarded_review_ids),
            rerun_review_ids=list(plan.rerun_review_ids),
            nothing_to_rerun_review_ids=list(plan.nothing_to_rerun_review_ids),
            exhausted_work_ids=list(plan.exhausted_work_ids),
            planned=plan.planned_counts,
            applied=await db.get_cross_source_review_conversion(plan.report_id) is not None,
        )

    @memory_router.post(
        "/cross-source-review-conversion/apply",
        response_model=CrossSourceReviewConversionReceiptResponse,
        dependencies=[Depends(refuse_memory_writes_during_reindex)],
    )
    async def apply_cross_source_review_conversion(
        body: CrossSourceReviewConversionApplyRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Apply one reported conversion; the same report applies once."""

        actor = _require_maintenance_operator(request)
        try:
            receipt = await apply_conversion(
                db,
                report_id=body.report_id,
                label_overrides=body.label_overrides,
                actor=actor,
                max_attempts=DEFAULT_RELATION_DISCOVERY_BUDGET.max_attempts,
            )
        except ReviewLabelOverrideError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except CrossSourceReviewConversionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return _conversion_receipt_response(receipt)

    @memory_router.post(
        "/cross-source-review-conversion/delete",
        response_model=CrossSourceReviewDeletionResponse,
    )
    async def delete_converted_cross_source_reviews(
        body: CrossSourceReviewDeletionRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Delete the Review rows of a complete conversion after the evaluation set is frozen."""

        from memforge.evals.cross_document_relation_cases import RELATION_CASE_POLICY_VERSION

        actor = _require_maintenance_operator(request)
        cohort = await db.get_agent_evaluation_cohort(body.cohort_id)
        if cohort is None or cohort.selection_policy_version != RELATION_CASE_POLICY_VERSION:
            raise HTTPException(status_code=409, detail="relation evaluation cohort not found")
        try:
            deleted = await delete_converted_reviews(db, report_id=body.report_id, actor=actor)
        except CrossSourceReviewConversionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return CrossSourceReviewDeletionResponse(report_id=body.report_id, deleted_review_count=deleted)

    @relation_discovery_router.get("/work", response_model=RelationDiscoveryWorkListResponse)
    async def list_relation_discovery_work(
        request: Request,
        state: Literal["exhausted", "completed", "failed"],
        error_code: str | None = None,
        updated_from: datetime | None = None,
        updated_to: datetime | None = None,
        classifier_version: str | None = None,
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
        db: Database = Depends(get_db),
    ):
        """List discovery work by state with its last error; exhausted work has no retry left."""

        _require_maintenance_operator(request)
        selection = _relation_discovery_selection(
            state=state,
            error_code=error_code,
            updated_from=updated_from,
            updated_to=updated_to,
            classifier_version=classifier_version,
        )
        work = await db.list_relation_discovery_work(selection, limit=limit, offset=offset)
        return RelationDiscoveryWorkListResponse(
            data=[_relation_discovery_work_detail(item) for item in work],
            total=await db.count_relation_discovery_work(selection),
            limit=limit,
            offset=offset,
        )

    @relation_discovery_router.post("/work/rerun", response_model=RelationDiscoveryRerunResponse)
    async def rerun_relation_discovery_work(
        body: RelationDiscoveryWorkFilter,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Queue selected completed or exhausted work again, auditing each item's last state."""

        actor = _require_maintenance_operator(request)
        selection = _relation_discovery_selection(**body.model_dump())
        return RelationDiscoveryRerunResponse(
            rerun_count=await db.rerun_relation_discovery_work(selection, actor=actor)
        )

    @memory_router.post(
        "/{memory_id}/retire",
        response_model=MemoryLifecycleResponse,
        dependencies=[Depends(refuse_memory_writes_during_reindex)],
    )
    async def retire_memory_route(
        memory_id: str,
        req: MemoryRetireRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        visible = await _filter_visible_ids(db, [memory_id], _workspace_default_scope(request, include_private=True))
        if memory_id not in visible:
            raise HTTPException(status_code=404, detail="Memory not found")
        service = await _build_lifecycle_service(
            db,
            config,
            runtime_provider,
            audit_context=_request_audit_context(request),
        )
        try:
            result = await service.retire_memory(
                memory_id,
                reason=req.reason,
                expected_content_hash=req.expected_content_hash,
                actor_user_id=resolve_request_principal(request),
            )
        except MemoryLifecycleNotFound:
            raise HTTPException(status_code=404, detail="Memory not found")
        except MemoryLifecycleConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return MemoryLifecycleResponse(memory_id=result.memory_id, status=result.status)

    @memory_router.post(
        "/{memory_id}/corrections/propose",
        response_model=MemoryCorrectionProposalResponse,
        dependencies=[Depends(refuse_memory_writes_during_reindex)],
    )
    async def propose_memory_correction_route(
        memory_id: str,
        req: MemoryCorrectionProposalRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        visible = await _filter_visible_ids(
            db,
            [memory_id],
            _workspace_default_scope(request, include_private=True),
        )
        if memory_id not in visible:
            raise HTTPException(status_code=404, detail="Memory not found")
        service = await _build_lifecycle_service(
            db,
            config,
            runtime_provider,
            audit_context=_request_audit_context(request),
        )
        try:
            result = await service.propose_memory_correction(
                memory_id,
                replacement_content=req.replacement_content,
                provenance=req.provenance,
                reason=req.reason,
                expected_content_hash=req.expected_content_hash,
                authority=_request_correction_authority(request),
                replacement_kind=req.replacement_kind,
            )
        except MemoryLifecycleNotFound:
            raise HTTPException(status_code=404, detail="Memory not found")
        except MemoryLifecycleConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        return MemoryCorrectionProposalResponse(
            memory_id=result.memory_id,
            replacement_memory_id=result.replacement_memory_id,
            outcome=result.outcome,
            status=result.status,
            replacement_kind=result.replacement_kind,
            review_id=result.review_id,
        )

    @memory_router.delete(
        "/{memory_id}/purge",
        dependencies=[Depends(refuse_memory_writes_during_reindex)],
    )
    async def purge_memory(
        memory_id: str,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        """Hard-purge a visible memory for privacy/compliance removal.

        Purging is irreversible: a private memory can be purged only by its
        owner, and a workspace memory only by a workspace administrator. A
        memory that an active source still supports is refused, because the
        next sync would extract it again; remove the content at the source or
        retire the source instead. Everyday removal goes through
        ``POST /memories/{memory_id}/retire``.
        """
        visible = await _filter_visible_ids(db, [memory_id], _lifecycle_visibility_scope(request))
        memory = await db.get_memory(memory_id) if memory_id in visible else None
        if memory is None:
            raise HTTPException(status_code=404, detail="Memory not found")
        if memory.visibility == Visibility.PRIVATE.value:
            if memory.owner_user_id != resolve_request_principal(request):
                raise HTTPException(status_code=403, detail="memory_owner_authority_required")
        else:
            _require_workspace_memory_administration(request)
        support_state = (await db.get_active_memory_support_states((memory_id,)))[memory_id]
        if support_state.unit_ids:
            raise HTTPException(status_code=409, detail="source_backed_memory_requires_lifecycle_review")

        memory_store = await _build_memory_store(
            db,
            config,
            runtime_provider,
            audit_context=_request_audit_context(request),
        )
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
        include_private: bool = Query(default=True, description=INCLUDE_PRIVATE_DESCRIPTION),
        limit: int = 50,
        offset: int = 0,
        db: Database = Depends(get_db),
    ):
        """List memories with pagination and filters.

        Filters: type (fact/decision/convention/procedure), status, source,
        project, free-text search. Supports limit/offset pagination.

        Without ``status`` the list holds active Memories, so its total for a
        project equals that project's ``memory_count``; ``status`` lists the
        Memories in exactly that lifecycle status instead.

        The access predicate gates every row: workspace rows are visible
        across every project (the ranker handles project relevance, not
        the predicate), and the caller's own private rows are included
        unless ``include_private=false``. Other users' private rows are
        never listed.
        """
        scope = _workspace_default_scope(
            request,
            include_private=include_private,
            allowed_statuses=(normalize_memory_status(status),) if status else ACTIVE_MEMORY_STATUSES,
        )
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
            data=[
                _memory_to_response(
                    m,
                    *page.origins.get(m.id, (None, None)),
                    sources=page.sources.get(m.id, ()),
                    relations=page.relations.get(m.id, ()),
                    open_review_id=page.open_reviews.get(m.id),
                )
                for m in page.memories
            ],
            total=page.total,
            limit=limit,
            offset=offset,
        )

    # ===================================================================
    # 3. Entity Endpoints
    # ===================================================================

    @entity_router.get("", response_model=EntityListResponse)
    async def list_entities(
        request: Request,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Database = Depends(get_db),
    ):
        """List the Entities the caller can discover, with optional name search."""
        entity_rows, total = await db.list_entities(
            scope=_workspace_default_scope(request, include_private=True),
            search=search,
            limit=limit,
            offset=offset,
        )
        entities = [
            EntityResponse(
                id=ent.id,
                canonical_name=ent.canonical_name,
                display_name=ent.display_name,
                created_at=_dt_iso(ent.created_at),
            )
            for ent in entity_rows
        ]

        return EntityListResponse(data=entities, total=total)

    @entity_router.get("/{entity_id}", response_model=EntityDetailResponse)
    async def get_entity(entity_id: int, request: Request, db: Database = Depends(get_db)):
        """Get entity detail with aliases and the count of linked Memories the caller can query."""
        ent, linked_count = await _require_visible_entity(request, db, entity_id)

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

        return EntityDetailResponse(
            id=ent.id,
            canonical_name=ent.canonical_name,
            display_name=ent.display_name,
            created_at=_dt_iso(ent.created_at),
            aliases=alias_responses,
            linked_memory_count=linked_count,
            can_curate=can_manage_workspace(resolve_request_workspace_role(request)),
        )

    @entity_router.post("/merge", response_model=EntityMergeResponse)
    async def merge_entities(
        req: MergeEntitiesRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Merge two entities: reassign all references from source to target.

        All memory_entities rows, aliases, and document references pointing to
        source_id are moved to target_id. The source entity is then deleted.
        Both entities must be discoverable by the caller, and only a workspace
        administrator may merge.
        """
        if req.source_id == req.target_id:
            raise HTTPException(
                status_code=400,
                detail="Source and target entities must differ",
            )
        _require_entity_curation(request)
        await _require_visible_entity(request, db, req.source_id, missing_detail="Source entity not found")
        await _require_visible_entity(request, db, req.target_id, missing_detail="Target entity not found")
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

    @entity_router.get("/{entity_id}/aliases", response_model=EntityAliasListResponse)
    async def list_entity_aliases(
        entity_id: int,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """List all aliases for an entity."""
        await _require_visible_entity(request, db, entity_id)

        aliases = await db.get_aliases_for_entity(entity_id)
        return EntityAliasListResponse(
            data=[
                EntityAliasResponse(
                    alias=a.alias,
                    alias_normalized=a.alias_normalized,
                    source=a.source,
                    created_at=_dt_iso(a.created_at),
                )
                for a in aliases
            ]
        )

    @entity_router.post("/{entity_id}/aliases", response_model=EntityAliasAddedResponse)
    async def add_entity_alias(
        entity_id: int,
        req: AddAliasRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Add a workspace-wide manual alias for an entity."""
        _require_entity_curation(request)
        await _require_visible_entity(request, db, entity_id)

        normalized = canonicalize_entity_name(req.alias)
        await db.insert_alias(
            alias=req.alias,
            alias_normalized=normalized,
            canonical_id=entity_id,
            source="admin_manual",
        )
        return {"ok": True, "alias": req.alias, "alias_normalized": normalized}

    @entity_router.delete("/{entity_id}/aliases/{alias}", response_model=OkResponse)
    async def remove_entity_alias(
        entity_id: int,
        alias: str,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Remove a workspace-wide alias from an entity."""
        _require_entity_curation(request)
        await _require_visible_entity(request, db, entity_id)

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
                execution_kinds=list(g.execution_kinds),
                managed=source_type_descriptor(g.name).family is None,
            )
            for g in genes
        ]

    @gene_router.get("/{name}/config-schema", response_model=GeneConfigSchemaResponse)
    async def get_gene_config_schema(name: str):
        """The form description of a source type, which the admin UI draws without per-type code.

        Besides what the Gene declares, each field says whether it is a scope
        field and whether it is fixed after create, from the type's descriptor.
        """
        if name not in GENE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Gene '{name}' not found")

        schema = GENE_REGISTRY[name].config_schema()
        descriptor = source_type_descriptor(name)
        scope_fields = set(descriptor.scope_fields)
        fixed_fields = set(descriptor.local_agent.immutable_execution_fields) if descriptor.local_agent else set()

        def conditions(items: Iterable[ConfigCondition]) -> list[ConfigConditionResponse]:
            return [ConfigConditionResponse(field=c.field, values=list(c.values)) for c in items]

        return GeneConfigSchemaResponse(
            groups=[
                ConfigGroupResponse(
                    key=g.key,
                    label=g.label,
                    order=g.order,
                    description=g.description,
                    purpose=g.purpose,
                    panels=[ConfigPanelResponse(name=p.name, visible_when=conditions(p.visible_when)) for p in g.panels],
                )
                for g in schema.groups
            ],
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
                    choices=[
                        ConfigOptionResponse(
                            value=o.value,
                            label=o.label,
                            help_text=o.help_text,
                            visible_when=conditions(o.visible_when),
                        )
                        for o in f.choices
                    ],
                    visible_when=conditions(f.visible_when),
                    widget=f.widget,
                    advanced=f.advanced,
                    scope=f.key in scope_fields,
                    fixed_after_create=f.key in fixed_fields,
                    options=[o.value for o in f.choices],
                )
                for f in schema.fields
            ],
            project_field=schema.project_field,
        )

    @gene_router.post("/{name}/config-check", response_model=ConfigCheckResponse)
    async def check_gene_config(
        name: str,
        req: ConfigCheckRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """What saving this configuration would store and refuse, without saving it.

        It runs save's own steps (``complete`` and every configuration check)
        on a copy, and writes nothing: no Source, no inbox, no encrypted
        secret. ``derived`` holds the declared non-secret fields whose
        effective value the steps change, such as the page tree a pasted
        Confluence page URL names; ``facts`` states what the type can tell
        about the configuration; ``issues`` holds every refusal. With
        ``source_id``, the stored configuration completes the copy as it does
        on update, and the caller needs the access that update needs.
        """
        if name not in GENE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Gene '{name}' not found")
        existing_config: dict[str, Any] | None = None
        if req.source_id:
            existing = await db.get_source(req.source_id)
            if not existing:
                raise HTTPException(status_code=404, detail="Source not found")
            _require_source_management(request, existing)
            _require_local_agent_connection_management(request, existing)
            if existing["type"] != name:
                raise HTTPException(status_code=400, detail="Config-check source type does not match the existing source")
            existing_config = existing["config"]

        submitted = copy.deepcopy(req.config)
        config = copy.deepcopy(req.config)
        issues: tuple[FieldIssue, ...] = ()
        try:
            config = _complete_source_config(name, config, req.source_id or "config-check", existing_config)
            _validate_source_config(name, config, existing_config=existing_config)
        except SourceConfigInvalid as exc:
            issues = exc.issues

        descriptor = source_type_descriptor(name)
        derived: dict[str, Any] = {}
        for field in GENE_REGISTRY[name].config_schema().fields:
            if field.field_type == ConfigFieldType.SECRET:
                continue
            before = comparison_string(effective_value(field, submitted, descriptor), field.field_type)
            after = comparison_string(effective_value(field, config, descriptor), field.field_type)
            if before != after:
                derived[field.key] = config.get(field.key)

        facts: list[tuple[str, str]] = []
        if descriptor.config.facts is not None:
            facts = descriptor.config.facts(config)
        return ConfigCheckResponse(
            derived=derived,
            facts=[ConfigFactResponse(label=label, value=value) for label, value in facts],
            issues=[FieldIssueResponse(key=issue.key, message=issue.message) for issue in issues],
        )

    @gene_router.post("/{name}/preview-discovery", response_model=DiscoveryPreviewResponse)
    async def preview_gene_discovery(
        name: str,
        req: DiscoveryPreviewRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Preview the documents a source config would discover without saving it."""
        if name not in GENE_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Gene '{name}' not found")

        preview_config = dict(req.config)
        if req.source_id:
            existing = await db.get_source(req.source_id)
            if not existing:
                raise HTTPException(status_code=404, detail="Source not found")
            _require_source_management(request, existing)
            _require_local_agent_connection_management(request, existing)
            if existing["type"] != name:
                raise HTTPException(status_code=400, detail="Preview source type does not match the existing source")
            stored_config = decrypt_source_config_for_runtime(
                existing["config"],
                secret_fields=_source_secret_fields(name),
            )
            preview_config = {**stored_config, **preview_config}

        gene = None
        try:
            _validate_source_config(name, preview_config)
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
        except SourceConfigInvalid as exc:
            raise _config_invalid_response(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except browser_session.BrowserSessionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Discovery preview failed for gene %s", name)
            raise HTTPException(status_code=502, detail=f"Discovery preview failed: {exc}") from exc
        finally:
            if gene is not None:
                await gene.close()

        return DiscoveryPreviewResponse(
            source_type=name,
            count=count,
            truncated=count > len(items),
            items=items,
        )

    @gene_router.post("/github_repo/browse-tree", response_model=GitHubRepoTreeResponse)
    async def browse_github_repo_tree(
        req: GitHubRepoTreeRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """List selectable repository paths for the GitHub Repository source UI.

        The tree is read with the Source's own client and TLS. A tree GitHub
        truncates is walked folder by folder within
        ``BROWSE_TREE_REQUEST_BUDGET`` requests; a larger repository answers
        400, and its paths are typed instead.
        """
        gene = None
        try:
            config = dict(req.config)
            if req.source_id:
                existing = await db.get_source(req.source_id)
                if not existing:
                    raise HTTPException(status_code=404, detail="Source not found")
                _require_source_management(request, existing)
                _require_local_agent_connection_management(request, existing)
                if existing["type"] != "github_repo":
                    raise ValueError("Repository browse source type does not match the existing source")
                stored_config = decrypt_source_config_for_runtime(
                    existing["config"],
                    secret_fields=_source_secret_fields("github_repo"),
                )
                config = {**stored_config, **config}
            _validate_source_config("github_repo", config)
            if str(config.get("connection_mode") or "cloud_pull").strip().lower() != "cloud_pull":
                raise ValueError("GitHub Repository folder browsing is available only for Public internet access")
            gene = cast(GitHubRepoGene, create_gene("github_repo", config, source_id="browse-github-repo"))
            await gene.authenticate()
            github = gene.github_api()
            ref = str(config.get("ref") or "").strip()
            if not ref:
                ref = await github.default_branch()
            entries = await github.tree(ref, max_requests=BROWSE_TREE_REQUEST_BUDGET)
            items = _github_repo_tree_items(
                entries,
                limit=req.limit,
                include_extensions=github_include_extensions(config),
            )
        except HTTPException:
            raise
        except SourceConfigInvalid as exc:
            raise _config_invalid_response(exc) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except GitHubTreeBudgetExceeded as exc:
            raise HTTPException(
                status_code=400,
                detail=f"{exc} to browse. Type the folder or file paths to sync instead.",
            ) from exc
        except Exception as exc:
            logger.exception("GitHub Repository tree browse failed")
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            if gene is not None:
                await gene.close()

        return GitHubRepoTreeResponse(
            ref=ref,
            count=len(items),
            truncated=len(items) > req.limit,
            items=items[: req.limit],
        )

    @source_router.get("", response_model=SourceListResponse, response_model_exclude_unset=True)
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
            workspace_id=sync_service.workspace_id,
        )

        # Attach memory_count and sync status to each source
        from memforge.agent_sessions import AGENT_SESSION_SOURCE_TYPE

        jira_auth_service = JiraAuthSessionService(db)
        for s in sources:
            original_config = s.get("config", {})
            auth_mode = jira_auth_mode(original_config) if s["type"] == "jira" else None
            if s.get("capabilities", {}).get("can_configure"):
                s["config"] = redact_source_config(
                    original_config,
                    secret_fields=_source_secret_fields(s["type"]),
                    validate_encryption=True,
                )
                s["config"] = _public_source_config(s["type"], s["config"])
            else:
                s["config"] = {}
            if s.get("scope_transition") is not None:
                s["scope_transition"] = _viewed_scope_transition(s, s["scope_transition"])
            # Surface the originating client for agent-session sources so the UI
            # can pick a per-client brand mark without decoding opaque source ids.
            if s["type"] == AGENT_SESSION_SOURCE_TYPE:
                s["client"] = original_config.get("client")
            else:
                s["client"] = None
            if (
                s["type"] == "jira"
                and auth_mode == "browser_cookie"
                and s.get("capabilities", {}).get("can_configure_connection")
            ):
                try:
                    session = await jira_auth_service.get_status(str(s.get("config", {}).get("base_url") or ""))
                    s["connection_status"] = connection_status_from_browser_session(session)
                except ValueError:
                    s["connection_status"] = {
                        "state": "action_required",
                        "reason": "configuration",
                    }

        return {"data": sources}

    @source_router.get("/image-documents", response_model=ImageDocumentsResponse)
    async def count_image_documents(
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Count the current Documents with an image a model can read, in the Sources the caller can reprocess."""
        documents = await image_documents_by_source(
            db,
            viewer_id=resolve_request_principal(request),
            viewer_role=resolve_request_workspace_role(request),
        )
        return ImageDocumentsResponse(
            document_count=sum(len(ids) for ids in documents.values()),
            source_count=len(documents),
        )

    @source_router.post("/image-documents/reprocess", status_code=202, response_model=ImageDocumentsReprocessResponse)
    async def reprocess_image_documents(
        request: Request,
        db: Database = Depends(get_db),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """Queue a reprocess of the Documents with an image, one run per Source.

        It is what a model that newly reads images needs, since a sync reads only
        Documents that changed. A Source whose run cannot be queued now, such as
        one with a run already pending, is reported and left as it is. A queued
        Document that cannot be reprocessed from what was stored is recorded as
        a failed item of its run.
        """
        documents = await image_documents_by_source(
            db,
            viewer_id=resolve_request_principal(request),
            viewer_role=resolve_request_workspace_role(request),
        )
        queued_documents = 0
        queued_sources = 0
        unqueued: list[UnqueuedSource] = []
        for source_id, document_ids in documents.items():
            if await db.get_open_projection_scope_transition(source_id) is not None:
                unqueued.append(UnqueuedSource(source_id=source_id, reason="projection_scope_transition_open"))
                continue
            try:
                await sync_service.enqueue_reprocess(source_id, document_ids)
            except SourcePausedError:
                unqueued.append(UnqueuedSource(source_id=source_id, reason="source_paused"))
                continue
            except SourceNotActiveError:
                unqueued.append(UnqueuedSource(source_id=source_id, reason="source_not_active"))
                continue
            except SourceActivityConflict:
                unqueued.append(UnqueuedSource(source_id=source_id, reason="source_busy"))
                continue
            except SourceSyncBoundaryError:
                unqueued.append(UnqueuedSource(source_id=source_id, reason="source_not_synced"))
                continue
            except ValueError:
                unqueued.append(UnqueuedSource(source_id=source_id, reason="source_missing"))
                continue
            queued_documents += len(document_ids)
            queued_sources += 1
        return ImageDocumentsReprocessResponse(
            document_count=queued_documents, source_count=queued_sources, unqueued=unqueued,
        )

    async def _searchable_source_rows(
        request: Request,
        db: Database,
        *,
        sync_service: SyncService,
    ) -> list[dict[str, Any]]:
        rows = await list_source_admin_rows(
            db,
            sync_service=sync_service,
            viewer_id=resolve_request_principal(request),
            viewer_role=resolve_request_workspace_role(request),
            workspace_id=sync_service.workspace_id,
        )
        searchable: list[dict[str, Any]] = []
        for row in rows:
            source_id = str(row.get("id") or "")
            if not source_id or row.get("status") != "active" or not row.get("enabled_for_me", False):
                continue
            searchable.append(
                {
                    "source_id": source_id,
                    "name": row.get("name") or "",
                    "type": row.get("type") or "",
                    "status": row.get("status") or "",
                    "doc_count": int(row.get("doc_count") or 0),
                    "memory_count": int(row.get("memory_count") or 0),
                    "last_synced_at": row.get("last_sync"),
                }
            )
        return searchable

    @source_router.get("/searchable")
    async def list_searchable_sources(
        request: Request,
        db: Database = Depends(get_db),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """List search-eligible sources for MCP/source-id discovery."""
        return {"data": await _searchable_source_rows(request, db, sync_service=sync_service)}

    def _online_evaluation_truncated(events: Sequence[object], assessments: Sequence[object]) -> bool:
        return ONLINE_EVALUATION_ROW_LIMIT in (len(events), len(assessments))

    def _online_evaluation_window(days: int) -> tuple[datetime, datetime]:
        if not 1 <= days <= 90:
            raise HTTPException(status_code=400, detail="days must be between 1 and 90")
        occurred_to = datetime.now(timezone.utc)
        return occurred_to - timedelta(days=days), occurred_to

    async def _online_evaluation_rows(
        db: Database,
        *,
        principal: str,
        occurred_from: datetime,
        occurred_to: datetime,
        source_id: str | None = None,
        source_type: str | None = None,
    ):
        from memforge.evals.agent_evaluation import (
            AgentAssessmentQuery,
            AgentRuntimeEventQuery,
        )

        events = await db.list_agent_runtime_events(
            AgentRuntimeEventQuery(
                occurred_from=occurred_from,
                occurred_to=occurred_to,
                requesting_user_id=principal,
                include_private=True,
                source_id=source_id,
                source_type=source_type,
                newest_first=True,
                limit=ONLINE_EVALUATION_ROW_LIMIT,
            )
        )
        assessments = await db.list_agent_assessments(
            AgentAssessmentQuery(
                occurred_from=occurred_from,
                occurred_to=occurred_to,
                requesting_user_id=principal,
                include_private=True,
                source_id=source_id,
                newest_first=True,
                limit=ONLINE_EVALUATION_ROW_LIMIT,
            )
        )
        return events, assessments

    @evaluation_router.get("/online-overview", response_model=WorkspaceAgentEvaluationResponse)
    async def get_workspace_agent_evaluation(
        request: Request,
        days: int = 1,
        source_id: str | None = None,
        source_type: str | None = None,
        db: Database = Depends(get_db),
    ):
        """Return one authorized workspace evaluation overview."""

        from memforge.evals.agent_evaluation import (
            build_workspace_online_evaluation_view,
        )

        occurred_from, occurred_to = _online_evaluation_window(days)
        principal = resolve_request_principal(request)
        discoverable_sources = [
            source
            for source in await db.list_sources()
            if source_is_discoverable(source, viewer_id=principal)
        ]
        available_source_types = sorted(
            {
                str(source.get("type") or "unknown")
                for source in discoverable_sources
            }
        )
        if source_id is not None:
            selected = [
                source
                for source in discoverable_sources
                if str(source.get("id") or "") == source_id
            ]
            if not selected:
                raise HTTPException(status_code=404, detail="Source not found")
            discoverable_sources = selected
        if source_type is not None:
            discoverable_sources = [
                source
                for source in discoverable_sources
                if str(source.get("type") or "") == source_type
            ]
        events, assessments = await _online_evaluation_rows(
            db,
            principal=principal,
            occurred_from=occurred_from,
            occurred_to=occurred_to,
            source_id=source_id,
            source_type=source_type,
        )
        view = build_workspace_online_evaluation_view(
            discoverable_sources,
            events,
            assessments,
        )
        summary = dict(view["summary"])
        summary["truncated"] = _online_evaluation_truncated(events, assessments)
        summary["row_limit"] = ONLINE_EVALUATION_ROW_LIMIT
        return {
            "scope": {
                "kind": "workspace",
                "source_id": source_id,
                "source_type": source_type,
            },
            "window": {
                "from": occurred_from.isoformat(),
                "to": occurred_to.isoformat(),
                "days": days,
            },
            "summary": summary,
            "coverage": view["coverage"],
            "issue_groups": view["issue_groups"],
            "available_source_types": available_source_types,
            "sources": view["sources"],
            "runtime_events": view["runtime_events"],
            "assessments": view["assessments"],
        }

    # Deprecated (admin-ui-v1): only the V1 per-source evaluation dialog calls
    # this route, and V1 never renders that dialog. The admin UI reads
    # GET /agent-evaluations/online-overview, scoped by source_id; this route
    # is removed with the V1 admin UI.
    @source_router.get("/{source_id}/agent-evaluation", deprecated=True)
    async def get_source_agent_evaluation(
        source_id: str,
        request: Request,
        days: int = 30,
        db: Database = Depends(get_db),
    ):
        """Return a bounded, authorized online-assessment view for one Source."""

        from memforge.evals.agent_evaluation import build_source_online_evaluation_view

        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_discoverability(request, source)
        occurred_from, occurred_to = _online_evaluation_window(days)
        principal = resolve_request_principal(request)
        events, assessments = await _online_evaluation_rows(
            db,
            principal=principal,
            occurred_from=occurred_from,
            occurred_to=occurred_to,
            source_id=source_id,
        )
        view = build_source_online_evaluation_view(events, assessments)
        summary = dict(view["summary"])
        summary["truncated"] = _online_evaluation_truncated(events, assessments)
        summary["row_limit"] = ONLINE_EVALUATION_ROW_LIMIT
        return {
            "source_id": source_id,
            "window": {
                "from": occurred_from.isoformat(),
                "to": occurred_to.isoformat(),
                "days": days,
            },
            "summary": summary,
            "coverage": view["coverage"],
            "issue_groups": view["issue_groups"],
            "runtime_events": view["runtime_events"],
            "assessments": view["assessments"],
        }

    async def _require_evaluation_result_management(
        result_id: str,
        request: Request,
        db: Database,
    ) -> None:
        result = await db.get_agent_evaluation_result(result_id)
        if result is None:
            raise HTTPException(status_code=404, detail="Evaluation result not found")
        case = await db.get_agent_evaluation_case(result.case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="Evaluation case not found")
        source = await db.get_source(case.source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_management(request, source)

    def _langfuse_annotation_exchange(
        request: Request,
        db: Database,
    ):
        from memforge.evals.external_annotation import (
            ExternalAnnotationExchange,
            LangfuseAnnotationAdapter,
        )
        from memforge.evals.offline_evaluation import OfflineAgentEvaluation

        adapter = getattr(request.app.state, "langfuse_annotation_adapter", None)
        if adapter is None:
            adapter = LangfuseAnnotationAdapter.from_env()
            request.app.state.langfuse_annotation_adapter = adapter
        return ExternalAnnotationExchange(
            db,
            OfflineAgentEvaluation(db, executors={}),
            adapter,
        )

    @evaluation_router.post("/annotation-policies/langfuse")
    async def approve_langfuse_annotation_policy(
        body: AgentEvaluationLangfusePolicyRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Approve one exact Source-to-queue protected-content disclosure."""

        from memforge.evals.offline_evaluation import (
            OfflineAgentEvaluation,
            agent_evaluation_content_policy_to_payload,
        )

        source = await db.get_source(body.source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_management(request, source)
        try:
            policy = await OfflineAgentEvaluation(db, executors={}).approve_langfuse_human_calibration_content(
                source_id=body.source_id,
                policy_version=body.policy_version,
                queue_id=body.queue_id,
                approved_by=resolve_request_principal(request),
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return agent_evaluation_content_policy_to_payload(policy)

    @evaluation_router.post("/annotations/langfuse/export")
    async def export_langfuse_annotation(
        body: AgentEvaluationLangfuseExportRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Export one blinded task while retaining content-free durable state."""

        from memforge.evals.offline_evaluation import external_annotation_task_to_payload

        await _require_evaluation_result_management(body.result_id, request, db)
        principal = resolve_request_principal(request)
        try:
            task = await _langfuse_annotation_exchange(request, db).export(
                result_id=body.result_id,
                content_policy_id=body.content_policy_id,
                criterion=body.criterion,
                rubric_version=body.rubric_version,
                actor_user_id=principal,
                provider_reviewer_id=body.provider_reviewer_id,
                queue_id=body.queue_id,
                score_config_id=body.score_config_id,
                lease_owner=f"api:{uuid.uuid4().hex}",
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            status = 409 if "leased" in str(exc) or "ambiguous" in str(exc) else 503
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Langfuse annotation export failed")
            raise HTTPException(status_code=502, detail="Langfuse annotation export failed") from exc
        return external_annotation_task_to_payload(task)

    @evaluation_router.get("/annotations/langfuse/{task_id}")
    async def get_langfuse_annotation_task(
        task_id: str,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Return one authorized content-free annotation bridge record."""

        from memforge.evals.offline_evaluation import external_annotation_task_to_payload

        task = await db.get_external_annotation_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Annotation task not found")
        await _require_evaluation_result_management(task.result_id, request, db)
        return external_annotation_task_to_payload(task)

    @evaluation_router.post("/annotations/langfuse/{task_id}/import")
    async def import_langfuse_annotation(
        task_id: str,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Import one completed reviewer Score as an immutable assessment."""

        from memforge.evals.offline_evaluation import external_annotation_task_to_payload

        task = await db.get_external_annotation_task(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Annotation task not found")
        await _require_evaluation_result_management(task.result_id, request, db)
        try:
            imported = await _langfuse_annotation_exchange(request, db).import_completed(
                task_id=task_id,
                submitted_by=resolve_request_principal(request),
                lease_owner=f"api:{uuid.uuid4().hex}",
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            status = 409 if "leased" in str(exc) else 503
            raise HTTPException(status_code=status, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Langfuse annotation import failed")
            raise HTTPException(status_code=502, detail="Langfuse annotation import failed") from exc
        return external_annotation_task_to_payload(imported)

    @evaluation_router.post("/relation-cases/seed")
    async def seed_relation_evaluation_cases(
        body: CrossSourceReviewLabelsRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Pin decided Cross-Source Conflict Reviews as labeled cases and freeze their cohort."""

        from memforge.evals.cross_document_relation_cases import seed_cross_document_relation_cases
        from memforge.evals.offline_evaluation import OfflineAgentEvaluation

        actor = _require_maintenance_operator(request)
        try:
            report = await seed_cross_document_relation_cases(
                db,
                OfflineAgentEvaluation(db, executors={}),
                actor=actor,
                label_overrides=body.label_overrides,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "cohort_id": report.cohort_id,
            "pinned_case_count": report.pinned_case_count,
            "label_counts": dict(report.label_counts),
            "skipped": dict(report.skipped),
        }

    @evaluation_router.post("/relation-group-cases/seed")
    async def seed_relation_group_evaluation_cases(
        body: RelationGroupCasesSeedRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Pin labelled discovery groups as relation group cases and freeze their cohort.

        The server reads every Memory and its classifier input itself; the
        request carries Memory ids, labels and the content hashes the labels
        were made for only.
        """

        from memforge.evals.cross_document_relation_cases import (
            RelationGroupLabels,
            seed_cross_document_relation_group_cases,
        )
        from memforge.evals.offline_evaluation import OfflineAgentEvaluation

        actor = _require_maintenance_operator(request)
        try:
            report = await seed_cross_document_relation_group_cases(
                db,
                OfflineAgentEvaluation(db, executors={}),
                groups=[
                    RelationGroupLabels(
                        challenger_memory_id=group.challenger_memory_id,
                        candidate_memory_ids=tuple(group.candidate_memory_ids),
                        labels=group.labels,
                        content_hashes=group.content_hashes,
                    )
                    for group in body.groups
                ],
                labelled_by=body.labelled_by,
                actor=actor,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "cohort_id": report.cohort_id,
            "pinned_group_count": report.pinned_group_count,
            "candidate_count": report.candidate_count,
            "labelled_pair_count": report.labelled_pair_count,
            "label_counts": dict(report.label_counts),
            "skipped": dict(report.skipped),
            "skipped_groups": dict(report.skipped_groups),
        }

    @evaluation_router.post("/derivation-cases/seed")
    async def seed_derivation_evaluation_cases(
        body: DerivationCasesSeedRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Pin a derivation of each named Document as a case and freeze their cohort."""

        from memforge.evals.offline_evaluation import (
            AgentEvaluationPopulation,
            AgentEvaluationRole,
            OfflineAgentEvaluation,
        )
        from memforge.evals.source_unit_derivation_cases import (
            DerivationCaseRequest as PinRequest,
            seed_source_unit_derivation_cases,
        )

        actor = _require_maintenance_operator(request)
        try:
            report = await seed_source_unit_derivation_cases(
                db,
                OfflineAgentEvaluation(db, executors={}),
                requests=[
                    PinRequest(
                        doc_id=case.doc_id,
                        source_id=case.source_id,
                        title=case.title,
                        derivation_id=case.derivation_id,
                        rubric=case.rubric,
                        population=AgentEvaluationPopulation(case.population),
                    )
                    for case in body.cases
                ],
                role=AgentEvaluationRole(body.role),
                selection_policy_version=body.selection_policy_version,
                actor=actor,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "cohort_id": report.cohort_id,
            "cases": [
                {
                    "doc_id": case.doc_id,
                    "case_id": case.case_id,
                    "ground_truth_revision_id": case.ground_truth_revision_id,
                    "population": case.population.value,
                }
                for case in report.cases
            ],
            "skipped": {reason: list(doc_ids) for reason, doc_ids in report.skipped.items()},
        }

    @evaluation_router.post("/support-cases/seed")
    async def seed_support_evaluation_cases(
        body: SupportCasesSeedRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Pin an applied update of each named Document as a Support case and freeze their cohort."""

        from memforge.evals.offline_evaluation import (
            AgentEvaluationPopulation,
            AgentEvaluationRole,
            OfflineAgentEvaluation,
        )
        from memforge.evals.source_unit_support_cases import (
            SupportCaseRequest as PinRequest,
            seed_source_unit_support_cases,
        )

        actor = _require_maintenance_operator(request)
        try:
            report = await seed_source_unit_support_cases(
                db,
                OfflineAgentEvaluation(db, executors={}),
                requests=[
                    PinRequest(
                        doc_id=case.doc_id,
                        source_id=case.source_id,
                        title=case.title,
                        derivation_id=case.derivation_id,
                        rubric=case.rubric,
                        population=AgentEvaluationPopulation(case.population),
                    )
                    for case in body.cases
                ],
                role=AgentEvaluationRole(body.role),
                selection_policy_version=body.selection_policy_version,
                actor=actor,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "cohort_id": report.cohort_id,
            "cases": [
                {
                    "doc_id": case.doc_id,
                    "derivation_id": case.derivation_id,
                    "case_id": case.case_id,
                    "ground_truth_revision_id": case.ground_truth_revision_id,
                    "population": case.population.value,
                }
                for case in report.cases
            ],
            "skipped": {reason: list(labels) for reason, labels in report.skipped.items()},
        }

    @evaluation_router.post("/reconciliation-cases/seed")
    async def seed_reconciliation_evaluation_cases(
        body: ReconciliationCasesSeedRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Import each hand-written situation as a reconciliation case and freeze their cohort."""

        from memforge.evals.offline_evaluation import (
            AgentEvaluationPopulation,
            AgentEvaluationRole,
            OfflineAgentEvaluation,
        )
        from memforge.evals.reconciliation_cases import (
            ReconciliationCaseRequest as ImportRequest,
            seed_reconciliation_cases,
        )

        actor = _require_maintenance_operator(request)
        try:
            report = await seed_reconciliation_cases(
                db,
                OfflineAgentEvaluation(db, executors={}),
                requests=[
                    ImportRequest(
                        doc_id=case.doc_id,
                        source_id=case.source_id,
                        title=case.title,
                        input=case.input,
                        rubric=case.rubric,
                        population=AgentEvaluationPopulation(case.population),
                    )
                    for case in body.cases
                ],
                role=AgentEvaluationRole(body.role),
                selection_policy_version=body.selection_policy_version,
                actor=actor,
            )
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "cohort_id": report.cohort_id,
            "cases": [
                {
                    "doc_id": case.doc_id,
                    "case_id": case.case_id,
                    "ground_truth_revision_id": case.ground_truth_revision_id,
                    "population": case.population.value,
                }
                for case in report.cases
            ],
            "skipped": {reason: list(labels) for reason, labels in report.skipped.items()},
        }

    @evaluation_router.post("/runs", status_code=202)
    async def admit_agent_evaluation_run(
        body: AgentEvaluationRunCreateRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
    ):
        """Admit one idempotent run; execution remains worker-owned.

        A candidate manifest without ``code_revision`` is pinned to the release
        this server runs, the code its worker replays the cohort with, so a run
        admitted after a deployment is a new run rather than the earlier one.
        One without ``model`` is pinned to the route of the generation model in
        use.
        """

        from memforge.evals.offline_evaluation import (
            OfflineAgentEvaluation,
            SemanticJudgeSpec,
            agent_evaluation_run_execution_to_payload,
            agent_evaluation_run_to_payload,
        )
        from memforge.release import current_release
        from memforge.runtime import resolve_models

        pins: dict[str, object] = {"code_revision": current_release()}
        if "model" not in body.candidate_manifest:
            generation = (await resolve_models(db, config)).generation
            if generation is not None:
                pins["model"] = generation.route
        try:
            semantic_judge = (
                SemanticJudgeSpec.from_manifest(body.semantic_judge)
                if body.semantic_judge is not None
                else None
            )
            run, execution = await OfflineAgentEvaluation(db, executors={}).admit_run(
                cohort_id=body.cohort_id,
                candidate_manifest={**pins, **body.candidate_manifest},
                evaluator_suite=body.evaluator_suite,
                evaluator_version=body.evaluator_version,
                created_by=resolve_request_principal(request),
                semantic_judge_spec=semantic_judge,
                replicate_count=body.replicate_count,
                baseline_run_id=body.baseline_run_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "run": agent_evaluation_run_to_payload(run),
            "execution": (
                agent_evaluation_run_execution_to_payload(execution)
                if execution is not None
                else None
            ),
        }

    @evaluation_router.get("/runs/{run_id}")
    async def get_agent_evaluation_run(
        run_id: str,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Return one authorized, content-free run status and report."""

        from memforge.evals.offline_evaluation import (
            OfflineAgentEvaluation,
            agent_evaluation_report_public_payload,
        )

        evaluation = OfflineAgentEvaluation(db, executors={})
        try:
            report = await evaluation.read_report(
                run_id,
                requesting_user_id=resolve_request_principal(request),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        execution = await db.get_agent_evaluation_run_execution(run_id)
        return agent_evaluation_report_public_payload(report, execution)

    @evaluation_router.get("/runs/{run_id}/case-outputs")
    async def get_agent_evaluation_case_outputs(
        run_id: str,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Return each case's pinned input, accepted rubric and candidate output for failure analysis."""

        from memforge.evals.offline_evaluation import (
            OfflineAgentEvaluation,
            agent_evaluation_case_outputs_payload,
        )

        actor = _require_maintenance_operator(request)
        try:
            outputs = await OfflineAgentEvaluation(db, executors={}).read_case_outputs(
                run_id,
                requesting_user_id=actor,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return agent_evaluation_case_outputs_payload(run_id, outputs)

    @source_router.get("/{source_id}/memory-lifecycle")
    async def get_source_memory_lifecycle(
        source_id: str,
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Return independent scope transition, review, and delivery axes."""

        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_management(request, source)
        reviews = await db.list_lifecycle_reviews(source_id)
        vector_tasks = await db.list_lifecycle_vector_tasks(source_id=source_id, limit=200)
        scope_transitions = await db.list_projection_scope_transitions(source_id)
        return {
            "source_id": source_id,
            "scope_transitions": [
                {
                    "id": transition.id,
                    "previous_scope": dict(transition.previous_scope),
                    "target_scope": dict(transition.target_scope),
                    "status": transition.status.value,
                    "run_id": transition.run_id,
                    "error": transition.error,
                    "created_at": transition.created_at,
                    "updated_at": transition.updated_at,
                    "completed_at": transition.completed_at,
                }
                for transition in scope_transitions
            ],
            "reviews": [
                {
                    "id": review.id,
                    "lifecycle_plan_id": review.lifecycle_plan_id,
                    "incumbent_memory_id": review.incumbent_memory_id,
                    "status": review.status.value,
                    "staged_evidence": dict(review.staged_evidence),
                    "reason": review.reason,
                    "review_note": review.review_note,
                    "reviewer": review.reviewer,
                    "presentation": present_lifecycle_review(
                        staged_evidence=review.staged_evidence,
                        reason=review.reason,
                    ).to_dict(),
                    "created_at": review.created_at,
                    "resolved_at": review.resolved_at,
                }
                for review in reviews
            ],
            "vector_outbox": [
                {
                    "id": task.id,
                    "lifecycle_plan_id": task.lifecycle_plan_id,
                    "memory_id": task.memory_id,
                    "operation": task.operation.value,
                    "status": task.status.value,
                    "attempts": task.attempts,
                    "error": task.error,
                }
                for task in vector_tasks
            ],
        }

    async def _decide_lifecycle_review(
        source_id: str,
        review_id: str,
        request: Request,
        db: Database,
        *,
        action: Literal["use_latest_state", "keep_current_state"],
        expected_fingerprint: str,
        note: str | None,
        config: AppConfig | None = None,
        runtime_provider: RuntimeProvider | None = None,
    ) -> dict[str, object]:
        from memforge.memory.lifecycle_review import build_lifecycle_review_approval_plan

        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        review = await db.get_lifecycle_review(review_id)
        if review is None:
            raise HTTPException(status_code=404, detail="Lifecycle review not found")
        if review.source_id != source_id:
            raise HTTPException(status_code=404, detail="Lifecycle review not found")
        await _require_lifecycle_review_visibility(request, db, review)
        _require_review_decision_authority(request, [source])
        if lifecycle_review_decision_fingerprint(review) != expected_fingerprint:
            raise HTTPException(status_code=409, detail="Review decision fingerprint is stale")
        target_status = (
            LifecycleReviewStatus.APPROVED if action == "use_latest_state" else LifecycleReviewStatus.REJECTED
        )
        if review.status is target_status:
            return {
                "source_id": source_id,
                "review_id": review_id,
                "status": review.status.value,
                **(
                    {"lifecycle_plan_id": f"lifecycle-review-approval-{review.id}"}
                    if target_status is LifecycleReviewStatus.APPROVED
                    else {}
                ),
            }
        if review.status is not LifecycleReviewStatus.PENDING:
            raise HTTPException(
                status_code=409,
                detail=f"Lifecycle review is already {review.status.value}",
            )
        payload = await db.get_lifecycle_plan_payload(review.lifecycle_plan_id)
        if payload is None or not isinstance(payload.get("scope"), dict):
            raise HTTPException(status_code=409, detail="Lifecycle review plan is unavailable")
        if payload["scope"].get("source_id") != source_id:
            raise HTTPException(status_code=404, detail="Lifecycle review not found")
        reviewer = resolve_request_principal(request)
        if action == "keep_current_state":
            if not note or not note.strip():
                raise HTTPException(status_code=400, detail="A note is required to keep the current state")
            try:
                rejected = await db.resolve_lifecycle_review(
                    review_id, LifecycleReviewStatus.REJECTED, reviewer=reviewer, review_note=note.strip(),
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return {"source_id": source_id, "review_id": review_id, "status": rejected.status.value}
        try:
            plan = build_lifecycle_review_approval_plan(
                review, payload, reviewer=reviewer, review_note=note.strip() if note else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return await _apply_lifecycle_review_plan(
            source_id, review, plan, db, config=config, runtime_provider=runtime_provider,
        )

    async def _apply_lifecycle_review_plan(
        source_id: str,
        review: LifecycleReview,
        plan: LifecyclePlan,
        db: Database,
        *,
        config: AppConfig | None,
        runtime_provider: RuntimeProvider | None,
    ) -> dict[str, object]:
        """Apply one approval's Plan and deliver its vector changes.

        An approval whose guard no longer holds marks the Review stale.
        """
        if config is None or runtime_provider is None:
            raise RuntimeError("Lifecycle approval requires the configured runtime")
        try:
            await db.apply_lifecycle_plan(plan)
        except ValueError as exc:
            detail = str(exc)
            if "stale guard" in detail or "already" in detail:
                try:
                    await db.resolve_lifecycle_review(review.id, LifecycleReviewStatus.STALE)
                except ValueError:
                    pass
            raise HTTPException(status_code=409, detail=detail) from exc
        memory_store = await _build_memory_store(db, config, runtime_provider)
        delivery = await memory_store.attempt_lifecycle_vector_delivery(plan.id)
        if delivery.pending:
            logger.warning(
                "Lifecycle review %s applied but vector delivery remains pending error_types=%s",
                review.id,
                delivery.error_types,
            )
        decided = await db.get_lifecycle_review(review.id)
        assert decided is not None
        return {
            "source_id": source_id,
            "review_id": review.id,
            "status": decided.status.value,
            "lifecycle_plan_id": plan.id,
        }

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
        _require_source_discoverability(request, source)

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
        _require_source_discoverability(request, source)

        rows = await db.list_resolved_projects_for_source(
            source_id,
            include_private=True,
            owner_user_id=resolve_request_principal(request),
        )
        projects = [ResolvedProjectResponse(project_key=key, memory_count=count) for key, count in rows]
        return ResolvedProjectsResponse(source_id=source_id, projects=projects)

    @source_router.put("/{source_id}/subscription", response_model=SourceSubscriptionChangeResponse)
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
        _require_source_discoverability(request, source)
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

    @source_router.put("/{source_id}/pin", response_model=SourcePinResponse)
    async def pin_source_for_viewer(
        request: Request,
        source_id: str,
        db: Database = Depends(get_db),
    ):
        """Pin one source in the authenticated viewer's Source List."""
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_discoverability(request, source)
        await db.set_source_pinned_for_user(
            source_id,
            resolve_request_principal(request),
            True,
        )
        return {"source_id": source_id, "pinned": True}

    @source_router.delete("/{source_id}/pin", response_model=SourcePinResponse)
    async def unpin_source_for_viewer(
        request: Request,
        source_id: str,
        db: Database = Depends(get_db),
    ):
        """Unpin one source in the authenticated viewer's Source List."""
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_discoverability(request, source)
        await db.set_source_pinned_for_user(
            source_id,
            resolve_request_principal(request),
            False,
        )
        return {"source_id": source_id, "pinned": False}

    async def _source_access_transition_service(
        db: Database,
        config: AppConfig,
        runtime_provider: RuntimeProvider,
        sync_service: SyncService,
    ) -> SourceAccessTransitionService:
        return SourceAccessTransitionService(
            db=db,
            memory_store=await _build_memory_store(db, config, runtime_provider),
            sync_service=sync_service,
        )

    @source_router.post(
        "/{source_id}/access-transitions",
        status_code=202,
        response_model=SourceAccessTransitionResponse,
    )
    async def start_source_access_transition(
        request: Request,
        source_id: str,
        req: SourceAccessTransitionRequest,
        background_tasks: BackgroundTasks,
        idempotency_key: str = Header(alias="Idempotency-Key", min_length=1, max_length=200),
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """Start a durable whole-history Source access change."""
        source = await db.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_management(request, source)
        service = await _source_access_transition_service(
            db,
            config,
            runtime_provider,
            sync_service,
        )
        try:
            transition = await service.start(
                source_id=source_id,
                actor_user_id=resolve_request_principal(request),
                target_policy=req.target_policy,
                idempotency_key=idempotency_key,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        background_tasks.add_task(
            _run_source_access_background,
            transition["operation_id"],
            service.run,
        )
        return transition

    @source_router.get("/{source_id}/access-transition", response_model=SourceCurrentAccessTransitionResponse)
    async def get_source_access_transition(
        request: Request,
        source_id: str,
        db: Database = Depends(get_db),
    ):
        """Return the Source's current access operation, if any."""
        source = await db.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_discoverability(request, source)
        return {"transition": await db.get_active_source_access_transition(source_id)}

    @source_router.get(
        "/{source_id}/access-transitions/{operation_id}",
        response_model=SourceAccessTransitionOperationResponse,
    )
    async def get_source_access_transition_operation(
        request: Request,
        source_id: str,
        operation_id: str,
        db: Database = Depends(get_db),
    ):
        """Return one durable access operation, including terminal history."""
        source = await db.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_discoverability(request, source)
        transition = await db.get_source_access_transition(operation_id)
        if transition is None or transition["source_id"] != source_id:
            raise HTTPException(status_code=404, detail="Access transition not found")
        return {"transition": transition}

    @source_router.post(
        "/{source_id}/access-transitions/{operation_id}/retry",
        response_model=SourceAccessTransitionQueuedResponse,
    )
    async def retry_source_access_transition(
        request: Request,
        source_id: str,
        operation_id: str,
        background_tasks: BackgroundTasks,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """Retry a failed access operation without widening visibility."""
        source = await db.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_discoverability(request, source)
        if source.get("owner_user_id") != resolve_request_principal(request):
            raise _source_management_forbidden()
        transition = await db.get_source_access_transition(operation_id)
        if transition is None or transition["source_id"] != source_id:
            raise HTTPException(status_code=404, detail="Access transition not found")
        service = await _source_access_transition_service(
            db,
            config,
            runtime_provider,
            sync_service,
        )
        try:
            if transition["status"] != "failed":
                raise SourceAccessTransitionError("source_access_transition_not_retryable")
        except SourceAccessTransitionError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        background_tasks.add_task(
            _run_source_access_background,
            operation_id,
            service.retry,
        )
        return {"operation_id": operation_id, "status": "queued"}

    @source_router.post(
        "/{source_id}/access-transitions/{operation_id}/revert",
        response_model=SourceAccessTransitionQueuedResponse,
    )
    async def revert_source_access_transition(
        request: Request,
        source_id: str,
        operation_id: str,
        background_tasks: BackgroundTasks,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """Restore the complete previous policy for a failed operation."""
        source = await db.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_discoverability(request, source)
        if source.get("owner_user_id") != resolve_request_principal(request):
            raise _source_management_forbidden()
        transition = await db.get_source_access_transition(operation_id)
        if transition is None or transition["source_id"] != source_id:
            raise HTTPException(status_code=404, detail="Access transition not found")
        service = await _source_access_transition_service(
            db,
            config,
            runtime_provider,
            sync_service,
        )
        if transition["status"] != "failed":
            raise HTTPException(
                status_code=409,
                detail="source_access_transition_not_revertible",
            )
        background_tasks.add_task(
            _run_source_access_background,
            operation_id,
            service.revert,
        )
        return {"operation_id": operation_id, "status": "queued"}

    @source_list_router.get("/preferences", response_model=SourceListPreferencesResponse)
    async def get_source_list_preferences(
        request: Request,
        db: Database = Depends(get_db),
    ):
        """Return the authenticated viewer's Source List preferences."""
        sort_mode = await db.get_source_list_sort_mode(resolve_request_principal(request))
        return {"sort_mode": sort_mode}

    @source_list_router.put("/preferences", response_model=SourceListPreferencesResponse)
    async def set_source_list_preferences(
        request: Request,
        req: SourceListPreferenceRequest,
        db: Database = Depends(get_db),
    ):
        """Persist the authenticated viewer's Source List sort mode."""
        await db.set_source_list_sort_mode(
            resolve_request_principal(request),
            req.sort_mode,
        )
        return {"sort_mode": req.sort_mode}

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
                        "WHERE m.updated_at >= ? AND ms.source_id = ?"
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

    @source_router.post("", response_model=SourceCreatedResponse)
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
            incoming_config = _complete_source_config(req.type, req.config, source_id)
            _validate_source_config(req.type, incoming_config)
            source_config = prepare_source_config_for_storage(
                incoming_config,
                secret_fields=_source_secret_fields(req.type),
            )
            _validate_source_project_binding(req.project_binding)
            source_config = _store_source_config(req.type, source_config, source_id, config)
        except SourceConfigInvalid as exc:
            raise _config_invalid_response(exc) from exc
        except (SecretConfigurationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        creator_user_id = resolve_request_principal(request)
        source_for_classification = {"type": req.type, "config": source_config}
        await db.upsert_source(
            id=source_id,
            type=req.type,
            name=req.name,
            config_json=json.dumps(source_config),
            access_policy=req.access_policy,
            owner_user_id=creator_user_id,
            project_binding=req.project_binding,
            created_by_user_id=creator_user_id,
            execution_owner_user_id=(
                creator_user_id if is_local_agent_backed_source(source_for_classification) else None
            ),
        )
        if req.sync_schedule is not None:
            await db.set_source_sync_schedule(
                source_id,
                enabled=req.sync_schedule.enabled,
                interval_minutes=req.sync_schedule.interval_minutes,
            )
        return {
            "id": source_id,
            "name": req.name,
            "type": req.type,
            "access_policy": req.access_policy,
        }

    @source_router.post("/{source_id}/scope-change-preview", response_model=SourceScopeChangePreviewResponse)
    async def preview_source_scope_change(
        request: Request,
        source_id: str,
        req: SourceScopeChangePreviewRequest,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
    ):
        """What saving this configuration would do to the Source's scope, without saving it.

        The configuration form reads it before saving, to warn about a mode
        that cannot list a scope change, about replacing the change in
        progress, and about when removal happens. It decides as
        ``PUT /sources/{id}`` does, from the same stored facts; a
        configuration the source type rejects answers 400.
        """

        existing = await db.get_source(source_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_management(request, existing)
        _require_local_agent_connection_management(request, existing)
        open_transition = await db.get_open_projection_scope_transition(source_id) is not None
        removes = scope_change_removes(str(existing["type"]))
        try:
            change = await _plan_config_change(db, existing, req.config, config, preview=True)
        except HTTPException as exc:
            if exc.status_code != 409:
                raise
            return {
                "outcome": "refused",
                "open_transition": open_transition,
                "removes": removes,
                "refusal": _scope_change_refusal(exc.detail),
            }
        return {"outcome": _scope_change_outcome(change), "open_transition": open_transition, "removes": removes}

    @source_router.put("/{source_id}", response_model=SourceUpdatedResponse)
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
        if req.config is not None:
            _require_local_agent_connection_management(request, existing)

        _validate_source_status(req.status)
        name = req.name or existing["name"]
        if req.config is not None:
            change = await _plan_config_change(db, existing, req.config, config)
        else:
            change = _ConfigChange(
                config=existing["config"],
                scope_changed=False,
                scope_change=_ScopeChange(transition=None),
            )
        src_config = change.config
        scope_changed = change.scope_changed
        scope_change = change.scope_change
        if req.expected_scope_transition_id is not None:
            # A scope change supersedes the transition it read, compared and
            # set when it is written; any other save compares what it reads.
            if scope_changed:
                seen = scope_change.superseded_transition_id
            else:
                open_transition = await db.get_open_projection_scope_transition(source_id)
                seen = open_transition.id if open_transition is not None else None
            if seen != req.expected_scope_transition_id:
                raise HTTPException(status_code=409, detail="projection_scope_transition_stale")
        auth_secret_changed = req.config is not None and _source_credentials_changed(
            existing["type"],
            req.config,
            existing["config"],
        )
        if scope_changed or auth_secret_changed:
            await sync_service.cancel_source(source_id)

        if _request_includes_field(req, "project_binding"):
            try:
                _validate_source_project_binding(req.project_binding)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            await db.upsert_source(
                id=source_id,
                type=existing["type"],
                name=name,
                config_json=json.dumps(src_config),
                access_policy=existing["access_policy"],
                owner_user_id=existing["owner_user_id"],
                access_state=existing["access_state"],
                status=req.status if _request_includes_field(req, "status") else None,
                project_binding=(
                    req.project_binding
                    if _request_includes_field(req, "project_binding")
                    else existing.get("project_binding")
                ),
                projection_scope_transition=scope_change.transition,
                expected_open_transition_id=scope_change.superseded_transition_id,
            )
        except SourceSyncRunActive as exc:
            raise _source_sync_running_http_error(source_id, exc.run_id) from exc
        except ProjectionScopeTransitionStale as exc:
            # The open transition changed after this change read it: read
            # the Source again and decide with fresh facts.
            raise HTTPException(status_code=409, detail="projection_scope_transition_stale") from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        if _request_includes_field(req, "status") and req.status == SOURCE_PAUSED_STATUS:
            # A pause also stops the sync in progress; the Source syncs again once resumed.
            await _stop_source_sync(request, db, sync_service, existing)
        if _request_includes_field(req, "sync_schedule"):
            schedule = req.sync_schedule or SourceSyncScheduleRequest()
            await db.set_source_sync_schedule(
                source_id,
                enabled=schedule.enabled,
                interval_minutes=schedule.interval_minutes,
            )
        reverted_to_last_applied_scope = scope_change.superseded_transition_id is not None and (
            scope_change.transition is None
        )
        if (
            auth_secret_changed
            or reverted_to_last_applied_scope
            or (scope_changed and not _is_document_source_type(existing["type"]))
        ):
            # The next run reads everything. A new credential may read what
            # the old one could not. A return to the last applied scope opens
            # no transition, while the superseded transition's runs may have
            # moved the cursor and settled failed Documents of that scope. A
            # document Source otherwise keeps its cursor through a scope
            # change: its transition runs discover the whole new scope, and
            # the run that completes it moves the cursor.
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
        """Functionally remove a Source and retire Memories left unsupported."""
        existing = await db.get_source(source_id)
        if not existing or existing.get("status") == "retired":
            return {
                "ok": True,
                "deleted_source": source_id,
                "already_deleted": True,
            }
        _require_source_management(request, existing)
        await sync_service.cancel_source(source_id)
        release_atlassian_request_limiter(
            str(existing["config"].get("base_url") or ""),
            owner_id=source_id,
        )

        activity_id = f"source-delete-{uuid.uuid4().hex}"
        try:
            source_activity = await db.acquire_source_activity(
                activity_id=activity_id,
                source_id=source_id,
                kind=SourceActivityKind.MAINTENANCE,
                capability=activity_id,
                lease_seconds=900,
            )
            memory_store = await _build_memory_store(db, config, runtime_provider)
            await memory_store.delete_source_cascade(
                source_id,
                source_activity=source_activity,
            )
        except (SourceActivityConflict, ValueError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        finally:
            await db.release_source_activity(
                activity_id=activity_id,
                capability=activity_id,
            )
        await SourceArtifactCleanupService(db, document_store).run_pending(limit=1000)
        return {"ok": True, "deleted_source": source_id}

    async def _enqueue_sync_local_agent_job(
        request: Request,
        db: Database,
        **job: Any,
    ) -> tuple[str, bool]:
        enqueuer = getattr(request.app.state, "local_agent_job_enqueuer", None)
        try:
            if enqueuer is not None:
                return await enqueuer(**job)
            return await db.enqueue_local_agent_job(**job)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    async def _enqueue_local_source_collection_job(
        request: Request,
        db: Database,
        source: dict[str, Any],
        *,
        force_full_sync: bool,
        dry_run: bool = False,
    ) -> dict[str, Any] | None:
        operation = local_agent_sync_operation(source["type"], source.get("config"))
        if operation is None:
            return None
        if source.get("status") != "active":
            raise HTTPException(status_code=409, detail="source_not_active")
        owner = execution_owner_user_id(source)
        if owner is None:
            raise HTTPException(
                status_code=409,
                detail="local_agent_sync_execution_owner_required",
            )
        job_id = f"laj-{uuid.uuid4().hex}"
        payload = await local_agent_sync_job_payload(
            db,
            source,
            {"force_full_sync": force_full_sync, "retry_failed_items": True, **({"dry_run": True} if dry_run else {})},
        )
        job_id, created = await _enqueue_sync_local_agent_job(
            request,
            db,
            job_id=job_id,
            source_id=source["id"],
            source_type=source["type"],
            operation=operation,
            payload=payload,
            created_by_user_id=resolve_request_principal(request),
            execution_owner_user_id=owner,
        )
        return {
            "ok": True,
            "message": "Local collection enqueued",
            "source_id": source["id"],
            "job_id": job_id,
            "status": "queued",
            "coalesced": not created,
        }

    @source_router.get("/{source_id}/sync-runs/{run_id}", response_model=SourceSyncRunStateResponse)
    async def read_source_sync_run(
        request: Request,
        source_id: str,
        run_id: str,
        db: Database = Depends(get_db),
        workspace_id: str = Depends(get_workspace_id),
    ):
        """Resolve an exact admission receipt without advancing or replaying it."""
        source = await db.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_sync_execution(request, source)
        run = await db.get_source_sync_run(run_id)
        if run is None or run.source_id != source_id or run.workspace_id != workspace_id:
            raise HTTPException(status_code=404, detail="Sync run not found")
        return {"run_id": run.run_id, "status": run.status}

    @source_router.post(
        "/{source_id}/sync",
        status_code=202,
        response_model=SourceSyncRunReceiptResponse | LocalCollectionReceiptResponse,
    )
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
        _require_source_sync_execution(request, source)
        _require_source_sync_support(source)

        retry_run_id = req.retry_target.execution_id if req and req.retry_target else None
        dry_run = bool(req and req.dry_run)
        if dry_run and (retry_run_id is not None or _conversation_gene(source) is None):
            # A dry run plans one new device collection of a conversation Source.
            raise HTTPException(status_code=400, detail="dry_run_requires_a_new_conversation_collection")
        if retry_run_id is None:
            local_job = await _enqueue_local_source_collection_job(
                request, db, source, force_full_sync=bool(req and req.force_full_sync), dry_run=dry_run
            )
            if local_job is not None:
                return local_job
            if dry_run:
                raise HTTPException(status_code=400, detail="dry_run_requires_a_new_conversation_collection")

        try:
            run = await sync_service.enqueue_source(
                source_id,
                trigger="manual",
                force_full_sync=bool(req and req.force_full_sync),
                retry_run_id=retry_run_id,
            )
        except SourcePausedError:
            raise _source_paused_http_error()
        except (
            SourceActivityConflict,
            SourceSyncUnsupportedError,
            ValueError,
        ) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "ok": True,
            "message": "Sync enqueued",
            "source_id": source_id,
            "run_id": run.run_id,
            "status": run.status,
            "created_at": run.created_at.isoformat(),
            "coalesced": run.coalesced,
        }

    @source_router.post("/{source_id}/sync/stop", response_model=SourceSyncStopResponse)
    async def stop_source_sync(
        request: Request,
        source_id: str,
        db: Database = Depends(get_db),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """Stop the Source's sync in progress. The Source stays active and later syncs run as scheduled.

        A person who manages the Source, or who runs its sync, can stop it.
        A server run that has not started, and the collection jobs of the
        Source's local daemon, end at once. A running server run ends at its
        worker's next heartbeat: what it committed stays, and it removes
        nothing and does not advance the sync cursor.
        """
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_discoverability(request, source)
        # Whoever can pause the Source can stop its sync, as can whoever runs it.
        if not can_manage_source(
            source,
            viewer_id=resolve_request_principal(request),
            viewer_role=resolve_request_workspace_role(request),
        ):
            _require_source_sync_execution(request, source)
        _require_source_sync_support(source)
        return await _stop_source_sync(request, db, sync_service, source)

    @source_router.post("/{source_id}/process", status_code=202)
    async def process_collected_local_source(
        request: Request,
        source_id: str,
        req: SourceSyncRequest | None = Body(default=None),
        db: Database = Depends(get_db),
        workspace_id: str = Depends(get_workspace_id),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """Enqueue server processing after the owner daemon persisted raw input."""
        source = await db.get_source(source_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        if local_agent_sync_operation(source["type"], source.get("config")) is None:
            raise HTTPException(status_code=400, detail="source_is_not_local_agent_backed")
        _require_source_sync_execution(request, source)
        _require_source_sync_support(source)
        current_source = await db.get_source(source_id)
        if current_source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        try:
            input_snapshot_id = local_agent_collection_attempt_id(
                current_source["type"],
                req.local_agent_job_id if req else None,
                req.local_agent_attempt_count if req else None,
                req.sync_snapshot_id if req else None,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        current_source = await db.get_source(source_id)
        if current_source is None:
            raise HTTPException(status_code=404, detail="Source not found")
        lease_payload = await _require_current_local_agent_lease(
            request,
            db,
            source=current_source,
            job_id=req.local_agent_job_id if req else None,
            attempt_count=req.local_agent_attempt_count if req else None,
        )
        if lease_payload.get("dry_run"):
            # A dry run reports its plan; nothing it collected is processed.
            raise HTTPException(status_code=409, detail="dry_run_job_is_not_processed")
        if _conversation_gene(current_source) is not None:
            await _require_complete_collection(
                _collection_intake(db, StoredCollections(db, workspace_id=workspace_id)),
                source_id,
                input_snapshot_id,
            )
        else:
            await _require_ready_snapshot_manifest(
                db,
                source=current_source,
                source_id=source_id,
                workspace_id=workspace_id,
                input_snapshot_id=input_snapshot_id,
                req=req,
                lease_payload=lease_payload,
            )
        try:
            run = await sync_service.enqueue_source(
                source_id,
                trigger="local_agent",
                force_full_sync=bool(
                    lease_payload.get("force_full_sync")
                    if "force_full_sync" in lease_payload
                    else req and req.force_full_sync
                ),
                input_snapshot_id=input_snapshot_id,
                source_config_revision=str(lease_payload["source_config_revision"]),
                predecessor_activity_id=(req.local_agent_job_id if req else None),
                retry_failed_items=bool(lease_payload.get("retry_failed_items")),
            )
        except SourcePausedError:
            raise _source_paused_http_error()
        except (
            SourceActivityConflict,
            SourceSyncUnsupportedError,
        ) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "ok": True,
            "source_id": source_id,
            "run_id": run.run_id,
            "status": run.status,
            "created_at": run.created_at.isoformat(),
            "coalesced": run.coalesced,
        }

    @source_router.post(
        "/{source_id}/force-resync",
        status_code=202,
        response_model=SourceSyncRunReceiptResponse | LocalCollectionReceiptResponse,
    )
    async def trigger_force_resync(
        request: Request,
        source_id: str,
        db: Database = Depends(get_db),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """Enqueue a full source sync without clearing the prior successful cursor."""
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_sync_execution(request, source)
        _require_source_sync_support(source)

        local_job = await _enqueue_local_source_collection_job(
            request,
            db,
            source,
            force_full_sync=True,
        )
        if local_job is not None:
            return local_job

        try:
            run = await sync_service.enqueue_source(
                source_id,
                trigger="force",
                force_full_sync=True,
            )
        except SourcePausedError:
            raise _source_paused_http_error()
        except (
            SourceActivityConflict,
            SourceSyncUnsupportedError,
        ) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "ok": True,
            "message": "Sync enqueued",
            "source_id": source_id,
            "run_id": run.run_id,
            "status": run.status,
            "created_at": run.created_at.isoformat(),
            "coalesced": run.coalesced,
        }

    @source_router.post(
        "/{source_id}/reprocess",
        status_code=202,
        responses={
            200: {"model": ReprocessPreviewResponse, "description": "The dry run's preview."},
            202: {"model": SourceReprocessQueuedResponse, "description": "The queued run."},
        },
    )
    async def reprocess_source_documents(
        request: Request,
        source_id: str,
        req: SourceReprocessRequest,
        response: Response,
        db: Database = Depends(get_db),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
        sync_service: SyncService = Depends(get_sync_service),
    ):
        """Reprocess the Source Units of these Documents with the current adapter and compiler.

        A Source that can ask its provider for one Document by id reads each
        Document's current state from the provider; any other Source reads the
        stored input, so it needs no local daemon. ``all_documents`` takes every
        current Document of the Source. ``dry_run`` reports what the run would
        read and an estimate of its model calls without writing or contacting
        the provider. A queued Document the run cannot reprocess is recorded as
        a failed item of the run.
        """
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_discoverability(request, source)
        _require_source_management(request, source)
        _require_source_sync_support(source)
        if req.all_documents:
            if req.document_ids:
                raise HTTPException(status_code=400, detail="document_ids_with_all_documents")
            document_ids = tuple(sorted(await db.list_indexed_doc_ids(source_id)))
            if not document_ids:
                raise HTTPException(status_code=409, detail="source_has_no_documents")
        else:
            document_ids = tuple(
                dict.fromkeys(document_id.strip() for document_id in req.document_ids if document_id.strip())
            )
            if not document_ids:
                raise HTTPException(status_code=400, detail="document_ids_required")
        if await db.get_open_projection_scope_transition(source_id) is not None:
            raise HTTPException(status_code=409, detail="projection_scope_transition_open")
        if req.dry_run:
            response.status_code = 200
            source_config = source.get("config") or {}
            try:
                gene = create_gene(
                    source["type"],
                    decrypt_source_config_for_runtime(
                        source_config,
                        secret_fields=source_secret_fields(source["type"], GENE_REGISTRY),
                    ),
                    source_id=source_id,
                )
            except (SourceConfigurationError, ValueError) as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            gene.bind_document_store(artifact_store)
            return asdict(await reprocess_preview(
                db, artifact_store, gene, source_id=source_id, document_ids=document_ids,
                rediscovers=source_rediscovers_documents(source["type"], source_config),
                projection_adapter=DEFAULT_SOURCE_PROJECTION_ADAPTER,
                projection_scope=_sync_scope_config(source["type"], source_config),
                access_context=source_access_context(source),
            ))
        try:
            run = await sync_service.enqueue_reprocess(source_id, document_ids)
        except SourcePausedError:
            raise _source_paused_http_error()
        except (
            SourceActivityConflict,
            SourceSyncBoundaryError,
            ValueError,
        ) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "ok": True,
            "message": "Reprocess enqueued",
            "source_id": source_id,
            "run_id": run.run_id,
            "status": run.status,
            "created_at": run.created_at.isoformat(),
            "document_ids": list(run.reprocess_document_ids),
        }

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
        if req.enabled:
            if not source_type_supports_sync(str(source.get("type") or "")):
                raise HTTPException(status_code=409, detail="source_sync_not_supported")
        await db.set_source_sync_schedule(
            source_id,
            enabled=req.enabled,
            interval_minutes=req.interval_minutes,
        )
        return {"ok": True, "source_id": source_id}

    @source_router.post("/{source_id}/adapter/artifacts")
    async def push_local_source_artifact(
        source_id: str,
        request: Request,
        source_unit_key: str,
        provider_key: str,
        provider_revision: str,
        parent_observation_type: str,
        parent_provider_key: str,
        filename: str,
        media_type: str,
        local_agent_job_id: str | None = None,
        local_agent_attempt_count: int | None = None,
        db: Database = Depends(get_db),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
        workspace_id: str = Depends(get_workspace_id),
    ):
        """Receive one bounded binary input for a later local-source package."""

        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        collection = _local_agent_collection(source.get("type"))
        if not (collection is not None and collection.accepts_artifact_inputs) and _conversation_gene(source) is None:
            raise HTTPException(status_code=400, detail="source does not support local Artifact inputs")
        _require_source_sync_execution(request, source)
        _require_source_sync_support(source)
        await _require_current_local_agent_lease(
            request,
            db,
            source=source,
            job_id=local_agent_job_id,
            attempt_count=local_agent_attempt_count,
        )
        normalized_media_type = normalize_source_artifact_media_type(media_type)
        request_media_type = normalize_source_artifact_media_type(request.headers.get("content-type"))
        if normalized_media_type not in SUPPORTED_SOURCE_ARTIFACT_MEDIA_TYPES:
            raise HTTPException(status_code=400, detail="unsupported Source Artifact media type")
        if request_media_type != normalized_media_type:
            raise HTTPException(status_code=400, detail="Source Artifact Content-Type mismatch")
        required_values = {
            "source_unit_key": source_unit_key,
            "provider_key": provider_key,
            "provider_revision": provider_revision,
            "parent_observation_type": parent_observation_type,
            "parent_provider_key": parent_provider_key,
            "filename": filename,
        }
        if any(not str(value).strip() for value in required_values.values()):
            raise HTTPException(status_code=400, detail="Source Artifact identity is incomplete")
        try:
            declared_length = parse_source_artifact_content_length(request.headers.get("content-length"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if declared_length is not None and declared_length > MAX_SOURCE_ARTIFACT_STORAGE_BYTES:
            raise HTTPException(status_code=413, detail="Source Artifact exceeds storage limit")

        content = tempfile.SpooledTemporaryFile(
            max_size=SOURCE_ARTIFACT_SPOOL_MEMORY_BYTES,
            mode="w+b",
        )
        digest = hashlib.sha256()
        observed_size = 0
        try:
            async for chunk in request.stream():
                if not chunk:
                    continue
                observed_size += len(chunk)
                if observed_size > MAX_SOURCE_ARTIFACT_STORAGE_BYTES:
                    raise HTTPException(status_code=413, detail="Source Artifact exceeds storage limit")
                digest.update(chunk)
                content.write(chunk)
            if declared_length is not None and observed_size != declared_length:
                raise HTTPException(status_code=400, detail="Source Artifact transport length mismatch")
            content.seek(0)
            input_sha256 = await retain_artifact_input(
                db,
                artifact_store,
                source_id=source_id,
                workspace_id=workspace_id,
                identity={key: str(value).strip() for key, value in required_values.items()},
                media_type=normalized_media_type,
                content=content,
                size_bytes=observed_size,
                content_sha256=digest.hexdigest(),
            )
            return {
                "input_sha256": input_sha256,
                "content_sha256": digest.hexdigest(),
                "size_bytes": observed_size,
            }
        finally:
            content.close()

    @source_router.post("/{source_id}/adapter/packages")
    async def push_local_source_package(
        source_id: str,
        req: LocalSourcePackageRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
        workspace_id: str = Depends(get_workspace_id),
    ):
        """Receive one local-source package pushed by the local daemon/adapter.

        The service owns the inbox layout and package format. Local clients
        send raw file text or structured source payloads; source genes perform
        canonical markdown normalization during sync.
        """
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        source_type = source.get("type")
        collection = _local_agent_collection(source_type)
        if collection is None or collection.package_submitter is None:
            raise HTTPException(
                status_code=400,
                detail=(f"source {source_id} is type {source_type!r}, not a local adapter source"),
            )
        _require_source_sync_execution(request, source)
        _require_source_sync_support(source)
        await _require_current_local_agent_lease(
            request,
            db,
            source=source,
            job_id=req.local_agent_job_id,
            attempt_count=req.local_agent_attempt_count,
        )
        if source.get("status") == SOURCE_PAUSED_STATUS:
            raise HTTPException(status_code=400, detail="Source is paused")
        try:
            input_snapshot_id = local_agent_collection_attempt_id(
                source_type,
                req.local_agent_job_id,
                req.local_agent_attempt_count,
                req.sync_snapshot_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

        try:
            source_artifacts = await _resolve_local_source_artifact_inputs(
                db,
                source_id=source_id,
                workspace_id=workspace_id,
                input_hashes=req.artifact_input_hashes,
                expected_source_unit_key=str(req.artifact_source_unit_key or "").strip(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        submit_package = load_reference(collection.package_submitter)
        try:
            result = await submit_package(
                req,
                db=db,
                workspace_id=workspace_id,
                config=config,
                source=source,
                document_store=artifact_store,
                sync_snapshot_id=input_snapshot_id,
                source_artifacts=source_artifacts,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        if result.get("package_uri"):
            package_uri = str(result["package_uri"])

            async def discard_unretained_package() -> None:
                await db.enqueue_source_artifact_cleanup_task(
                    source_id=source_id,
                    artifact_uri=package_uri,
                )

            try:
                current_source = await db.get_source(source_id)
                if current_source is None:
                    raise HTTPException(status_code=404, detail="Source not found")
                await _require_current_local_agent_lease(
                    request,
                    db,
                    source=current_source,
                    job_id=req.local_agent_job_id,
                    attempt_count=req.local_agent_attempt_count,
                )
                raw_sha256 = local_agent_semantic_input_sha256(
                    result.get("doc_id"),
                    result.get("semantic_hash"),
                )
                if not raw_sha256:
                    raise HTTPException(
                        status_code=500,
                        detail="local_agent_package_hash_missing",
                    )
                manifest_entry = result.get("package_manifest_entry")
                package_sha256 = str(result.get("package_sha256") or "").strip()
                if not package_sha256:
                    raise HTTPException(
                        status_code=500,
                        detail="local_agent_package_artifact_hash_missing",
                    )
                if isinstance(manifest_entry, dict):
                    manifest_entry = {
                        **manifest_entry,
                        "package_sha256": package_sha256,
                    }
                retained_input = await db.create_source_sync_input(
                    source_id=source_id,
                    workspace_id=workspace_id,
                    raw_uri=package_uri,
                    raw_sha256=raw_sha256,
                    raw_content_type="application/json",
                    metadata={
                        "doc_id": result.get("doc_id"),
                        "source_type": source_type,
                        "package_path": result.get("package_path"),
                        "submitted_at": result.get("submitted_at"),
                        "submitted_by": req.submitted_by,
                        "package_sha256": package_sha256,
                        "manifest_entry": (manifest_entry if isinstance(manifest_entry, dict) else {}),
                    },
                    sync_snapshot_id=input_snapshot_id,
                )
            except HTTPException:
                await discard_unretained_package()
                raise
            except ValueError as exc:
                await discard_unretained_package()
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            except Exception:
                await discard_unretained_package()
                raise
            if retained_input.raw_uri != package_uri:
                try:
                    retained_input, retained_package_hash = await _attest_retained_local_source_input(
                        db=db,
                        artifact_store=artifact_store,
                        source_type=source_type,
                        source_id=source_id,
                        retained_input=retained_input,
                    )
                except ValueError as exc:
                    await discard_unretained_package()
                    raise HTTPException(status_code=409, detail=str(exc)) from exc
                except Exception:
                    await discard_unretained_package()
                    raise
                await discard_unretained_package()
                result["package_uri"] = retained_input.raw_uri
                result["package_sha256"] = retained_package_hash

        public_result = dict(result)
        public_result.pop("package_manifest_entry", None)
        return {**public_result, "sync_started": False}

    async def _conversation_upload(
        request: Request,
        db: Database,
        *,
        source_id: str,
        workspace_id: str,
        attempt: ConversationCollectionAttemptRequest,
    ) -> _ConversationUpload:
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        gene = _conversation_gene(source)
        if gene is None:
            raise HTTPException(status_code=400, detail="source_is_not_a_conversation_source")
        _require_source_sync_execution(request, source)
        _require_source_sync_support(source)
        lease_payload = await _require_current_local_agent_lease(
            request,
            db,
            source=source,
            job_id=attempt.local_agent_job_id,
            attempt_count=attempt.local_agent_attempt_count,
        )
        collections = StoredCollections(db, workspace_id=workspace_id)
        return _ConversationUpload(
            gene=gene,
            collections=collections,
            intake=_collection_intake(db, collections),
            collection_attempt_id=local_agent_sync_snapshot_id(
                attempt.local_agent_job_id, attempt.local_agent_attempt_count
            ),
            config_revision=str(lease_payload["source_config_revision"]),
        )

    @source_router.post("/{source_id}/conversation-collection/request")
    async def conversation_collection_request(
        source_id: str,
        req: ConversationCollectionAttemptRequest,
        request: Request,
        db: Database = Depends(get_db),
        workspace_id: str = Depends(get_workspace_id),
    ):
        """The request one device collection attempt answers; asking again returns the same request."""

        upload = await _conversation_upload(request, db, source_id=source_id, workspace_id=workspace_id, attempt=req)
        collection_request = await upload.gene.issue_collection_request(
            upload.collections, upload.collection_attempt_id, now=datetime.now(timezone.utc)
        )
        return {
            "collection_attempt_id": upload.collection_attempt_id,
            "request": request_to_payload(collection_request),
        }

    @source_router.post("/{source_id}/conversation-collection/plan")
    async def plan_conversation_collection(
        source_id: str,
        req: ConversationCollectionAttemptRequest,
        request: Request,
        db: Database = Depends(get_db),
        document_store: DocumentArtifactStore = Depends(get_document_store),
        workspace_id: str = Depends(get_workspace_id),
    ):
        """What one uploaded device collection attempt would change, without processing it.

        The plan is the one a sync run would make now: the run's collection
        family binds the attempt and starts the open Projection Scope
        transition, with its removals, as the run does.
        """

        upload = await _conversation_upload(request, db, source_id=source_id, workspace_id=workspace_id, attempt=req)
        await _require_complete_collection(upload.intake, source_id, upload.collection_attempt_id)
        family = collection_family(upload.gene, db=db, doc_store=document_store, source_id=source_id)
        family.prepare(
            collection_attempt=CollectionAttempt(workspace_id, upload.collection_attempt_id),
            input_snapshot=None,
        )
        transition = await db.get_open_projection_scope_transition(source_id)
        if transition is not None:
            family.begin_scope_transition(transition)
        return {
            "collection_attempt_id": upload.collection_attempt_id,
            **plan_report(await upload.gene.preview(), source_id=source_id),
        }

    @source_router.post("/{source_id}/conversation-collection/manifest")
    async def receive_conversation_manifest(
        source_id: str,
        req: ConversationManifestRequest,
        request: Request,
        db: Database = Depends(get_db),
        workspace_id: str = Depends(get_workspace_id),
    ):
        """Keep the Collection Manifest of one device collection attempt and name what the server lacks."""

        upload = await _conversation_upload(request, db, source_id=source_id, workspace_id=workspace_id, attempt=req)
        try:
            missing = await upload.intake.receive_manifest(
                req.manifest,
                source_id=source_id,
                collection_attempt_id=upload.collection_attempt_id,
                config_revision=upload.config_revision,
                record_schema=upload.gene.record_schema,
                received_at=datetime.now(timezone.utc),
            )
        except ManifestFormatError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except CollectionRejected as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "collection_attempt_id": upload.collection_attempt_id,
            "missing": [
                {
                    "conversation_id": item.identity.conversation_id,
                    "message_id": item.identity.message_id,
                    "body": item.body,
                    "artifacts": [
                        {"provider_key": artifact.provider_key, "provider_revision": artifact.provider_revision}
                        for artifact in item.artifacts
                    ],
                }
                for item in missing
            ],
        }

    @source_router.post("/{source_id}/conversation-collection/bodies")
    async def receive_conversation_bodies(
        source_id: str,
        req: ConversationBodiesRequest,
        request: Request,
        db: Database = Depends(get_db),
        workspace_id: str = Depends(get_workspace_id),
    ):
        """Keep message bodies of one device collection attempt, each matching its manifest message."""

        upload = await _conversation_upload(request, db, source_id=source_id, workspace_id=workspace_id, attempt=req)
        try:
            await upload.intake.receive_bodies(
                {
                    MessageIdentity(body.conversation_id, body.message_id): CollectedBody(
                        record=body.record,
                        artifacts=tuple(
                            ArtifactLink(
                                provider_key=link.provider_key,
                                provider_revision=link.provider_revision,
                                input_sha256=link.input_sha256,
                            )
                            for link in body.artifacts
                        ),
                    )
                    for body in req.bodies
                },
                source_id=source_id,
                collection_attempt_id=upload.collection_attempt_id,
            )
        except CollectionRejected as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"collection_attempt_id": upload.collection_attempt_id, "received": len(req.bodies)}

    @source_router.get("/{source_id}/projection-inventory")
    async def retired_projection_inventory(source_id: str):
        """Tell an agent that reads the projection inventory to upgrade: the server owns every conversation decision."""

        raise HTTPException(status_code=409, detail="local_agent_upgrade_required")

    @source_router.post("/{source_id}/adapter/manifest")
    async def prepare_local_source_manifest(
        source_id: str,
        req: LocalSourceManifestRequest,
        request: Request,
        db: Database = Depends(get_db),
        workspace_id: str = Depends(get_workspace_id),
    ):
        """Reuse unchanged immutable packages in one complete, fenced collection."""
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        if _conversation_gene(source) is not None:
            # A conversation Source uploads a Collection Manifest; an agent
            # that sends it document snapshots is told to upgrade.
            raise HTTPException(status_code=409, detail="local_agent_upgrade_required")
        _require_source_sync_execution(request, source)
        _require_source_sync_support(source)
        lease_payload = await _require_current_local_agent_lease(
            request,
            db,
            source=source,
            job_id=req.local_agent_job_id,
            attempt_count=req.local_agent_attempt_count,
        )
        try:
            snapshot_id = local_agent_collection_attempt_id(
                source.get("type"),
                req.local_agent_job_id,
                req.local_agent_attempt_count,
                req.sync_snapshot_id,
            )
            items = [SnapshotManifestItem(doc_id=item.doc_id, revision=item.revision) for item in req.items]
            manifest_items = [(item.doc_id, item.revision, UPSERT) for item in items]
            retained_inputs = await db.find_source_sync_input_attestations(
                source_id=source_id,
                workspace_id=workspace_id,
                manifest_items=manifest_items,
            )
            plan = plan_snapshot_manifest(
                items,
                retained_inputs,
                require_source_time=local_package_requires_source_time(
                    source.get("type"),
                    req.package_contract_version,
                ),
            )
            excluded_listings = tuple(
                ExcludedScopeListing(
                    scope_key=listing.scope_key, doc_ids=frozenset(listing.doc_ids), rejection=listing.rejection
                )
                for listing in req.excluded_listings
            )
            validate_excluded_listings_for_job(
                excluded_listings,
                (
                    str(entry.get("scope_key") or "")
                    for entry in lease_payload.get(LOCAL_AGENT_EXCLUDED_SCOPES_PAYLOAD_FIELD) or ()
                    if isinstance(entry, Mapping)
                ),
            )
            manifest_sha256 = snapshot_manifest_sha256(items, excluded_listings)
            confirm = await _snapshot_confirm_set(
                db,
                source=source,
                workspace_id=workspace_id,
                snapshot_id=snapshot_id,
                listed_doc_ids=frozenset(item.doc_id for item in items),
                excluded_listings=excluded_listings,
                package_contract_version=req.package_contract_version,
            )
            await db.attach_source_sync_inputs_to_snapshot(
                source_id=source_id,
                workspace_id=workspace_id,
                snapshot_id=snapshot_id,
                memberships=list(plan.reused_memberships),
                coverage=COMPLETE_SNAPSHOT,
                manifest_count=len(req.items),
                manifest_sha256=manifest_sha256,
                manifest_items=manifest_items,
                local_agent_job_id=req.local_agent_job_id,
                local_agent_attempt_count=req.local_agent_attempt_count,
                source_config_revision=str(lease_payload["source_config_revision"]),
                confirm=confirm,
                excluded_listings=excluded_listings,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "source_id": source_id,
            "snapshot_id": snapshot_id,
            "coverage": COMPLETE_SNAPSHOT,
            "manifest_count": len(req.items),
            "manifest_sha256": manifest_sha256,
            "reused_count": len(plan.reused_memberships),
            "required_doc_ids": list(plan.required_doc_ids),
            "confirm": [{"doc_id": item.doc_id, "provider_key": item.provider_key} for item in confirm],
        }

    @source_router.post("/{source_id}/adapter/manifest/confirmations")
    async def submit_local_source_confirmations(
        source_id: str,
        req: LocalSourceConfirmationsRequest,
        request: Request,
        db: Database = Depends(get_db),
        workspace_id: str = Depends(get_workspace_id),
    ):
        """Store a device's answers for the items its manifest response asked it to confirm.

        The answers bind to the job attempt and the digest of the manifest. An
        identical replay is accepted; an item the manifest did not ask about,
        another digest or attempt, and answers that differ from the stored
        ones are refused with 409.
        """
        source = await db.get_source(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")
        _require_source_sync_execution(request, source)
        _require_source_sync_support(source)
        await _require_current_local_agent_lease(
            request,
            db,
            source=source,
            job_id=req.local_agent_job_id,
            attempt_count=req.local_agent_attempt_count,
        )
        try:
            snapshot_id = local_agent_collection_attempt_id(
                source.get("type"),
                req.local_agent_job_id,
                req.local_agent_attempt_count,
                req.sync_snapshot_id,
            )
            await db.write_source_sync_snapshot_confirmations(
                source_id=source_id,
                workspace_id=workspace_id,
                snapshot_id=snapshot_id,
                manifest_sha256=req.manifest_sha256,
                local_agent_job_id=req.local_agent_job_id,
                local_agent_attempt_count=req.local_agent_attempt_count,
                answers=tuple(
                    SnapshotConfirmation(
                        item.doc_id, item.provider_key, SnapshotConfirmationAnswer(item.answer)
                    )
                    for item in req.answers
                ),
                failure_cause=req.failure_cause,
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "source_id": source_id,
            "snapshot_id": snapshot_id,
            "answered_count": len(req.answers),
            "failed": req.failure_cause is not None,
        }

    # ===================================================================
    # 4b. Removed agent-session capture
    # ===================================================================

    @agent_session_router.post(
        "/windows",
        status_code=410,
        deprecated=True,
        responses={410: {"model": AgentSessionCaptureRemovedResponse}},
    )
    async def refuse_agent_session_window() -> None:
        """Refuse an automatic session-window upload from an earlier memory plugin.

        MemForge no longer captures agent sessions. The route reads no body,
        creates no Source, calls no model and takes no Source activity; it
        answers ``410`` with ``agent_session_capture_removed`` so the upload
        stops at the plugin. It is removed in the next release.
        """
        _refuse_removed_agent_session_capture()

    @hook_router.post(
        "/receipts",
        status_code=410,
        deprecated=True,
        responses={410: {"model": AgentSessionCaptureRemovedResponse}},
    )
    async def refuse_agent_hook_receipt() -> None:
        """Refuse a turn-end hook receipt from an earlier memory plugin.

        Same contract as ``POST /api/v1/agent-sessions/windows``: no body is
        read and nothing is recorded. It is removed in the next release.
        """
        _refuse_removed_agent_session_capture()

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
    # Projects
    # ===================================================================

    @projects_router.get("", response_model=ProjectListResponse)
    async def list_projects_route(
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        """List projects with the number of memories the caller can see in each."""
        adapters = await _build_project_adapters(db, config, runtime_provider)
        rows = await adapters.relational.list_projects()
        memory_counts = await count_project_memories(
            db,
            scope=_workspace_default_scope(request, include_private=True),
            project_keys=[project.key for project in rows],
        )
        return ProjectListResponse(
            data=[
                ProjectListItemResponse(
                    **_project_to_response(project).model_dump(),
                    memory_count=memory_counts[project.key],
                )
                for project in rows
            ],
            can_manage=_can_manage_projects(request),
        )

    @projects_router.post("", response_model=ProjectResponse, status_code=201)
    async def create_project_route(
        request: Request,
        req: ProjectCreateRequest,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        _require_project_management(request)
        key = (req.key or _derive_project_key(req.name)).strip()
        if not key:
            raise HTTPException(status_code=400, detail="project key cannot be empty")
        if key.upper() in RESERVED_PROJECT_KEYS:
            raise _reserved_project_conflict(key.upper())
        adapters = await _build_project_adapters(db, config, runtime_provider)
        try:
            created = await adapters.relational.create_project(
                key=key,
                name=req.name,
                is_shared=(req.kind == "shared"),
            )
        except ValueError:
            raise HTTPException(
                status_code=409,
                detail=f"A project with the code {key} already exists. Pick another name or code.",
            )
        return _project_to_response(created)

    @projects_router.patch("/{project_id}", response_model=ProjectResponse)
    async def update_project_route(
        request: Request,
        project_id: str,
        req: ProjectUpdateRequest,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        _require_project_management(request)
        adapters = await _build_project_adapters(db, config, runtime_provider)
        await _require_mutable_project(adapters.relational, project_id)
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

    @projects_router.get("/{project_id}/deletion-impact", response_model=ProjectDeletionImpactResponse)
    async def get_project_deletion_impact(
        request: Request,
        project_id: str,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        """Count what deleting the project changes, so the admin can confirm it knowingly."""
        _require_project_management(request)
        adapters = await _build_project_adapters(db, config, runtime_provider)
        await _require_mutable_project(adapters.relational, project_id)
        memory_ids = await adapters.relational.list_project_memory_ids(project_id)
        released_source_ids = await adapters.relational.list_sources_released_by_project_deletion(project_id)
        return ProjectDeletionImpactResponse(memory_count=len(memory_ids), source_count=len(released_source_ids))

    @projects_router.delete(
        "/{project_id}",
        response_model=ProjectDeleteResponse,
        dependencies=[Depends(refuse_memory_writes_during_reindex)],
    )
    async def delete_project_route(
        request: Request,
        project_id: str,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        _require_project_management(request)
        adapters = await _build_project_adapters(db, config, runtime_provider)
        await _require_mutable_project(adapters.relational, project_id)
        try:
            affected = await adapters.relational.list_project_memory_ids(project_id)
        except LookupError:
            raise HTTPException(status_code=404, detail="project not found")
        memory_store = await _build_memory_store(db, config, runtime_provider)
        # Vector metadata moves first so a failure here aborts the
        # transaction with both stores still pointing at the original
        # project. Only after the vector channel reports success does the
        # relational transaction rebucket the memories, release the Source
        # bindings and drop the project row together.
        await memory_store.rebucket_project_memories(
            affected,
            UNSORTED_PROJECT_KEY,
        )
        released_source_ids = await adapters.relational.commit_project_deletion(project_id, affected)
        return ProjectDeleteResponse(
            id=project_id,
            rebucketed_count=len(affected),
            rebucketed_memory_ids=affected,
            released_source_count=len(released_source_ids),
        )

    # ===================================================================
    # Memory Reviews
    # ===================================================================

    @review_router.get("", response_model=MemoryReviewListResponse)
    async def list_memory_reviews(
        request: Request,
        status: str | None = "open",
        origin: Literal["memory", "lifecycle"] | None = None,
        source_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
        db: Database = Depends(get_db),
    ):
        """List caller-visible Reviews with an exact post-stale-filter total.

        Eligibility is evaluated across every matching candidate before exact
        pagination. Lifecycle staged evidence is hydrated only for this page;
        Memory version reads and queue enumeration still grow with the queue.
        """
        if limit < 1 or limit > MAX_REVIEW_PAGE_SIZE:
            raise HTTPException(status_code=400, detail=f"limit must be between 1 and {MAX_REVIEW_PAGE_SIZE}")
        if offset < 0:
            raise HTTPException(status_code=400, detail="offset must not be negative")
        if status not in {None, "all", "open", "pending", "stale", "approved", "rejected"}:
            raise HTTPException(status_code=400, detail="unsupported review status")

        queue = await read_review_queue(
            db,
            scope=_lifecycle_visibility_scope(request),
            status=status if status and status != "all" else None,
            origin=origin,
            source_id=source_id,
        )
        return MemoryReviewListResponse(
            data=await _review_queue_page(request, db, queue, queue.entries[offset : offset + limit]),
            total=len(queue.entries),
            limit=limit,
            offset=offset,
        )

    @review_router.get("/{review_id}", response_model=MemoryReviewDetailResponse)
    async def get_memory_review(
        review_id: str,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
    ):
        review = await _get_supersede_review(db, review_id)
        lifecycle_review = await db.get_lifecycle_review(review_id)
        if review is not None and lifecycle_review is not None:
            raise HTTPException(status_code=409, detail="Review ID is ambiguous across storage origins")
        if review is None:
            if lifecycle_review is None:
                raise HTTPException(status_code=404, detail="Review not found")
            payload = await db.get_lifecycle_plan_payload(lifecycle_review.lifecycle_plan_id)
            source_id = (
                str(payload.get("scope", {}).get("source_id") or "")
                if isinstance(payload, Mapping) and isinstance(payload.get("scope"), Mapping)
                else ""
            )
            source = await db.get_source(source_id) if source_id else None
            if source is None:
                raise HTTPException(status_code=409, detail="Lifecycle review source is unavailable")
            _require_source_discoverability(request, source)
            await _require_lifecycle_review_visibility(request, db, lifecycle_review)
            return await _lifecycle_review_response(
                db,
                lifecycle_review,
                source,
                request=request,
                detail=True,
                config=config,
                artifact_store=artifact_store,
            )

        incumbent, challenger, related_memories = await _require_memory_review_visibility(
            request,
            db,
            review,
        )
        return await _memory_review_detail_response(
            db,
            review,
            request=request,
            incumbent=incumbent,
            challenger=challenger,
            related_challengers=related_memories,
            config=config,
            artifact_store=artifact_store,
        )

    @review_router.post(
        "/{review_id}/approve",
        response_model=MemoryReviewDetailResponse,
        dependencies=[Depends(refuse_memory_writes_during_reindex)],
    )
    async def approve_memory_review(
        review_id: str,
        request: Request,
        req: MemoryReviewDecisionRequest,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        lifecycle_review = await db.get_lifecycle_review(review_id)
        review_before = await db.get_memory_review(review_id)
        if lifecycle_review is not None and review_before is not None:
            raise HTTPException(status_code=409, detail="Review ID is ambiguous across storage origins")
        if lifecycle_review is not None:
            payload = await db.get_lifecycle_plan_payload(lifecycle_review.lifecycle_plan_id)
            scope = payload.get("scope") if isinstance(payload, Mapping) else None
            source_id = str(scope.get("source_id") or "") if isinstance(scope, Mapping) else ""
            await _decide_lifecycle_review(
                source_id,
                review_id,
                request,
                db,
                action="use_latest_state",
                expected_fingerprint=req.expected_fingerprint,
                note=req.note,
                config=config,
                runtime_provider=runtime_provider,
            )
            updated = await db.get_lifecycle_review(review_id)
            source = await db.get_source(source_id)
            assert updated is not None and source is not None
            return await _lifecycle_review_response(
                db,
                updated,
                source,
                request=request,
                detail=True,
                config=config,
                artifact_store=artifact_store,
            )

        if review_before is None:
            raise HTTPException(status_code=404, detail="Review not found")
        _, _, related_before = await _require_memory_review_management(request, db, review_before)
        if memory_review_decision_fingerprint(review_before, related_before) != req.expected_fingerprint:
            raise HTTPException(status_code=409, detail="Review decision fingerprint is stale")
        if review_before.status == "approved":
            return await get_memory_review(review_id, request, db, config, artifact_store)
        if review_before.status != "pending":
            raise HTTPException(status_code=409, detail=f"Review is already {review_before.status}")
        service = await _build_review_service(db, config, runtime_provider)
        try:
            result = await service.approve(
                review_id,
                reviewer=resolve_request_principal(request),
                note=req.note,
                expected_fingerprint=req.expected_fingerprint,
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
        _, _, related_after = await _require_memory_review_visibility(request, db, review)
        return await _memory_review_detail_response(
            db,
            review,
            request=request,
            incumbent=result.incumbent,
            challenger=result.challenger,
            related_challengers=related_after,
            config=config,
            artifact_store=artifact_store,
        )

    @review_router.post(
        "/{review_id}/reject",
        response_model=MemoryReviewDetailResponse,
        dependencies=[Depends(refuse_memory_writes_during_reindex)],
    )
    async def reject_memory_review(
        review_id: str,
        request: Request,
        req: MemoryReviewDecisionRequest,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        if not req.note or not req.note.strip():
            raise HTTPException(status_code=400, detail="A note is required to keep the current state")

        lifecycle_review = await db.get_lifecycle_review(review_id)
        review_before = await db.get_memory_review(review_id)
        if lifecycle_review is not None and review_before is not None:
            raise HTTPException(status_code=409, detail="Review ID is ambiguous across storage origins")
        if lifecycle_review is not None:
            payload = await db.get_lifecycle_plan_payload(lifecycle_review.lifecycle_plan_id)
            scope = payload.get("scope") if isinstance(payload, Mapping) else None
            source_id = str(scope.get("source_id") or "") if isinstance(scope, Mapping) else ""
            await _decide_lifecycle_review(
                source_id,
                review_id,
                request,
                db,
                action="keep_current_state",
                expected_fingerprint=req.expected_fingerprint,
                note=req.note,
                config=config,
                runtime_provider=runtime_provider,
            )
            updated = await db.get_lifecycle_review(review_id)
            source = await db.get_source(source_id)
            assert updated is not None and source is not None
            return await _lifecycle_review_response(
                db,
                updated,
                source,
                request=request,
                detail=True,
                config=config,
                artifact_store=artifact_store,
            )

        if review_before is None:
            raise HTTPException(status_code=404, detail="Review not found")
        _, _, related_before = await _require_memory_review_management(request, db, review_before)
        if memory_review_decision_fingerprint(review_before, related_before) != req.expected_fingerprint:
            raise HTTPException(status_code=409, detail="Review decision fingerprint is stale")
        if review_before.status == "rejected":
            return await get_memory_review(review_id, request, db, config, artifact_store)
        if review_before.status != "pending":
            raise HTTPException(status_code=409, detail=f"Review is already {review_before.status}")
        service = await _build_review_service(db, config, runtime_provider)
        try:
            result = await service.reject(
                review_id,
                reviewer=resolve_request_principal(request),
                note=req.note,
                expected_fingerprint=req.expected_fingerprint,
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
        _, _, related_after = await _require_memory_review_visibility(request, db, review)
        return await _memory_review_detail_response(
            db,
            review,
            request=request,
            incumbent=result.incumbent,
            challenger=result.challenger,
            related_challengers=related_after,
            config=config,
            artifact_store=artifact_store,
        )

    async def _validate_review_manifest_decision(
        item: MemoryReviewManifestDecision,
        request: Request,
        db: Database,
    ) -> MemoryReviewDecisionResult:
        memory_review = await _get_supersede_review(db, item.review_id)
        lifecycle_review = await db.get_lifecycle_review(item.review_id)
        if memory_review is not None and lifecycle_review is not None:
            return MemoryReviewDecisionResult(
                review_id=item.review_id,
                decision=item.decision,
                outcome="invalid",
                message="Review ID is ambiguous across storage origins",
            )
        if memory_review is None and lifecycle_review is None:
            return MemoryReviewDecisionResult(
                review_id=item.review_id,
                decision=item.decision,
                outcome="not_found",
                message="Review not found",
            )

        try:
            if lifecycle_review is not None:
                source_id = lifecycle_review.source_id or ""
                source = await db.get_source(source_id) if source_id else None
                if source is None:
                    return MemoryReviewDecisionResult(
                        review_id=item.review_id,
                        decision=item.decision,
                        outcome="invalid",
                        message="Lifecycle Review source is unavailable",
                    )
                await _require_lifecycle_review_visibility(request, db, lifecycle_review)
                _require_review_decision_authority(request, [source])
                fingerprint = lifecycle_review_decision_fingerprint(lifecycle_review)
                if fingerprint != item.expected_fingerprint:
                    return MemoryReviewDecisionResult(
                        review_id=item.review_id,
                        decision=item.decision,
                        outcome="stale",
                        status=lifecycle_review.status.value,
                        message="Review decision fingerprint is stale",
                    )
                target_status = "approved" if item.decision == "approve" else "rejected"
                if lifecycle_review.status.value == target_status:
                    return MemoryReviewDecisionResult(
                        review_id=item.review_id,
                        decision=item.decision,
                        outcome="already_applied",
                        status=target_status,
                    )
                if lifecycle_review.status.value == "stale":
                    return MemoryReviewDecisionResult(
                        review_id=item.review_id,
                        decision=item.decision,
                        outcome="stale",
                        status=lifecycle_review.status.value,
                        message="Lifecycle Review is stale",
                    )
                if lifecycle_review.status.value != "pending":
                    return MemoryReviewDecisionResult(
                        review_id=item.review_id,
                        decision=item.decision,
                        outcome="invalid",
                        status=lifecycle_review.status.value,
                        message=f"Review is already {lifecycle_review.status.value}",
                    )
                presentation = present_lifecycle_review(
                    staged_evidence=lifecycle_review.staged_evidence,
                    reason=lifecycle_review.reason,
                )
            else:
                assert memory_review is not None
                incumbent, challenger, related_challengers = await _require_memory_review_management(
                    request,
                    db,
                    memory_review,
                )
                fingerprint = memory_review_decision_fingerprint(
                    memory_review,
                    related_challengers,
                )
                if fingerprint != item.expected_fingerprint:
                    return MemoryReviewDecisionResult(
                        review_id=item.review_id,
                        decision=item.decision,
                        outcome="stale",
                        status=memory_review.status,
                        message="Review decision fingerprint is stale",
                    )
                target_status = "approved" if item.decision == "approve" else "rejected"
                if memory_review.status == target_status:
                    return MemoryReviewDecisionResult(
                        review_id=item.review_id,
                        decision=item.decision,
                        outcome="already_applied",
                        status=target_status,
                    )
                if memory_review.status != "pending":
                    return MemoryReviewDecisionResult(
                        review_id=item.review_id,
                        decision=item.decision,
                        outcome="invalid",
                        status=memory_review.status,
                        message=f"Review is already {memory_review.status}",
                    )
                if is_memory_review_stale(memory_review, incumbent, challenger):
                    return MemoryReviewDecisionResult(
                        review_id=item.review_id,
                        decision=item.decision,
                        outcome="stale",
                        status=memory_review.status,
                        message="Review participants changed after analysis",
                    )
                if (
                    memory_review.expected_support_set_hash is not None
                    and await db.get_memory_support_set_hash(memory_review.incumbent_memory_id)
                    != memory_review.expected_support_set_hash
                ):
                    return MemoryReviewDecisionResult(
                        review_id=item.review_id,
                        decision=item.decision,
                        outcome="stale",
                        status=memory_review.status,
                        message="The incumbent Support Set changed after the correction was proposed",
                    )
                if any(memory.status != "pending_review" for memory in related_challengers):
                    return MemoryReviewDecisionResult(
                        review_id=item.review_id,
                        decision=item.decision,
                        outcome="stale",
                        status=memory_review.status,
                        message="A related Review participant changed lifecycle state",
                    )
                presentation = present_memory_review(
                    reason=memory_review.reason,
                    source_backed_correction=memory_review.expected_support_set_hash is not None,
                )
        except HTTPException as exc:
            if exc.status_code == 403:
                outcome = "forbidden"
            elif exc.status_code == 404:
                outcome = "not_found"
            else:
                outcome = "invalid"
            return MemoryReviewDecisionResult(
                review_id=item.review_id,
                decision=item.decision,
                outcome=outcome,
                message=str(exc.detail),
            )

        action = next(
            (candidate for candidate in presentation.actions if candidate.decision == item.decision),
            None,
        )
        if action is None:
            return MemoryReviewDecisionResult(
                review_id=item.review_id,
                decision=item.decision,
                outcome="invalid",
                status="pending",
                message="Decision is not valid for this Review kind",
            )
        if action.requires_note and not (item.note and item.note.strip()):
            return MemoryReviewDecisionResult(
                review_id=item.review_id,
                decision=item.decision,
                outcome="invalid",
                status="pending",
                message=f"A note is required for {action.label}",
                decision_label=action.label,
                consequence=action.consequence,
            )
        return MemoryReviewDecisionResult(
            review_id=item.review_id,
            decision=item.decision,
            outcome="ready",
            status="pending",
            decision_label=action.label,
            consequence=action.consequence,
        )

    def _manifest_response(
        mode: Literal["validate", "apply"],
        results: list[MemoryReviewDecisionResult],
        *,
        validation_receipt: str | None = None,
    ) -> MemoryReviewDecisionManifestResponse:
        counts = {
            outcome: sum(item.outcome == outcome for item in results)
            for outcome in {
                "ready",
                "applied",
                "already_applied",
                "stale",
                "forbidden",
            }
        }
        failed = sum(item.outcome in {"not_found", "invalid", "failed"} for item in results)
        return MemoryReviewDecisionManifestResponse(
            mode=mode,
            results=results,
            ready=counts["ready"],
            applied=counts["applied"],
            already_applied=counts["already_applied"],
            stale=counts["stale"],
            forbidden=counts["forbidden"],
            failed=failed,
            validation_receipt=validation_receipt,
        )

    def _review_manifest_receipt(
        req: MemoryReviewDecisionManifestRequest,
        request: Request,
        config: AppConfig,
    ) -> str:
        payload = {
            "principal": resolve_request_principal(request),
            "workspace_id": get_workspace_id(request),
            "decisions": [item.model_dump(mode="json") for item in req.decisions],
        }
        canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        digest = hmac.new(
            config.server.jwt_secret.encode("utf-8"),
            canonical.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return f"review-manifest-v1:{digest}"

    @review_router.post(
        "/decisions/validate",
        response_model=MemoryReviewDecisionManifestResponse,
    )
    async def validate_memory_review_decisions(
        req: MemoryReviewDecisionManifestRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
    ):
        """Validate one bounded Decision Manifest without lifecycle mutation."""

        results = [await _validate_review_manifest_decision(item, request, db) for item in req.decisions]
        return _manifest_response(
            "validate",
            results,
            validation_receipt=_review_manifest_receipt(req, request, config),
        )

    @review_router.post(
        "/decisions/apply",
        response_model=MemoryReviewDecisionManifestResponse,
        dependencies=[Depends(refuse_memory_writes_during_reindex)],
    )
    async def apply_memory_review_decisions(
        req: MemoryReviewDecisionManifestRequest,
        request: Request,
        db: Database = Depends(get_db),
        config: AppConfig = Depends(get_config),
        artifact_store: DocumentArtifactStore = Depends(get_document_store),
        runtime_provider: RuntimeProvider = Depends(get_runtime_provider),
    ):
        """Apply a confirmed manifest through independent single-Review decisions."""

        expected_receipt = _review_manifest_receipt(req, request, config)
        if not req.validation_receipt or not hmac.compare_digest(
            req.validation_receipt,
            expected_receipt,
        ):
            raise HTTPException(
                status_code=409,
                detail="Decision Manifest must be validated unchanged before apply",
            )

        results: list[MemoryReviewDecisionResult] = []
        for item in req.decisions:
            validation = await _validate_review_manifest_decision(item, request, db)
            if validation.outcome != "ready":
                results.append(validation)
                continue
            try:
                decision_request = MemoryReviewDecisionRequest(
                    expected_fingerprint=item.expected_fingerprint,
                    note=item.note,
                )
                if item.decision == "approve":
                    detail = await approve_memory_review(
                        item.review_id,
                        request,
                        decision_request,
                        db,
                        config,
                        artifact_store,
                        runtime_provider,
                    )
                else:
                    detail = await reject_memory_review(
                        item.review_id,
                        request,
                        decision_request,
                        db,
                        config,
                        artifact_store,
                        runtime_provider,
                    )
                results.append(
                    MemoryReviewDecisionResult(
                        review_id=item.review_id,
                        decision=item.decision,
                        outcome="applied",
                        status=detail.status,
                        decision_label=validation.decision_label,
                        consequence=validation.consequence,
                    )
                )
            except HTTPException as exc:
                detail = exc.detail
                message = (
                    str(detail.get("message") or detail.get("error") or detail)
                    if isinstance(detail, Mapping)
                    else str(detail)
                )
                if exc.status_code == 403:
                    outcome = "forbidden"
                elif exc.status_code == 404:
                    outcome = "not_found"
                elif exc.status_code == 409 and "stale" in message.lower():
                    outcome = "stale"
                elif exc.status_code in {400, 409}:
                    outcome = "invalid"
                else:
                    outcome = "failed"
                results.append(
                    MemoryReviewDecisionResult(
                        review_id=item.review_id,
                        decision=item.decision,
                        outcome=outcome,
                        message=message,
                    )
                )
            except Exception as exc:
                logger.exception("Review manifest item failed: %s", item.review_id)
                results.append(
                    MemoryReviewDecisionResult(
                        review_id=item.review_id,
                        decision=item.decision,
                        outcome="failed",
                        message=str(exc),
                    )
                )
        return _manifest_response("apply", results)

    # -- Include all routers --
    # Local tool — no auth required. All routers accessible directly.
    def _shape_local_agent_job(job: dict[str, Any]) -> dict[str, Any]:
        return {
            "job_id": job["job_id"],
            "workspace_id": job["workspace_id"],
            "source_id": job["source_id"],
            "source_type": job["source_type"],
            "operation": job["operation"],
            "status": job["status"],
            "payload": job.get("payload") or {},
            "execution_owner_user_id": job["execution_owner_user_id"],
            "result": job.get("result") or {},
            "last_error": job.get("last_error"),
            "next_attempt_at": job.get("next_attempt_at"),
            "leased_until": job.get("leased_until"),
            "attempt_count": int(job.get("attempt_count") or 0),
            "created_at": job.get("created_at"),
            "updated_at": job.get("updated_at"),
            "finished_at": job.get("finished_at"),
        }

    @local_agent_router.post(
        "/jobs",
        status_code=201,
        response_model=LocalAgentJobCreateResponse,
        response_model_exclude_unset=True,
    )
    async def create_local_agent_job(
        req: LocalAgentJobCreateRequest,
        request: Request,
        db: Database = Depends(get_db),
        current_workspace_id: str = Depends(get_workspace_id),
    ):
        requester = resolve_request_principal(request)
        source_id = req.source_id.strip()
        payload = dict(req.payload)
        if req.retry_job_id is not None and req.operation not in LOCAL_AGENT_SYNC_OPERATIONS:
            raise HTTPException(status_code=400, detail="retry_target_requires_sync_operation")
        if req.operation in LOCAL_AGENT_SYNC_OPERATIONS:
            if not source_id:
                raise HTTPException(status_code=400, detail="local_agent_sync_requires_source_id")
            source = await db.get_source(source_id)
            if source is None:
                raise HTTPException(status_code=404, detail="local_agent_source_not_found")
            _require_source_discoverability(request, source)
            operation = local_agent_sync_operation(source["type"], source.get("config"))
            if operation != req.operation or source["type"] != req.source_type:
                raise HTTPException(status_code=400, detail="local_agent_operation_source_mismatch")
            owner = execution_owner_user_id(source)
            if owner is None:
                raise HTTPException(status_code=409, detail="local_agent_sync_execution_owner_required")
            if requester != owner:
                raise HTTPException(status_code=403, detail="local_agent_sync_execution_owner_forbidden")
            payload = await local_agent_sync_job_payload(db, source, payload)
        else:
            expected_type = LOCAL_AGENT_SETUP_OPERATION_SOURCE_TYPES.get(req.operation)
            if expected_type is None:
                raise HTTPException(status_code=400, detail="unsupported_local_agent_operation")
            if expected_type != req.source_type:
                raise HTTPException(status_code=400, detail="local_agent_operation_source_type_mismatch")
            if resolve_request_workspace_role(request) not in {"workspace_admin", "member"}:
                raise HTTPException(status_code=403, detail="local_agent_job_forbidden")
            owner = requester
        payload.pop("created_by_user_id", None)
        payload.pop("execution_owner_user_id", None)
        job_id = f"laj-{uuid.uuid4().hex}"
        workspace_id = (req.workspace_id or current_workspace_id or "default").strip() or "default"
        if req.operation in LOCAL_AGENT_SYNC_OPERATIONS:
            job_id, created = await _enqueue_sync_local_agent_job(
                request,
                db,
                job_id=job_id,
                workspace_id=workspace_id,
                source_id=source_id,
                source_type=req.source_type,
                operation=req.operation,
                payload=payload,
                created_by_user_id=requester,
                execution_owner_user_id=owner,
                retry_job_id=req.retry_job_id,
                trigger="manual",
            )
            current_job = await db.get_local_agent_job(job_id)
            return {
                "job_id": job_id,
                "status": current_job["status"],
                "created_at": current_job["created_at"],
                "coalesced": not created,
            }
        await db.insert_local_agent_job(
            job_id=job_id,
            workspace_id=workspace_id,
            source_id=source_id,
            source_type=req.source_type,
            operation=req.operation,
            payload=payload,
            created_by_user_id=requester,
            execution_owner_user_id=owner,
        )
        return {"job_id": job_id, "status": "queued", "coalesced": False}

    @local_agent_router.post("/jobs/lease")
    async def lease_local_agent_jobs(
        req: LocalAgentJobLeaseRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        requester = resolve_request_principal(request)
        deadline = time.monotonic() + req.wait_seconds
        while True:
            jobs = await db.lease_local_agent_jobs(
                user_id=requester,
                limit=req.limit,
                lease_seconds=req.lease_seconds,
            )
            if jobs or req.wait_seconds <= 0 or time.monotonic() >= deadline:
                return {"jobs": [_shape_local_agent_job(job) for job in jobs]}
            if await request.is_disconnected():
                return {"jobs": []}
            await asyncio.sleep(min(0.5, max(deadline - time.monotonic(), 0)))

    @local_agent_router.post("/jobs/{job_id}/heartbeat")
    async def heartbeat_local_agent_job(
        job_id: str,
        req: LocalAgentJobHeartbeatRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        requester = resolve_request_principal(request)
        try:
            progress = normalize_sync_progress_snapshot(req.progress) if req.progress is not None else None
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        wrote = await db.heartbeat_local_agent_job(
            job_id=job_id,
            user_id=requester,
            attempt_count=req.attempt_count,
            lease_seconds=req.lease_seconds,
            progress=progress,
        )
        if not wrote:
            raise HTTPException(status_code=404, detail="local_agent_job_not_found")
        return {"ok": True, "job_id": job_id, "status": "leased"}

    @local_agent_router.post("/jobs/{job_id}/complete")
    async def complete_local_agent_job(
        job_id: str,
        req: LocalAgentJobCompleteRequest,
        request: Request,
        db: Database = Depends(get_db),
    ):
        wrote = await db.complete_local_agent_job(
            job_id=job_id,
            user_id=resolve_request_principal(request),
            attempt_count=req.attempt_count,
            status=req.status,
            result=req.result,
            error=req.error,
            retryable=bool(req.result.get("retryable")),
        )
        if not wrote:
            current = await db.get_local_agent_job(job_id)
            if (
                not current
                or int(current.get("attempt_count") or 0) != req.attempt_count
                or current.get("execution_owner_user_id") != resolve_request_principal(request)
            ):
                raise HTTPException(status_code=404, detail="local_agent_job_not_found")
            status = str(current.get("status") or "")
            if status not in {"queued", "succeeded", "failed"}:
                raise HTTPException(status_code=404, detail="local_agent_job_not_found")
        else:
            status = local_agent_completion_status(
                req.status,
                retryable=bool(req.result.get("retryable")),
                attempt_count=req.attempt_count,
            )
        return {"ok": True, "job_id": job_id, "status": status}

    @local_agent_router.get("/jobs/current", response_model=LocalAgentJobListResponse)
    async def read_current_local_agent_jobs(
        request: Request,
        db: Database = Depends(get_db),
        workspace_id: str = Depends(get_workspace_id),
    ):
        jobs = await db.list_current_local_agent_jobs(
            workspace_id=workspace_id,
            user_id=resolve_request_principal(request),
        )
        return {"data": [_shape_local_agent_job(job) for job in jobs]}

    @local_agent_router.get("/jobs/{job_id}")
    async def read_local_agent_job(
        job_id: str,
        request: Request,
        db: Database = Depends(get_db),
    ):
        job = await db.get_local_agent_job(job_id)
        requester = resolve_request_principal(request)
        if job is None or requester not in {job["created_by_user_id"], job["execution_owner_user_id"]}:
            raise HTTPException(status_code=404, detail="local_agent_job_not_found")
        return _shape_local_agent_job(job)

    @local_agent_router.get("/status", response_model=LocalAgentDaemonStatusResponse)
    async def read_local_agent_status(
        request: Request,
        db: Database = Depends(get_db),
    ):
        now = datetime.now(timezone.utc)
        last_seen_at = await db.get_local_agent_heartbeat(resolve_request_principal(request))
        last_seen = datetime.fromisoformat(last_seen_at) if last_seen_at else None
        online = bool(last_seen and (now - last_seen).total_seconds() <= LOCAL_AGENT_STATUS_STALE_SECONDS)
        return {
            "status": "online" if online else "offline",
            "last_seen_at": last_seen_at,
            "checked_at": now.isoformat(),
            "stale_after_seconds": LOCAL_AGENT_STATUS_STALE_SECONDS,
        }

    app.include_router(auth_router)
    app.include_router(health_router)
    app.include_router(document_router)
    app.include_router(source_artifact_router)
    app.include_router(source_unit_router)
    app.include_router(memory_router)
    app.include_router(relation_discovery_router)
    app.include_router(review_router)
    app.include_router(entity_router)
    app.include_router(gene_router)
    app.include_router(source_router)
    app.include_router(evaluation_router)
    app.include_router(source_list_router)
    app.include_router(agent_session_router)
    app.include_router(hook_router)
    app.include_router(recent_change_router)
    app.include_router(model_settings_router)
    app.include_router(projects_router)
    app.include_router(local_agent_router)

    # -- Detailed stats endpoint (observability) --

    @app.get("/api/v1/stats/detailed")
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
