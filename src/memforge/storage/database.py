"""SQLite database for documents, memories, entities, sync state, and configuration.

Mirrors the schema from architecture.md Section 9. Uses aiosqlite with WAL mode.
FK enforcement is OFF — all cascades are implemented manually in delete methods.
FTS5 rows must be manually synced on insert/update/delete of memories.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import uuid
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import aiosqlite

from memforge.agent_session_contract import (
    AGENT_SESSION_WINDOW_SOURCE_KIND,
    successful_agent_session_activity_at,
)
from memforge.local_agent.source_contract import (
    local_agent_completion_status,
    local_agent_source_config_revision,
    local_agent_sync_job_payload,
    local_agent_sync_operation,
    source_sync_input_metadata_with_artifact_attestation,
)
from memforge.storage.admin_source import is_pause_only_source_update
from memforge.source_activity import (
    SourceActivityConflict,
    SourceActivityKind,
    SourceActivityLease,
)
from memforge.sync_progress import normalize_sync_progress_snapshot
from memforge.models import (
    AgentHookReceipt,
    AgentSessionReceipt,
    DocumentMetadata,
    DocumentRecord,
    Entity,
    EntityAlias,
    Memory,
    MemoryReview,
    MemoryReviewRelatedChallenger,
    MemorySource,
    Project,
    ReplacementKind,
    SHARED_PROJECT_KEY,
    SourceArtifactCleanupTask,
    SourceDeletionResult,
    SourceLifecycleResetResult,
    SourceSyncInput,
    SourceSyncRun,
    SyncState,
    UNSORTED_PROJECT_KEY,
    Visibility,
    canonicalize_entity_name,
    content_hash,
    slugify,
    source_artifact_cleanup_task_id,
)
from memforge.memory.evidence import (
    ActiveSupportEvidence,
    AuthorityCase,
    CandidateBucket,
    CandidateMemory,
    EvidenceContentProvenance,
    EvidenceReference,
    EvidenceRole,
    EvidenceRelationRecord,
    EvidenceUnit,
    LifecycleAction,
    MemorySupportAssertion,
    RelationCandidateRecord,
    RelationDirection,
    RelationOutcomeBundle,
    RelationRunRecord,
    RelationType,
    ReviewCase,
    evidence_relation_retry_identity,
    evidence_reference_id_for,
    relation_bundle_snapshot_audit,
    relation_candidate_retry_identity,
    validate_persisted_evidence_relation,
    validate_evidence_references,
)
from memforge.memory.lifecycle_plan import (
    build_unprovable_cutover_resolution,
    ClaimIdentityPolicy,
    ContestedSupportEdge,
    contested_supports_from_staged_evidence,
    CutoverFindingReason,
    CutoverFindingStatus,
    LifecycleCutoverFinding,
    LifecycleBackfillJob,
    LifecycleBackfillJobStatus,
    LegacyMemoryProvenance,
    LifecycleGate,
    LifecycleGateState,
    LifecycleMutationType,
    LifecyclePlan,
    LifecycleReview,
    LifecycleReviewStatus,
    LifecycleVectorOperation,
    LifecycleVectorTask,
    LifecycleVectorTaskStatus,
    lifecycle_plan_to_payload,
    pending_review_contested_supports,
    unprovable_cutover_retirement_plan_id,
    validate_unprovable_cutover_evidence,
)
from memforge.memory.relation_discovery_contract import (
    CURRENT_RELATION_EVIDENCE_PREDICATE_SQL,
    RelationDiscoveryRequest,
    RelationDiscoveryWork,
    RelationDiscoveryWorkStatus,
    resolve_relation_discovery_actor_user_id,
)
from memforge.memory.audit import MemoryAuditEvent
from memforge.memory.lifecycle import allowed_search_statuses, normalize_memory_status
from memforge.memory.review_contract import (
    CrossSourceReviewMemorySnapshot,
    CrossSourceReviewSupportSnapshot,
    validate_cross_source_review_write,
    validate_pending_review_retry,
)
from memforge.retrieval.access_predicate import visible_sql
from memforge.retrieval.metadata_text import metadata_alias_text, metadata_compact_text
from memforge.source_access import infer_legacy_source_access
from memforge.source_projection import (
    AnchorKind,
    ProjectionCoverage,
    ProjectionScopeTransition,
    ProjectionScopeTransitionStatus,
    SourceObservationRevision,
    SourceAnchor,
    SourceProjection,
    SourceUnit,
    SourceUnitInventoryFilter,
    SourceUnitInventoryPage,
    SourceUnitRevision,
    source_projection_from_payload,
    source_projection_to_payload,
)
from memforge.storage.admin_memory import (
    MemoryAdminListFilters,
    MemoryAdminQueryPage,
)
from memforge.storage.admin_source import (
    SOURCE_LIST_DEFAULT_SORT_MODE,
    SOURCE_SYNC_SCHEDULE_DEFAULT_INTERVAL_MINUTES,
    SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES,
    SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES,
    SourceListSortMode,
    validate_source_list_sort_mode,
)

logger = logging.getLogger(__name__)


async def _drain_task_despite_cancellation(task: asyncio.Future[Any]) -> Any:
    """Wait for transaction cleanup even if the caller is cancelled repeatedly."""

    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


def _agent_session_source_id_for_owner(client: str, owner_user_id: str) -> str:
    owner_fingerprint = hashlib.sha256(owner_user_id.encode("utf-8")).hexdigest()[:16]
    return f"src-agent-sessions-{slugify(client)[:32]}-{owner_fingerprint}"


def _legacy_agent_session_client(source_id: str) -> str:
    prefix = "src-agent-sessions-"
    if source_id == "src-agent-sessions":
        return ""
    if not source_id.startswith(prefix):
        return ""
    suffix = source_id[len(prefix) :]
    if suffix.startswith("claude-code"):
        return "claude-code"
    if suffix.startswith("codex"):
        return "codex"
    return suffix.rsplit("-", 1)[0]


# The three current outcomes an uploaded agent-session window can record.
# Knowledge completeness ("how much was kept vs dropped as no_output") is read
# from these. Older receipts used "package_created"; reads normalize that value
# into knowledge_patched.
AGENT_SESSION_OUTCOME_KNOWLEDGE_PATCHED = "knowledge_patched"
AGENT_SESSION_OUTCOME_LEGACY_PACKAGE_CREATED = "package_created"
AGENT_SESSION_OUTCOME_NO_OUTPUT = "no_output"
AGENT_SESSION_OUTCOME_FAILED = "failed"


def _with_relation_snapshot_audit(bundle: RelationOutcomeBundle) -> RelationOutcomeBundle:
    """Return a bundle whose relation-run audit contains the canonical snapshot hashes."""
    snapshot_audit = relation_bundle_snapshot_audit(candidates=bundle.candidates, relations=bundle.relations)
    audit = dict(bundle.relation_run.audit)
    for key, value in snapshot_audit.items():
        existing = audit.get(key)
        if existing is not None and existing != value:
            raise RuntimeError(
                "relation_run_id collision for "
                f"{bundle.relation_run.id}: supplied audit does not match relation snapshot ({key})"
            )
        audit[key] = value
    return replace(bundle, relation_run=replace(bundle.relation_run, audit=audit))


def _with_empty_relation_snapshot_audit(run: RelationRunRecord) -> RelationRunRecord:
    """Return a standalone relation run with the canonical empty snapshot audit."""
    audit = dict(run.audit)
    for key, value in relation_bundle_snapshot_audit(candidates=(), relations=()).items():
        existing = audit.get(key)
        if existing is not None and existing != value:
            raise RuntimeError(
                f"relation_run_id collision for {run.id}: supplied audit does not match empty relation snapshot ({key})"
            )
        audit[key] = value
    return replace(run, audit=audit)


AGENT_SESSION_OUTCOMES = (
    AGENT_SESSION_OUTCOME_KNOWLEDGE_PATCHED,
    AGENT_SESSION_OUTCOME_NO_OUTPUT,
    AGENT_SESSION_OUTCOME_FAILED,
)
_RELATION_SNAPSHOT_AUDIT_KEYS = frozenset(
    {
        "candidate_snapshot_hash",
        "relation_snapshot_hash",
    }
)


def _relation_result_memory_id(run: RelationRunRecord) -> str | None:
    value = run.result_memory_id
    return value if isinstance(value, str) and value else None


def _relation_run_value(row: Mapping[str, Any], key: str) -> Any:
    return row[key]


def _relation_run_user_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in audit.items() if key not in _RELATION_SNAPSHOT_AUDIT_KEYS}


def _assert_relation_run_retry_matches(row: Mapping[str, Any], run: RelationRunRecord) -> None:
    lifecycle_action = run.lifecycle_action.value if run.lifecycle_action is not None else None
    review_case = run.review_case.value if run.review_case is not None else None
    expected = {
        "evidence_unit_id": run.evidence_unit_id,
        "access_context_hash": run.access_context_hash,
        "candidate_count": run.candidate_count,
        "mandatory_candidate_count": run.mandatory_candidate_count,
        "checked_candidate_count": run.checked_candidate_count,
        "incomplete_mandatory_buckets_json": json.dumps(list(run.incomplete_mandatory_buckets), sort_keys=True),
        "classifier_version": run.classifier_version,
        "lifecycle_action": lifecycle_action,
        "review_case": review_case,
        "status": run.status,
        "result_memory_id": _relation_result_memory_id(run),
    }
    mismatches = [key for key, value in expected.items() if _relation_run_value(row, key) != value]
    if mismatches:
        raise RuntimeError(
            "relation_run_id collision for "
            f"{run.id}: existing run does not match retry payload ({', '.join(mismatches)})"
        )
    existing_audit = json.loads(row["audit_json"] or "{}")
    if _relation_run_user_audit(existing_audit) != _relation_run_user_audit(dict(run.audit)):
        raise RuntimeError(
            f"relation_run_id collision for {run.id}: existing run does not match retry payload (audit_json)"
        )
    for key in _RELATION_SNAPSHOT_AUDIT_KEYS:
        if key not in existing_audit:
            raise RuntimeError(f"relation_run_id collision for {run.id}: committed audit is missing {key}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _utc_iso(dt: datetime | None) -> str:
    value = dt or datetime.now(timezone.utc)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime values must include timezone information")
    return value.astimezone(timezone.utc).isoformat()


def _lifecycle_memory_version(row: Any | None) -> str:
    if row is None:
        return ""
    digest = hashlib.sha256(
        "\x1f".join(
            (
                str(row["status"] or ""),
                str(row["content_hash"] or ""),
                str(row["updated_at"] or ""),
            )
        ).encode("utf-8")
    ).hexdigest()[:16]
    return f"memory-version-{digest}"


def _non_empty_string(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _manifest_doc_id(metadata: Mapping[str, object]) -> str | None:
    entry = metadata.get("manifest_entry")
    nested = _non_empty_string(entry.get("doc_id")) if isinstance(entry, Mapping) else None
    return nested or _non_empty_string(metadata.get("doc_id"))


def _validate_replacement_kind(value: str) -> ReplacementKind:
    if value not in {"revision", "supersession"}:
        raise ValueError(f"Unsupported memory replacement kind: {value}")
    return value  # type: ignore[return-value]


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    return date.fromisoformat(str(s)[:10])


def _source_schedule_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "enabled": bool(row.get("sync_schedule_enabled")),
        "interval_minutes": int(
            row.get("sync_schedule_interval_minutes") or SOURCE_SYNC_SCHEDULE_DEFAULT_INTERVAL_MINUTES
        ),
        "next_run_at": row.get("sync_schedule_next_at"),
        "updated_at": row.get("sync_schedule_updated_at"),
    }


def _source_sync_run_from_row(row: Mapping[str, Any], *, coalesced: bool = False) -> SourceSyncRun:
    data = dict(row)
    return SourceSyncRun(
        run_id=str(data["run_id"]),
        workspace_id=str(data["workspace_id"]),
        source_id=str(data["source_id"]),
        trigger=str(data["trigger"]),
        status=str(data["status"]),
        force_full_sync=bool(data["force_full_sync"]),
        input_snapshot_id=data.get("input_snapshot_id"),
        rerun_input_snapshot_id=data.get("rerun_input_snapshot_id"),
        input_generation_watermark=data.get("input_generation_watermark"),
        rerun_input_generation_watermark=data.get("rerun_input_generation_watermark"),
        source_config_revision=data.get("source_config_revision"),
        rerun_source_config_revision=data.get("rerun_source_config_revision"),
        predecessor_activity_id=data.get("predecessor_activity_id"),
        rerun_predecessor_activity_id=data.get("rerun_predecessor_activity_id"),
        coalesced=coalesced,
        lease_owner=data.get("lease_owner"),
        lease_expires_at=_parse_dt(data.get("lease_expires_at")),
        lease_attempt_count=int(data.get("lease_attempt_count") or 0),
        recovery_count=int(data.get("recovery_count") or 0),
        rerun_requested=bool(data.get("rerun_requested")),
        next_attempt_at=_parse_dt(data.get("next_attempt_at")),
        error_message=data.get("error_message"),
        progress=json.loads(data["progress_json"]) if data.get("progress_json") else None,
        progress_revision=int(data.get("progress_revision") or 0),
        progress_updated_at=_parse_dt(data.get("progress_updated_at")),
        created_at=_parse_dt(data.get("created_at")),
        updated_at=_parse_dt(data.get("updated_at")),
        started_at=_parse_dt(data.get("started_at")),
        completed_at=_parse_dt(data.get("completed_at")),
    )


def _source_sync_input_from_row(row: Mapping[str, Any]) -> SourceSyncInput:
    data = dict(row)
    try:
        metadata = json.loads(data.get("metadata_json") or "{}")
    except json.JSONDecodeError:
        metadata = {}
    if not isinstance(metadata, dict):
        metadata = {}
    return SourceSyncInput(
        input_id=str(data["input_id"]),
        workspace_id=str(data["workspace_id"]),
        source_id=str(data["source_id"]),
        input_generation=int(data["input_generation"]),
        raw_uri=str(data["raw_uri"]),
        raw_sha256=str(data["raw_sha256"]),
        raw_content_type=str(data["raw_content_type"]),
        metadata=metadata,
        created_at=_parse_dt(data.get("created_at")),
    )


def _local_agent_job_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    data = dict(row)
    for field in ("payload_json", "result_json"):
        try:
            parsed = json.loads(data.get(field) or "{}")
        except json.JSONDecodeError:
            parsed = {}
        data[field.removesuffix("_json")] = parsed if isinstance(parsed, dict) else {}
    return data


_VALID_VISIBILITIES = frozenset({Visibility.WORKSPACE.value, Visibility.PRIVATE.value})


def _validate_visibility(visibility: str, owner_user_id: str | None) -> None:
    """Enforce the owner/visibility invariant before any memory write."""
    if visibility not in _VALID_VISIBILITIES:
        raise ValueError(f"visibility must be one of {sorted(_VALID_VISIBILITIES)}, got {visibility!r}")
    if (visibility == Visibility.PRIVATE.value) != (owner_user_id is not None):
        raise ValueError(
            "owner_user_id must be set iff visibility is private "
            f"(visibility={visibility!r}, owner_user_id={owner_user_id!r})"
        )


def _normalize_project_key(project_key: str | None) -> str:
    """Every persisted memory carries a non-NULL project_key; an unsupplied key
    lands in the UNSORTED backlog so the access predicate can use simple IN
    semantics without SQL three-valued NULL traps."""
    return project_key or UNSORTED_PROJECT_KEY


def _entity_from_row(d: dict) -> Entity:
    """Deserialize an entity row, handling both old (entity_type) and new (tags) columns."""
    tags_raw = d.get("tags", "[]")
    try:
        tags = json.loads(tags_raw) if tags_raw else []
    except (json.JSONDecodeError, TypeError):
        # Fallback: use entity_type as single-element list
        tags = [d.get("entity_type", "unknown")]
    return Entity(
        id=d["id"],
        canonical_name=d["canonical_name"],
        tags=tags if isinstance(tags, list) else [tags],
        display_name=d["display_name"],
        created_at=_parse_dt(d.get("created_at")),
    )


def _admin_fts_query(value: str) -> str:
    terms = value.strip().split()
    if not terms:
        return '""'
    quoted_terms = []
    for term in terms:
        escaped = term.replace('"', '""')
        quoted_terms.append(f'"{escaped}"')
    return " ".join(quoted_terms)


def _admin_like_pattern(value: str) -> str:
    escaped = value.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _enabled_source_visibility_condition(
    disabled_source_ids: list[str],
) -> tuple[str | None, list[str]]:
    if not disabled_source_ids:
        return None, []
    placeholders = ", ".join("?" for _ in disabled_source_ids)
    return (
        f"""(
            NOT EXISTS (
                SELECT 1
                FROM memory_sources ms_any
                WHERE ms_any.memory_id = m.id
            )
            OR EXISTS (
                SELECT 1
                FROM memory_sources ms_enabled
                WHERE ms_enabled.memory_id = m.id
                  AND (ms_enabled.source_id IS NULL OR ms_enabled.source_id NOT IN ({placeholders}))
            )
        )""",
        list(disabled_source_ids),
    )


# ---------------------------------------------------------------------------
# Schema (v1)
# ---------------------------------------------------------------------------

SCHEMA = """
-- ---------------------------------------------------------------
-- Documents
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS documents (
    doc_id              TEXT PRIMARY KEY,
    source              TEXT NOT NULL,
    source_url          TEXT NOT NULL,
    title               TEXT NOT NULL,
    space_or_project    TEXT NOT NULL,
    author              TEXT,
    last_modified       TEXT NOT NULL,
    labels              TEXT,                -- JSON array
    version             TEXT NOT NULL,
    content_hash        TEXT NOT NULL,
    token_count         INTEGER,
    raw_content_uri     TEXT,
    raw_content_type    TEXT,
    normalized_content_uri TEXT,
    pdf_content_uri     TEXT,
    last_synced         TEXT NOT NULL,
    created_at          TEXT DEFAULT (datetime('now')),
    updated_at          TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS document_metadata (
    doc_id          TEXT PRIMARY KEY REFERENCES documents(doc_id),
    summary         TEXT NOT NULL,
    tags            TEXT NOT NULL,           -- JSON array
    entities        TEXT NOT NULL,           -- JSON array of {name, type}
    doc_type        TEXT NOT NULL,
    complexity      TEXT NOT NULL,
    enriched_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_relationships (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_doc_id   TEXT REFERENCES documents(doc_id),
    target_doc_id   TEXT,
    target_title    TEXT NOT NULL,
    relation_type   TEXT NOT NULL,
    confidence      REAL NOT NULL,
    link_source     TEXT NOT NULL DEFAULT 'enrichment'
);

CREATE TABLE IF NOT EXISTS changelog (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id              TEXT REFERENCES documents(doc_id),
    change_type         TEXT NOT NULL,
    previous_version    TEXT,
    current_version     TEXT,
    content_diff        TEXT,
    ai_change_summary   TEXT,
    detected_at         TEXT NOT NULL,
    title               TEXT,
    source              TEXT
);

-- ---------------------------------------------------------------
-- Entities
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name  TEXT NOT NULL UNIQUE,
    entity_type     TEXT DEFAULT 'unknown',    -- DEPRECATED: kept for migration compat
    tags            TEXT NOT NULL DEFAULT '[]', -- JSON array of soft tags
    display_name    TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS entity_aliases (
    alias            TEXT NOT NULL,
    alias_normalized TEXT NOT NULL,
    canonical_id     INTEGER NOT NULL REFERENCES entities(id),
    source           TEXT NOT NULL,
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (alias_normalized, canonical_id)
);

CREATE VIRTUAL TABLE IF NOT EXISTS entity_alias_search_fts USING fts5(
    entity_id UNINDEXED,
    canonical_name UNINDEXED,
    alias_normalized UNINDEXED,
    search_text,
    tokenize='porter unicode61'
);

-- ---------------------------------------------------------------
-- Memories
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memories (
    id                  TEXT PRIMARY KEY,
    memory_type         TEXT NOT NULL,
    content             TEXT NOT NULL,
    content_hash        TEXT NOT NULL,
    tags                TEXT NOT NULL DEFAULT '[]',
    visibility          TEXT NOT NULL DEFAULT 'workspace',
    owner_user_id       TEXT,
    project_key         TEXT,
    repo_identifier     TEXT,
    confidence          REAL NOT NULL DEFAULT 0.7,
    corroboration_count INTEGER NOT NULL DEFAULT 1,
    contradiction_count INTEGER NOT NULL DEFAULT 0,
    valid_from          TEXT,
    valid_until         TEXT,
    superseded_by       TEXT REFERENCES memories(id),
    status              TEXT NOT NULL DEFAULT 'active',
    retirement_reason   TEXT,
    retired_at          TEXT,
    superseded_at       TEXT,
    replacement_reason  TEXT,
    replacement_kind    TEXT,
    extraction_context  TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK (visibility IN ('private','workspace')),
    CHECK ((visibility = 'private') = (owner_user_id IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS memory_sources (
    memory_id   TEXT NOT NULL REFERENCES memories(id),
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
    source_id   TEXT,
    source_type TEXT NOT NULL,
    excerpt     TEXT,
    support_kind TEXT NOT NULL DEFAULT 'extracted',
    added_at    TEXT NOT NULL DEFAULT (datetime('now')),
    source_updated_at TEXT,
    PRIMARY KEY (memory_id, doc_id)
);

CREATE TABLE IF NOT EXISTS source_artifact_cleanup_tasks (
    task_id        TEXT PRIMARY KEY,
    source_id      TEXT NOT NULL,
    artifact_uri   TEXT NOT NULL,
    attempt_count  INTEGER NOT NULL DEFAULT 0,
    last_error     TEXT,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE (source_id, artifact_uri)
);

CREATE TABLE IF NOT EXISTS evidence_units (
    id                   TEXT PRIMARY KEY,
    source_id            TEXT NOT NULL,
    doc_id               TEXT,
    doc_revision_id      TEXT,
    source_type          TEXT NOT NULL,
    client               TEXT,
    repo_identifier      TEXT,
    source_anchor        TEXT,
    source_lineage_id    TEXT,
    source_metadata_json TEXT NOT NULL DEFAULT '{}',
    project_key          TEXT,
    visibility           TEXT NOT NULL DEFAULT 'workspace',
    owner_user_id        TEXT,
    observed_at          TEXT,
    extractor_run_id     TEXT,
    access_context_hash  TEXT,
    content              TEXT NOT NULL,
    excerpt              TEXT,
    evidence_provenance  TEXT NOT NULL,
    created_at           TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at           TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK (visibility IN ('private','workspace')),
    CHECK ((visibility = 'private') = (owner_user_id IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS evidence_relations (
    evidence_unit_id        TEXT NOT NULL REFERENCES evidence_units(id),
    memory_id               TEXT NOT NULL REFERENCES memories(id),
    relation_type           TEXT NOT NULL,
    relation_direction      TEXT,
    authority_case          TEXT NOT NULL,
    is_authoritative_support INTEGER NOT NULL DEFAULT 0,
    source_lineage_id       TEXT,
    confidence              REAL,
    reason                  TEXT,
    proposed_memory_content TEXT,
    excerpt                 TEXT,
    classifier_version      TEXT NOT NULL,
    relation_run_id         TEXT NOT NULL,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (evidence_unit_id, memory_id),
    CHECK (relation_type IN ('supports','equivalent','refines','contradicts')),
    CHECK (relation_direction IS NULL OR relation_direction IN ('symmetric','challenger_to_candidate','candidate_to_challenger'))
);

CREATE TABLE IF NOT EXISTS relation_runs (
    id                                TEXT PRIMARY KEY,
    evidence_unit_id                  TEXT NOT NULL REFERENCES evidence_units(id),
    access_context_hash               TEXT,
    candidate_count                   INTEGER NOT NULL DEFAULT 0,
    mandatory_candidate_count         INTEGER NOT NULL DEFAULT 0,
    checked_candidate_count           INTEGER NOT NULL DEFAULT 0,
    incomplete_mandatory_buckets_json TEXT NOT NULL DEFAULT '[]',
    classifier_version                TEXT,
    lifecycle_action                  TEXT,
    review_case                       TEXT,
    status                            TEXT NOT NULL,
    result_memory_id                  TEXT,
    audit_json                        TEXT NOT NULL DEFAULT '{}',
    started_at                        TEXT NOT NULL DEFAULT (datetime('now')),
    completed_at                      TEXT
);

CREATE TABLE IF NOT EXISTS relation_run_relations (
    relation_run_id         TEXT NOT NULL REFERENCES relation_runs(id),
    evidence_unit_id        TEXT NOT NULL REFERENCES evidence_units(id),
    memory_id               TEXT NOT NULL REFERENCES memories(id),
    relation_type           TEXT NOT NULL,
    relation_direction      TEXT,
    authority_case          TEXT NOT NULL,
    is_authoritative_support INTEGER NOT NULL DEFAULT 0,
    source_lineage_id       TEXT,
    confidence              REAL,
    reason                  TEXT,
    proposed_memory_content TEXT,
    excerpt                 TEXT,
    classifier_version      TEXT NOT NULL,
    created_at              TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (relation_run_id, evidence_unit_id, memory_id),
    CHECK (relation_type IN ('supports','equivalent','refines','contradicts')),
    CHECK (relation_direction IS NULL OR relation_direction IN ('symmetric','challenger_to_candidate','candidate_to_challenger'))
);

CREATE TABLE IF NOT EXISTS relation_candidates (
    relation_run_id TEXT NOT NULL REFERENCES relation_runs(id),
    evidence_unit_id TEXT NOT NULL REFERENCES evidence_units(id),
    memory_id       TEXT NOT NULL REFERENCES memories(id),
    bucket          TEXT NOT NULL,
    bucket_rank     INTEGER NOT NULL,
    candidate_rank  INTEGER NOT NULL,
    score           REAL,
    is_mandatory    INTEGER NOT NULL DEFAULT 0,
    bucket_complete INTEGER NOT NULL DEFAULT 0,
    was_checked     INTEGER NOT NULL DEFAULT 0,
    reason          TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (relation_run_id, bucket, memory_id)
);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id   TEXT NOT NULL REFERENCES memories(id),
    entity_id   INTEGER NOT NULL REFERENCES entities(id),
    PRIMARY KEY (memory_id, entity_id)
);

-- BM25 full-text search
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    memory_id UNINDEXED,
    content,
    entities_text,
    tags_text,
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_search_metadata_fts USING fts5(
    memory_id UNINDEXED,
    source_id UNINDEXED,
    doc_id UNINDEXED,
    source_type UNINDEXED,
    metadata_title_tokens,
    metadata_external_id_tokens,
    metadata_path_tokens,
    metadata_source_name_tokens,
    metadata_label_context_tokens,
    tokenize='porter unicode61'
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_search_metadata_alias_fts USING fts5(
    memory_id UNINDEXED,
    source_id UNINDEXED,
    doc_id UNINDEXED,
    source_type UNINDEXED,
    metadata_alias_tokens,
    tokenize='porter unicode61'
);

CREATE TABLE IF NOT EXISTS memory_search_metadata_trigram (
    memory_id        TEXT NOT NULL,
    source_id        TEXT,
    doc_id           TEXT NOT NULL,
    source_type      TEXT NOT NULL,
    metadata_compact TEXT NOT NULL,
    PRIMARY KEY (memory_id, doc_id)
);

-- ---------------------------------------------------------------
-- Sources & Sync
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sources (
    id              TEXT PRIMARY KEY,
    type            TEXT NOT NULL,
    name            TEXT NOT NULL,
    config          TEXT NOT NULL,           -- JSON
    status          TEXT NOT NULL DEFAULT 'active',
    last_sync       TEXT,
    doc_count       INTEGER DEFAULT 0,
    project_binding TEXT,                    -- JSON: {"mode": "fixed", ...} or {"mode": "by_field", ...}
    created_by_user_id TEXT,
    owner_user_id   TEXT NOT NULL,
    access_policy   TEXT NOT NULL CHECK (access_policy IN ('private', 'workspace')),
    access_state    TEXT NOT NULL DEFAULT 'active'
                    CHECK (access_state IN ('active', 'changing', 'orphaned_private')),
    execution_owner_user_id TEXT,
    sync_schedule_enabled INTEGER NOT NULL DEFAULT 0,
    sync_schedule_interval_minutes INTEGER NOT NULL DEFAULT 1440,
    sync_schedule_next_at TEXT,
    sync_schedule_updated_at TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS source_access_transitions (
    operation_id          TEXT PRIMARY KEY,
    source_id             TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    idempotency_key       TEXT NOT NULL,
    actor_user_id         TEXT NOT NULL,
    previous_policy       TEXT NOT NULL CHECK (previous_policy IN ('private', 'workspace')),
    target_policy         TEXT NOT NULL CHECK (target_policy IN ('private', 'workspace')),
    previous_source_status TEXT NOT NULL,
    status                TEXT NOT NULL
                          CHECK (status IN ('queued', 'running', 'failed', 'completed', 'reverted')),
    total_memories        INTEGER NOT NULL DEFAULT 0,
    processed_memories    INTEGER NOT NULL DEFAULT 0,
    error_code            TEXT,
    error_message         TEXT,
    created_at            TEXT NOT NULL,
    updated_at            TEXT NOT NULL,
    completed_at          TEXT,
    UNIQUE (source_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_source_access_transitions_status
    ON source_access_transitions(status, updated_at);

CREATE TABLE IF NOT EXISTS source_access_transition_memory_map (
    operation_id       TEXT NOT NULL REFERENCES source_access_transitions(operation_id) ON DELETE CASCADE,
    source_id          TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    original_memory_id TEXT NOT NULL,
    target_memory_id   TEXT NOT NULL,
    PRIMARY KEY (operation_id, original_memory_id),
    UNIQUE (operation_id, target_memory_id)
);

CREATE TABLE IF NOT EXISTS source_projection_runs (
    id              TEXT PRIMARY KEY,
    source_id       TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    source_type     TEXT NOT NULL,
    coverage        TEXT NOT NULL,
    scope_json      TEXT NOT NULL,
    checkpoint_json TEXT NOT NULL,
    payload_json    TEXT NOT NULL,
    payload_hash    TEXT NOT NULL,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_projection_runs_source
    ON source_projection_runs(source_id, created_at);

CREATE TABLE IF NOT EXISTS projection_scope_transitions (
    id                  TEXT PRIMARY KEY,
    source_id           TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    previous_scope_json TEXT NOT NULL,
    target_scope_json   TEXT NOT NULL,
    status              TEXT NOT NULL CHECK (status IN ('pending', 'running', 'applied', 'failed')),
    run_id              TEXT,
    coverage            TEXT,
    error               TEXT,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    completed_at        TEXT
);
CREATE INDEX IF NOT EXISTS idx_projection_scope_transitions_source
    ON projection_scope_transitions(source_id, created_at);

CREATE TABLE IF NOT EXISTS source_units (
    id                  TEXT PRIMARY KEY,
    source_id           TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    unit_type           TEXT NOT NULL,
    provider_key        TEXT NOT NULL,
    locator_json        TEXT NOT NULL DEFAULT '{}',
    current_revision_id TEXT,
    updated_at          TEXT NOT NULL,
    UNIQUE (source_id, provider_key)
);

CREATE TABLE IF NOT EXISTS source_unit_document_lineage_history (
    source_id      TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    document_id    TEXT NOT NULL,
    source_unit_id TEXT NOT NULL REFERENCES source_units(id) ON DELETE CASCADE,
    is_current     INTEGER NOT NULL DEFAULT 1,
    first_seen_at  TEXT NOT NULL,
    last_seen_at   TEXT NOT NULL,
    PRIMARY KEY (source_id, document_id, source_unit_id)
);
CREATE INDEX IF NOT EXISTS idx_source_unit_document_history_unit
    ON source_unit_document_lineage_history(source_unit_id, is_current, last_seen_at);

CREATE TABLE IF NOT EXISTS source_observations (
    id                  TEXT PRIMARY KEY,
    source_id           TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    source_unit_id      TEXT NOT NULL REFERENCES source_units(id) ON DELETE CASCADE,
    observation_type    TEXT NOT NULL,
    provider_key        TEXT NOT NULL,
    locator_json        TEXT NOT NULL DEFAULT '{}',
    current_revision_id TEXT,
    updated_at          TEXT NOT NULL,
    UNIQUE (source_unit_id, provider_key)
);

CREATE TABLE IF NOT EXISTS source_observation_revisions (
    id              TEXT PRIMARY KEY,
    observation_id  TEXT NOT NULL REFERENCES source_observations(id) ON DELETE CASCADE,
    semantic_hash   TEXT NOT NULL,
    content         TEXT NOT NULL,
    metadata_json   TEXT NOT NULL DEFAULT '{}',
    observed_at     TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_unit_revisions (
    id                            TEXT PRIMARY KEY,
    source_unit_id                TEXT NOT NULL REFERENCES source_units(id) ON DELETE CASCADE,
    semantic_hash                 TEXT NOT NULL,
    location_hash                 TEXT,
    membership_hash               TEXT,
    access_hash                   TEXT,
    observation_revision_ids_json TEXT NOT NULL,
    observed_at                   TEXT,
    created_at                    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_projection_relations (
    projection_run_id    TEXT NOT NULL REFERENCES source_projection_runs(id) ON DELETE CASCADE,
    relation_index       INTEGER NOT NULL,
    relation_type        TEXT NOT NULL,
    from_id              TEXT NOT NULL,
    to_id                TEXT NOT NULL,
    provider_relation_id TEXT,
    metadata_json        TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (projection_run_id, relation_index)
);

CREATE TABLE IF NOT EXISTS source_revision_deltas (
    projection_run_id TEXT NOT NULL REFERENCES source_projection_runs(id) ON DELETE CASCADE,
    delta_index       INTEGER NOT NULL,
    source_unit_id    TEXT NOT NULL,
    payload_json      TEXT NOT NULL,
    PRIMARY KEY (projection_run_id, delta_index)
);

CREATE TABLE IF NOT EXISTS source_lifecycle_gates (
    source_id   TEXT PRIMARY KEY REFERENCES sources(id) ON DELETE CASCADE,
    state       TEXT NOT NULL CHECK (state IN ('gated', 'enabled')),
    reason      TEXT,
    audited_at  TEXT,
    enabled_at  TEXT,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lifecycle_cutover_findings (
    id                        TEXT PRIMARY KEY,
    source_id                 TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    memory_id                 TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    reason                    TEXT NOT NULL,
    status                    TEXT NOT NULL CHECK (status IN ('open', 'resolved')),
    available_provenance_json TEXT NOT NULL DEFAULT '{}',
    mapping_attempt_json      TEXT NOT NULL DEFAULT '{}',
    observation_id            TEXT,
    source_unit_id            TEXT,
    created_at                TEXT NOT NULL,
    updated_at                TEXT NOT NULL,
    resolved_at               TEXT
);
CREATE INDEX IF NOT EXISTS idx_cutover_findings_source_status
    ON lifecycle_cutover_findings(source_id, status);

CREATE TABLE IF NOT EXISTS lifecycle_backfill_jobs (
    id               TEXT PRIMARY KEY,
    source_id        TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    status           TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    scanned_memories INTEGER NOT NULL DEFAULT 0,
    mapped_memories  INTEGER NOT NULL DEFAULT 0,
    finding_count    INTEGER NOT NULL DEFAULT 0,
    error            TEXT,
    created_at       TEXT NOT NULL,
    started_at       TEXT,
    completed_at     TEXT,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_backfill_jobs_source
    ON lifecycle_backfill_jobs(source_id, created_at);

CREATE TABLE IF NOT EXISTS evidence_references (
    id                          TEXT PRIMARY KEY,
    evidence_unit_id            TEXT NOT NULL REFERENCES evidence_units(id) ON DELETE CASCADE,
    role                        TEXT NOT NULL CHECK (role IN ('primary', 'required', 'context')),
    anchor_kind                 TEXT NOT NULL,
    observation_id              TEXT NOT NULL REFERENCES source_observations(id) ON DELETE CASCADE,
    observation_revision_id     TEXT NOT NULL REFERENCES source_observation_revisions(id),
    fragment_id                 TEXT,
    range_start                 INTEGER,
    range_end                   INTEGER,
    created_at                  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_support_assertions (
    id                    TEXT PRIMARY KEY,
    memory_id             TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    evidence_reference_id TEXT NOT NULL REFERENCES evidence_references(id) ON DELETE CASCADE,
    source_id             TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    access_context_hash   TEXT NOT NULL,
    active                INTEGER NOT NULL DEFAULT 1,
    created_at            TEXT NOT NULL,
    removed_at            TEXT,
    UNIQUE (memory_id, evidence_reference_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_support_assertions_active
    ON memory_support_assertions(memory_id, active);

CREATE TABLE IF NOT EXISTS lifecycle_plans (
    id                TEXT PRIMARY KEY,
    reconciliation_scope_id TEXT NOT NULL,
    source_id         TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    source_unit_id    TEXT NOT NULL,
    target_unit_revision_id TEXT,
    status            TEXT NOT NULL CHECK (status IN ('staged', 'applied', 'rejected', 'stale')),
    payload_json      TEXT NOT NULL,
    payload_hash      TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    applied_at        TEXT,
    error             TEXT
);

CREATE TABLE IF NOT EXISTS lifecycle_reviews (
    id                  TEXT PRIMARY KEY,
    lifecycle_plan_id   TEXT NOT NULL REFERENCES lifecycle_plans(id) ON DELETE CASCADE,
    incumbent_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    status              TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected', 'stale')),
    staged_evidence_json TEXT NOT NULL DEFAULT '{}',
    reason              TEXT,
    created_at          TEXT NOT NULL,
    resolved_at         TEXT
);

CREATE TABLE IF NOT EXISTS lifecycle_vector_outbox (
    id                TEXT PRIMARY KEY,
    lifecycle_plan_id TEXT NOT NULL REFERENCES lifecycle_plans(id) ON DELETE CASCADE,
    memory_id         TEXT NOT NULL,
    operation         TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
    status            TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'failed')),
    attempts          INTEGER NOT NULL DEFAULT 0,
    error             TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL,
    UNIQUE (lifecycle_plan_id, memory_id, operation)
);
CREATE INDEX IF NOT EXISTS idx_lifecycle_vector_outbox_status
    ON lifecycle_vector_outbox(status, created_at);

CREATE TABLE IF NOT EXISTS relation_discovery_work (
    id                       TEXT PRIMARY KEY,
    lifecycle_plan_id        TEXT NOT NULL REFERENCES lifecycle_plans(id) ON DELETE CASCADE,
    memory_id                TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    expected_content_hash    TEXT NOT NULL,
    source_id                TEXT NOT NULL,
    source_unit_id           TEXT NOT NULL,
    source_unit_revision_id  TEXT,
    doc_id                   TEXT NOT NULL,
    actor_user_id            TEXT,
    entity_ids_json          TEXT NOT NULL DEFAULT '[]',
    status                   TEXT NOT NULL CHECK (status IN ('pending','running','completed','failed','obsolete')),
    attempts                 INTEGER NOT NULL DEFAULT 0,
    lease_owner              TEXT,
    lease_token              TEXT,
    lease_until              TEXT,
    next_attempt_at          TEXT,
    error                    TEXT,
    created_at               TEXT NOT NULL,
    updated_at               TEXT NOT NULL,
    completed_at             TEXT,
    UNIQUE (lifecycle_plan_id, memory_id, expected_content_hash)
);
CREATE INDEX IF NOT EXISTS idx_relation_discovery_work_status
    ON relation_discovery_work(status, lease_until, created_at);

CREATE TABLE IF NOT EXISTS source_deletion_vector_outbox (
    id          TEXT PRIMARY KEY,
    source_id   TEXT NOT NULL,
    memory_id   TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'failed')),
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_source_deletion_vector_outbox_status
    ON source_deletion_vector_outbox(status, created_at);

CREATE TABLE IF NOT EXISTS source_subscriptions (
    source_id   TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    user_id     TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (source_id, user_id)
);

CREATE TABLE IF NOT EXISTS source_list_pins (
    source_id   TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    user_id     TEXT NOT NULL,
    pinned_at   TEXT NOT NULL,
    PRIMARY KEY (source_id, user_id)
);

CREATE INDEX IF NOT EXISTS idx_source_list_pins_user
    ON source_list_pins(user_id, pinned_at);

CREATE TABLE IF NOT EXISTS source_list_preferences (
    user_id     TEXT PRIMARY KEY,
    sort_mode   TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sources_sync_schedule_due
    ON sources(sync_schedule_enabled, sync_schedule_next_at);

CREATE TABLE IF NOT EXISTS auth_sessions (
    provider            TEXT NOT NULL,
    origin              TEXT NOT NULL,
    secret_encrypted    TEXT NOT NULL,
    principal_id        TEXT,
    principal_name      TEXT,
    principal_email     TEXT,
    browser             TEXT,
    status              TEXT NOT NULL,
    captured_at         TEXT NOT NULL,
    validated_at        TEXT,
    last_error          TEXT,
    updated_at          TEXT NOT NULL,
    PRIMARY KEY (provider, origin)
);

CREATE TABLE IF NOT EXISTS agent_session_receipts (
    doc_id                  TEXT PRIMARY KEY,
    source_id               TEXT NOT NULL,
    client                  TEXT NOT NULL,
    session_id              TEXT NOT NULL,
    trigger                 TEXT NOT NULL,
    workspace               TEXT NOT NULL,
    repo                    TEXT,
    branch                  TEXT,
    commit_sha              TEXT,
    history_window_kind     TEXT NOT NULL,
    history_window_start    TEXT,
    history_window_end      TEXT,
    submitted_at            TEXT NOT NULL,
    document_hash           TEXT NOT NULL,
    source_kind             TEXT NOT NULL,
    document_uri            TEXT NOT NULL,
    metadata                TEXT NOT NULL DEFAULT '{}',
    updated_at              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_hook_receipts (
    receipt_id      TEXT PRIMARY KEY,
    client          TEXT NOT NULL,
    session_id      TEXT NOT NULL,
    hook            TEXT NOT NULL,
    workspace       TEXT NOT NULL,
    repo            TEXT,
    branch          TEXT,
    commit_sha      TEXT,
    submitted_at    TEXT NOT NULL,
    metadata        TEXT NOT NULL DEFAULT '{}',
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_concepts (
    id                  TEXT PRIMARY KEY,
    source_id           TEXT NOT NULL,
    owner_user_id       TEXT NOT NULL,
    visibility          TEXT NOT NULL DEFAULT 'private',
    workspace           TEXT NOT NULL,
    repo_identifier     TEXT,
    concept_type        TEXT NOT NULL,
    concept_path        TEXT NOT NULL,
    title               TEXT NOT NULL,
    markdown_body       TEXT NOT NULL,
    frontmatter_json    TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
    last_observed_at    TEXT NOT NULL,
    CHECK (visibility IN ('private', 'workspace'))
);

CREATE TABLE IF NOT EXISTS agent_claims (
    id                  TEXT PRIMARY KEY,
    concept_id          TEXT NOT NULL REFERENCES agent_concepts(id),
    display_anchor      TEXT NOT NULL,
    claim_text          TEXT NOT NULL,
    memory_type         TEXT NOT NULL,
    tags                TEXT NOT NULL DEFAULT '[]',
    confidence          REAL NOT NULL DEFAULT 0.7,
    memory_id           TEXT NOT NULL REFERENCES memories(id),
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
    last_observed_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_claim_citations (
    claim_id        TEXT NOT NULL REFERENCES agent_claims(id),
    citation_url    TEXT NOT NULL,
    observed_at     TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (claim_id, citation_url)
);

CREATE TABLE IF NOT EXISTS sync_state (
    source              TEXT PRIMARY KEY,
    last_sync_at        TEXT,
    last_sync_status    TEXT,
    docs_processed      INTEGER,
    docs_updated        INTEGER,
    error_message       TEXT
);

CREATE TABLE IF NOT EXISTS sync_history (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source              TEXT NOT NULL,
    status              TEXT NOT NULL,
    docs_processed      INTEGER NOT NULL DEFAULT 0,
    docs_updated        INTEGER NOT NULL DEFAULT 0,
    docs_failed         INTEGER NOT NULL DEFAULT 0,
    memories_extracted  INTEGER NOT NULL DEFAULT 0,
    error_message       TEXT,
    failed_docs         TEXT,                -- JSON array
    started_at          TEXT NOT NULL,
    finished_at         TEXT NOT NULL,
    run_id              TEXT
);

CREATE TABLE IF NOT EXISTS source_sync_runs (
    run_id                  TEXT PRIMARY KEY,
    workspace_id            TEXT NOT NULL,
    source_id               TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    trigger                 TEXT NOT NULL,
    status                  TEXT NOT NULL,
    force_full_sync         INTEGER NOT NULL DEFAULT 0,
    input_snapshot_id       TEXT,
    rerun_input_snapshot_id TEXT,
    input_generation_watermark INTEGER,
    rerun_input_generation_watermark INTEGER,
    source_config_revision TEXT,
    rerun_source_config_revision TEXT,
    predecessor_activity_id TEXT,
    rerun_predecessor_activity_id TEXT,
    lease_owner             TEXT,
    lease_expires_at        TEXT,
    lease_attempt_count     INTEGER NOT NULL DEFAULT 0,
    recovery_count          INTEGER NOT NULL DEFAULT 0,
    rerun_requested         INTEGER NOT NULL DEFAULT 0,
    next_attempt_at         TEXT,
    error_message           TEXT,
    progress_json           TEXT,
    progress_revision       INTEGER NOT NULL DEFAULT 0,
    progress_updated_at     TEXT,
    created_at              TEXT NOT NULL,
    updated_at              TEXT NOT NULL,
    started_at              TEXT,
    completed_at            TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_source_sync_runs_active
    ON source_sync_runs(workspace_id, source_id)
    WHERE status IN ('pending', 'running');

CREATE TABLE IF NOT EXISTS source_sync_inputs (
    input_id            TEXT PRIMARY KEY,
    workspace_id        TEXT NOT NULL,
    source_id           TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    input_generation    INTEGER NOT NULL,
    raw_uri             TEXT NOT NULL,
    raw_sha256          TEXT NOT NULL,
    raw_content_type    TEXT NOT NULL,
    metadata_json       TEXT NOT NULL DEFAULT '{}',
    created_at          TEXT NOT NULL,
    UNIQUE(workspace_id, source_id, input_generation)
);

CREATE INDEX IF NOT EXISTS idx_source_sync_inputs_source
    ON source_sync_inputs(workspace_id, source_id, input_generation);

CREATE UNIQUE INDEX IF NOT EXISTS idx_source_sync_inputs_raw_hash
    ON source_sync_inputs(workspace_id, source_id, raw_sha256);

CREATE TABLE IF NOT EXISTS source_sync_snapshot_items (
    workspace_id        TEXT NOT NULL,
    source_id           TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    snapshot_id         TEXT NOT NULL,
    doc_id              TEXT NOT NULL,
    input_id            TEXT NOT NULL REFERENCES source_sync_inputs(input_id) ON DELETE CASCADE,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (workspace_id, source_id, snapshot_id, doc_id)
);

CREATE INDEX IF NOT EXISTS idx_source_sync_snapshot_items_input
    ON source_sync_snapshot_items(input_id);

CREATE TABLE IF NOT EXISTS local_agent_jobs (
    job_id TEXT PRIMARY KEY,
    workspace_id TEXT NOT NULL DEFAULT 'default',
    source_id TEXT NOT NULL DEFAULT '',
    source_type TEXT NOT NULL,
    operation TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    payload_json TEXT NOT NULL DEFAULT '{}',
    created_by_user_id TEXT NOT NULL,
    execution_owner_user_id TEXT NOT NULL,
    lease_owner_user_id TEXT,
    leased_until TEXT,
    attempt_count INTEGER NOT NULL DEFAULT 0,
    result_json TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_local_agent_jobs_owner_status
    ON local_agent_jobs(execution_owner_user_id, status, leased_until);
CREATE TABLE IF NOT EXISTS local_agent_heartbeats (
    user_id TEXT PRIMARY KEY,
    last_seen_at TEXT NOT NULL
);

-- ---------------------------------------------------------------
-- Config singletons
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schedule_config (
    id          INTEGER PRIMARY KEY CHECK (id = 1),
    enabled     INTEGER NOT NULL DEFAULT 0,
    frequency   TEXT NOT NULL DEFAULT 'daily',
    time        TEXT NOT NULL DEFAULT '02:00',
    day_of_week INTEGER NOT NULL DEFAULT 0,
    timezone    TEXT NOT NULL DEFAULT 'UTC'
);

CREATE TABLE IF NOT EXISTS llm_config (
    id                  INTEGER PRIMARY KEY CHECK (id = 1),
    enrichment_model    TEXT,
    enrichment_base_url TEXT,
    enrichment_api_key  TEXT,
    embedding_model     TEXT,
    embedding_base_url  TEXT,
    embedding_api_key   TEXT
);

-- ---------------------------------------------------------------
-- Users
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS users (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT UNIQUE NOT NULL,
    display_name    TEXT,
    password_hash   TEXT NOT NULL,
    role            TEXT NOT NULL DEFAULT 'viewer',
    last_login      TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------
-- Projects: per-row metadata for the relevance bucket on each memory.
-- SHARED is the team-wide bucket (never down-weighted, always satisfies
-- the access predicate). UNSORTED is the unmapped backlog (open and
-- visible, but down-weighted like any cross-project hit until an admin
-- binds the field value).
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS projects (
    id            TEXT PRIMARY KEY,
    key           TEXT NOT NULL UNIQUE,
    name          TEXT NOT NULL,
    is_shared     INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ---------------------------------------------------------------
-- Schema migrations tracking
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_migrations (
    version     INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    applied_at  TEXT NOT NULL
);

-- ---------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);
CREATE INDEX IF NOT EXISTS idx_documents_space ON documents(space_or_project);
CREATE INDEX IF NOT EXISTS idx_changelog_doc ON changelog(doc_id);
CREATE INDEX IF NOT EXISTS idx_changelog_detected ON changelog(detected_at);
CREATE INDEX IF NOT EXISTS idx_relationships_source ON document_relationships(source_doc_id);
CREATE INDEX IF NOT EXISTS idx_relationships_target ON document_relationships(target_doc_id);
CREATE INDEX IF NOT EXISTS idx_sync_history_finished ON sync_history(finished_at);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(canonical_name);
CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(memory_type);
CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status);
CREATE INDEX IF NOT EXISTS idx_memories_project ON memories(project_key);
-- Indexes for columns added after the initial schema, including visibility,
-- are created by their migrations. SCHEMA runs before
-- migrations, so upgrading databases may not have those columns here yet.
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memory_sources_doc ON memory_sources(doc_id);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_normalized ON entity_aliases(alias_normalized);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_compact ON entity_aliases(REPLACE(alias_normalized, ' ', ''));
CREATE INDEX IF NOT EXISTS idx_entities_canonical_compact ON entities(REPLACE(canonical_name, ' ', ''));
CREATE INDEX IF NOT EXISTS idx_auth_sessions_status ON auth_sessions(status);
CREATE INDEX IF NOT EXISTS idx_source_subscriptions_user ON source_subscriptions(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_session_receipts_session ON agent_session_receipts(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_session_receipts_source ON agent_session_receipts(source_id);
CREATE INDEX IF NOT EXISTS idx_agent_hook_receipts_session ON agent_hook_receipts(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_hook_receipts_hook ON agent_hook_receipts(hook);
CREATE INDEX IF NOT EXISTS idx_agent_concepts_owner_repo ON agent_concepts(owner_user_id, repo_identifier);
CREATE INDEX IF NOT EXISTS idx_agent_claims_concept ON agent_claims(concept_id);
CREATE INDEX IF NOT EXISTS idx_agent_claims_memory ON agent_claims(memory_id);
CREATE INDEX IF NOT EXISTS idx_relation_runs_result_memory ON relation_runs(result_memory_id);
-- Cross-document contradiction tracking
CREATE TABLE IF NOT EXISTS memory_contradictions (
    memory_id_a    TEXT NOT NULL REFERENCES memories(id),
    memory_id_b    TEXT NOT NULL REFERENCES memories(id),
    classification TEXT NOT NULL,
    resolution     TEXT DEFAULT 'pending',
    detected_at    TEXT NOT NULL DEFAULT (datetime('now')),
    resolved_at    TEXT,
    reason         TEXT,
    PRIMARY KEY (memory_id_a, memory_id_b)
);

-- ---------------------------------------------------------------
-- Memory reviews - human-gated lifecycle decisions
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_reviews (
    id                              TEXT PRIMARY KEY,
    kind                            TEXT NOT NULL,
    status                          TEXT NOT NULL,
    incumbent_memory_id             TEXT NOT NULL REFERENCES memories(id),
    challenger_memory_id            TEXT NOT NULL REFERENCES memories(id),
    reason                          TEXT,
    review_note                     TEXT,
    reviewer                        TEXT,
    expected_incumbent_updated_at   TEXT,
    expected_challenger_updated_at  TEXT,
    replacement_kind                TEXT NOT NULL DEFAULT 'supersession',
    created_at                      TEXT NOT NULL,
    resolved_at                     TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_reviews_status ON memory_reviews(status);
CREATE INDEX IF NOT EXISTS idx_memory_reviews_incumbent ON memory_reviews(incumbent_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_reviews_challenger ON memory_reviews(challenger_memory_id);

CREATE TABLE IF NOT EXISTS memory_review_related_challengers (
    review_id              TEXT NOT NULL REFERENCES memory_reviews(id),
    challenger_memory_id   TEXT NOT NULL REFERENCES memories(id),
    reason                 TEXT,
    created_at             TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (review_id, challenger_memory_id),
    UNIQUE (challenger_memory_id)
);

CREATE INDEX IF NOT EXISTS idx_memory_review_related_review
    ON memory_review_related_challengers(review_id);

-- ---------------------------------------------------------------
-- Memory audit ledger - append-only evaluation events
-- ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS memory_audit_events (
    event_id          TEXT PRIMARY KEY,
    operation_id      TEXT NOT NULL,
    parent_event_id   TEXT,
    occurred_at       TEXT NOT NULL,
    actor_type        TEXT,
    actor_id          TEXT,
    run_id            TEXT,
    trace_id          TEXT,
    source_id         TEXT,
    doc_id            TEXT,
    memory_id         TEXT,
    candidate_id      TEXT,
    review_id         TEXT,
    support_kind      TEXT,
    event_type        TEXT NOT NULL,
    decision          TEXT,
    reason            TEXT,
    payload_class     TEXT,
    before_snapshot   TEXT,
    after_snapshot    TEXT,
    evidence_refs     TEXT NOT NULL DEFAULT '[]',
    model             TEXT,
    prompt_hash       TEXT,
    config_hash       TEXT,
    thresholds        TEXT,
    status            TEXT NOT NULL,
    payload           TEXT NOT NULL DEFAULT '{}',
    error             TEXT
);

CREATE INDEX IF NOT EXISTS idx_memory_audit_operation ON memory_audit_events(operation_id);
CREATE INDEX IF NOT EXISTS idx_memory_audit_memory ON memory_audit_events(memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_audit_doc ON memory_audit_events(doc_id);
CREATE INDEX IF NOT EXISTS idx_memory_audit_type ON memory_audit_events(event_type);
"""

# ---------------------------------------------------------------------------
# Migrations - empty for v1; start from v2 onwards.
# Each entry: (version, description, [sql_statements])
# ---------------------------------------------------------------------------

MIGRATIONS: Sequence[tuple[int, str, list[str]]] = [
    (
        1,
        "Add tags column to entities, deprecate entity_type",
        [
            "ALTER TABLE entities ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'",
            "UPDATE entities SET tags = json_array(entity_type) WHERE tags = '[]'",
            "DROP INDEX IF EXISTS idx_entities_type",
        ],
    ),
    (
        2,
        "Add memory_contradictions table",
        [
            """CREATE TABLE IF NOT EXISTS memory_contradictions (
            memory_id_a TEXT NOT NULL REFERENCES memories(id),
            memory_id_b TEXT NOT NULL REFERENCES memories(id),
            classification TEXT NOT NULL,
            resolution TEXT DEFAULT 'pending',
            detected_at TEXT NOT NULL DEFAULT (datetime('now')),
            resolved_at TEXT,
            reason TEXT,
            PRIMARY KEY (memory_id_a, memory_id_b)
        )""",
        ],
    ),
    (
        3,
        "Add lean memory lifecycle metadata",
        [
            "ALTER TABLE memories ADD COLUMN retirement_reason TEXT",
            "ALTER TABLE memories ADD COLUMN retired_at TEXT",
            "ALTER TABLE memories ADD COLUMN superseded_at TEXT",
            "ALTER TABLE memories ADD COLUMN replacement_reason TEXT",
            "UPDATE memories SET status = 'retired' WHERE status = 'decayed'",
        ],
    ),
    (
        4,
        "Add agent session receipt lineage",
        [
            """CREATE TABLE IF NOT EXISTS agent_session_receipts (
            doc_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL,
            client TEXT NOT NULL,
            session_id TEXT NOT NULL,
            trigger TEXT NOT NULL,
            workspace TEXT NOT NULL,
            repo TEXT,
            branch TEXT,
            commit_sha TEXT,
            history_window_kind TEXT NOT NULL,
            history_window_start TEXT,
            history_window_end TEXT,
            submitted_at TEXT NOT NULL,
            document_hash TEXT NOT NULL,
            source_kind TEXT NOT NULL,
            document_uri TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        )""",
            "CREATE INDEX IF NOT EXISTS idx_agent_session_receipts_session ON agent_session_receipts(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_agent_session_receipts_source ON agent_session_receipts(source_id)",
        ],
    ),
    (
        5,
        "Add memory_reviews table for human-gated lifecycle decisions",
        [
            """CREATE TABLE IF NOT EXISTS memory_reviews (
            id                              TEXT PRIMARY KEY,
            kind                            TEXT NOT NULL,
            status                          TEXT NOT NULL,
            incumbent_memory_id             TEXT NOT NULL REFERENCES memories(id),
            challenger_memory_id            TEXT NOT NULL REFERENCES memories(id),
            reason                          TEXT,
            review_note                     TEXT,
            reviewer                        TEXT,
            expected_incumbent_updated_at   TEXT,
            expected_challenger_updated_at  TEXT,
            replacement_kind                TEXT NOT NULL DEFAULT 'supersession',
            created_at                      TEXT NOT NULL,
            resolved_at                     TEXT
        )""",
            "CREATE INDEX IF NOT EXISTS idx_memory_reviews_status ON memory_reviews(status)",
            "CREATE INDEX IF NOT EXISTS idx_memory_reviews_incumbent ON memory_reviews(incumbent_memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_reviews_challenger ON memory_reviews(challenger_memory_id)",
        ],
    ),
    (
        6,
        "Add provenance support ownership kind",
        [
            "ALTER TABLE memory_sources ADD COLUMN support_kind TEXT NOT NULL DEFAULT 'extracted'",
            "CREATE INDEX IF NOT EXISTS idx_memory_sources_doc_kind ON memory_sources(doc_id, support_kind)",
        ],
    ),
    (
        7,
        "Add memory audit event ledger",
        [
            """CREATE TABLE IF NOT EXISTS memory_audit_events (
            event_id          TEXT PRIMARY KEY,
            operation_id      TEXT NOT NULL,
            parent_event_id   TEXT,
            occurred_at       TEXT NOT NULL,
            actor_type        TEXT,
            actor_id          TEXT,
            run_id            TEXT,
            trace_id          TEXT,
            source_id         TEXT,
            doc_id            TEXT,
            memory_id         TEXT,
            candidate_id      TEXT,
            review_id         TEXT,
            support_kind      TEXT,
            event_type        TEXT NOT NULL,
            decision          TEXT,
            reason            TEXT,
            payload_class     TEXT,
            before_snapshot   TEXT,
            after_snapshot    TEXT,
            evidence_refs     TEXT NOT NULL DEFAULT '[]',
            model             TEXT,
            prompt_hash       TEXT,
            config_hash       TEXT,
            thresholds        TEXT,
            status            TEXT NOT NULL,
            payload           TEXT NOT NULL DEFAULT '{}',
            error             TEXT
        )""",
            "CREATE INDEX IF NOT EXISTS idx_memory_audit_operation ON memory_audit_events(operation_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_audit_memory ON memory_audit_events(memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_audit_doc ON memory_audit_events(doc_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_audit_type ON memory_audit_events(event_type)",
        ],
    ),
    (
        8,
        "Add shared auth sessions",
        [
            """CREATE TABLE IF NOT EXISTS auth_sessions (
            provider            TEXT NOT NULL,
            origin              TEXT NOT NULL,
            secret_encrypted    TEXT NOT NULL,
            principal_id        TEXT,
            principal_name      TEXT,
            principal_email     TEXT,
            browser             TEXT,
            status              TEXT NOT NULL,
            captured_at         TEXT NOT NULL,
            validated_at        TEXT,
            last_error          TEXT,
            updated_at          TEXT NOT NULL,
            PRIMARY KEY (provider, origin)
        )""",
            "CREATE INDEX IF NOT EXISTS idx_auth_sessions_status ON auth_sessions(status)",
        ],
    ),
    (
        9,
        "Add agent hook lifecycle receipts",
        [
            """CREATE TABLE IF NOT EXISTS agent_hook_receipts (
            receipt_id TEXT PRIMARY KEY,
            client TEXT NOT NULL,
            session_id TEXT NOT NULL,
            hook TEXT NOT NULL,
            workspace TEXT NOT NULL,
            repo TEXT,
            branch TEXT,
            commit_sha TEXT,
            submitted_at TEXT NOT NULL,
            metadata TEXT NOT NULL DEFAULT '{}',
            updated_at TEXT NOT NULL
        )""",
            "CREATE INDEX IF NOT EXISTS idx_agent_hook_receipts_session ON agent_hook_receipts(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_agent_hook_receipts_hook ON agent_hook_receipts(hook)",
        ],
    ),
    (
        10,
        "Add related challengers for grouped review cases",
        [
            """CREATE TABLE IF NOT EXISTS memory_review_related_challengers (
            review_id              TEXT NOT NULL REFERENCES memory_reviews(id),
            challenger_memory_id   TEXT NOT NULL REFERENCES memories(id),
            reason                 TEXT,
            created_at             TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (review_id, challenger_memory_id),
            UNIQUE (challenger_memory_id)
        )""",
            """CREATE INDEX IF NOT EXISTS idx_memory_review_related_review
           ON memory_review_related_challengers(review_id)""",
        ],
    ),
    (
        11,
        "Add client column and index to documents table",
        [
            "ALTER TABLE documents ADD COLUMN client TEXT",
            "CREATE INDEX IF NOT EXISTS idx_documents_source_client ON documents(source, client)",
        ],
    ),
    (
        12,
        "Split singleton agent-session source into per-client sources",
        [
            # For each known client that has documents under the singleton source,
            # upsert its per-client source row and re-point those documents to it.
            # Documents whose client value is not a recognised slug are left under
            # the singleton so no data is lost. The singleton row itself is removed
            # only when it has zero documents remaining.
            #
            # Step 1: populate documents.client from agent_session_receipts for rows
            # that are still under the singleton and do not yet have a client value.
            """UPDATE documents
           SET client = (
               SELECT asr.client
               FROM agent_session_receipts asr
               WHERE asr.doc_id = documents.doc_id
               LIMIT 1
           )
           WHERE documents.source = 'src-agent-sessions'
             AND documents.client IS NULL""",
            # Step 2: upsert the codex per-client source (idempotent).
            """INSERT INTO sources (id, type, name, config)
           SELECT
               'src-agent-sessions-codex',
               'agent_session',
               'Codex Session Summaries',
               (SELECT config FROM sources WHERE id = 'src-agent-sessions')
           WHERE EXISTS (
               SELECT 1 FROM sources WHERE id = 'src-agent-sessions'
           )
           ON CONFLICT(id) DO NOTHING""",
            # Step 3: upsert the claude-code per-client source (idempotent).
            """INSERT INTO sources (id, type, name, config)
           SELECT
               'src-agent-sessions-claude-code',
               'agent_session',
               'Claude Code Session Summaries',
               (SELECT config FROM sources WHERE id = 'src-agent-sessions')
           WHERE EXISTS (
               SELECT 1 FROM sources WHERE id = 'src-agent-sessions'
           )
           ON CONFLICT(id) DO NOTHING""",
            # Step 4: re-point codex documents to the codex source.
            """UPDATE documents
           SET source = 'src-agent-sessions-codex'
           WHERE source = 'src-agent-sessions'
             AND client = 'codex'""",
            # Step 5: re-point claude-code documents to the claude-code source.
            """UPDATE documents
           SET source = 'src-agent-sessions-claude-code'
           WHERE source = 'src-agent-sessions'
             AND client = 'claude-code'""",
            # Step 6: remove the singleton source only when it has no remaining
            # documents (clients other than the two known ones stay attached to it).
            """DELETE FROM sources
           WHERE id = 'src-agent-sessions'
             AND NOT EXISTS (
                 SELECT 1 FROM documents WHERE source = 'src-agent-sessions'
             )""",
        ],
    ),
    (
        13,
        "Rename agent-session sources to drop 'Summaries' and re-split any singleton remnants",
        [
            # The display name was tightened from 'X Session Summaries' to 'X Session'.
            # This migration also re-runs the singleton split because a server running
            # the pre-split code path could recreate src-agent-sessions and write
            # new documents to it before being restarted; the SQL below is identical
            # to migration 12 and idempotent on a fully-split database.
            """UPDATE sources SET name = 'Codex Session'
           WHERE id = 'src-agent-sessions-codex'""",
            """UPDATE sources SET name = 'Claude Code Session'
           WHERE id = 'src-agent-sessions-claude-code'""",
            """UPDATE documents
           SET client = (
               SELECT asr.client
               FROM agent_session_receipts asr
               WHERE asr.doc_id = documents.doc_id
               LIMIT 1
           )
           WHERE documents.source = 'src-agent-sessions'
             AND documents.client IS NULL""",
            """INSERT INTO sources (id, type, name, config)
           SELECT
               'src-agent-sessions-codex',
               'agent_session',
               'Codex Session',
               (SELECT config FROM sources WHERE id = 'src-agent-sessions')
           WHERE EXISTS (
               SELECT 1 FROM sources WHERE id = 'src-agent-sessions'
           )
           ON CONFLICT(id) DO NOTHING""",
            """INSERT INTO sources (id, type, name, config)
           SELECT
               'src-agent-sessions-claude-code',
               'agent_session',
               'Claude Code Session',
               (SELECT config FROM sources WHERE id = 'src-agent-sessions')
           WHERE EXISTS (
               SELECT 1 FROM sources WHERE id = 'src-agent-sessions'
           )
           ON CONFLICT(id) DO NOTHING""",
            """UPDATE documents
           SET source = 'src-agent-sessions-codex'
           WHERE source = 'src-agent-sessions'
             AND client = 'codex'""",
            """UPDATE documents
           SET source = 'src-agent-sessions-claude-code'
           WHERE source = 'src-agent-sessions'
             AND client = 'claude-code'""",
            """DELETE FROM sources
           WHERE id = 'src-agent-sessions'
             AND NOT EXISTS (
                 SELECT 1 FROM documents WHERE source = 'src-agent-sessions'
             )""",
        ],
    ),
    (
        14,
        "Add visibility and owner columns to memories",
        [
            "ALTER TABLE memories ADD COLUMN visibility TEXT NOT NULL DEFAULT 'workspace'",
            "ALTER TABLE memories ADD COLUMN owner_user_id TEXT",
            "CREATE INDEX IF NOT EXISTS idx_memories_access ON memories(status, visibility)",
            "CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner_user_id)",
            "DROP INDEX IF EXISTS idx_memories_scope",
        ],
    ),
    (
        15,
        "Backfill visibility and project_key from legacy scope",
        [
            "UPDATE memories SET visibility = 'workspace' WHERE visibility IS NULL OR visibility = ''",
            "UPDATE memories SET owner_user_id = NULL WHERE visibility = 'workspace'",
            "UPDATE memories SET project_key = substr(scope, 9) WHERE project_key IS NULL AND scope LIKE 'project:%'",
            "UPDATE memories SET project_key = 'SHARED' WHERE project_key IS NULL AND scope = 'team'",
            "UPDATE memories SET project_key = 'UNSORTED' WHERE project_key IS NULL",
        ],
    ),
    (
        16,
        "Backfill NULL project_key to UNSORTED and add the projects stub table",
        [
            # The CREATE TABLE matches SCHEMA above; running it in a migration covers
            # any database that already passed connect() before SCHEMA carried it.
            "CREATE TABLE IF NOT EXISTS projects (project_key TEXT PRIMARY KEY)",
            f"UPDATE memories SET project_key = '{UNSORTED_PROJECT_KEY}' WHERE project_key IS NULL",
        ],
    ),
    (
        17,
        "Replace stub projects table with full schema and seed reserved rows",
        [
            # Rebuild the stub table under the full schema in one step. The
            # two reserved rows are seeded immediately so the resolver's
            # `UNSORTED` default and the SHARED bucket are valid foreign-key
            # targets the moment the migration completes.
            "DROP TABLE IF EXISTS projects",
            (
                "CREATE TABLE projects ("
                "    id            TEXT PRIMARY KEY,"
                "    key           TEXT NOT NULL UNIQUE,"
                "    name          TEXT NOT NULL,"
                "    is_shared     INTEGER NOT NULL DEFAULT 0,"
                "    created_at    TEXT NOT NULL DEFAULT (datetime('now'))"
                ")"
            ),
            # Seed the two reserved rows. INSERT OR IGNORE keeps the migration
            # idempotent against any future re-application.
            (
                "INSERT OR IGNORE INTO projects (id, key, name, is_shared) "
                f"VALUES ('proj-shared',   '{SHARED_PROJECT_KEY}',   'Shared',   1)"
            ),
            (
                "INSERT OR IGNORE INTO projects (id, key, name, is_shared) "
                f"VALUES ('proj-unsorted', '{UNSORTED_PROJECT_KEY}', 'Unsorted', 0)"
            ),
            # The sources table gains project_binding. Legacy rows read NULL.
            "ALTER TABLE sources ADD COLUMN project_binding TEXT",
        ],
    ),
    (
        18,
        "Track source creator for shared source management",
        [
            "ALTER TABLE sources ADD COLUMN created_by_user_id TEXT",
        ],
    ),
    (
        19,
        "Track per-user source subscriptions",
        [
            """CREATE TABLE IF NOT EXISTS source_subscriptions (
            source_id   TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
            user_id     TEXT NOT NULL,
            enabled     INTEGER NOT NULL DEFAULT 1,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (source_id, user_id)
        )""",
            "CREATE INDEX IF NOT EXISTS idx_source_subscriptions_user ON source_subscriptions(user_id)",
        ],
    ),
    (
        20,
        "Add per-source sync schedules",
        [
            "ALTER TABLE sources ADD COLUMN sync_schedule_enabled INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE sources ADD COLUMN sync_schedule_interval_minutes INTEGER NOT NULL DEFAULT 1440",
            "ALTER TABLE sources ADD COLUMN sync_schedule_next_at TEXT",
            "ALTER TABLE sources ADD COLUMN sync_schedule_updated_at TEXT",
            "CREATE INDEX IF NOT EXISTS idx_sources_sync_schedule_due ON sources(sync_schedule_enabled, sync_schedule_next_at)",
        ],
    ),
    (
        21,
        "Add memory curation lineage metadata",
        [
            "ALTER TABLE memories ADD COLUMN repo_identifier TEXT",
            "ALTER TABLE memories ADD COLUMN memory_level TEXT NOT NULL DEFAULT 'atomic'",
            "ALTER TABLE memories ADD COLUMN curation_cluster_id TEXT",
            """CREATE TABLE IF NOT EXISTS memory_derivations (
            parent_memory_id TEXT NOT NULL REFERENCES memories(id),
            child_memory_id  TEXT NOT NULL REFERENCES memories(id),
            relation         TEXT NOT NULL DEFAULT 'summarizes',
            created_at       TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (parent_memory_id, child_memory_id, relation)
        )""",
            """CREATE TABLE IF NOT EXISTS memory_curation_runs (
            id                   TEXT PRIMARY KEY,
            policy_id            TEXT NOT NULL,
            source_type          TEXT NOT NULL,
            client               TEXT,
            repo_identifier      TEXT,
            project_key          TEXT,
            candidate_count      INTEGER NOT NULL,
            created_memory_count INTEGER NOT NULL,
            skipped_reason       TEXT,
            error                TEXT,
            started_at           TEXT NOT NULL,
            completed_at         TEXT
        )""",
            "CREATE INDEX IF NOT EXISTS idx_memories_repo ON memories(repo_identifier)",
            "CREATE INDEX IF NOT EXISTS idx_memories_curation_cluster ON memories(curation_cluster_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_derivations_child ON memory_derivations(child_memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_curation_runs_scope ON memory_curation_runs(source_type, client, repo_identifier, project_key)",
        ],
    ),
    (
        22,
        "Add private agent knowledge bundle concept claims",
        [
            """CREATE TABLE IF NOT EXISTS agent_concepts (
            id                  TEXT PRIMARY KEY,
            source_id           TEXT NOT NULL,
            owner_user_id       TEXT NOT NULL,
            visibility          TEXT NOT NULL DEFAULT 'private',
            workspace           TEXT NOT NULL,
            repo_identifier     TEXT,
            concept_type        TEXT NOT NULL,
            concept_path        TEXT NOT NULL,
            title               TEXT NOT NULL,
            markdown_body       TEXT NOT NULL,
            frontmatter_json    TEXT NOT NULL DEFAULT '{}',
            created_at          TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
            last_observed_at    TEXT NOT NULL,
            CHECK (visibility = 'private')
        )""",
            """CREATE TABLE IF NOT EXISTS agent_claims (
            id                  TEXT PRIMARY KEY,
            concept_id          TEXT NOT NULL REFERENCES agent_concepts(id),
            display_anchor      TEXT NOT NULL,
            claim_text          TEXT NOT NULL,
            memory_type         TEXT NOT NULL,
            tags                TEXT NOT NULL DEFAULT '[]',
            confidence          REAL NOT NULL DEFAULT 0.7,
            memory_id           TEXT NOT NULL REFERENCES memories(id),
            created_at          TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
            last_observed_at    TEXT NOT NULL
        )""",
            """CREATE TABLE IF NOT EXISTS agent_claim_citations (
            claim_id        TEXT NOT NULL REFERENCES agent_claims(id),
            citation_url    TEXT NOT NULL,
            observed_at     TEXT NOT NULL,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (claim_id, citation_url)
        )""",
            "CREATE INDEX IF NOT EXISTS idx_agent_concepts_owner_repo ON agent_concepts(owner_user_id, repo_identifier)",
            "CREATE INDEX IF NOT EXISTS idx_agent_claims_concept ON agent_claims(concept_id)",
            "CREATE INDEX IF NOT EXISTS idx_agent_claims_memory ON agent_claims(memory_id)",
        ],
    ),
    (
        23,
        "Add structured memory replacement kind",
        [
            "ALTER TABLE memories ADD COLUMN replacement_kind TEXT",
        ],
    ),
    (
        24,
        "Persist source identity on memory provenance rows",
        [
            "ALTER TABLE memory_sources ADD COLUMN source_id TEXT",
            """UPDATE memory_sources
           SET source_id = (
               SELECT documents.source
               FROM documents
               WHERE documents.doc_id = memory_sources.doc_id
               LIMIT 1
           )
           WHERE source_id IS NULL""",
            "CREATE INDEX IF NOT EXISTS idx_memory_sources_source ON memory_sources(source_id)",
        ],
    ),
    (
        25,
        "Project materialized memory id on relation runs",
        [
            "ALTER TABLE relation_runs ADD COLUMN result_memory_id TEXT",
            """UPDATE relation_runs
           SET result_memory_id = json_extract(audit_json, '$.result_memory_id')
           WHERE result_memory_id IS NULL
             AND audit_json IS NOT NULL""",
            "CREATE INDEX IF NOT EXISTS idx_relation_runs_result_memory ON relation_runs(result_memory_id)",
        ],
    ),
    (
        26,
        "Backfill relation-run candidate and relation snapshot audit hashes",
        [],
    ),
    (
        27,
        "Track explicit source observation time on memory provenance",
        [
            "ALTER TABLE memory_sources ADD COLUMN source_updated_at TEXT",
        ],
    ),
    (
        28,
        "Backfill canonical source id on memory provenance rows",
        [
            """UPDATE memory_sources
           SET source_id = (
               SELECT documents.source
               FROM documents
               WHERE documents.doc_id = memory_sources.doc_id
               LIMIT 1
           )
           WHERE (source_id IS NULL OR source_id = '')
             AND EXISTS (
               SELECT 1
               FROM documents
               WHERE documents.doc_id = memory_sources.doc_id
             )""",
        ],
    ),
    (
        29,
        "Persist replacement kind on memory reviews",
        [
            "ALTER TABLE memory_reviews ADD COLUMN replacement_kind TEXT NOT NULL DEFAULT 'supersession'",
        ],
    ),
    (
        30,
        "Add metadata keyword search projection",
        [
            """CREATE VIRTUAL TABLE IF NOT EXISTS memory_search_metadata_fts USING fts5(
                memory_id UNINDEXED,
                source_id UNINDEXED,
                doc_id UNINDEXED,
                source_type UNINDEXED,
                metadata_title_tokens,
                metadata_external_id_tokens,
                metadata_path_tokens,
                metadata_source_name_tokens,
                metadata_label_context_tokens,
                tokenize='porter unicode61'
            )""",
        ],
    ),
    (
        31,
        "Add metadata alias and substring search projections",
        [
            """CREATE VIRTUAL TABLE IF NOT EXISTS memory_search_metadata_alias_fts USING fts5(
                memory_id UNINDEXED,
                source_id UNINDEXED,
                doc_id UNINDEXED,
                source_type UNINDEXED,
                metadata_alias_tokens,
                tokenize='porter unicode61'
            )""",
            """CREATE TABLE IF NOT EXISTS memory_search_metadata_trigram (
                memory_id        TEXT NOT NULL,
                source_id        TEXT,
                doc_id           TEXT NOT NULL,
                source_type      TEXT NOT NULL,
                metadata_compact TEXT NOT NULL,
                PRIMARY KEY (memory_id, doc_id)
            )""",
        ],
    ),
    (
        32,
        "Add compact entity alias lookup indexes",
        [
            "CREATE INDEX IF NOT EXISTS idx_entity_aliases_compact ON entity_aliases(REPLACE(alias_normalized, ' ', ''))",
            "CREATE INDEX IF NOT EXISTS idx_entities_canonical_compact ON entities(REPLACE(canonical_name, ' ', ''))",
        ],
    ),
    (
        33,
        "Add entity alias lexical search projection",
        [
            """CREATE VIRTUAL TABLE IF NOT EXISTS entity_alias_search_fts USING fts5(
                entity_id UNINDEXED,
                canonical_name UNINDEXED,
                alias_normalized UNINDEXED,
                search_text,
                tokenize='porter unicode61'
            )""",
            "DELETE FROM entity_alias_search_fts",
            """INSERT INTO entity_alias_search_fts (
                   entity_id,
                   canonical_name,
                   alias_normalized,
                   search_text
               )
               SELECT
                   e.id,
                   e.canonical_name,
                   e.canonical_name,
                   COALESCE(e.canonical_name, '') || ' ' || COALESCE(e.display_name, '')
               FROM entities e
               UNION ALL
               SELECT
                   ea.canonical_id,
                   e.canonical_name,
                   ea.alias_normalized,
                   COALESCE(ea.alias, '') || ' ' || COALESCE(ea.alias_normalized, '')
               FROM entity_aliases ea
               JOIN entities e ON e.id = ea.canonical_id""",
        ],
    ),
    (
        34,
        "Rebuild entity alias lexical search projection without tag tokens",
        [
            "DELETE FROM entity_alias_search_fts",
            """INSERT INTO entity_alias_search_fts (
                   entity_id,
                   canonical_name,
                   alias_normalized,
                   search_text
               )
               SELECT
                   e.id,
                   e.canonical_name,
                   e.canonical_name,
                   COALESCE(e.canonical_name, '') || ' ' || COALESCE(e.display_name, '')
               FROM entities e
               UNION ALL
               SELECT
                   ea.canonical_id,
                   e.canonical_name,
                   ea.alias_normalized,
                   COALESCE(ea.alias, '') || ' ' || COALESCE(ea.alias_normalized, '')
               FROM entity_aliases ea
               JOIN entities e ON e.id = ea.canonical_id""",
        ],
    ),
    (
        35,
        "Track local source execution owner",
        [
            "ALTER TABLE sources ADD COLUMN execution_owner_user_id TEXT",
            """UPDATE sources
               SET execution_owner_user_id = created_by_user_id
               WHERE execution_owner_user_id IS NULL
                 AND created_by_user_id IS NOT NULL""",
        ],
    ),
    (
        36,
        "Add durable source sync runs",
        [
            """CREATE TABLE IF NOT EXISTS source_sync_runs (
                run_id                  TEXT PRIMARY KEY,
                workspace_id            TEXT NOT NULL,
                source_id               TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                trigger                 TEXT NOT NULL,
                status                  TEXT NOT NULL,
                force_full_sync         INTEGER NOT NULL DEFAULT 0,
                input_snapshot_id       TEXT,
                rerun_input_snapshot_id TEXT,
                lease_owner             TEXT,
                lease_expires_at        TEXT,
                lease_attempt_count     INTEGER NOT NULL DEFAULT 0,
                recovery_count          INTEGER NOT NULL DEFAULT 0,
                rerun_requested         INTEGER NOT NULL DEFAULT 0,
                next_attempt_at         TEXT,
                error_message           TEXT,
                created_at              TEXT NOT NULL,
                updated_at              TEXT NOT NULL,
                started_at              TEXT,
                completed_at            TEXT
            )""",
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_source_sync_runs_active
               ON source_sync_runs(workspace_id, source_id)
               WHERE status IN ('pending', 'running')""",
            """CREATE TABLE IF NOT EXISTS source_sync_inputs (
                input_id            TEXT PRIMARY KEY,
                workspace_id        TEXT NOT NULL,
                source_id           TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                input_generation    INTEGER NOT NULL,
                raw_uri             TEXT NOT NULL,
                raw_sha256          TEXT NOT NULL,
                raw_content_type    TEXT NOT NULL,
                metadata_json       TEXT NOT NULL DEFAULT '{}',
                created_at          TEXT NOT NULL,
                UNIQUE(workspace_id, source_id, input_generation)
            )""",
            """CREATE INDEX IF NOT EXISTS idx_source_sync_inputs_source
               ON source_sync_inputs(workspace_id, source_id, input_generation)""",
            """CREATE UNIQUE INDEX IF NOT EXISTS idx_source_sync_inputs_raw_hash
               ON source_sync_inputs(workspace_id, source_id, raw_sha256)""",
            """CREATE TABLE IF NOT EXISTS source_sync_snapshot_items (
                workspace_id        TEXT NOT NULL,
                source_id           TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                snapshot_id         TEXT NOT NULL,
                doc_id              TEXT NOT NULL,
                input_id            TEXT NOT NULL REFERENCES source_sync_inputs(input_id) ON DELETE CASCADE,
                created_at          TEXT NOT NULL,
                PRIMARY KEY (workspace_id, source_id, snapshot_id, doc_id)
            )""",
            """CREATE INDEX IF NOT EXISTS idx_source_sync_snapshot_items_input
               ON source_sync_snapshot_items(input_id)""",
        ],
    ),
    (
        37,
        "Add local agent job broker",
        [
            """CREATE TABLE IF NOT EXISTS local_agent_jobs (
                job_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL DEFAULT 'default',
                source_id TEXT NOT NULL DEFAULT '',
                source_type TEXT NOT NULL,
                operation TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                payload_json TEXT NOT NULL DEFAULT '{}',
                created_by_user_id TEXT NOT NULL,
                execution_owner_user_id TEXT NOT NULL,
                lease_owner_user_id TEXT,
                leased_until TEXT,
                attempt_count INTEGER NOT NULL DEFAULT 0,
                result_json TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                finished_at TEXT
            )""",
            """CREATE INDEX IF NOT EXISTS idx_local_agent_jobs_owner_status
               ON local_agent_jobs(execution_owner_user_id, status, leased_until)""",
            """CREATE TABLE IF NOT EXISTS local_agent_heartbeats (
                user_id TEXT PRIMARY KEY,
                last_seen_at TEXT NOT NULL
            )""",
        ],
    ),
    # Migration 36 is expanded in fresh installs, while 38 upgrades databases
    # that already recorded the earlier 36. Duplicate columns are intentionally
    # handled by the migration runner's additive-schema compatibility path.
    (
        38,
        "Track local source input snapshots",
        [
            "ALTER TABLE source_sync_runs ADD COLUMN input_snapshot_id TEXT",
            "ALTER TABLE source_sync_runs ADD COLUMN rerun_input_snapshot_id TEXT",
            """CREATE TABLE IF NOT EXISTS source_sync_snapshot_items (
                workspace_id        TEXT NOT NULL,
                source_id           TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                snapshot_id         TEXT NOT NULL,
                doc_id              TEXT NOT NULL,
                input_id            TEXT NOT NULL REFERENCES source_sync_inputs(input_id) ON DELETE CASCADE,
                created_at          TEXT NOT NULL,
                PRIMARY KEY (workspace_id, source_id, snapshot_id, doc_id)
            )""",
            """CREATE INDEX IF NOT EXISTS idx_source_sync_snapshot_items_input
               ON source_sync_snapshot_items(input_id)""",
        ],
    ),
    (
        39,
        "Persist source sync progress",
        [
            "ALTER TABLE source_sync_runs ADD COLUMN progress_json TEXT",
            "ALTER TABLE source_sync_runs ADD COLUMN progress_revision INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE source_sync_runs ADD COLUMN progress_updated_at TEXT",
        ],
    ),
    (
        40,
        "Persist personal Source List pins",
        [
            """CREATE TABLE IF NOT EXISTS source_list_pins (
                source_id   TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                user_id     TEXT NOT NULL,
                pinned_at   TEXT NOT NULL,
                PRIMARY KEY (source_id, user_id)
            )""",
            "CREATE INDEX IF NOT EXISTS idx_source_list_pins_user ON source_list_pins(user_id, pinned_at)",
            """CREATE TABLE IF NOT EXISTS source_list_preferences (
                user_id     TEXT PRIMARY KEY,
                sort_mode   TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            )""",
        ],
    ),
    (
        41,
        "Remove obsolete GitHub repository clone paths",
        [
            """UPDATE sources
               SET config = json_remove(config, '$.repo_path')
               WHERE type = 'github_repo'
                 AND json_type(config, '$.repo_path') IS NOT NULL""",
        ],
    ),
    (
        42,
        "Materialize Source access policy and ownership",
        [
            "ALTER TABLE sources ADD COLUMN owner_user_id TEXT",
            "ALTER TABLE sources ADD COLUMN access_policy TEXT",
            "ALTER TABLE sources ADD COLUMN access_state TEXT",
            "CREATE INDEX IF NOT EXISTS idx_sources_access ON sources(access_policy, owner_user_id, access_state)",
        ],
    ),
    (
        43,
        "Add durable Source access transitions",
        [
            """CREATE TABLE IF NOT EXISTS source_access_transitions (
                operation_id          TEXT PRIMARY KEY,
                source_id             TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                idempotency_key       TEXT NOT NULL,
                actor_user_id         TEXT NOT NULL,
                previous_policy       TEXT NOT NULL CHECK (previous_policy IN ('private', 'workspace')),
                target_policy         TEXT NOT NULL CHECK (target_policy IN ('private', 'workspace')),
                previous_source_status TEXT NOT NULL,
                status                TEXT NOT NULL
                                      CHECK (status IN ('queued', 'running', 'failed', 'completed', 'reverted')),
                total_memories        INTEGER NOT NULL DEFAULT 0,
                processed_memories    INTEGER NOT NULL DEFAULT 0,
                error_code            TEXT,
                error_message         TEXT,
                created_at            TEXT NOT NULL,
                updated_at            TEXT NOT NULL,
                completed_at          TEXT,
                UNIQUE (source_id, idempotency_key)
            )""",
            "CREATE INDEX IF NOT EXISTS idx_source_access_transitions_status "
            "ON source_access_transitions(status, updated_at)",
        ],
    ),
    (
        44,
        "Allow managed-capture concepts to follow Source access",
        [
            "PRAGMA foreign_keys = OFF",
            """CREATE TABLE agent_concepts_access_new (
                id                  TEXT PRIMARY KEY,
                source_id           TEXT NOT NULL,
                owner_user_id       TEXT NOT NULL,
                visibility          TEXT NOT NULL DEFAULT 'private',
                workspace           TEXT NOT NULL,
                repo_identifier     TEXT,
                concept_type        TEXT NOT NULL,
                concept_path        TEXT NOT NULL,
                title               TEXT NOT NULL,
                markdown_body       TEXT NOT NULL,
                frontmatter_json    TEXT NOT NULL DEFAULT '{}',
                created_at          TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
                last_observed_at    TEXT NOT NULL,
                CHECK (visibility IN ('private', 'workspace'))
            )""",
            """INSERT INTO agent_concepts_access_new
               SELECT * FROM agent_concepts""",
            "DROP TABLE agent_concepts",
            "ALTER TABLE agent_concepts_access_new RENAME TO agent_concepts",
            "CREATE INDEX IF NOT EXISTS idx_agent_concepts_owner_repo "
            "ON agent_concepts(owner_user_id, repo_identifier)",
            "PRAGMA foreign_keys = ON",
        ],
    ),
    (
        45,
        "Partition legacy coding-session sources by owner",
        ["SELECT 1"],
    ),
    (
        46,
        "Persist reversible Source access memory splits",
        [
            """CREATE TABLE IF NOT EXISTS source_access_transition_memory_map (
                operation_id       TEXT NOT NULL REFERENCES source_access_transitions(operation_id) ON DELETE CASCADE,
                source_id          TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                original_memory_id TEXT NOT NULL,
                target_memory_id   TEXT NOT NULL,
                PRIMARY KEY (operation_id, original_memory_id),
                UNIQUE (operation_id, target_memory_id)
            )""",
        ],
    ),
    (
        47,
        "Add durable Source Projection lineage",
        [
            """CREATE TABLE IF NOT EXISTS source_projection_runs (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                source_type TEXT NOT NULL,
                coverage TEXT NOT NULL,
                scope_json TEXT NOT NULL,
                checkpoint_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_source_projection_runs_source ON source_projection_runs(source_id, created_at)",
            """CREATE TABLE IF NOT EXISTS source_units (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                unit_type TEXT NOT NULL,
                provider_key TEXT NOT NULL,
                locator_json TEXT NOT NULL DEFAULT '{}',
                current_revision_id TEXT,
                updated_at TEXT NOT NULL,
                UNIQUE (source_id, provider_key)
            )""",
            """CREATE TABLE IF NOT EXISTS source_observations (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                source_unit_id TEXT NOT NULL REFERENCES source_units(id) ON DELETE CASCADE,
                observation_type TEXT NOT NULL,
                provider_key TEXT NOT NULL,
                locator_json TEXT NOT NULL DEFAULT '{}',
                current_revision_id TEXT,
                updated_at TEXT NOT NULL,
                UNIQUE (source_unit_id, provider_key)
            )""",
            """CREATE TABLE IF NOT EXISTS source_observation_revisions (
                id TEXT PRIMARY KEY,
                observation_id TEXT NOT NULL REFERENCES source_observations(id) ON DELETE CASCADE,
                semantic_hash TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                observed_at TEXT,
                created_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS source_unit_revisions (
                id TEXT PRIMARY KEY,
                source_unit_id TEXT NOT NULL REFERENCES source_units(id) ON DELETE CASCADE,
                semantic_hash TEXT NOT NULL,
                location_hash TEXT,
                membership_hash TEXT,
                access_hash TEXT,
                observation_revision_ids_json TEXT NOT NULL,
                observed_at TEXT,
                created_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS source_projection_relations (
                projection_run_id TEXT NOT NULL REFERENCES source_projection_runs(id) ON DELETE CASCADE,
                relation_index INTEGER NOT NULL,
                relation_type TEXT NOT NULL,
                from_id TEXT NOT NULL,
                to_id TEXT NOT NULL,
                provider_relation_id TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY (projection_run_id, relation_index)
            )""",
            """CREATE TABLE IF NOT EXISTS source_revision_deltas (
                projection_run_id TEXT NOT NULL REFERENCES source_projection_runs(id) ON DELETE CASCADE,
                delta_index INTEGER NOT NULL,
                source_unit_id TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (projection_run_id, delta_index)
            )""",
        ],
    ),
    (
        48,
        "Add lifecycle cutover gates findings and support assertions",
        [
            """CREATE TABLE IF NOT EXISTS source_lifecycle_gates (
                source_id TEXT PRIMARY KEY REFERENCES sources(id) ON DELETE CASCADE,
                state TEXT NOT NULL CHECK (state IN ('gated', 'enabled')),
                reason TEXT,
                audited_at TEXT,
                enabled_at TEXT,
                updated_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS lifecycle_cutover_findings (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                reason TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('open', 'resolved')),
                available_provenance_json TEXT NOT NULL DEFAULT '{}',
                mapping_attempt_json TEXT NOT NULL DEFAULT '{}',
                observation_id TEXT,
                source_unit_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                resolved_at TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_cutover_findings_source_status ON lifecycle_cutover_findings(source_id, status)",
            """CREATE TABLE IF NOT EXISTS evidence_references (
                id TEXT PRIMARY KEY,
                evidence_unit_id TEXT NOT NULL REFERENCES evidence_units(id) ON DELETE CASCADE,
                role TEXT NOT NULL CHECK (role IN ('primary', 'required', 'context')),
                anchor_kind TEXT NOT NULL,
                observation_id TEXT NOT NULL REFERENCES source_observations(id) ON DELETE CASCADE,
                observation_revision_id TEXT NOT NULL REFERENCES source_observation_revisions(id),
                fragment_id TEXT,
                range_start INTEGER,
                range_end INTEGER,
                created_at TEXT NOT NULL
            )""",
            """CREATE TABLE IF NOT EXISTS memory_support_assertions (
                id TEXT PRIMARY KEY,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                evidence_reference_id TEXT NOT NULL REFERENCES evidence_references(id) ON DELETE CASCADE,
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                access_context_hash TEXT NOT NULL,
                active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                removed_at TEXT,
                UNIQUE (memory_id, evidence_reference_id)
            )""",
            "CREATE INDEX IF NOT EXISTS idx_memory_support_assertions_active ON memory_support_assertions(memory_id, active)",
        ],
    ),
    (
        49,
        "Add atomic lifecycle plans and reviews",
        [
            """CREATE TABLE IF NOT EXISTS lifecycle_plans (
                id TEXT PRIMARY KEY,
                reconciliation_scope_id TEXT NOT NULL,
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                source_unit_id TEXT NOT NULL,
                target_unit_revision_id TEXT,
                status TEXT NOT NULL CHECK (status IN ('staged', 'applied', 'rejected', 'stale')),
                payload_json TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                applied_at TEXT,
                error TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS lifecycle_reviews (
                id TEXT PRIMARY KEY,
                lifecycle_plan_id TEXT NOT NULL REFERENCES lifecycle_plans(id) ON DELETE CASCADE,
                incumbent_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                status TEXT NOT NULL CHECK (status IN ('pending', 'approved', 'rejected', 'stale')),
                staged_evidence_json TEXT NOT NULL DEFAULT '{}',
                reason TEXT,
                created_at TEXT NOT NULL,
                resolved_at TEXT
            )""",
        ],
    ),
    (
        50,
        "Add durable lifecycle vector outbox",
        [
            """CREATE TABLE IF NOT EXISTS lifecycle_vector_outbox (
                id TEXT PRIMARY KEY,
                lifecycle_plan_id TEXT NOT NULL REFERENCES lifecycle_plans(id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL,
                operation TEXT NOT NULL CHECK (operation IN ('upsert', 'delete')),
                status TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'failed')),
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE (lifecycle_plan_id, memory_id, operation)
            )""",
            "CREATE INDEX IF NOT EXISTS idx_lifecycle_vector_outbox_status "
            "ON lifecycle_vector_outbox(status, created_at)",
        ],
    ),
    (
        51,
        "Add durable lifecycle backfill jobs",
        [
            """CREATE TABLE IF NOT EXISTS lifecycle_backfill_jobs (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed')),
                scanned_memories INTEGER NOT NULL DEFAULT 0,
                mapped_memories INTEGER NOT NULL DEFAULT 0,
                finding_count INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_lifecycle_backfill_jobs_source "
            "ON lifecycle_backfill_jobs(source_id, created_at)",
        ],
    ),
    (
        52,
        "Add durable Projection Scope transitions",
        [
            """CREATE TABLE IF NOT EXISTS projection_scope_transitions (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                previous_scope_json TEXT NOT NULL,
                target_scope_json TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('pending', 'running', 'applied', 'failed')),
                run_id TEXT,
                coverage TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT
            )""",
            "CREATE INDEX IF NOT EXISTS idx_projection_scope_transitions_source "
            "ON projection_scope_transitions(source_id, created_at)",
        ],
    ),
    (
        53,
        "Remove consolidated memory persistence",
        [
            "DROP INDEX IF EXISTS idx_memories_curation_cluster",
            "DROP INDEX IF EXISTS idx_memory_derivations_child",
            "DROP INDEX IF EXISTS idx_memory_curation_runs_scope",
            "DROP TABLE IF EXISTS memory_derivations",
            "DROP TABLE IF EXISTS memory_curation_runs",
        ],
    ),
    (
        54,
        "Track Source Unit document lineage across moves",
        [
            """CREATE TABLE IF NOT EXISTS source_unit_document_lineage (
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                document_id TEXT NOT NULL,
                source_unit_id TEXT NOT NULL REFERENCES source_units(id) ON DELETE CASCADE,
                is_current INTEGER NOT NULL DEFAULT 1,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                PRIMARY KEY (source_id, document_id)
            )""",
            "CREATE INDEX IF NOT EXISTS idx_source_unit_document_lineage_unit "
            "ON source_unit_document_lineage(source_unit_id, is_current, last_seen_at)",
            """INSERT OR IGNORE INTO source_unit_document_lineage (
                   source_id, document_id, source_unit_id, is_current,
                   first_seen_at, last_seen_at
               )
               SELECT source_id,
                      json_extract(locator_json, '$.document_id'),
                      id,
                      1,
                      updated_at,
                      updated_at
               FROM source_units
               WHERE json_extract(locator_json, '$.document_id') IS NOT NULL
                 AND json_extract(locator_json, '$.document_id') != ''""",
        ],
    ),
    (
        55,
        "Allow document locators to have multiple Source Unit incarnations",
        [
            """CREATE TABLE IF NOT EXISTS source_unit_document_lineage_history (
                source_id TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
                document_id TEXT NOT NULL,
                source_unit_id TEXT NOT NULL REFERENCES source_units(id) ON DELETE CASCADE,
                is_current INTEGER NOT NULL DEFAULT 1,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                PRIMARY KEY (source_id, document_id, source_unit_id)
            )""",
            """INSERT OR IGNORE INTO source_unit_document_lineage_history (
                   source_id, document_id, source_unit_id, is_current,
                   first_seen_at, last_seen_at
               )
               SELECT source_id, document_id, source_unit_id, is_current,
                      first_seen_at, last_seen_at
               FROM source_unit_document_lineage""",
            "DROP INDEX IF EXISTS idx_source_unit_document_lineage_unit",
            "DROP TABLE IF EXISTS source_unit_document_lineage",
            "CREATE INDEX IF NOT EXISTS idx_source_unit_document_history_unit "
            "ON source_unit_document_lineage_history(source_unit_id, is_current, last_seen_at)",
        ],
    ),
    (
        56,
        "Add durable source-deletion vector cleanup outbox",
        [
            """CREATE TABLE IF NOT EXISTS source_deletion_vector_outbox (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                memory_id TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('pending', 'completed', 'failed')),
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )""",
            "CREATE INDEX IF NOT EXISTS idx_source_deletion_vector_outbox_status "
            "ON source_deletion_vector_outbox(status, created_at)",
        ],
    ),
    (
        57,
        "Track source sync consumed input boundary",
        [
            "ALTER TABLE source_sync_runs ADD COLUMN input_generation_watermark INTEGER",
            "ALTER TABLE source_sync_runs ADD COLUMN rerun_input_generation_watermark INTEGER",
            "ALTER TABLE source_sync_runs ADD COLUMN source_config_revision TEXT",
            "ALTER TABLE source_sync_runs ADD COLUMN rerun_source_config_revision TEXT",
        ],
    ),
    (
        58,
        "Add durable source activity admission",
        [
            "ALTER TABLE sources ADD COLUMN activity_epoch INTEGER NOT NULL DEFAULT 0",
            """CREATE TABLE IF NOT EXISTS source_activity_leases (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                kind TEXT NOT NULL CHECK (kind IN ('sync', 'external_collection', 'agent_patch', 'maintenance')),
                epoch INTEGER NOT NULL,
                capability TEXT,
                lease_until TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (source_id) REFERENCES sources(id) ON DELETE CASCADE
            )""",
            "CREATE INDEX IF NOT EXISTS idx_source_activity_leases_source "
            "ON source_activity_leases(source_id, lease_until)",
        ],
    ),
    (
        59,
        "Track source sync predecessor activity",
        [
            "ALTER TABLE source_sync_runs ADD COLUMN predecessor_activity_id TEXT",
            "ALTER TABLE source_sync_runs ADD COLUMN rerun_predecessor_activity_id TEXT",
        ],
    ),
    (
        60,
        "Add immutable exact revision replay ledgers",
        [
            """CREATE TABLE IF NOT EXISTS lifecycle_replay_ledgers (
                lifecycle_plan_id TEXT PRIMARY KEY REFERENCES lifecycle_plans(id) ON DELETE CASCADE,
                claim_count INTEGER NOT NULL CHECK (claim_count >= 0),
                created_at TEXT NOT NULL,
                invalidated_at TEXT,
                invalidation_reason TEXT
            )""",
            """CREATE TABLE IF NOT EXISTS lifecycle_replay_claims (
                lifecycle_plan_id TEXT NOT NULL REFERENCES lifecycle_replay_ledgers(lifecycle_plan_id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                content_hash TEXT NOT NULL,
                evidence_reference_ids_json TEXT NOT NULL,
                access_context_hash TEXT NOT NULL,
                PRIMARY KEY (lifecycle_plan_id, memory_id)
            )""",
            """CREATE INDEX IF NOT EXISTS idx_lifecycle_plans_exact_replay
                ON lifecycle_plans(
                    source_id, source_unit_id, target_unit_revision_id, status
                )""",
            """CREATE INDEX IF NOT EXISTS idx_lifecycle_replay_claims_memory
                ON lifecycle_replay_claims(memory_id)""",
        ],
    ),
    (
        61,
        "Remove exact revision replay ledgers",
        [
            "DROP INDEX IF EXISTS idx_lifecycle_replay_claims_memory",
            "DROP INDEX IF EXISTS idx_lifecycle_plans_exact_replay",
            "DROP TABLE IF EXISTS lifecycle_replay_claims",
            "DROP TABLE IF EXISTS lifecycle_replay_ledgers",
        ],
    ),
    (
        62,
        "Reset legacy directional contradiction summaries",
        [],
    ),
    (
        63,
        "Record explicit semantic relation direction",
        [
            "ALTER TABLE evidence_relations ADD COLUMN relation_direction TEXT",
            "ALTER TABLE relation_run_relations ADD COLUMN relation_direction TEXT",
        ],
    ),
    (
        64,
        "Add durable progressive relation discovery work",
        [
            """CREATE TABLE IF NOT EXISTS relation_discovery_work (
                id TEXT PRIMARY KEY,
                lifecycle_plan_id TEXT NOT NULL REFERENCES lifecycle_plans(id) ON DELETE CASCADE,
                memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
                expected_content_hash TEXT NOT NULL,
                source_id TEXT NOT NULL,
                source_unit_id TEXT NOT NULL,
                source_unit_revision_id TEXT,
                doc_id TEXT NOT NULL,
                actor_user_id TEXT,
                entity_ids_json TEXT NOT NULL DEFAULT '[]',
                status TEXT NOT NULL CHECK (status IN ('pending','running','completed','failed','obsolete')),
                attempts INTEGER NOT NULL DEFAULT 0,
                lease_owner TEXT,
                lease_token TEXT,
                lease_until TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                completed_at TEXT,
                UNIQUE (lifecycle_plan_id, memory_id, expected_content_hash)
            )""",
            "CREATE INDEX IF NOT EXISTS idx_relation_discovery_work_status "
            "ON relation_discovery_work(status, lease_until, created_at)",
        ],
    ),
    (
        65,
        "Back off failed progressive relation discovery work",
        [
            "ALTER TABLE relation_discovery_work ADD COLUMN next_attempt_at TEXT",
            "DROP INDEX IF EXISTS idx_relation_discovery_work_status",
            "CREATE INDEX IF NOT EXISTS idx_relation_discovery_work_status "
            "ON relation_discovery_work(status, next_attempt_at, lease_until, created_at)",
        ],
    ),
]


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------


class _ActiveSourceSyncRunChanged(RuntimeError):
    pass


class Database:
    """Async SQLite database layer for MemForge."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self, *, run_migrations: bool = True) -> None:
        """Open the database, optionally in a non-mutating evaluation mode."""

        path = Path(self.db_path).expanduser()
        if run_migrations:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = await aiosqlite.connect(str(path))
        else:
            read_only_uri = f"{path.resolve().as_uri()}?mode=ro"
            self._db = await aiosqlite.connect(read_only_uri, uri=True)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA foreign_keys = ON")
        if not run_migrations:
            await self._db.execute("PRAGMA query_only = ON")
            return
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.executescript(SCHEMA)
        await self._run_migrations()
        await self._assert_memory_source_ids_resolved()
        await self._db.commit()

    async def _run_migrations(self) -> None:
        """Apply pending schema migrations tracked in schema_migrations."""
        applied: set[int] = set()
        async with self.db.execute("SELECT version FROM schema_migrations") as cur:
            async for row in cur:
                applied.add(row[0])

        for version, description, statements in MIGRATIONS:
            if version in applied:
                continue
            if version == 53:
                async with self.db.execute("PRAGMA table_info(memories)") as cursor:
                    memory_columns = {str(row[1]) async for row in cursor}
                if "memory_level" in memory_columns:
                    async with self.db.execute("SELECT id FROM memories WHERE memory_level = 'consolidated'") as cursor:
                        consolidated_memory_ids = [str(row[0]) async for row in cursor]
                    for memory_id in consolidated_memory_ids:
                        await self.purge_memory(memory_id)
            for sql in statements:
                try:
                    await self.db.execute(sql)
                except Exception as e:
                    message = str(e).lower()
                    legacy_scope_backfill = version == 15 and "no such column" in message and "scope" in sql.lower()
                    if "duplicate column" in message or legacy_scope_backfill:
                        logger.debug(
                            "Migration %d: expected-absent column on this DB, skipping: %s",
                            version,
                            sql,
                        )
                    else:
                        raise
            if version == 53:
                async with self.db.execute("PRAGMA table_info(memories)") as cursor:
                    memory_columns = {str(row[1]) async for row in cursor}
                for column_name in ("curation_cluster_id", "memory_level"):
                    if column_name in memory_columns:
                        await self.db.execute(f"ALTER TABLE memories DROP COLUMN {column_name}")
            if version == 26:
                await self._backfill_relation_run_snapshot_audit()
            if version in (30, 31):
                await self._rebuild_memory_metadata_fts_unlocked()
            if version == 42:
                await self._migrate_source_access_policy_unlocked()
            if version == 45:
                await self._partition_legacy_agent_session_sources_unlocked()
            if version == 62:
                await self._reset_legacy_contradiction_summaries_unlocked()
            await self.db.execute(
                "INSERT INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?)",
                (version, description, _now_iso()),
            )
            await self.db.commit()
            if version == 44:
                # SQLite ignores attempts to enable foreign keys while a
                # transaction is active. Migration 44 disables them before a
                # table rebuild, so restore the connection invariant only
                # after the migration transaction has committed.
                await self.db.execute("PRAGMA foreign_keys = ON")
                async with self.db.execute("PRAGMA foreign_keys") as cursor:
                    foreign_keys = await cursor.fetchone()
                if not foreign_keys or int(foreign_keys[0]) != 1:
                    raise RuntimeError("failed to restore SQLite foreign key enforcement")
            logger.info("Applied migration %d: %s", version, description)
        await self._assert_memory_source_ids_resolved()

    async def _reset_legacy_contradiction_summaries_unlocked(self) -> None:
        """Discard directional summary residue without assuming every legacy cache column exists."""
        await self.db.execute("DELETE FROM memory_contradictions")
        async with self.db.execute("PRAGMA table_info(memories)") as cursor:
            columns = {str(row[1]) async for row in cursor}
        if "contradiction_count" in columns:
            await self.db.execute("UPDATE memories SET contradiction_count = 0 WHERE contradiction_count <> 0")

    async def _migrate_source_access_policy_unlocked(self) -> None:
        """Materialize legacy Source access without guessing.

        Existing Memory visibility is the effective pre-migration policy. An
        empty configured Source preserves the old workspace writer behavior.
        Coding-session Sources are private here and are partitioned by owner in
        migration 45. Missing ownership or mixed historical access stops the
        migration so operators can repair the data explicitly.
        """

        async with self.db.execute("SELECT * FROM sources ORDER BY id") as cursor:
            sources = [dict(row) async for row in cursor]

        for source in sources:
            source_id = str(source["id"])
            explicit_owner = str(source.get("owner_user_id") or "").strip()
            explicit_policy = str(source.get("access_policy") or "").strip()
            explicit_state = str(source.get("access_state") or "").strip()
            if explicit_policy or explicit_owner or explicit_state:
                if (
                    explicit_policy not in {"private", "workspace"}
                    or not explicit_owner
                    or explicit_state not in {"active", "changing", "orphaned_private"}
                ):
                    raise RuntimeError(f"source access metadata is incomplete or invalid: {source_id}")
                continue

            if str(source.get("type") or "") == "agent_session" and not (
                str(source.get("created_by_user_id") or "").strip()
                or str(source.get("execution_owner_user_id") or "").strip()
            ):
                # Pre-access coding Sources were shared per client and may not
                # have Source-level provenance. Migration 45 derives their
                # owners from receipt and Memory provenance before partitioning
                # them into private per-owner Sources.
                continue

            async with self.db.execute(
                """SELECT DISTINCT m.visibility, m.owner_user_id
                     FROM memories m
                     JOIN memory_sources ms ON ms.memory_id = m.id
                    WHERE ms.source_id = ? AND m.status = 'active'""",
                (source_id,),
            ) as cursor:
                access_rows = [dict(row) async for row in cursor]
            try:
                policy_value, owner_user_id = infer_legacy_source_access(
                    source_id=source_id,
                    source_type=str(source.get("type") or ""),
                    provenance_owner_user_id=(
                        source.get("created_by_user_id") or source.get("execution_owner_user_id")
                    ),
                    memory_access=((str(row.get("visibility") or ""), row.get("owner_user_id")) for row in access_rows),
                )
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
            policy = policy_value.value

            await self.db.execute(
                """UPDATE sources
                      SET owner_user_id = ?, access_policy = ?, access_state = 'active'
                    WHERE id = ?""",
                (owner_user_id, policy, source_id),
            )

    async def _partition_legacy_agent_session_sources_unlocked(self) -> None:
        """Split pre-access coding Sources into one private Source per owner.

        This is an explicit one-time migration. Runtime reads never infer an owner
        or fall back to a legacy shared Source.
        """

        async with self.db.execute("SELECT * FROM sources WHERE type = 'agent_session' ORDER BY id") as cursor:
            legacy_sources = [dict(row) async for row in cursor]

        for source in legacy_sources:
            source_id = str(source["id"])
            source_owner = str(source.get("owner_user_id") or "").strip()
            config = json.loads(source.get("config") or "{}")
            client = str(config.get("client") or "").strip()
            if not client:
                async with self.db.execute(
                    "SELECT client FROM documents WHERE source = ? AND client IS NOT NULL LIMIT 1",
                    (source_id,),
                ) as cursor:
                    client_row = await cursor.fetchone()
                client = str(client_row["client"] if client_row else "").strip()
            if not client:
                client = _legacy_agent_session_client(source_id)
            if not client:
                raise RuntimeError(f"cannot partition legacy agent-session source without client: {source_id}")
            if source_owner and source_id == _agent_session_source_id_for_owner(client, source_owner):
                continue

            async with self.db.execute(
                """SELECT d.doc_id,
                          NULLIF(json_extract(r.metadata, '$.user_id'), '') AS receipt_owner_user_id
                   FROM documents d
                   LEFT JOIN agent_session_receipts r ON r.doc_id = d.doc_id
                   WHERE d.source = ?""",
                (source_id,),
            ) as cursor:
                document_rows = [dict(row) async for row in cursor]

            document_owners: dict[str, str] = {}
            for document in document_rows:
                doc_id = str(document["doc_id"])
                owner = str(document.get("receipt_owner_user_id") or "").strip()
                if not owner:
                    async with self.db.execute(
                        """SELECT DISTINCT m.owner_user_id
                           FROM memory_sources ms
                           JOIN memories m ON m.id = ms.memory_id
                           WHERE ms.doc_id = ? AND m.owner_user_id IS NOT NULL""",
                        (doc_id,),
                    ) as cursor:
                        memory_owners = {
                            str(row["owner_user_id"] or "").strip()
                            async for row in cursor
                            if str(row["owner_user_id"] or "").strip()
                        }
                    if len(memory_owners) > 1:
                        raise RuntimeError(f"agent-session document has multiple owners during migration: {doc_id}")
                    owner = next(iter(memory_owners), source_owner)
                if not owner:
                    raise RuntimeError(f"cannot partition legacy agent-session document without an owner: {doc_id}")
                document_owners[doc_id] = owner

            owners = {owner for owner in document_owners.values() if owner}
            async with self.db.execute(
                "SELECT DISTINCT owner_user_id FROM agent_concepts WHERE source_id = ?",
                (source_id,),
            ) as cursor:
                async for row in cursor:
                    owner = str(row["owner_user_id"]).strip()
                    if owner:
                        owners.add(owner)
            async with self.db.execute(
                "SELECT DISTINCT owner_user_id FROM evidence_units WHERE source_id = ? AND owner_user_id IS NOT NULL",
                (source_id,),
            ) as cursor:
                async for row in cursor:
                    owner = str(row["owner_user_id"]).strip()
                    if owner:
                        owners.add(owner)
            if source_owner:
                owners.add(source_owner)
            if not owners:
                raise RuntimeError(f"cannot partition legacy agent-session source without owner evidence: {source_id}")

            for owner in sorted(owners):
                target_source_id = _agent_session_source_id_for_owner(client, owner)
                target_config = dict(config)
                target_config["client"] = client
                documents_dir = str(target_config.get("documents_dir") or "").strip()
                if documents_dir:
                    target_config["documents_dir"] = str(Path(documents_dir).parent / target_source_id)
                await self.db.execute(
                    """INSERT INTO sources (
                           id, type, name, config, status, last_sync, doc_count,
                           project_binding, created_by_user_id, owner_user_id,
                           access_policy, access_state, execution_owner_user_id,
                           sync_schedule_enabled, sync_schedule_interval_minutes,
                           sync_schedule_next_at, sync_schedule_updated_at, created_at
                       ) VALUES (?, 'agent_session', ?, ?, ?, ?, ?, ?, ?, ?,
                                 'private', 'active', ?, ?, ?, ?, ?, ?)
                       ON CONFLICT(id) DO NOTHING""",
                    (
                        target_source_id,
                        source["name"],
                        json.dumps(target_config, sort_keys=True),
                        source["status"],
                        source.get("last_sync"),
                        source.get("doc_count", 0),
                        source.get("project_binding"),
                        owner,
                        owner,
                        owner,
                        source.get("sync_schedule_enabled", 0),
                        source.get("sync_schedule_interval_minutes", 1440),
                        source.get("sync_schedule_next_at"),
                        source.get("sync_schedule_updated_at"),
                        source.get("created_at") or _now_iso(),
                    ),
                )
                owned_doc_ids = [doc_id for doc_id, doc_owner in document_owners.items() if doc_owner == owner]
                for doc_id in owned_doc_ids:
                    await self.db.execute(
                        "UPDATE documents SET source = ? WHERE doc_id = ?",
                        (target_source_id, doc_id),
                    )
                    await self.db.execute(
                        "UPDATE agent_session_receipts SET source_id = ? WHERE doc_id = ?",
                        (target_source_id, doc_id),
                    )
                    await self.db.execute(
                        "UPDATE memory_sources SET source_id = ? WHERE doc_id = ?",
                        (target_source_id, doc_id),
                    )
                await self.db.execute(
                    """UPDATE agent_concepts
                       SET source_id = ?, visibility = 'private'
                       WHERE source_id = ? AND owner_user_id = ?""",
                    (target_source_id, source_id, owner),
                )
                await self.db.execute(
                    """UPDATE evidence_units
                       SET source_id = ?, visibility = 'private', owner_user_id = ?,
                           access_context_hash = NULL
                       WHERE source_id = ? AND owner_user_id = ?""",
                    (target_source_id, owner, source_id, owner),
                )
                await self.db.execute(
                    """UPDATE local_agent_jobs SET source_id = ?
                       WHERE source_id = ? AND execution_owner_user_id = ?""",
                    (target_source_id, source_id, owner),
                )
                await self.db.execute(
                    """INSERT OR IGNORE INTO source_subscriptions
                       SELECT ?, user_id, enabled, created_at, updated_at
                       FROM source_subscriptions WHERE source_id = ? AND user_id = ?""",
                    (target_source_id, source_id, owner),
                )
                await self.db.execute(
                    """INSERT OR IGNORE INTO source_list_pins
                       SELECT ?, user_id, pinned_at
                       FROM source_list_pins WHERE source_id = ? AND user_id = ?""",
                    (target_source_id, source_id, owner),
                )
                await self.db.execute(
                    """UPDATE memories
                       SET visibility = 'private', owner_user_id = ?
                       WHERE id IN (
                           SELECT ms.memory_id FROM memory_sources ms
                           WHERE ms.source_id = ?
                       )
                         AND NOT EXISTS (
                           SELECT 1
                           FROM memory_sources other_ms
                           JOIN sources other_s ON other_s.id = other_ms.source_id
                           WHERE other_ms.memory_id = memories.id
                             AND (
                               other_s.access_policy = 'workspace'
                               OR other_s.owner_user_id <> ?
                             )
                         )""",
                    (owner, target_source_id, owner),
                )

            await self.db.execute(
                "DELETE FROM source_subscriptions WHERE source_id = ?",
                (source_id,),
            )
            await self.db.execute(
                "DELETE FROM source_list_pins WHERE source_id = ?",
                (source_id,),
            )
            async with self.db.execute(
                """SELECT
                       (SELECT COUNT(*) FROM documents WHERE source = ?) +
                       (SELECT COUNT(*) FROM memory_sources WHERE source_id = ?) +
                       (SELECT COUNT(*) FROM evidence_units WHERE source_id = ?) +
                       (SELECT COUNT(*) FROM agent_concepts WHERE source_id = ?) +
                       (SELECT COUNT(*) FROM local_agent_jobs WHERE source_id = ?) AS remaining""",
                (source_id, source_id, source_id, source_id, source_id),
            ) as cursor:
                remaining_row = await cursor.fetchone()
            if int(remaining_row["remaining"] if remaining_row else 0) != 0:
                raise RuntimeError(f"legacy agent-session source was not fully partitioned: {source_id}")
            await self.db.execute("DELETE FROM sources WHERE id = ?", (source_id,))

    async def _assert_memory_source_ids_resolved(self) -> None:
        """Fail startup when exact source-id provenance cannot be trusted."""

        async with self.db.execute(
            "SELECT COUNT(*) AS total FROM memory_sources WHERE source_id IS NULL OR source_id = ''"
        ) as cursor:
            row = await cursor.fetchone()
        unresolved = int(row["total"] if row else 0)
        if unresolved:
            raise RuntimeError(
                "memory_sources contains rows without source_id after migration: "
                f"{unresolved}. Repair source provenance before starting MemForge."
            )

    async def _backfill_relation_run_snapshot_audit(self) -> None:
        """Backfill immutable retry snapshot hashes for pre-hardening relation runs."""

        async with self.db.execute("SELECT id, audit_json FROM relation_runs") as cursor:
            rows = [row async for row in cursor]
        for row in rows:
            run_id = row["id"]
            audit = json.loads(row["audit_json"] or "{}")
            candidates = await self._get_relation_candidates_unlocked(run_id)
            relations = await self._get_relation_run_relations_unlocked(run_id)
            snapshot_audit = relation_bundle_snapshot_audit(candidates=candidates, relations=relations)
            changed = False
            for key, value in snapshot_audit.items():
                existing = audit.get(key)
                if existing is not None and existing != value:
                    raise RuntimeError(
                        "relation_run_id collision for "
                        f"{run_id}: committed audit does not match stored relation snapshot ({key})"
                    )
                if existing is None:
                    audit[key] = value
                    changed = True
            if changed:
                await self.db.execute(
                    "UPDATE relation_runs SET audit_json = ? WHERE id = ?",
                    (json.dumps(audit, sort_keys=True), run_id),
                )

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    @property
    def db(self) -> aiosqlite.Connection:
        assert self._db is not None, "Database not connected"
        return self._db

    # ==================================================================
    # Documents
    # ==================================================================

    async def _assert_document_source_writable_unlocked(
        self,
        source_id: str,
        *,
        require_configured_source: bool,
    ) -> None:
        # A no-op write is intentional: it acquires SQLite's cross-process
        # writer fence before source status is read and holds it until the
        # caller commits the document mutation. A plain SELECT here would let
        # another process delete the Source between validation and INSERT.
        cursor = await self.db.execute(
            "UPDATE sources SET status = status WHERE id = ?",
            (source_id,),
        )
        source_exists = bool(cursor.rowcount)
        async with self.db.execute(
            "SELECT status FROM sources WHERE id = ?",
            (source_id,),
        ) as cursor:
            source_row = await cursor.fetchone()
        if not source_exists or source_row is None:
            if require_configured_source:
                await self.db.rollback()
                raise ValueError(f"Source does not exist: {source_id}")
            return
        if source_row["status"] != "active":
            await self.db.rollback()
            if source_row["status"] == "deleting":
                raise ValueError(f"Source is being deleted: {source_id}")
            raise ValueError(f"Source is not active: {source_id}")

    async def upsert_document(
        self,
        doc: DocumentRecord,
        *,
        require_configured_source: bool = False,
        source_activity: SourceActivityLease | None = None,
    ) -> None:
        async with self._write_lock:
            try:
                await self._assert_document_source_writable_unlocked(
                    doc.source,
                    require_configured_source=require_configured_source,
                )
                await self._assert_source_activity_fence_unlocked(
                    doc.source,
                    source_activity,
                )
                await self.db.execute(
                    """INSERT INTO documents (
                    doc_id, source, source_url, title, space_or_project,
                    author, last_modified, labels, version, content_hash,
                    token_count, raw_content_uri, raw_content_type,
                    normalized_content_uri, pdf_content_uri, last_synced,
                    client, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source=excluded.source, source_url=excluded.source_url,
                    title=excluded.title, space_or_project=excluded.space_or_project,
                    author=excluded.author, last_modified=excluded.last_modified,
                    labels=excluded.labels, version=excluded.version,
                    content_hash=excluded.content_hash, token_count=excluded.token_count,
                    raw_content_uri=excluded.raw_content_uri,
                    raw_content_type=excluded.raw_content_type,
                    normalized_content_uri=excluded.normalized_content_uri,
                    pdf_content_uri=excluded.pdf_content_uri,
                    last_synced=excluded.last_synced,
                    client=COALESCE(excluded.client, documents.client),
                    updated_at=excluded.updated_at""",
                    (
                        doc.doc_id,
                        doc.source,
                        doc.source_url,
                        doc.title,
                        doc.space_or_project,
                        doc.author,
                        doc.last_modified.isoformat(),
                        json.dumps(doc.labels),
                        doc.version,
                        doc.content_hash,
                        doc.token_count,
                        doc.raw_content_uri,
                        doc.raw_content_type,
                        doc.normalized_content_uri,
                        doc.pdf_content_uri,
                        doc.last_synced.isoformat(),
                        doc.client,
                        _now_iso(),
                    ),
                )
                await self._refresh_metadata_fts_for_doc_unlocked(doc.doc_id)
                await self._assert_source_activity_fence_unlocked(
                    doc.source,
                    source_activity,
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        async with self.db.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return self._row_to_document(row)

    async def restore_document_snapshot(
        self,
        doc: DocumentRecord,
        *,
        require_configured_source: bool = False,
    ) -> None:
        """Restore one document row from a captured snapshot."""
        async with self._write_lock:
            await self._assert_document_source_writable_unlocked(
                doc.source,
                require_configured_source=require_configured_source,
            )
            await self.db.execute(
                """INSERT INTO documents (
                    doc_id, source, source_url, title, space_or_project, author,
                    last_modified, labels, version, content_hash, token_count,
                    raw_content_uri, raw_content_type, normalized_content_uri,
                    pdf_content_uri, last_synced, client, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    source=excluded.source, source_url=excluded.source_url,
                    title=excluded.title, space_or_project=excluded.space_or_project,
                    author=excluded.author, last_modified=excluded.last_modified,
                    labels=excluded.labels, version=excluded.version,
                    content_hash=excluded.content_hash, token_count=excluded.token_count,
                    raw_content_uri=excluded.raw_content_uri,
                    raw_content_type=excluded.raw_content_type,
                    normalized_content_uri=excluded.normalized_content_uri,
                    pdf_content_uri=excluded.pdf_content_uri,
                    last_synced=excluded.last_synced,
                    client=COALESCE(excluded.client, documents.client),
                    created_at=COALESCE(excluded.created_at, documents.created_at),
                    updated_at=excluded.updated_at""",
                (
                    doc.doc_id,
                    doc.source,
                    doc.source_url,
                    doc.title,
                    doc.space_or_project,
                    doc.author,
                    doc.last_modified.isoformat(),
                    json.dumps(doc.labels),
                    doc.version,
                    doc.content_hash,
                    doc.token_count,
                    doc.raw_content_uri,
                    doc.raw_content_type,
                    doc.normalized_content_uri,
                    doc.pdf_content_uri,
                    doc.last_synced.isoformat(),
                    doc.client,
                    doc.created_at.isoformat() if doc.created_at else None,
                    doc.updated_at.isoformat() if doc.updated_at else None,
                ),
            )
            await self.db.commit()

    async def get_document_side_table_snapshots(
        self,
        doc_ids: list[str],
        *,
        source_id: str | None = None,
    ) -> dict[str, list[dict]]:
        """Capture document-adjacent rows removed by document/source cascades."""
        unique_doc_ids = list(dict.fromkeys(doc_ids))
        snapshots = {
            "document_metadata": [],
            "document_relationships": [],
            "changelog": [],
            "agent_session_receipts": [],
            "sync_state": [],
            "sync_history": [],
        }

        async def fetch_rows(sql: str, params: tuple) -> list[dict]:
            rows: list[dict] = []
            async with self.db.execute(sql, params) as cursor:
                async for row in cursor:
                    rows.append(dict(row))
            return rows

        if unique_doc_ids:
            placeholders = ",".join("?" for _ in unique_doc_ids)
            params = tuple(unique_doc_ids)
            snapshots["document_metadata"] = await fetch_rows(
                f"SELECT * FROM document_metadata WHERE doc_id IN ({placeholders})",
                params,
            )
            snapshots["document_relationships"] = await fetch_rows(
                f"""SELECT * FROM document_relationships
                    WHERE source_doc_id IN ({placeholders})
                       OR target_doc_id IN ({placeholders})""",
                params + params,
            )
            snapshots["changelog"] = await fetch_rows(
                f"SELECT * FROM changelog WHERE doc_id IN ({placeholders})",
                params,
            )
            if source_id:
                snapshots["agent_session_receipts"] = await fetch_rows(
                    f"""SELECT * FROM agent_session_receipts
                        WHERE source_id = ? OR doc_id IN ({placeholders})""",
                    (source_id, *unique_doc_ids),
                )
            else:
                snapshots["agent_session_receipts"] = await fetch_rows(
                    f"SELECT * FROM agent_session_receipts WHERE doc_id IN ({placeholders})",
                    params,
                )
        elif source_id:
            snapshots["agent_session_receipts"] = await fetch_rows(
                "SELECT * FROM agent_session_receipts WHERE source_id = ?",
                (source_id,),
            )

        if source_id:
            snapshots["sync_state"] = await fetch_rows(
                "SELECT * FROM sync_state WHERE source = ?",
                (source_id,),
            )
            snapshots["sync_history"] = await fetch_rows(
                "SELECT * FROM sync_history WHERE source = ?",
                (source_id,),
            )

        return snapshots

    async def restore_document_side_table_snapshots(self, snapshots: dict[str, list[dict]]) -> None:
        """Restore document-adjacent rows captured before a cascade delete."""
        async with self._write_lock:
            for table in (
                "document_metadata",
                "document_relationships",
                "changelog",
                "agent_session_receipts",
                "sync_state",
                "sync_history",
            ):
                for row in snapshots.get(table, []):
                    columns = list(row.keys())
                    placeholders = ",".join("?" for _ in columns)
                    assignments = ",".join(f"{column}=excluded.{column}" for column in columns)
                    await self.db.execute(
                        f"""INSERT INTO {table} ({",".join(columns)})
                            VALUES ({placeholders})
                            ON CONFLICT DO UPDATE SET {assignments}""",
                        tuple(row[column] for column in columns),
                    )
            await self.db.commit()

    async def get_content_hash(self, doc_id: str) -> str | None:
        async with self.db.execute("SELECT content_hash FROM documents WHERE doc_id = ?", (doc_id,)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

    async def list_documents(
        self,
        source: str | None = None,
        search: str | None = None,
        limit: int = 100,
    ) -> list[DocumentRecord]:
        query = "SELECT * FROM documents WHERE 1=1"
        params: list = []
        if source:
            query += " AND source = ?"
            params.append(source)
        if search:
            query += " AND (title LIKE ? OR doc_id LIKE ?)"
            like = f"%{search}%"
            params.extend([like, like])
        query += " ORDER BY last_modified DESC LIMIT ?"
        params.append(limit)

        results: list[DocumentRecord] = []
        async with self.db.execute(query, params) as cursor:
            async for row in cursor:
                results.append(self._row_to_document(row))
        return results

    async def insert_local_agent_job(
        self,
        *,
        job_id: str,
        source_id: str,
        source_type: str,
        operation: str,
        payload: Mapping[str, Any],
        created_by_user_id: str,
        execution_owner_user_id: str,
        workspace_id: str = "default",
        now: datetime | None = None,
    ) -> bool:
        now_iso = _utc_iso(now)
        async with self._write_lock:
            cursor = await self.db.execute(
                """INSERT OR IGNORE INTO local_agent_jobs (
                    job_id, workspace_id, source_id, source_type, operation,
                    payload_json, created_by_user_id, execution_owner_user_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    workspace_id,
                    source_id,
                    source_type,
                    operation,
                    json.dumps(dict(payload), sort_keys=True),
                    created_by_user_id,
                    execution_owner_user_id,
                    now_iso,
                    now_iso,
                ),
            )
            await self.db.commit()
        return bool(cursor.rowcount)

    async def enqueue_local_agent_job(
        self,
        *,
        job_id: str,
        source_id: str,
        source_type: str,
        operation: str,
        payload: Mapping[str, Any],
        created_by_user_id: str,
        execution_owner_user_id: str,
        workspace_id: str = "default",
    ) -> tuple[str, bool]:
        now_iso = _utc_iso(None)
        requested_force = bool(payload.get("force_full_sync"))
        async with self._write_lock:
            try:
                source_lock = await self.db.execute(
                    "UPDATE sources SET status = status WHERE id = ?",
                    (source_id,),
                )
                if source_lock.rowcount != 1:
                    raise ValueError(f"Source not found: {source_id}")
                async with self.db.execute(
                    """SELECT id FROM lifecycle_backfill_jobs
                       WHERE source_id = ? AND status IN ('queued', 'running')
                       ORDER BY created_at LIMIT 1""",
                    (source_id,),
                ) as cursor:
                    lifecycle_job = await cursor.fetchone()
                if lifecycle_job is not None:
                    raise ValueError(f"source lifecycle maintenance active: {lifecycle_job['id']}")
            except Exception:
                await self.db.rollback()
                raise
            async with self.db.execute(
                """SELECT * FROM local_agent_jobs
                   WHERE workspace_id = ? AND source_id = ? AND operation = ?
                     AND execution_owner_user_id = ? AND status IN ('queued', 'leased')
                   ORDER BY created_at LIMIT 1""",
                (workspace_id, source_id, operation, execution_owner_user_id),
            ) as cursor:
                existing = await cursor.fetchone()
            if existing is not None:
                existing_payload = json.loads(existing["payload_json"] or "{}")
                if existing["status"] == "queued":
                    replacement = dict(payload)
                    if bool(existing_payload.get("force_full_sync")):
                        replacement["force_full_sync"] = True
                    if replacement != existing_payload:
                        await self.db.execute(
                            """UPDATE local_agent_jobs SET payload_json = ?, updated_at = ?
                               WHERE job_id = ? AND status IN ('queued', 'leased')""",
                            (
                                json.dumps(replacement, sort_keys=True),
                                now_iso,
                                existing["job_id"],
                            ),
                        )
                        await self.db.commit()
                    return str(existing["job_id"]), False
                same_revision = existing_payload.get("source_config_revision") == payload.get("source_config_revision")
                if same_revision and (not requested_force or bool(existing_payload.get("force_full_sync"))):
                    return str(existing["job_id"]), False
            await self.db.execute(
                """INSERT INTO local_agent_jobs (
                    job_id, workspace_id, source_id, source_type, operation,
                    payload_json, created_by_user_id, execution_owner_user_id,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    job_id,
                    workspace_id,
                    source_id,
                    source_type,
                    operation,
                    json.dumps(dict(payload), sort_keys=True),
                    created_by_user_id,
                    execution_owner_user_id,
                    now_iso,
                    now_iso,
                ),
            )
            await self.db.commit()
        return job_id, True

    async def get_local_agent_job(self, job_id: str) -> dict[str, Any] | None:
        async with self.db.execute("SELECT * FROM local_agent_jobs WHERE job_id = ?", (job_id,)) as cursor:
            row = await cursor.fetchone()
        return _local_agent_job_from_row(row) if row else None

    async def list_current_local_agent_jobs(
        self,
        *,
        workspace_id: str,
        user_id: str,
    ) -> list[dict[str, Any]]:
        async with self.db.execute(
            """SELECT * FROM local_agent_jobs
               WHERE workspace_id = ? AND execution_owner_user_id = ? AND source_id <> ''
               ORDER BY CASE WHEN status IN ('queued', 'leased') THEN 0 ELSE 1 END,
                        created_at DESC LIMIT 200""",
            (workspace_id, user_id),
        ) as cursor:
            rows = await cursor.fetchall()
        latest_by_source: dict[str, dict[str, Any]] = {}
        for row in rows:
            job = _local_agent_job_from_row(row)
            latest_by_source.setdefault(str(job["source_id"]), job)
        return list(latest_by_source.values())

    async def lease_local_agent_jobs(
        self,
        *,
        user_id: str,
        limit: int,
        lease_seconds: int,
        now: datetime | None = None,
    ) -> list[dict[str, Any]]:
        lease_time = now or datetime.now(timezone.utc)
        now_iso = _utc_iso(lease_time)
        leased_until = _utc_iso(lease_time + timedelta(seconds=lease_seconds))
        leased_ids: list[str] = []
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO local_agent_heartbeats (user_id, last_seen_at)
                   VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET
                   last_seen_at = excluded.last_seen_at""",
                (user_id, now_iso),
            )
            async with self.db.execute(
                """SELECT j.job_id FROM local_agent_jobs j
                   WHERE j.execution_owner_user_id = ?
                     AND (j.status = 'queued' OR
                          (j.status = 'leased' AND j.leased_until <= ?))
                     AND NOT EXISTS (
                         SELECT 1 FROM local_agent_jobs active
                         WHERE active.workspace_id = j.workspace_id
                           AND active.source_id = j.source_id
                           AND active.operation = j.operation
                           AND active.execution_owner_user_id = j.execution_owner_user_id
                           AND active.status = 'leased'
                           AND active.leased_until > ?
                           AND active.job_id <> j.job_id
                     )
                   ORDER BY j.created_at, j.job_id LIMIT ?""",
                (user_id, now_iso, now_iso, limit),
            ) as cursor:
                rows = await cursor.fetchall()
            for row in rows:
                cursor = await self.db.execute(
                    """UPDATE local_agent_jobs SET status = 'leased',
                       lease_owner_user_id = ?, leased_until = ?,
                       attempt_count = attempt_count + 1, result_json = NULL,
                       last_error = NULL, finished_at = NULL, updated_at = ?
                       WHERE job_id = ? AND execution_owner_user_id = ?
                         AND (status = 'queued' OR
                              (status = 'leased' AND leased_until <= ?))
                         AND NOT EXISTS (
                             SELECT 1 FROM local_agent_jobs active
                             WHERE active.workspace_id = local_agent_jobs.workspace_id
                               AND active.source_id = local_agent_jobs.source_id
                               AND active.operation = local_agent_jobs.operation
                               AND active.execution_owner_user_id = local_agent_jobs.execution_owner_user_id
                               AND active.status = 'leased'
                               AND active.leased_until > ?
                               AND active.job_id <> local_agent_jobs.job_id
                         )""",
                    (
                        user_id,
                        leased_until,
                        now_iso,
                        row["job_id"],
                        user_id,
                        now_iso,
                        now_iso,
                    ),
                )
                if cursor.rowcount:
                    leased_ids.append(str(row["job_id"]))
            await self.db.commit()
        jobs = [await self.get_local_agent_job(job_id) for job_id in leased_ids]
        return [job for job in jobs if job is not None]

    async def heartbeat_local_agent_job(
        self,
        *,
        job_id: str,
        user_id: str,
        attempt_count: int,
        lease_seconds: int,
        progress: Mapping[str, Any] | None = None,
        now: datetime | None = None,
    ) -> bool:
        heartbeat_at = now or datetime.now(timezone.utc)
        now_iso = _utc_iso(heartbeat_at)
        leased_until = _utc_iso(heartbeat_at + timedelta(seconds=lease_seconds))
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO local_agent_heartbeats (user_id, last_seen_at)
                   VALUES (?, ?) ON CONFLICT(user_id) DO UPDATE SET
                   last_seen_at = excluded.last_seen_at""",
                (user_id, now_iso),
            )
            result_json = None
            if progress is not None:
                result_json = json.dumps(
                    {"progress": normalize_sync_progress_snapshot(progress)},
                    sort_keys=True,
                    separators=(",", ":"),
                )
            cursor = await self.db.execute(
                """UPDATE local_agent_jobs SET leased_until = ?,
                       result_json = COALESCE(?, result_json), updated_at = ?
                   WHERE job_id = ? AND status = 'leased'
                     AND lease_owner_user_id = ? AND attempt_count = ?""",
                (leased_until, result_json, now_iso, job_id, user_id, attempt_count),
            )
            await self.db.commit()
        return bool(cursor.rowcount)

    async def complete_local_agent_job(
        self,
        *,
        job_id: str,
        user_id: str,
        attempt_count: int,
        status: str,
        result: Mapping[str, Any],
        error: str | None,
        retryable: bool = False,
        now: datetime | None = None,
    ) -> bool:
        now_iso = _utc_iso(now)
        stored_status = local_agent_completion_status(
            status,
            retryable=retryable,
            attempt_count=attempt_count,
        )
        should_retry = stored_status == "queued"
        finished_at = None if should_retry else now_iso
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE local_agent_jobs SET status = ?, result_json = ?,
                   last_error = ?, lease_owner_user_id = NULL,
                   leased_until = NULL, finished_at = ?, updated_at = ?
                   WHERE job_id = ? AND status = 'leased'
                     AND lease_owner_user_id = ? AND attempt_count = ?""",
                (
                    stored_status,
                    json.dumps(dict(result), sort_keys=True),
                    error,
                    finished_at,
                    now_iso,
                    job_id,
                    user_id,
                    attempt_count,
                ),
            )
            await self.db.commit()
        return bool(cursor.rowcount)

    async def get_local_agent_heartbeat(self, user_id: str) -> str | None:
        async with self.db.execute(
            "SELECT last_seen_at FROM local_agent_heartbeats WHERE user_id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return str(row["last_seen_at"]) if row else None

    async def enqueue_due_local_agent_jobs(
        self,
        *,
        now: datetime | None = None,
        limit: int = 50,
        workspace_id: str = "default",
    ) -> int:
        claim_time = now or datetime.now(timezone.utc)
        due_at = _utc_iso(claim_time)
        enqueued = 0
        async with self._write_lock:
            async with self.db.execute(
                """SELECT * FROM sources s WHERE status = 'active'
                   AND sync_schedule_enabled = 1
                   AND sync_schedule_next_at IS NOT NULL
                   AND sync_schedule_next_at <= ?
                   AND NOT EXISTS (
                       SELECT 1 FROM lifecycle_backfill_jobs j
                       WHERE j.source_id = s.id
                         AND j.status IN ('queued', 'running')
                   )
                   ORDER BY s.sync_schedule_next_at, s.created_at LIMIT ?""",
                (due_at, limit),
            ) as cursor:
                rows = [dict(row) async for row in cursor]
            for row in rows:
                operation = local_agent_sync_operation(row["type"], row["config"])
                if operation is None:
                    continue
                owner = str(row.get("execution_owner_user_id") or "").strip()
                expected_due = str(row["sync_schedule_next_at"])
                interval = int(
                    row.get("sync_schedule_interval_minutes") or SOURCE_SYNC_SCHEDULE_DEFAULT_INTERVAL_MINUTES
                )
                if not owner:
                    await self.db.execute(
                        """UPDATE sources SET sync_schedule_next_at = ?,
                           sync_schedule_updated_at = ?
                           WHERE id = ? AND status = 'active'
                             AND sync_schedule_enabled = 1
                             AND sync_schedule_next_at = ?""",
                        (
                            _utc_iso(claim_time + timedelta(minutes=interval)),
                            due_at,
                            row["id"],
                            expected_due,
                        ),
                    )
                    continue
                job_id = (
                    "laj-schedule-"
                    + uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"memforge:{workspace_id}:{row['id']}:{expected_due}",
                    ).hex
                )
                cursor = await self.db.execute(
                    """UPDATE sources SET sync_schedule_next_at = ?,
                       sync_schedule_updated_at = ?
                       WHERE id = ? AND status = 'active'
                         AND sync_schedule_enabled = 1
                         AND sync_schedule_next_at = ?""",
                    (
                        _utc_iso(claim_time + timedelta(minutes=interval)),
                        due_at,
                        row["id"],
                        expected_due,
                    ),
                )
                if cursor.rowcount:
                    await self.db.execute(
                        """INSERT OR IGNORE INTO local_agent_jobs (
                            job_id, workspace_id, source_id, source_type, operation,
                            payload_json, created_by_user_id, execution_owner_user_id,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            job_id,
                            workspace_id,
                            row["id"],
                            row["type"],
                            operation,
                            json.dumps(local_agent_sync_job_payload(row), sort_keys=True),
                            owner,
                            owner,
                            due_at,
                            due_at,
                        ),
                    )
                    enqueued += 1
            await self.db.commit()
        return enqueued

    async def count_documents(self, source: str | None = None) -> int:
        """Return the number of indexed documents, optionally scoped to a source."""
        if source:
            query = "SELECT COUNT(*) FROM documents WHERE source = ?"
            params: tuple[str, ...] = (source,)
        else:
            query = "SELECT COUNT(*) FROM documents"
            params = ()

        async with self.db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return int(row[0]) if row else 0

    async def list_indexed_doc_ids(self, source_id: str) -> set[str]:
        """Return the complete current document identity set for one source."""
        async with self.db.execute(
            "SELECT doc_id FROM documents WHERE source = ?",
            (source_id,),
        ) as cursor:
            return {str(row[0]) async for row in cursor}

    async def list_indexed_document_versions(self, source_id: str) -> dict[str, str]:
        """Return the complete current document/version identity for one source."""
        async with self.db.execute(
            "SELECT doc_id, version FROM documents WHERE source = ?",
            (source_id,),
        ) as cursor:
            return {str(row[0]): str(row[1]) async for row in cursor}

    async def delete_document(self, doc_id: str) -> list[str]:
        """Delete a document and manually cascade to related tables.

        Returns memory IDs retired because this document was their last valid
        source support.
        """
        async with self._write_lock:
            try:
                memory_ids: list[str] = []
                async with self.db.execute(
                    "SELECT source, raw_content_uri, normalized_content_uri, pdf_content_uri "
                    "FROM documents WHERE doc_id = ?",
                    (doc_id,),
                ) as cursor:
                    document_row = await cursor.fetchone()
                async with self.db.execute(
                    "SELECT 1 FROM memory_support_assertions msa "
                    "JOIN evidence_references er ON er.id = msa.evidence_reference_id "
                    "JOIN evidence_units eu ON eu.id = er.evidence_unit_id "
                    "WHERE msa.active = 1 AND eu.doc_id = ? LIMIT 1",
                    (doc_id,),
                ) as cursor:
                    active_projected_support = await cursor.fetchone()
                if active_projected_support is not None:
                    raise ValueError(
                        "active projected support remains; apply a Lifecycle Plan before deleting the document"
                    )
                if document_row is not None:
                    source_id = str(document_row["source"])
                    for artifact_uri in dict.fromkeys(
                        str(uri)
                        for uri in (
                            document_row["raw_content_uri"],
                            document_row["normalized_content_uri"],
                            document_row["pdf_content_uri"],
                        )
                        if uri
                    ):
                        await self.db.execute(
                            "INSERT OR IGNORE INTO source_artifact_cleanup_tasks "
                            "(task_id, source_id, artifact_uri) VALUES (?, ?, ?)",
                            (
                                source_artifact_cleanup_task_id(source_id, artifact_uri),
                                source_id,
                                artifact_uri,
                            ),
                        )
                async with self.db.execute(
                    "SELECT memory_id FROM memory_sources WHERE doc_id = ?",
                    (doc_id,),
                ) as cursor:
                    async for row in cursor:
                        memory_ids.append(row[0])

                await self.db.execute("DELETE FROM memory_search_metadata_fts WHERE doc_id = ?", (doc_id,))
                await self.db.execute("DELETE FROM memory_search_metadata_alias_fts WHERE doc_id = ?", (doc_id,))
                await self.db.execute("DELETE FROM memory_search_metadata_trigram WHERE doc_id = ?", (doc_id,))
                await self.db.execute("DELETE FROM memory_sources WHERE doc_id = ?", (doc_id,))
                await self._delete_evidence_graph_for_doc_ids_unlocked([doc_id])
                retired_ids = await self._refresh_support_after_source_removal_unlocked(memory_ids)
                await self.db.execute("DELETE FROM document_metadata WHERE doc_id = ?", (doc_id,))
                await self.db.execute(
                    "DELETE FROM document_relationships WHERE source_doc_id = ? OR target_doc_id = ?",
                    (doc_id, doc_id),
                )
                await self.db.execute("DELETE FROM changelog WHERE doc_id = ?", (doc_id,))
                await self.db.execute("DELETE FROM agent_session_receipts WHERE doc_id = ?", (doc_id,))
                await self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                await self.db.commit()
                return retired_ids
            except Exception:
                await self.db.rollback()
                raise

    async def delete_projected_document(self, doc_id: str) -> None:
        """Delete document artifacts after an applied projected lifecycle plan.

        Unlike the legacy deletion path, this method never infers Memory
        lifecycle and never deletes Source Projection or Evidence lineage. The
        caller must first apply a gate-checked Lifecycle Plan that removes every
        document support edge; the invariant is checked again in this
        transaction before the document record is removed.
        """

        async with self._write_lock:
            try:
                async with self.db.execute(
                    "SELECT 1 FROM memory_sources WHERE doc_id = ? LIMIT 1",
                    (doc_id,),
                ) as cursor:
                    if await cursor.fetchone() is not None:
                        raise ValueError("active document support remains after lifecycle plan")
                async with self.db.execute(
                    "SELECT source, raw_content_uri, normalized_content_uri, pdf_content_uri "
                    "FROM documents WHERE doc_id = ?",
                    (doc_id,),
                ) as cursor:
                    document_row = await cursor.fetchone()
                if document_row is None:
                    return
                source_id = str(document_row["source"])
                for artifact_uri in dict.fromkeys(
                    str(uri)
                    for uri in (
                        document_row["raw_content_uri"],
                        document_row["normalized_content_uri"],
                        document_row["pdf_content_uri"],
                    )
                    if uri
                ):
                    async with self.db.execute(
                        """SELECT 1 FROM documents
                           WHERE doc_id != ?
                             AND (raw_content_uri = ? OR normalized_content_uri = ? OR pdf_content_uri = ?)
                           LIMIT 1""",
                        (doc_id, artifact_uri, artifact_uri, artifact_uri),
                    ) as cursor:
                        if await cursor.fetchone() is not None:
                            continue
                    await self.db.execute(
                        "INSERT OR IGNORE INTO source_artifact_cleanup_tasks "
                        "(task_id, source_id, artifact_uri) VALUES (?, ?, ?)",
                        (
                            source_artifact_cleanup_task_id(source_id, artifact_uri),
                            source_id,
                            artifact_uri,
                        ),
                    )
                await self.db.execute("DELETE FROM memory_search_metadata_fts WHERE doc_id = ?", (doc_id,))
                await self.db.execute(
                    "DELETE FROM memory_search_metadata_alias_fts WHERE doc_id = ?",
                    (doc_id,),
                )
                await self.db.execute(
                    "DELETE FROM memory_search_metadata_trigram WHERE doc_id = ?",
                    (doc_id,),
                )
                await self.db.execute("DELETE FROM document_metadata WHERE doc_id = ?", (doc_id,))
                await self.db.execute(
                    "DELETE FROM document_relationships WHERE source_doc_id = ? OR target_doc_id = ?",
                    (doc_id, doc_id),
                )
                await self.db.execute("DELETE FROM changelog WHERE doc_id = ?", (doc_id,))
                await self.db.execute("DELETE FROM agent_session_receipts WHERE doc_id = ?", (doc_id,))
                await self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def rebind_projected_document_support(
        self,
        old_doc_id: str,
        new_doc_id: str,
    ) -> None:
        """Move legacy document provenance after a stable Source Unit rename.

        Source Projection and Evidence lineage remain pinned to immutable old
        revisions. Only the compatibility ``memory_sources`` edge follows the
        current document locator.
        """
        if old_doc_id == new_doc_id:
            return
        async with self._write_lock:
            try:
                rows = await self.db.execute_fetchall(
                    "SELECT doc_id, source FROM documents WHERE doc_id IN (?, ?)",
                    (old_doc_id, new_doc_id),
                )
                sources = {str(row["doc_id"]): str(row["source"]) for row in rows}
                if new_doc_id not in sources:
                    raise ValueError("target document must exist before provenance rebind")
                if old_doc_id not in sources:
                    return
                if sources[old_doc_id] != sources[new_doc_id]:
                    raise ValueError("document lineage cannot cross configured Sources")
                await self.db.execute(
                    """DELETE FROM memory_sources AS old_support
                       WHERE old_support.doc_id = ?
                         AND EXISTS (
                             SELECT 1 FROM memory_sources AS new_support
                             WHERE new_support.memory_id = old_support.memory_id
                               AND new_support.doc_id = ?
                         )""",
                    (old_doc_id, new_doc_id),
                )
                await self.db.execute(
                    "UPDATE memory_sources SET doc_id = ? WHERE doc_id = ?",
                    (new_doc_id, old_doc_id),
                )
                await self._refresh_metadata_fts_for_doc_unlocked(new_doc_id)
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def upsert_metadata(self, meta: DocumentMetadata) -> None:
        async with self._write_lock:
            entities_json = json.dumps([{"name": e.canonical_name, "tags": e.tags} for e in meta.entities])
            await self.db.execute(
                """INSERT INTO document_metadata (
                    doc_id, summary, tags, entities, doc_type, complexity, enriched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    summary=excluded.summary, tags=excluded.tags,
                    entities=excluded.entities, doc_type=excluded.doc_type,
                    complexity=excluded.complexity, enriched_at=excluded.enriched_at""",
                (
                    meta.doc_id,
                    meta.summary,
                    json.dumps(meta.tags),
                    entities_json,
                    meta.doc_type,
                    meta.complexity,
                    meta.enriched_at.isoformat() if meta.enriched_at else _now_iso(),
                ),
            )
            await self.db.commit()

    async def get_metadata(self, doc_id: str) -> DocumentMetadata | None:
        async with self.db.execute("SELECT * FROM document_metadata WHERE doc_id = ?", (doc_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            raw_entities = json.loads(d["entities"])
            entities = [
                Entity(
                    id=0,
                    canonical_name=e["name"],
                    tags=e.get("tags", [e.get("type", "unknown")]),  # backward compat
                    display_name=e["name"],
                )
                for e in raw_entities
            ]
            return DocumentMetadata(
                doc_id=d["doc_id"],
                summary=d["summary"],
                tags=json.loads(d["tags"]),
                entities=entities,
                doc_type=d["doc_type"],
                complexity=d["complexity"],
                enriched_at=_parse_dt(d["enriched_at"]),
            )

    # ==================================================================
    # Source Projection lineage
    # ==================================================================

    async def record_source_projection(
        self,
        projection: SourceProjection,
        *,
        expected_source_activity_epoch: int | None = None,
        _manage_transaction: bool = True,
    ) -> None:
        """Persist one complete provider-neutral projection atomically.

        Observation and unit revisions are immutable. Stable observations and
        units may update their locator/current-revision pointers without changing
        provider identity. Reusing a run id for a different payload is rejected.
        """

        payload = source_projection_to_payload(projection)
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        now = _now_iso()
        transaction_lock = self._write_lock if _manage_transaction else nullcontext()
        async with transaction_lock:
            try:
                # First write acquires SQLite's cross-process writer fence for
                # this transaction before any projection snapshot is read.
                await self.db.execute(
                    "UPDATE sources SET status = status WHERE id = ?",
                    (projection.source_id,),
                )
                if expected_source_activity_epoch is not None:
                    async with self.db.execute(
                        "SELECT activity_epoch FROM sources WHERE id = ?",
                        (projection.source_id,),
                    ) as cursor:
                        source_epoch = await cursor.fetchone()
                    current_epoch = int(source_epoch["activity_epoch"] or 0)
                    if current_epoch != expected_source_activity_epoch:
                        raise SourceActivityConflict(
                            "source activity epoch changed: "
                            f"expected {expected_source_activity_epoch}, current {current_epoch}"
                        )
                async with self.db.execute(
                    "SELECT payload_hash FROM source_projection_runs WHERE id = ?",
                    (projection.run_id,),
                ) as cursor:
                    existing_run = await cursor.fetchone()
                if existing_run is not None:
                    if existing_run["payload_hash"] != payload_hash:
                        raise ValueError("projection retry payload mismatch")
                    if _manage_transaction:
                        await self.db.commit()
                    return

                await self.db.execute(
                    """INSERT INTO source_projection_runs (
                        id, source_id, source_type, coverage, scope_json,
                        checkpoint_json, payload_json, payload_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        projection.run_id,
                        projection.source_id,
                        projection.source_type,
                        projection.coverage.value,
                        json.dumps(dict(projection.scope), sort_keys=True),
                        json.dumps(dict(projection.checkpoint), sort_keys=True),
                        payload_json,
                        payload_hash,
                        now,
                    ),
                )

                for unit in projection.source_units:
                    await self.db.execute(
                        """INSERT OR IGNORE INTO source_units (
                            id, source_id, unit_type, provider_key, locator_json,
                            current_revision_id, updated_at
                        ) VALUES (?, ?, ?, ?, ?, NULL, ?)""",
                        (
                            unit.id,
                            unit.source_id,
                            unit.unit_type,
                            unit.provider_key,
                            json.dumps(dict(unit.locator), sort_keys=True),
                            now,
                        ),
                    )
                    await self._assert_projection_identity_unlocked(
                        table="source_units",
                        row_id=unit.id,
                        expected={
                            "source_id": unit.source_id,
                            "unit_type": unit.unit_type,
                            "provider_key": unit.provider_key,
                        },
                    )
                    document_id = str(unit.locator.get("document_id") or "").strip()
                    if document_id:
                        if projection.coverage is ProjectionCoverage.TOMBSTONED_DELTA:
                            await self.db.execute(
                                """UPDATE source_unit_document_lineage_history
                                   SET is_current = 0, last_seen_at = ?
                                   WHERE source_id = ? AND source_unit_id = ?""",
                                (now, unit.source_id, unit.id),
                            )
                            await self.db.execute(
                                "UPDATE source_units SET locator_json = ?, updated_at = ? WHERE id = ?",
                                (json.dumps(dict(unit.locator), sort_keys=True), now, unit.id),
                            )
                            continue
                        async with self.db.execute(
                            """SELECT first_seen_at
                               FROM source_unit_document_lineage_history
                               WHERE source_id = ? AND document_id = ?
                                 AND source_unit_id = ?""",
                            (unit.source_id, document_id, unit.id),
                        ) as cursor:
                            existing_document_lineage = await cursor.fetchone()
                        await self.db.execute(
                            """UPDATE source_unit_document_lineage_history
                               SET is_current = 0
                               WHERE source_id = ? AND (
                                   source_unit_id = ? OR document_id = ?
                               ) AND NOT (source_unit_id = ? AND document_id = ?)""",
                            (
                                unit.source_id,
                                unit.id,
                                document_id,
                                unit.id,
                                document_id,
                            ),
                        )
                        await self.db.execute(
                            """INSERT INTO source_unit_document_lineage_history (
                                   source_id, document_id, source_unit_id, is_current,
                                   first_seen_at, last_seen_at
                               ) VALUES (?, ?, ?, 1, ?, ?)
                               ON CONFLICT(source_id, document_id, source_unit_id) DO UPDATE SET
                                   is_current = 1,
                                   last_seen_at = excluded.last_seen_at""",
                            (
                                unit.source_id,
                                document_id,
                                unit.id,
                                str(existing_document_lineage["first_seen_at"])
                                if existing_document_lineage is not None
                                else now,
                                now,
                            ),
                        )
                    await self.db.execute(
                        "UPDATE source_units SET locator_json = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(dict(unit.locator), sort_keys=True), now, unit.id),
                    )

                for observation in projection.observations:
                    await self.db.execute(
                        """INSERT OR IGNORE INTO source_observations (
                            id, source_id, source_unit_id, observation_type,
                            provider_key, locator_json, current_revision_id, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, NULL, ?)""",
                        (
                            observation.id,
                            observation.source_id,
                            observation.source_unit_id,
                            observation.observation_type,
                            observation.provider_key,
                            json.dumps(dict(observation.locator), sort_keys=True),
                            now,
                        ),
                    )
                    await self._assert_projection_identity_unlocked(
                        table="source_observations",
                        row_id=observation.id,
                        expected={
                            "source_id": observation.source_id,
                            "source_unit_id": observation.source_unit_id,
                            "observation_type": observation.observation_type,
                            "provider_key": observation.provider_key,
                        },
                    )
                    await self.db.execute(
                        "UPDATE source_observations SET locator_json = ?, updated_at = ? WHERE id = ?",
                        (json.dumps(dict(observation.locator), sort_keys=True), now, observation.id),
                    )

                for revision in projection.observation_revisions:
                    metadata_json = json.dumps(dict(revision.metadata), sort_keys=True)
                    await self.db.execute(
                        """INSERT OR IGNORE INTO source_observation_revisions (
                            id, observation_id, semantic_hash, content, metadata_json,
                            observed_at, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            revision.id,
                            revision.observation_id,
                            revision.semantic_hash,
                            revision.content,
                            metadata_json,
                            revision.observed_at,
                            now,
                        ),
                    )
                    await self._assert_projection_identity_unlocked(
                        table="source_observation_revisions",
                        row_id=revision.id,
                        expected={
                            "observation_id": revision.observation_id,
                            "semantic_hash": revision.semantic_hash,
                            "content": revision.content,
                            "metadata_json": metadata_json,
                        },
                    )
                    await self.db.execute(
                        "UPDATE source_observations SET current_revision_id = ?, updated_at = ? WHERE id = ?",
                        (revision.id, now, revision.observation_id),
                    )

                for revision in projection.source_unit_revisions:
                    revision_ids_json = json.dumps(list(revision.observation_revision_ids))
                    await self.db.execute(
                        """INSERT OR IGNORE INTO source_unit_revisions (
                            id, source_unit_id, semantic_hash, location_hash,
                            membership_hash, access_hash,
                            observation_revision_ids_json, observed_at, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            revision.id,
                            revision.source_unit_id,
                            revision.semantic_hash,
                            revision.location_hash,
                            revision.membership_hash,
                            revision.access_hash,
                            revision_ids_json,
                            revision.observed_at,
                            now,
                        ),
                    )
                    await self._assert_projection_identity_unlocked(
                        table="source_unit_revisions",
                        row_id=revision.id,
                        expected={
                            "source_unit_id": revision.source_unit_id,
                            "semantic_hash": revision.semantic_hash,
                            "location_hash": revision.location_hash,
                            "membership_hash": revision.membership_hash,
                            "access_hash": revision.access_hash,
                            "observation_revision_ids_json": revision_ids_json,
                        },
                    )
                    await self.db.execute(
                        "UPDATE source_units SET current_revision_id = ?, updated_at = ? WHERE id = ?",
                        (revision.id, now, revision.source_unit_id),
                    )

                for index, relation in enumerate(projection.relations):
                    await self.db.execute(
                        """INSERT INTO source_projection_relations (
                            projection_run_id, relation_index, relation_type,
                            from_id, to_id, provider_relation_id, metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            projection.run_id,
                            index,
                            relation.relation_type.value,
                            relation.from_id,
                            relation.to_id,
                            relation.provider_relation_id,
                            json.dumps(dict(relation.metadata), sort_keys=True),
                        ),
                    )
                delta_payloads = payload["deltas"]
                assert isinstance(delta_payloads, list)
                for index, delta in enumerate(projection.deltas):
                    await self.db.execute(
                        """INSERT INTO source_revision_deltas (
                            projection_run_id, delta_index, source_unit_id, payload_json
                        ) VALUES (?, ?, ?, ?)""",
                        (
                            projection.run_id,
                            index,
                            delta.source_unit_id,
                            json.dumps(delta_payloads[index], sort_keys=True, separators=(",", ":")),
                        ),
                    )
                    if delta.removed_observation_ids and delta.coverage.proves_absence:
                        placeholders = ", ".join("?" for _ in delta.removed_observation_ids)
                        await self.db.execute(
                            f"""UPDATE source_observations
                                SET current_revision_id = NULL, updated_at = ?
                                WHERE source_unit_id = ? AND id IN ({placeholders})""",
                            (now, delta.source_unit_id, *delta.removed_observation_ids),
                        )
                if _manage_transaction:
                    await self.db.commit()
            except Exception:
                if _manage_transaction:
                    await self.db.rollback()
                raise

    async def _assert_projection_identity_unlocked(
        self,
        *,
        table: str,
        row_id: str,
        expected: Mapping[str, object],
    ) -> None:
        allowed_tables = {
            "source_units",
            "source_observations",
            "source_observation_revisions",
            "source_unit_revisions",
        }
        if table not in allowed_tables:
            raise ValueError("unsupported projection identity table")
        async with self.db.execute(f"SELECT * FROM {table} WHERE id = ?", (row_id,)) as cursor:
            row = await cursor.fetchone()
        if row is None or any(row[key] != value for key, value in expected.items()):
            raise ValueError(f"immutable projection identity mismatch: {table}:{row_id}")

    async def get_source_projection(self, run_id: str) -> SourceProjection | None:
        async with self.db.execute(
            "SELECT payload_json FROM source_projection_runs WHERE id = ?",
            (run_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return source_projection_from_payload(json.loads(row["payload_json"]))

    async def create_projection_scope_transition(
        self,
        transition: ProjectionScopeTransition,
    ) -> ProjectionScopeTransition:
        if transition.status is not ProjectionScopeTransitionStatus.PENDING:
            raise ValueError("new projection scope transition must be pending")
        now = _now_iso()
        previous_json = json.dumps(dict(transition.previous_scope), sort_keys=True, separators=(",", ":"))
        target_json = json.dumps(dict(transition.target_scope), sort_keys=True, separators=(",", ":"))
        async with self._write_lock:
            try:
                await self.db.execute(
                    """INSERT OR IGNORE INTO projection_scope_transitions (
                        id, source_id, previous_scope_json, target_scope_json, status,
                        run_id, coverage, error, created_at, updated_at, completed_at
                    ) VALUES (?, ?, ?, ?, 'pending', NULL, NULL, NULL, ?, ?, NULL)""",
                    (
                        transition.id,
                        transition.source_id,
                        previous_json,
                        target_json,
                        transition.created_at or now,
                        now,
                    ),
                )
                async with self.db.execute(
                    "SELECT * FROM projection_scope_transitions WHERE id = ?",
                    (transition.id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if (
                    row is None
                    or row["source_id"] != transition.source_id
                    or row["previous_scope_json"] != previous_json
                    or row["target_scope_json"] != target_json
                ):
                    raise ValueError("projection scope transition retry identity mismatch")
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        return self._row_to_projection_scope_transition(row)

    async def get_open_projection_scope_transition(
        self,
        source_id: str,
    ) -> ProjectionScopeTransition | None:
        async with self.db.execute(
            """SELECT * FROM projection_scope_transitions
               WHERE source_id = ? AND status IN ('pending', 'running', 'failed')
               ORDER BY created_at, id LIMIT 1""",
            (source_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_projection_scope_transition(row) if row is not None else None

    async def list_projection_scope_transitions(
        self,
        source_id: str,
        *,
        limit: int = 20,
    ) -> list[ProjectionScopeTransition]:
        rows = await self.db.execute_fetchall(
            """SELECT * FROM projection_scope_transitions
               WHERE source_id = ? ORDER BY created_at DESC, id DESC LIMIT ?""",
            (source_id, limit),
        )
        return [self._row_to_projection_scope_transition(row) for row in rows]

    async def start_projection_scope_transition(
        self,
        transition_id: str,
        *,
        run_id: str,
    ) -> ProjectionScopeTransition:
        now = _now_iso()
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE projection_scope_transitions
                   SET status = 'running', run_id = ?, coverage = NULL, error = NULL,
                       updated_at = ?, completed_at = NULL
                   WHERE id = ? AND status IN ('pending', 'failed', 'running')
                     AND (status != 'running' OR run_id = ?)""",
                (run_id, now, transition_id, run_id),
            )
            if cursor.rowcount != 1:
                await self.db.rollback()
                raise ValueError("projection scope transition is not startable")
            await self.db.commit()
        stored = await self._get_projection_scope_transition(transition_id)
        assert stored is not None
        return stored

    async def complete_projection_scope_transition(
        self,
        transition_id: str,
        *,
        run_id: str,
        coverage: ProjectionCoverage,
    ) -> ProjectionScopeTransition:
        if not coverage.proves_absence:
            raise ValueError("scope transition requires complete snapshot coverage")
        now = _now_iso()
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE projection_scope_transitions
                   SET status = 'applied', coverage = ?, error = NULL,
                       updated_at = ?, completed_at = ?
                   WHERE id = ? AND status = 'running' AND run_id = ?""",
                (coverage.value, now, now, transition_id, run_id),
            )
            if cursor.rowcount != 1:
                await self.db.rollback()
                raise ValueError("projection scope transition completion stale guard failed")
            await self.db.commit()
        stored = await self._get_projection_scope_transition(transition_id)
        assert stored is not None
        return stored

    async def fail_projection_scope_transition(
        self,
        transition_id: str,
        *,
        run_id: str,
        coverage: ProjectionCoverage,
        error: str,
    ) -> ProjectionScopeTransition:
        now = _now_iso()
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE projection_scope_transitions
                   SET status = 'failed', coverage = ?, error = ?, updated_at = ?
                   WHERE id = ? AND status = 'running' AND run_id = ?""",
                (coverage.value, error, now, transition_id, run_id),
            )
            if cursor.rowcount != 1:
                await self.db.rollback()
                raise ValueError("projection scope transition failure stale guard failed")
            await self.db.commit()
        stored = await self._get_projection_scope_transition(transition_id)
        assert stored is not None
        return stored

    async def _get_projection_scope_transition(
        self,
        transition_id: str,
    ) -> ProjectionScopeTransition | None:
        async with self.db.execute(
            "SELECT * FROM projection_scope_transitions WHERE id = ?",
            (transition_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_projection_scope_transition(row) if row is not None else None

    async def get_current_source_unit_revision(
        self,
        source_unit_id: str,
    ) -> SourceUnitRevision | None:
        async with self.db.execute(
            """SELECT sur.*
               FROM source_units su
               JOIN source_unit_revisions sur ON sur.id = su.current_revision_id
               WHERE su.id = ?""",
            (source_unit_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return SourceUnitRevision(
            id=row["id"],
            source_unit_id=row["source_unit_id"],
            semantic_hash=row["semantic_hash"],
            observation_revision_ids=tuple(json.loads(row["observation_revision_ids_json"])),
            location_hash=row["location_hash"],
            membership_hash=row["membership_hash"],
            access_hash=row["access_hash"],
            observed_at=row["observed_at"],
        )

    async def get_current_source_observation_revisions(
        self,
        source_unit_id: str,
    ) -> Mapping[str, SourceObservationRevision]:
        revisions: dict[str, SourceObservationRevision] = {}
        async with self.db.execute(
            """SELECT so.id AS source_observation_id, sor.*
               FROM source_observations so
               JOIN source_observation_revisions sor ON sor.id = so.current_revision_id
               WHERE so.source_unit_id = ?
               ORDER BY so.id""",
            (source_unit_id,),
        ) as cursor:
            async for row in cursor:
                revisions[row["source_observation_id"]] = SourceObservationRevision(
                    id=row["id"],
                    observation_id=row["observation_id"],
                    semantic_hash=row["semantic_hash"],
                    content=row["content"],
                    observed_at=row["observed_at"],
                    metadata=json.loads(row["metadata_json"] or "{}"),
                )
        return revisions

    async def find_source_unit_by_document_id(
        self,
        source_id: str,
        document_id: str,
        *,
        current_only: bool = False,
    ) -> SourceUnit | None:
        current_clause = " AND lineage.is_current = 1" if current_only else ""
        async with self.db.execute(
            f"""SELECT su.*
               FROM source_unit_document_lineage_history lineage
               JOIN source_units su ON su.id = lineage.source_unit_id
               WHERE lineage.source_id = ? AND lineage.document_id = ?{current_clause}
               ORDER BY lineage.is_current DESC, lineage.last_seen_at DESC, su.id
               LIMIT 1""",
            (source_id, document_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return SourceUnit(
            id=row["id"],
            source_id=row["source_id"],
            unit_type=row["unit_type"],
            provider_key=row["provider_key"],
            locator=json.loads(row["locator_json"] or "{}"),
        )

    async def list_source_unit_document_ids(
        self,
        source_unit_id: str,
    ) -> tuple[str, ...]:
        rows = await self.db.execute_fetchall(
            """SELECT document_id
               FROM source_unit_document_lineage_history
               WHERE source_unit_id = ?
               ORDER BY is_current DESC, last_seen_at DESC, document_id""",
            (source_unit_id,),
        )
        return tuple(str(row["document_id"]) for row in rows)

    async def list_current_source_unit_observation_ids(
        self,
        source_id: str,
    ) -> dict[str, tuple[str, ...]]:
        rows = await self.db.execute_fetchall(
            """SELECT su.id AS source_unit_id, so.id AS observation_id
               FROM source_units su
               LEFT JOIN source_observations so
                 ON so.source_unit_id = su.id AND so.current_revision_id IS NOT NULL
               WHERE su.source_id = ? AND su.current_revision_id IS NOT NULL
               ORDER BY su.id, so.id""",
            (source_id,),
        )
        result: dict[str, list[str]] = {}
        for row in rows:
            unit_id = str(row["source_unit_id"])
            result.setdefault(unit_id, [])
            if row["observation_id"] is not None:
                result[unit_id].append(str(row["observation_id"]))
        return {unit_id: tuple(ids) for unit_id, ids in result.items()}

    async def list_current_source_units(
        self,
        source_id: str,
    ) -> tuple[SourceUnit, ...]:
        rows = await self.db.execute_fetchall(
            """SELECT su.*
               FROM source_units su
               WHERE su.source_id = ?
                 AND su.current_revision_id IS NOT NULL
                 AND EXISTS (
                     SELECT 1 FROM source_observations so
                     WHERE so.source_unit_id = su.id
                       AND so.current_revision_id IS NOT NULL
                 )
               ORDER BY su.id""",
            (source_id,),
        )
        return tuple(
            SourceUnit(
                id=row["id"],
                source_id=row["source_id"],
                unit_type=row["unit_type"],
                provider_key=row["provider_key"],
                locator=json.loads(row["locator_json"] or "{}"),
            )
            for row in rows
        )

    async def list_current_source_units_page(
        self,
        source_id: str,
        *,
        filters: SourceUnitInventoryFilter,
        cursor: str | None = None,
        limit: int = 200,
    ) -> SourceUnitInventoryPage:
        if limit < 1 or limit > 500:
            raise ValueError("source unit inventory limit must be between 1 and 500")
        conditions = [
            "su.source_id = ?",
            "su.current_revision_id IS NOT NULL",
            "EXISTS (SELECT 1 FROM source_observations so "
            "WHERE so.source_unit_id = su.id "
            "AND so.current_revision_id IS NOT NULL)",
        ]
        params: list[object] = [source_id]
        if filters.unit_type:
            conditions.append("su.unit_type = ?")
            params.append(filters.unit_type)
        for key, value in sorted(filters.locator_equals.items()):
            if not key.replace("_", "").isalnum():
                raise ValueError("source unit locator filter key is invalid")
            conditions.append("json_extract(su.locator_json, ?) = ?")
            params.extend((f"$.{key}", value))
        temporal_filters = (
            ("observed_from", "<=", filters.observed_from_lte),
            ("observed_to", ">=", filters.observed_to_gte),
            ("observed_to", "<", filters.observed_to_lt),
        )
        for key, operator, value in temporal_filters:
            if value is None:
                continue
            conditions.append(f"json_extract(su.locator_json, '$.{key}') {operator} ?")
            params.append(value)
        if cursor:
            conditions.append("su.id > ?")
            params.append(cursor)
        params.append(limit + 1)
        rows = await self.db.execute_fetchall(
            "SELECT su.* FROM source_units su WHERE " + " AND ".join(conditions) + " ORDER BY su.id LIMIT ?",
            tuple(params),
        )
        has_more = len(rows) > limit
        page_rows = rows[:limit]
        units = tuple(
            SourceUnit(
                id=row["id"],
                source_id=row["source_id"],
                unit_type=row["unit_type"],
                provider_key=row["provider_key"],
                locator=json.loads(row["locator_json"] or "{}"),
            )
            for row in page_rows
        )
        return SourceUnitInventoryPage(
            units=units,
            next_cursor=units[-1].id if has_more and units else None,
        )

    async def list_legacy_memory_provenance(
        self,
        source_id: str,
    ) -> list[LegacyMemoryProvenance]:
        rows = await self.db.execute_fetchall(
            """SELECT DISTINCT
                   m.id AS memory_id, ms.doc_id, ms.source_id, ms.source_type,
                   m.content, ms.excerpt, m.visibility, m.owner_user_id,
                   m.project_key, m.repo_identifier
               FROM memory_sources ms
               JOIN memories m ON m.id = ms.memory_id
               WHERE ms.source_id = ? AND m.status = 'active'
               ORDER BY m.id, ms.doc_id""",
            (source_id,),
        )
        return [
            LegacyMemoryProvenance(
                memory_id=row["memory_id"],
                doc_id=row["doc_id"],
                source_id=row["source_id"],
                source_type=row["source_type"],
                content=row["content"],
                excerpt=row["excerpt"],
                visibility=row["visibility"],
                owner_user_id=row["owner_user_id"],
                project_key=row["project_key"],
                repo_identifier=row["repo_identifier"],
            )
            for row in rows
        ]

    async def count_active_source_memories_without_support(self, source_id: str) -> int:
        """Count active Memories whose same-source support invariant is absent."""

        async with self.db.execute(
            """SELECT COUNT(DISTINCT ms.memory_id) AS count
               FROM memory_sources ms
               JOIN memories m ON m.id = ms.memory_id
               WHERE ms.source_id = ?
                 AND m.status = 'active'
                 AND NOT EXISTS (
                     SELECT 1 FROM memory_support_assertions msa
                     WHERE msa.memory_id = ms.memory_id
                       AND msa.source_id = ms.source_id
                       AND msa.active = 1
                 )""",
            (source_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["count"] if row is not None else 0)

    async def count_active_source_memories(self, source_id: str) -> int:
        """Count active source-backed Memories without loading their LOB content."""

        async with self.db.execute(
            """SELECT COUNT(DISTINCT ms.memory_id) AS count
               FROM memory_sources ms
               JOIN memories m ON m.id = ms.memory_id
               WHERE ms.source_id = ? AND m.status = 'active'""",
            (source_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return int(row["count"] if row is not None else 0)

    async def get_lifecycle_gate(self, source_id: str) -> LifecycleGate:
        async with self.db.execute(
            "SELECT * FROM source_lifecycle_gates WHERE source_id = ?",
            (source_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return LifecycleGate(
                source_id=source_id,
                state=LifecycleGateState.GATED,
                reason="lifecycle cutover audit has not completed",
            )
        return LifecycleGate(
            source_id=row["source_id"],
            state=LifecycleGateState(row["state"]),
            reason=row["reason"],
            enabled_at=row["enabled_at"],
            audited_at=row["audited_at"],
        )

    async def _assert_source_activity_fence_unlocked(
        self,
        source_id: str,
        source_activity: SourceActivityLease | None,
    ) -> None:
        if source_activity is None:
            return
        if source_activity.source_id != source_id:
            raise SourceActivityConflict("source activity fence source mismatch")
        source_lock = await self.db.execute(
            "UPDATE sources SET status = status WHERE id = ?",
            (source_id,),
        )
        if source_lock.rowcount != 1:
            raise SourceActivityConflict(f"Source not found: {source_id}")
        async with self.db.execute(
            "SELECT activity_epoch FROM sources WHERE id = ?",
            (source_id,),
        ) as cursor:
            source = await cursor.fetchone()
        async with self.db.execute(
            """SELECT source_id, capability, epoch, lease_until
               FROM source_activity_leases WHERE id = ?""",
            (source_activity.id,),
        ) as cursor:
            lease = await cursor.fetchone()
        if (
            source is None
            or int(source["activity_epoch"] or 0) != source_activity.epoch
            or lease is None
            or str(lease["source_id"]) != source_id
            or lease["capability"] != source_activity.capability
            or int(lease["epoch"]) != source_activity.epoch
            or str(lease["lease_until"]) <= _now_iso()
        ):
            raise SourceActivityConflict(f"source activity fence is not current: {source_activity.id}")

    async def enable_lifecycle_gate(
        self,
        source_id: str,
        *,
        source_activity: SourceActivityLease | None = None,
    ) -> LifecycleGate:
        """Enable destructive lifecycle only after the durable audit closes."""

        async with self._write_lock:
            try:
                await self._assert_source_activity_fence_unlocked(
                    source_id,
                    source_activity,
                )
                async with self.db.execute(
                    """SELECT COUNT(*) AS count
                       FROM lifecycle_cutover_findings
                       WHERE source_id = ? AND status = 'open'""",
                    (source_id,),
                ) as cursor:
                    finding_count = int((await cursor.fetchone())["count"])
                if finding_count:
                    raise ValueError("open lifecycle cutover findings block the lifecycle gate")
                async with self.db.execute(
                    """SELECT DISTINCT ms.memory_id
                       FROM memory_sources ms
                       JOIN memories m ON m.id = ms.memory_id
                       WHERE ms.source_id = ?
                         AND m.status = 'active'
                         AND NOT EXISTS (
                             SELECT 1 FROM memory_support_assertions msa
                             WHERE msa.memory_id = ms.memory_id
                               AND msa.source_id = ms.source_id
                               AND msa.active = 1
                         )
                       LIMIT 1""",
                    (source_id,),
                ) as cursor:
                    missing_lineage = await cursor.fetchone()
                if missing_lineage is not None:
                    raise ValueError("source-backed Memory lacks validated support lineage")
                now = _now_iso()
                await self.db.execute(
                    """INSERT INTO source_lifecycle_gates (
                        source_id, state, reason, audited_at, enabled_at, updated_at
                    ) VALUES (?, 'enabled', NULL, ?, ?, ?)
                    ON CONFLICT(source_id) DO UPDATE SET
                        state='enabled', reason=NULL, audited_at=excluded.audited_at,
                        enabled_at=COALESCE(source_lifecycle_gates.enabled_at, excluded.enabled_at),
                        updated_at=excluded.updated_at""",
                    (source_id, now, now, now),
                )
                await self._assert_source_activity_fence_unlocked(
                    source_id,
                    source_activity,
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        return await self.get_lifecycle_gate(source_id)

    async def gate_destructive_lifecycle(
        self,
        source_id: str,
        *,
        reason: str,
        source_activity: SourceActivityLease | None = None,
    ) -> LifecycleGate:
        now = _now_iso()
        async with self._write_lock:
            try:
                await self._assert_source_activity_fence_unlocked(
                    source_id,
                    source_activity,
                )
                await self.db.execute(
                    """INSERT INTO source_lifecycle_gates (
                        source_id, state, reason, audited_at, enabled_at, updated_at
                    ) VALUES (?, 'gated', ?, ?, NULL, ?)
                    ON CONFLICT(source_id) DO UPDATE SET
                        state='gated', reason=excluded.reason,
                        audited_at=excluded.audited_at, updated_at=excluded.updated_at""",
                    (source_id, reason, now, now),
                )
                await self._assert_source_activity_fence_unlocked(
                    source_id,
                    source_activity,
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        return await self.get_lifecycle_gate(source_id)

    async def upsert_lifecycle_cutover_finding(
        self,
        finding: LifecycleCutoverFinding,
        *,
        source_activity: SourceActivityLease | None = None,
    ) -> None:
        if finding.status is not CutoverFindingStatus.OPEN:
            raise ValueError("lifecycle finding upsert accepts only open findings")
        now = _now_iso()
        created_at = finding.created_at or now
        async with self._write_lock:
            try:
                await self._assert_source_activity_fence_unlocked(
                    finding.source_id,
                    source_activity,
                )
                async with self.db.execute(
                    """SELECT source_id, memory_id, reason, status
                       FROM lifecycle_cutover_findings WHERE id = ?""",
                    (finding.id,),
                ) as cursor:
                    existing = await cursor.fetchone()
                if existing is not None and (
                    str(existing["source_id"]) != finding.source_id or str(existing["memory_id"]) != finding.memory_id
                ):
                    raise ValueError("lifecycle finding identity mismatch")
                if (
                    existing is not None
                    and CutoverFindingStatus(str(existing["status"])) is CutoverFindingStatus.RESOLVED
                ):
                    await self._assert_source_activity_fence_unlocked(
                        finding.source_id,
                        source_activity,
                    )
                    await self.db.commit()
                    return
                await self.db.execute(
                    """INSERT INTO lifecycle_cutover_findings (
                        id, source_id, memory_id, reason, status,
                        available_provenance_json, mapping_attempt_json,
                        observation_id, source_unit_id, created_at, updated_at, resolved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        reason=excluded.reason,
                        available_provenance_json=excluded.available_provenance_json,
                        mapping_attempt_json=excluded.mapping_attempt_json,
                        updated_at=excluded.updated_at
                    WHERE lifecycle_cutover_findings.status = 'open'""",
                    (
                        finding.id,
                        finding.source_id,
                        finding.memory_id,
                        finding.reason.value,
                        finding.status.value,
                        json.dumps(dict(finding.available_provenance), sort_keys=True),
                        json.dumps(dict(finding.mapping_attempt), sort_keys=True),
                        finding.observation_id,
                        finding.source_unit_id,
                        created_at,
                        finding.updated_at or now,
                        finding.resolved_at,
                    ),
                )
                await self.db.execute(
                    """INSERT INTO source_lifecycle_gates (
                        source_id, state, reason, audited_at, enabled_at, updated_at
                    ) VALUES (?, 'gated', 'open lifecycle cutover finding', ?, NULL, ?)
                    ON CONFLICT(source_id) DO UPDATE SET
                        state='gated', reason=excluded.reason,
                        audited_at=excluded.audited_at, updated_at=excluded.updated_at""",
                    (finding.source_id, now, now),
                )
                await self._assert_source_activity_fence_unlocked(
                    finding.source_id,
                    source_activity,
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def get_lifecycle_cutover_finding(
        self,
        finding_id: str,
    ) -> LifecycleCutoverFinding | None:
        async with self.db.execute(
            "SELECT * FROM lifecycle_cutover_findings WHERE id = ?",
            (finding_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_lifecycle_cutover_finding(row) if row is not None else None

    async def list_lifecycle_cutover_findings(
        self,
        source_id: str,
        *,
        status: CutoverFindingStatus | None = None,
    ) -> list[LifecycleCutoverFinding]:
        params: list[object] = [source_id]
        status_clause = ""
        if status is not None:
            status_clause = " AND status = ?"
            params.append(status.value)
        rows = await self.db.execute_fetchall(
            "SELECT * FROM lifecycle_cutover_findings WHERE source_id = ?" + status_clause + " ORDER BY created_at, id",
            tuple(params),
        )
        return [self._row_to_lifecycle_cutover_finding(row) for row in rows]

    async def create_lifecycle_backfill_job(
        self,
        job: LifecycleBackfillJob,
    ) -> LifecycleBackfillJob:
        return await self._create_lifecycle_backfill_job(
            job,
            cancel_active_sync=False,
        )

    async def create_source_rebaseline_job(
        self,
        job: LifecycleBackfillJob,
    ) -> LifecycleBackfillJob:
        """Atomically fence active sync work and admit destructive maintenance."""

        return await self._create_lifecycle_backfill_job(
            job,
            cancel_active_sync=True,
        )

    async def _create_lifecycle_backfill_job(
        self,
        job: LifecycleBackfillJob,
        *,
        cancel_active_sync: bool,
    ) -> LifecycleBackfillJob:
        if job.status is not LifecycleBackfillJobStatus.QUEUED:
            raise ValueError("new lifecycle backfill job must be queued")
        now = _now_iso()
        async with self._write_lock:
            try:
                source_lock = await self.db.execute(
                    "UPDATE sources SET status = status WHERE id = ?",
                    (job.source_id,),
                )
                if source_lock.rowcount != 1:
                    raise ValueError(f"Source not found: {job.source_id}")
                async with self.db.execute(
                    """SELECT run_id FROM source_sync_runs
                       WHERE source_id = ? AND status IN ('pending', 'running')
                       ORDER BY created_at LIMIT 1""",
                    (job.source_id,),
                ) as cursor:
                    active_run = await cursor.fetchone()
                if active_run is not None and not cancel_active_sync:
                    raise ValueError(f"source sync run already active: {active_run['run_id']}")
                async with self.db.execute(
                    """SELECT job_id FROM local_agent_jobs
                       WHERE source_id = ? AND status IN ('queued', 'leased')
                       ORDER BY created_at LIMIT 1""",
                    (job.source_id,),
                ) as cursor:
                    active_local_job = await cursor.fetchone()
                if active_local_job is not None:
                    raise ValueError(f"local agent job already active: {active_local_job['job_id']}")
                async with self.db.execute(
                    """SELECT operation_id FROM source_access_transitions
                       WHERE source_id = ?
                         AND status IN ('queued', 'running', 'failed')
                       ORDER BY created_at LIMIT 1""",
                    (job.source_id,),
                ) as cursor:
                    active_access = await cursor.fetchone()
                if active_access is not None:
                    raise ValueError(f"source access transition already active: {active_access['operation_id']}")
                async with self.db.execute(
                    "SELECT source_id, status FROM lifecycle_backfill_jobs WHERE id = ?",
                    (job.id,),
                ) as cursor:
                    existing = await cursor.fetchone()
                if existing is not None:
                    if existing["source_id"] != job.source_id:
                        raise ValueError("lifecycle backfill job retry identity mismatch")
                    if existing["status"] in {
                        LifecycleBackfillJobStatus.QUEUED.value,
                        LifecycleBackfillJobStatus.RUNNING.value,
                    }:
                        await self._acquire_source_activity_unlocked(
                            activity_id=job.id,
                            source_id=job.source_id,
                            kind=SourceActivityKind.MAINTENANCE,
                            capability=job.id,
                            lease_seconds=900,
                            bump_epoch=True,
                        )
                    await self.db.commit()
                    stored = await self.get_lifecycle_backfill_job(job.id)
                    assert stored is not None
                    return stored
                async with self.db.execute(
                    "SELECT id FROM lifecycle_backfill_jobs "
                    "WHERE source_id = ? AND status IN ('queued', 'running') "
                    "AND id <> ? LIMIT 1",
                    (job.source_id, job.id),
                ) as cursor:
                    active = await cursor.fetchone()
                if active is not None:
                    raise ValueError(f"source lifecycle job already active: {active['id']}")
                if active_run is not None:
                    await self._cancel_source_sync_run_for_maintenance_unlocked(
                        str(active_run["run_id"]),
                        maintenance_job_id=job.id,
                    )
                if cancel_active_sync:
                    # A normal sync holds this lease in addition to its durable
                    # SourceSyncRun lease.  Revoking only SYNC leaves collection
                    # and agent-patch state machines independent; the epoch bump
                    # below fences any already-computed stale lifecycle commit.
                    await self.db.execute(
                        "DELETE FROM source_activity_leases WHERE source_id = ? AND kind = ?",
                        (job.source_id, SourceActivityKind.SYNC.value),
                    )
                await self._acquire_source_activity_unlocked(
                    activity_id=job.id,
                    source_id=job.source_id,
                    kind=SourceActivityKind.MAINTENANCE,
                    capability=job.id,
                    lease_seconds=900,
                    bump_epoch=True,
                )
                await self.db.execute(
                    """INSERT INTO lifecycle_backfill_jobs (
                        id, source_id, status, scanned_memories, mapped_memories,
                        finding_count, error, created_at, started_at, completed_at, updated_at
                    ) VALUES (?, ?, 'queued', 0, 0, 0, NULL, ?, NULL, NULL, ?)""",
                    (job.id, job.source_id, job.created_at or now, now),
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        stored = await self.get_lifecycle_backfill_job(job.id)
        assert stored is not None
        return stored

    async def _cancel_source_sync_run_for_maintenance_unlocked(
        self,
        run_id: str,
        *,
        maintenance_job_id: str,
    ) -> None:
        now = _now_iso()
        cursor = await self.db.execute(
            """UPDATE source_sync_runs
               SET status = 'failed',
                   input_snapshot_id = CASE
                       WHEN rerun_requested = 1
                       THEN COALESCE(rerun_input_snapshot_id, input_snapshot_id)
                       ELSE input_snapshot_id
                   END,
                   input_generation_watermark = CASE
                       WHEN rerun_requested = 1
                       THEN COALESCE(
                           rerun_input_generation_watermark,
                           input_generation_watermark
                       )
                       ELSE input_generation_watermark
                   END,
                   source_config_revision = CASE
                       WHEN rerun_requested = 1
                       THEN COALESCE(rerun_source_config_revision, source_config_revision)
                       ELSE source_config_revision
                   END,
                   predecessor_activity_id = CASE
                       WHEN rerun_requested = 1
                       THEN COALESCE(
                           rerun_predecessor_activity_id,
                           predecessor_activity_id
                       )
                       ELSE predecessor_activity_id
                   END,
                   rerun_requested = 0,
                   rerun_input_snapshot_id = NULL,
                   rerun_input_generation_watermark = NULL,
                   rerun_source_config_revision = NULL,
                   rerun_predecessor_activity_id = NULL,
                   lease_owner = NULL,
                   lease_expires_at = NULL,
                   next_attempt_at = NULL,
                   error_message = ?,
                   completed_at = ?,
                   updated_at = ?
               WHERE run_id = ? AND status IN ('pending', 'running')""",
            (
                f"cancelled_by_source_lifecycle_maintenance:{maintenance_job_id}",
                now,
                now,
                run_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ValueError(f"source sync run changed during maintenance admission: {run_id}")

    async def _acquire_source_activity_unlocked(
        self,
        *,
        activity_id: str,
        source_id: str,
        kind: SourceActivityKind,
        capability: str | None,
        lease_seconds: int,
        expected_epoch: int | None = None,
        bump_epoch: bool = False,
    ) -> SourceActivityLease:
        now = datetime.now(timezone.utc)
        now_iso = _utc_iso(now)
        lease_until = now + timedelta(seconds=max(1, lease_seconds))
        async with self.db.execute(
            "SELECT activity_epoch FROM sources WHERE id = ?",
            (source_id,),
        ) as cursor:
            source = await cursor.fetchone()
        if source is None:
            raise SourceActivityConflict(f"Source not found: {source_id}")
        epoch = int(source["activity_epoch"] or 0)
        if expected_epoch is not None and epoch != expected_epoch:
            raise SourceActivityConflict(f"source activity epoch changed: expected {expected_epoch}, current {epoch}")
        # Expiry removes admission authority. Purge stale rows while holding
        # the Source write boundary so a deterministic job/activity id can be
        # reacquired after a crash without colliding with its old primary key.
        await self.db.execute(
            "DELETE FROM source_activity_leases WHERE source_id = ? AND lease_until <= ?",
            (source_id, now_iso),
        )
        async with self.db.execute(
            """SELECT id, kind, epoch, capability, lease_until
               FROM source_activity_leases
               WHERE source_id = ? AND lease_until > ?
               ORDER BY created_at LIMIT 1""",
            (source_id, now_iso),
        ) as cursor:
            active = await cursor.fetchone()
        if active is not None and str(active["id"]) != activity_id:
            raise SourceActivityConflict(f"source activity already active: {active['id']} ({active['kind']})")
        if active is not None:
            if str(active["kind"]) != kind.value or int(active["epoch"]) != epoch or active["capability"] != capability:
                raise SourceActivityConflict("source activity retry identity mismatch")
            await self.db.execute(
                "UPDATE source_activity_leases SET lease_until = ?, updated_at = ? WHERE id = ?",
                (_utc_iso(lease_until), now_iso, activity_id),
            )
        else:
            if bump_epoch:
                epoch += 1
                await self.db.execute(
                    "UPDATE sources SET activity_epoch = ? WHERE id = ?",
                    (epoch, source_id),
                )
            await self.db.execute(
                """INSERT INTO source_activity_leases (
                       id, source_id, kind, epoch, capability, lease_until,
                       created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    activity_id,
                    source_id,
                    kind.value,
                    epoch,
                    capability,
                    _utc_iso(lease_until),
                    now_iso,
                    now_iso,
                ),
            )
        return SourceActivityLease(
            id=activity_id,
            source_id=source_id,
            kind=kind,
            epoch=epoch,
            capability=capability,
            lease_until=lease_until,
        )

    async def acquire_source_activity(
        self,
        *,
        activity_id: str,
        source_id: str,
        kind: SourceActivityKind,
        capability: str | None = None,
        lease_seconds: int = 900,
        expected_epoch: int | None = None,
    ) -> SourceActivityLease:
        async with self._write_lock:
            try:
                source_lock = await self.db.execute(
                    "UPDATE sources SET status = status WHERE id = ?",
                    (source_id,),
                )
                if source_lock.rowcount != 1:
                    raise SourceActivityConflict(f"Source not found: {source_id}")
                lease = await self._acquire_source_activity_unlocked(
                    activity_id=activity_id,
                    source_id=source_id,
                    kind=kind,
                    capability=capability,
                    lease_seconds=lease_seconds,
                    expected_epoch=expected_epoch,
                )
                await self.db.commit()
                return lease
            except Exception:
                await self.db.rollback()
                raise

    async def renew_source_activity(
        self,
        *,
        activity_id: str,
        capability: str | None = None,
        lease_seconds: int = 900,
    ) -> SourceActivityLease:
        now = datetime.now(timezone.utc)
        lease_until = now + timedelta(seconds=max(1, lease_seconds))
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE source_activity_leases
                   SET lease_until = ?, updated_at = ?
                   WHERE id = ? AND capability IS ? AND lease_until > ?""",
                (
                    _utc_iso(lease_until),
                    _utc_iso(now),
                    activity_id,
                    capability,
                    _utc_iso(now),
                ),
            )
            if cursor.rowcount != 1:
                await self.db.rollback()
                raise SourceActivityConflict(f"source activity lease is not current: {activity_id}")
            async with self.db.execute(
                "SELECT source_id, kind, epoch FROM source_activity_leases WHERE id = ?",
                (activity_id,),
            ) as read_cursor:
                row = await read_cursor.fetchone()
            await self.db.commit()
        assert row is not None
        return SourceActivityLease(
            id=activity_id,
            source_id=str(row["source_id"]),
            kind=SourceActivityKind(str(row["kind"])),
            epoch=int(row["epoch"]),
            capability=capability,
            lease_until=lease_until,
        )

    async def release_source_activity(
        self,
        *,
        activity_id: str,
        capability: str | None = None,
    ) -> bool:
        async with self._write_lock:
            cursor = await self.db.execute(
                "DELETE FROM source_activity_leases WHERE id = ? AND capability IS ?",
                (activity_id, capability),
            )
            await self.db.commit()
        return bool(cursor.rowcount)

    async def get_source_activity_epoch(self, source_id: str) -> int:
        async with self.db.execute(
            "SELECT activity_epoch FROM sources WHERE id = ?",
            (source_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise SourceActivityConflict(f"Source not found: {source_id}")
        return int(row["activity_epoch"] or 0)

    async def start_lifecycle_backfill_job(self, job_id: str) -> LifecycleBackfillJob:
        now = _now_iso()
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE lifecycle_backfill_jobs
                   SET status = 'running', started_at = ?, updated_at = ?, error = NULL
                   WHERE id = ? AND status = 'queued'""",
                (now, now, job_id),
            )
            if cursor.rowcount != 1:
                await self.db.rollback()
                raise ValueError("lifecycle backfill job is not queued")
            await self.db.commit()
        stored = await self.get_lifecycle_backfill_job(job_id)
        assert stored is not None
        return stored

    async def complete_lifecycle_backfill_job(
        self,
        job_id: str,
        *,
        scanned_memories: int,
        mapped_memories: int,
        finding_count: int,
    ) -> LifecycleBackfillJob:
        now = _now_iso()
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE lifecycle_backfill_jobs
                   SET status = 'completed', scanned_memories = ?, mapped_memories = ?,
                       finding_count = ?, error = NULL, completed_at = ?, updated_at = ?
                   WHERE id = ? AND status = 'running'""",
                (scanned_memories, mapped_memories, finding_count, now, now, job_id),
            )
            if cursor.rowcount != 1:
                await self.db.rollback()
                raise ValueError("lifecycle backfill job is not running")
            lease_cursor = await self.db.execute(
                "DELETE FROM source_activity_leases WHERE id = ? AND capability = ?",
                (job_id, job_id),
            )
            if lease_cursor.rowcount != 1:
                await self.db.rollback()
                raise SourceActivityConflict(f"source lifecycle activity lease is not current: {job_id}")
            await self.db.commit()
        stored = await self.get_lifecycle_backfill_job(job_id)
        assert stored is not None
        return stored

    async def fail_lifecycle_backfill_job(
        self,
        job_id: str,
        *,
        error: str,
    ) -> LifecycleBackfillJob:
        now = _now_iso()
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE lifecycle_backfill_jobs
                   SET status = 'failed', error = ?, completed_at = ?, updated_at = ?
                   WHERE id = ? AND status IN ('queued', 'running')""",
                (error, now, now, job_id),
            )
            if cursor.rowcount != 1:
                await self.db.rollback()
                raise ValueError("lifecycle backfill job cannot fail from its current state")
            lease_cursor = await self.db.execute(
                "DELETE FROM source_activity_leases WHERE id = ? AND capability = ?",
                (job_id, job_id),
            )
            if lease_cursor.rowcount != 1:
                await self.db.rollback()
                raise SourceActivityConflict(f"source lifecycle activity lease is not current: {job_id}")
            await self.db.commit()
        stored = await self.get_lifecycle_backfill_job(job_id)
        assert stored is not None
        return stored

    async def recover_stale_lifecycle_backfill_job(
        self,
        job_id: str,
        *,
        error: str,
    ) -> LifecycleBackfillJob:
        """Fail an orphaned lifecycle job only after its lease lost authority."""

        now = datetime.now(timezone.utc)
        now_iso = _utc_iso(now)
        async with self._write_lock:
            try:
                async with self.db.execute(
                    "SELECT * FROM lifecycle_backfill_jobs WHERE id = ?",
                    (job_id,),
                ) as cursor:
                    job = await cursor.fetchone()
                if job is None:
                    raise LookupError(f"unknown lifecycle backfill job: {job_id}")
                status = LifecycleBackfillJobStatus(str(job["status"]))
                if status is LifecycleBackfillJobStatus.FAILED:
                    await self.db.rollback()
                    stored = self._row_to_lifecycle_backfill_job(job)
                    return stored
                if status not in {
                    LifecycleBackfillJobStatus.QUEUED,
                    LifecycleBackfillJobStatus.RUNNING,
                }:
                    raise ValueError("only an active lifecycle backfill job can be recovered")
                source_id = str(job["source_id"])
                source_lock = await self.db.execute(
                    "UPDATE sources SET status = status WHERE id = ?",
                    (source_id,),
                )
                if source_lock.rowcount != 1:
                    raise ValueError(f"Source not found: {source_id}")
                async with self.db.execute(
                    "SELECT * FROM lifecycle_backfill_jobs WHERE id = ?",
                    (job_id,),
                ) as cursor:
                    job = await cursor.fetchone()
                if job is None:
                    raise LookupError(f"unknown lifecycle backfill job: {job_id}")
                status = LifecycleBackfillJobStatus(str(job["status"]))
                if status not in {
                    LifecycleBackfillJobStatus.QUEUED,
                    LifecycleBackfillJobStatus.RUNNING,
                }:
                    raise ValueError("lifecycle backfill job changed during stale recovery")
                if str(job["source_id"]) != source_id:
                    raise SourceActivityConflict(f"source lifecycle activity retry identity mismatch: {job_id}")
                async with self.db.execute(
                    "SELECT source_id, capability, lease_until FROM source_activity_leases WHERE id = ?",
                    (job_id,),
                ) as cursor:
                    lease = await cursor.fetchone()
                if lease is not None:
                    if str(lease["source_id"]) != str(job["source_id"]) or lease["capability"] != job_id:
                        raise SourceActivityConflict(f"source lifecycle activity retry identity mismatch: {job_id}")
                    if str(lease["lease_until"]) > now_iso:
                        raise SourceActivityConflict(f"source lifecycle activity lease is still current: {job_id}")
                    lease_cursor = await self.db.execute(
                        "DELETE FROM source_activity_leases WHERE id = ? AND capability = ? AND lease_until <= ?",
                        (job_id, job_id, now_iso),
                    )
                    if lease_cursor.rowcount != 1:
                        raise SourceActivityConflict(
                            f"source lifecycle activity lease changed during recovery: {job_id}"
                        )
                fenced = await self.db.execute(
                    """UPDATE sources
                       SET activity_epoch = COALESCE(activity_epoch, 0) + 1
                       WHERE id = ?""",
                    (source_id,),
                )
                if fenced.rowcount != 1:
                    raise ValueError(f"Source not found: {source_id}")
                cursor = await self.db.execute(
                    """UPDATE lifecycle_backfill_jobs
                       SET status = 'failed', error = ?, completed_at = ?, updated_at = ?
                       WHERE id = ? AND status IN ('queued', 'running')""",
                    (error, now_iso, now_iso, job_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("lifecycle backfill job changed during stale recovery")
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        stored = await self.get_lifecycle_backfill_job(job_id)
        assert stored is not None
        return stored

    async def list_stale_lifecycle_backfill_job_ids(
        self,
        *,
        limit: int = 100,
    ) -> tuple[str, ...]:
        """List active lifecycle jobs whose execution lease has expired or disappeared."""

        now_iso = _now_iso()
        bounded_limit = max(1, min(int(limit), 1000))
        rows = await self.db.execute_fetchall(
            """SELECT job.id
               FROM lifecycle_backfill_jobs job
               LEFT JOIN source_activity_leases lease ON lease.id = job.id
               WHERE job.status IN ('queued', 'running')
                 AND (lease.id IS NULL OR lease.lease_until <= ?)
               ORDER BY job.created_at, job.id
               LIMIT ?""",
            (now_iso, bounded_limit),
        )
        return tuple(str(row["id"]) for row in rows)

    async def get_lifecycle_backfill_job(
        self,
        job_id: str,
    ) -> LifecycleBackfillJob | None:
        async with self.db.execute(
            "SELECT * FROM lifecycle_backfill_jobs WHERE id = ?",
            (job_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_lifecycle_backfill_job(row) if row is not None else None

    async def get_active_lifecycle_backfill_job(
        self,
        source_id: str,
    ) -> LifecycleBackfillJob | None:
        async with self.db.execute(
            """SELECT * FROM lifecycle_backfill_jobs
               WHERE source_id = ? AND status IN ('queued', 'running')
               ORDER BY created_at DESC, id DESC LIMIT 1""",
            (source_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_lifecycle_backfill_job(row) if row is not None else None

    async def list_lifecycle_backfill_jobs(
        self,
        source_id: str,
        *,
        limit: int = 20,
    ) -> list[LifecycleBackfillJob]:
        rows = await self.db.execute_fetchall(
            """SELECT * FROM lifecycle_backfill_jobs
               WHERE source_id = ? ORDER BY created_at DESC, id DESC LIMIT ?""",
            (source_id, limit),
        )
        return [self._row_to_lifecycle_backfill_job(row) for row in rows]

    async def resolve_lifecycle_cutover_finding(
        self,
        finding_id: str,
        *,
        observation_id: str,
        source_unit_id: str,
        source_activity: SourceActivityLease | None = None,
    ) -> LifecycleCutoverFinding:
        async with self._write_lock:
            try:
                async with self.db.execute(
                    "SELECT * FROM lifecycle_cutover_findings WHERE id = ?",
                    (finding_id,),
                ) as cursor:
                    finding = await cursor.fetchone()
                if finding is None:
                    raise LookupError(f"unknown lifecycle cutover finding: {finding_id}")
                source_id = str(finding["source_id"])
                await self._assert_source_activity_fence_unlocked(
                    source_id,
                    source_activity,
                )
                async with self.db.execute(
                    """SELECT 1
                       FROM memory_support_assertions msa
                       JOIN evidence_references er ON er.id = msa.evidence_reference_id
                       JOIN source_observations so ON so.id = er.observation_id
                       WHERE msa.memory_id = ? AND msa.source_id = ? AND msa.active = 1
                         AND er.observation_id = ? AND so.source_unit_id = ?
                       LIMIT 1""",
                    (
                        finding["memory_id"],
                        finding["source_id"],
                        observation_id,
                        source_unit_id,
                    ),
                ) as cursor:
                    lineage = await cursor.fetchone()
                if lineage is None:
                    raise ValueError("finding requires validated support lineage before resolution")
                now = _now_iso()
                await self.db.execute(
                    """UPDATE lifecycle_cutover_findings
                       SET status = 'resolved', observation_id = ?, source_unit_id = ?,
                           updated_at = ?, resolved_at = ?
                       WHERE id = ? AND status = 'open'""",
                    (observation_id, source_unit_id, now, now, finding_id),
                )
                await self._assert_source_activity_fence_unlocked(
                    source_id,
                    source_activity,
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        resolved = await self.get_lifecycle_cutover_finding(finding_id)
        assert resolved is not None
        return resolved

    async def retire_unprovable_lifecycle_cutover_finding(
        self,
        finding_id: str,
        *,
        source_id: str,
        reconstruction_attempt_id: str,
        operator_id: str,
        unavailable_documents: Mapping[str, str],
    ) -> LifecycleCutoverFinding:
        """Retire one unprovable Agent Session Memory and preserve its finding history."""

        resolution = build_unprovable_cutover_resolution(
            reconstruction_attempt_id=reconstruction_attempt_id,
            operator_id=operator_id,
            unavailable_documents=unavailable_documents,
        )
        normalized_unavailable = resolution["unavailable_documents"]
        assert isinstance(normalized_unavailable, dict)

        async with self._write_lock:
            try:
                async with self.db.execute(
                    "SELECT * FROM lifecycle_cutover_findings WHERE id = ?",
                    (finding_id,),
                ) as cursor:
                    finding = await cursor.fetchone()
                if finding is None:
                    raise LookupError(f"unknown lifecycle cutover finding: {finding_id}")
                if str(finding["source_id"]) != source_id:
                    raise ValueError("cutover finding source identity mismatch")
                existing_attempt = json.loads(finding["mapping_attempt_json"] or "{}")
                if str(finding["status"]) == CutoverFindingStatus.RESOLVED.value:
                    if existing_attempt.get("resolution") != resolution:
                        raise ValueError("idempotent retirement evidence mismatch")
                    await self.db.rollback()
                    return self._row_to_lifecycle_cutover_finding(finding)
                if str(finding["reason"]) != CutoverFindingReason.MISSING_SOURCE_PROVENANCE.value:
                    raise ValueError("only missing_source_provenance findings may be retired as unprovable")

                cursor = await self.db.execute(
                    "UPDATE sources SET status = status WHERE id = ?",
                    (source_id,),
                )
                if cursor.rowcount != 1:
                    raise ValueError(f"Source not found: {source_id}")
                cursor = await self.db.execute(
                    "UPDATE memories SET status = status WHERE id = ?",
                    (finding["memory_id"],),
                )
                if cursor.rowcount != 1:
                    raise ValueError("unprovable retirement Memory is unavailable")
                async with self.db.execute(
                    "SELECT * FROM lifecycle_cutover_findings WHERE id = ?",
                    (finding_id,),
                ) as cursor:
                    locked_finding = await cursor.fetchone()
                if (
                    locked_finding is None
                    or str(locked_finding["status"]) != CutoverFindingStatus.OPEN.value
                    or str(locked_finding["source_id"]) != source_id
                    or str(locked_finding["memory_id"]) != str(finding["memory_id"])
                ):
                    raise ValueError("unprovable retirement finding stale guard failed")
                finding = locked_finding
                existing_attempt = json.loads(finding["mapping_attempt_json"] or "{}")

                async with self.db.execute(
                    "SELECT type, status, access_state FROM sources WHERE id = ?",
                    (source_id,),
                ) as cursor:
                    source = await cursor.fetchone()
                if (
                    source is None
                    or str(source["type"]) != "agent_session"
                    or str(source["status"]) != "active"
                    or str(source["access_state"] or "active") != "active"
                ):
                    raise ValueError("unprovable retirement requires an active Agent Session source")

                async with self.db.execute(
                    "SELECT status FROM memories WHERE id = ?",
                    (finding["memory_id"],),
                ) as cursor:
                    memory = await cursor.fetchone()
                if memory is None or str(memory["status"]) != "active":
                    raise ValueError("unprovable retirement requires an active Memory")

                async with self.db.execute(
                    "SELECT doc_id, source_id, source_type FROM memory_sources WHERE memory_id = ?",
                    (finding["memory_id"],),
                ) as cursor:
                    source_rows = [dict(row) async for row in cursor]
                available = json.loads(finding["available_provenance_json"] or "{}")
                validate_unprovable_cutover_evidence(
                    available_provenance=available,
                    mapping_attempt=existing_attempt,
                    source_rows=source_rows,
                    source_id=source_id,
                    unavailable_documents=normalized_unavailable,
                )

                async with self.db.execute(
                    "SELECT 1 FROM memory_support_assertions WHERE memory_id = ? AND active = 1 LIMIT 1",
                    (finding["memory_id"],),
                ) as cursor:
                    if await cursor.fetchone() is not None:
                        raise ValueError("unprovable retirement rejected while active support remains")

                now = _now_iso()
                existing_attempt["resolution"] = resolution
                plan_id = unprovable_cutover_retirement_plan_id(finding_id)
                payload = {
                    "operation": "retire_unprovable_lifecycle_cutover_finding",
                    "finding_id": finding_id,
                    "source_id": source_id,
                    "memory_id": str(finding["memory_id"]),
                    "resolution": resolution,
                }
                payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
                payload_hash = hashlib.sha256(payload_json.encode()).hexdigest()
                await self.db.execute(
                    """INSERT INTO lifecycle_plans (
                           id, reconciliation_scope_id, source_id, source_unit_id,
                           target_unit_revision_id, status, payload_json, payload_hash,
                           created_at, applied_at, error
                       ) VALUES (?, 'cutover_unprovable_retirement', ?, ?, NULL, 'staged', ?, ?, ?, NULL, NULL)""",
                    (plan_id, source_id, f"cutover-finding:{finding_id}", payload_json, payload_hash, now),
                )
                cursor = await self.db.execute(
                    """UPDATE memories SET status = 'retired',
                           retirement_reason = 'unprovable_source_lineage',
                           retired_at = ?, valid_until = ?, updated_at = ?
                       WHERE id = ? AND status = 'active'""",
                    (now, _today_iso(), now, finding["memory_id"]),
                )
                if cursor.rowcount != 1:
                    raise ValueError("unprovable retirement Memory stale guard failed")
                await self._stale_pending_reviews_unlocked(
                    (str(finding["memory_id"]),),
                    now=now,
                )
                await self._rebuild_memory_fts_unlocked(
                    str(finding["memory_id"]),
                    search_visible_statuses=set(allowed_search_statuses()),
                )
                cursor = await self.db.execute(
                    """UPDATE lifecycle_cutover_findings
                       SET status = 'resolved', mapping_attempt_json = ?,
                           observation_id = NULL, source_unit_id = NULL,
                           updated_at = ?, resolved_at = ?
                       WHERE id = ? AND status = 'open'""",
                    (json.dumps(existing_attempt, sort_keys=True), now, now, finding_id),
                )
                if cursor.rowcount != 1:
                    raise ValueError("unprovable retirement finding stale guard failed")
                await self._enqueue_lifecycle_vector_task_unlocked(
                    plan_id,
                    str(finding["memory_id"]),
                    LifecycleVectorOperation.DELETE,
                    now=now,
                )
                await self.db.execute(
                    "UPDATE lifecycle_plans SET status = 'applied', applied_at = ? WHERE id = ?",
                    (now, plan_id),
                )
                await self.db.commit()
            except BaseException:
                rollback_task = asyncio.create_task(self.db.rollback())
                await _drain_task_despite_cancellation(rollback_task)
                raise
        resolved = await self.get_lifecycle_cutover_finding(finding_id)
        assert resolved is not None
        return resolved

    async def record_evidence_references(
        self,
        evidence_unit_id: str,
        references: Sequence[EvidenceReference],
        *,
        source_activity: SourceActivityLease | None = None,
    ) -> tuple[EvidenceReference, ...]:
        revision_ids = {item.anchor.observation_revision_id for item in references}
        if revision_ids:
            placeholders = ", ".join("?" for _ in revision_ids)
            async with self.db.execute(
                f"SELECT id FROM source_observation_revisions WHERE id IN ({placeholders})",
                tuple(revision_ids),
            ) as cursor:
                available = {row["id"] async for row in cursor}
        else:
            available = set()
        validated = validate_evidence_references(tuple(references), available_revision_ids=available)
        persisted = tuple(
            EvidenceReference(
                id=item.id or evidence_reference_id_for(evidence_unit_id, item),
                evidence_unit_id=evidence_unit_id,
                role=item.role,
                anchor=item.anchor,
            )
            for item in validated
        )
        async with self._write_lock:
            try:
                async with self.db.execute(
                    "SELECT source_id FROM evidence_units WHERE id = ?",
                    (evidence_unit_id,),
                ) as cursor:
                    evidence_unit = await cursor.fetchone()
                if evidence_unit is None:
                    raise ValueError("evidence reference requires a persisted Evidence Unit")
                source_id = str(evidence_unit["source_id"])
                await self._assert_source_activity_fence_unlocked(
                    source_id,
                    source_activity,
                )
                for item in persisted:
                    anchor = item.anchor
                    await self.db.execute(
                        """INSERT OR IGNORE INTO evidence_references (
                            id, evidence_unit_id, role, anchor_kind, observation_id,
                            observation_revision_id, fragment_id, range_start, range_end, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            item.id,
                            evidence_unit_id,
                            item.role.value,
                            anchor.kind.value,
                            anchor.observation_id,
                            anchor.observation_revision_id,
                            anchor.fragment_id,
                            anchor.range_start,
                            anchor.range_end,
                            _now_iso(),
                        ),
                    )
                await self._assert_source_activity_fence_unlocked(
                    source_id,
                    source_activity,
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        return persisted

    async def upsert_memory_support_assertion(
        self,
        assertion: MemorySupportAssertion,
        *,
        source_activity: SourceActivityLease | None = None,
    ) -> None:
        async with self._write_lock:
            try:
                await self._assert_source_activity_fence_unlocked(
                    assertion.source_id,
                    source_activity,
                )
                async with self.db.execute(
                    "SELECT id FROM memories WHERE id = ?",
                    (assertion.memory_id,),
                ) as cursor:
                    memory = await cursor.fetchone()
                if memory is None:
                    raise ValueError("support assertion references unknown Memory")
                async with self.db.execute(
                    """SELECT er.role, eu.source_id AS evidence_source_id
                       FROM evidence_references er
                       JOIN evidence_units eu ON eu.id = er.evidence_unit_id
                       WHERE er.id = ?""",
                    (assertion.evidence_reference_id,),
                ) as cursor:
                    reference = await cursor.fetchone()
                if reference is None:
                    raise ValueError("support assertion references unknown evidence")
                if EvidenceRole(reference["role"]) is EvidenceRole.CONTEXT:
                    raise ValueError("context evidence cannot grant Memory support")
                if reference["evidence_source_id"] != assertion.source_id:
                    raise ValueError("support assertion source does not match evidence source")
                now = _now_iso()
                await self.db.execute(
                    """INSERT INTO memory_support_assertions (
                        id, memory_id, evidence_reference_id, source_id,
                        access_context_hash, active, created_at, removed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(memory_id, evidence_reference_id) DO UPDATE SET
                        access_context_hash=excluded.access_context_hash,
                        active=excluded.active,
                        removed_at=excluded.removed_at""",
                    (
                        assertion.id,
                        assertion.memory_id,
                        assertion.evidence_reference_id,
                        assertion.source_id,
                        assertion.access_context_hash,
                        1 if assertion.active else 0,
                        assertion.created_at or now,
                        assertion.removed_at,
                    ),
                )
                await self._assert_source_activity_fence_unlocked(
                    assertion.source_id,
                    source_activity,
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def get_memory_support_set_hash(self, memory_id: str) -> str:
        return await self._memory_support_set_hash_unlocked(memory_id)

    async def get_active_memory_support_reference_ids(self, memory_id: str) -> tuple[str, ...]:
        rows = await self.db.execute_fetchall(
            """SELECT evidence_reference_id
               FROM memory_support_assertions
               WHERE memory_id = ? AND active = 1
               ORDER BY evidence_reference_id""",
            (memory_id,),
        )
        return tuple(row["evidence_reference_id"] for row in rows)

    async def get_active_memory_support_evidence(
        self,
        memory_id: str,
        *,
        source_id: str | None = None,
    ) -> tuple[ActiveSupportEvidence, ...]:
        params: list[object] = [memory_id]
        source_clause = ""
        if source_id is not None:
            source_clause = " AND msa.source_id = ?"
            params.append(source_id)
        rows = await self.db.execute_fetchall(
            """SELECT msa.memory_id, msa.source_id,
                      er.id AS reference_id, er.evidence_unit_id, er.role,
                      er.anchor_kind, er.observation_id,
                      er.observation_revision_id, er.fragment_id,
                      er.range_start, er.range_end, eu.excerpt
               FROM memory_support_assertions msa
               JOIN evidence_references er ON er.id = msa.evidence_reference_id
               JOIN evidence_units eu ON eu.id = er.evidence_unit_id
               WHERE msa.memory_id = ? AND msa.active = 1"""
            + source_clause
            + " ORDER BY msa.source_id, er.id",
            tuple(params),
        )
        return tuple(
            ActiveSupportEvidence(
                memory_id=str(row["memory_id"]),
                source_id=str(row["source_id"]),
                reference_id=str(row["reference_id"]),
                evidence_unit_id=str(row["evidence_unit_id"]),
                role=EvidenceRole(str(row["role"])),
                anchor=SourceAnchor(
                    kind=AnchorKind(str(row["anchor_kind"])),
                    observation_id=str(row["observation_id"]),
                    observation_revision_id=str(row["observation_revision_id"]),
                    fragment_id=row["fragment_id"],
                    range_start=row["range_start"],
                    range_end=row["range_end"],
                ),
                excerpt=row["excerpt"],
            )
            for row in rows
        )

    async def get_source_unit_support_reference_ids(
        self,
        source_unit_id: str,
    ) -> Mapping[str, tuple[str, ...]]:
        rows = await self.db.execute_fetchall(
            """SELECT msa.memory_id, msa.evidence_reference_id
               FROM memory_support_assertions msa
               JOIN evidence_references er ON er.id = msa.evidence_reference_id
               JOIN source_observations so ON so.id = er.observation_id
               WHERE so.source_unit_id = ? AND msa.active = 1
               ORDER BY msa.memory_id, msa.evidence_reference_id""",
            (source_unit_id,),
        )
        values: dict[str, list[str]] = defaultdict(list)
        for row in rows:
            values[row["memory_id"]].append(row["evidence_reference_id"])
        return {memory_id: tuple(reference_ids) for memory_id, reference_ids in values.items()}

    async def _memory_support_set_hash_unlocked(self, memory_id: str) -> str:
        values: list[tuple[str, str, str]] = []
        async with self.db.execute(
            """SELECT evidence_reference_id, source_id, access_context_hash
               FROM memory_support_assertions
               WHERE memory_id = ? AND active = 1
               ORDER BY evidence_reference_id, source_id, access_context_hash""",
            (memory_id,),
        ) as cursor:
            async for row in cursor:
                values.append(
                    (
                        row["evidence_reference_id"],
                        row["source_id"],
                        row["access_context_hash"],
                    )
                )
        payload = json.dumps(values, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def get_lifecycle_plan_status(self, plan_id: str) -> str | None:
        async with self.db.execute(
            "SELECT status FROM lifecycle_plans WHERE id = ?",
            (plan_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return row["status"] if row is not None else None

    async def apply_source_projection_lifecycle(
        self,
        projection: SourceProjection,
        plan: LifecyclePlan,
        *,
        expected_source_activity_epoch: int | None = None,
    ) -> None:
        """Advance Source Projection and Memory lifecycle in one transaction."""

        if projection.source_id != plan.scope.source_id:
            raise ValueError("projection and lifecycle plan belong to different sources")
        if len(projection.deltas) != 1:
            raise ValueError("atomic projected lifecycle requires exactly one Revision Delta")
        delta = projection.deltas[0]
        if (
            delta.source_unit_id != plan.scope.source_unit_id
            or delta.current_unit_revision_id != plan.scope.target_unit_revision_id
        ):
            raise ValueError("projection and lifecycle plan target different Source Unit revisions")
        async with self._write_lock:
            try:
                await self.record_source_projection(
                    projection,
                    expected_source_activity_epoch=expected_source_activity_epoch,
                    _manage_transaction=False,
                )
                await self._apply_lifecycle_plan_with_identity_policy(
                    plan,
                    _manage_transaction=False,
                    claim_identity_policy=ClaimIdentityPolicy.ORDINARY_EXTRACTION,
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def apply_agent_claim_source_projection_lifecycle(
        self,
        projection: SourceProjection,
        plan: LifecyclePlan,
        *,
        memory_id: str,
        relation_outcome: RelationOutcomeBundle | None,
        claim_id: str,
        concept_id: str,
        display_anchor: str,
        claim_text: str,
        memory_type: str,
        tags: list[str],
        confidence: float,
        observed_at: datetime,
        citations: list[str] | None = None,
        concept_projection: dict[str, Any] | None = None,
        concept_markdown_body: str | None = None,
    ) -> None:
        """Commit projected Memory lifecycle and the Agent Knowledge view atomically."""

        plan.validate()
        if projection.source_id != plan.scope.source_id or len(projection.deltas) != 1:
            raise ValueError("agent projection and lifecycle plan scope mismatch")
        delta = projection.deltas[0]
        if (
            delta.source_unit_id != plan.scope.source_unit_id
            or delta.current_unit_revision_id != plan.scope.target_unit_revision_id
        ):
            raise ValueError("agent projection and lifecycle plan revision mismatch")
        created_memory_ids = {
            mutation.memory_id
            for mutation in plan.mutations
            if mutation.mutation_type is LifecycleMutationType.CREATE_MEMORY
        }
        retired_memory_ids = {
            mutation.memory_id
            for mutation in plan.mutations
            if mutation.mutation_type is LifecycleMutationType.RETIRE_MEMORY
        }
        if memory_id not in created_memory_ids | retired_memory_ids:
            raise ValueError("agent claim must reference a Memory created or retired by its Lifecycle Plan")
        observed = _utc_iso(observed_at)
        async with self._write_lock:
            try:
                await self.record_source_projection(
                    projection,
                    _manage_transaction=False,
                )
                await self._apply_lifecycle_plan_with_identity_policy(
                    plan,
                    _manage_transaction=False,
                    claim_identity_policy=ClaimIdentityPolicy.EXPLICIT_CONCEPT,
                )
                if relation_outcome is not None:
                    await self._record_relation_outcome_bundle_unlocked(relation_outcome)
                if concept_projection is not None:
                    await self._upsert_agent_concept_unlocked(
                        **concept_projection,
                        observed=observed,
                    )
                await self._upsert_agent_claim_unlocked(
                    claim_id=claim_id,
                    concept_id=concept_id,
                    display_anchor=display_anchor,
                    claim_text=claim_text,
                    memory_type=memory_type,
                    tags=tags,
                    confidence=confidence,
                    memory_id=memory_id,
                    observed=observed,
                )
                for citation_url in citations or ():
                    await self._add_agent_claim_citation_unlocked(
                        claim_id=claim_id,
                        citation_url=citation_url,
                        observed=observed,
                    )
                if concept_markdown_body is not None:
                    await self._update_agent_concept_markdown_unlocked(
                        concept_id=concept_id,
                        markdown_body=concept_markdown_body,
                        observed=observed,
                    )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def apply_lifecycle_plan(
        self,
        plan: LifecyclePlan,
    ) -> None:
        await self._apply_lifecycle_plan_with_identity_policy(
            plan,
            _manage_transaction=True,
            claim_identity_policy=ClaimIdentityPolicy.ORDINARY_EXTRACTION,
        )

    async def _apply_lifecycle_plan_with_identity_policy(
        self,
        plan: LifecyclePlan,
        *,
        _manage_transaction: bool,
        claim_identity_policy: ClaimIdentityPolicy,
    ) -> None:
        """Validate stale guards and commit the complete lifecycle plan once."""

        plan.validate()
        payload = lifecycle_plan_to_payload(plan)
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        transaction_lock = self._write_lock if _manage_transaction else nullcontext()
        async with transaction_lock:
            try:
                await self.db.execute(
                    "UPDATE sources SET status = status WHERE id = ?",
                    (plan.scope.source_id,),
                )
                async with self.db.execute(
                    "SELECT payload_hash, status FROM lifecycle_plans WHERE id = ?",
                    (plan.id,),
                ) as cursor:
                    existing = await cursor.fetchone()
                if existing is not None:
                    if existing["payload_hash"] != payload_hash:
                        raise ValueError("lifecycle plan retry payload mismatch")
                    if existing["status"] == "applied":
                        if _manage_transaction:
                            await self.db.commit()
                        return
                    raise ValueError(f"lifecycle plan is already {existing['status']}")

                async with self.db.execute(
                    "SELECT state FROM source_lifecycle_gates WHERE source_id = ?",
                    (plan.scope.source_id,),
                ) as cursor:
                    gate_row = await cursor.fetchone()
                gate_state = LifecycleGateState(gate_row["state"]) if gate_row else LifecycleGateState.GATED
                if gate_state is not plan.gate_state:
                    raise ValueError("lifecycle plan gate stale guard failed")
                async with self.db.execute(
                    "SELECT current_revision_id FROM source_units WHERE id = ?",
                    (plan.scope.source_unit_id,),
                ) as cursor:
                    unit = await cursor.fetchone()
                current_unit_revision = unit["current_revision_id"] if unit is not None else None
                if current_unit_revision != plan.scope.target_unit_revision_id:
                    raise ValueError("lifecycle plan Source Unit revision stale guard failed")
                if plan.stale_guard.observation_revision_ids:
                    placeholders = ", ".join("?" for _ in plan.stale_guard.observation_revision_ids)
                    async with self.db.execute(
                        f"""SELECT COUNT(*) AS count FROM source_observations
                            WHERE current_revision_id IN ({placeholders})""",
                        tuple(plan.stale_guard.observation_revision_ids),
                    ) as cursor:
                        current_count = int((await cursor.fetchone())["count"])
                    if current_count != len(set(plan.stale_guard.observation_revision_ids)):
                        raise ValueError("lifecycle plan observation revision stale guard failed")
                for memory_id, expected_hash in plan.stale_guard.support_set_hashes.items():
                    actual_hash = await self._memory_support_set_hash_unlocked(memory_id)
                    if actual_hash != expected_hash:
                        raise ValueError(f"lifecycle plan support stale guard failed: {memory_id}")
                for memory_id, expected_version in plan.stale_guard.memory_versions.items():
                    async with self.db.execute(
                        "SELECT status, content_hash, updated_at FROM memories WHERE id = ?",
                        (memory_id,),
                    ) as cursor:
                        memory_row = await cursor.fetchone()
                    actual_version = _lifecycle_memory_version(memory_row)
                    if actual_version != expected_version:
                        raise ValueError(f"lifecycle plan Memory stale guard failed: {memory_id}")
                if claim_identity_policy is ClaimIdentityPolicy.ORDINARY_EXTRACTION:
                    await self._assert_no_active_exact_claim_conflicts_unlocked(plan)

                now = _now_iso()
                await self.db.execute(
                    """INSERT INTO lifecycle_plans (
                        id, reconciliation_scope_id, source_id, source_unit_id,
                        target_unit_revision_id, status, payload_json, payload_hash,
                        created_at, applied_at, error
                    ) VALUES (?, ?, ?, ?, ?, 'staged', ?, ?, ?, NULL, NULL)""",
                    (
                        plan.id,
                        plan.scope.id,
                        plan.scope.source_id,
                        plan.scope.source_unit_id,
                        plan.scope.target_unit_revision_id,
                        payload_json,
                        payload_hash,
                        now,
                    ),
                )
                await self._stage_lifecycle_evidence_unlocked(plan, now=now)
                for mutation in plan.mutations:
                    await self._apply_lifecycle_mutation_unlocked(plan.id, mutation, now=now)
                for request in plan.relation_discovery_requests:
                    await self._enqueue_relation_discovery_work_unlocked(
                        plan.id,
                        request,
                        now=now,
                    )
                for memory_id in {mutation.memory_id for mutation in plan.mutations}:
                    async with self.db.execute(
                        "SELECT COUNT(*) AS total FROM memory_sources WHERE memory_id = ?",
                        (memory_id,),
                    ) as cursor:
                        source_count = await cursor.fetchone()
                    await self.db.execute(
                        "UPDATE memories SET corroboration_count = ? WHERE id = ?",
                        (int(source_count["total"] or 0), memory_id),
                    )
                await self._validate_projected_support_invariant_unlocked(plan)
                await self.db.execute(
                    "UPDATE lifecycle_plans SET status = 'applied', applied_at = ? WHERE id = ?",
                    (now, plan.id),
                )
                if _manage_transaction:
                    await self.db.commit()
            except Exception:
                if _manage_transaction:
                    await self.db.rollback()
                raise

    async def _assert_no_active_exact_claim_conflicts_unlocked(
        self,
        plan: LifecyclePlan,
    ) -> None:
        """Fail stale CREATE plans after another Unit committed the exact claim."""

        planned_claims: set[tuple[str, str, str | None, str | None]] = set()
        for mutation in plan.mutations:
            if mutation.mutation_type is not LifecycleMutationType.CREATE_MEMORY:
                continue
            raw = mutation.payload.get("memory")
            if not isinstance(raw, Mapping):
                raise ValueError("create_memory mutation requires memory payload")
            claim = (
                str(raw.get("content_hash") or content_hash(str(raw.get("content") or ""))),
                str(raw.get("visibility") or "workspace"),
                raw.get("owner_user_id") if isinstance(raw.get("owner_user_id"), str) else None,
                raw.get("repo_identifier") if isinstance(raw.get("repo_identifier"), str) else None,
            )
            if claim in planned_claims:
                raise ValueError("lifecycle plan contains duplicate exact claim creates")
            planned_claims.add(claim)
            async with self.db.execute(
                """SELECT m.id FROM memories AS m
                    WHERE m.content_hash = ? AND m.status = 'active'
                      AND m.visibility = ? AND m.owner_user_id IS ?
                      AND m.repo_identifier IS ? AND m.id <> ?
                      AND NOT EXISTS (
                          SELECT 1 FROM agent_claims AS ac
                          WHERE ac.memory_id = m.id
                      )
                    ORDER BY m.created_at, m.id LIMIT 1""",
                (*claim, mutation.memory_id),
            ) as cursor:
                if await cursor.fetchone() is not None:
                    raise ValueError(
                        "lifecycle plan exact claim stale guard failed: canonical active Memory now exists"
                    )

    async def _validate_projected_support_invariant_unlocked(
        self,
        plan: LifecyclePlan,
    ) -> None:
        """Fail closed when committed source support is not current, stable lineage.

        A pending Review intentionally preserves only its exact incumbent edge
        while a human decides. The current Plan's staged Reviews and prior
        applied Reviews whose own Source Unit revision remains current share
        that contract. Every other surviving same-source assertion must point
        to an Observation current in its own stable Source Unit. A newly
        activated Memory must additionally gain support in the Plan's Unit.
        """

        created_ids = {
            mutation.memory_id
            for mutation in plan.mutations
            if mutation.mutation_type is LifecycleMutationType.CREATE_MEMORY
        }
        reactivated_ids = {
            mutation.memory_id
            for mutation in plan.mutations
            if mutation.mutation_type is LifecycleMutationType.REACTIVATE_MEMORY
        }
        candidate_ids = (
            set(plan.coverage_proof.mandatory_incumbent_ids)
            | created_ids
            | reactivated_ids
            | {
                mutation.memory_id
                for mutation in plan.mutations
                if mutation.mutation_type is LifecycleMutationType.ATTACH_SUPPORT
            }
        )
        contested_supports = set(pending_review_contested_supports(plan))
        contested_supports.update(
            await self._durable_pending_review_contested_supports_unlocked(
                source_id=plan.scope.source_id,
                memory_ids=candidate_ids,
            )
        )
        for memory_id in sorted(candidate_ids):
            async with self.db.execute(
                "SELECT status FROM memories WHERE id = ?",
                (memory_id,),
            ) as cursor:
                memory = await cursor.fetchone()
            if memory is None or memory["status"] != "active":
                continue
            async with self.db.execute(
                """SELECT msa.evidence_reference_id,
                          er.observation_id,
                          eu.source_id AS evidence_source_id,
                          so.source_id AS observation_source_id,
                          su.source_id AS unit_source_id,
                          eu.source_lineage_id,
                          so.source_unit_id,
                          er.observation_revision_id,
                          sor.observation_id AS revision_observation_id,
                          so.current_revision_id
                   FROM memory_support_assertions msa
                   LEFT JOIN evidence_references er ON er.id = msa.evidence_reference_id
                   LEFT JOIN evidence_units eu ON eu.id = er.evidence_unit_id
                   LEFT JOIN source_observations so ON so.id = er.observation_id
                   LEFT JOIN source_observation_revisions sor
                     ON sor.id = er.observation_revision_id
                   LEFT JOIN source_units su ON su.id = so.source_unit_id
                   WHERE msa.memory_id = ? AND msa.source_id = ? AND msa.active = 1""",
                (memory_id, plan.scope.source_id),
            ) as cursor:
                supports = await cursor.fetchall()
            structurally_valid = [
                support
                for support in supports
                if support["evidence_source_id"] == plan.scope.source_id
                and support["observation_source_id"] == plan.scope.source_id
                and support["unit_source_id"] == plan.scope.source_id
                and support["source_lineage_id"] == support["source_unit_id"]
                and support["revision_observation_id"] == support["observation_id"]
            ]
            current_supports = [
                support
                for support in structurally_valid
                if support["observation_revision_id"] == support["current_revision_id"]
            ]
            current_scope_total = sum(
                support["source_unit_id"] == plan.scope.source_unit_id for support in current_supports
            )
            if memory_id in created_ids | reactivated_ids and current_scope_total == 0:
                raise ValueError(f"projected lifecycle activated Memory without source support: {memory_id}")
            accepted_supports = [
                support
                for support in structurally_valid
                if support["observation_revision_id"] == support["current_revision_id"]
                or ContestedSupportEdge(
                    memory_id=memory_id,
                    source_id=plan.scope.source_id,
                    source_unit_id=support["source_unit_id"],
                    evidence_reference_id=support["evidence_reference_id"],
                )
                in contested_supports
            ]
            if len(accepted_supports) != len(supports):
                raise ValueError(f"projected lifecycle left stale or ambiguous source support: {memory_id}")

    async def _durable_pending_review_contested_supports_unlocked(
        self,
        *,
        source_id: str,
        memory_ids: set[str],
    ) -> frozenset[ContestedSupportEdge]:
        """Load exact contested edges from previously applied pending Reviews."""

        ordered_memory_ids = sorted(memory_ids)
        if not ordered_memory_ids:
            return frozenset()
        placeholders = ", ".join("?" for _ in ordered_memory_ids)
        rows = await self.db.execute_fetchall(
            f"""SELECT lr.incumbent_memory_id,
                       lp.source_id AS plan_source_id,
                       lp.source_unit_id AS plan_source_unit_id,
                       lr.staged_evidence_json
                  FROM lifecycle_reviews lr
                  JOIN lifecycle_plans lp ON lp.id = lr.lifecycle_plan_id
                  JOIN source_units su ON su.id = lp.source_unit_id
                 WHERE lr.status = 'pending'
                   AND lp.status = 'applied'
                   AND lp.source_id = ?
                   AND su.source_id = lp.source_id
                   AND lp.target_unit_revision_id = su.current_revision_id
                   AND lr.incumbent_memory_id IN ({placeholders})""",
            (source_id, *ordered_memory_ids),
        )
        contested: set[ContestedSupportEdge] = set()
        for row in rows:
            staged = json.loads(row["staged_evidence_json"] or "{}")
            if not isinstance(staged, Mapping):
                raise ValueError("pending lifecycle review requires staged_evidence")
            contested.update(
                contested_supports_from_staged_evidence(
                    incumbent_memory_id=row["incumbent_memory_id"],
                    source_id=row["plan_source_id"],
                    source_unit_id=row["plan_source_unit_id"],
                    staged_evidence=staged,
                )
            )
        return frozenset(contested)

    async def _stage_lifecycle_evidence_unlocked(
        self,
        plan: LifecyclePlan,
        *,
        now: str,
    ) -> None:
        """Stage revision-pinned Evidence inside the Lifecycle Plan transaction."""

        for unit in plan.evidence_units:
            await self.db.execute(
                """INSERT INTO evidence_units (
                    id, source_id, doc_id, doc_revision_id, source_type, client,
                    repo_identifier, source_anchor, source_lineage_id,
                    source_metadata_json, project_key, visibility, owner_user_id,
                    observed_at, extractor_run_id, access_context_hash, content,
                    excerpt, evidence_provenance, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    observed_at=excluded.observed_at,
                    extractor_run_id=excluded.extractor_run_id,
                    access_context_hash=excluded.access_context_hash,
                    updated_at=excluded.updated_at""",
                (
                    unit.id,
                    unit.source_id,
                    unit.doc_id,
                    unit.doc_revision_id,
                    unit.source_type,
                    unit.client,
                    unit.repo_identifier,
                    unit.source_anchor,
                    unit.source_lineage_id,
                    json.dumps(dict(unit.source_metadata), sort_keys=True),
                    _normalize_project_key(unit.project_key),
                    unit.visibility,
                    unit.owner_user_id,
                    unit.observed_at,
                    unit.extractor_run_id,
                    unit.access_context_hash,
                    unit.content,
                    unit.excerpt,
                    unit.evidence_provenance.value,
                    now,
                    now,
                ),
            )
        for reference in plan.evidence_references:
            anchor = reference.anchor
            await self.db.execute(
                """INSERT OR IGNORE INTO evidence_references (
                    id, evidence_unit_id, role, anchor_kind, observation_id,
                    observation_revision_id, fragment_id, range_start, range_end, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    reference.id,
                    reference.evidence_unit_id,
                    reference.role.value,
                    anchor.kind.value,
                    anchor.observation_id,
                    anchor.observation_revision_id,
                    anchor.fragment_id,
                    anchor.range_start,
                    anchor.range_end,
                    now,
                ),
            )

    async def _apply_lifecycle_mutation_unlocked(self, plan_id: str, mutation, *, now: str) -> None:
        mutation_type = mutation.mutation_type
        if mutation_type is LifecycleMutationType.CREATE_MEMORY:
            raw = mutation.payload.get("memory")
            if not isinstance(raw, Mapping):
                raise ValueError("create_memory mutation requires memory payload")
            content = str(raw.get("content") or "")
            memory = Memory(
                id=mutation.memory_id,
                memory_type=str(raw.get("memory_type") or "fact"),
                content=content,
                content_hash=str(raw.get("content_hash") or content_hash(content)),
                visibility=str(raw.get("visibility") or "workspace"),
                owner_user_id=raw.get("owner_user_id") if isinstance(raw.get("owner_user_id"), str) else None,
                project_key=raw.get("project_key") if isinstance(raw.get("project_key"), str) else None,
                repo_identifier=raw.get("repo_identifier") if isinstance(raw.get("repo_identifier"), str) else None,
                entity_refs=[str(value) for value in raw.get("entity_refs", [])],
                tags=[str(value) for value in raw.get("tags", [])],
                confidence=float(raw.get("confidence", 0.7)),
                valid_from=_parse_date(raw.get("valid_from")),
                valid_until=_parse_date(raw.get("valid_until")),
                extraction_context=(
                    str(raw.get("extraction_context")) if raw.get("extraction_context") is not None else None
                ),
            )
            await self._insert_memory_unlocked(memory)
            await self._link_memory_entities_unlocked(
                memory.id,
                [int(value) for value in raw.get("entity_ids", [])],
            )
            document_source = raw.get("document_source")
            if isinstance(document_source, Mapping):
                doc_id = document_source.get("doc_id")
                source_type = document_source.get("source_type")
                if not isinstance(doc_id, str) or not doc_id:
                    raise ValueError("create_memory document_source requires doc_id")
                if not isinstance(source_type, str) or not source_type:
                    raise ValueError("create_memory document_source requires source_type")
                await self._add_memory_source_unlocked(
                    memory.id,
                    doc_id,
                    source_type,
                    (str(document_source.get("excerpt")) if document_source.get("excerpt") is not None else None),
                    source_updated_at=_parse_dt(document_source.get("source_updated_at")),
                )
            await self._enqueue_lifecycle_vector_task_unlocked(
                plan_id,
                memory.id,
                LifecycleVectorOperation.UPSERT,
                now=now,
            )
            return
        if mutation_type is LifecycleMutationType.REACTIVATE_MEMORY:
            expected_content_hash = mutation.payload.get("expected_content_hash")
            if not isinstance(expected_content_hash, str) or not expected_content_hash:
                raise ValueError("reactivate_memory requires expected_content_hash")
            expected_retirement_reason = mutation.payload.get(
                "expected_retirement_reason",
                "source_rebaseline",
            )
            if not isinstance(expected_retirement_reason, str) or not expected_retirement_reason:
                raise ValueError("reactivate_memory requires expected_retirement_reason")
            cursor = await self.db.execute(
                """UPDATE memories
                      SET status = 'active', retirement_reason = NULL,
                          retired_at = NULL, valid_until = NULL, updated_at = ?
                    WHERE id = ? AND status = 'retired'
                      AND retirement_reason = ?
                      AND content_hash = ?""",
                (
                    now,
                    mutation.memory_id,
                    expected_retirement_reason,
                    expected_content_hash,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("reactivate Memory stale guard failed")
            await self._rebuild_memory_fts_unlocked(
                mutation.memory_id,
                search_visible_statuses=set(allowed_search_statuses()),
            )
            return
        if mutation_type is LifecycleMutationType.ATTACH_SUPPORT:
            access_hash = mutation.payload.get("access_context_hash")
            if not isinstance(access_hash, str) or not access_hash:
                raise ValueError("attach_support mutation requires access_context_hash")
            for reference_id in mutation.evidence_reference_ids:
                async with self.db.execute(
                    """SELECT er.role, eu.source_id, eu.doc_id, eu.source_type,
                              eu.excerpt, eu.observed_at
                       FROM evidence_references er
                       JOIN evidence_units eu ON eu.id = er.evidence_unit_id
                       WHERE er.id = ?""",
                    (reference_id,),
                ) as cursor:
                    reference = await cursor.fetchone()
                if reference is None or EvidenceRole(reference["role"]) is EvidenceRole.CONTEXT:
                    raise ValueError("attach_support requires support-granting evidence")
                if reference["source_id"] != mutation.source_id:
                    raise ValueError("attach_support evidence belongs to another source")
                support_id = (
                    "support-"
                    + hashlib.sha256(f"{mutation.memory_id}\x1f{reference_id}".encode("utf-8")).hexdigest()[:20]
                )
                await self.db.execute(
                    """INSERT INTO memory_support_assertions (
                        id, memory_id, evidence_reference_id, source_id,
                        access_context_hash, active, created_at, removed_at
                    ) VALUES (?, ?, ?, ?, ?, 1, ?, NULL)
                    ON CONFLICT(memory_id, evidence_reference_id) DO UPDATE SET
                        access_context_hash=excluded.access_context_hash,
                        active=1, removed_at=NULL""",
                    (
                        support_id,
                        mutation.memory_id,
                        reference_id,
                        mutation.source_id,
                        access_hash,
                        now,
                    ),
                )
                await self._corroborate_memory_unlocked(
                    mutation.memory_id,
                    str(reference["doc_id"]),
                    str(reference["source_type"]),
                    reference["excerpt"],
                    support_kind="corroborated",
                    source_updated_at=_parse_dt(mutation.payload.get("source_updated_at")),
                )
            return
        if mutation_type is LifecycleMutationType.REMOVE_SUPPORT:
            placeholders = ", ".join("?" for _ in mutation.evidence_reference_ids)
            cursor = await self.db.execute(
                f"""UPDATE memory_support_assertions
                    SET active = 0, removed_at = ?
                    WHERE memory_id = ? AND source_id = ? AND active = 1
                      AND evidence_reference_id IN ({placeholders})""",
                (now, mutation.memory_id, mutation.source_id, *mutation.evidence_reference_ids),
            )
            if cursor.rowcount != len(set(mutation.evidence_reference_ids)):
                raise ValueError("remove_support mutation did not match complete active support set")
            document_id = mutation.payload.get("document_id")
            if isinstance(document_id, str) and document_id:
                await self.db.execute(
                    "DELETE FROM memory_sources WHERE memory_id = ? AND doc_id = ? AND source_id = ?",
                    (mutation.memory_id, document_id, mutation.source_id),
                )
                await self.db.execute(
                    "DELETE FROM memory_search_metadata_fts WHERE memory_id = ? AND doc_id = ?",
                    (mutation.memory_id, document_id),
                )
                await self.db.execute(
                    "DELETE FROM memory_search_metadata_alias_fts WHERE memory_id = ? AND doc_id = ?",
                    (mutation.memory_id, document_id),
                )
                await self.db.execute(
                    "DELETE FROM memory_search_metadata_trigram WHERE memory_id = ? AND doc_id = ?",
                    (mutation.memory_id, document_id),
                )
            return
        if mutation_type is LifecycleMutationType.SUPERSEDE_MEMORY:
            assert mutation.replacement_memory_id is not None
            async with self.db.execute(
                """SELECT 1 FROM memory_support_assertions
                   WHERE memory_id = ? AND active = 1 LIMIT 1""",
                (mutation.memory_id,),
            ) as cursor:
                if await cursor.fetchone() is not None:
                    raise ValueError("cannot supersede Memory while active support remains")
            async with self.db.execute(
                "SELECT 1 FROM memories WHERE id = ?",
                (mutation.replacement_memory_id,),
            ) as cursor:
                if await cursor.fetchone() is None:
                    raise ValueError("supersede replacement Memory does not exist in plan")
            cursor = await self.db.execute(
                """UPDATE memories SET status = 'superseded', superseded_by = ?,
                    valid_until = ?, superseded_at = ?, replacement_reason = ?,
                    replacement_kind = ?, updated_at = ?
                    WHERE id = ? AND status = 'active'""",
                (
                    mutation.replacement_memory_id,
                    _today_iso(),
                    now,
                    str(mutation.payload.get("reason") or "authoritative source replacement"),
                    str(mutation.payload.get("replacement_kind") or "supersession"),
                    now,
                    mutation.memory_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("supersede Memory stale guard failed")
            await self._stale_pending_reviews_unlocked(
                (mutation.memory_id,),
                now=now,
            )
            await self._rebuild_memory_fts_unlocked(
                mutation.memory_id,
                search_visible_statuses=set(allowed_search_statuses()),
            )
            await self._enqueue_lifecycle_vector_task_unlocked(
                plan_id,
                mutation.memory_id,
                LifecycleVectorOperation.DELETE,
                now=now,
            )
            return
        if mutation_type is LifecycleMutationType.RETIRE_MEMORY:
            async with self.db.execute(
                """SELECT 1 FROM memory_support_assertions
                   WHERE memory_id = ? AND active = 1 LIMIT 1""",
                (mutation.memory_id,),
            ) as cursor:
                if await cursor.fetchone() is not None:
                    raise ValueError("cannot retire Memory while active support remains")
            cursor = await self.db.execute(
                """UPDATE memories SET status = 'retired', retirement_reason = ?,
                    retired_at = ?, valid_until = ?, updated_at = ?
                    WHERE id = ? AND status = 'active'""",
                (
                    str(mutation.payload.get("reason") or "support_removed"),
                    now,
                    _today_iso(),
                    now,
                    mutation.memory_id,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("retire Memory stale guard failed")
            await self._stale_pending_reviews_unlocked(
                (mutation.memory_id,),
                now=now,
            )
            await self._rebuild_memory_fts_unlocked(
                mutation.memory_id,
                search_visible_statuses=set(allowed_search_statuses()),
            )
            await self._enqueue_lifecycle_vector_task_unlocked(
                plan_id,
                mutation.memory_id,
                LifecycleVectorOperation.DELETE,
                now=now,
            )
            return
        if mutation_type is LifecycleMutationType.CREATE_REVIEW:
            review_id = str(mutation.payload.get("review_id") or f"review-{plan_id}-{mutation.memory_id}")
            await self.db.execute(
                """INSERT INTO lifecycle_reviews (
                    id, lifecycle_plan_id, incumbent_memory_id, status,
                    staged_evidence_json, reason, created_at, resolved_at
                ) VALUES (?, ?, ?, 'pending', ?, ?, ?, NULL)""",
                (
                    review_id,
                    plan_id,
                    mutation.memory_id,
                    json.dumps(mutation.payload.get("staged_evidence", {}), sort_keys=True),
                    str(mutation.payload.get("reason") or "lifecycle review required"),
                    now,
                ),
            )
            return
        if mutation_type is LifecycleMutationType.RESOLVE_REVIEW:
            review_id = mutation.payload.get("review_id")
            status = mutation.payload.get("status")
            if not isinstance(review_id, str) or not review_id:
                raise ValueError("resolve_review mutation requires review_id")
            if status != LifecycleReviewStatus.APPROVED.value:
                raise ValueError("atomic lifecycle review resolution only supports approval")
            cursor = await self.db.execute(
                "UPDATE lifecycle_reviews SET status = 'approved', resolved_at = ? "
                "WHERE id = ? AND incumbent_memory_id = ? AND status = 'pending'",
                (now, review_id, mutation.memory_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("lifecycle review approval stale guard failed")
            return
        if mutation_type is LifecycleMutationType.REFRESH_MEMORY_INDEX:
            async with self.db.execute(
                "SELECT 1 FROM memories WHERE id = ? AND status = 'active'",
                (mutation.memory_id,),
            ) as cursor:
                if await cursor.fetchone() is None:
                    raise ValueError("refresh_memory_index requires an active Memory")
            await self._enqueue_lifecycle_vector_task_unlocked(
                plan_id,
                mutation.memory_id,
                LifecycleVectorOperation.UPSERT,
                now=now,
            )
            return
        raise ValueError(f"unsupported lifecycle mutation: {mutation_type.value}")

    async def _stale_pending_reviews_unlocked(
        self,
        memory_ids: Sequence[str],
        *,
        now: str,
    ) -> None:
        """Close all review work whose decision target became terminal."""
        unique_ids = tuple(sorted(set(memory_ids)))
        if not unique_ids:
            return
        placeholders = ", ".join("?" for _ in unique_ids)
        await self.db.execute(
            f"""UPDATE memory_reviews
                   SET status = 'stale', resolved_at = COALESCE(resolved_at, ?)
                 WHERE status = 'pending'
                   AND (
                       incumbent_memory_id IN ({placeholders})
                       OR challenger_memory_id IN ({placeholders})
                       OR id IN (
                           SELECT review_id
                             FROM memory_review_related_challengers
                            WHERE challenger_memory_id IN ({placeholders})
                       )
                   )""",
            (now, *unique_ids, *unique_ids, *unique_ids),
        )
        await self.db.execute(
            f"""UPDATE lifecycle_reviews
                   SET status = 'stale', resolved_at = COALESCE(resolved_at, ?)
                 WHERE status = 'pending'
                   AND incumbent_memory_id IN ({placeholders})
                   AND incumbent_memory_id IN (
                       SELECT id FROM memories
                        WHERE status IN ('retired', 'superseded')
                   )""",
            (now, *unique_ids),
        )

    async def _enqueue_lifecycle_vector_task_unlocked(
        self,
        plan_id: str,
        memory_id: str,
        operation: LifecycleVectorOperation,
        *,
        now: str,
    ) -> None:
        task_id = (
            "lvout-" + hashlib.sha256(f"{plan_id}\x1f{memory_id}\x1f{operation.value}".encode("utf-8")).hexdigest()[:20]
        )
        await self.db.execute(
            """INSERT OR IGNORE INTO lifecycle_vector_outbox (
                   id, lifecycle_plan_id, memory_id, operation, status,
                   attempts, error, created_at, updated_at
               ) VALUES (?, ?, ?, ?, 'pending', 0, NULL, ?, ?)""",
            (task_id, plan_id, memory_id, operation.value, now, now),
        )

    async def _enqueue_relation_discovery_work_unlocked(
        self,
        lifecycle_plan_id: str,
        request: RelationDiscoveryRequest,
        *,
        now: str,
    ) -> None:
        await self.db.execute(
            """INSERT OR IGNORE INTO relation_discovery_work (
                   id, lifecycle_plan_id, memory_id, expected_content_hash,
                   source_id, source_unit_id, source_unit_revision_id, doc_id,
                   actor_user_id, entity_ids_json, status, attempts,
                   lease_owner, lease_token, lease_until, error,
                   created_at, updated_at, completed_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', 0,
                         NULL, NULL, NULL, NULL, ?, ?, NULL)""",
            (
                request.id,
                lifecycle_plan_id,
                request.memory_id,
                request.expected_content_hash,
                request.source_id,
                request.source_unit_id,
                request.source_unit_revision_id,
                request.doc_id,
                request.actor_user_id,
                json.dumps(list(request.entity_ids)),
                now,
                now,
            ),
        )

    async def get_lifecycle_plan_payload(self, lifecycle_plan_id: str) -> Mapping[str, object] | None:
        async with self.db.execute(
            "SELECT payload_json FROM lifecycle_plans WHERE id = ?", (lifecycle_plan_id,)
        ) as cursor:
            row = await cursor.fetchone()
        return json.loads(row["payload_json"]) if row is not None else None

    async def get_lifecycle_review(self, review_id: str) -> LifecycleReview | None:
        async with self.db.execute("SELECT * FROM lifecycle_reviews WHERE id = ?", (review_id,)) as cursor:
            row = await cursor.fetchone()
        return self._row_to_lifecycle_review(row) if row is not None else None

    async def list_lifecycle_reviews(
        self,
        source_id: str,
        *,
        status: LifecycleReviewStatus | None = None,
    ) -> list[LifecycleReview]:
        params: list[object] = [source_id]
        status_clause = ""
        if status is not None:
            status_clause = " AND lr.status = ?"
            params.append(status.value)
        rows = await self.db.execute_fetchall(
            "SELECT lr.* FROM lifecycle_reviews lr "
            "JOIN lifecycle_plans lp ON lp.id = lr.lifecycle_plan_id "
            "WHERE lp.source_id = ?" + status_clause + " ORDER BY lr.created_at, lr.id",
            tuple(params),
        )
        return [self._row_to_lifecycle_review(row) for row in rows]

    async def resolve_lifecycle_review(
        self,
        review_id: str,
        status: LifecycleReviewStatus,
    ) -> LifecycleReview:
        if status not in {LifecycleReviewStatus.REJECTED, LifecycleReviewStatus.STALE}:
            raise ValueError("direct lifecycle review resolution must be rejected or stale")
        async with self._write_lock:
            cursor = await self.db.execute(
                "UPDATE lifecycle_reviews SET status = ?, resolved_at = ? WHERE id = ? AND status = 'pending'",
                (status.value, _now_iso(), review_id),
            )
            if cursor.rowcount != 1:
                raise ValueError("lifecycle review is not pending")
            await self.db.commit()
        review = await self.get_lifecycle_review(review_id)
        assert review is not None
        return review

    async def list_lifecycle_vector_tasks(
        self,
        *,
        source_id: str | None = None,
        lifecycle_plan_id: str | None = None,
        limit: int = 100,
    ) -> list[LifecycleVectorTask]:
        conditions = "WHERE lvo.status IN ('pending', 'failed')"
        params: list[object] = []
        if source_id is not None:
            conditions += " AND lp.source_id = ?"
            params.append(source_id)
        if lifecycle_plan_id is not None:
            conditions += " AND lvo.lifecycle_plan_id = ?"
            params.append(lifecycle_plan_id)
        selects = [
            f"""SELECT lvo.id, lvo.lifecycle_plan_id, lvo.memory_id,
                       lvo.operation, lvo.status, lvo.attempts, lvo.error,
                       lvo.created_at, lvo.updated_at
                  FROM lifecycle_vector_outbox lvo
                  JOIN lifecycle_plans lp ON lp.id = lvo.lifecycle_plan_id
                  {conditions}"""
        ]
        if lifecycle_plan_id is None:
            deletion_conditions = "WHERE status IN ('pending', 'failed')"
            if source_id is not None:
                deletion_conditions += " AND source_id = ?"
                params.append(source_id)
            selects.append(
                f"""SELECT id, 'source-delete:' || source_id AS lifecycle_plan_id,
                           memory_id, 'delete' AS operation, status, attempts,
                           error, created_at, updated_at
                      FROM source_deletion_vector_outbox
                      {deletion_conditions}"""
            )
        params.append(limit)
        rows = await self.db.execute_fetchall(
            "SELECT * FROM (" + " UNION ALL ".join(selects) + ") "
            "ORDER BY CASE status WHEN 'pending' THEN 0 ELSE 1 END, "
            "CASE status WHEN 'failed' THEN updated_at ELSE created_at END, id LIMIT ?",
            tuple(params),
        )
        return [
            LifecycleVectorTask(
                id=row["id"],
                lifecycle_plan_id=row["lifecycle_plan_id"],
                memory_id=row["memory_id"],
                operation=LifecycleVectorOperation(row["operation"]),
                status=LifecycleVectorTaskStatus(row["status"]),
                attempts=int(row["attempts"]),
                error=row["error"],
            )
            for row in rows
        ]

    async def complete_lifecycle_vector_task(self, task_id: str) -> None:
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE lifecycle_vector_outbox
                   SET status = 'completed', attempts = attempts + 1,
                       updated_at = ?
                   WHERE id = ? AND status IN ('pending', 'failed')""",
                (_now_iso(), task_id),
            )
            if cursor.rowcount != 1:
                cursor = await self.db.execute(
                    """UPDATE source_deletion_vector_outbox
                       SET status = 'completed', attempts = attempts + 1,
                           updated_at = ?
                       WHERE id = ? AND status IN ('pending', 'failed')""",
                    (_now_iso(), task_id),
                )
            if cursor.rowcount != 1:
                raise ValueError("lifecycle vector task is not pending")
            await self.db.commit()

    async def fail_lifecycle_vector_task(self, task_id: str, error: str) -> None:
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE lifecycle_vector_outbox
                   SET status = 'failed', attempts = attempts + 1,
                       error = COALESCE(error, ?), updated_at = ?
                   WHERE id = ? AND status IN ('pending', 'failed')""",
                (error[:4000], _now_iso(), task_id),
            )
            if cursor.rowcount != 1:
                cursor = await self.db.execute(
                    """UPDATE source_deletion_vector_outbox
                       SET status = 'failed', attempts = attempts + 1,
                           error = COALESCE(error, ?), updated_at = ?
                       WHERE id = ? AND status IN ('pending', 'failed')""",
                    (error[:4000], _now_iso(), task_id),
                )
            if cursor.rowcount != 1:
                raise ValueError("lifecycle vector task is not pending")
            await self.db.commit()

    async def has_ready_relation_discovery_work(self, *, max_attempts: int) -> bool:
        """Return whether durable relation work is ready for a lease attempt."""

        if max_attempts < 1:
            raise ValueError("relation discovery readiness requires positive max attempts")
        now = _now_iso()
        rows = await self.db.execute_fetchall(
            """SELECT 1 AS ready FROM relation_discovery_work
               WHERE attempts < ? AND (
                     status = 'pending'
                  OR (status = 'failed' AND (next_attempt_at IS NULL OR next_attempt_at <= ?))
                  OR (status = 'running' AND lease_until < ?)
               )
               LIMIT 1""",
            (max_attempts, now, now),
        )
        return bool(rows)

    async def lease_relation_discovery_work(
        self,
        *,
        worker_id: str,
        limit: int,
        lease_seconds: int,
        max_attempts: int,
    ) -> list[RelationDiscoveryWork]:
        if not worker_id or limit < 1 or lease_seconds < 1 or max_attempts < 1:
            raise ValueError("relation discovery lease requires positive bounds and worker id")
        now = _now_iso()
        lease_until = (datetime.now(timezone.utc) + timedelta(seconds=lease_seconds)).isoformat()
        leased: list[RelationDiscoveryWork] = []
        async with self._write_lock:
            try:
                await self.db.execute("BEGIN IMMEDIATE")
                rows = await self.db.execute_fetchall(
                    """SELECT * FROM relation_discovery_work
                       WHERE attempts < ?
                         AND (
                           status = 'pending'
                           OR (status = 'failed' AND (next_attempt_at IS NULL OR next_attempt_at <= ?))
                           OR (status = 'running' AND lease_until < ?)
                         )
                       ORDER BY created_at, id
                       LIMIT ?""",
                    (max_attempts, now, now, limit),
                )
                for row in rows:
                    lease_token = uuid.uuid4().hex
                    cursor = await self.db.execute(
                        """UPDATE relation_discovery_work
                              SET status = 'running', attempts = attempts + 1,
                                  lease_owner = ?, lease_token = ?, lease_until = ?,
                                  updated_at = ?
                            WHERE id = ? AND attempts = ?
                              AND (
                                status = 'pending'
                                OR (status = 'failed' AND (next_attempt_at IS NULL OR next_attempt_at <= ?))
                                OR (status = 'running' AND lease_until < ?)
                              )""",
                        (
                            worker_id,
                            lease_token,
                            lease_until,
                            now,
                            row["id"],
                            row["attempts"],
                            now,
                            now,
                        ),
                    )
                    if cursor.rowcount != 1:
                        continue
                    leased.append(
                        self._row_to_relation_discovery_work(
                            {
                                **dict(row),
                                "status": RelationDiscoveryWorkStatus.RUNNING.value,
                                "attempts": int(row["attempts"]) + 1,
                                "lease_owner": worker_id,
                                "lease_token": lease_token,
                                "lease_until": lease_until,
                                "next_attempt_at": row["next_attempt_at"],
                                "updated_at": now,
                            }
                        )
                    )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        return leased

    async def complete_relation_discovery_work(
        self,
        work_id: str,
        *,
        worker_id: str,
        lease_token: str,
        relation_outcome: RelationOutcomeBundle,
        reviews: Sequence[MemoryReview] = (),
    ) -> None:
        """Atomically publish one relation result and fence its leased work."""

        async with self._write_lock:
            try:
                await self.db.execute("BEGIN IMMEDIATE")
                row = await self._relation_discovery_lease_row_unlocked(
                    work_id,
                    worker_id=worker_id,
                    lease_token=lease_token,
                )
                await self._assert_relation_discovery_outcome_current_unlocked(
                    row,
                    relation_outcome,
                )
                await self._record_relation_outcome_bundle_unlocked(
                    relation_outcome,
                    update_evidence_unit=False,
                )
                for review in reviews:
                    if review.kind == "cross_source_conflict":
                        memories, supports = await self._load_cross_source_review_snapshots_unlocked(review)
                        validate_cross_source_review_write(
                            review,
                            relation_outcome,
                            memories=memories,
                            supports=supports,
                        )
                    async with self.db.execute(
                        "SELECT * FROM memory_reviews WHERE id = ?",
                        (review.id,),
                    ) as cursor:
                        existing_review = await cursor.fetchone()
                    if existing_review is not None:
                        validate_pending_review_retry(
                            review,
                            self._row_to_review(existing_review),
                        )
                        continue
                    now = _now_iso()
                    await self.db.execute(
                        """INSERT INTO memory_reviews (
                            id, kind, status, incumbent_memory_id,
                            challenger_memory_id, reason, review_note, reviewer,
                            expected_incumbent_updated_at,
                            expected_challenger_updated_at, replacement_kind,
                            created_at, resolved_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            review.id,
                            review.kind,
                            review.status,
                            review.incumbent_memory_id,
                            review.challenger_memory_id,
                            review.reason,
                            review.review_note,
                            review.reviewer,
                            review.expected_incumbent_updated_at,
                            review.expected_challenger_updated_at,
                            _validate_replacement_kind(review.replacement_kind),
                            review.created_at.isoformat() if review.created_at else now,
                            review.resolved_at.isoformat() if review.resolved_at else None,
                        ),
                    )
                completed_at = _now_iso()
                cursor = await self.db.execute(
                    """UPDATE relation_discovery_work
                          SET status = 'completed', lease_owner = NULL,
                              lease_token = NULL, lease_until = NULL,
                              next_attempt_at = NULL,
                              updated_at = ?, completed_at = ?
                        WHERE id = ? AND status = 'running'
                          AND lease_owner = ? AND lease_token = ?""",
                    (
                        completed_at,
                        completed_at,
                        work_id,
                        worker_id,
                        lease_token,
                    ),
                )
                if cursor.rowcount != 1:
                    raise ValueError("relation discovery lease was lost")
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def fail_relation_discovery_work(
        self,
        work_id: str,
        *,
        worker_id: str,
        lease_token: str,
        error: str,
        next_attempt_at: str | None,
        exhausted: bool,
    ) -> None:
        await self._finish_relation_discovery_work(
            work_id,
            worker_id=worker_id,
            lease_token=lease_token,
            status=RelationDiscoveryWorkStatus.FAILED,
            message=error,
            next_attempt_at=None if exhausted else next_attempt_at,
        )

    async def obsolete_relation_discovery_work(
        self,
        work_id: str,
        *,
        worker_id: str,
        lease_token: str,
        reason: str,
    ) -> None:
        await self._finish_relation_discovery_work(
            work_id,
            worker_id=worker_id,
            lease_token=lease_token,
            status=RelationDiscoveryWorkStatus.OBSOLETE,
            message=reason,
            next_attempt_at=None,
        )

    async def _finish_relation_discovery_work(
        self,
        work_id: str,
        *,
        worker_id: str,
        lease_token: str,
        status: RelationDiscoveryWorkStatus,
        message: str,
        next_attempt_at: str | None,
    ) -> None:
        if status not in {
            RelationDiscoveryWorkStatus.FAILED,
            RelationDiscoveryWorkStatus.OBSOLETE,
        }:
            raise ValueError("invalid relation discovery terminal status")
        now = _now_iso()
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE relation_discovery_work
                      SET status = ?, lease_owner = NULL, lease_token = NULL,
                          lease_until = NULL, next_attempt_at = ?,
                          error = COALESCE(error, ?), updated_at = ?,
                          completed_at = CASE WHEN ? = 'obsolete' THEN ? ELSE NULL END
                    WHERE id = ? AND status = 'running'
                      AND lease_owner = ? AND lease_token = ?
                      AND lease_until > ?""",
                (
                    status.value,
                    next_attempt_at,
                    message[:4000],
                    now,
                    status.value,
                    now,
                    work_id,
                    worker_id,
                    lease_token,
                    now,
                ),
            )
            if cursor.rowcount != 1:
                raise ValueError("relation discovery lease was lost")
            await self.db.commit()

    async def _relation_discovery_lease_row_unlocked(
        self,
        work_id: str,
        *,
        worker_id: str,
        lease_token: str,
    ):
        now = _now_iso()
        async with self.db.execute(
            """SELECT * FROM relation_discovery_work
               WHERE id = ? AND status = 'running'
                 AND lease_owner = ? AND lease_token = ?
                 AND lease_until > ?""",
            (work_id, worker_id, lease_token, now),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise ValueError("relation discovery lease was lost")
        return row

    async def _assert_relation_discovery_outcome_current_unlocked(
        self,
        work_row,
        outcome: RelationOutcomeBundle,
    ) -> None:
        async with self.db.execute(
            """SELECT status, content_hash, visibility, owner_user_id,
                      repo_identifier
                 FROM memories WHERE id = ?""",
            (work_row["memory_id"],),
        ) as cursor:
            challenger = await cursor.fetchone()
        if (
            challenger is None
            or challenger["status"] != "active"
            or challenger["content_hash"] != work_row["expected_content_hash"]
        ):
            raise ValueError("relation discovery challenger is stale")
        unit = outcome.evidence_unit
        async with self.db.execute(
            "SELECT current_revision_id FROM source_units WHERE id = ? AND source_id = ?",
            (work_row["source_unit_id"], work_row["source_id"]),
        ) as cursor:
            source_unit = await cursor.fetchone()
        if source_unit is None or source_unit["current_revision_id"] != work_row["source_unit_revision_id"]:
            raise ValueError("relation discovery Source Unit revision is stale")
        if (
            unit.id != outcome.relation_run.evidence_unit_id
            or unit.source_id != work_row["source_id"]
            or unit.source_lineage_id != work_row["source_unit_id"]
        ):
            raise ValueError("relation discovery evidence lineage is stale")
        async with self.db.execute(
            "SELECT * FROM evidence_units WHERE id = ?",
            (unit.id,),
        ) as cursor:
            persisted_unit_row = await cursor.fetchone()
        if persisted_unit_row is None or self._row_to_evidence_unit(persisted_unit_row) != unit:
            raise ValueError("relation discovery evidence snapshot is stale")
        async with self.db.execute(
            f"""SELECT 1
                 FROM memory_support_assertions msa
                 JOIN evidence_references er
                   ON er.id = msa.evidence_reference_id
                 JOIN evidence_units eu ON eu.id = er.evidence_unit_id
                 JOIN source_observations so ON so.id = er.observation_id
                WHERE {CURRENT_RELATION_EVIDENCE_PREDICATE_SQL}
                  AND eu.id = ?
                LIMIT 1""",
            (
                work_row["memory_id"],
                work_row["source_id"],
                work_row["source_id"],
                work_row["source_unit_id"],
                unit.id,
            ),
        ) as cursor:
            if await cursor.fetchone() is None:
                raise ValueError("relation discovery evidence is no longer current")
        candidate_ids = tuple(dict.fromkeys(candidate.memory_id for candidate in outcome.candidates))
        if not candidate_ids:
            return
        placeholders = ", ".join("?" for _ in candidate_ids)
        rows = await self.db.execute_fetchall(
            f"""SELECT id, status, visibility, owner_user_id, repo_identifier
                  FROM memories WHERE id IN ({placeholders})""",
            candidate_ids,
        )
        by_id = {row["id"]: row for row in rows}
        for memory_id in candidate_ids:
            candidate = by_id.get(memory_id)
            if (
                candidate is None
                or candidate["status"] != "active"
                or candidate["visibility"] != challenger["visibility"]
                or candidate["owner_user_id"] != challenger["owner_user_id"]
                or candidate["repo_identifier"] != challenger["repo_identifier"]
            ):
                raise ValueError("relation discovery candidate is stale")
        selected_provenance = {
            (
                item.memory_id,
                item.source_id,
                item.doc_id,
                item.source_lineage_id,
                item.visibility,
                item.owner_user_id,
                item.repo_identifier,
                item.doc_revision_id,
                item.source_anchor,
                json.dumps(dict(item.source_metadata), sort_keys=True),
            )
            for item in outcome.candidate_provenance
        }
        if {item.memory_id for item in outcome.candidate_provenance} != set(candidate_ids):
            raise ValueError("relation discovery candidate provenance is incomplete")
        provenance_rows = await self.db.execute_fetchall(
            f"""SELECT m.id AS memory_id, m.visibility, m.owner_user_id,
                       m.repo_identifier, ms.source_id, ms.doc_id,
                       ms.doc_id AS source_lineage_id, NULL AS doc_revision_id,
                       NULL AS source_anchor, NULL AS source_metadata_json
                  FROM memories AS m
                  LEFT JOIN memory_sources AS ms ON ms.memory_id = m.id
                 WHERE m.id IN ({placeholders}) AND m.status = 'active'""",
            candidate_ids,
        )
        current_provenance = {
            (
                item.memory_id,
                item.source_id,
                item.doc_id,
                item.source_lineage_id,
                item.visibility,
                item.owner_user_id,
                item.repo_identifier,
                item.doc_revision_id,
                item.source_anchor,
                json.dumps(dict(item.source_metadata), sort_keys=True),
            )
            for item in (self._row_to_candidate_memory(row) for row in provenance_rows)
        }
        if not selected_provenance.issubset(current_provenance):
            raise ValueError("relation discovery candidate provenance is stale")
        actor_user_id = resolve_relation_discovery_actor_user_id(
            visibility=str(challenger["visibility"]),
            owner_user_id=challenger["owner_user_id"],
            requested_actor_user_id=work_row["actor_user_id"],
        )
        if actor_user_id:
            disabled_rows = await self.db.execute_fetchall(
                "SELECT source_id FROM source_subscriptions WHERE user_id = ? AND enabled = 0",
                (actor_user_id,),
            )
            disabled_source_ids = {str(row["source_id"]) for row in disabled_rows}
            if any(
                item.source_id in disabled_source_ids
                for item in outcome.candidate_provenance
                if item.source_id is not None
            ):
                raise ValueError("relation discovery candidate source access is stale")

    @staticmethod
    def _row_to_relation_discovery_work(row: Mapping[str, Any]) -> RelationDiscoveryWork:
        request = RelationDiscoveryRequest(
            id=str(row["id"]),
            memory_id=str(row["memory_id"]),
            expected_content_hash=str(row["expected_content_hash"]),
            source_id=str(row["source_id"]),
            source_unit_id=str(row["source_unit_id"]),
            source_unit_revision_id=row["source_unit_revision_id"],
            doc_id=str(row["doc_id"]),
            actor_user_id=row["actor_user_id"],
            entity_ids=tuple(int(value) for value in json.loads(row["entity_ids_json"])),
        )
        return RelationDiscoveryWork(
            request=request,
            lifecycle_plan_id=str(row["lifecycle_plan_id"]),
            status=RelationDiscoveryWorkStatus(row["status"]),
            attempts=int(row["attempts"]),
            lease_owner=row["lease_owner"],
            lease_token=row["lease_token"],
            lease_until=row["lease_until"],
            next_attempt_at=row["next_attempt_at"],
            error=row["error"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            completed_at=row["completed_at"],
        )

    def _row_to_lifecycle_cutover_finding(self, row) -> LifecycleCutoverFinding:
        return LifecycleCutoverFinding(
            id=row["id"],
            source_id=row["source_id"],
            memory_id=row["memory_id"],
            reason=CutoverFindingReason(row["reason"]),
            status=CutoverFindingStatus(row["status"]),
            available_provenance=json.loads(row["available_provenance_json"] or "{}"),
            mapping_attempt=json.loads(row["mapping_attempt_json"] or "{}"),
            observation_id=row["observation_id"],
            source_unit_id=row["source_unit_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            resolved_at=row["resolved_at"],
        )

    def _row_to_lifecycle_review(self, row) -> LifecycleReview:
        return LifecycleReview(
            id=row["id"],
            lifecycle_plan_id=row["lifecycle_plan_id"],
            incumbent_memory_id=row["incumbent_memory_id"],
            status=LifecycleReviewStatus(row["status"]),
            staged_evidence=json.loads(row["staged_evidence_json"] or "{}"),
            reason=row["reason"],
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
        )

    def _row_to_lifecycle_backfill_job(self, row) -> LifecycleBackfillJob:
        return LifecycleBackfillJob(
            id=row["id"],
            source_id=row["source_id"],
            status=LifecycleBackfillJobStatus(row["status"]),
            scanned_memories=int(row["scanned_memories"]),
            mapped_memories=int(row["mapped_memories"]),
            finding_count=int(row["finding_count"]),
            error=row["error"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    # ==================================================================
    # Memories
    # ==================================================================

    async def upsert_evidence_unit(
        self,
        unit: EvidenceUnit,
        *,
        source_activity: SourceActivityLease | None = None,
    ) -> None:
        """Persist one scoped evidence item before relation classification."""
        metadata_json = json.dumps(dict(unit.source_metadata), sort_keys=True)
        provenance = unit.evidence_provenance.value
        async with self._write_lock:
            try:
                await self._assert_source_activity_fence_unlocked(
                    unit.source_id,
                    source_activity,
                )
                now = _now_iso()
                await self.db.execute(
                    """INSERT INTO evidence_units (
                        id, source_id, doc_id, doc_revision_id, source_type, client,
                        repo_identifier, source_anchor, source_lineage_id,
                        source_metadata_json, project_key, visibility, owner_user_id,
                        observed_at, extractor_run_id, access_context_hash, content,
                        excerpt, evidence_provenance, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        observed_at=excluded.observed_at,
                        extractor_run_id=excluded.extractor_run_id,
                        access_context_hash=excluded.access_context_hash,
                        updated_at=excluded.updated_at""",
                    (
                        unit.id,
                        unit.source_id,
                        unit.doc_id,
                        unit.doc_revision_id,
                        unit.source_type,
                        unit.client,
                        unit.repo_identifier,
                        unit.source_anchor,
                        unit.source_lineage_id,
                        metadata_json,
                        _normalize_project_key(unit.project_key),
                        unit.visibility,
                        unit.owner_user_id,
                        unit.observed_at,
                        unit.extractor_run_id,
                        unit.access_context_hash,
                        unit.content,
                        unit.excerpt,
                        provenance,
                        now,
                        now,
                    ),
                )
                await self._assert_source_activity_fence_unlocked(
                    unit.source_id,
                    source_activity,
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def get_evidence_unit(self, evidence_unit_id: str) -> EvidenceUnit | None:
        async with self.db.execute(
            "SELECT * FROM evidence_units WHERE id = ?",
            (evidence_unit_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_evidence_unit(row)

    @staticmethod
    def _row_to_evidence_unit(row) -> EvidenceUnit:
        return EvidenceUnit(
            id=row["id"],
            source_id=row["source_id"],
            doc_id=row["doc_id"],
            doc_revision_id=row["doc_revision_id"],
            source_type=row["source_type"],
            client=row["client"],
            repo_identifier=row["repo_identifier"],
            source_anchor=row["source_anchor"],
            source_lineage_id=row["source_lineage_id"],
            source_metadata=json.loads(row["source_metadata_json"] or "{}"),
            project_key=row["project_key"],
            visibility=row["visibility"],
            owner_user_id=row["owner_user_id"],
            observed_at=row["observed_at"],
            extractor_run_id=row["extractor_run_id"],
            access_context_hash=row["access_context_hash"],
            content=row["content"],
            excerpt=row["excerpt"],
            evidence_provenance=EvidenceContentProvenance(row["evidence_provenance"]),
        )

    async def get_current_relation_evidence_unit(
        self,
        memory_id: str,
        *,
        source_id: str,
        source_unit_id: str,
    ) -> EvidenceUnit | None:
        rows = await self.db.execute_fetchall(
            f"""SELECT DISTINCT eu.id
                 FROM memory_support_assertions msa
                 JOIN evidence_references er
                   ON er.id = msa.evidence_reference_id
                 JOIN evidence_units eu ON eu.id = er.evidence_unit_id
                 JOIN source_observations so ON so.id = er.observation_id
                WHERE {CURRENT_RELATION_EVIDENCE_PREDICATE_SQL}
                ORDER BY eu.id""",
            (memory_id, source_id, source_id, source_unit_id),
        )
        if not rows:
            return None
        if len(rows) != 1:
            raise ValueError("current relation evidence is ambiguous")
        return await self.get_evidence_unit(str(rows[0]["id"]))

    async def replace_evidence_relations(
        self,
        evidence_unit_id: str,
        relations: Sequence[EvidenceRelationRecord],
    ) -> None:
        """Replace the complete current relation set for one Evidence Unit."""
        async with self._write_lock:
            try:
                await self.db.execute(
                    "DELETE FROM evidence_relations WHERE evidence_unit_id = ?",
                    (evidence_unit_id,),
                )
                for relation in relations:
                    validate_persisted_evidence_relation(relation)
                    await self.db.execute(
                        """INSERT INTO evidence_relations (
                            evidence_unit_id, memory_id, relation_type, relation_direction, authority_case,
                            is_authoritative_support, source_lineage_id, confidence,
                            reason, proposed_memory_content, excerpt, classifier_version,
                            relation_run_id, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            evidence_unit_id,
                            relation.memory_id,
                            relation.relation_type.value,
                            relation.direction.value if relation.direction is not None else None,
                            relation.authority_case.value,
                            1 if relation.is_authoritative_support else 0,
                            relation.source_lineage_id,
                            relation.confidence,
                            relation.reason,
                            relation.proposed_memory_content,
                            relation.excerpt,
                            relation.classifier_version,
                            relation.relation_run_id,
                            relation.created_at or _now_iso(),
                        ),
                    )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def get_evidence_relations(self, evidence_unit_id: str) -> list[EvidenceRelationRecord]:
        rows: list[EvidenceRelationRecord] = []
        async with self.db.execute(
            "SELECT * FROM evidence_relations WHERE evidence_unit_id = ? ORDER BY memory_id",
            (evidence_unit_id,),
        ) as cursor:
            async for row in cursor:
                rows.append(
                    EvidenceRelationRecord(
                        evidence_unit_id=row["evidence_unit_id"],
                        memory_id=row["memory_id"],
                        relation_type=RelationType(row["relation_type"]),
                        direction=(
                            RelationDirection(row["relation_direction"])
                            if row["relation_direction"] is not None
                            else None
                        ),
                        authority_case=AuthorityCase(row["authority_case"]),
                        is_authoritative_support=bool(row["is_authoritative_support"]),
                        source_lineage_id=row["source_lineage_id"],
                        confidence=row["confidence"],
                        reason=row["reason"],
                        proposed_memory_content=row["proposed_memory_content"],
                        excerpt=row["excerpt"],
                        classifier_version=row["classifier_version"],
                        relation_run_id=row["relation_run_id"],
                        created_at=row["created_at"],
                    )
                )
        return rows

    async def get_evidence_relations_by_memory(self, memory_id: str) -> list[EvidenceRelationRecord]:
        """Return current evidence-relation edges attached to one Memory."""
        rows: list[EvidenceRelationRecord] = []
        async with self.db.execute(
            "SELECT * FROM evidence_relations WHERE memory_id = ? ORDER BY evidence_unit_id",
            (memory_id,),
        ) as cursor:
            async for row in cursor:
                rows.append(
                    EvidenceRelationRecord(
                        evidence_unit_id=row["evidence_unit_id"],
                        memory_id=row["memory_id"],
                        relation_type=RelationType(row["relation_type"]),
                        direction=(
                            RelationDirection(row["relation_direction"])
                            if row["relation_direction"] is not None
                            else None
                        ),
                        authority_case=AuthorityCase(row["authority_case"]),
                        is_authoritative_support=bool(row["is_authoritative_support"]),
                        source_lineage_id=row["source_lineage_id"],
                        confidence=row["confidence"],
                        reason=row["reason"],
                        proposed_memory_content=row["proposed_memory_content"],
                        excerpt=row["excerpt"],
                        classifier_version=row["classifier_version"],
                        relation_run_id=row["relation_run_id"],
                        created_at=row["created_at"],
                    )
                )
        return rows

    async def has_materialized_evidence_unit(self, evidence_unit_id: str) -> bool:
        """Return true once an Evidence Unit has produced any durable Memory.

        This is intentionally lifecycle-status agnostic. A superseded Memory is
        still the historical materialization of the Evidence Unit, so retrying
        the same Evidence Unit must not create a second Memory merely because
        the first materialization is no longer active.
        """
        async with self.db.execute(
            """SELECT 1
               FROM relation_runs
               WHERE evidence_unit_id = ?
                 AND lifecycle_action IN (
                     'attach_support', 'create_memory', 'create_revision',
                     'supersede_memory', 'retire_memory'
                 )
               LIMIT 1""",
            (evidence_unit_id,),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def _delete_evidence_graph_for_memory_unlocked(self, memory_id: str) -> None:
        """Remove evidence graph content that materialized one Memory."""
        async with self.db.execute(
            """SELECT DISTINCT evidence_unit_id
               FROM relation_runs
               WHERE result_memory_id = ?
                 AND lifecycle_action IN (
                     'attach_support', 'create_memory', 'create_revision',
                     'supersede_memory', 'retire_memory'
                 )""",
            (memory_id,),
        ) as cursor:
            unit_ids = [row[0] async for row in cursor]
        await self._delete_evidence_graph_for_unit_ids_unlocked(unit_ids)
        await self.db.execute("DELETE FROM evidence_relations WHERE memory_id = ?", (memory_id,))
        await self.db.execute("DELETE FROM relation_run_relations WHERE memory_id = ?", (memory_id,))
        await self.db.execute("DELETE FROM relation_candidates WHERE memory_id = ?", (memory_id,))

    async def _delete_evidence_graph_for_memory_doc_unlocked(self, memory_id: str, doc_id: str) -> None:
        """Remove evidence graph content for a single memory-source link."""
        async with self.db.execute(
            """SELECT DISTINCT eu.id
               FROM evidence_units eu
               JOIN relation_runs rr ON rr.evidence_unit_id = eu.id
               WHERE eu.doc_id = ?
                 AND rr.result_memory_id = ?
                 AND rr.lifecycle_action IN (
                     'attach_support', 'create_memory', 'create_revision',
                     'supersede_memory', 'retire_memory'
                 )""",
            (doc_id, memory_id),
        ) as cursor:
            unit_ids = [row[0] async for row in cursor]
        await self._delete_evidence_graph_for_unit_ids_unlocked(unit_ids)
        await self.db.execute(
            """DELETE FROM evidence_relations
               WHERE memory_id = ?
                 AND evidence_unit_id IN (
                     SELECT id FROM evidence_units WHERE doc_id = ?
                 )""",
            (memory_id, doc_id),
        )
        await self.db.execute(
            """DELETE FROM relation_run_relations
               WHERE memory_id = ?
                 AND evidence_unit_id IN (
                     SELECT id FROM evidence_units WHERE doc_id = ?
                 )""",
            (memory_id, doc_id),
        )
        await self.db.execute(
            """DELETE FROM relation_candidates
               WHERE memory_id = ?
                 AND evidence_unit_id IN (
                     SELECT id FROM evidence_units WHERE doc_id = ?
                 )""",
            (memory_id, doc_id),
        )

    async def _delete_evidence_graph_for_doc_ids_unlocked(self, doc_ids: Sequence[str]) -> None:
        """Remove evidence graph content derived from deleted documents."""
        unique_doc_ids = tuple(dict.fromkeys(doc_ids))
        if not unique_doc_ids:
            return
        placeholders = ", ".join("?" for _ in unique_doc_ids)
        async with self.db.execute(
            f"SELECT id FROM evidence_units WHERE doc_id IN ({placeholders})",
            unique_doc_ids,
        ) as cursor:
            unit_ids = [row[0] async for row in cursor]
        await self._delete_evidence_graph_for_unit_ids_unlocked(unit_ids)

    async def _delete_evidence_graph_for_source_id_unlocked(self, source_id: str) -> None:
        """Remove evidence graph content owned by a deleted source."""
        async with self.db.execute(
            "SELECT id FROM evidence_units WHERE source_id = ?",
            (source_id,),
        ) as cursor:
            unit_ids = [row[0] async for row in cursor]
        await self._delete_evidence_graph_for_unit_ids_unlocked(unit_ids)

    async def _delete_evidence_graph_for_unit_ids_unlocked(self, unit_ids: Sequence[str]) -> None:
        """Delete units plus their relation-run and candidate audit graph."""
        unique_unit_ids = tuple(dict.fromkeys(unit_ids))
        if not unique_unit_ids:
            return
        placeholders = ", ".join("?" for _ in unique_unit_ids)
        await self.db.execute(
            f"DELETE FROM relation_candidates WHERE evidence_unit_id IN ({placeholders})",
            unique_unit_ids,
        )
        await self.db.execute(
            f"DELETE FROM evidence_relations WHERE evidence_unit_id IN ({placeholders})",
            unique_unit_ids,
        )
        await self.db.execute(
            f"DELETE FROM relation_run_relations WHERE evidence_unit_id IN ({placeholders})",
            unique_unit_ids,
        )
        await self.db.execute(
            f"DELETE FROM relation_runs WHERE evidence_unit_id IN ({placeholders})",
            unique_unit_ids,
        )
        await self.db.execute(
            f"DELETE FROM evidence_units WHERE id IN ({placeholders})",
            unique_unit_ids,
        )

    async def restore_evidence_relation_snapshot(self, relation: EvidenceRelationRecord) -> None:
        """Restore one current evidence-relation edge during write rollback."""
        validate_persisted_evidence_relation(relation)
        async with self._write_lock:
            try:
                await self.db.execute(
                    """INSERT OR IGNORE INTO evidence_relations (
                        evidence_unit_id, memory_id, relation_type, relation_direction, authority_case,
                        is_authoritative_support, source_lineage_id, confidence,
                        reason, proposed_memory_content, excerpt, classifier_version,
                        relation_run_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        relation.evidence_unit_id,
                        relation.memory_id,
                        relation.relation_type.value,
                        relation.direction.value if relation.direction is not None else None,
                        relation.authority_case.value,
                        1 if relation.is_authoritative_support else 0,
                        relation.source_lineage_id,
                        relation.confidence,
                        relation.reason,
                        relation.proposed_memory_content,
                        relation.excerpt,
                        relation.classifier_version,
                        relation.relation_run_id,
                        relation.created_at or _now_iso(),
                    ),
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def record_relation_run(self, run: RelationRunRecord) -> None:
        run = _with_empty_relation_snapshot_audit(run)
        lifecycle_action = run.lifecycle_action.value if run.lifecycle_action is not None else None
        review_case = run.review_case.value if run.review_case is not None else None
        async with self._write_lock:
            async with self.db.execute(
                "SELECT * FROM relation_runs WHERE id = ?",
                (run.id,),
            ) as cursor:
                existing_run = await cursor.fetchone()
            if existing_run is not None:
                _assert_relation_run_retry_matches(existing_run, run)
                return
            await self.db.execute(
                """INSERT INTO relation_runs (
                    id, evidence_unit_id, access_context_hash, candidate_count,
                    mandatory_candidate_count, checked_candidate_count,
                    incomplete_mandatory_buckets_json, classifier_version,
                    lifecycle_action, review_case, status, result_memory_id,
                    audit_json, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    run.id,
                    run.evidence_unit_id,
                    run.access_context_hash,
                    run.candidate_count,
                    run.mandatory_candidate_count,
                    run.checked_candidate_count,
                    json.dumps(list(run.incomplete_mandatory_buckets), sort_keys=True),
                    run.classifier_version,
                    lifecycle_action,
                    review_case,
                    run.status,
                    _relation_result_memory_id(run),
                    json.dumps(dict(run.audit), sort_keys=True),
                    run.started_at or _now_iso(),
                    run.completed_at,
                ),
            )
            await self.db.commit()

    async def get_relation_run(self, relation_run_id: str) -> RelationRunRecord | None:
        async with self.db.execute(
            "SELECT * FROM relation_runs WHERE id = ?",
            (relation_run_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        lifecycle_action = LifecycleAction(row["lifecycle_action"]) if row["lifecycle_action"] is not None else None
        review_case = ReviewCase(row["review_case"]) if row["review_case"] is not None else None
        return RelationRunRecord(
            id=row["id"],
            evidence_unit_id=row["evidence_unit_id"],
            access_context_hash=row["access_context_hash"],
            candidate_count=row["candidate_count"],
            mandatory_candidate_count=row["mandatory_candidate_count"],
            checked_candidate_count=row["checked_candidate_count"],
            incomplete_mandatory_buckets=tuple(json.loads(row["incomplete_mandatory_buckets_json"] or "[]")),
            classifier_version=row["classifier_version"],
            lifecycle_action=lifecycle_action,
            review_case=review_case,
            status=row["status"],
            result_memory_id=row["result_memory_id"],
            audit=json.loads(row["audit_json"] or "{}"),
            started_at=row["started_at"],
            completed_at=row["completed_at"],
        )

    async def replace_relation_candidates(
        self,
        relation_run_id: str,
        candidates: Sequence[RelationCandidateRecord],
    ) -> None:
        """Replace the auditable candidate universe for one relation run."""
        async with self._write_lock:
            try:
                await self.db.execute(
                    "DELETE FROM relation_candidates WHERE relation_run_id = ?",
                    (relation_run_id,),
                )
                for candidate in candidates:
                    await self.db.execute(
                        """INSERT INTO relation_candidates (
                            relation_run_id, evidence_unit_id, memory_id, bucket,
                            bucket_rank, candidate_rank, score, is_mandatory,
                            bucket_complete, was_checked, reason
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            relation_run_id,
                            candidate.evidence_unit_id,
                            candidate.memory_id,
                            candidate.bucket.value,
                            candidate.bucket_rank,
                            candidate.candidate_rank,
                            candidate.score,
                            1 if candidate.is_mandatory else 0,
                            1 if candidate.bucket_complete else 0,
                            1 if candidate.was_checked else 0,
                            candidate.reason,
                        ),
                    )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def get_relation_candidates(
        self,
        relation_run_id: str,
    ) -> list[RelationCandidateRecord]:
        return await self._get_relation_candidates_unlocked(relation_run_id)

    async def _get_relation_candidates_unlocked(
        self,
        relation_run_id: str,
    ) -> list[RelationCandidateRecord]:
        rows: list[RelationCandidateRecord] = []
        async with self.db.execute(
            """SELECT * FROM relation_candidates
               WHERE relation_run_id = ?
               ORDER BY bucket_rank, candidate_rank, memory_id""",
            (relation_run_id,),
        ) as cursor:
            async for row in cursor:
                rows.append(
                    RelationCandidateRecord(
                        relation_run_id=row["relation_run_id"],
                        evidence_unit_id=row["evidence_unit_id"],
                        memory_id=row["memory_id"],
                        bucket=CandidateBucket(row["bucket"]),
                        bucket_rank=row["bucket_rank"],
                        candidate_rank=row["candidate_rank"],
                        score=row["score"],
                        is_mandatory=bool(row["is_mandatory"]),
                        bucket_complete=bool(row["bucket_complete"]),
                        was_checked=bool(row["was_checked"]),
                        reason=row["reason"],
                    )
                )
        return rows

    async def _get_relation_run_relations_unlocked(
        self,
        relation_run_id: str,
    ) -> list[EvidenceRelationRecord]:
        rows: list[EvidenceRelationRecord] = []
        async with self.db.execute(
            """SELECT * FROM relation_run_relations
               WHERE relation_run_id = ?
               ORDER BY memory_id, relation_type, relation_run_id""",
            (relation_run_id,),
        ) as cursor:
            async for row in cursor:
                rows.append(
                    EvidenceRelationRecord(
                        evidence_unit_id=row["evidence_unit_id"],
                        memory_id=row["memory_id"],
                        relation_type=RelationType(row["relation_type"]),
                        direction=(
                            RelationDirection(row["relation_direction"])
                            if row["relation_direction"] is not None
                            else None
                        ),
                        authority_case=AuthorityCase(row["authority_case"]),
                        is_authoritative_support=bool(row["is_authoritative_support"]),
                        source_lineage_id=row["source_lineage_id"],
                        confidence=row["confidence"],
                        reason=row["reason"],
                        proposed_memory_content=row["proposed_memory_content"],
                        excerpt=row["excerpt"],
                        classifier_version=row["classifier_version"],
                        relation_run_id=row["relation_run_id"],
                        created_at=row["created_at"],
                    )
                )
        return rows

    async def record_relation_outcome_bundle(self, bundle: RelationOutcomeBundle) -> None:
        """Persist one complete relation outcome in a single transaction."""
        async with self._write_lock:
            try:
                await self._record_relation_outcome_bundle_unlocked(bundle)
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def _record_relation_outcome_bundle_unlocked(
        self,
        bundle: RelationOutcomeBundle,
        *,
        update_evidence_unit: bool = True,
    ) -> None:
        bundle = _with_relation_snapshot_audit(bundle)
        unit = bundle.evidence_unit
        run = bundle.relation_run
        async with self.db.execute(
            "SELECT * FROM relation_runs WHERE id = ?",
            (run.id,),
        ) as cursor:
            existing_run = await cursor.fetchone()
        if existing_run is not None:
            _assert_relation_run_retry_matches(existing_run, run)
            existing_audit = json.loads(existing_run["audit_json"] or "{}")
            await self._assert_relation_bundle_retry_matches_unlocked(bundle, existing_audit)
            return

        lifecycle_action = run.lifecycle_action.value if run.lifecycle_action is not None else None
        review_case = run.review_case.value if run.review_case is not None else None
        provenance = (
            unit.evidence_provenance.value
            if isinstance(unit.evidence_provenance, EvidenceContentProvenance)
            else str(unit.evidence_provenance)
        )
        now = _now_iso()
        if update_evidence_unit:
            await self.db.execute(
                """INSERT INTO evidence_units (
                        id, source_id, doc_id, doc_revision_id, source_type, client,
                        repo_identifier, source_anchor, source_lineage_id,
                        source_metadata_json, project_key, visibility, owner_user_id,
                        observed_at, extractor_run_id, access_context_hash,
                        content, excerpt, evidence_provenance, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        observed_at=excluded.observed_at,
                        extractor_run_id=excluded.extractor_run_id,
                        access_context_hash=excluded.access_context_hash,
                        updated_at=excluded.updated_at""",
                (
                    unit.id,
                    unit.source_id,
                    unit.doc_id,
                    unit.doc_revision_id,
                    unit.source_type,
                    unit.client,
                    unit.repo_identifier,
                    unit.source_anchor,
                    unit.source_lineage_id,
                    json.dumps(dict(unit.source_metadata), sort_keys=True),
                    _normalize_project_key(unit.project_key),
                    unit.visibility,
                    unit.owner_user_id,
                    unit.observed_at,
                    unit.extractor_run_id,
                    unit.access_context_hash,
                    unit.content,
                    unit.excerpt,
                    provenance,
                    now,
                    now,
                ),
            )
        await self.db.execute(
            """INSERT INTO relation_runs (
                        id, evidence_unit_id, access_context_hash, candidate_count,
                        mandatory_candidate_count, checked_candidate_count,
                        incomplete_mandatory_buckets_json, classifier_version,
                        lifecycle_action, review_case, status, result_memory_id,
                        audit_json, started_at, completed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run.id,
                run.evidence_unit_id,
                run.access_context_hash,
                run.candidate_count,
                run.mandatory_candidate_count,
                run.checked_candidate_count,
                json.dumps(list(run.incomplete_mandatory_buckets), sort_keys=True),
                run.classifier_version,
                lifecycle_action,
                review_case,
                run.status,
                _relation_result_memory_id(run),
                json.dumps(dict(run.audit), sort_keys=True),
                run.started_at or now,
                run.completed_at,
            ),
        )
        await self.db.execute(
            "DELETE FROM relation_candidates WHERE relation_run_id = ?",
            (run.id,),
        )
        for candidate in bundle.candidates:
            await self.db.execute(
                """INSERT INTO relation_candidates (
                            relation_run_id, evidence_unit_id, memory_id, bucket,
                            bucket_rank, candidate_rank, score, is_mandatory,
                            bucket_complete, was_checked, reason
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    candidate.relation_run_id,
                    candidate.evidence_unit_id,
                    candidate.memory_id,
                    candidate.bucket.value,
                    candidate.bucket_rank,
                    candidate.candidate_rank,
                    candidate.score,
                    1 if candidate.is_mandatory else 0,
                    1 if candidate.bucket_complete else 0,
                    1 if candidate.was_checked else 0,
                    candidate.reason,
                ),
            )
        await self.db.execute(
            "DELETE FROM evidence_relations WHERE evidence_unit_id = ?",
            (unit.id,),
        )
        for relation in bundle.relations:
            validate_persisted_evidence_relation(relation)
            await self.db.execute(
                """INSERT INTO relation_run_relations (
                            relation_run_id, evidence_unit_id, memory_id, relation_type, relation_direction,
                            authority_case, is_authoritative_support, source_lineage_id,
                            confidence, reason, proposed_memory_content, excerpt,
                            classifier_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    relation.relation_run_id,
                    relation.evidence_unit_id,
                    relation.memory_id,
                    relation.relation_type.value,
                    relation.direction.value if relation.direction is not None else None,
                    relation.authority_case.value,
                    1 if relation.is_authoritative_support else 0,
                    relation.source_lineage_id,
                    relation.confidence,
                    relation.reason,
                    relation.proposed_memory_content,
                    relation.excerpt,
                    relation.classifier_version,
                    relation.created_at or now,
                ),
            )
            await self.db.execute(
                """INSERT INTO evidence_relations (
                            evidence_unit_id, memory_id, relation_type, relation_direction, authority_case,
                            is_authoritative_support, source_lineage_id, confidence,
                            reason, proposed_memory_content, excerpt, classifier_version,
                            relation_run_id, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    relation.evidence_unit_id,
                    relation.memory_id,
                    relation.relation_type.value,
                    relation.direction.value if relation.direction is not None else None,
                    relation.authority_case.value,
                    1 if relation.is_authoritative_support else 0,
                    relation.source_lineage_id,
                    relation.confidence,
                    relation.reason,
                    relation.proposed_memory_content,
                    relation.excerpt,
                    relation.classifier_version,
                    relation.relation_run_id,
                    relation.created_at or now,
                ),
            )

    async def _assert_relation_bundle_retry_matches_unlocked(
        self,
        bundle: RelationOutcomeBundle,
        committed_audit: Mapping[str, object],
    ) -> None:
        stored_candidates = await self._get_relation_candidates_unlocked(bundle.relation_run.id)
        expected_candidates = list(bundle.candidates)
        committed_candidate_hash = committed_audit.get("candidate_snapshot_hash")
        if committed_candidate_hash is None:
            raise RuntimeError(
                "relation_run_id collision for "
                f"{bundle.relation_run.id}: committed audit is missing candidate_snapshot_hash"
            )
        candidate_hashes = relation_bundle_snapshot_audit(candidates=stored_candidates, relations=[])
        if committed_candidate_hash != candidate_hashes["candidate_snapshot_hash"]:
            raise RuntimeError(
                "relation_run_id collision for "
                f"{bundle.relation_run.id}: committed run snapshot was modified (relation_candidates)"
            )

        if [relation_candidate_retry_identity(candidate) for candidate in stored_candidates] != [
            relation_candidate_retry_identity(candidate) for candidate in expected_candidates
        ]:
            raise RuntimeError(
                "relation_run_id collision for "
                f"{bundle.relation_run.id}: existing run does not match retry payload (relation_candidates)"
            )

        stored_relations = await self._get_relation_run_relations_unlocked(bundle.relation_run.id)
        committed_relation_hash = committed_audit.get("relation_snapshot_hash")
        if committed_relation_hash is None:
            raise RuntimeError(
                "relation_run_id collision for "
                f"{bundle.relation_run.id}: committed audit is missing relation_snapshot_hash"
            )
        relation_hashes = relation_bundle_snapshot_audit(candidates=[], relations=stored_relations)
        if committed_relation_hash != relation_hashes["relation_snapshot_hash"]:
            raise RuntimeError(
                "relation_run_id collision for "
                f"{bundle.relation_run.id}: committed run snapshot was modified (evidence_relations)"
            )

        expected_relations = sorted(bundle.relations, key=evidence_relation_retry_identity)
        stored_relations = sorted(stored_relations, key=evidence_relation_retry_identity)
        if [evidence_relation_retry_identity(relation) for relation in stored_relations] != [
            evidence_relation_retry_identity(relation) for relation in expected_relations
        ]:
            raise RuntimeError(
                "relation_run_id collision for "
                f"{bundle.relation_run.id}: existing run does not match retry payload (evidence_relations)"
            )

    async def insert_memory(self, mem: Memory) -> str:
        """Insert a memory and its FTS5 row. Returns the memory id."""
        async with self._write_lock:
            try:
                await self._insert_memory_unlocked(mem)
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        return mem.id

    async def _insert_memory_unlocked(self, mem: Memory) -> None:
        _validate_visibility(mem.visibility, mem.owner_user_id)
        project_key = _normalize_project_key(mem.project_key)
        now = _now_iso()
        status = normalize_memory_status(mem.status)
        await self.db.execute(
            """INSERT INTO memories (
                id, memory_type, content, content_hash, tags, visibility, owner_user_id,
                project_key, repo_identifier,
                confidence, corroboration_count,
                contradiction_count, valid_from, valid_until,
                superseded_by, status, retirement_reason, retired_at,
                superseded_at, replacement_reason, replacement_kind, extraction_context,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                mem.id,
                mem.memory_type,
                mem.content,
                mem.content_hash,
                json.dumps(mem.tags),
                mem.visibility,
                mem.owner_user_id,
                project_key,
                mem.repo_identifier,
                mem.confidence,
                mem.corroboration_count,
                mem.contradiction_count,
                mem.valid_from.isoformat() if mem.valid_from else None,
                mem.valid_until.isoformat() if mem.valid_until else None,
                mem.superseded_by,
                status,
                mem.retirement_reason,
                mem.retired_at.isoformat() if mem.retired_at else None,
                mem.superseded_at.isoformat() if mem.superseded_at else None,
                mem.replacement_reason,
                mem.replacement_kind,
                mem.extraction_context,
                mem.created_at.isoformat() if mem.created_at else now,
                mem.updated_at.isoformat() if mem.updated_at else now,
            ),
        )
        entities_text = " ".join(mem.entity_refs)
        tags_text = " ".join(mem.tags)
        await self.db.execute(
            "INSERT INTO memories_fts (memory_id, content, entities_text, tags_text) VALUES (?, ?, ?, ?)",
            (mem.id, mem.content, entities_text, tags_text),
        )

    async def _upsert_memory_preserving_created_at_unlocked(self, mem: Memory) -> None:
        """Create or refresh a lifecycle-produced Memory without rewriting CREATED_AT."""
        _validate_visibility(mem.visibility, mem.owner_user_id)
        project_key = _normalize_project_key(mem.project_key)
        now = _now_iso()
        status = normalize_memory_status(mem.status)
        await self.db.execute(
            """INSERT INTO memories (
                id, memory_type, content, content_hash, tags, visibility, owner_user_id,
                project_key, repo_identifier,
                confidence, corroboration_count,
                contradiction_count, valid_from, valid_until,
                superseded_by, status, retirement_reason, retired_at,
                superseded_at, replacement_reason, replacement_kind, extraction_context,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                memory_type=excluded.memory_type,
                content=excluded.content,
                content_hash=excluded.content_hash,
                tags=excluded.tags,
                visibility=excluded.visibility,
                owner_user_id=excluded.owner_user_id,
                project_key=excluded.project_key,
                repo_identifier=excluded.repo_identifier,
                confidence=excluded.confidence,
                corroboration_count=excluded.corroboration_count,
                contradiction_count=excluded.contradiction_count,
                valid_from=excluded.valid_from,
                valid_until=excluded.valid_until,
                superseded_by=excluded.superseded_by,
                status=excluded.status,
                retirement_reason=excluded.retirement_reason,
                retired_at=excluded.retired_at,
                superseded_at=excluded.superseded_at,
                replacement_reason=excluded.replacement_reason,
                replacement_kind=excluded.replacement_kind,
                extraction_context=excluded.extraction_context,
                updated_at=excluded.updated_at""",
            (
                mem.id,
                mem.memory_type,
                mem.content,
                mem.content_hash,
                json.dumps(mem.tags),
                mem.visibility,
                mem.owner_user_id,
                project_key,
                mem.repo_identifier,
                mem.confidence,
                mem.corroboration_count,
                mem.contradiction_count,
                mem.valid_from.isoformat() if mem.valid_from else None,
                mem.valid_until.isoformat() if mem.valid_until else None,
                mem.superseded_by,
                status,
                mem.retirement_reason,
                mem.retired_at.isoformat() if mem.retired_at else None,
                mem.superseded_at.isoformat() if mem.superseded_at else None,
                mem.replacement_reason,
                mem.replacement_kind,
                mem.extraction_context,
                mem.created_at.isoformat() if mem.created_at else now,
                mem.updated_at.isoformat() if mem.updated_at else now,
            ),
        )

    async def insert_memory_with_source_and_relation(
        self,
        mem: Memory,
        *,
        doc_id: str,
        source_type: str,
        excerpt: str | None = None,
        entity_ids: Sequence[int] | None = None,
        relation_outcome: RelationOutcomeBundle | None = None,
        source_updated_at: datetime | None,
        review: MemoryReview | None = None,
        related_review_id: str | None = None,
        related_review_reason: str | None = None,
    ) -> str:
        """Insert a memory, source provenance, entities, relation audit, and optional review atomically."""
        if review is not None and related_review_id is not None:
            raise ValueError("review and related_review_id are mutually exclusive")
        async with self._write_lock:
            try:
                await self._assert_legacy_source_write_allowed_unlocked(doc_id)
                await self._upsert_memory_preserving_created_at_unlocked(mem)
                await self._add_memory_source_unlocked(
                    mem.id,
                    doc_id,
                    source_type,
                    excerpt,
                    source_updated_at=source_updated_at,
                )
                await self._link_memory_entities_unlocked(mem.id, entity_ids)
                await self._rebuild_memory_fts_unlocked(
                    mem.id,
                    search_visible_statuses=set(allowed_search_statuses()),
                )
                if relation_outcome is not None:
                    await self._record_relation_outcome_bundle_unlocked(relation_outcome)
                if related_review_id is not None:
                    now = _now_iso()
                    async with self.db.execute(
                        """SELECT review_id FROM memory_review_related_challengers
                           WHERE challenger_memory_id = ?""",
                        (mem.id,),
                    ) as cursor:
                        existing = await cursor.fetchone()
                    if existing:
                        existing_review_id = existing["review_id"]
                        if existing_review_id != related_review_id:
                            raise ValueError(f"Challenger {mem.id} is already attached to review {existing_review_id}")
                        await self.db.execute(
                            """UPDATE memory_review_related_challengers
                               SET reason = COALESCE(?, reason)
                               WHERE review_id = ? AND challenger_memory_id = ?""",
                            (related_review_reason, related_review_id, mem.id),
                        )
                    else:
                        await self.db.execute(
                            """INSERT INTO memory_review_related_challengers (
                                review_id, challenger_memory_id, reason, created_at
                            ) VALUES (?, ?, ?, ?)""",
                            (related_review_id, mem.id, related_review_reason, now),
                        )
                if review is not None:
                    now = _now_iso()
                    created_at = review.created_at.isoformat() if review.created_at else now
                    await self.db.execute(
                        """INSERT INTO memory_reviews (
                            id, kind, status, incumbent_memory_id, challenger_memory_id,
                            reason, review_note, reviewer,
                            expected_incumbent_updated_at, expected_challenger_updated_at,
                            replacement_kind, created_at, resolved_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            kind=excluded.kind,
                            status=excluded.status,
                            incumbent_memory_id=excluded.incumbent_memory_id,
                            challenger_memory_id=excluded.challenger_memory_id,
                            reason=excluded.reason,
                            review_note=excluded.review_note,
                            reviewer=excluded.reviewer,
                            expected_incumbent_updated_at=excluded.expected_incumbent_updated_at,
                            expected_challenger_updated_at=excluded.expected_challenger_updated_at,
                            replacement_kind=excluded.replacement_kind,
                            resolved_at=excluded.resolved_at
                        WHERE memory_reviews.status = 'pending'""",
                        (
                            review.id,
                            review.kind,
                            review.status,
                            review.incumbent_memory_id,
                            review.challenger_memory_id,
                            review.reason,
                            review.review_note,
                            review.reviewer,
                            review.expected_incumbent_updated_at,
                            review.expected_challenger_updated_at,
                            _validate_replacement_kind(review.replacement_kind),
                            created_at,
                            review.resolved_at.isoformat() if review.resolved_at else None,
                        ),
                    )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        return mem.id

    async def insert_memory_and_upsert_agent_claim(
        self,
        mem: Memory,
        *,
        doc_id: str,
        source_type: str,
        excerpt: str | None,
        relation_outcome: RelationOutcomeBundle | None,
        claim_id: str,
        concept_id: str,
        display_anchor: str,
        claim_text: str,
        memory_type: str,
        tags: list[str],
        confidence: float,
        observed_at: datetime,
        source_updated_at: datetime | None,
        citations: list[str] | None = None,
        concept_projection: dict[str, Any] | None = None,
        concept_markdown_body: str | None = None,
        entity_ids: list[int] | None = None,
    ) -> str:
        """Insert an agent-session memory and its claim projection atomically."""
        _validate_visibility(mem.visibility, mem.owner_user_id)
        observed = _utc_iso(observed_at)
        async with self._write_lock:
            try:
                await self._assert_legacy_source_write_allowed_unlocked(doc_id)
                await self._upsert_memory_preserving_created_at_unlocked(mem)
                await self.db.execute(
                    """INSERT INTO memory_sources (
                        memory_id, doc_id, source_id, source_type, excerpt, support_kind, source_updated_at
                    ) VALUES (?, ?, (SELECT source FROM documents WHERE doc_id = ?), ?, ?, 'extracted', ?)
                    ON CONFLICT(memory_id, doc_id) DO UPDATE SET
                        source_id = excluded.source_id,
                        source_type = excluded.source_type,
                        excerpt = excluded.excerpt,
                        support_kind = excluded.support_kind,
                        source_updated_at = excluded.source_updated_at""",
                    (
                        mem.id,
                        doc_id,
                        doc_id,
                        source_type,
                        excerpt,
                        _utc_iso(source_updated_at) if source_updated_at is not None else None,
                    ),
                )
                if entity_ids:
                    for entity_id in entity_ids:
                        await self.db.execute(
                            "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
                            (mem.id, entity_id),
                        )
                await self._rebuild_memory_fts_unlocked(
                    mem.id,
                    search_visible_statuses=set(allowed_search_statuses()),
                )
                if relation_outcome is not None:
                    await self._record_relation_outcome_bundle_unlocked(relation_outcome)
                if concept_projection is not None:
                    await self._upsert_agent_concept_unlocked(**concept_projection, observed=observed)
                await self._upsert_agent_claim_unlocked(
                    claim_id=claim_id,
                    concept_id=concept_id,
                    display_anchor=display_anchor,
                    claim_text=claim_text,
                    memory_type=memory_type,
                    tags=tags,
                    confidence=confidence,
                    memory_id=mem.id,
                    observed=observed,
                )
                for citation_url in citations or []:
                    await self._add_agent_claim_citation_unlocked(
                        claim_id=claim_id,
                        citation_url=citation_url,
                        observed=observed,
                    )
                if concept_markdown_body is not None:
                    await self._update_agent_concept_markdown_unlocked(
                        concept_id=concept_id,
                        markdown_body=concept_markdown_body,
                        observed=observed,
                    )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        return mem.id

    async def rebuild_memory_fts(
        self,
        memory_id: str,
        *,
        search_visible_statuses: set[str],
    ) -> bool:
        """Rebuild one memory's FTS row from SQLite's canonical memory state."""
        async with self._write_lock:
            rebuilt = await self._rebuild_memory_fts_unlocked(
                memory_id,
                search_visible_statuses=search_visible_statuses,
            )
            await self.db.commit()
            return rebuilt

    async def _rebuild_memory_fts_unlocked(
        self,
        memory_id: str,
        *,
        search_visible_statuses: set[str],
    ) -> bool:
        async with self.db.execute(
            "SELECT content, tags, status FROM memories WHERE id = ?",
            (memory_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return False

        await self.db.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory_id,))
        if normalize_memory_status(row["status"]) in search_visible_statuses:
            entity_names: list[str] = []
            async with self.db.execute(
                """SELECT e.canonical_name
                   FROM memory_entities me
                   JOIN entities e ON me.entity_id = e.id
                   WHERE me.memory_id = ?
                   ORDER BY e.id""",
                (memory_id,),
            ) as entity_cursor:
                async for entity_row in entity_cursor:
                    entity_names.append(entity_row[0])

            tags = json.loads(row["tags"] or "[]")
            await self.db.execute(
                "INSERT INTO memories_fts (memory_id, content, entities_text, tags_text) VALUES (?, ?, ?, ?)",
                (memory_id, row["content"], " ".join(entity_names), " ".join(tags)),
            )
        return True

    async def prune_memory_fts_orphans(self) -> int:
        """Remove FTS rows whose memory ID is not present in SQLite memories."""
        async with self._write_lock:
            cursor = await self.db.execute(
                """DELETE FROM memories_fts
                   WHERE NOT EXISTS (
                       SELECT 1 FROM memories WHERE memories.id = memories_fts.memory_id
                   )"""
            )
            await self.db.commit()
            return cursor.rowcount if cursor.rowcount is not None else 0

    async def get_memory(self, memory_id: str) -> Memory | None:
        async with self.db.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return self._row_to_memory(row)

    async def find_active_exact_claim_candidate(
        self,
        memory_content_hash: str,
        *,
        visibility: str,
        owner_user_id: str | None,
        repo_identifier: str | None,
        excluded_memory_ids: Sequence[str] = (),
    ) -> Memory | None:
        """Return the canonical active exact claim in the same access context."""

        exclusions = tuple(dict.fromkeys(excluded_memory_ids))
        exclusion_clause = ""
        params: list[Any] = [
            memory_content_hash,
            visibility,
            owner_user_id,
            repo_identifier,
        ]
        if exclusions:
            exclusion_clause = " AND m.id NOT IN (" + ", ".join("?" for _ in exclusions) + ")"
            params.extend(exclusions)
        async with self.db.execute(
            """SELECT m.* FROM memories AS m
                WHERE m.content_hash = ?
                  AND m.status = 'active'
                  AND m.visibility = ?
                  AND m.owner_user_id IS ?
                  AND m.repo_identifier IS ?
                  AND NOT EXISTS (
                      SELECT 1 FROM agent_claims AS ac
                      WHERE ac.memory_id = m.id
                  )"""
            + exclusion_clause
            + " ORDER BY m.created_at, m.id LIMIT 1",
            params,
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_memory(row) if row else None

    async def list_active_ordinary_claim_memories(
        self,
        memory_ids: Sequence[str],
    ) -> list[Memory]:
        """Return active non-Agent-Claim Memories in caller-provided order."""

        ordered_ids = tuple(dict.fromkeys(memory_ids))
        if not ordered_ids:
            return []
        placeholders = ", ".join("?" for _ in ordered_ids)
        rows: dict[str, Memory] = {}
        async with self.db.execute(
            f"""SELECT m.* FROM memories AS m
                WHERE m.id IN ({placeholders})
                  AND m.status = 'active'
                  AND NOT EXISTS (
                      SELECT 1 FROM agent_claims AS ac
                      WHERE ac.memory_id = m.id
                  )""",
            ordered_ids,
        ) as cursor:
            async for row in cursor:
                memory = self._row_to_memory(row)
                rows[memory.id] = memory
        return [rows[memory_id] for memory_id in ordered_ids if memory_id in rows]

    async def list_active_memories(
        self,
        memory_ids: Sequence[str],
    ) -> list[Memory]:
        """Return active Memories in caller-provided order."""

        ordered_ids = tuple(dict.fromkeys(memory_ids))
        if not ordered_ids:
            return []
        placeholders = ", ".join("?" for _ in ordered_ids)
        rows: dict[str, Memory] = {}
        async with self.db.execute(
            f"""SELECT m.* FROM memories AS m
                WHERE m.id IN ({placeholders})
                  AND m.status = 'active'""",
            ordered_ids,
        ) as cursor:
            async for row in cursor:
                memory = self._row_to_memory(row)
                rows[memory.id] = memory
        return [rows[memory_id] for memory_id in ordered_ids if memory_id in rows]

    async def list_active_candidate_memories(
        self,
        memory_ids: Sequence[str],
    ) -> list[CandidateMemory]:
        """Return lightweight provenance rows for active candidate IDs."""

        ordered_ids = tuple(dict.fromkeys(memory_ids))
        if not ordered_ids:
            return []
        placeholders = ", ".join("?" for _ in ordered_ids)
        candidates: list[CandidateMemory] = []
        async with self.db.execute(
            f"""SELECT m.id AS memory_id,
                       m.visibility,
                       m.owner_user_id,
                       m.repo_identifier,
                       ms.source_id,
                       ms.doc_id,
                       ms.doc_id AS source_lineage_id,
                       NULL AS doc_revision_id,
                       NULL AS source_anchor,
                       NULL AS source_metadata_json
                  FROM memories AS m
                  LEFT JOIN memory_sources AS ms ON ms.memory_id = m.id
                 WHERE m.id IN ({placeholders})
                   AND m.status = 'active'
                 ORDER BY m.id, ms.source_id, ms.doc_id""",
            ordered_ids,
        ) as cursor:
            async for row in cursor:
                candidates.append(self._row_to_candidate_memory(row))
        return candidates

    async def find_rebaseline_reactivation_candidate(
        self,
        memory_content_hash: str,
        *,
        visibility: str,
        owner_user_id: str | None,
        repo_identifier: str | None,
    ) -> Memory | None:
        """Return the canonical exact claim retired only for source rebaseline."""

        async with self.db.execute(
            """SELECT m.* FROM memories AS m
                WHERE m.content_hash = ?
                  AND m.status = 'retired'
                  AND m.retirement_reason = 'source_rebaseline'
                  AND m.visibility = ?
                  AND m.owner_user_id IS ?
                  AND m.repo_identifier IS ?
                  AND NOT EXISTS (
                      SELECT 1 FROM agent_claims AS ac
                      WHERE ac.memory_id = m.id
                  )
                ORDER BY m.created_at, m.id
                LIMIT 1""",
            (
                memory_content_hash,
                visibility,
                owner_user_id,
                repo_identifier,
            ),
        ) as cursor:
            row = await cursor.fetchone()
        return self._row_to_memory(row) if row else None

    async def get_memories_by_source_doc(
        self,
        doc_id: str,
        *,
        support_kind: str | None = "extracted",
    ) -> list[Memory]:
        results: list[Memory] = []
        params: list[str] = [doc_id]
        kind_clause = ""
        if support_kind is not None:
            kind_clause = " AND ms.support_kind = ?"
            params.append(support_kind)
        async with self.db.execute(
            """SELECT m.* FROM memories m
               JOIN memory_sources ms ON m.id = ms.memory_id
               WHERE ms.doc_id = ?"""
            + kind_clause,
            params,
        ) as cursor:
            async for row in cursor:
                results.append(self._row_to_memory(row))
        return results

    async def get_candidate_memories_by_source_doc(
        self,
        *,
        doc_id: str,
        support_kind: str | None = None,
    ) -> list[CandidateMemory]:
        """Return complete same-document candidate Memories for relation checks."""
        params: list[str] = [doc_id]
        kind_clause = ""
        if support_kind is not None:
            kind_clause = " AND ms.support_kind = ?"
            params.append(support_kind)
        candidates: list[CandidateMemory] = []
        async with self.db.execute(
            """SELECT DISTINCT
                      m.id, m.visibility, m.owner_user_id, m.repo_identifier,
                      m.status, ms.source_id, ms.doc_id
                 FROM memories m
                 JOIN memory_sources ms ON m.id = ms.memory_id
                WHERE ms.doc_id = ?
                  AND m.status = 'active'"""
            + kind_clause
            + """
                ORDER BY m.id""",
            params,
        ) as cursor:
            async for row in cursor:
                candidates.append(
                    CandidateMemory(
                        memory_id=row["id"],
                        source_id=row["source_id"],
                        doc_id=row["doc_id"],
                        source_lineage_id=row["doc_id"],
                        visibility=row["visibility"],
                        owner_user_id=row["owner_user_id"],
                        repo_identifier=row["repo_identifier"],
                    )
                )
        return candidates

    async def get_candidate_memories_by_source_anchor(
        self,
        *,
        source_id: str,
        source_anchor: str,
    ) -> list[CandidateMemory]:
        """Return complete exact-anchor candidates from current evidence relations."""
        candidates: list[CandidateMemory] = []
        async with self.db.execute(
            """SELECT DISTINCT
                      m.id AS memory_id, m.visibility, m.owner_user_id,
                      m.repo_identifier, eu.source_id, eu.doc_id,
                      eu.doc_revision_id, eu.source_anchor,
                      eu.source_lineage_id, eu.source_metadata_json
                 FROM evidence_units eu
                 JOIN evidence_relations er ON er.evidence_unit_id = eu.id
                 JOIN memories m ON m.id = er.memory_id
                WHERE eu.source_id = ?
                  AND eu.source_anchor = ?
                  AND m.status = 'active'
                ORDER BY m.id""",
            (source_id, source_anchor),
        ) as cursor:
            async for row in cursor:
                candidates.append(self._row_to_candidate_memory(row))
        return candidates

    async def get_candidate_memories_by_agent_claim(
        self,
        *,
        claim_anchor: str,
    ) -> list[CandidateMemory]:
        """Return complete private same-agent-claim candidates."""
        candidates: list[CandidateMemory] = []
        async with self.db.execute(
            """SELECT DISTINCT
                      m.id AS memory_id, m.visibility, m.owner_user_id,
                      m.repo_identifier, eu.source_id, eu.doc_id,
                      eu.doc_revision_id, eu.source_anchor,
                      eu.source_lineage_id, eu.source_metadata_json
                 FROM evidence_units eu
                 JOIN evidence_relations er ON er.evidence_unit_id = eu.id
                 JOIN memories m ON m.id = er.memory_id
                WHERE eu.source_type = 'agent_session'
                  AND eu.source_lineage_id = ?
                  AND m.status = 'active'
                ORDER BY m.id""",
            (claim_anchor,),
        ) as cursor:
            async for row in cursor:
                candidates.append(self._row_to_candidate_memory(row))
        return candidates

    async def get_candidate_memories_by_existing_relation_graph(
        self,
        *,
        evidence_unit_id: str,
    ) -> list[CandidateMemory]:
        """Return active candidates already related to this Evidence Unit."""
        candidates: list[CandidateMemory] = []
        async with self.db.execute(
            """SELECT DISTINCT
                      m.id AS memory_id, m.visibility, m.owner_user_id,
                      m.repo_identifier, eu.source_id, eu.doc_id,
                      eu.doc_revision_id, eu.source_anchor,
                      eu.source_lineage_id, eu.source_metadata_json
                 FROM evidence_units eu
                 JOIN evidence_relations er ON er.evidence_unit_id = eu.id
                 JOIN memories m ON m.id = er.memory_id
                WHERE eu.id = ?
                  AND m.status = 'active'
                ORDER BY m.id""",
            (evidence_unit_id,),
        ) as cursor:
            async for row in cursor:
                candidates.append(self._row_to_candidate_memory(row))
        return candidates

    async def get_memories_by_entity(self, entity_id: int) -> list[Memory]:
        results: list[Memory] = []
        async with self.db.execute(
            """SELECT m.* FROM memories m
               JOIN memory_entities me ON m.id = me.memory_id
               WHERE me.entity_id = ?""",
            (entity_id,),
        ) as cursor:
            async for row in cursor:
                results.append(self._row_to_memory(row))
        return results

    async def update_memory_content(
        self,
        memory_id: str,
        new_content: str,
        new_confidence: float | None,
        new_tags: list[str] | None,
    ) -> None:
        async with self._write_lock:
            from memforge.models import content_hash

            async with self.db.execute(
                "SELECT content_hash, confidence, tags FROM memories WHERE id = ?",
                (memory_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                return
            if row["content_hash"] != content_hash(new_content):
                await self._assert_no_active_source_support_unlocked(memory_id)

            confidence = new_confidence if new_confidence is not None else row["confidence"]
            tags = new_tags if new_tags is not None else json.loads(row["tags"] or "[]")
            now = _now_iso()
            await self.db.execute(
                """UPDATE memories SET
                    content = ?, content_hash = ?, confidence = ?,
                    tags = ?, updated_at = ?
                   WHERE id = ?""",
                (
                    new_content,
                    content_hash(new_content),
                    confidence,
                    json.dumps(tags),
                    now,
                    memory_id,
                ),
            )
            await self._rebuild_memory_fts_unlocked(
                memory_id,
                search_visible_statuses=set(allowed_search_statuses()),
            )
            await self.db.commit()

    async def update_memory_status(
        self,
        memory_id: str,
        status: str,
        *,
        reason: str | None = None,
    ) -> None:
        async with self._write_lock:
            await self._update_memory_status_unlocked(memory_id, status, reason=reason)
            await self.db.commit()

    async def update_memory_status_with_relation_outcome(
        self,
        memory_id: str,
        status: str,
        *,
        reason: str | None = None,
        relation_outcome: RelationOutcomeBundle,
    ) -> None:
        async with self._write_lock:
            try:
                await self._update_memory_status_unlocked(
                    memory_id,
                    status,
                    reason=reason,
                    allowed_current_statuses=("active", "pending_review"),
                )
                await self._record_relation_outcome_bundle_unlocked(relation_outcome)
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def _update_memory_status_unlocked(
        self,
        memory_id: str,
        status: str,
        *,
        reason: str | None = None,
        allowed_current_statuses: Sequence[str] | None = None,
    ) -> None:
        canonical = normalize_memory_status(status)
        if canonical != "active":
            await self._assert_no_active_source_support_unlocked(memory_id)
        now = _now_iso()
        guard_clause = ""
        guard_params: tuple[str, ...] = ()
        if allowed_current_statuses:
            placeholders = ", ".join("?" for _ in allowed_current_statuses)
            guard_clause = f" AND status IN ({placeholders})"
            guard_params = tuple(allowed_current_statuses)
        if canonical == "retired":
            cursor = await self.db.execute(
                f"""UPDATE memories SET
                    status = ?, retirement_reason = COALESCE(?, retirement_reason, 'admin_hidden'),
                    retired_at = COALESCE(retired_at, ?), updated_at = ?
                   WHERE id = ?{guard_clause}""",
                (canonical, reason, now, now, memory_id, *guard_params),
            )
        elif canonical == "active":
            cursor = await self.db.execute(
                f"""UPDATE memories SET
                    status = ?, retirement_reason = NULL, retired_at = NULL,
                    superseded_by = NULL, superseded_at = NULL,
                    replacement_reason = NULL, replacement_kind = NULL, updated_at = ?
                   WHERE id = ?{guard_clause}""",
                (canonical, now, memory_id, *guard_params),
            )
        else:
            if allowed_current_statuses:
                cursor = await self.db.execute(
                    f"UPDATE memories SET status = ?, updated_at = ? WHERE id = ?{guard_clause}",
                    (canonical, now, memory_id, *guard_params),
                )
            else:
                await self.db.execute(
                    "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                    (canonical, now, memory_id),
                )
                return
        if allowed_current_statuses and cursor.rowcount != 1:
            raise RuntimeError(f"memory {memory_id} cannot transition to {canonical} from its current lifecycle state")

    async def purge_memory(self, memory_id: str) -> bool:
        """Hard-delete a memory and its local indexes/provenance."""
        async with self._write_lock:
            async with self.db.execute(
                "SELECT id FROM memories WHERE id = ?",
                (memory_id,),
            ) as cursor:
                exists = await cursor.fetchone()
            if not exists:
                return False

            now = _now_iso()
            await self.db.execute(
                """UPDATE memories SET
                    superseded_by = NULL, superseded_at = NULL,
                    replacement_reason = NULL, replacement_kind = NULL,
                    status = 'retired',
                    retirement_reason = 'privacy_removed',
                    retired_at = COALESCE(retired_at, ?), updated_at = ?
                   WHERE superseded_by = ?""",
                (now, now, memory_id),
            )
            await self.db.execute(
                "DELETE FROM memory_contradictions WHERE memory_id_a = ? OR memory_id_b = ?",
                (memory_id, memory_id),
            )
            await self.db.execute(
                "DELETE FROM memory_sources WHERE memory_id = ?",
                (memory_id,),
            )
            await self._delete_evidence_graph_for_memory_unlocked(memory_id)
            await self.db.execute(
                "DELETE FROM memory_entities WHERE memory_id = ?",
                (memory_id,),
            )
            await self.db.execute(
                "DELETE FROM memory_review_related_challengers WHERE challenger_memory_id = ?",
                (memory_id,),
            )
            await self.db.execute(
                """DELETE FROM memory_review_related_challengers
                   WHERE review_id IN (
                       SELECT id FROM memory_reviews
                       WHERE incumbent_memory_id = ? OR challenger_memory_id = ?
                   )""",
                (memory_id, memory_id),
            )
            await self.db.execute(
                """DELETE FROM memory_reviews
                   WHERE incumbent_memory_id = ? OR challenger_memory_id = ?""",
                (memory_id, memory_id),
            )
            await self.db.execute(
                "DELETE FROM memories_fts WHERE memory_id = ?",
                (memory_id,),
            )
            await self.db.execute(
                """DELETE FROM agent_claim_citations
                   WHERE claim_id IN (
                       SELECT id FROM agent_claims WHERE memory_id = ?
                   )""",
                (memory_id,),
            )
            await self.db.execute(
                "DELETE FROM agent_claims WHERE memory_id = ?",
                (memory_id,),
            )
            await self.db.execute(
                "DELETE FROM memories WHERE id = ?",
                (memory_id,),
            )
            await self.db.commit()
            return True

    async def restore_memory_snapshot(
        self,
        memory: Memory,
        *,
        search_visible_statuses: set[str],
    ) -> None:
        """Restore one memory row and its FTS visibility from a captured snapshot."""
        _validate_visibility(memory.visibility, memory.owner_user_id)
        project_key = _normalize_project_key(memory.project_key)
        entity_names = await self.get_memory_entity_names(memory.id)
        tags_text = " ".join(memory.tags)
        entities_text = " ".join(entity_names)
        search_visible = memory.status in search_visible_statuses
        async with self._write_lock:
            await self.db.execute(
                """UPDATE memories SET
                    memory_type = ?, content = ?, content_hash = ?, tags = ?,
                    visibility = ?, owner_user_id = ?, project_key = ?,
                    repo_identifier = ?,
                    confidence = ?,
                    corroboration_count = ?, contradiction_count = ?,
                    valid_from = ?, valid_until = ?, superseded_by = ?,
                    status = ?, retirement_reason = ?, retired_at = ?,
                    superseded_at = ?, replacement_reason = ?, replacement_kind = ?, extraction_context = ?,
                    updated_at = ?
                   WHERE id = ?""",
                (
                    memory.memory_type,
                    memory.content,
                    memory.content_hash,
                    json.dumps(memory.tags),
                    memory.visibility,
                    memory.owner_user_id,
                    project_key,
                    memory.repo_identifier,
                    memory.confidence,
                    memory.corroboration_count,
                    memory.contradiction_count,
                    memory.valid_from.isoformat() if memory.valid_from else None,
                    memory.valid_until.isoformat() if memory.valid_until else None,
                    memory.superseded_by,
                    memory.status,
                    memory.retirement_reason,
                    memory.retired_at.isoformat() if memory.retired_at else None,
                    memory.superseded_at.isoformat() if memory.superseded_at else None,
                    memory.replacement_reason,
                    memory.replacement_kind,
                    memory.extraction_context,
                    memory.updated_at.isoformat() if memory.updated_at else None,
                    memory.id,
                ),
            )
            await self.db.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory.id,))
            if search_visible:
                await self.db.execute(
                    "INSERT INTO memories_fts (memory_id, content, entities_text, tags_text) VALUES (?, ?, ?, ?)",
                    (memory.id, memory.content, entities_text, tags_text),
                )
            await self.db.commit()

    async def corroborate_memory(
        self,
        memory_id: str,
        doc_id: str,
        source_type: str,
        excerpt: str | None = None,
        *,
        support_kind: str = "extracted",
        source_updated_at: datetime | None,
    ) -> str:
        """Add a supporting source and count only distinct source documents."""
        async with self._write_lock:
            outcome = await self._corroborate_memory_unlocked(
                memory_id,
                doc_id,
                source_type,
                excerpt,
                support_kind=support_kind,
                source_updated_at=source_updated_at,
            )
            await self.db.commit()
            return outcome

    async def corroborate_memory_with_relation_outcome(
        self,
        memory_id: str,
        doc_id: str,
        source_type: str,
        excerpt: str | None = None,
        *,
        support_kind: str = "extracted",
        relation_outcome: RelationOutcomeBundle,
        source_updated_at: datetime | None,
    ) -> str:
        """Add a supporting source and its Evidence Relation audit atomically."""
        async with self._write_lock:
            try:
                outcome = await self._corroborate_memory_unlocked(
                    memory_id,
                    doc_id,
                    source_type,
                    excerpt,
                    support_kind=support_kind,
                    source_updated_at=source_updated_at,
                )
                await self._record_relation_outcome_bundle_unlocked(relation_outcome)
                await self.db.commit()
                return outcome
            except Exception:
                await self.db.rollback()
                raise

    async def _corroborate_memory_unlocked(
        self,
        memory_id: str,
        doc_id: str,
        source_type: str,
        excerpt: str | None = None,
        *,
        support_kind: str = "extracted",
        source_updated_at: datetime | None,
    ) -> str:
        async with self.db.execute(
            """SELECT excerpt, support_kind, source_updated_at
               FROM memory_sources
               WHERE memory_id = ? AND doc_id = ?""",
            (memory_id, doc_id),
        ) as cursor:
            existing = await cursor.fetchone()

        if existing:
            existing_excerpt = existing["excerpt"]
            existing_kind = existing["support_kind"]
            existing_source_updated_at = existing["source_updated_at"]
            next_kind = existing_kind
            if existing_kind == "corroborated" and support_kind == "extracted":
                next_kind = "extracted"
            next_source_updated_at = _utc_iso(source_updated_at) if source_updated_at is not None else None

            should_update_excerpt = bool(
                excerpt
                and excerpt != existing_excerpt
                and (not existing_excerpt or len(excerpt) > len(existing_excerpt))
            )
            if (
                should_update_excerpt
                or next_kind != existing_kind
                or next_source_updated_at != existing_source_updated_at
            ):
                await self.db.execute(
                    """UPDATE memory_sources
                       SET source_type = ?, excerpt = ?, support_kind = ?, source_updated_at = ?
                       WHERE memory_id = ? AND doc_id = ?""",
                    (
                        source_type,
                        excerpt if should_update_excerpt else existing_excerpt,
                        next_kind,
                        next_source_updated_at,
                        memory_id,
                        doc_id,
                    ),
                )
                return "updated"
            return "unchanged"

        cursor = await self.db.execute(
            """INSERT OR IGNORE INTO memory_sources (
                memory_id, doc_id, source_id, source_type, excerpt, support_kind, source_updated_at
            ) VALUES (?, ?, (SELECT source FROM documents WHERE doc_id = ?), ?, ?, ?, ?)""",
            (
                memory_id,
                doc_id,
                doc_id,
                source_type,
                excerpt,
                support_kind,
                _utc_iso(source_updated_at) if source_updated_at is not None else None,
            ),
        )
        await self._refresh_memory_metadata_fts_unlocked(memory_id, doc_id)
        if cursor.rowcount:
            await self.db.execute(
                """UPDATE memories SET
                    corroboration_count = corroboration_count + 1,
                    updated_at = ?
                   WHERE id = ?""",
                (_now_iso(), memory_id),
            )
        return "inserted" if cursor.rowcount else "unchanged"

    async def supersede_memory(
        self,
        old_id: str,
        new_memory: Memory,
        *,
        replacement_reason: str | None = None,
        replacement_kind: ReplacementKind,
    ) -> None:
        """Mark old memory as superseded and insert the new one."""
        async with self._write_lock:
            try:
                await self._supersede_memory_unlocked(
                    old_id,
                    new_memory,
                    replacement_reason=replacement_reason,
                    replacement_kind=replacement_kind,
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def _supersede_memory_unlocked(
        self,
        old_id: str,
        new_memory: Memory,
        *,
        replacement_reason: str | None = None,
        replacement_kind: ReplacementKind,
    ) -> None:
        if old_id == new_memory.id:
            raise ValueError("cannot supersede a memory with itself")
        await self._assert_no_active_source_support_unlocked(old_id)
        replacement_kind = _validate_replacement_kind(replacement_kind)
        _validate_visibility(new_memory.visibility, new_memory.owner_user_id)
        now = _now_iso()
        await self._upsert_memory_preserving_created_at_unlocked(new_memory)
        await self.db.execute(
            """UPDATE memories SET
                status = 'superseded', superseded_by = ?, valid_until = ?,
                superseded_at = ?, replacement_reason = ?, replacement_kind = ?, updated_at = ?
               WHERE id = ?""",
            (
                new_memory.id,
                _today_iso(),
                now,
                replacement_reason,
                replacement_kind,
                now,
                old_id,
            ),
        )
        await self._stale_pending_reviews_unlocked(
            (old_id,),
            now=now,
        )
        await self._rebuild_memory_fts_unlocked(
            new_memory.id,
            search_visible_statuses=set(allowed_search_statuses()),
        )

    async def _assert_no_active_source_support_unlocked(self, memory_id: str) -> None:
        async with self.db.execute(
            """SELECT 1 FROM memory_support_assertions
               WHERE memory_id = ? AND active = 1 LIMIT 1""",
            (memory_id,),
        ) as cursor:
            if await cursor.fetchone() is not None:
                raise ValueError(
                    "direct terminal Memory transition rejected while active source support remains; "
                    "projected lifecycle required"
                )

    async def _assert_legacy_source_write_allowed_unlocked(self, doc_id: str) -> None:
        async with self.db.execute(
            """SELECT g.state
               FROM documents d
               JOIN source_lifecycle_gates g ON g.source_id = d.source
               WHERE d.doc_id = ?""",
            (doc_id,),
        ) as cursor:
            gate = await cursor.fetchone()
        if gate is not None and gate["state"] == LifecycleGateState.ENABLED.value:
            raise ValueError(
                "direct configured-source Memory write rejected after cutover; projected lifecycle required"
            )

    async def supersede_memory_with_source_and_relation(
        self,
        old_id: str,
        new_memory: Memory,
        *,
        replacement_kind: ReplacementKind,
        doc_id: str,
        source_type: str,
        excerpt: str | None = None,
        replacement_reason: str | None = None,
        carry_revision_sources: bool = False,
        entity_ids: Sequence[int] | None = None,
        relation_outcome: RelationOutcomeBundle | None = None,
        source_updated_at: datetime | None,
    ) -> None:
        """Supersede a memory and persist replacement provenance/relation audit atomically."""
        async with self._write_lock:
            try:
                await self._supersede_memory_unlocked(
                    old_id,
                    new_memory,
                    replacement_reason=replacement_reason,
                    replacement_kind=replacement_kind,
                )
                if carry_revision_sources:
                    async with self.db.execute(
                        "SELECT * FROM memory_sources WHERE memory_id = ?",
                        (old_id,),
                    ) as cursor:
                        async for row in cursor:
                            d = dict(row)
                            if d["doc_id"] == doc_id:
                                continue
                            await self._add_memory_source_unlocked(
                                new_memory.id,
                                d["doc_id"],
                                d["source_type"],
                                d["excerpt"],
                                support_kind=d.get("support_kind", "extracted"),
                                source_updated_at=_parse_dt(d.get("source_updated_at")),
                            )
                await self._add_memory_source_unlocked(
                    new_memory.id,
                    doc_id,
                    source_type,
                    excerpt,
                    source_updated_at=source_updated_at,
                )
                await self._link_memory_entities_unlocked(new_memory.id, entity_ids)
                await self._rebuild_memory_fts_unlocked(
                    new_memory.id,
                    search_visible_statuses=set(allowed_search_statuses()),
                )
                if relation_outcome is not None:
                    await self._record_relation_outcome_bundle_unlocked(relation_outcome)
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def resolve_current_memory_id(self, memory_id: str) -> str | None:
        """Return the active head of a memory supersession chain.

        Memory IDs are immutable version IDs. Callers that hold an older ID can
        use this resolver to reach the current version without guessing through
        search results.
        """
        current = memory_id
        seen: set[str] = set()
        while current:
            if current in seen:
                raise RuntimeError(f"Memory supersession chain contains a cycle at {current}")
            seen.add(current)
            memory = await self.get_memory(current)
            if memory is None:
                return None
            if memory.status != "superseded" or not memory.superseded_by:
                return memory.id
            current = memory.superseded_by
        return None

    async def promote_quarantined_challenger(
        self,
        *,
        incumbent_id: str,
        challenger: Memory,
        replacement_reason: str | None = None,
        replacement_kind: ReplacementKind,
    ) -> None:
        """Promote a pending_review challenger to active and supersede the incumbent.

        The challenger row already exists in SQLite (it was inserted as part of
        reconciliation). This rewires lifecycle metadata and the FTS5 index so
        default retrieval picks up the challenger and drops the incumbent.

        ``entities_text`` is derived from ``memory_entities`` so the rebuilt
        FTS row keeps the entity coverage that was wired up at extraction
        time, even when the caller passed a Memory loaded via ``get_memory``
        (which leaves ``entity_refs`` empty by design).
        """
        replacement_kind = _validate_replacement_kind(replacement_kind)
        async with self._write_lock:
            now = _now_iso()
            await self.db.execute(
                """UPDATE memories SET
                    status = 'active', retirement_reason = NULL, retired_at = NULL,
                    updated_at = ?
                   WHERE id = ?""",
                (now, challenger.id),
            )
            await self.db.execute(
                """UPDATE memories SET
                    status = 'superseded', superseded_by = ?, superseded_at = ?,
                    valid_until = ?, replacement_reason = ?, replacement_kind = ?, updated_at = ?
                   WHERE id = ?""",
                (
                    challenger.id,
                    now,
                    _today_iso(),
                    replacement_reason,
                    replacement_kind,
                    now,
                    incumbent_id,
                ),
            )
            await self.db.execute(
                "DELETE FROM memories_fts WHERE memory_id = ?",
                (incumbent_id,),
            )
            await self.db.execute(
                "DELETE FROM memories_fts WHERE memory_id = ?",
                (challenger.id,),
            )
            entity_names: list[str] = []
            async with self.db.execute(
                """SELECT e.canonical_name
                   FROM memory_entities me
                   JOIN entities e ON me.entity_id = e.id
                   WHERE me.memory_id = ?
                   ORDER BY e.id""",
                (challenger.id,),
            ) as cursor:
                async for row in cursor:
                    entity_names.append(row[0])
            entities_text = " ".join(entity_names)
            tags_text = " ".join(challenger.tags)
            await self.db.execute(
                "INSERT INTO memories_fts (memory_id, content, entities_text, tags_text) VALUES (?, ?, ?, ?)",
                (challenger.id, challenger.content, entities_text, tags_text),
            )
            await self.db.commit()

    async def list_memories(
        self,
        type: str | None = None,
        status: str | None = None,
        source: str | None = None,
        project: str | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        query = "SELECT DISTINCT m.* FROM memories m"
        joins: list[str] = []
        conditions: list[str] = ["1=1"]
        params: list = []

        if source:
            joins.append("JOIN memory_sources ms ON m.id = ms.memory_id")
            joins.append("JOIN documents d ON ms.doc_id = d.doc_id")
            conditions.append("d.source = ?")
            params.append(source)
        if type:
            conditions.append("m.memory_type = ?")
            params.append(type)
        if status:
            conditions.append("m.status = ?")
            params.append(normalize_memory_status(status))
        if project:
            conditions.append("m.project_key = ?")
            params.append(project)

        join_clause = " ".join(joins)
        where_clause = " AND ".join(conditions)
        query = f"{query} {join_clause} WHERE {where_clause} ORDER BY m.updated_at DESC LIMIT ?"
        params.append(limit)

        results: list[Memory] = []
        async with self.db.execute(query, params) as cursor:
            async for row in cursor:
                results.append(self._row_to_memory(row))
        return results

    async def query_memory_admin_page(
        self,
        *,
        scope,
        filters: MemoryAdminListFilters,
        limit: int,
        offset: int,
    ) -> MemoryAdminQueryPage:
        predicate_sql, predicate_params = visible_sql(scope, "m")
        subscription_condition, subscription_params = _enabled_source_visibility_condition(
            await self.list_disabled_source_ids_for_user(scope.user_id)
        )
        if filters.search:
            fts_query = _admin_fts_query(filters.search)
            like_query = _admin_like_pattern(filters.search)
            conditions: list[str] = [
                predicate_sql,
                """(
                    m.id IN (
                        SELECT memory_id
                        FROM memories_fts
                        WHERE memories_fts MATCH ?
                    )
                    OR EXISTS (
                        SELECT 1
                        FROM memory_sources ms_search
                        JOIN documents d_search ON ms_search.doc_id = d_search.doc_id
                        WHERE ms_search.memory_id = m.id
                          AND (
                            d_search.doc_id LIKE ? ESCAPE '\\'
                            OR d_search.title LIKE ? ESCAPE '\\'
                            OR d_search.source_url LIKE ? ESCAPE '\\'
                          )
                    )
                )""",
            ]
            params: list[Any] = [
                *predicate_params,
                fts_query,
                like_query,
                like_query,
                like_query,
            ]
            if subscription_condition:
                conditions.append(subscription_condition)
                params.extend(subscription_params)

            if filters.source:
                conditions.append(
                    """EXISTS (
                        SELECT 1
                        FROM memory_sources ms_filter
                        JOIN documents d_filter ON ms_filter.doc_id = d_filter.doc_id
                        WHERE ms_filter.memory_id = m.id
                          AND d_filter.source = ?
                    )"""
                )
                params.append(filters.source)
            if filters.memory_type:
                conditions.append("m.memory_type = ?")
                params.append(filters.memory_type)
            if filters.status:
                conditions.append("m.status = ?")
                params.append(normalize_memory_status(filters.status))
            if filters.project:
                conditions.append("m.project_key = ?")
                params.append(filters.project)

            where_clause = " AND ".join(conditions)
            memories: list[Memory] = []
            query = (
                f"SELECT DISTINCT m.* FROM memories m WHERE {where_clause} ORDER BY m.updated_at DESC LIMIT ? OFFSET ?"
            )
            async with self.db.execute(query, [*params, limit, offset]) as cursor:
                async for row in cursor:
                    memories.append(self._row_to_memory(row))

            count_query = f"SELECT COUNT(DISTINCT m.id) FROM memories m WHERE {where_clause}"
            async with self.db.execute(count_query, params) as cursor:
                total_row = await cursor.fetchone()
                total = total_row[0] if total_row else 0
            return MemoryAdminQueryPage(memories=memories, total=total)

        query = "SELECT DISTINCT m.* FROM memories m"
        joins: list[str] = []
        conditions: list[str] = [predicate_sql]
        params = list(predicate_params)
        if subscription_condition:
            conditions.append(subscription_condition)
            params.extend(subscription_params)

        if filters.source:
            joins.append("JOIN memory_sources ms ON m.id = ms.memory_id")
            joins.append("JOIN documents d ON ms.doc_id = d.doc_id")
            conditions.append("d.source = ?")
            params.append(filters.source)
        if filters.memory_type:
            conditions.append("m.memory_type = ?")
            params.append(filters.memory_type)
        if filters.status:
            conditions.append("m.status = ?")
            params.append(normalize_memory_status(filters.status))
        if filters.project:
            conditions.append("m.project_key = ?")
            params.append(filters.project)

        join_clause = " ".join(joins)
        where_clause = " AND ".join(conditions)

        count_q = f"SELECT COUNT(DISTINCT m.id) FROM memories m {join_clause} WHERE {where_clause}"
        async with self.db.execute(count_q, params) as cursor:
            total_row = await cursor.fetchone()
            total = total_row[0] if total_row else 0

        full_q = f"{query} {join_clause} WHERE {where_clause} ORDER BY m.updated_at DESC LIMIT ? OFFSET ?"
        memories = []
        async with self.db.execute(full_q, [*params, limit, offset]) as cursor:
            async for row in cursor:
                memories.append(self._row_to_memory(row))
        return MemoryAdminQueryPage(memories=memories, total=total)

    async def count_memories(
        self,
        type: str | None = None,
        status: str | None = None,
    ) -> int:
        query = "SELECT COUNT(*) FROM memories WHERE 1=1"
        params: list = []
        if type:
            query += " AND memory_type = ?"
            params.append(type)
        if status:
            query += " AND status = ?"
            params.append(normalize_memory_status(status))
        async with self.db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def get_memories_with_deleted_sources(self) -> list[Memory]:
        """Return active memories whose source documents no longer exist."""
        results: list[Memory] = []
        async with self.db.execute(
            """SELECT m.* FROM memories m
               WHERE m.status = 'active'
               AND NOT EXISTS (
                   SELECT 1 FROM memory_sources ms
                   JOIN documents d ON ms.doc_id = d.doc_id
                   WHERE ms.memory_id = m.id
               )"""
        ) as cursor:
            async for row in cursor:
                results.append(self._row_to_memory(row))
        return results

    async def retire_expired_memories(self) -> int:
        """Retire active memories whose ``valid_until`` date has passed."""
        async with self._write_lock:
            now = _now_iso()
            today = _today_iso()
            async with self.db.execute(
                """SELECT id FROM memories
                   WHERE status = 'active'
                     AND valid_until IS NOT NULL
                     AND valid_until < ?""",
                (today,),
            ) as rows:
                expiring_ids = tuple([str(row["id"]) async for row in rows])
            cursor = await self.db.execute(
                """UPDATE memories SET
                    status = 'retired', retirement_reason = 'expired',
                    retired_at = COALESCE(retired_at, ?), updated_at = ?
                   WHERE status = 'active'
                   AND valid_until IS NOT NULL
                   AND valid_until < ?""",
                (now, now, today),
            )
            await self._stale_pending_reviews_unlocked(
                expiring_ids,
                now=now,
            )
            await self.db.commit()
            return cursor.rowcount

    async def get_expired_memories(self) -> list[Memory]:
        """Return active memories whose valid_until date has passed."""
        results: list[Memory] = []
        today = _today_iso()
        async with self.db.execute(
            """SELECT * FROM memories
               WHERE status = 'active'
               AND valid_until IS NOT NULL
               AND valid_until < ?""",
            (today,),
        ) as cursor:
            async for row in cursor:
                results.append(self._row_to_memory(row))
        return results

    # ==================================================================
    # Memory Sources
    # ==================================================================

    async def add_memory_source(
        self,
        memory_id: str,
        doc_id: str,
        source_type: str,
        excerpt: str | None = None,
        *,
        support_kind: str = "extracted",
        source_updated_at: datetime | None,
    ) -> None:
        async with self._write_lock:
            await self._add_memory_source_unlocked(
                memory_id,
                doc_id,
                source_type,
                excerpt,
                support_kind=support_kind,
                source_updated_at=source_updated_at,
            )
            await self.db.commit()

    async def _add_memory_source_unlocked(
        self,
        memory_id: str,
        doc_id: str,
        source_type: str,
        excerpt: str | None = None,
        *,
        support_kind: str = "extracted",
        source_updated_at: datetime | None,
    ) -> None:
        await self.db.execute(
            """INSERT INTO memory_sources (
                memory_id, doc_id, source_id, source_type, excerpt, support_kind, source_updated_at
            ) VALUES (?, ?, (SELECT source FROM documents WHERE doc_id = ?), ?, ?, ?, ?)
            ON CONFLICT(memory_id, doc_id) DO UPDATE SET
                source_id = excluded.source_id,
                source_type = excluded.source_type,
                excerpt = excluded.excerpt,
                support_kind = excluded.support_kind,
                source_updated_at = excluded.source_updated_at""",
            (
                memory_id,
                doc_id,
                doc_id,
                source_type,
                excerpt,
                support_kind,
                _utc_iso(source_updated_at) if source_updated_at is not None else None,
            ),
        )
        await self._refresh_memory_metadata_fts_unlocked(memory_id, doc_id)

    async def _refresh_memory_metadata_fts_unlocked(self, memory_id: str, doc_id: str) -> None:
        await self.db.execute(
            "DELETE FROM memory_search_metadata_fts WHERE memory_id = ? AND doc_id = ?",
            (memory_id, doc_id),
        )
        await self.db.execute(
            "DELETE FROM memory_search_metadata_alias_fts WHERE memory_id = ? AND doc_id = ?",
            (memory_id, doc_id),
        )
        await self.db.execute(
            "DELETE FROM memory_search_metadata_trigram WHERE memory_id = ? AND doc_id = ?",
            (memory_id, doc_id),
        )
        rows = await self.db.execute_fetchall(
            """SELECT
                   ms.memory_id,
                   ms.source_id,
                   ms.doc_id,
                   ms.source_type,
                   d.title,
                   d.source_url,
                   d.space_or_project,
                   d.labels,
                   s.name AS source_name
               FROM memory_sources ms
               JOIN documents d ON d.doc_id = ms.doc_id
               LEFT JOIN sources s ON s.id = ms.source_id
              WHERE ms.memory_id = ? AND ms.doc_id = ?""",
            (memory_id, doc_id),
        )
        if not rows:
            return
        record = rows[0]
        labels_text = ""
        try:
            labels = json.loads(record["labels"] or "[]")
            if isinstance(labels, list):
                labels_text = " ".join(str(label) for label in labels)
        except json.JSONDecodeError:
            labels_text = str(record["labels"] or "")
        await self.db.execute(
            """INSERT INTO memory_search_metadata_fts (
                   memory_id,
                   source_id,
                   doc_id,
                   source_type,
                   metadata_title_tokens,
                   metadata_external_id_tokens,
                   metadata_path_tokens,
                   metadata_source_name_tokens,
                   metadata_label_context_tokens
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record["memory_id"],
                record["source_id"],
                record["doc_id"],
                record["source_type"],
                record["title"] or "",
                record["doc_id"] or "",
                " ".join(part for part in (record["space_or_project"] or "", record["source_url"] or "") if part),
                record["source_name"] or "",
                labels_text,
            ),
        )
        metadata_values = (
            record["title"] or "",
            record["doc_id"] or "",
            record["space_or_project"] or "",
            record["source_url"] or "",
            record["source_name"] or "",
            labels_text,
        )
        await self.db.execute(
            """INSERT INTO memory_search_metadata_alias_fts (
                   memory_id,
                   source_id,
                   doc_id,
                   source_type,
                   metadata_alias_tokens
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                record["memory_id"],
                record["source_id"],
                record["doc_id"],
                record["source_type"],
                metadata_alias_text(metadata_values),
            ),
        )
        await self.db.execute(
            """INSERT OR REPLACE INTO memory_search_metadata_trigram (
                   memory_id,
                   source_id,
                   doc_id,
                   source_type,
                   metadata_compact
               ) VALUES (?, ?, ?, ?, ?)""",
            (
                record["memory_id"],
                record["source_id"],
                record["doc_id"],
                record["source_type"],
                metadata_compact_text(metadata_values),
            ),
        )

    async def _refresh_metadata_fts_for_doc_unlocked(self, doc_id: str) -> None:
        rows = await self.db.execute_fetchall(
            "SELECT memory_id FROM memory_sources WHERE doc_id = ?",
            (doc_id,),
        )
        for row in rows:
            await self._refresh_memory_metadata_fts_unlocked(row["memory_id"], doc_id)

    async def _refresh_metadata_fts_for_source_unlocked(self, source_id: str) -> None:
        rows = await self.db.execute_fetchall(
            "SELECT memory_id, doc_id FROM memory_sources WHERE source_id = ?",
            (source_id,),
        )
        for row in rows:
            await self._refresh_memory_metadata_fts_unlocked(
                row["memory_id"],
                row["doc_id"],
            )

    async def rebuild_memory_metadata_fts(self) -> None:
        """Rebuild the source-metadata FTS projection from durable support rows."""
        async with self._write_lock:
            await self._rebuild_memory_metadata_fts_unlocked()
            await self.db.commit()

    async def _rebuild_memory_metadata_fts_unlocked(self) -> None:
        await self.db.execute("DELETE FROM memory_search_metadata_fts")
        await self.db.execute("DELETE FROM memory_search_metadata_alias_fts")
        await self.db.execute("DELETE FROM memory_search_metadata_trigram")
        rows = await self.db.execute_fetchall("SELECT memory_id, doc_id FROM memory_sources ORDER BY memory_id, doc_id")
        for row in rows:
            await self._refresh_memory_metadata_fts_unlocked(
                row["memory_id"],
                row["doc_id"],
            )

    async def _link_memory_entities_unlocked(
        self,
        memory_id: str,
        entity_ids: Sequence[int] | None,
    ) -> None:
        for entity_id in entity_ids or ():
            await self.db.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
                (memory_id, entity_id),
            )

    async def restore_memory_source_snapshot(self, source: MemorySource) -> None:
        """Restore one memory source row from a captured snapshot."""
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO memory_sources (
                    memory_id, doc_id, source_id, source_type, excerpt, support_kind, added_at, source_updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, doc_id) DO UPDATE SET
                    source_id=excluded.source_id,
                    source_type=excluded.source_type,
                    excerpt=excluded.excerpt,
                    support_kind=excluded.support_kind,
                    added_at=excluded.added_at,
                    source_updated_at=excluded.source_updated_at""",
                (
                    source.memory_id,
                    source.doc_id,
                    source.source_id,
                    source.source_type,
                    source.excerpt,
                    source.support_kind,
                    source.added_at.isoformat() if source.added_at else _now_iso(),
                    _utc_iso(source.source_updated_at) if source.source_updated_at else None,
                ),
            )
            await self.db.commit()

    async def get_memory_sources(self, memory_id: str) -> list[MemorySource]:
        results: list[MemorySource] = []
        async with self.db.execute(
            """SELECT * FROM memory_sources
               WHERE memory_id = ?
               ORDER BY
                   CASE WHEN support_kind = 'extracted' THEN 0 ELSE 1 END,
                   added_at DESC,
                   doc_id ASC""",
            (memory_id,),
        ) as cursor:
            async for row in cursor:
                d = dict(row)
                results.append(
                    MemorySource(
                        memory_id=d["memory_id"],
                        doc_id=d["doc_id"],
                        source_type=d["source_type"],
                        source_id=d.get("source_id"),
                        excerpt=d["excerpt"],
                        support_kind=d.get("support_kind", "extracted"),
                        added_at=_parse_dt(d["added_at"]),
                        source_updated_at=_parse_dt(d.get("source_updated_at")),
                    )
                )
        return results

    async def get_memory_ids_for_doc(self, doc_id: str) -> list[str]:
        ids: list[str] = []
        async with self.db.execute(
            "SELECT memory_id FROM memory_sources WHERE doc_id = ?",
            (doc_id,),
        ) as cursor:
            async for row in cursor:
                ids.append(str(row[0]))
        return list(dict.fromkeys(ids))

    # ==================================================================
    # Agent Knowledge Bundle
    # ==================================================================

    async def upsert_agent_concept(
        self,
        *,
        concept_id: str,
        source_id: str,
        owner_user_id: str,
        workspace: str,
        repo_identifier: str | None,
        concept_type: str,
        concept_path: str,
        title: str,
        markdown_body: str,
        frontmatter: dict[str, Any],
        observed_at: datetime,
    ) -> None:
        """Insert or update a private agent-session concept."""
        observed = _utc_iso(observed_at)
        async with self._write_lock:
            try:
                await self._upsert_agent_concept_unlocked(
                    concept_id=concept_id,
                    source_id=source_id,
                    owner_user_id=owner_user_id,
                    workspace=workspace,
                    repo_identifier=repo_identifier,
                    concept_type=concept_type,
                    concept_path=concept_path,
                    title=title,
                    markdown_body=markdown_body,
                    frontmatter=frontmatter,
                    observed=observed,
                )
                await self.db.commit()
            except BaseException:
                rollback_task = asyncio.create_task(self.db.rollback())
                await _drain_task_despite_cancellation(rollback_task)
                raise

    async def _upsert_agent_concept_unlocked(
        self,
        *,
        concept_id: str,
        source_id: str,
        owner_user_id: str,
        workspace: str,
        repo_identifier: str | None,
        concept_type: str,
        concept_path: str,
        title: str,
        markdown_body: str,
        frontmatter: dict[str, Any],
        observed: str,
    ) -> None:
        async with self.db.execute(
            "SELECT access_policy, owner_user_id FROM sources WHERE id = ?",
            (source_id,),
        ) as cursor:
            source_access = await cursor.fetchone()
        if source_access is None:
            raise ValueError(f"agent concept source does not exist: {source_id}")
        if str(source_access["owner_user_id"]) != owner_user_id:
            raise ValueError("agent concept owner must match source owner")
        visibility = str(source_access["access_policy"])
        await self.db.execute(
            """INSERT INTO agent_concepts (
                id, source_id, owner_user_id, visibility, workspace,
                repo_identifier, concept_type, concept_path, title,
                markdown_body, frontmatter_json, created_at, updated_at,
                last_observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_id=excluded.source_id,
                owner_user_id=excluded.owner_user_id,
                visibility=excluded.visibility,
                workspace=excluded.workspace,
                repo_identifier=excluded.repo_identifier,
                concept_type=excluded.concept_type,
                concept_path=excluded.concept_path,
                title=excluded.title,
                markdown_body=excluded.markdown_body,
                frontmatter_json=excluded.frontmatter_json,
                updated_at=excluded.updated_at,
                last_observed_at=excluded.last_observed_at""",
            (
                concept_id,
                source_id,
                owner_user_id,
                visibility,
                workspace,
                repo_identifier,
                concept_type,
                concept_path,
                title,
                markdown_body,
                json.dumps(frontmatter, sort_keys=True),
                observed,
                observed,
                observed,
            ),
        )

    async def get_agent_concept(self, concept_id: str) -> dict[str, Any] | None:
        async with self.db.execute(
            "SELECT * FROM agent_concepts WHERE id = ?",
            (concept_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_agent_concepts(
        self,
        *,
        viewer_user_id: str,
        repo_identifier: str | None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        concepts: list[dict[str, Any]] = []
        async with self.db.execute(
            """SELECT c.* FROM agent_concepts c
               JOIN sources s ON s.id = c.source_id
               LEFT JOIN source_subscriptions ss
                 ON ss.source_id = s.id AND ss.user_id = ?
               WHERE (
                   (s.access_state = 'active'
                    AND (s.access_policy = 'workspace' OR s.owner_user_id = ?))
                   OR (s.access_state = 'changing' AND s.owner_user_id = ?)
               )
                 AND COALESCE(ss.enabled, 1) = 1
                 AND COALESCE(c.repo_identifier, '') = COALESCE(?, '')
               ORDER BY c.updated_at DESC, c.id
               LIMIT ?""",
            (viewer_user_id, viewer_user_id, viewer_user_id, repo_identifier, limit),
        ) as cursor:
            async for row in cursor:
                concepts.append(dict(row))
        return concepts

    async def update_agent_concept_markdown(
        self,
        *,
        concept_id: str,
        markdown_body: str,
        observed_at: datetime,
    ) -> None:
        observed = _utc_iso(observed_at)
        async with self._write_lock:
            await self._update_agent_concept_markdown_unlocked(
                concept_id=concept_id,
                markdown_body=markdown_body,
                observed=observed,
            )
            await self.db.commit()

    async def _update_agent_concept_markdown_unlocked(
        self,
        *,
        concept_id: str,
        markdown_body: str,
        observed: str,
    ) -> None:
        async with self.db.execute("SELECT 1 FROM agent_concepts WHERE id = ?", (concept_id,)) as cursor:
            if await cursor.fetchone() is None:
                raise RuntimeError(f"agent concept projection target missing: {concept_id}")
        await self.db.execute(
            """UPDATE agent_concepts SET
                markdown_body = CASE
                    WHEN updated_at IS NULL OR updated_at <= ? THEN ?
                    ELSE markdown_body
                END,
                updated_at = CASE
                    WHEN updated_at IS NULL OR updated_at <= ? THEN ?
                    ELSE updated_at
                END,
                last_observed_at = CASE
                    WHEN last_observed_at IS NULL OR last_observed_at <= ? THEN ?
                    ELSE last_observed_at
                END
               WHERE id = ?""",
            (observed, markdown_body, observed, observed, observed, observed, concept_id),
        )

    async def upsert_agent_claim(
        self,
        *,
        claim_id: str,
        concept_id: str,
        display_anchor: str,
        claim_text: str,
        memory_type: str,
        tags: list[str],
        confidence: float,
        memory_id: str,
        observed_at: datetime,
    ) -> None:
        observed = _utc_iso(observed_at)
        async with self._write_lock:
            await self._upsert_agent_claim_unlocked(
                claim_id=claim_id,
                concept_id=concept_id,
                display_anchor=display_anchor,
                claim_text=claim_text,
                memory_type=memory_type,
                tags=tags,
                confidence=confidence,
                memory_id=memory_id,
                observed=observed,
            )
            await self.db.commit()

    async def _upsert_agent_claim_unlocked(
        self,
        *,
        claim_id: str,
        concept_id: str,
        display_anchor: str,
        claim_text: str,
        memory_type: str,
        tags: list[str],
        confidence: float,
        memory_id: str,
        observed: str,
    ) -> None:
        await self.db.execute(
            """INSERT INTO agent_claims (
                id, concept_id, display_anchor, claim_text, memory_type,
                tags, confidence, memory_id, created_at, updated_at,
                last_observed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                concept_id=excluded.concept_id,
                display_anchor=excluded.display_anchor,
                claim_text=excluded.claim_text,
                memory_type=excluded.memory_type,
                tags=excluded.tags,
                confidence=excluded.confidence,
                memory_id=excluded.memory_id,
                updated_at=excluded.updated_at,
                last_observed_at=excluded.last_observed_at""",
            (
                claim_id,
                concept_id,
                display_anchor,
                claim_text,
                memory_type,
                json.dumps(tags),
                confidence,
                memory_id,
                observed,
                observed,
                observed,
            ),
        )

    async def supersede_memory_and_upsert_agent_claim(
        self,
        old_id: str,
        new_memory: Memory,
        *,
        doc_id: str,
        source_type: str,
        excerpt: str | None,
        carry_revision_sources: bool,
        entity_ids: Sequence[int] | None = None,
        replacement_reason: str | None,
        replacement_kind: ReplacementKind,
        claim_id: str,
        concept_id: str,
        display_anchor: str,
        claim_text: str,
        memory_type: str,
        tags: list[str],
        confidence: float,
        observed_at: datetime,
        source_updated_at: datetime | None,
        citations: list[str] | None = None,
        concept_markdown_body: str | None = None,
        relation_outcome: RelationOutcomeBundle | None = None,
    ) -> None:
        """Supersede a memory and move its agent-claim projection atomically."""
        if old_id == new_memory.id:
            raise ValueError("cannot supersede a memory with itself")
        replacement_kind = _validate_replacement_kind(replacement_kind)
        _validate_visibility(new_memory.visibility, new_memory.owner_user_id)
        project_key = _normalize_project_key(new_memory.project_key)
        observed = _utc_iso(observed_at)
        async with self._write_lock:
            try:
                async with self.db.execute(
                    "SELECT created_at FROM agent_claims WHERE id = ?",
                    (claim_id,),
                ) as cursor:
                    existing_claim = await cursor.fetchone()
                created_at = existing_claim["created_at"] if existing_claim else observed
                now = _now_iso()
                new_status = normalize_memory_status(new_memory.status)
                await self.db.execute(
                    """INSERT INTO memories (
                    id, memory_type, content, content_hash, tags, visibility, owner_user_id,
                    project_key, repo_identifier,
                    confidence, corroboration_count,
                    contradiction_count, valid_from, valid_until,
                    superseded_by, status, retirement_reason, retired_at,
                    superseded_at, replacement_reason, replacement_kind, extraction_context,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        new_memory.id,
                        new_memory.memory_type,
                        new_memory.content,
                        new_memory.content_hash,
                        json.dumps(new_memory.tags),
                        new_memory.visibility,
                        new_memory.owner_user_id,
                        project_key,
                        new_memory.repo_identifier,
                        new_memory.confidence,
                        new_memory.corroboration_count,
                        new_memory.contradiction_count,
                        new_memory.valid_from.isoformat() if new_memory.valid_from else None,
                        new_memory.valid_until.isoformat() if new_memory.valid_until else None,
                        new_memory.superseded_by,
                        new_status,
                        new_memory.retirement_reason,
                        new_memory.retired_at.isoformat() if new_memory.retired_at else None,
                        new_memory.superseded_at.isoformat() if new_memory.superseded_at else None,
                        new_memory.replacement_reason,
                        new_memory.replacement_kind,
                        new_memory.extraction_context,
                        new_memory.created_at.isoformat() if new_memory.created_at else now,
                        now,
                    ),
                )
                await self.db.execute(
                    """UPDATE memories SET
                    status = 'superseded', superseded_by = ?, valid_until = ?,
                    superseded_at = ?, replacement_reason = ?, replacement_kind = ?, updated_at = ?
                   WHERE id = ?""",
                    (new_memory.id, now, now, replacement_reason, replacement_kind, now, old_id),
                )
                await self._stale_pending_reviews_unlocked(
                    (old_id,),
                    now=now,
                )
                await self.db.execute(
                    """INSERT INTO agent_claims (
                    id, concept_id, display_anchor, claim_text, memory_type,
                    tags, confidence, memory_id, created_at, updated_at,
                    last_observed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    concept_id=excluded.concept_id,
                    display_anchor=excluded.display_anchor,
                    claim_text=excluded.claim_text,
                    memory_type=excluded.memory_type,
                    tags=excluded.tags,
                    confidence=excluded.confidence,
                    memory_id=excluded.memory_id,
                    updated_at=excluded.updated_at,
                    last_observed_at=excluded.last_observed_at""",
                    (
                        claim_id,
                        concept_id,
                        display_anchor,
                        claim_text,
                        memory_type,
                        json.dumps(tags),
                        confidence,
                        new_memory.id,
                        created_at,
                        observed,
                        observed,
                    ),
                )
                if carry_revision_sources:
                    async with self.db.execute(
                        "SELECT * FROM memory_sources WHERE memory_id = ? AND doc_id <> ?",
                        (old_id, doc_id),
                    ) as cursor:
                        async for row in cursor:
                            await self._add_memory_source_unlocked(
                                new_memory.id,
                                row["doc_id"],
                                row["source_type"],
                                row["excerpt"],
                                support_kind=row["support_kind"] or "extracted",
                                source_updated_at=_parse_dt(row["source_updated_at"]),
                            )
                await self._add_memory_source_unlocked(
                    new_memory.id,
                    doc_id,
                    source_type,
                    excerpt,
                    support_kind="extracted",
                    source_updated_at=source_updated_at,
                )
                await self._link_memory_entities_unlocked(new_memory.id, entity_ids)
                await self._rebuild_memory_fts_unlocked(
                    new_memory.id,
                    search_visible_statuses=set(allowed_search_statuses()),
                )
                for citation_url in citations or []:
                    await self._add_agent_claim_citation_unlocked(
                        claim_id=claim_id,
                        citation_url=citation_url,
                        observed=observed,
                    )
                if concept_markdown_body is not None:
                    await self._update_agent_concept_markdown_unlocked(
                        concept_id=concept_id,
                        markdown_body=concept_markdown_body,
                        observed=observed,
                    )
                if relation_outcome is not None:
                    await self._record_relation_outcome_bundle_unlocked(relation_outcome)
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def get_agent_claim(self, claim_id: str) -> dict[str, Any] | None:
        async with self.db.execute(
            "SELECT * FROM agent_claims WHERE id = ?",
            (claim_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_agent_claim(row)

    async def get_agent_claim_by_memory_id(self, memory_id: str) -> dict[str, Any] | None:
        async with self.db.execute(
            "SELECT * FROM agent_claims WHERE memory_id = ? ORDER BY updated_at DESC, id LIMIT 1",
            (memory_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if not row:
            return None
        return self._row_to_agent_claim(row)

    async def list_agent_claims(self, concept_id: str, *, include_inactive: bool = False) -> list[dict[str, Any]]:
        claims: list[dict[str, Any]] = []
        if include_inactive:
            query = "SELECT ac.* FROM agent_claims ac WHERE ac.concept_id = ? ORDER BY ac.created_at, ac.id"
            params = (concept_id,)
        else:
            query = """SELECT ac.* FROM agent_claims ac
                       JOIN memories m ON m.id = ac.memory_id
                       WHERE ac.concept_id = ? AND m.status = 'active'
                       ORDER BY ac.created_at, ac.id"""
            params = (concept_id,)
        async with self.db.execute(query, params) as cursor:
            async for row in cursor:
                claims.append(self._row_to_agent_claim(row))
        return claims

    async def add_agent_claim_citation(
        self,
        *,
        claim_id: str,
        citation_url: str,
        observed_at: datetime,
    ) -> None:
        observed = _utc_iso(observed_at)
        async with self._write_lock:
            await self._add_agent_claim_citation_unlocked(
                claim_id=claim_id,
                citation_url=citation_url,
                observed=observed,
            )
            await self.db.commit()

    async def _add_agent_claim_citation_unlocked(
        self,
        *,
        claim_id: str,
        citation_url: str,
        observed: str,
    ) -> None:
        await self.db.execute(
            """INSERT INTO agent_claim_citations (
                claim_id, citation_url, observed_at, created_at
            ) VALUES (?, ?, ?, ?)
            ON CONFLICT(claim_id, citation_url) DO UPDATE SET
                observed_at=excluded.observed_at""",
            (claim_id, citation_url, observed, observed),
        )

    async def list_agent_claim_citations(self, claim_id: str) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        async with self.db.execute(
            """SELECT claim_id, citation_url, observed_at, created_at
               FROM agent_claim_citations
               WHERE claim_id = ?
               ORDER BY created_at, citation_url""",
            (claim_id,),
        ) as cursor:
            async for row in cursor:
                citations.append(dict(row))
        return citations

    def _row_to_agent_claim(self, row) -> dict[str, Any]:
        data = dict(row)
        try:
            data["tags"] = json.loads(data.get("tags") or "[]")
        except (TypeError, json.JSONDecodeError):
            data["tags"] = []
        return data

    async def get_origin_source_pairs(
        self, memory_ids: list[str]
    ) -> dict[str, list[tuple[str, str | None, str | None]]]:
        """Return each memory's (source_type, support_kind, client) triples, ordered
        oldest-first by (added_at, doc_id), for a batch of memories in one query.
        The client value comes from documents.client for agent-submitted sources.
        Memories with no sources are absent from the result."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" for _ in memory_ids)
        grouped: dict[str, list[tuple[str, str | None, str | None]]] = {}
        async with self.db.execute(
            f"""SELECT ms.memory_id, ms.source_type, ms.support_kind, d.client
                FROM memory_sources ms
                LEFT JOIN documents d ON d.doc_id = ms.doc_id
                WHERE ms.memory_id IN ({placeholders})
                ORDER BY ms.added_at ASC, ms.doc_id ASC""",
            memory_ids,
        ) as cursor:
            async for row in cursor:
                d = dict(row)
                grouped.setdefault(d["memory_id"], []).append(
                    (d["source_type"], d.get("support_kind"), d.get("client"))
                )
        return grouped

    async def get_corroborated_sources_by_doc(self, doc_id: str) -> list[MemorySource]:
        results: list[MemorySource] = []
        async with self.db.execute(
            """SELECT * FROM memory_sources
               WHERE doc_id = ? AND support_kind = 'corroborated'""",
            (doc_id,),
        ) as cursor:
            async for row in cursor:
                d = dict(row)
                results.append(
                    MemorySource(
                        memory_id=d["memory_id"],
                        doc_id=d["doc_id"],
                        source_type=d["source_type"],
                        source_id=d.get("source_id"),
                        excerpt=d["excerpt"],
                        support_kind=d.get("support_kind", "corroborated"),
                        added_at=_parse_dt(d["added_at"]),
                        source_updated_at=_parse_dt(d.get("source_updated_at")),
                    )
                )
        return results

    async def get_source_support_candidates(
        self,
        *,
        doc_id: str,
        entity_ids: list[int],
        project_key: str | None = None,
        limit: int = 30,
        writer_visibility: str | None = None,
        writer_owner_user_id: str | None = None,
        writer_project_key: str | None = None,
        excluded_source_ids: Sequence[str] = (),
    ) -> list[Memory]:
        """Rank active memories that may be supported by the current document.

        When ``writer_visibility`` is provided, the candidate pool is narrowed
        to the same visibility tier as the writer; private writers see only
        their own owner's set, and workspace writers see only candidates in
        their own project. Callers that omit the writer args (legacy and tests)
        receive the unscoped pool.
        """
        if not entity_ids:
            return []

        placeholders = ",".join("?" for _ in entity_ids)
        scope_clauses: list[str] = []
        scope_params: list[Any] = []
        if writer_visibility is not None:
            scope_clauses.append("AND m.visibility = ?")
            scope_params.append(writer_visibility)
            if writer_visibility == Visibility.PRIVATE.value and writer_owner_user_id is not None:
                scope_clauses.append("AND m.owner_user_id = ?")
                scope_params.append(writer_owner_user_id)
            if writer_visibility == Visibility.WORKSPACE.value:
                # NULL project_key is normalized to UNSORTED at persistence
                # time; resolve the writer side the same way so the candidate
                # pool stays inside one project boundary.
                scope_clauses.append("AND m.project_key = ?")
                scope_params.append(writer_project_key or UNSORTED_PROJECT_KEY)
        if excluded_source_ids:
            placeholders_sources = ",".join("?" for _ in excluded_source_ids)
            scope_clauses.append(
                f"""AND (
                    NOT EXISTS (
                        SELECT 1 FROM memory_sources ms_any
                        WHERE ms_any.memory_id = m.id
                    )
                    OR EXISTS (
                        SELECT 1 FROM memory_sources ms_enabled
                        WHERE ms_enabled.memory_id = m.id
                          AND (ms_enabled.source_id IS NULL OR ms_enabled.source_id NOT IN ({placeholders_sources}))
                    )
                )"""
            )
            scope_params.extend(excluded_source_ids)
        scope_sql = ("\n              " + "\n              ".join(scope_clauses)) if scope_clauses else ""
        sql = f"""
            SELECT m.*,
                   COUNT(DISTINCT me.entity_id) AS entity_overlap,
                   CASE WHEN ? IS NOT NULL AND m.project_key = ? THEN 1 ELSE 0 END AS same_project
            FROM memories m
            JOIN memory_entities me ON m.id = me.memory_id
            WHERE me.entity_id IN ({placeholders})
              AND m.status = 'active'{scope_sql}
              AND NOT EXISTS (
                  SELECT 1 FROM memory_sources ms
                  WHERE ms.memory_id = m.id AND ms.doc_id = ?
              )
            GROUP BY m.id
            ORDER BY same_project DESC,
                     entity_overlap DESC,
                     m.corroboration_count DESC,
                     m.confidence DESC,
                     m.updated_at DESC
            LIMIT ?
        """
        params = [project_key, project_key, *entity_ids, *scope_params, doc_id, limit]
        results: list[Memory] = []
        async with self.db.execute(sql, params) as cursor:
            async for row in cursor:
                results.append(self._row_to_memory(row))
        return results

    async def remove_memory_source(
        self,
        memory_id: str,
        doc_id: str,
        *,
        retire_reason: str = "source_deleted",
    ) -> bool:
        """Remove one source link and refresh support-derived memory state.

        Returns ``True`` when the memory was retired.
        """
        async with self._write_lock:
            await self._delete_evidence_graph_for_memory_doc_unlocked(memory_id, doc_id)
            await self.db.execute(
                "DELETE FROM memory_search_metadata_fts WHERE memory_id = ? AND doc_id = ?",
                (memory_id, doc_id),
            )
            await self.db.execute(
                "DELETE FROM memory_search_metadata_alias_fts WHERE memory_id = ? AND doc_id = ?",
                (memory_id, doc_id),
            )
            await self.db.execute(
                "DELETE FROM memory_search_metadata_trigram WHERE memory_id = ? AND doc_id = ?",
                (memory_id, doc_id),
            )
            await self.db.execute(
                "DELETE FROM memory_sources WHERE memory_id = ? AND doc_id = ?",
                (memory_id, doc_id),
            )
            retired = await self._refresh_memory_support_state_unlocked(
                memory_id,
                retire_reason=retire_reason,
            )
            await self.db.commit()
            return retired

    async def refresh_memory_support_state(
        self,
        memory_id: str,
        *,
        retire_reason: str = "source_deleted",
    ) -> bool:
        """Recompute source counts and retire memories with no valid source support."""
        async with self._write_lock:
            retired = await self._refresh_memory_support_state_unlocked(
                memory_id,
                retire_reason=retire_reason,
            )
            await self.db.commit()
            return retired

    async def _refresh_memory_support_state_unlocked(
        self,
        memory_id: str,
        *,
        retire_reason: str,
    ) -> bool:
        async with self.db.execute(
            """SELECT
                   COUNT(*) AS total_count
               FROM memory_sources ms
               JOIN documents d ON ms.doc_id = d.doc_id
               WHERE ms.memory_id = ?""",
            (memory_id,),
        ) as cursor:
            row = await cursor.fetchone()

        total_count = int(row["total_count"] or 0) if row else 0
        now = _now_iso()
        retired = total_count == 0
        if retired:
            await self.db.execute(
                """UPDATE memories SET
                    status = 'retired', retirement_reason = ?,
                    retired_at = COALESCE(retired_at, ?),
                    corroboration_count = ?, updated_at = ?
                   WHERE id = ?""",
                (retire_reason, now, total_count, now, memory_id),
            )
            await self._stale_pending_reviews_unlocked(
                (memory_id,),
                now=now,
            )
        else:
            await self.db.execute(
                """UPDATE memories SET
                    corroboration_count = ?, updated_at = ?
                   WHERE id = ?""",
                (total_count, now, memory_id),
            )
        await self._rebuild_memory_fts_unlocked(
            memory_id,
            search_visible_statuses=set(allowed_search_statuses()),
        )
        return retired

    async def _refresh_support_after_source_removal_unlocked(
        self,
        memory_ids: list[str],
        *,
        retire_reason: str = "source_deleted",
    ) -> list[str]:
        """Refresh cached support and return memories retired by source loss."""
        if not memory_ids:
            return []

        retired_ids: list[str] = []
        for memory_id in set(memory_ids):
            retired = await self._refresh_memory_support_state_unlocked(
                memory_id,
                retire_reason=retire_reason,
            )
            if retired:
                retired_ids.append(memory_id)
        return retired_ids

    # ==================================================================
    # Memory Entities
    # ==================================================================

    async def link_memory_entity(self, memory_id: str, entity_id: int) -> None:
        async with self._write_lock:
            await self.db.execute(
                "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) VALUES (?, ?)",
                (memory_id, entity_id),
            )
            await self.db.commit()

    async def get_memory_entity_ids(self, memory_id: str) -> list[int]:
        results: list[int] = []
        async with self.db.execute(
            "SELECT entity_id FROM memory_entities WHERE memory_id = ?",
            (memory_id,),
        ) as cursor:
            async for row in cursor:
                results.append(row[0])
        return results

    async def get_memory_entity_names(self, memory_id: str) -> list[str]:
        """Return canonical entity names linked to a memory, in insertion order."""
        results: list[str] = []
        async with self.db.execute(
            """SELECT e.canonical_name
               FROM memory_entities me
               JOIN entities e ON me.entity_id = e.id
               WHERE me.memory_id = ?
               ORDER BY e.id""",
            (memory_id,),
        ) as cursor:
            async for row in cursor:
                results.append(row[0])
        return results

    # ==================================================================
    # Entities
    # ==================================================================

    async def upsert_entity(
        self,
        canonical_name: str,
        display_name: str,
        tags: list[str] | None = None,
    ) -> int:
        """Insert or update an entity. Returns the entity id."""
        tags_json = json.dumps(tags or [])
        # Also write entity_type for backward compat (first tag or 'unknown')
        entity_type = tags[0] if tags else "unknown"
        async with self._write_lock:
            try:
                await self.db.execute(
                    """INSERT INTO entities (canonical_name, entity_type, tags, display_name)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(canonical_name) DO UPDATE SET
                       entity_type=excluded.entity_type,
                       tags=excluded.tags,
                       display_name=excluded.display_name""",
                    (canonical_name, entity_type, tags_json, display_name),
                )
                async with self.db.execute(
                    "SELECT id FROM entities WHERE canonical_name = ?",
                    (canonical_name,),
                ) as cursor:
                    row = await cursor.fetchone()
                    assert row is not None
                    entity_id = int(row[0])
                await self._refresh_entity_alias_search_unlocked(entity_id)
                await self.db.commit()
                return entity_id
            except BaseException:
                rollback_task = asyncio.create_task(self.db.rollback())
                await _drain_task_despite_cancellation(rollback_task)
                raise

    async def _refresh_entity_alias_search_unlocked(self, entity_id: int) -> None:
        await self.db.execute(
            "DELETE FROM entity_alias_search_fts WHERE entity_id = ?",
            (entity_id,),
        )
        async with self.db.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)) as cursor:
            entity_row = await cursor.fetchone()
        if entity_row is None:
            return
        search_text = " ".join(
            part
            for part in (
                entity_row["canonical_name"] or "",
                entity_row["display_name"] or "",
            )
            if part
        )
        await self.db.execute(
            """INSERT INTO entity_alias_search_fts (
                   entity_id,
                   canonical_name,
                   alias_normalized,
                   search_text
               ) VALUES (?, ?, ?, ?)""",
            (
                entity_id,
                entity_row["canonical_name"],
                entity_row["canonical_name"],
                search_text,
            ),
        )
        async with self.db.execute(
            "SELECT alias, alias_normalized FROM entity_aliases WHERE canonical_id = ?",
            (entity_id,),
        ) as cursor:
            async for row in cursor:
                alias_text = " ".join(part for part in (row["alias"] or "", row["alias_normalized"] or "") if part)
                await self.db.execute(
                    """INSERT INTO entity_alias_search_fts (
                           entity_id,
                           canonical_name,
                           alias_normalized,
                           search_text
                       ) VALUES (?, ?, ?, ?)""",
                    (
                        entity_id,
                        entity_row["canonical_name"],
                        row["alias_normalized"],
                        alias_text,
                    ),
                )

    async def get_entity_by_canonical(self, canonical_name: str) -> Entity | None:
        async with self.db.execute("SELECT * FROM entities WHERE canonical_name = ?", (canonical_name,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return _entity_from_row(dict(row))

    async def get_entity_by_alias(self, alias_normalized: str) -> EntityAlias | None:
        async with self.db.execute(
            "SELECT * FROM entity_aliases WHERE alias_normalized = ? LIMIT 1",
            (alias_normalized,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            return EntityAlias(
                alias=d["alias"],
                alias_normalized=d["alias_normalized"],
                canonical_id=d["canonical_id"],
                source=d["source"],
                created_at=_parse_dt(d["created_at"]),
            )

    async def get_entities_by_tag(self, tag: str) -> list[Entity]:
        """Get entities that have a specific tag."""
        results: list[Entity] = []
        # Use LIKE with JSON pattern for simple tag matching
        async with self.db.execute(
            """SELECT * FROM entities WHERE tags LIKE ? ORDER BY display_name""",
            (f'%"{tag}"%',),
        ) as cursor:
            async for row in cursor:
                results.append(_entity_from_row(dict(row)))
        return results

    async def get_all_entities(self) -> list[Entity]:
        results: list[Entity] = []
        async with self.db.execute("SELECT * FROM entities ORDER BY canonical_name") as cursor:
            async for row in cursor:
                results.append(_entity_from_row(dict(row)))
        return results

    async def list_entities(
        self,
        *,
        tag: str | None = None,
        search: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[Entity], int]:
        """List entities for the admin API without exposing the DB connection."""
        query = "SELECT * FROM entities WHERE 1=1"
        params: list[Any] = []
        if tag:
            query += " AND tags LIKE ?"
            params.append(f'%"{tag}"%')
        if search:
            query += " AND (canonical_name LIKE ? OR display_name LIKE ?)"
            like = f"%{search}%"
            params.extend([like, like])

        count_q = query.replace("SELECT *", "SELECT COUNT(*)")
        async with self.db.execute(count_q, params) as cursor:
            total_row = await cursor.fetchone()
            total = total_row[0] if total_row else 0

        query += " ORDER BY display_name LIMIT ? OFFSET ?"
        page_params = [*params, limit, offset]
        entities: list[Entity] = []
        async with self.db.execute(query, page_params) as cursor:
            async for row in cursor:
                entities.append(_entity_from_row(dict(row)))
        return entities, total

    async def get_entity(self, entity_id: int) -> Entity | None:
        async with self.db.execute("SELECT * FROM entities WHERE id = ?", (entity_id,)) as cursor:
            row = await cursor.fetchone()
            return _entity_from_row(dict(row)) if row else None

    async def count_memories_for_entity(self, entity_id: int) -> int:
        async with self.db.execute("SELECT COUNT(*) FROM memory_entities WHERE entity_id = ?", (entity_id,)) as cursor:
            count_row = await cursor.fetchone()
            return count_row[0] if count_row else 0

    async def merge_entities(self, *, source_id: int, target_id: int) -> dict:
        """Merge one entity into another and return source/target names."""
        if source_id == target_id:
            raise ValueError("Source and target entities must differ")

        async with self._write_lock:
            try:
                async with self.db.execute(
                    "SELECT * FROM entities WHERE id = ?",
                    (source_id,),
                ) as cursor:
                    source_row = await cursor.fetchone()
                if source_row is None:
                    raise LookupError("Source entity not found")
                source = _entity_from_row(dict(source_row))
                async with self.db.execute(
                    "SELECT * FROM entities WHERE id = ?",
                    (target_id,),
                ) as cursor:
                    target_row = await cursor.fetchone()
                if target_row is None:
                    raise LookupError("Target entity not found")
                target = _entity_from_row(dict(target_row))

                await self.db.execute(
                    """UPDATE OR IGNORE memory_entities
                       SET entity_id = ?
                       WHERE entity_id = ?""",
                    (target_id, source_id),
                )
                await self.db.execute(
                    "DELETE FROM memory_entities WHERE entity_id = ?",
                    (source_id,),
                )
                await self.db.execute(
                    """UPDATE OR IGNORE entity_aliases
                       SET canonical_id = ?
                       WHERE canonical_id = ?""",
                    (target_id, source_id),
                )
                await self.db.execute(
                    "DELETE FROM entity_aliases WHERE canonical_id = ?",
                    (source_id,),
                )
                await self.db.execute(
                    """INSERT OR IGNORE INTO entity_aliases
                       (alias, alias_normalized, canonical_id, source)
                       VALUES (?, ?, ?, 'admin_manual')""",
                    (
                        source.canonical_name,
                        canonicalize_entity_name(source.canonical_name),
                        target_id,
                    ),
                )
                await self.db.execute("DELETE FROM entities WHERE id = ?", (source_id,))
                await self.db.execute("DELETE FROM entity_alias_search_fts WHERE entity_id = ?", (source_id,))
                await self._refresh_entity_alias_search_unlocked(target_id)
                await self.db.commit()
            except BaseException:
                rollback_task = asyncio.create_task(self.db.rollback())
                await _drain_task_despite_cancellation(rollback_task)
                raise

        return {
            "source_id": source_id,
            "source_name": source.canonical_name,
            "target_id": target_id,
            "target_name": target.canonical_name,
        }

    async def remove_entity_alias(self, *, entity_id: int, alias_normalized: str) -> bool:
        async with self._write_lock:
            try:
                result = await self.db.execute(
                    "DELETE FROM entity_aliases WHERE alias_normalized = ? AND canonical_id = ?",
                    (alias_normalized, entity_id),
                )
                if result.rowcount > 0:
                    await self._refresh_entity_alias_search_unlocked(entity_id)
                await self.db.commit()
                return result.rowcount > 0
            except BaseException:
                rollback_task = asyncio.create_task(self.db.rollback())
                await _drain_task_despite_cancellation(rollback_task)
                raise

    async def insert_alias(
        self,
        alias: str,
        alias_normalized: str,
        canonical_id: int,
        source: str,
    ) -> None:
        async with self._write_lock:
            await self.db.execute(
                """INSERT OR IGNORE INTO entity_aliases (
                    alias, alias_normalized, canonical_id, source
                ) VALUES (?, ?, ?, ?)""",
                (alias, alias_normalized, canonical_id, source),
            )
            await self._refresh_entity_alias_search_unlocked(canonical_id)
            await self.db.commit()

    async def get_aliases_for_entity(self, entity_id: int) -> list[EntityAlias]:
        results: list[EntityAlias] = []
        async with self.db.execute("SELECT * FROM entity_aliases WHERE canonical_id = ?", (entity_id,)) as cursor:
            async for row in cursor:
                d = dict(row)
                results.append(
                    EntityAlias(
                        alias=d["alias"],
                        alias_normalized=d["alias_normalized"],
                        canonical_id=d["canonical_id"],
                        source=d["source"],
                        created_at=_parse_dt(d["created_at"]),
                    )
                )
        return results

    async def get_all_aliases(self) -> list[tuple[str, int]]:
        """Return all (alias_normalized, canonical_id) pairs for entity detection."""
        results: list[tuple[str, int]] = []
        async with self.db.execute("SELECT alias_normalized, canonical_id FROM entity_aliases") as cursor:
            async for row in cursor:
                results.append((row["alias_normalized"], row["canonical_id"]))
        return results

    # ==================================================================
    # Sources
    # ==================================================================

    async def upsert_source(
        self,
        id: str,
        type: str,
        name: str,
        config_json: str,
        access_policy: str,
        owner_user_id: str,
        access_state: str = "active",
        status: str | None = None,
        project_binding: Mapping[str, Any] | None = None,
        created_by_user_id: str | None = None,
        execution_owner_user_id: str | None = None,
        projection_scope_transition: ProjectionScopeTransition | None = None,
    ) -> None:
        """Insert or update a source row.

        `project_binding` is the structured rule the project resolver
        consults when memories are extracted from this source. `None`
        leaves the source unbound and resolves writes to `UNSORTED`.
        """
        from memforge.source_access import memory_visibility_for_source

        memory_visibility_for_source(
            {
                "access_policy": access_policy,
                "owner_user_id": owner_user_id,
                "access_state": access_state,
            }
        )
        binding_json = json.dumps(dict(project_binding)) if project_binding else None
        if projection_scope_transition is not None and projection_scope_transition.source_id != id:
            raise ValueError("projection scope transition source mismatch")
        if (
            projection_scope_transition is not None
            and projection_scope_transition.status is not ProjectionScopeTransitionStatus.PENDING
        ):
            raise ValueError("new projection scope transition must be pending")
        async with self._write_lock:
            try:
                source_lock = await self.db.execute(
                    "UPDATE sources SET status = status WHERE id = ?",
                    (id,),
                )
                if source_lock.rowcount:
                    async with self.db.execute(
                        """SELECT type, name, config, project_binding,
                                  created_by_user_id, owner_user_id, access_policy,
                                  access_state, execution_owner_user_id
                           FROM sources WHERE id = ?""",
                        (id,),
                    ) as cursor:
                        current_source = await cursor.fetchone()
                    async with self.db.execute(
                        """SELECT id FROM lifecycle_backfill_jobs
                           WHERE source_id = ? AND status IN ('queued', 'running')
                           ORDER BY created_at LIMIT 1""",
                        (id,),
                    ) as cursor:
                        lifecycle_job = await cursor.fetchone()
                    if lifecycle_job is not None:
                        raise ValueError(f"source lifecycle maintenance active: {lifecycle_job['id']}")
                    async with self.db.execute(
                        """SELECT run_id FROM source_sync_runs
                           WHERE source_id = ? AND status IN ('pending', 'running')
                           ORDER BY created_at LIMIT 1""",
                        (id,),
                    ) as cursor:
                        active_run = await cursor.fetchone()
                    if active_run is not None:
                        current_creator = current_source["created_by_user_id"]
                        current_execution_owner = current_source["execution_owner_user_id"]
                        pause_only = is_pause_only_source_update(
                            current={
                                "type": current_source["type"],
                                "name": current_source["name"],
                                "config": current_source["config"],
                                "project_binding": current_source["project_binding"],
                                "created_by_user_id": current_creator,
                                "owner_user_id": current_source["owner_user_id"],
                                "access_policy": current_source["access_policy"],
                                "access_state": current_source["access_state"],
                                "execution_owner_user_id": current_execution_owner,
                            },
                            proposed={
                                "type": type,
                                "name": name,
                                "config": config_json,
                                "project_binding": binding_json,
                                "created_by_user_id": current_creator or created_by_user_id,
                                "owner_user_id": owner_user_id,
                                "access_policy": access_policy,
                                "access_state": access_state,
                                "execution_owner_user_id": current_execution_owner or execution_owner_user_id,
                            },
                            requested_status=status,
                        )
                        if not pause_only:
                            raise ValueError(f"source sync run already active: {active_run['run_id']}")
            except Exception:
                await self.db.rollback()
                raise
            try:
                await self.db.execute(
                    """INSERT INTO sources (
                       id, type, name, config, status, project_binding,
                       created_by_user_id, owner_user_id, access_policy,
                       access_state, execution_owner_user_id
                   )
                   VALUES (?, ?, ?, ?, COALESCE(?, 'active'), ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                   type=excluded.type,
                   name=excluded.name,
                   config=excluded.config,
                   status=CASE
                       WHEN ? IS NULL THEN sources.status
                       ELSE excluded.status
                   END,
                   project_binding=excluded.project_binding,
                   created_by_user_id=COALESCE(sources.created_by_user_id, excluded.created_by_user_id),
                   owner_user_id=excluded.owner_user_id,
                   access_policy=excluded.access_policy,
                   access_state=excluded.access_state,
                   execution_owner_user_id=COALESCE(
                       sources.execution_owner_user_id,
                       excluded.execution_owner_user_id
                   )""",
                    (
                        id,
                        type,
                        name,
                        config_json,
                        status,
                        binding_json,
                        created_by_user_id,
                        owner_user_id,
                        access_policy,
                        access_state,
                        execution_owner_user_id,
                        status,
                    ),
                )
                if projection_scope_transition is not None:
                    previous_json = json.dumps(
                        dict(projection_scope_transition.previous_scope),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    target_json = json.dumps(
                        dict(projection_scope_transition.target_scope),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    async with self.db.execute(
                        """SELECT * FROM projection_scope_transitions
                           WHERE source_id = ?
                             AND status IN ('pending', 'running', 'failed')
                           ORDER BY created_at, id LIMIT 1""",
                        (id,),
                    ) as cursor:
                        open_transition = await cursor.fetchone()
                    if open_transition is not None and open_transition["id"] != projection_scope_transition.id:
                        raise ValueError("projection_scope_transition_active")
                    await self.db.execute(
                        """INSERT OR IGNORE INTO projection_scope_transitions (
                            id, source_id, previous_scope_json, target_scope_json, status,
                            run_id, coverage, error, created_at, updated_at, completed_at
                        ) VALUES (?, ?, ?, ?, 'pending', NULL, NULL, NULL, ?, ?, NULL)""",
                        (
                            projection_scope_transition.id,
                            id,
                            previous_json,
                            target_json,
                            projection_scope_transition.created_at or _now_iso(),
                            _now_iso(),
                        ),
                    )
                    async with self.db.execute(
                        "SELECT * FROM projection_scope_transitions WHERE id = ?",
                        (projection_scope_transition.id,),
                    ) as cursor:
                        stored_transition = await cursor.fetchone()
                    if (
                        stored_transition is None
                        or stored_transition["source_id"] != id
                        or stored_transition["previous_scope_json"] != previous_json
                        or stored_transition["target_scope_json"] != target_json
                    ):
                        raise ValueError("projection scope transition retry identity mismatch")
                stale_metadata_rows = await self.db.execute_fetchall(
                    """SELECT 1
                     FROM memory_search_metadata_fts
                    WHERE source_id = ?
                      AND metadata_source_name_tokens IS NOT ?
                    LIMIT 1""",
                    (id, name),
                )
                if stale_metadata_rows:
                    await self._refresh_metadata_fts_for_source_unlocked(id)
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def get_source(self, source_id: str) -> dict | None:
        async with self.db.execute("SELECT * FROM sources WHERE id = ?", (source_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            d["config"] = json.loads(d["config"])
            d["project_binding"] = json.loads(d["project_binding"]) if d.get("project_binding") else None
            d["sync_schedule"] = _source_schedule_from_row(d)
            return d

    async def create_source_access_transition(
        self,
        *,
        operation_id: str,
        source_id: str,
        idempotency_key: str,
        actor_user_id: str,
        target_policy: str,
    ) -> dict[str, Any]:
        """Start one fail-closed, idempotent Source access transition."""
        from memforge.source_access import SourceAccessPolicy, source_access_policy

        target = SourceAccessPolicy(target_policy)
        now = _now_iso()
        async with self._write_lock:
            try:
                source_lock = await self.db.execute(
                    "UPDATE sources SET status = status WHERE id = ?",
                    (source_id,),
                )
                if source_lock.rowcount != 1:
                    raise LookupError("source_not_found")
                async with self.db.execute(
                    """SELECT id FROM lifecycle_backfill_jobs
                       WHERE source_id = ? AND status IN ('queued', 'running')
                       ORDER BY created_at LIMIT 1""",
                    (source_id,),
                ) as cursor:
                    lifecycle_job = await cursor.fetchone()
                if lifecycle_job is not None:
                    raise ValueError(f"source lifecycle maintenance active: {lifecycle_job['id']}")
                async with self.db.execute(
                    """SELECT run_id FROM source_sync_runs
                       WHERE source_id = ? AND status IN ('pending', 'running')
                       ORDER BY created_at LIMIT 1""",
                    (source_id,),
                ) as cursor:
                    active_run = await cursor.fetchone()
                if active_run is not None:
                    raise ValueError(f"source sync run already active: {active_run['run_id']}")
                async with self.db.execute(
                    "SELECT * FROM source_access_transitions WHERE source_id = ? AND idempotency_key = ?",
                    (source_id, idempotency_key),
                ) as cursor:
                    existing = await cursor.fetchone()
                if existing:
                    transition = dict(existing)
                    if transition["target_policy"] != target.value:
                        raise ValueError("idempotency_key_reused_for_different_target")
                    return transition

                async with self.db.execute(
                    "SELECT * FROM sources WHERE id = ?",
                    (source_id,),
                ) as cursor:
                    source_row = await cursor.fetchone()
                if source_row is None:
                    raise LookupError("source_not_found")
                source = dict(source_row)
                previous = source_access_policy(source)
                if previous is target:
                    raise ValueError("source_access_policy_unchanged")
                if source.get("access_state") != "active":
                    raise RuntimeError("source_access_transition_already_active")
                async with self.db.execute(
                    "SELECT operation_id FROM source_access_transitions "
                    "WHERE source_id = ? AND status IN ('queued', 'running', 'failed') LIMIT 1",
                    (source_id,),
                ) as cursor:
                    active = await cursor.fetchone()
                if active:
                    raise RuntimeError("source_access_transition_already_active")

                async with self.db.execute(
                    "SELECT COUNT(DISTINCT memory_id) AS total FROM memory_sources WHERE source_id = ?",
                    (source_id,),
                ) as cursor:
                    count_row = await cursor.fetchone()
                total = int(count_row["total"] if count_row else 0)
                await self.db.execute(
                    """INSERT INTO source_access_transitions (
                           operation_id, source_id, idempotency_key, actor_user_id,
                           previous_policy, target_policy, previous_source_status,
                           status, total_memories, processed_memories,
                           created_at, updated_at
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, 0, ?, ?)""",
                    (
                        operation_id,
                        source_id,
                        idempotency_key,
                        actor_user_id,
                        previous.value,
                        target.value,
                        source["status"],
                        total,
                        now,
                        now,
                    ),
                )
                await self.db.execute(
                    "UPDATE sources SET access_state = 'changing', status = 'paused' WHERE id = ?",
                    (source_id,),
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        transition = await self.get_source_access_transition(operation_id)
        assert transition is not None
        return transition

    async def get_source_access_transition(self, operation_id: str) -> dict[str, Any] | None:
        async with self.db.execute(
            "SELECT * FROM source_access_transitions WHERE operation_id = ?",
            (operation_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_active_source_access_transition(self, source_id: str) -> dict[str, Any] | None:
        async with self.db.execute(
            """SELECT * FROM source_access_transitions
               WHERE source_id = ? AND status IN ('queued', 'running', 'failed')
               ORDER BY created_at DESC LIMIT 1""",
            (source_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return dict(row) if row else None

    async def mark_source_access_transition_running(self, operation_id: str) -> None:
        async with self._write_lock:
            await self.db.execute(
                """UPDATE source_access_transitions
                   SET status = 'running', error_code = NULL, error_message = NULL,
                       updated_at = ?
                   WHERE operation_id = ? AND status IN ('queued', 'failed', 'running')""",
                (_now_iso(), operation_id),
            )
            await self.db.commit()

    async def mark_source_access_transition_failed(
        self,
        operation_id: str,
        *,
        error_code: str,
        error_message: str,
    ) -> None:
        async with self._write_lock:
            await self.db.execute(
                """UPDATE source_access_transitions
                   SET status = 'failed', error_code = ?, error_message = ?, updated_at = ?
                   WHERE operation_id = ? AND status <> 'completed'""",
                (error_code[:100], error_message[:500], _now_iso(), operation_id),
            )
            await self.db.commit()

    async def advance_source_access_transition_progress(self, operation_id: str) -> None:
        """Record one relational/vector access projection as converged."""
        async with self._write_lock:
            await self.db.execute(
                """UPDATE source_access_transitions
                   SET processed_memories = MIN(total_memories, processed_memories + 1),
                       updated_at = ?
                   WHERE operation_id = ? AND status = 'running'""",
                (_now_iso(), operation_id),
            )
            await self.db.commit()

    async def reconcile_source_memory_access(
        self,
        *,
        operation_id: str,
        source_id: str,
        target_policy: str,
        source_owner_user_id: str,
    ) -> list[str]:
        """Atomically move one Source's support into its target access context."""
        from memforge.source_access import SourceAccessPolicy

        target = SourceAccessPolicy(target_policy)
        target_visibility = target.value
        target_owner = source_owner_user_id if target is SourceAccessPolicy.PRIVATE else None
        affected_ids: list[str] = []
        async with self._write_lock:
            try:
                async with self.db.execute(
                    "SELECT previous_policy FROM source_access_transitions WHERE operation_id = ?",
                    (operation_id,),
                ) as cursor:
                    transition_row = await cursor.fetchone()
                if transition_row is None:
                    raise LookupError("source_access_transition_not_found")
                reverting = str(transition_row["previous_policy"]) == target.value
                async with self.db.execute(
                    """SELECT original_memory_id, target_memory_id
                       FROM source_access_transition_memory_map
                       WHERE operation_id = ?""",
                    (operation_id,),
                ) as cursor:
                    split_by_target = {
                        str(row["target_memory_id"]): str(row["original_memory_id"]) async for row in cursor
                    }
                target_by_original = {
                    original_memory_id: target_memory_id
                    for target_memory_id, original_memory_id in split_by_target.items()
                }
                async with self.db.execute(
                    "SELECT DISTINCT memory_id FROM memory_sources WHERE source_id = ? ORDER BY memory_id",
                    (source_id,),
                ) as cursor:
                    memory_ids = [str(row["memory_id"]) async for row in cursor]

                for memory_id in memory_ids:
                    memory = await self.get_memory(memory_id)
                    if memory is None:
                        continue
                    original_memory_id = split_by_target.get(memory_id) if reverting else None
                    if original_memory_id is not None:
                        await self.db.execute(
                            """INSERT OR IGNORE INTO memory_sources (
                                   memory_id, doc_id, source_id, source_type, excerpt,
                                   support_kind, added_at, source_updated_at
                               )
                               SELECT ?, doc_id, source_id, source_type, excerpt,
                                      support_kind, added_at, source_updated_at
                               FROM memory_sources
                               WHERE memory_id = ? AND source_id = ?""",
                            (original_memory_id, memory_id, source_id),
                        )
                        await self.db.execute(
                            "DELETE FROM memory_sources WHERE memory_id = ? AND source_id = ?",
                            (memory_id, source_id),
                        )
                        await self.db.execute(
                            """INSERT OR REPLACE INTO evidence_relations (
                                   evidence_unit_id, memory_id, relation_type, relation_direction,
                                   authority_case, is_authoritative_support,
                                   source_lineage_id, confidence, reason,
                                   proposed_memory_content, excerpt, classifier_version,
                                   relation_run_id, created_at
                               )
                               SELECT er.evidence_unit_id, ?, er.relation_type, er.relation_direction,
                                      er.authority_case, er.is_authoritative_support,
                                      er.source_lineage_id, er.confidence, er.reason,
                                      er.proposed_memory_content, er.excerpt,
                                      er.classifier_version, er.relation_run_id, er.created_at
                               FROM evidence_relations er
                               JOIN evidence_units eu ON eu.id = er.evidence_unit_id
                               WHERE er.memory_id = ? AND eu.source_id = ?""",
                            (original_memory_id, memory_id, source_id),
                        )
                        await self.db.execute(
                            """DELETE FROM evidence_relations
                               WHERE memory_id = ? AND evidence_unit_id IN (
                                   SELECT id FROM evidence_units WHERE source_id = ?
                               )""",
                            (memory_id, source_id),
                        )
                        await self.db.execute(
                            """UPDATE agent_claims SET memory_id = ?
                               WHERE memory_id = ? AND concept_id IN (
                                   SELECT id FROM agent_concepts WHERE source_id = ?
                               )""",
                            (original_memory_id, memory_id, source_id),
                        )
                        await self.db.execute(
                            """UPDATE relation_runs SET result_memory_id = ?
                               WHERE result_memory_id = ? AND evidence_unit_id IN (
                                   SELECT id FROM evidence_units WHERE source_id = ?
                               )""",
                            (original_memory_id, memory_id, source_id),
                        )
                        await self._refresh_memory_support_state_unlocked(
                            original_memory_id,
                            retire_reason="source_access_transition_revert",
                        )
                        await self._refresh_memory_support_state_unlocked(
                            memory_id,
                            retire_reason="source_access_transition_revert",
                        )
                        affected_ids.extend([original_memory_id, memory_id])
                        continue
                    if reverting and memory_id in target_by_original:
                        affected_ids.extend([memory_id, target_by_original[memory_id]])
                        continue
                    if not reverting and memory_id in split_by_target:
                        affected_ids.extend([split_by_target[memory_id], memory_id])
                        continue
                    context_matches = memory.visibility == target_visibility and memory.owner_user_id == target_owner
                    async with self.db.execute(
                        "SELECT COUNT(*) AS total FROM memory_sources WHERE memory_id = ? AND source_id <> ?",
                        (memory_id, source_id),
                    ) as cursor:
                        other_row = await cursor.fetchone()
                    has_other_support = bool(other_row and int(other_row["total"]) > 0)

                    target_memory_id = memory_id
                    if not context_matches and has_other_support:
                        target_memory_id = (
                            "mem-access-"
                            + uuid.uuid5(
                                uuid.NAMESPACE_URL,
                                f"{operation_id}:{source_id}:{memory_id}",
                            ).hex[:24]
                        )
                        if await self.get_memory(target_memory_id) is None:
                            target_memory = replace(
                                memory,
                                id=target_memory_id,
                                visibility=target_visibility,
                                owner_user_id=target_owner,
                                corroboration_count=1,
                                created_at=datetime.now(timezone.utc),
                                updated_at=datetime.now(timezone.utc),
                            )
                            await self._insert_memory_unlocked(target_memory)
                            await self.db.execute(
                                "INSERT OR IGNORE INTO memory_entities (memory_id, entity_id) "
                                "SELECT ?, entity_id FROM memory_entities WHERE memory_id = ?",
                                (target_memory_id, memory_id),
                            )
                        await self.db.execute(
                            """INSERT OR IGNORE INTO source_access_transition_memory_map (
                                   operation_id, source_id, original_memory_id, target_memory_id
                               ) VALUES (?, ?, ?, ?)""",
                            (operation_id, source_id, memory_id, target_memory_id),
                        )
                        await self.db.execute(
                            "UPDATE memory_sources SET memory_id = ? WHERE memory_id = ? AND source_id = ?",
                            (target_memory_id, memory_id, source_id),
                        )
                        await self.db.execute(
                            """INSERT OR REPLACE INTO evidence_relations (
                                   evidence_unit_id, memory_id, relation_type, relation_direction,
                                   authority_case, is_authoritative_support,
                                   source_lineage_id, confidence, reason,
                                   proposed_memory_content, excerpt, classifier_version,
                                   relation_run_id, created_at
                               )
                               SELECT er.evidence_unit_id, ?, er.relation_type, er.relation_direction,
                                      er.authority_case, er.is_authoritative_support,
                                      er.source_lineage_id, er.confidence, er.reason,
                                      er.proposed_memory_content, er.excerpt,
                                      er.classifier_version, er.relation_run_id, er.created_at
                               FROM evidence_relations er
                               JOIN evidence_units eu ON eu.id = er.evidence_unit_id
                               WHERE er.memory_id = ? AND eu.source_id = ?""",
                            (target_memory_id, memory_id, source_id),
                        )
                        await self.db.execute(
                            """DELETE FROM evidence_relations
                               WHERE memory_id = ? AND evidence_unit_id IN (
                                   SELECT id FROM evidence_units WHERE source_id = ?
                               )""",
                            (memory_id, source_id),
                        )
                        await self.db.execute(
                            """UPDATE agent_claims SET memory_id = ?
                               WHERE memory_id = ? AND concept_id IN (
                                   SELECT id FROM agent_concepts WHERE source_id = ?
                               )""",
                            (target_memory_id, memory_id, source_id),
                        )
                        await self.db.execute(
                            """UPDATE relation_runs SET result_memory_id = ?
                               WHERE result_memory_id = ? AND evidence_unit_id IN (
                                   SELECT id FROM evidence_units WHERE source_id = ?
                               )""",
                            (target_memory_id, memory_id, source_id),
                        )
                        await self._refresh_memory_support_state_unlocked(
                            memory_id,
                            retire_reason="source_access_transition",
                        )
                    elif not context_matches:
                        await self.db.execute(
                            """UPDATE memories
                               SET visibility = ?, owner_user_id = ?, updated_at = ?
                               WHERE id = ?""",
                            (target_visibility, target_owner, _now_iso(), memory_id),
                        )

                    await self._refresh_memory_support_state_unlocked(
                        target_memory_id,
                        retire_reason="source_access_transition",
                    )
                    affected_ids.extend([memory_id, target_memory_id])

                await self.db.execute(
                    """UPDATE evidence_units
                       SET visibility = ?, owner_user_id = ?, access_context_hash = NULL,
                           updated_at = ?
                       WHERE source_id = ?""",
                    (target_visibility, target_owner, _now_iso(), source_id),
                )
                await self.db.execute(
                    """UPDATE agent_concepts
                       SET visibility = ?, updated_at = ?
                       WHERE source_id = ?""",
                    (target_visibility, _now_iso(), source_id),
                )
                unique_ids = list(dict.fromkeys(affected_ids))
                for affected_id in unique_ids:
                    await self._rebuild_memory_fts_unlocked(
                        affected_id,
                        search_visible_statuses=set(allowed_search_statuses()),
                    )
                await self.db.execute(
                    """UPDATE source_access_transitions
                       SET total_memories = ?, processed_memories = 0, updated_at = ?
                       WHERE operation_id = ?""",
                    (len(unique_ids), _now_iso(), operation_id),
                )
                await self.db.commit()
                return unique_ids
            except Exception:
                await self.db.rollback()
                raise

    async def complete_source_access_transition(self, operation_id: str) -> None:
        """Activate a fully reconciled transition and restore Source sync state."""
        now = _now_iso()
        async with self._write_lock:
            try:
                async with self.db.execute(
                    "SELECT * FROM source_access_transitions WHERE operation_id = ?",
                    (operation_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise LookupError("source_access_transition_not_found")
                transition = dict(row)
                if transition["status"] == "completed":
                    return
                await self.db.execute(
                    """UPDATE sources
                       SET access_policy = ?, access_state = 'active', status = ?
                       WHERE id = ?""",
                    (
                        transition["target_policy"],
                        transition["previous_source_status"],
                        transition["source_id"],
                    ),
                )
                await self.db.execute(
                    """UPDATE source_access_transitions
                       SET status = 'completed', error_code = NULL, error_message = NULL,
                           updated_at = ?, completed_at = ?
                       WHERE operation_id = ?""",
                    (now, now, operation_id),
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def mark_source_access_transition_reverted(self, operation_id: str) -> None:
        now = _now_iso()
        async with self._write_lock:
            try:
                async with self.db.execute(
                    "SELECT * FROM source_access_transitions WHERE operation_id = ?",
                    (operation_id,),
                ) as cursor:
                    row = await cursor.fetchone()
                if row is None:
                    raise LookupError("source_access_transition_not_found")
                transition = dict(row)
                await self.db.execute(
                    """UPDATE sources
                       SET access_policy = ?, access_state = 'active', status = ?
                       WHERE id = ?""",
                    (
                        transition["previous_policy"],
                        transition["previous_source_status"],
                        transition["source_id"],
                    ),
                )
                await self.db.execute(
                    """UPDATE source_access_transitions
                       SET status = 'reverted', error_code = NULL, error_message = NULL,
                           updated_at = ?, completed_at = ?
                       WHERE operation_id = ?""",
                    (now, now, operation_id),
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def restore_source_snapshot(self, source: dict) -> None:
        """Restore one source row from a captured snapshot."""
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO sources
                   (id, type, name, config, status, last_sync, doc_count, project_binding,
                    created_by_user_id, owner_user_id, access_policy, access_state,
                    execution_owner_user_id, sync_schedule_enabled,
                    sync_schedule_interval_minutes, sync_schedule_next_at,
                    sync_schedule_updated_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                   type=excluded.type,
                   name=excluded.name,
                   config=excluded.config,
                   status=excluded.status,
                   last_sync=excluded.last_sync,
                   doc_count=excluded.doc_count,
                   project_binding=excluded.project_binding,
                   created_by_user_id=excluded.created_by_user_id,
                   owner_user_id=excluded.owner_user_id,
                   access_policy=excluded.access_policy,
                   access_state=excluded.access_state,
                   execution_owner_user_id=excluded.execution_owner_user_id,
                   sync_schedule_enabled=excluded.sync_schedule_enabled,
                   sync_schedule_interval_minutes=excluded.sync_schedule_interval_minutes,
                   sync_schedule_next_at=excluded.sync_schedule_next_at,
                   sync_schedule_updated_at=excluded.sync_schedule_updated_at,
                   created_at=excluded.created_at""",
                (
                    source["id"],
                    source["type"],
                    source["name"],
                    json.dumps(source["config"]),
                    source["status"],
                    source["last_sync"],
                    source["doc_count"],
                    (json.dumps(source["project_binding"]) if source.get("project_binding") else None),
                    source.get("created_by_user_id"),
                    source["owner_user_id"],
                    source["access_policy"],
                    source["access_state"],
                    source.get("execution_owner_user_id"),
                    int((source.get("sync_schedule") or {}).get("enabled") or 0),
                    int((source.get("sync_schedule") or {}).get("interval_minutes") or 1440),
                    (source.get("sync_schedule") or {}).get("next_run_at"),
                    (source.get("sync_schedule") or {}).get("updated_at"),
                    source["created_at"],
                ),
            )
            await self.db.commit()

    async def list_sources(self) -> list[dict]:
        results: list[dict] = []
        async with self.db.execute("SELECT * FROM sources ORDER BY created_at") as cursor:
            async for row in cursor:
                d = dict(row)
                d["config"] = json.loads(d["config"])
                d["project_binding"] = json.loads(d["project_binding"]) if d.get("project_binding") else None
                d["sync_schedule"] = _source_schedule_from_row(d)
                results.append(d)
        return results

    async def list_searchable_source_ids_for_user(
        self,
        source_ids: list[str],
        user_id: str,
    ) -> set[str]:
        if not source_ids:
            return set()
        ordered_unique = tuple(dict.fromkeys(source_ids))
        placeholders = ", ".join("?" for _ in ordered_unique)
        async with self.db.execute(
            f"""SELECT s.id
               FROM sources s
               LEFT JOIN source_subscriptions ss
                 ON ss.source_id = s.id AND ss.user_id = ?
               WHERE s.id IN ({placeholders})
                 AND s.status = 'active'
                 AND COALESCE(ss.enabled, 1) = 1""",
            (user_id, *ordered_unique),
        ) as cursor:
            return {str(row["id"]) async for row in cursor}

    async def set_source_sync_schedule(
        self,
        source_id: str,
        *,
        enabled: bool,
        interval_minutes: int,
        next_run_at: datetime | None = None,
    ) -> None:
        if interval_minutes < SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES:
            raise ValueError(
                f"source sync schedule interval must be at least {SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES} minutes"
            )
        if interval_minutes > SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES:
            raise ValueError(
                f"source sync schedule interval must be at most {SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES} minutes"
            )
        async with self._write_lock:
            async with self.db.execute(
                """SELECT sync_schedule_enabled, sync_schedule_interval_minutes,
                          sync_schedule_next_at
                   FROM sources WHERE id = ?""",
                (source_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                raise ValueError(f"Source not found: {source_id}")
            existing = dict(row)
            existing_enabled = bool(existing.get("sync_schedule_enabled"))
            existing_interval = int(
                existing.get("sync_schedule_interval_minutes") or SOURCE_SYNC_SCHEDULE_DEFAULT_INTERVAL_MINUTES
            )
            existing_next_at = existing.get("sync_schedule_next_at")
            if not enabled:
                stored_next_at = None
            elif next_run_at is not None:
                stored_next_at = next_run_at.isoformat()
            elif existing_enabled and existing_interval == interval_minutes and existing_next_at:
                stored_next_at = existing_next_at
            else:
                stored_next_at = (datetime.now(timezone.utc) + timedelta(minutes=interval_minutes)).isoformat()
            await self.db.execute(
                """UPDATE sources SET
                   sync_schedule_enabled = ?,
                   sync_schedule_interval_minutes = ?,
                   sync_schedule_next_at = ?,
                   sync_schedule_updated_at = ?
                   WHERE id = ?""",
                (
                    int(enabled),
                    interval_minutes,
                    stored_next_at,
                    _now_iso(),
                    source_id,
                ),
            )
            await self.db.commit()

    async def claim_due_scheduled_sources(
        self,
        *,
        now: datetime | None = None,
        limit: int = 50,
        exclude_source_ids: set[str] | None = None,
    ) -> list[dict]:
        claim_time = now or datetime.now(timezone.utc)
        due_at = claim_time.isoformat()
        exclude_ids = tuple(sorted(exclude_source_ids or ()))
        exclude_sql = ""
        if exclude_ids:
            exclude_sql = " AND id NOT IN (" + ", ".join("?" for _ in exclude_ids) + ")"
        results: list[dict] = []
        async with self._write_lock:
            async with self.db.execute(
                f"""SELECT * FROM sources
                   WHERE status = 'active'
                     AND sync_schedule_enabled = 1
                     AND sync_schedule_next_at IS NOT NULL
                     AND sync_schedule_next_at <= ?
                     {exclude_sql}
                   ORDER BY sync_schedule_next_at, created_at
                   LIMIT ?""",
                (due_at, *exclude_ids, limit),
            ) as cursor:
                rows = [dict(row) async for row in cursor]
            for d in rows:
                interval_minutes = int(
                    d.get("sync_schedule_interval_minutes") or SOURCE_SYNC_SCHEDULE_DEFAULT_INTERVAL_MINUTES
                )
                next_at = claim_time + timedelta(minutes=interval_minutes)
                updated_at = _now_iso()
                update_cursor = await self.db.execute(
                    """UPDATE sources SET
                       sync_schedule_next_at = ?,
                       sync_schedule_updated_at = ?
                       WHERE id = ?
                         AND status = 'active'
                         AND sync_schedule_enabled = 1
                         AND sync_schedule_next_at = ?""",
                    (
                        next_at.isoformat(),
                        updated_at,
                        d["id"],
                        d["sync_schedule_next_at"],
                    ),
                )
                if update_cursor.rowcount:
                    d["sync_schedule_next_at"] = next_at.isoformat()
                    d["sync_schedule_updated_at"] = updated_at
                    d["config"] = json.loads(d["config"])
                    d["project_binding"] = json.loads(d["project_binding"]) if d.get("project_binding") else None
                    d["sync_schedule"] = _source_schedule_from_row(d)
                    results.append(d)
            await self.db.commit()
        return results

    async def enqueue_due_source_sync_runs(
        self,
        *,
        now: datetime | None = None,
        limit: int = 50,
        workspace_id: str = "default",
        exclude_source_ids: set[str] | None = None,
    ) -> list[SourceSyncRun]:
        claim_time = now or datetime.now(timezone.utc)
        due_at = claim_time.isoformat()
        exclude_ids = tuple(sorted(exclude_source_ids or ()))
        exclude_sql = ""
        if exclude_ids:
            exclude_sql = " AND id NOT IN (" + ", ".join("?" for _ in exclude_ids) + ")"
        runs: list[SourceSyncRun] = []
        async with self._write_lock:
            try:
                async with self.db.execute(
                    f"""SELECT * FROM sources
                       WHERE status = 'active'
                         AND sync_schedule_enabled = 1
                         AND sync_schedule_next_at IS NOT NULL
                         AND sync_schedule_next_at <= ?
                         {exclude_sql}
                       ORDER BY sync_schedule_next_at, created_at
                       LIMIT ?""",
                    (due_at, *exclude_ids, limit),
                ) as cursor:
                    rows = [dict(row) async for row in cursor]
                for row in rows:
                    if local_agent_sync_operation(row["type"], row["config"]) is not None:
                        continue
                    interval_minutes = int(
                        row.get("sync_schedule_interval_minutes") or SOURCE_SYNC_SCHEDULE_DEFAULT_INTERVAL_MINUTES
                    )
                    next_at = claim_time + timedelta(minutes=interval_minutes)
                    updated_at = _now_iso()
                    update_cursor = await self.db.execute(
                        """UPDATE sources SET
                           sync_schedule_next_at = ?,
                           sync_schedule_updated_at = ?
                           WHERE id = ?
                             AND status = 'active'
                             AND sync_schedule_enabled = 1
                             AND sync_schedule_next_at = ?""",
                        (
                            next_at.isoformat(),
                            updated_at,
                            row["id"],
                            row["sync_schedule_next_at"],
                        ),
                    )
                    if not update_cursor.rowcount:
                        continue
                    runs.append(
                        await self._enqueue_source_sync_run_locked(
                            source_id=str(row["id"]),
                            workspace_id=workspace_id,
                            trigger="schedule",
                            now=updated_at,
                        )
                    )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        return runs

    async def advance_source_sync_schedule(
        self,
        source_id: str,
        *,
        expected_next_run_at: str,
        now: datetime | None = None,
    ) -> bool:
        claim_time = now or datetime.now(timezone.utc)
        async with self._write_lock:
            async with self.db.execute(
                "SELECT sync_schedule_interval_minutes FROM sources WHERE id = ?",
                (source_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return False
            interval_minutes = int(
                row["sync_schedule_interval_minutes"] or SOURCE_SYNC_SCHEDULE_DEFAULT_INTERVAL_MINUTES
            )
            next_at = claim_time + timedelta(minutes=interval_minutes)
            updated_at = _now_iso()
            cursor = await self.db.execute(
                """UPDATE sources SET
                   sync_schedule_next_at = ?,
                   sync_schedule_updated_at = ?
                   WHERE id = ?
                     AND status = 'active'
                     AND sync_schedule_enabled = 1
                     AND sync_schedule_next_at = ?""",
                (
                    next_at.isoformat(),
                    updated_at,
                    source_id,
                    expected_next_run_at,
                ),
            )
            await self.db.commit()
            return bool(cursor.rowcount)

    async def is_source_enabled_for_user(self, source_id: str, user_id: str) -> bool:
        async with self.db.execute(
            "SELECT enabled FROM source_subscriptions WHERE source_id = ? AND user_id = ?",
            (source_id, user_id),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return True
            return bool(row["enabled"])

    async def set_source_subscription(self, source_id: str, user_id: str, enabled: bool) -> None:
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO source_subscriptions
                   (source_id, user_id, enabled, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(source_id, user_id) DO UPDATE SET
                   enabled=excluded.enabled,
                   updated_at=excluded.updated_at""",
                (source_id, user_id, int(enabled), _now_iso()),
            )
            await self.db.commit()

    async def is_source_pinned_for_user(self, source_id: str, user_id: str) -> bool:
        async with self.db.execute(
            "SELECT 1 FROM source_list_pins WHERE source_id = ? AND user_id = ?",
            (source_id, user_id),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def set_source_pinned_for_user(self, source_id: str, user_id: str, pinned: bool) -> None:
        async with self._write_lock:
            if pinned:
                await self.db.execute(
                    """INSERT INTO source_list_pins (source_id, user_id, pinned_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(source_id, user_id) DO NOTHING""",
                    (source_id, user_id, _now_iso()),
                )
            else:
                await self.db.execute(
                    "DELETE FROM source_list_pins WHERE source_id = ? AND user_id = ?",
                    (source_id, user_id),
                )
            await self.db.commit()

    async def get_source_list_sort_mode(self, user_id: str) -> SourceListSortMode:
        async with self.db.execute(
            "SELECT sort_mode FROM source_list_preferences WHERE user_id = ?",
            (user_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return SOURCE_LIST_DEFAULT_SORT_MODE
        return validate_source_list_sort_mode(str(row["sort_mode"]))

    async def set_source_list_sort_mode(self, user_id: str, sort_mode: SourceListSortMode) -> None:
        normalized = validate_source_list_sort_mode(sort_mode)
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO source_list_preferences (user_id, sort_mode, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET
                   sort_mode=excluded.sort_mode,
                   updated_at=excluded.updated_at""",
                (user_id, normalized, _now_iso()),
            )
            await self.db.commit()

    async def list_disabled_source_ids_for_user(self, user_id: str) -> list[str]:
        results: list[str] = []
        async with self.db.execute(
            "SELECT source_id FROM source_subscriptions WHERE user_id = ? AND enabled = 0",
            (user_id,),
        ) as cursor:
            async for row in cursor:
                results.append(str(row["source_id"]))
        return results

    async def count_source_memories(
        self,
        source_id: str,
        *,
        include_private: bool = False,
        owner_user_id: str | None = None,
    ) -> int:
        visible_statuses = allowed_search_statuses(False)
        if not visible_statuses:
            return 0
        status_placeholders = ", ".join("?" for _ in visible_statuses)
        visibility_sql = "m.visibility <> ?"
        params: list[Any] = [source_id, *visible_statuses, Visibility.PRIVATE.value]
        if include_private and owner_user_id:
            visibility_sql = "(m.visibility <> ? OR m.owner_user_id = ?)"
            params = [source_id, *visible_statuses, Visibility.PRIVATE.value, owner_user_id]
        async with self.db.execute(
            f"""
            SELECT COUNT(DISTINCT ms.memory_id)
            FROM memory_sources ms
            JOIN memories m ON m.id = ms.memory_id
            WHERE ms.source_id = ?
              AND m.status IN ({status_placeholders})
              AND {visibility_sql}
            """,
            params,
        ) as cursor:
            row = await cursor.fetchone()
            return int(row[0]) if row else 0

    async def list_source_projects(
        self,
        source_id: str,
        *,
        include_private: bool = False,
        owner_user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        visibility_sql = "m.id IS NULL OR m.visibility <> ?"
        params: list[Any] = [source_id, Visibility.PRIVATE.value]
        if include_private and owner_user_id:
            visibility_sql = "m.id IS NULL OR m.visibility <> ? OR m.owner_user_id = ?"
            params = [source_id, Visibility.PRIVATE.value, owner_user_id]
        projects: list[dict[str, Any]] = []
        async with self.db.execute(
            f"""
            SELECT
                COALESCE(NULLIF(TRIM(d.space_or_project), ''), 'Unspecified') AS project,
                COUNT(DISTINCT d.doc_id) AS document_count,
                COUNT(DISTINCT ms.memory_id) AS memory_count,
                MAX(d.last_modified) AS last_observed_at
            FROM documents d
            LEFT JOIN memory_sources ms ON ms.doc_id = d.doc_id
            LEFT JOIN memories m ON m.id = ms.memory_id
            WHERE d.source = ?
              AND ({visibility_sql})
            GROUP BY COALESCE(NULLIF(TRIM(d.space_or_project), ''), 'Unspecified')
            ORDER BY last_observed_at DESC, project ASC
            """,
            params,
        ) as cursor:
            async for row in cursor:
                projects.append(
                    {
                        "project": str(row["project"]),
                        "document_count": int(row["document_count"]),
                        "memory_count": int(row["memory_count"]),
                        "last_observed_at": row["last_observed_at"],
                    }
                )
        return projects

    async def list_resolved_projects_for_source(
        self,
        source_id: str,
        *,
        include_private: bool = False,
        owner_user_id: str | None = None,
    ) -> list[tuple[str, int]]:
        """Group memories from a source by their resolved `project_key`.

        Distinct from `list_source_projects`, which reports the raw
        `documents.space_or_project` field as observed at sync time. This
        view follows provenance through `memory_sources` and reads the
        resolver's verdict on each memory, so the admin can see where
        writes actually landed under the active `project_binding`.
        """
        visibility_sql = "m.visibility <> ?"
        params: list[Any] = [source_id, Visibility.PRIVATE.value]
        if include_private and owner_user_id:
            visibility_sql = "(m.visibility <> ? OR m.owner_user_id = ?)"
            params = [source_id, Visibility.PRIVATE.value, owner_user_id]
        rows: list[tuple[str, int]] = []
        async with self.db.execute(
            f"""
            SELECT m.project_key AS project_key,
                   COUNT(DISTINCT m.id) AS memory_count
            FROM memories m
            JOIN memory_sources ms ON ms.memory_id = m.id
            JOIN documents d ON d.doc_id = ms.doc_id
            WHERE d.source = ?
              AND {visibility_sql}
            GROUP BY m.project_key
            ORDER BY memory_count DESC, project_key ASC
            """,
            params,
        ) as cursor:
            async for row in cursor:
                key = row["project_key"] or UNSORTED_PROJECT_KEY
                rows.append((str(key), int(row["memory_count"])))
        return rows

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    async def create_project(self, *, key: str, name: str, is_shared: bool = False) -> Project:
        """Insert a project row, raising ValueError if `key` already exists."""
        proj_id = f"proj-{uuid.uuid4().hex[:12]}"
        try:
            async with self._write_lock:
                await self.db.execute(
                    "INSERT INTO projects (id, key, name, is_shared) VALUES (?, ?, ?, ?)",
                    (proj_id, key, name, 1 if is_shared else 0),
                )
                await self.db.commit()
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"project key {key!r} already exists") from exc
        created = await self.get_project(proj_id)
        if created is None:
            raise RuntimeError(f"project {proj_id!r} disappeared after insert")
        return created

    async def get_project(self, project_id: str) -> Project | None:
        async with self.db.execute(
            "SELECT id, key, name, is_shared, created_at FROM projects WHERE id = ?",
            (project_id,),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return Project(
            id=row["id"],
            key=row["key"],
            name=row["name"],
            is_shared=bool(row["is_shared"]),
            created_at=_parse_dt(row["created_at"]),
        )

    async def list_projects(self) -> list[Project]:
        out: list[Project] = []
        async with self.db.execute("SELECT id, key, name, is_shared, created_at FROM projects ORDER BY key") as cur:
            async for row in cur:
                out.append(
                    Project(
                        id=row["id"],
                        key=row["key"],
                        name=row["name"],
                        is_shared=bool(row["is_shared"]),
                        created_at=_parse_dt(row["created_at"]),
                    )
                )
        return out

    async def update_project(
        self,
        project_id: str,
        *,
        name: str | None = None,
        is_shared: bool | None = None,
    ) -> Project | None:
        fields: list[str] = []
        params: list[Any] = []
        if name is not None:
            fields.append("name = ?")
            params.append(name)
        if is_shared is not None:
            fields.append("is_shared = ?")
            params.append(1 if is_shared else 0)
        if not fields:
            return await self.get_project(project_id)
        params.append(project_id)
        async with self._write_lock:
            await self.db.execute(
                f"UPDATE projects SET {', '.join(fields)} WHERE id = ?",
                params,
            )
            await self.db.commit()
        return await self.get_project(project_id)

    async def list_project_memory_ids(self, project_id: str) -> list[str]:
        """Return memory ids attached to a project, validating that the
        project is real and not a reserved bucket.

        Pairs with `commit_project_deletion`: the caller (the project
        delete handler) reads the affected ids first, hands them to the
        owning vector service so embedding metadata moves to UNSORTED,
        then asks the database to commit the relational rebucket and
        drop the project row. Reserved keys (SHARED, UNSORTED) raise
        `ValueError`; an unknown id raises `LookupError`.
        """
        target = await self.get_project(project_id)
        if target is None:
            raise LookupError(f"project {project_id!r} not found")
        if target.key in (SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY):
            raise ValueError(f"project {target.key!r} is reserved and cannot be deleted")
        affected_ids: list[str] = []
        async with self.db.execute("SELECT id FROM memories WHERE project_key = ?", (target.key,)) as cur:
            async for row in cur:
                affected_ids.append(row["id"])
        return affected_ids

    async def commit_project_deletion(self, project_id: str, affected_ids: Sequence[str]) -> None:
        """Rebucket the named memories to UNSORTED and drop the project
        row, in one transaction.

        `affected_ids` is the snapshot the caller already moved on the
        vector side. Rebucketing by id rather than by `project_key`
        means a memory inserted under this project after the snapshot
        was taken stays untouched here, so the relational and vector
        channels never disagree about which rows this delete moved.

        Reserved keys (SHARED, UNSORTED) raise `ValueError`. Calling
        with an empty `affected_ids` list still drops the project row
        so a project that owned no memories deletes cleanly.
        """
        target = await self.get_project(project_id)
        if target is None:
            return
        if target.key in (SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY):
            raise ValueError(f"project {target.key!r} is reserved and cannot be deleted")
        async with self._write_lock:
            if affected_ids:
                placeholders = ",".join("?" for _ in affected_ids)
                await self.db.execute(
                    f"UPDATE memories SET project_key = ? WHERE id IN ({placeholders}) AND project_key = ?",
                    (UNSORTED_PROJECT_KEY, *affected_ids, target.key),
                )
            await self.db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
            await self.db.commit()

    async def delete_source_cascade(self, source_id: str) -> SourceDeletionResult:
        """Delete a source and cascade to all documents + memories linked to those docs.

        Returns memory IDs retired because the source removal left them without
        valid support.
        """
        async with self._write_lock:
            try:
                source_lock = await self.db.execute(
                    "UPDATE sources SET status = status WHERE id = ?",
                    (source_id,),
                )
                if source_lock.rowcount:
                    async with self.db.execute(
                        """SELECT id FROM lifecycle_backfill_jobs
                           WHERE source_id = ? AND status IN ('queued', 'running')
                           ORDER BY created_at LIMIT 1""",
                        (source_id,),
                    ) as cursor:
                        lifecycle_job = await cursor.fetchone()
                    if lifecycle_job is not None:
                        raise ValueError(f"source lifecycle maintenance active: {lifecycle_job['id']}")
                    async with self.db.execute(
                        """SELECT run_id FROM source_sync_runs
                           WHERE source_id = ? AND status IN ('pending', 'running')
                           ORDER BY created_at LIMIT 1""",
                        (source_id,),
                    ) as cursor:
                        active_run = await cursor.fetchone()
                    if active_run is not None:
                        raise ValueError(f"source sync run already active: {active_run['run_id']}")
                await self.db.execute(
                    "UPDATE sources SET status = 'deleting' WHERE id = ?",
                    (source_id,),
                )
                retired_ids: list[str] = []
                doc_ids: list[str] = []
                artifact_uris: list[str] = []
                async with self.db.execute(
                    "SELECT doc_id, raw_content_uri, normalized_content_uri, pdf_content_uri "
                    "FROM documents WHERE source = ?",
                    (source_id,),
                ) as cursor:
                    async for row in cursor:
                        doc_ids.append(row["doc_id"])
                        artifact_uris.extend(
                            str(uri)
                            for uri in (
                                row["raw_content_uri"],
                                row["normalized_content_uri"],
                                row["pdf_content_uri"],
                            )
                            if uri
                        )
                async with self.db.execute(
                    "SELECT raw_uri FROM source_sync_inputs WHERE source_id = ?",
                    (source_id,),
                ) as cursor:
                    async for row in cursor:
                        if row["raw_uri"]:
                            artifact_uris.append(str(row["raw_uri"]))

                for artifact_uri in dict.fromkeys(artifact_uris):
                    task_id = source_artifact_cleanup_task_id(source_id, artifact_uri)
                    await self.db.execute(
                        "INSERT OR IGNORE INTO source_artifact_cleanup_tasks "
                        "(task_id, source_id, artifact_uri) VALUES (?, ?, ?)",
                        (task_id, source_id, artifact_uri),
                    )

                # Agent Knowledge is a source-owned projection. These tables
                # intentionally predate source foreign keys, so remove the
                # claim view explicitly before deleting the source row.
                await self.db.execute(
                    """DELETE FROM agent_claim_citations
                       WHERE claim_id IN (
                           SELECT ac.id FROM agent_claims ac
                           JOIN agent_concepts c ON c.id = ac.concept_id
                           WHERE c.source_id = ?
                       )""",
                    (source_id,),
                )
                await self.db.execute(
                    """DELETE FROM agent_claims
                       WHERE concept_id IN (
                           SELECT id FROM agent_concepts WHERE source_id = ?
                       )""",
                    (source_id,),
                )
                await self.db.execute(
                    "DELETE FROM agent_concepts WHERE source_id = ?",
                    (source_id,),
                )

                await self._delete_evidence_graph_for_source_id_unlocked(source_id)
                for doc_id in doc_ids:
                    memory_ids: list[str] = []
                    async with self.db.execute(
                        "SELECT memory_id FROM memory_sources WHERE doc_id = ?",
                        (doc_id,),
                    ) as cursor:
                        async for row in cursor:
                            memory_ids.append(row[0])

                    await self.db.execute("DELETE FROM memory_search_metadata_fts WHERE doc_id = ?", (doc_id,))
                    await self.db.execute("DELETE FROM memory_search_metadata_alias_fts WHERE doc_id = ?", (doc_id,))
                    await self.db.execute("DELETE FROM memory_search_metadata_trigram WHERE doc_id = ?", (doc_id,))
                    await self.db.execute("DELETE FROM memory_sources WHERE doc_id = ?", (doc_id,))
                    await self._delete_evidence_graph_for_doc_ids_unlocked([doc_id])

                    retired_ids.extend(await self._refresh_support_after_source_removal_unlocked(memory_ids))

                    await self.db.execute("DELETE FROM document_metadata WHERE doc_id = ?", (doc_id,))
                    await self.db.execute(
                        "DELETE FROM document_relationships WHERE source_doc_id = ? OR target_doc_id = ?",
                        (doc_id, doc_id),
                    )
                    await self.db.execute("DELETE FROM changelog WHERE doc_id = ?", (doc_id,))
                    await self.db.execute("DELETE FROM agent_session_receipts WHERE doc_id = ?", (doc_id,))
                    await self.db.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))

                await self.db.execute("DELETE FROM agent_session_receipts WHERE source_id = ?", (source_id,))
                await self.db.execute("DELETE FROM sync_state WHERE source = ?", (source_id,))
                await self.db.execute("DELETE FROM sync_history WHERE source = ?", (source_id,))
                await self.db.execute("DELETE FROM source_list_pins WHERE source_id = ?", (source_id,))
                now = _now_iso()
                for memory_id in dict.fromkeys(retired_ids):
                    await self.db.execute(
                        """INSERT INTO source_deletion_vector_outbox (
                               id, source_id, memory_id, status, created_at, updated_at
                           ) VALUES (?, ?, ?, 'pending', ?, ?)""",
                        (
                            f"source-delete-vector-{uuid.uuid4().hex}",
                            source_id,
                            memory_id,
                            now,
                            now,
                        ),
                    )
                await self.db.execute("DELETE FROM sources WHERE id = ?", (source_id,))
                await self.db.commit()
                return SourceDeletionResult(
                    retired_memory_ids=tuple(dict.fromkeys(retired_ids)),
                    retired_search_cleanup_required=True,
                )
            except Exception:
                await self.db.rollback()
                raise

    async def rebaseline_source_lifecycle(
        self,
        source_id: str,
        *,
        source_activity: SourceActivityLease | None = None,
    ) -> SourceLifecycleResetResult:
        """Reset derived source lifecycle while preserving source identity and content.

        This is an explicit recovery operation for replayable sources. It keeps
        configuration, access, subscriptions, documents, artifacts, and finding
        history; removes derived Memory support and Source Projection state;
        retires memories left unsupported; and closes the destructive gate until
        a complete replay succeeds.
        """

        async with self._write_lock:
            try:
                await self._assert_source_activity_fence_unlocked(
                    source_id,
                    source_activity,
                )
                async with self.db.execute(
                    "SELECT type FROM sources WHERE id = ?",
                    (source_id,),
                ) as cursor:
                    source = await cursor.fetchone()
                if source is None:
                    raise LookupError(f"source {source_id!r} not found")
                if str(source["type"]) == "agent_session":
                    raise ValueError("agent_session requires managed claim lineage repair, not source rebaseline")
                doc_ids: list[str] = []
                async with self.db.execute(
                    "SELECT doc_id FROM documents WHERE source = ?",
                    (source_id,),
                ) as cursor:
                    doc_ids = [str(row[0]) async for row in cursor]
                memory_ids: set[str] = set()
                async with self.db.execute(
                    """SELECT DISTINCT ms.memory_id
                         FROM memory_sources ms
                         LEFT JOIN documents d ON d.doc_id = ms.doc_id
                        WHERE ms.source_id = ? OR d.source = ?""",
                    (source_id, source_id),
                ) as cursor:
                    async for row in cursor:
                        memory_ids.add(str(row[0]))
                async with self.db.execute(
                    "SELECT DISTINCT memory_id FROM memory_support_assertions WHERE source_id = ?",
                    (source_id,),
                ) as cursor:
                    async for row in cursor:
                        memory_ids.add(str(row[0]))

                now = _now_iso()
                async with self.db.execute(
                    """SELECT id, mapping_attempt_json
                       FROM lifecycle_cutover_findings
                       WHERE source_id = ? AND status = 'open'""",
                    (source_id,),
                ) as cursor:
                    open_findings = [dict(row) async for row in cursor]
                for finding in open_findings:
                    attempt = json.loads(finding["mapping_attempt_json"] or "{}")
                    attempt["resolution"] = "source_rebaseline"
                    attempt["resolved_at"] = now
                    await self.db.execute(
                        """UPDATE lifecycle_cutover_findings
                           SET status = 'resolved', mapping_attempt_json = ?,
                               updated_at = ?, resolved_at = ?
                           WHERE id = ?""",
                        (json.dumps(attempt, sort_keys=True), now, now, finding["id"]),
                    )

                # Preserve prior source-rebaseline cleanup plans and their
                # outbox tasks.  Chroma is external to SQLite, so a failed
                # vector delete must remain retryable even when an operator
                # starts the rebaseline again.
                await self.db.execute(
                    """DELETE FROM lifecycle_vector_outbox
                       WHERE lifecycle_plan_id IN (
                           SELECT id FROM lifecycle_plans
                           WHERE source_id = ?
                             AND reconciliation_scope_id <> 'source_rebaseline_cleanup'
                       )""",
                    (source_id,),
                )
                await self.db.execute(
                    """DELETE FROM lifecycle_reviews
                       WHERE lifecycle_plan_id IN (
                           SELECT id FROM lifecycle_plans WHERE source_id = ?
                       )""",
                    (source_id,),
                )
                await self.db.execute(
                    """DELETE FROM lifecycle_plans
                       WHERE source_id = ?
                         AND reconciliation_scope_id <> 'source_rebaseline_cleanup'""",
                    (source_id,),
                )
                await self.db.execute(
                    "DELETE FROM memory_support_assertions WHERE source_id = ?",
                    (source_id,),
                )
                await self.db.execute(
                    """DELETE FROM evidence_references
                       WHERE evidence_unit_id IN (
                           SELECT id FROM evidence_units WHERE source_id = ?
                       )""",
                    (source_id,),
                )
                await self._delete_evidence_graph_for_source_id_unlocked(source_id)
                await self.db.execute(
                    """DELETE FROM source_projection_relations
                       WHERE projection_run_id IN (
                           SELECT id FROM source_projection_runs WHERE source_id = ?
                       )""",
                    (source_id,),
                )
                await self.db.execute(
                    """DELETE FROM source_revision_deltas
                       WHERE projection_run_id IN (
                           SELECT id FROM source_projection_runs WHERE source_id = ?
                       )""",
                    (source_id,),
                )
                await self.db.execute("DELETE FROM source_projection_runs WHERE source_id = ?", (source_id,))
                await self.db.execute(
                    """DELETE FROM source_observation_revisions
                       WHERE observation_id IN (
                           SELECT id FROM source_observations WHERE source_id = ?
                       )""",
                    (source_id,),
                )
                await self.db.execute(
                    """DELETE FROM source_unit_revisions
                       WHERE source_unit_id IN (
                           SELECT id FROM source_units WHERE source_id = ?
                       )""",
                    (source_id,),
                )
                await self.db.execute("DELETE FROM source_observations WHERE source_id = ?", (source_id,))
                await self.db.execute(
                    "DELETE FROM source_unit_document_lineage_history WHERE source_id = ?",
                    (source_id,),
                )
                await self.db.execute("DELETE FROM source_units WHERE source_id = ?", (source_id,))
                await self.db.execute("DELETE FROM projection_scope_transitions WHERE source_id = ?", (source_id,))
                await self.db.execute(
                    """DELETE FROM memory_sources
                        WHERE source_id = ?
                           OR doc_id IN (SELECT doc_id FROM documents WHERE source = ?)""",
                    (source_id, source_id),
                )
                for doc_id in doc_ids:
                    await self.db.execute("DELETE FROM memory_search_metadata_fts WHERE doc_id = ?", (doc_id,))
                    await self.db.execute("DELETE FROM memory_search_metadata_alias_fts WHERE doc_id = ?", (doc_id,))
                    await self.db.execute("DELETE FROM memory_search_metadata_trigram WHERE doc_id = ?", (doc_id,))

                retired_ids = await self._refresh_support_after_source_removal_unlocked(
                    list(memory_ids),
                    retire_reason="source_rebaseline",
                )
                if retired_ids:
                    cleanup_plan_id = f"source-rebaseline-cleanup-{uuid.uuid4().hex}"
                    cleanup_payload = {
                        "operation": "source_rebaseline_vector_cleanup",
                        "source_id": source_id,
                        "retired_memory_ids": sorted(retired_ids),
                    }
                    cleanup_payload_json = json.dumps(cleanup_payload, sort_keys=True)
                    await self.db.execute(
                        """INSERT INTO lifecycle_plans (
                               id, reconciliation_scope_id, source_id, source_unit_id,
                               target_unit_revision_id, status, payload_json, payload_hash,
                               created_at, applied_at, error
                           ) VALUES (?, 'source_rebaseline_cleanup', ?, ?, NULL,
                                     'applied', ?, ?, ?, ?, NULL)""",
                        (
                            cleanup_plan_id,
                            source_id,
                            f"source-rebaseline:{source_id}",
                            cleanup_payload_json,
                            hashlib.sha256(cleanup_payload_json.encode("utf-8")).hexdigest(),
                            now,
                            now,
                        ),
                    )
                    for memory_id in retired_ids:
                        await self._enqueue_lifecycle_vector_task_unlocked(
                            cleanup_plan_id,
                            memory_id,
                            LifecycleVectorOperation.DELETE,
                            now=now,
                        )
                await self.db.execute("DELETE FROM sync_state WHERE source = ?", (source_id,))
                await self.db.execute(
                    """INSERT INTO source_lifecycle_gates (
                           source_id, state, reason, audited_at, enabled_at, updated_at
                       ) VALUES (?, 'gated', ?, ?, NULL, ?)
                       ON CONFLICT(source_id) DO UPDATE SET
                           state='gated', reason=excluded.reason,
                           audited_at=excluded.audited_at, updated_at=excluded.updated_at""",
                    (
                        source_id,
                        "source rebaseline requires a complete successful replay",
                        now,
                        now,
                    ),
                )
                await self.db.execute(
                    "UPDATE sources SET last_sync = NULL WHERE id = ?",
                    (source_id,),
                )
                await self._assert_source_activity_fence_unlocked(
                    source_id,
                    source_activity,
                )
                await self.db.commit()
                return SourceLifecycleResetResult(
                    retired_memory_ids=tuple(sorted(retired_ids)),
                    retired_search_cleanup_required=True,
                )
            except Exception:
                await self.db.rollback()
                raise

    async def list_source_artifact_cleanup_tasks(
        self,
        *,
        limit: int = 100,
    ) -> list[SourceArtifactCleanupTask]:
        if limit <= 0:
            return []
        tasks: list[SourceArtifactCleanupTask] = []
        async with self.db.execute(
            "SELECT task_id, source_id, artifact_uri, attempt_count, last_error, created_at, updated_at "
            "FROM source_artifact_cleanup_tasks ORDER BY created_at, task_id LIMIT ?",
            (limit,),
        ) as cursor:
            async for row in cursor:
                tasks.append(
                    SourceArtifactCleanupTask(
                        task_id=str(row["task_id"]),
                        source_id=str(row["source_id"]),
                        artifact_uri=str(row["artifact_uri"]),
                        attempt_count=int(row["attempt_count"]),
                        last_error=row["last_error"],
                        created_at=_parse_dt(row["created_at"]),
                        updated_at=_parse_dt(row["updated_at"]),
                    )
                )
        return tasks

    async def enqueue_source_artifact_cleanup_task(
        self,
        *,
        source_id: str,
        artifact_uri: str,
    ) -> None:
        task_id = source_artifact_cleanup_task_id(source_id, artifact_uri)
        async with self._write_lock:
            await self.db.execute(
                "INSERT OR IGNORE INTO source_artifact_cleanup_tasks "
                "(task_id, source_id, artifact_uri) VALUES (?, ?, ?)",
                (task_id, source_id, artifact_uri),
            )
            await self.db.commit()

    async def complete_source_artifact_cleanup_task(self, task_id: str) -> None:
        async with self._write_lock:
            await self.db.execute(
                "DELETE FROM source_artifact_cleanup_tasks WHERE task_id = ?",
                (task_id,),
            )
            await self.db.commit()

    async def fail_source_artifact_cleanup_task(self, task_id: str, error: str) -> None:
        async with self._write_lock:
            await self.db.execute(
                "UPDATE source_artifact_cleanup_tasks SET "
                "attempt_count = attempt_count + 1, last_error = ?, updated_at = datetime('now') "
                "WHERE task_id = ?",
                (error, task_id),
            )
            await self.db.commit()

    async def update_source_doc_count(self, source_id: str, count: int) -> None:
        async with self._write_lock:
            await self.db.execute(
                "UPDATE sources SET doc_count = ? WHERE id = ?",
                (count, source_id),
            )
            await self.db.commit()

    async def reset_source_sync_cursor(self, source_id: str) -> None:
        """Force the next sync for a source to run as a full sync."""
        async with self._write_lock:
            await self.db.execute(
                "DELETE FROM sync_state WHERE source = ?",
                (source_id,),
            )
            await self.db.execute(
                "DELETE FROM sync_history WHERE source = ?",
                (source_id,),
            )
            await self.db.execute(
                "UPDATE sources SET last_sync = NULL WHERE id = ?",
                (source_id,),
            )
            await self.db.commit()

    # ==================================================================
    # Auth sessions
    # ==================================================================

    async def upsert_auth_session(
        self,
        *,
        provider: str,
        origin: str,
        secret_encrypted: str,
        principal_id: str | None,
        principal_name: str | None,
        principal_email: str | None,
        browser: str | None,
        status: str,
        captured_at: str,
        validated_at: str | None,
        last_error: str | None,
    ) -> None:
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO auth_sessions (
                    provider, origin, secret_encrypted, principal_id, principal_name,
                    principal_email, browser, status, captured_at, validated_at,
                    last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, origin) DO UPDATE SET
                    secret_encrypted=excluded.secret_encrypted,
                    principal_id=excluded.principal_id,
                    principal_name=excluded.principal_name,
                    principal_email=excluded.principal_email,
                    browser=excluded.browser,
                    status=excluded.status,
                    captured_at=excluded.captured_at,
                    validated_at=excluded.validated_at,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at""",
                (
                    provider,
                    origin,
                    secret_encrypted,
                    principal_id,
                    principal_name,
                    principal_email,
                    browser,
                    status,
                    captured_at,
                    validated_at,
                    last_error,
                    _now_iso(),
                ),
            )
            await self.db.commit()

    async def upsert_auth_session_and_reset_sources(
        self,
        *,
        provider: str,
        origin: str,
        secret_encrypted: str,
        principal_id: str | None,
        principal_name: str | None,
        principal_email: str | None,
        browser: str | None,
        status: str,
        captured_at: str,
        validated_at: str | None,
        last_error: str | None,
        reset_source_ids: list[str],
    ) -> None:
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO auth_sessions (
                    provider, origin, secret_encrypted, principal_id, principal_name,
                    principal_email, browser, status, captured_at, validated_at,
                    last_error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(provider, origin) DO UPDATE SET
                    secret_encrypted=excluded.secret_encrypted,
                    principal_id=excluded.principal_id,
                    principal_name=excluded.principal_name,
                    principal_email=excluded.principal_email,
                    browser=excluded.browser,
                    status=excluded.status,
                    captured_at=excluded.captured_at,
                    validated_at=excluded.validated_at,
                    last_error=excluded.last_error,
                    updated_at=excluded.updated_at""",
                (
                    provider,
                    origin,
                    secret_encrypted,
                    principal_id,
                    principal_name,
                    principal_email,
                    browser,
                    status,
                    captured_at,
                    validated_at,
                    last_error,
                    _now_iso(),
                ),
            )
            for source_id in reset_source_ids:
                await self.db.execute("DELETE FROM sync_state WHERE source = ?", (source_id,))
                await self.db.execute("DELETE FROM sync_history WHERE source = ?", (source_id,))
                await self.db.execute("UPDATE sources SET last_sync = NULL WHERE id = ?", (source_id,))
            await self.db.commit()

    async def get_auth_session(self, provider: str, origin: str) -> dict | None:
        async with self.db.execute(
            "SELECT * FROM auth_sessions WHERE provider = ? AND origin = ?",
            (provider, origin),
        ) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def list_auth_sessions(self, provider: str | None = None) -> list[dict]:
        query = "SELECT * FROM auth_sessions"
        params: list = []
        if provider:
            query += " WHERE provider = ?"
            params.append(provider)
        query += " ORDER BY origin"
        rows: list[dict] = []
        async with self.db.execute(query, params) as cursor:
            async for row in cursor:
                rows.append(dict(row))
        return rows

    async def mark_auth_session_status(
        self,
        *,
        provider: str,
        origin: str,
        status: str,
        last_error: str | None,
    ) -> None:
        async with self._write_lock:
            await self.db.execute(
                """UPDATE auth_sessions
                   SET status = ?, last_error = ?, updated_at = ?
                   WHERE provider = ? AND origin = ?""",
                (status, last_error, _now_iso(), provider, origin),
            )
            await self.db.commit()

    async def delete_auth_session(self, provider: str, origin: str) -> bool:
        """Delete a stored auth session. Returns True if a row was removed."""
        async with self._write_lock:
            cursor = await self.db.execute(
                "DELETE FROM auth_sessions WHERE provider = ? AND origin = ?",
                (provider, origin),
            )
            await self.db.commit()
            return cursor.rowcount > 0

    # ==================================================================
    # Agent session receipt lineage
    # ==================================================================

    async def upsert_agent_session_receipt(self, receipt: AgentSessionReceipt) -> None:
        """Insert or update lineage for a generated agent session document."""
        activity_at = successful_agent_session_activity_at(receipt)
        async with self._write_lock:
            try:
                await self.db.execute(
                    """INSERT INTO agent_session_receipts (
                        doc_id, source_id, client, session_id, trigger, workspace,
                        repo, branch, commit_sha, history_window_kind,
                        history_window_start, history_window_end, submitted_at,
                        document_hash, source_kind, document_uri, metadata, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(doc_id) DO UPDATE SET
                        source_id=excluded.source_id,
                        client=excluded.client,
                        session_id=excluded.session_id,
                        trigger=excluded.trigger,
                        workspace=excluded.workspace,
                        repo=excluded.repo,
                        branch=excluded.branch,
                        commit_sha=excluded.commit_sha,
                        history_window_kind=excluded.history_window_kind,
                        history_window_start=excluded.history_window_start,
                        history_window_end=excluded.history_window_end,
                        submitted_at=excluded.submitted_at,
                        document_hash=excluded.document_hash,
                        source_kind=excluded.source_kind,
                        document_uri=excluded.document_uri,
                        metadata=excluded.metadata,
                        updated_at=excluded.updated_at""",
                    (
                        receipt.doc_id,
                        receipt.source_id,
                        receipt.client,
                        receipt.session_id,
                        receipt.trigger,
                        receipt.workspace,
                        receipt.repo,
                        receipt.branch,
                        receipt.commit_sha,
                        receipt.history_window_kind,
                        receipt.history_window_start,
                        receipt.history_window_end,
                        receipt.submitted_at,
                        receipt.document_hash,
                        receipt.source_kind,
                        receipt.document_uri,
                        json.dumps(receipt.metadata),
                        receipt.updated_at or _now_iso(),
                    ),
                )
                if activity_at is not None:
                    await self.db.execute(
                        """UPDATE sources
                           SET last_sync = CASE
                               WHEN last_sync IS NULL OR last_sync < ? THEN ?
                               ELSE last_sync
                           END
                           WHERE id = ?""",
                        (activity_at, activity_at, receipt.source_id),
                    )
                await self.db.commit()
            except BaseException:
                rollback_task = asyncio.create_task(self.db.rollback())
                await _drain_task_despite_cancellation(rollback_task)
                raise

    async def get_agent_session_receipt(self, doc_id: str) -> dict | None:
        """Return receipt metadata for one generated agent session document."""
        async with self.db.execute(
            "SELECT * FROM agent_session_receipts WHERE doc_id = ?",
            (doc_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return self._row_to_agent_session_receipt(row)

    async def list_agent_session_receipts(
        self,
        source_id: str | None = None,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List generated agent session document receipts."""
        query = "SELECT * FROM agent_session_receipts WHERE 1=1"
        params: list = []
        if source_id:
            query += " AND source_id = ?"
            params.append(source_id)
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY submitted_at DESC LIMIT ?"
        params.append(limit)

        results: list[dict] = []
        async with self.db.execute(query, params) as cursor:
            async for row in cursor:
                results.append(self._row_to_agent_session_receipt(row))
        return results

    async def summarize_agent_session_outcomes(
        self,
        *,
        session_id: str | None = None,
        source_id: str | None = None,
    ) -> dict:
        """Return window outcome counts and the no_output fraction (completeness read).

        A read-only, on-demand pass over agent_session_receipts. Capture
        completeness is the client-side bookmark check; this is the other half,
        knowledge completeness: of the windows that were processed, how many kept
        a knowledge patch versus dropped everything as no_output. No stored verdict, no
        threshold, no background job. Explicit-document receipts and receipts
        without a recognized outcome are ignored so the fraction stays
        well-defined.

        When at least one failed receipt is present, ``latest_failure`` carries
        ``count``, ``reason`` (latest), and ``last_seen_at`` so the admin UI can
        surface an operational warning without a second query.
        """
        query = "SELECT metadata, updated_at FROM agent_session_receipts WHERE source_kind = ?"
        params: list = [AGENT_SESSION_WINDOW_SOURCE_KIND]
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        if source_id:
            query += " AND source_id = ?"
            params.append(source_id)

        counts = {outcome: 0 for outcome in AGENT_SESSION_OUTCOMES}
        latest_failure_reason: str | None = None
        latest_failure_seen_at: str | None = None
        async with self.db.execute(query, params) as cursor:
            async for row in cursor:
                try:
                    metadata = json.loads(row[0] or "{}")
                except (json.JSONDecodeError, TypeError):
                    metadata = {}
                outcome = metadata.get("outcome") if isinstance(metadata, dict) else None
                if outcome == AGENT_SESSION_OUTCOME_LEGACY_PACKAGE_CREATED:
                    outcome = AGENT_SESSION_OUTCOME_KNOWLEDGE_PATCHED
                if outcome in counts:
                    counts[outcome] += 1
                if outcome == AGENT_SESSION_OUTCOME_FAILED:
                    seen_at = row[1]
                    if seen_at and (latest_failure_seen_at is None or seen_at > latest_failure_seen_at):
                        latest_failure_seen_at = seen_at
                        reason = metadata.get("reason") if isinstance(metadata, dict) else None
                        latest_failure_reason = reason if isinstance(reason, str) else None

        total = sum(counts.values())
        processed_total = counts[AGENT_SESSION_OUTCOME_KNOWLEDGE_PATCHED] + counts[AGENT_SESSION_OUTCOME_NO_OUTPUT]
        no_output_fraction = counts[AGENT_SESSION_OUTCOME_NO_OUTPUT] / processed_total if processed_total else 0.0
        latest_failure: dict | None = None
        if counts[AGENT_SESSION_OUTCOME_FAILED]:
            latest_failure = {
                "count": counts[AGENT_SESSION_OUTCOME_FAILED],
                "reason": latest_failure_reason,
                "last_seen_at": latest_failure_seen_at,
            }
        return {
            "session_id": session_id,
            "source_id": source_id,
            "total": total,
            "processed_total": processed_total,
            "counts": counts,
            "no_output_fraction": no_output_fraction,
            "latest_failure": latest_failure,
        }

    async def upsert_agent_hook_receipt(self, receipt: AgentHookReceipt) -> None:
        """Insert or update lineage for a coding-agent lifecycle hook event."""
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO agent_hook_receipts (
                    receipt_id, client, session_id, hook, workspace, repo, branch,
                    commit_sha, submitted_at, metadata, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(receipt_id) DO UPDATE SET
                    client=excluded.client,
                    session_id=excluded.session_id,
                    hook=excluded.hook,
                    workspace=excluded.workspace,
                    repo=excluded.repo,
                    branch=excluded.branch,
                    commit_sha=excluded.commit_sha,
                    submitted_at=excluded.submitted_at,
                    metadata=excluded.metadata,
                    updated_at=excluded.updated_at""",
                (
                    receipt.receipt_id,
                    receipt.client,
                    receipt.session_id,
                    receipt.hook,
                    receipt.workspace,
                    receipt.repo,
                    receipt.branch,
                    receipt.commit_sha,
                    receipt.submitted_at,
                    json.dumps(receipt.metadata),
                    receipt.updated_at or _now_iso(),
                ),
            )
            await self.db.commit()

    async def list_agent_hook_receipts(
        self,
        session_id: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """List coding-agent lifecycle hook receipts."""
        query = "SELECT * FROM agent_hook_receipts WHERE 1=1"
        params: list = []
        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)
        query += " ORDER BY submitted_at DESC LIMIT ?"
        params.append(limit)

        results: list[dict] = []
        async with self.db.execute(query, params) as cursor:
            async for row in cursor:
                results.append(self._row_to_agent_hook_receipt(row))
        return results

    # ==================================================================
    # Sync
    # ==================================================================

    async def enqueue_source_sync_run(
        self,
        *,
        source_id: str,
        workspace_id: str = "default",
        trigger: str = "manual",
        force_full_sync: bool = False,
        input_snapshot_id: str | None = None,
        source_config_revision: str | None = None,
        predecessor_activity_id: str | None = None,
    ) -> SourceSyncRun:
        for _attempt in range(3):
            async with self._write_lock:
                try:
                    run = await self._enqueue_source_sync_run_locked(
                        source_id=source_id,
                        workspace_id=workspace_id,
                        trigger=trigger,
                        force_full_sync=force_full_sync,
                        input_snapshot_id=input_snapshot_id,
                        source_config_revision=source_config_revision,
                        predecessor_activity_id=predecessor_activity_id,
                    )
                    await self.db.commit()
                    return run
                except (_ActiveSourceSyncRunChanged, aiosqlite.IntegrityError):
                    await self.db.rollback()
                    continue
                except Exception:
                    await self.db.rollback()
                    raise
        raise RuntimeError("source sync run changed repeatedly during enqueue")

    async def _enqueue_source_sync_run_locked(
        self,
        *,
        source_id: str,
        workspace_id: str = "default",
        trigger: str = "manual",
        force_full_sync: bool = False,
        input_snapshot_id: str | None = None,
        source_config_revision: str | None = None,
        predecessor_activity_id: str | None = None,
        now: str | None = None,
    ) -> SourceSyncRun:
        now_iso = now or _now_iso()
        normalized_snapshot_id = _non_empty_string(input_snapshot_id)
        normalized_config_revision = _non_empty_string(source_config_revision)
        normalized_predecessor_activity_id = _non_empty_string(predecessor_activity_id)
        source_lock = await self.db.execute(
            "UPDATE sources SET status = status WHERE id = ?",
            (source_id,),
        )
        if source_lock.rowcount != 1:
            raise ValueError(f"Source not found: {source_id}")
        async with self.db.execute(
            "SELECT type, config, status FROM sources WHERE id = ?",
            (source_id,),
        ) as cursor:
            source = await cursor.fetchone()
        if source is None:
            raise ValueError(f"Source not found: {source_id}")
        if source["status"] != "active":
            raise ValueError(f"Source is not active: {source_id} ({source['status']})")
        if normalized_config_revision is not None:
            actual_config_revision = local_agent_source_config_revision(
                {
                    "type": source["type"],
                    "config": json.loads(source["config"]),
                }
            )
            if actual_config_revision != normalized_config_revision:
                raise ValueError("source config revision changed before sync enqueue")
        async with self.db.execute(
            """SELECT id FROM lifecycle_backfill_jobs
               WHERE source_id = ? AND status IN ('queued', 'running')
               ORDER BY created_at LIMIT 1""",
            (source_id,),
        ) as cursor:
            lifecycle_job = await cursor.fetchone()
        if lifecycle_job is not None:
            raise SourceActivityConflict(f"source lifecycle maintenance active: {lifecycle_job['id']}")
        async with self.db.execute(
            """SELECT MAX(input_generation) AS watermark
               FROM source_sync_inputs
               WHERE workspace_id = ? AND source_id = ?""",
            (workspace_id, source_id),
        ) as cursor:
            watermark_row = await cursor.fetchone()
        input_generation_watermark = (
            int(watermark_row["watermark"])
            if watermark_row is not None and watermark_row["watermark"] is not None
            else None
        )
        async with self.db.execute(
            """SELECT * FROM source_sync_runs
               WHERE workspace_id = ?
                 AND source_id = ?
                 AND status IN ('pending', 'running')
               ORDER BY created_at
               LIMIT 1""",
            (workspace_id, source_id),
        ) as cursor:
            existing = await cursor.fetchone()
        if existing:
            existing_snapshot_id = _non_empty_string(existing["input_snapshot_id"])
            pending_snapshot_id = _non_empty_string(existing["rerun_input_snapshot_id"])
            has_new_snapshot = normalized_snapshot_id is not None and normalized_snapshot_id not in {
                existing_snapshot_id,
                pending_snapshot_id,
            }
            force_requires_rerun = force_full_sync and not bool(existing["force_full_sync"])
            mark_rerun = existing["status"] == "running" and (
                has_new_snapshot
                or (trigger == "local_agent" and normalized_snapshot_id is None)
                or force_requires_rerun
            )
            cursor = await self.db.execute(
                """UPDATE source_sync_runs
                   SET force_full_sync = CASE WHEN ? THEN 1 ELSE force_full_sync END,
                       rerun_requested = CASE
                           WHEN status = 'pending' THEN 0
                           WHEN ? THEN 1
                           ELSE rerun_requested
                       END,
                       input_snapshot_id = CASE
                           WHEN status = 'pending' AND ? IS NOT NULL THEN ?
                           ELSE input_snapshot_id
                       END,
                       rerun_input_snapshot_id = CASE
                           WHEN status = 'pending' THEN NULL
                           WHEN status = 'running' AND ? IS NOT NULL THEN ?
                           ELSE rerun_input_snapshot_id
                       END,
                       input_generation_watermark = CASE
                           WHEN status = 'pending' THEN ?
                           ELSE input_generation_watermark
                       END,
                       rerun_input_generation_watermark = CASE
                           WHEN status = 'pending' THEN NULL
                           WHEN status = 'running' AND ? THEN ?
                           ELSE rerun_input_generation_watermark
                       END,
                       source_config_revision = CASE
                           WHEN status = 'pending' AND ? IS NOT NULL THEN ?
                           ELSE source_config_revision
                       END,
                       rerun_source_config_revision = CASE
                           WHEN status = 'pending' THEN NULL
                           WHEN status = 'running' AND ? AND ? IS NOT NULL THEN ?
                           ELSE rerun_source_config_revision
                       END,
                       predecessor_activity_id = CASE
                           WHEN status = 'pending' AND ? IS NOT NULL THEN ?
                           ELSE predecessor_activity_id
                       END,
                       rerun_predecessor_activity_id = CASE
                           WHEN status = 'pending' THEN NULL
                           WHEN status = 'running' AND ? THEN ?
                           ELSE rerun_predecessor_activity_id
                       END,
                       updated_at = ?
                   WHERE run_id = ? AND status IN ('pending', 'running')""",
                (
                    int(force_full_sync),
                    int(mark_rerun),
                    normalized_snapshot_id,
                    normalized_snapshot_id,
                    normalized_snapshot_id,
                    normalized_snapshot_id,
                    input_generation_watermark,
                    int(mark_rerun),
                    input_generation_watermark,
                    normalized_config_revision,
                    normalized_config_revision,
                    int(mark_rerun),
                    normalized_config_revision,
                    normalized_config_revision,
                    normalized_predecessor_activity_id,
                    normalized_predecessor_activity_id,
                    int(mark_rerun),
                    normalized_predecessor_activity_id,
                    now_iso,
                    existing["run_id"],
                ),
            )
            if not cursor.rowcount:
                raise _ActiveSourceSyncRunChanged
            async with self.db.execute(
                "SELECT * FROM source_sync_runs WHERE run_id = ?",
                (existing["run_id"],),
            ) as cursor:
                existing = await cursor.fetchone()
            return _source_sync_run_from_row(existing, coalesced=True)

        run_id = f"ssr-{uuid.uuid4().hex}"
        await self.db.execute(
            """INSERT INTO source_sync_runs (
                run_id, workspace_id, source_id, trigger, status,
                force_full_sync, input_snapshot_id, input_generation_watermark,
                source_config_revision, predecessor_activity_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                workspace_id,
                source_id,
                trigger,
                int(force_full_sync),
                normalized_snapshot_id,
                input_generation_watermark,
                normalized_config_revision,
                normalized_predecessor_activity_id,
                now_iso,
                now_iso,
            ),
        )
        async with self.db.execute(
            "SELECT * FROM source_sync_runs WHERE run_id = ?",
            (run_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        return _source_sync_run_from_row(row)

    async def get_source_sync_run(self, run_id: str) -> SourceSyncRun | None:
        async with self.db.execute(
            "SELECT * FROM source_sync_runs WHERE run_id = ?",
            (run_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return _source_sync_run_from_row(row) if row else None

    async def cancel_pending_source_sync_run(
        self,
        *,
        source_id: str,
        error_message: str,
    ) -> bool:
        """Terminate an unleased durable run so source configuration can change."""

        cancelled_at = _now_iso()
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE source_sync_runs
                   SET status = 'failed',
                       rerun_requested = 0,
                       rerun_input_snapshot_id = NULL,
                       rerun_input_generation_watermark = NULL,
                       rerun_source_config_revision = NULL,
                       rerun_predecessor_activity_id = NULL,
                       lease_owner = NULL,
                       lease_expires_at = NULL,
                       next_attempt_at = NULL,
                       error_message = ?,
                       completed_at = ?,
                       updated_at = ?
                   WHERE source_id = ? AND status = 'pending'""",
                (
                    error_message,
                    cancelled_at,
                    cancelled_at,
                    source_id,
                ),
            )
            await self.db.commit()
        return bool(cursor.rowcount)

    async def get_latest_source_sync_run(
        self,
        *,
        source_id: str,
        workspace_id: str = "default",
    ) -> SourceSyncRun | None:
        async with self.db.execute(
            """SELECT * FROM source_sync_runs
               WHERE workspace_id = ? AND source_id = ?
               ORDER BY created_at DESC, run_id DESC
               LIMIT 1""",
            (workspace_id, source_id),
        ) as cursor:
            row = await cursor.fetchone()
        return _source_sync_run_from_row(row) if row else None

    async def lease_next_source_sync_run(
        self,
        *,
        worker_id: str,
        workspace_id: str | None = None,
        lease_seconds: int = 300,
        now: datetime | None = None,
    ) -> SourceSyncRun | None:
        lease_started_at = now or datetime.now(timezone.utc)
        lease_started_iso = _utc_iso(lease_started_at)
        lease_expires_at = _utc_iso(lease_started_at + timedelta(seconds=lease_seconds))
        conditions = [
            "((status = 'pending' AND (next_attempt_at IS NULL OR next_attempt_at <= ?)) "
            "OR (status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?))",
            "NOT EXISTS ("
            "SELECT 1 FROM source_activity_leases predecessor "
            "WHERE predecessor.id = source_sync_runs.predecessor_activity_id "
            "AND predecessor.source_id = source_sync_runs.source_id "
            "AND predecessor.kind = 'external_collection' "
            "AND predecessor.lease_until > ?)",
        ]
        params: list[Any] = [lease_started_iso, lease_started_iso, lease_started_iso]
        if workspace_id is not None:
            conditions.append("workspace_id = ?")
            params.append(workspace_id)

        async with self._write_lock:
            async with self.db.execute(
                "SELECT * FROM source_sync_runs WHERE " + " AND ".join(conditions) + " ORDER BY created_at LIMIT 1",
                params,
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                return None

            recovery_increment = 1 if row["status"] == "running" else 0
            cursor = await self.db.execute(
                """UPDATE source_sync_runs
                   SET status = 'running',
                       lease_owner = ?,
                       lease_expires_at = ?,
                       lease_attempt_count = lease_attempt_count + 1,
                       recovery_count = recovery_count + ?,
                       next_attempt_at = NULL,
                       started_at = COALESCE(started_at, ?),
                       updated_at = ?
                   WHERE run_id = ?
                     AND (
                       (status = 'pending' AND (next_attempt_at IS NULL OR next_attempt_at <= ?))
                       OR
                       (status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?)
                     )
                     AND NOT EXISTS (
                       SELECT 1 FROM source_activity_leases predecessor
                       WHERE predecessor.id = ?
                         AND predecessor.source_id = ?
                         AND predecessor.kind = 'external_collection'
                         AND predecessor.lease_until > ?
                     )""",
                (
                    worker_id,
                    lease_expires_at,
                    recovery_increment,
                    lease_started_iso,
                    lease_started_iso,
                    row["run_id"],
                    lease_started_iso,
                    lease_started_iso,
                    row["predecessor_activity_id"],
                    row["source_id"],
                    lease_started_iso,
                ),
            )
            if not cursor.rowcount:
                await self.db.rollback()
                return None
            await self.db.commit()
            async with self.db.execute(
                "SELECT * FROM source_sync_runs WHERE run_id = ?",
                (row["run_id"],),
            ) as cursor:
                leased = await cursor.fetchone()
        return _source_sync_run_from_row(leased) if leased else None

    async def heartbeat_source_sync_run(
        self,
        run_id: str,
        *,
        worker_id: str,
        lease_attempt_count: int,
        lease_seconds: int = 300,
        now: datetime | None = None,
    ) -> bool:
        heartbeat_at = now or datetime.now(timezone.utc)
        heartbeat_iso = _utc_iso(heartbeat_at)
        lease_expires_at = _utc_iso(heartbeat_at + timedelta(seconds=lease_seconds))
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE source_sync_runs
                   SET lease_expires_at = ?,
                       updated_at = ?
                   WHERE run_id = ?
                     AND status = 'running'
                     AND lease_owner = ?
                     AND lease_attempt_count = ?""",
                (
                    lease_expires_at,
                    heartbeat_iso,
                    run_id,
                    worker_id,
                    lease_attempt_count,
                ),
            )
            await self.db.commit()
        return bool(cursor.rowcount)

    async def report_source_sync_run_progress(
        self,
        run_id: str,
        *,
        worker_id: str,
        lease_attempt_count: int,
        progress: Mapping[str, Any],
        now: datetime | None = None,
    ) -> bool:
        snapshot = normalize_sync_progress_snapshot(progress)
        progress_at = _utc_iso(now)
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE source_sync_runs
                   SET progress_json = ?,
                       progress_revision = progress_revision + 1,
                       progress_updated_at = ?
                   WHERE run_id = ?
                     AND status = 'running'
                     AND lease_owner = ?
                     AND lease_attempt_count = ?""",
                (
                    json.dumps(snapshot, sort_keys=True, separators=(",", ":")),
                    progress_at,
                    run_id,
                    worker_id,
                    lease_attempt_count,
                ),
            )
            await self.db.commit()
        return bool(cursor.rowcount)

    async def complete_source_sync_run(
        self,
        run_id: str,
        *,
        worker_id: str,
        lease_attempt_count: int,
        final_state: SyncState | None = None,
        completed_at: datetime | None = None,
    ) -> bool:
        completed_iso = _utc_iso(completed_at)
        if final_state is not None and final_state.last_sync_status != "success":
            raise ValueError("complete_source_sync_run requires a successful final state")
        status = "success"
        async with self._write_lock:
            async with self.db.execute(
                """SELECT * FROM source_sync_runs
                   WHERE run_id = ? AND status = 'running'
                     AND lease_owner = ? AND lease_attempt_count = ?
                     AND lease_expires_at > ?""",
                (run_id, worker_id, lease_attempt_count, completed_iso),
            ) as cursor:
                leased_run = await cursor.fetchone()
            if leased_run is None:
                return False
            if final_state is not None:
                await self.db.execute(
                    """INSERT INTO sync_state (
                        source, last_sync_at, last_sync_status,
                        docs_processed, docs_updated, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source) DO UPDATE SET
                        last_sync_at=excluded.last_sync_at,
                        last_sync_status=excluded.last_sync_status,
                        docs_processed=excluded.docs_processed,
                        docs_updated=excluded.docs_updated,
                        error_message=excluded.error_message""",
                    (
                        final_state.source,
                        final_state.last_sync_at.isoformat() if final_state.last_sync_at else None,
                        final_state.last_sync_status,
                        final_state.docs_processed,
                        final_state.docs_updated,
                        final_state.error_message,
                    ),
                )
                if final_state.last_sync_at and final_state.last_sync_status == "success":
                    await self.db.execute(
                        "UPDATE sources SET last_sync = ? WHERE id = ?",
                        (final_state.last_sync_at.isoformat(), final_state.source),
                    )
            cursor = await self.db.execute(
                """UPDATE source_sync_runs
                   SET status = ?,
                       lease_owner = NULL,
                       lease_expires_at = NULL,
                       next_attempt_at = NULL,
                       error_message = ?,
                       completed_at = ?,
                       updated_at = ?
                   WHERE run_id = ? AND status = 'running'
                     AND lease_owner = ? AND lease_attempt_count = ?
                     AND lease_expires_at > ?""",
                (
                    status,
                    final_state.error_message if final_state else None,
                    completed_iso,
                    completed_iso,
                    run_id,
                    worker_id,
                    lease_attempt_count,
                    completed_iso,
                ),
            )
            if not cursor.rowcount:
                await self.db.rollback()
                return False
            if status == "success":
                await self._enqueue_successor_for_completed_run(run_id, completed_iso)
            await self.db.commit()
        return True

    async def fail_source_sync_run(
        self,
        run_id: str,
        *,
        worker_id: str,
        lease_attempt_count: int,
        error_message: str,
        final_state: SyncState | None = None,
        retryable: bool = True,
        failed_at: datetime | None = None,
        next_attempt_at: datetime | None = None,
    ) -> bool:
        if retryable and next_attempt_at is None:
            raise ValueError("retryable source sync failure requires next_attempt_at")
        failed_iso = _utc_iso(failed_at)
        status = "pending" if retryable else "failed"
        completed_at = None if retryable else failed_iso
        next_attempt_iso = _utc_iso(next_attempt_at) if retryable and next_attempt_at else None
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE source_sync_runs
                   SET status = ?,
                       input_snapshot_id = CASE
                           WHEN ? AND rerun_requested = 1
                           THEN COALESCE(rerun_input_snapshot_id, input_snapshot_id)
                           ELSE input_snapshot_id
                       END,
                       input_generation_watermark = CASE
                           WHEN ? AND rerun_requested = 1
                           THEN COALESCE(
                               rerun_input_generation_watermark,
                               input_generation_watermark
                           )
                           ELSE input_generation_watermark
                       END,
                       source_config_revision = CASE
                           WHEN ? AND rerun_requested = 1
                           THEN COALESCE(
                               rerun_source_config_revision,
                               source_config_revision
                           )
                           ELSE source_config_revision
                       END,
                       predecessor_activity_id = CASE
                           WHEN ? AND rerun_requested = 1
                           THEN COALESCE(
                               rerun_predecessor_activity_id,
                               predecessor_activity_id
                           )
                           ELSE predecessor_activity_id
                       END,
                       rerun_requested = CASE WHEN ? THEN 0 ELSE rerun_requested END,
                       rerun_input_snapshot_id = CASE
                           WHEN ? THEN NULL ELSE rerun_input_snapshot_id
                       END,
                       rerun_input_generation_watermark = CASE
                           WHEN ? THEN NULL ELSE rerun_input_generation_watermark
                       END,
                       rerun_source_config_revision = CASE
                           WHEN ? THEN NULL ELSE rerun_source_config_revision
                       END,
                       rerun_predecessor_activity_id = CASE
                           WHEN ? THEN NULL ELSE rerun_predecessor_activity_id
                       END,
                       lease_owner = NULL,
                       lease_expires_at = NULL,
                       next_attempt_at = ?,
                       error_message = ?,
                       completed_at = ?,
                       updated_at = ?
                   WHERE run_id = ? AND status = 'running'
                     AND lease_owner = ? AND lease_attempt_count = ?
                     AND lease_expires_at > ?""",
                (
                    status,
                    int(retryable),
                    int(retryable),
                    int(retryable),
                    int(retryable),
                    int(retryable),
                    int(retryable),
                    int(retryable),
                    int(retryable),
                    int(retryable),
                    next_attempt_iso,
                    error_message,
                    completed_at,
                    failed_iso,
                    run_id,
                    worker_id,
                    lease_attempt_count,
                    failed_iso,
                ),
            )
            if not cursor.rowcount:
                await self.db.rollback()
                return False
            if final_state is not None:
                await self.db.execute(
                    """INSERT INTO sync_state (
                        source, last_sync_at, last_sync_status,
                        docs_processed, docs_updated, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source) DO UPDATE SET
                        last_sync_at=excluded.last_sync_at,
                        last_sync_status=excluded.last_sync_status,
                        docs_processed=excluded.docs_processed,
                        docs_updated=excluded.docs_updated,
                        error_message=excluded.error_message""",
                    (
                        final_state.source,
                        final_state.last_sync_at.isoformat() if final_state.last_sync_at else None,
                        final_state.last_sync_status,
                        final_state.docs_processed,
                        final_state.docs_updated,
                        final_state.error_message,
                    ),
                )
            if not retryable:
                await self._enqueue_successor_for_completed_run(run_id, failed_iso)
            await self.db.commit()
        return True

    async def _enqueue_successor_for_completed_run(self, run_id: str, now: str) -> None:
        async with self.db.execute(
            "SELECT * FROM source_sync_runs WHERE run_id = ?",
            (run_id,),
        ) as cursor:
            completed = await cursor.fetchone()
        if not completed or not bool(completed["rerun_requested"]):
            return
        successor_id = f"ssr-{uuid.uuid4().hex}"
        await self.db.execute(
            """INSERT INTO source_sync_runs (
                run_id, workspace_id, source_id, trigger, status,
                force_full_sync, input_snapshot_id, input_generation_watermark,
                source_config_revision, predecessor_activity_id,
                created_at, updated_at
            ) VALUES (?, ?, ?, 'rerun', 'pending', ?, ?, ?, ?, ?, ?, ?)""",
            (
                successor_id,
                completed["workspace_id"],
                completed["source_id"],
                int(completed["force_full_sync"]),
                completed["rerun_input_snapshot_id"],
                completed["rerun_input_generation_watermark"],
                completed["rerun_source_config_revision"],
                completed["rerun_predecessor_activity_id"],
                now,
                now,
            ),
        )

    async def create_source_sync_input(
        self,
        *,
        source_id: str,
        workspace_id: str = "default",
        raw_uri: str,
        raw_sha256: str,
        raw_content_type: str,
        metadata: dict[str, object] | None = None,
        sync_snapshot_id: str | None = None,
        expected_activity_epoch: int | None = None,
    ) -> SourceSyncInput:
        now = _now_iso()
        metadata_payload = dict(metadata or {})
        snapshot_id = _non_empty_string(sync_snapshot_id)
        snapshot_doc_id = _manifest_doc_id(metadata_payload)
        if snapshot_id and not snapshot_doc_id:
            raise ValueError("snapshot input requires doc_id")
        async with self._write_lock:
            try:
                source_lock = await self.db.execute(
                    "UPDATE sources SET status = status WHERE id = ?",
                    (source_id,),
                )
                if source_lock.rowcount != 1:
                    raise ValueError(f"Source not found: {source_id}")
                async with self.db.execute(
                    """SELECT id FROM lifecycle_backfill_jobs
                       WHERE source_id = ? AND status IN ('queued', 'running')
                       ORDER BY created_at LIMIT 1""",
                    (source_id,),
                ) as cursor:
                    lifecycle_job = await cursor.fetchone()
                if lifecycle_job is not None:
                    raise SourceActivityConflict(f"source lifecycle maintenance active: {lifecycle_job['id']}")
                if expected_activity_epoch is not None:
                    async with self.db.execute(
                        "SELECT activity_epoch FROM sources WHERE id = ?",
                        (source_id,),
                    ) as cursor:
                        epoch_row = await cursor.fetchone()
                    current_epoch = int(epoch_row["activity_epoch"] or 0)
                    if current_epoch != expected_activity_epoch:
                        raise SourceActivityConflict(
                            "source activity epoch changed: "
                            f"expected {expected_activity_epoch}, current {current_epoch}"
                        )
            except Exception:
                await self.db.rollback()
                raise
            try:
                async with self.db.execute(
                    """SELECT * FROM source_sync_inputs
                   WHERE workspace_id = ? AND source_id = ? AND raw_sha256 = ?""",
                    (workspace_id, source_id, raw_sha256),
                ) as cursor:
                    existing = await cursor.fetchone()
                if existing is not None:
                    if snapshot_id and snapshot_doc_id:
                        await self._record_source_sync_snapshot_item_unlocked(
                            workspace_id=workspace_id,
                            source_id=source_id,
                            snapshot_id=snapshot_id,
                            doc_id=snapshot_doc_id,
                            input_id=str(existing["input_id"]),
                            now=now,
                        )
                    await self.db.commit()
                    return _source_sync_input_from_row(existing)
                async with self.db.execute(
                    """SELECT COALESCE(MAX(input_generation), 0) + 1 AS next_generation
                   FROM source_sync_inputs
                   WHERE workspace_id = ? AND source_id = ?""",
                    (workspace_id, source_id),
                ) as cursor:
                    row = await cursor.fetchone()
                generation = int(row["next_generation"] if row else 1)
                input_id = f"ssi-{uuid.uuid4().hex}"
                await self.db.execute(
                    """INSERT INTO source_sync_inputs (
                    input_id, workspace_id, source_id, input_generation,
                    raw_uri, raw_sha256, raw_content_type, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        input_id,
                        workspace_id,
                        source_id,
                        generation,
                        raw_uri,
                        raw_sha256,
                        raw_content_type,
                        json.dumps(metadata_payload, sort_keys=True),
                        now,
                    ),
                )
                if snapshot_id and snapshot_doc_id:
                    await self._record_source_sync_snapshot_item_unlocked(
                        workspace_id=workspace_id,
                        source_id=source_id,
                        snapshot_id=snapshot_id,
                        doc_id=snapshot_doc_id,
                        input_id=input_id,
                        now=now,
                    )
                await self.db.commit()
                async with self.db.execute(
                    "SELECT * FROM source_sync_inputs WHERE input_id = ?",
                    (input_id,),
                ) as cursor:
                    inserted = await cursor.fetchone()
            except Exception:
                await self.db.rollback()
                raise
        assert inserted is not None
        return _source_sync_input_from_row(inserted)

    async def list_source_sync_inputs(
        self,
        *,
        source_id: str,
        workspace_id: str = "default",
        input_snapshot_id: str | None = None,
    ) -> list[SourceSyncInput]:
        results: list[SourceSyncInput] = []
        snapshot_id = _non_empty_string(input_snapshot_id)
        if snapshot_id:
            query = """SELECT i.* FROM source_sync_snapshot_items si
                       JOIN source_sync_inputs i ON i.input_id = si.input_id
                       WHERE si.workspace_id = ? AND si.source_id = ?
                         AND si.snapshot_id = ?
                       ORDER BY i.input_generation"""
            params: tuple[Any, ...] = (workspace_id, source_id, snapshot_id)
        else:
            query = """SELECT * FROM source_sync_inputs
                       WHERE workspace_id = ? AND source_id = ?
                       ORDER BY input_generation"""
            params = (workspace_id, source_id)
        async with self.db.execute(query, params) as cursor:
            async for row in cursor:
                results.append(_source_sync_input_from_row(row))
        return results

    async def attest_source_sync_input_artifact(
        self,
        *,
        source_id: str,
        input_id: str,
        package_sha256: str,
        expected_activity_epoch: int | None = None,
    ) -> SourceSyncInput:
        """Atomically fill or verify a retained input's own package hash."""
        async with self._write_lock:
            try:
                source_lock = await self.db.execute(
                    "UPDATE sources SET status = status WHERE id = ?",
                    (source_id,),
                )
                if source_lock.rowcount != 1:
                    raise ValueError(f"Source not found: {source_id}")
                async with self.db.execute(
                    """SELECT id FROM lifecycle_backfill_jobs
                       WHERE source_id = ? AND status IN ('queued', 'running')
                       ORDER BY created_at LIMIT 1""",
                    (source_id,),
                ) as cursor:
                    lifecycle_job = await cursor.fetchone()
                if lifecycle_job is not None:
                    raise SourceActivityConflict(f"source lifecycle maintenance active: {lifecycle_job['id']}")
                if expected_activity_epoch is not None:
                    async with self.db.execute(
                        "SELECT activity_epoch FROM sources WHERE id = ?",
                        (source_id,),
                    ) as cursor:
                        epoch_row = await cursor.fetchone()
                    current_epoch = int(epoch_row["activity_epoch"] or 0)
                    if current_epoch != expected_activity_epoch:
                        raise SourceActivityConflict(
                            "source activity epoch changed: "
                            f"expected {expected_activity_epoch}, current {current_epoch}"
                        )
                async with self.db.execute(
                    "SELECT * FROM source_sync_inputs WHERE input_id = ? AND source_id = ?",
                    (input_id, source_id),
                ) as cursor:
                    existing = await cursor.fetchone()
                if existing is None:
                    raise ValueError(f"Source sync input not found: {input_id}")
                current = _source_sync_input_from_row(existing)
                metadata = source_sync_input_metadata_with_artifact_attestation(
                    current.metadata,
                    package_sha256=package_sha256,
                    input_id=input_id,
                )
                async with self.db.execute(
                    """UPDATE source_sync_inputs SET metadata_json = ?
                       WHERE input_id = ? AND source_id = ?""",
                    (json.dumps(metadata, sort_keys=True), input_id, source_id),
                ) as cursor:
                    if cursor.rowcount != 1:
                        raise ValueError(f"Source sync input not found: {input_id}")
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        return replace(current, metadata=metadata)

    async def _record_source_sync_snapshot_item_unlocked(
        self,
        *,
        workspace_id: str,
        source_id: str,
        snapshot_id: str,
        doc_id: str,
        input_id: str,
        now: str,
    ) -> None:
        await self.db.execute(
            """INSERT INTO source_sync_snapshot_items (
                workspace_id, source_id, snapshot_id, doc_id, input_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(workspace_id, source_id, snapshot_id, doc_id)
            DO UPDATE SET input_id = excluded.input_id,
                          created_at = excluded.created_at""",
            (workspace_id, source_id, snapshot_id, doc_id, input_id, now),
        )

    async def upsert_sync_state(self, state: SyncState) -> None:
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO sync_state (
                    source, last_sync_at, last_sync_status,
                    docs_processed, docs_updated, error_message
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    last_sync_at=excluded.last_sync_at,
                    last_sync_status=excluded.last_sync_status,
                    docs_processed=excluded.docs_processed,
                    docs_updated=excluded.docs_updated,
                    error_message=excluded.error_message""",
                (
                    state.source,
                    state.last_sync_at.isoformat() if state.last_sync_at else None,
                    state.last_sync_status,
                    state.docs_processed,
                    state.docs_updated,
                    state.error_message,
                ),
            )
            if state.last_sync_at and state.last_sync_status == "success":
                await self.db.execute(
                    "UPDATE sources SET last_sync = ? WHERE id = ?",
                    (state.last_sync_at.isoformat(), state.source),
                )
            await self.db.commit()

    async def get_sync_state(self, source: str) -> SyncState | None:
        async with self.db.execute("SELECT * FROM sync_state WHERE source = ?", (source,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            return SyncState(
                source=d["source"],
                last_sync_at=_parse_dt(d["last_sync_at"]),
                last_sync_status=d["last_sync_status"],
                docs_processed=d["docs_processed"] or 0,
                docs_updated=d["docs_updated"] or 0,
                error_message=d["error_message"],
            )

    async def insert_sync_history(
        self,
        source: str,
        status: str,
        docs_processed: int,
        docs_updated: int,
        docs_failed: int,
        memories_extracted: int,
        error_message: str | None,
        failed_docs: list | None,
        started_at: str,
        finished_at: str,
        run_id: str | None = None,
    ) -> None:
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO sync_history (
                    source, status, docs_processed, docs_updated, docs_failed,
                    memories_extracted, error_message, failed_docs,
                    started_at, finished_at, run_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    source,
                    status,
                    docs_processed,
                    docs_updated,
                    docs_failed,
                    memories_extracted,
                    error_message,
                    json.dumps(failed_docs) if failed_docs else None,
                    started_at,
                    finished_at,
                    run_id,
                ),
            )
            await self.db.commit()

    async def get_sync_history(
        self,
        source: str | None = None,
        limit: int = 20,
    ) -> list[dict]:
        if source:
            query = "SELECT * FROM sync_history WHERE source = ? ORDER BY finished_at DESC LIMIT ?"
            params: list = [source, limit]
        else:
            query = "SELECT * FROM sync_history ORDER BY finished_at DESC LIMIT ?"
            params = [limit]

        results: list[dict] = []
        async with self.db.execute(query, params) as cursor:
            async for row in cursor:
                d = dict(row)
                d["failed_docs"] = json.loads(d["failed_docs"]) if d.get("failed_docs") else []
                results.append(d)
        return results

    # ==================================================================
    # Memory audit ledger
    # ==================================================================

    async def insert_memory_audit_event(self, event: MemoryAuditEvent) -> None:
        """Append one memory audit event."""
        occurred_at = _utc_iso(event.occurred_at)
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO memory_audit_events (
                    event_id, operation_id, parent_event_id, occurred_at,
                    actor_type, actor_id, run_id, trace_id, source_id, doc_id,
                    memory_id, candidate_id, review_id, support_kind,
                    event_type, decision, reason, payload_class,
                    before_snapshot, after_snapshot, evidence_refs,
                    model, prompt_hash, config_hash, thresholds,
                    status, payload, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.operation_id,
                    event.parent_event_id,
                    occurred_at,
                    event.actor_type,
                    event.actor_id,
                    event.run_id,
                    event.trace_id,
                    event.source_id,
                    event.doc_id,
                    event.memory_id,
                    event.candidate_id,
                    event.review_id,
                    event.support_kind,
                    event.event_type,
                    event.decision,
                    event.reason,
                    event.payload_class,
                    json.dumps(event.before_snapshot) if event.before_snapshot is not None else None,
                    json.dumps(event.after_snapshot) if event.after_snapshot is not None else None,
                    json.dumps(event.evidence_refs),
                    event.model,
                    event.prompt_hash,
                    event.config_hash,
                    json.dumps(event.thresholds) if event.thresholds is not None else None,
                    event.status,
                    json.dumps(event.payload),
                    event.error,
                ),
            )
            await self.db.commit()

    async def list_memory_audit_events(
        self,
        *,
        operation_id: str | None = None,
        memory_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[MemoryAuditEvent]:
        """List audit events for tests, reporting, and evaluation bundles."""
        clauses = ["1=1"]
        params: list = []
        if operation_id:
            clauses.append("operation_id = ?")
            params.append(operation_id)
        if memory_id:
            clauses.append("memory_id = ?")
            params.append(memory_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        params.append(limit)
        query = (
            "SELECT * FROM memory_audit_events WHERE "
            + " AND ".join(clauses)
            + " ORDER BY occurred_at ASC, event_id ASC LIMIT ?"
        )

        rows: list[MemoryAuditEvent] = []
        async with self.db.execute(query, params) as cursor:
            async for row in cursor:
                rows.append(self._row_to_audit_event(row))
        return rows

    async def redact_memory_audit_payloads(self, memory_id: str) -> None:
        """Remove sensitive payload fields for a purged memory while preserving event metadata."""
        async with self._write_lock:
            await self.db.execute(
                """UPDATE memory_audit_events SET
                    before_snapshot = NULL,
                    after_snapshot = NULL,
                    evidence_refs = ?,
                    thresholds = NULL,
                    payload = ?,
                    error = NULL
                   WHERE memory_id = ?""",
                (json.dumps([]), json.dumps({"redacted": True}), memory_id),
            )
            await self.db.commit()

    # ==================================================================
    # Config - schedule
    # ==================================================================

    async def get_schedule_config(self) -> dict:
        async with self.db.execute("SELECT * FROM schedule_config WHERE id = 1") as cursor:
            row = await cursor.fetchone()
            if not row:
                return {
                    "enabled": False,
                    "frequency": "daily",
                    "time": "02:00",
                    "day_of_week": 0,
                    "timezone": "UTC",
                }
            d = dict(row)
            return {
                "enabled": bool(d["enabled"]),
                "frequency": d["frequency"],
                "time": d["time"],
                "day_of_week": d["day_of_week"],
                "timezone": d.get("timezone", "UTC"),
            }

    async def set_schedule_config(self, config: dict) -> None:
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO schedule_config (
                    id, enabled, frequency, time, day_of_week, timezone
                ) VALUES (1, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    enabled=excluded.enabled, frequency=excluded.frequency,
                    time=excluded.time, day_of_week=excluded.day_of_week,
                    timezone=excluded.timezone""",
                (
                    int(config.get("enabled", False)),
                    config.get("frequency", "daily"),
                    config.get("time", "02:00"),
                    config.get("day_of_week", 0),
                    config.get("timezone", "UTC"),
                ),
            )
            await self.db.commit()

    # ==================================================================
    # ==================================================================
    # Contradictions
    # ==================================================================

    async def record_contradiction(
        self,
        memory_id_a: str,
        memory_id_b: str,
        classification: str,
        reason: str | None = None,
    ) -> None:
        """Record a contradiction between two memories and increment their counts."""
        if memory_id_a == memory_id_b:
            raise ValueError("contradiction requires two distinct Memories")
        memory_id_a, memory_id_b = sorted((memory_id_a, memory_id_b))
        async with self._write_lock:
            try:
                await self.db.execute("BEGIN IMMEDIATE")
                async with self.db.execute(
                    """SELECT classification
                         FROM memory_contradictions
                        WHERE memory_id_a = ? AND memory_id_b = ?""",
                    (memory_id_a, memory_id_b),
                ) as cursor:
                    existing = await cursor.fetchone()
                increment = False
                if existing is None:
                    await self.db.execute(
                        """INSERT INTO memory_contradictions
                           (memory_id_a, memory_id_b, classification, reason)
                           VALUES (?, ?, ?, ?)""",
                        (memory_id_a, memory_id_b, classification, reason),
                    )
                    increment = classification == "contradiction"
                elif existing["classification"] != "contradiction" and classification == "contradiction":
                    await self.db.execute(
                        """UPDATE memory_contradictions
                              SET classification = ?, reason = ?
                            WHERE memory_id_a = ? AND memory_id_b = ?""",
                        (
                            classification,
                            reason,
                            memory_id_a,
                            memory_id_b,
                        ),
                    )
                    increment = True
                if increment:
                    await self.db.execute(
                        "UPDATE memories SET contradiction_count = contradiction_count + 1 WHERE id IN (?, ?)",
                        (memory_id_a, memory_id_b),
                    )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    # ==================================================================
    # Memory reviews
    # ==================================================================

    async def insert_memory_review(self, review: MemoryReview) -> str:
        """Persist a new review record. ``created_at`` defaults to now if absent."""
        async with self._write_lock:
            now = _now_iso()
            created_at = review.created_at.isoformat() if review.created_at else now
            await self.db.execute(
                """INSERT INTO memory_reviews (
                    id, kind, status, incumbent_memory_id, challenger_memory_id,
                    reason, review_note, reviewer,
                    expected_incumbent_updated_at, expected_challenger_updated_at,
                    replacement_kind, created_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    review.id,
                    review.kind,
                    review.status,
                    review.incumbent_memory_id,
                    review.challenger_memory_id,
                    review.reason,
                    review.review_note,
                    review.reviewer,
                    review.expected_incumbent_updated_at,
                    review.expected_challenger_updated_at,
                    _validate_replacement_kind(review.replacement_kind),
                    created_at,
                    review.resolved_at.isoformat() if review.resolved_at else None,
                ),
            )
            await self.db.commit()
        return review.id

    async def record_memory_review_with_relation_outcome(
        self,
        review: MemoryReview,
        relation_outcome: RelationOutcomeBundle,
    ) -> str:
        """Atomically record a review work item and its relation audit.

        Unlike ``mark_memory_pending_review_with_case``, this operation never
        changes either Memory's lifecycle status.  It is the persistence seam
        for cross-source findings, where classification alone has no authority
        to quarantine, supersede, or retire a Memory.
        """
        async with self._write_lock:
            try:
                await self.db.execute("BEGIN IMMEDIATE")
                if review.kind == "cross_source_conflict":
                    memories, supports = await self._load_cross_source_review_snapshots_unlocked(review)
                    validate_cross_source_review_write(
                        review,
                        relation_outcome,
                        memories=memories,
                        supports=supports,
                    )
                async with self.db.execute(
                    "SELECT * FROM memory_reviews WHERE id = ?",
                    (review.id,),
                ) as cursor:
                    existing_review = await cursor.fetchone()
                if existing_review is not None:
                    validate_pending_review_retry(
                        review,
                        self._row_to_review(existing_review),
                    )
                    await self._record_relation_outcome_bundle_unlocked(relation_outcome)
                    await self.db.commit()
                    return review.id

                now = _now_iso()
                await self._record_relation_outcome_bundle_unlocked(relation_outcome)
                await self.db.execute(
                    """INSERT INTO memory_reviews (
                        id, kind, status, incumbent_memory_id, challenger_memory_id,
                        reason, review_note, reviewer,
                        expected_incumbent_updated_at, expected_challenger_updated_at,
                        replacement_kind, created_at, resolved_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        review.id,
                        review.kind,
                        review.status,
                        review.incumbent_memory_id,
                        review.challenger_memory_id,
                        review.reason,
                        review.review_note,
                        review.reviewer,
                        review.expected_incumbent_updated_at,
                        review.expected_challenger_updated_at,
                        _validate_replacement_kind(review.replacement_kind),
                        review.created_at.isoformat() if review.created_at else now,
                        review.resolved_at.isoformat() if review.resolved_at else None,
                    ),
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        return review.id

    async def _load_cross_source_review_snapshots_unlocked(
        self,
        review: MemoryReview,
    ) -> tuple[
        tuple[CrossSourceReviewMemorySnapshot, ...],
        tuple[CrossSourceReviewSupportSnapshot, ...],
    ]:
        memory_ids = (
            review.incumbent_memory_id,
            review.challenger_memory_id,
        )
        async with self.db.execute(
            """SELECT id, status, superseded_by, updated_at, visibility,
                      owner_user_id, repo_identifier, project_key
                 FROM memories
                WHERE id IN (?, ?)""",
            memory_ids,
        ) as cursor:
            memory_rows = await cursor.fetchall()
        memories = tuple(
            CrossSourceReviewMemorySnapshot(
                memory_id=str(row["id"]),
                status=str(row["status"]),
                superseded_by=row["superseded_by"],
                updated_at=row["updated_at"],
                visibility=str(row["visibility"]),
                owner_user_id=row["owner_user_id"],
                repo_identifier=row["repo_identifier"],
                project_key=row["project_key"],
            )
            for row in memory_rows
        )

        async with self.db.execute(
            """SELECT MSA.memory_id,
                      MSA.source_id AS assertion_source_id,
                      MSA.access_context_hash AS assertion_access_context_hash,
                      ER.evidence_unit_id,
                      EU.source_id AS evidence_source_id,
                      EU.source_lineage_id AS evidence_source_lineage_id,
                      EU.visibility AS evidence_visibility,
                      EU.owner_user_id AS evidence_owner_user_id,
                      EU.repo_identifier AS evidence_repo_identifier,
                      EU.project_key AS evidence_project_key,
                      EU.access_context_hash AS evidence_access_context_hash,
                      ER.observation_id,
                      SO.source_id AS observation_source_id,
                      ER.observation_revision_id,
                      SO.current_revision_id AS current_observation_revision_id,
                      SO.source_unit_id,
                      SU.source_id AS source_unit_source_id,
                      S.access_policy AS source_access_policy,
                      S.owner_user_id AS source_owner_user_id
                 FROM memory_support_assertions MSA
                 JOIN evidence_references ER
                   ON ER.id = MSA.evidence_reference_id
                 JOIN evidence_units EU
                   ON EU.id = ER.evidence_unit_id
                 JOIN source_observations SO
                   ON SO.id = ER.observation_id
                 JOIN source_units SU
                   ON SU.id = SO.source_unit_id
                 JOIN sources S
                   ON S.id = MSA.source_id
                WHERE MSA.memory_id IN (?, ?) AND MSA.active = 1
                ORDER BY MSA.memory_id, MSA.id""",
            memory_ids,
        ) as cursor:
            support_rows = await cursor.fetchall()
        supports = tuple(
            CrossSourceReviewSupportSnapshot(
                memory_id=str(row["memory_id"]),
                assertion_source_id=str(row["assertion_source_id"]),
                assertion_access_context_hash=str(row["assertion_access_context_hash"]),
                evidence_unit_id=str(row["evidence_unit_id"]),
                evidence_source_id=str(row["evidence_source_id"]),
                evidence_source_lineage_id=row["evidence_source_lineage_id"],
                evidence_visibility=str(row["evidence_visibility"]),
                evidence_owner_user_id=row["evidence_owner_user_id"],
                evidence_repo_identifier=row["evidence_repo_identifier"],
                evidence_project_key=row["evidence_project_key"],
                evidence_access_context_hash=row["evidence_access_context_hash"],
                observation_id=str(row["observation_id"]),
                observation_source_id=str(row["observation_source_id"]),
                observation_revision_id=str(row["observation_revision_id"]),
                current_observation_revision_id=row["current_observation_revision_id"],
                source_unit_id=str(row["source_unit_id"]),
                source_unit_source_id=str(row["source_unit_source_id"]),
                source_access_policy=str(row["source_access_policy"]),
                source_owner_user_id=str(row["source_owner_user_id"]),
            )
            for row in support_rows
        )
        return memories, supports

    async def get_memory_review(self, review_id: str) -> MemoryReview | None:
        async with self.db.execute(
            "SELECT * FROM memory_reviews WHERE id = ?",
            (review_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return self._row_to_review(row)

    async def get_pending_review_for_incumbent(self, incumbent_id: str) -> MemoryReview | None:
        async with self.db.execute(
            """SELECT * FROM memory_reviews
               WHERE incumbent_memory_id = ? AND status = 'pending'
               ORDER BY created_at DESC LIMIT 1""",
            (incumbent_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_review(row) if row else None

    async def get_pending_review_for_challenger(self, challenger_id: str) -> MemoryReview | None:
        async with self.db.execute(
            """SELECT * FROM memory_reviews
               WHERE challenger_memory_id = ? AND status = 'pending'
               ORDER BY created_at DESC LIMIT 1""",
            (challenger_id,),
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_review(row) if row else None

    async def get_open_review_for_incumbent_source_doc(
        self,
        *,
        incumbent_memory_id: str,
        doc_id: str,
        kind: str,
    ) -> MemoryReview | None:
        """Return the visible open review case for an incumbent and source doc."""
        async with self.db.execute(
            """SELECT DISTINCT mr.* FROM memory_reviews mr
               JOIN memory_sources ms
                 ON ms.memory_id = mr.challenger_memory_id
               WHERE mr.incumbent_memory_id = ?
                 AND mr.kind = ?
                 AND mr.status IN ('pending', 'stale')
                 AND ms.doc_id = ?
               ORDER BY mr.created_at DESC LIMIT 1""",
            (incumbent_memory_id, kind, doc_id),
        ) as cursor:
            row = await cursor.fetchone()
            return self._row_to_review(row) if row else None

    async def add_memory_review_related_challenger(
        self,
        review_id: str,
        challenger_memory_id: str,
        *,
        reason: str | None = None,
    ) -> None:
        """Attach an additional challenger to an existing visible review case."""
        async with self._write_lock:
            async with self.db.execute(
                """SELECT review_id FROM memory_review_related_challengers
                   WHERE challenger_memory_id = ?""",
                (challenger_memory_id,),
            ) as cursor:
                existing = await cursor.fetchone()
            if existing:
                existing_review_id = existing["review_id"]
                if existing_review_id != review_id:
                    raise ValueError(
                        f"Challenger {challenger_memory_id} is already attached to review {existing_review_id}"
                    )
                await self.db.execute(
                    """UPDATE memory_review_related_challengers
                       SET reason = COALESCE(?, reason)
                       WHERE review_id = ? AND challenger_memory_id = ?""",
                    (reason, review_id, challenger_memory_id),
                )
                await self.db.commit()
                return
            await self.db.execute(
                """INSERT INTO memory_review_related_challengers (
                    review_id, challenger_memory_id, reason, created_at
                ) VALUES (?, ?, ?, ?)""",
                (review_id, challenger_memory_id, reason, _now_iso()),
            )
            await self.db.commit()

    async def mark_memory_pending_review_with_case(
        self,
        memory_id: str,
        *,
        reason: str | None = None,
        relation_outcome: RelationOutcomeBundle | None = None,
        review: MemoryReview | None = None,
        related_review_id: str | None = None,
    ) -> None:
        """Atomically quarantine a memory and materialize its review work item."""
        if review is not None and related_review_id is not None:
            raise ValueError("review and related_review_id are mutually exclusive")
        async with self._write_lock:
            try:
                if review is not None:
                    async with self.db.execute(
                        "SELECT status FROM memory_reviews WHERE id = ?",
                        (review.id,),
                    ) as cursor:
                        existing_review = await cursor.fetchone()
                    if existing_review is not None and existing_review["status"] != "pending":
                        raise RuntimeError(
                            f"memory review {review.id} already exists with status {existing_review['status']}"
                        )
                now = _now_iso()
                await self.db.execute(
                    """UPDATE memories
                       SET status = ?, updated_at = ?
                       WHERE id = ? AND status IN ('active', 'pending_review')""",
                    ("pending_review", now, memory_id),
                )
                if relation_outcome is not None:
                    await self._record_relation_outcome_bundle_unlocked(relation_outcome)
                if related_review_id is not None:
                    async with self.db.execute(
                        """SELECT review_id FROM memory_review_related_challengers
                           WHERE challenger_memory_id = ?""",
                        (memory_id,),
                    ) as cursor:
                        existing = await cursor.fetchone()
                    if existing:
                        existing_review_id = existing["review_id"]
                        if existing_review_id != related_review_id:
                            raise ValueError(
                                f"Challenger {memory_id} is already attached to review {existing_review_id}"
                            )
                        await self.db.execute(
                            """UPDATE memory_review_related_challengers
                               SET reason = COALESCE(?, reason)
                               WHERE review_id = ? AND challenger_memory_id = ?""",
                            (reason, related_review_id, memory_id),
                        )
                    else:
                        await self.db.execute(
                            """INSERT INTO memory_review_related_challengers (
                                review_id, challenger_memory_id, reason, created_at
                            ) VALUES (?, ?, ?, ?)""",
                            (related_review_id, memory_id, reason, now),
                        )
                if review is not None:
                    created_at = review.created_at.isoformat() if review.created_at else now
                    cursor = await self.db.execute(
                        """INSERT INTO memory_reviews (
                            id, kind, status, incumbent_memory_id, challenger_memory_id,
                            reason, review_note, reviewer,
                            expected_incumbent_updated_at, expected_challenger_updated_at,
                            replacement_kind, created_at, resolved_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(id) DO UPDATE SET
                            kind=excluded.kind,
                            status=excluded.status,
                            incumbent_memory_id=excluded.incumbent_memory_id,
                            challenger_memory_id=excluded.challenger_memory_id,
                            reason=excluded.reason,
                            review_note=excluded.review_note,
                            reviewer=excluded.reviewer,
                            expected_incumbent_updated_at=excluded.expected_incumbent_updated_at,
                            expected_challenger_updated_at=excluded.expected_challenger_updated_at,
                            replacement_kind=excluded.replacement_kind,
                            resolved_at=excluded.resolved_at
                        WHERE memory_reviews.status = 'pending'""",
                        (
                            review.id,
                            review.kind,
                            review.status,
                            review.incumbent_memory_id,
                            review.challenger_memory_id,
                            review.reason,
                            review.review_note,
                            review.reviewer,
                            review.expected_incumbent_updated_at,
                            review.expected_challenger_updated_at,
                            _validate_replacement_kind(review.replacement_kind),
                            created_at,
                            review.resolved_at.isoformat() if review.resolved_at else None,
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError(f"memory review {review.id} was not created or refreshed")
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise

    async def list_memory_review_related_challengers(
        self,
        review_id: str,
    ) -> list[MemoryReviewRelatedChallenger]:
        results: list[MemoryReviewRelatedChallenger] = []
        async with self.db.execute(
            """SELECT * FROM memory_review_related_challengers
               WHERE review_id = ?
               ORDER BY created_at, challenger_memory_id""",
            (review_id,),
        ) as cursor:
            async for row in cursor:
                d = dict(row)
                results.append(
                    MemoryReviewRelatedChallenger(
                        review_id=d["review_id"],
                        challenger_memory_id=d["challenger_memory_id"],
                        reason=d["reason"],
                        created_at=_parse_dt(d["created_at"]),
                    )
                )
        return results

    async def list_memory_reviews(
        self,
        status: str | None = None,
        kind: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[MemoryReview]:
        query = "SELECT * FROM memory_reviews WHERE 1=1"
        params: list = []
        if status:
            if status == "open":
                query += " AND status IN ('pending', 'stale')"
            else:
                query += " AND status = ?"
                params.append(status)
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        results: list[MemoryReview] = []
        async with self.db.execute(query, params) as cursor:
            async for row in cursor:
                results.append(self._row_to_review(row))
        return results

    async def count_memory_reviews(
        self,
        status: str | None = None,
        kind: str | None = None,
    ) -> int:
        query = "SELECT COUNT(*) FROM memory_reviews WHERE 1=1"
        params: list = []
        if status:
            if status == "open":
                query += " AND status IN ('pending', 'stale')"
            else:
                query += " AND status = ?"
                params.append(status)
        if kind:
            query += " AND kind = ?"
            params.append(kind)
        async with self.db.execute(query, params) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else 0

    async def resolve_memory_review(
        self,
        review_id: str,
        *,
        status: str,
        reviewer: str | None,
        review_note: str | None,
    ) -> None:
        async with self._write_lock:
            now = _now_iso()
            await self.db.execute(
                """UPDATE memory_reviews SET
                    status = ?, reviewer = ?, review_note = ?, resolved_at = ?
                   WHERE id = ?""",
                (status, reviewer, review_note, now, review_id),
            )
            await self.db.commit()

    async def refresh_memory_review_expectations(
        self,
        review_id: str,
        *,
        expected_incumbent_updated_at: str | None,
        expected_challenger_updated_at: str | None,
    ) -> None:
        """Re-pin the review's optimistic-concurrency expectations to current state.

        Returns the review to ``pending`` and clears any reviewer/note/resolved
        timestamp left from the previous stale-marking attempt: the next
        decision starts fresh against the refreshed expectations.
        """
        async with self._write_lock:
            await self.db.execute(
                """UPDATE memory_reviews SET
                    expected_incumbent_updated_at = ?,
                    expected_challenger_updated_at = ?,
                    status = 'pending',
                    reviewer = NULL,
                    review_note = NULL,
                    resolved_at = NULL
                   WHERE id = ?""",
                (expected_incumbent_updated_at, expected_challenger_updated_at, review_id),
            )
            await self.db.commit()

    def _row_to_review(self, row) -> MemoryReview:
        d = dict(row)
        return MemoryReview(
            id=d["id"],
            kind=d["kind"],
            status=d["status"],
            incumbent_memory_id=d["incumbent_memory_id"],
            challenger_memory_id=d["challenger_memory_id"],
            reason=d.get("reason"),
            review_note=d.get("review_note"),
            reviewer=d.get("reviewer"),
            expected_incumbent_updated_at=d.get("expected_incumbent_updated_at"),
            expected_challenger_updated_at=d.get("expected_challenger_updated_at"),
            replacement_kind=_validate_replacement_kind(d.get("replacement_kind") or "supersession"),
            created_at=_parse_dt(d.get("created_at")),
            resolved_at=_parse_dt(d.get("resolved_at")),
        )

    def _row_to_audit_event(self, row) -> MemoryAuditEvent:
        d = dict(row)

        def load_json(key: str, fallback):
            raw = d.get(key)
            if raw is None:
                return fallback
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return fallback

        return MemoryAuditEvent(
            event_id=d["event_id"],
            operation_id=d["operation_id"],
            parent_event_id=d.get("parent_event_id"),
            occurred_at=_parse_dt(d.get("occurred_at")),
            actor_type=d.get("actor_type"),
            actor_id=d.get("actor_id"),
            run_id=d.get("run_id"),
            trace_id=d.get("trace_id"),
            source_id=d.get("source_id"),
            doc_id=d.get("doc_id"),
            memory_id=d.get("memory_id"),
            candidate_id=d.get("candidate_id"),
            review_id=d.get("review_id"),
            support_kind=d.get("support_kind"),
            event_type=d["event_type"],
            decision=d.get("decision"),
            reason=d.get("reason"),
            payload_class=d.get("payload_class"),
            before_snapshot=load_json("before_snapshot", None),
            after_snapshot=load_json("after_snapshot", None),
            evidence_refs=load_json("evidence_refs", []),
            model=d.get("model"),
            prompt_hash=d.get("prompt_hash"),
            config_hash=d.get("config_hash"),
            thresholds=load_json("thresholds", None),
            status=d["status"],
            payload=load_json("payload", {}),
            error=d.get("error"),
        )

    # ==================================================================
    # Config - LLM
    # ==================================================================

    async def get_llm_config(self) -> dict:
        async with self.db.execute("SELECT * FROM llm_config WHERE id = 1") as cursor:
            row = await cursor.fetchone()
            if not row:
                return {
                    "enrichment_model": None,
                    "enrichment_base_url": None,
                    "enrichment_api_key": None,
                    "embedding_model": None,
                    "embedding_base_url": None,
                    "embedding_api_key": None,
                }
            d = dict(row)
            return {
                "enrichment_model": d["enrichment_model"],
                "enrichment_base_url": d["enrichment_base_url"],
                "enrichment_api_key": d["enrichment_api_key"],
                "embedding_model": d["embedding_model"],
                "embedding_base_url": d["embedding_base_url"],
                "embedding_api_key": d["embedding_api_key"],
            }

    async def set_llm_config(self, config: dict) -> None:
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO llm_config (
                    id, enrichment_model, enrichment_base_url, enrichment_api_key,
                    embedding_model, embedding_base_url, embedding_api_key
                ) VALUES (1, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    enrichment_model=excluded.enrichment_model,
                    enrichment_base_url=excluded.enrichment_base_url,
                    enrichment_api_key=excluded.enrichment_api_key,
                    embedding_model=excluded.embedding_model,
                    embedding_base_url=excluded.embedding_base_url,
                    embedding_api_key=excluded.embedding_api_key""",
                (
                    config.get("enrichment_model"),
                    config.get("enrichment_base_url"),
                    config.get("enrichment_api_key"),
                    config.get("embedding_model"),
                    config.get("embedding_base_url"),
                    config.get("embedding_api_key"),
                ),
            )
            await self.db.commit()

    # ==================================================================
    # Users
    # ==================================================================

    async def create_user(
        self,
        username: str,
        display_name: str | None,
        password_hash: str,
        role: str = "viewer",
    ) -> int:
        async with self._write_lock:
            cursor = await self.db.execute(
                """INSERT INTO users (username, display_name, password_hash, role)
                   VALUES (?, ?, ?, ?)""",
                (username, display_name, password_hash, role),
            )
            await self.db.commit()
            return cursor.lastrowid or 0

    async def get_user_by_username(self, username: str) -> dict | None:
        async with self.db.execute("SELECT * FROM users WHERE username = ?", (username,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            d["created_at"] = _parse_dt(d.get("created_at"))
            d["last_login"] = _parse_dt(d.get("last_login"))
            return d

    async def list_users(self) -> list[dict]:
        results: list[dict] = []
        async with self.db.execute("SELECT * FROM users ORDER BY id") as cursor:
            async for row in cursor:
                d = dict(row)
                d["created_at"] = _parse_dt(d.get("created_at"))
                d["last_login"] = _parse_dt(d.get("last_login"))
                results.append(d)
        return results

    async def delete_user(self, user_id: int) -> None:
        async with self._write_lock:
            await self.db.execute("DELETE FROM users WHERE id = ?", (user_id,))
            await self.db.commit()

    # ==================================================================
    # Private helpers
    # ==================================================================

    def _row_to_document(self, row) -> DocumentRecord:
        d = dict(row)
        return DocumentRecord(
            doc_id=d["doc_id"],
            source=d["source"],
            source_url=d["source_url"],
            title=d["title"],
            space_or_project=d["space_or_project"],
            author=d["author"],
            last_modified=datetime.fromisoformat(d["last_modified"]),
            labels=json.loads(d.get("labels") or "[]"),
            version=d["version"],
            content_hash=d["content_hash"],
            token_count=d["token_count"],
            raw_content_uri=d["raw_content_uri"],
            raw_content_type=d["raw_content_type"],
            normalized_content_uri=d["normalized_content_uri"],
            pdf_content_uri=d.get("pdf_content_uri"),
            last_synced=datetime.fromisoformat(d["last_synced"]),
            client=d.get("client"),
            created_at=_parse_dt(d.get("created_at")),
            updated_at=_parse_dt(d.get("updated_at")),
        )

    def _row_to_memory(self, row) -> Memory:
        d = dict(row)
        return Memory(
            id=d["id"],
            memory_type=d["memory_type"],
            content=d["content"],
            content_hash=d["content_hash"],
            tags=json.loads(d.get("tags") or "[]"),
            visibility=d["visibility"],
            owner_user_id=d["owner_user_id"],
            project_key=d["project_key"],
            repo_identifier=d.get("repo_identifier"),
            confidence=d["confidence"],
            corroboration_count=d["corroboration_count"],
            contradiction_count=d["contradiction_count"],
            valid_from=_parse_date(d.get("valid_from")),
            valid_until=_parse_date(d.get("valid_until")),
            superseded_by=d.get("superseded_by"),
            status=d["status"],
            retirement_reason=d.get("retirement_reason"),
            retired_at=_parse_dt(d.get("retired_at")),
            superseded_at=_parse_dt(d.get("superseded_at")),
            replacement_reason=d.get("replacement_reason"),
            replacement_kind=d.get("replacement_kind"),
            extraction_context=d.get("extraction_context"),
            created_at=_parse_dt(d.get("created_at")),
            updated_at=_parse_dt(d.get("updated_at")),
        )

    def _row_to_candidate_memory(self, row) -> CandidateMemory:
        d = dict(row)
        try:
            source_metadata = json.loads(d.get("source_metadata_json") or "{}")
        except (TypeError, json.JSONDecodeError):
            source_metadata = {}
        return CandidateMemory(
            memory_id=d["memory_id"],
            source_id=d.get("source_id"),
            doc_id=d.get("doc_id"),
            source_lineage_id=d.get("source_lineage_id"),
            visibility=d["visibility"],
            owner_user_id=d.get("owner_user_id"),
            repo_identifier=d.get("repo_identifier"),
            doc_revision_id=d.get("doc_revision_id"),
            source_anchor=d.get("source_anchor"),
            source_metadata=source_metadata,
        )

    def _row_to_projection_scope_transition(self, row) -> ProjectionScopeTransition:
        d = dict(row)
        coverage = d.get("coverage")
        return ProjectionScopeTransition(
            id=d["id"],
            source_id=d["source_id"],
            previous_scope=json.loads(d["previous_scope_json"]),
            target_scope=json.loads(d["target_scope_json"]),
            status=ProjectionScopeTransitionStatus(d["status"]),
            run_id=d.get("run_id"),
            coverage=ProjectionCoverage(coverage) if coverage else None,
            error=d.get("error"),
            created_at=d.get("created_at"),
            updated_at=d.get("updated_at"),
            completed_at=d.get("completed_at"),
        )

    def _row_to_agent_session_receipt(self, row) -> dict:
        d = dict(row)
        try:
            metadata = json.loads(d.get("metadata") or "{}")
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        d["metadata"] = metadata
        return d

    def _row_to_agent_hook_receipt(self, row) -> dict:
        d = dict(row)
        try:
            metadata = json.loads(d.get("metadata") or "{}")
        except (json.JSONDecodeError, TypeError):
            metadata = {}
        d["metadata"] = metadata
        return d
