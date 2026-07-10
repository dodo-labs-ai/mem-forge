"""SQLite database for documents, memories, entities, sync state, and configuration.

Mirrors the schema from architecture.md Section 9. Uses aiosqlite with WAL mode.
FK enforcement is OFF — all cascades are implemented manually in delete methods.
FTS5 rows must be manually synced on insert/update/delete of memories.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import aiosqlite

from memforge.models import (
    AgentHookReceipt,
    AgentSessionReceipt,
    DocumentMetadata,
    DocumentRecord,
    Entity,
    EntityAlias,
    Memory,
    MemoryCurationRun,
    MemoryDerivation,
    MemoryReview,
    MemoryReviewRelatedChallenger,
    MemorySource,
    Project,
    ReplacementKind,
    SHARED_PROJECT_KEY,
    SourceSyncInput,
    SourceSyncRun,
    SyncState,
    UNSORTED_PROJECT_KEY,
    Visibility,
    canonicalize_entity_name,
)
from memforge.memory.evidence import (
    AuthorityCase,
    CandidateBucket,
    CandidateMemory,
    CandidatePage,
    EvidenceContentProvenance,
    EvidenceRelationRecord,
    EvidenceUnit,
    LifecycleAction,
    RelationCandidateRecord,
    RelationOutcomeBundle,
    RelationRunRecord,
    RelationType,
    ReviewCase,
    evidence_relation_retry_identity,
    relation_bundle_snapshot_audit,
    relation_candidate_retry_identity,
)
from memforge.memory.audit import MemoryAuditEvent
from memforge.memory.lifecycle import allowed_search_statuses, normalize_memory_status
from memforge.retrieval.access_predicate import visible_sql
from memforge.retrieval.metadata_text import metadata_alias_text, metadata_compact_text
from memforge.storage.admin_memory import (
    MemoryAdminListFilters,
    MemoryAdminQueryPage,
)
from memforge.storage.admin_source import (
    SOURCE_SYNC_SCHEDULE_DEFAULT_INTERVAL_MINUTES,
    SOURCE_SYNC_SCHEDULE_MAX_INTERVAL_MINUTES,
    SOURCE_SYNC_SCHEDULE_MIN_INTERVAL_MINUTES,
)

logger = logging.getLogger(__name__)

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
AGENT_SESSION_WINDOW_SOURCE_KIND = "generated_agent_window_summary"
_PERSISTED_RELATION_TYPES = {
    RelationType.SUPPORTS,
    RelationType.EQUIVALENT,
    RelationType.REFINES,
    RelationType.CONTRADICTS,
}
_RELATION_SNAPSHOT_AUDIT_KEYS = frozenset(
    {
        "candidate_snapshot_hash",
        "relation_snapshot_hash",
    }
)


def _validate_persisted_evidence_relation(relation: EvidenceRelationRecord) -> None:
    if relation.relation_type not in _PERSISTED_RELATION_TYPES:
        raise ValueError(f"relation_type {relation.relation_type.value!r} is not a persisted evidence relation")


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
        coalesced=coalesced,
        lease_owner=data.get("lease_owner"),
        lease_expires_at=_parse_dt(data.get("lease_expires_at")),
        lease_attempt_count=int(data.get("lease_attempt_count") or 0),
        recovery_count=int(data.get("recovery_count") or 0),
        rerun_requested=bool(data.get("rerun_requested")),
        next_attempt_at=_parse_dt(data.get("next_attempt_at")),
        error_message=data.get("error_message"),
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
    memory_level        TEXT NOT NULL DEFAULT 'atomic',
    curation_cluster_id TEXT,
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

CREATE TABLE IF NOT EXISTS memory_derivations (
    parent_memory_id TEXT NOT NULL REFERENCES memories(id),
    child_memory_id  TEXT NOT NULL REFERENCES memories(id),
    relation         TEXT NOT NULL DEFAULT 'summarizes',
    created_at       TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (parent_memory_id, child_memory_id, relation)
);

CREATE TABLE IF NOT EXISTS memory_curation_runs (
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
    CHECK (relation_type IN ('supports','equivalent','refines','contradicts'))
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
    CHECK (relation_type IN ('supports','equivalent','refines','contradicts'))
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
    execution_owner_user_id TEXT,
    sync_schedule_enabled INTEGER NOT NULL DEFAULT 0,
    sync_schedule_interval_minutes INTEGER NOT NULL DEFAULT 1440,
    sync_schedule_next_at TEXT,
    sync_schedule_updated_at TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS source_subscriptions (
    source_id   TEXT NOT NULL REFERENCES sources(id) ON DELETE CASCADE,
    user_id     TEXT NOT NULL,
    enabled     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (source_id, user_id)
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
    CHECK (visibility = 'private')
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
-- Indexes for columns added after the initial schema, including visibility and
-- curation metadata, are created by their migrations. SCHEMA runs before
-- migrations, so upgrading databases may not have those columns here yet.
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memory_sources_doc ON memory_sources(doc_id);
CREATE INDEX IF NOT EXISTS idx_memory_derivations_child ON memory_derivations(child_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_curation_runs_scope ON memory_curation_runs(source_type, client, repo_identifier, project_key);
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
        ],
    ),
]


# ---------------------------------------------------------------------------
# Database class
# ---------------------------------------------------------------------------


class Database:
    """Async SQLite database layer for MemForge."""

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._write_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    async def connect(self) -> None:
        """Open the database, enable WAL mode, create schema, and run migrations."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA foreign_keys = ON")
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
            if version == 26:
                await self._backfill_relation_run_snapshot_audit()
            if version in (30, 31):
                await self._rebuild_memory_metadata_fts_unlocked()
            await self.db.execute(
                "INSERT INTO schema_migrations (version, description, applied_at) VALUES (?, ?, ?)",
                (version, description, _now_iso()),
            )
            await self.db.commit()
            logger.info("Applied migration %d: %s", version, description)
        await self._assert_memory_source_ids_resolved()

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

    async def upsert_document(self, doc: DocumentRecord) -> None:
        async with self._write_lock:
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
            await self.db.commit()

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        async with self.db.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            return self._row_to_document(row)

    async def restore_document_snapshot(self, doc: DocumentRecord) -> None:
        """Restore one document row from a captured snapshot."""
        async with self._write_lock:
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

    async def delete_document(self, doc_id: str) -> list[str]:
        """Delete a document and manually cascade to related tables.

        Returns memory IDs retired because this document was their last valid
        source support.
        """
        async with self._write_lock:
            try:
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
    # Memories
    # ==================================================================

    async def upsert_evidence_unit(self, unit: EvidenceUnit) -> None:
        """Persist one scoped evidence item before relation classification."""
        metadata_json = json.dumps(dict(unit.source_metadata), sort_keys=True)
        provenance = unit.evidence_provenance.value
        async with self._write_lock:
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
            await self.db.commit()

    async def get_evidence_unit(self, evidence_unit_id: str) -> EvidenceUnit | None:
        async with self.db.execute(
            "SELECT * FROM evidence_units WHERE id = ?",
            (evidence_unit_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
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
                    _validate_persisted_evidence_relation(relation)
                    await self.db.execute(
                        """INSERT INTO evidence_relations (
                            evidence_unit_id, memory_id, relation_type, authority_case,
                            is_authoritative_support, source_lineage_id, confidence,
                            reason, proposed_memory_content, excerpt, classifier_version,
                            relation_run_id, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            evidence_unit_id,
                            relation.memory_id,
                            relation.relation_type.value,
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
        _validate_persisted_evidence_relation(relation)
        async with self._write_lock:
            try:
                await self.db.execute(
                    """INSERT OR IGNORE INTO evidence_relations (
                        evidence_unit_id, memory_id, relation_type, authority_case,
                        is_authoritative_support, source_lineage_id, confidence,
                        reason, proposed_memory_content, excerpt, classifier_version,
                        relation_run_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        relation.evidence_unit_id,
                        relation.memory_id,
                        relation.relation_type.value,
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

    async def _record_relation_outcome_bundle_unlocked(self, bundle: RelationOutcomeBundle) -> None:
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
            _validate_persisted_evidence_relation(relation)
            await self.db.execute(
                """INSERT INTO relation_run_relations (
                            relation_run_id, evidence_unit_id, memory_id, relation_type,
                            authority_case, is_authoritative_support, source_lineage_id,
                            confidence, reason, proposed_memory_content, excerpt,
                            classifier_version, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    relation.relation_run_id,
                    relation.evidence_unit_id,
                    relation.memory_id,
                    relation.relation_type.value,
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
                            evidence_unit_id, memory_id, relation_type, authority_case,
                            is_authoritative_support, source_lineage_id, confidence,
                            reason, proposed_memory_content, excerpt, classifier_version,
                            relation_run_id, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    relation.evidence_unit_id,
                    relation.memory_id,
                    relation.relation_type.value,
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
                project_key, repo_identifier, memory_level, curation_cluster_id,
                confidence, corroboration_count,
                contradiction_count, valid_from, valid_until,
                superseded_by, status, retirement_reason, retired_at,
                superseded_at, replacement_reason, replacement_kind, extraction_context,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                mem.memory_level,
                mem.curation_cluster_id,
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
                project_key, repo_identifier, memory_level, curation_cluster_id,
                confidence, corroboration_count,
                contradiction_count, valid_from, valid_until,
                superseded_by, status, retirement_reason, retired_at,
                superseded_at, replacement_reason, replacement_kind, extraction_context,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                memory_type=excluded.memory_type,
                content=excluded.content,
                content_hash=excluded.content_hash,
                tags=excluded.tags,
                visibility=excluded.visibility,
                owner_user_id=excluded.owner_user_id,
                project_key=excluded.project_key,
                repo_identifier=excluded.repo_identifier,
                memory_level=excluded.memory_level,
                curation_cluster_id=excluded.curation_cluster_id,
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
                mem.memory_level,
                mem.curation_cluster_id,
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

            async with self.db.execute("SELECT confidence, tags FROM memories WHERE id = ?", (memory_id,)) as cursor:
                row = await cursor.fetchone()
            if not row:
                return

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
                    superseded_by = NULL, status = 'retired',
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
                "DELETE FROM memory_derivations WHERE parent_memory_id = ? OR child_memory_id = ?",
                (memory_id, memory_id),
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
                    repo_identifier = ?, memory_level = ?, curation_cluster_id = ?,
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
                    memory.memory_level,
                    memory.curation_cluster_id,
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
        await self._rebuild_memory_fts_unlocked(
            new_memory.id,
            search_visible_statuses=set(allowed_search_statuses()),
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
            cursor = await self.db.execute(
                """UPDATE memories SET
                    status = 'retired', retirement_reason = 'expired',
                    retired_at = COALESCE(retired_at, ?), updated_at = ?
                   WHERE status = 'active'
                   AND valid_until IS NOT NULL
                   AND valid_until < ?""",
                (now, now, today),
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
                " ".join(
                    part
                    for part in (record["space_or_project"] or "", record["source_url"] or "")
                    if part
                ),
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
        rows = await self.db.execute_fetchall(
            "SELECT memory_id, doc_id FROM memory_sources ORDER BY memory_id, doc_id"
        )
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
        await self.db.execute(
            """INSERT INTO agent_concepts (
                id, source_id, owner_user_id, visibility, workspace,
                repo_identifier, concept_type, concept_path, title,
                markdown_body, frontmatter_json, created_at, updated_at,
                last_observed_at
            ) VALUES (?, ?, ?, 'private', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                source_id=excluded.source_id,
                owner_user_id=excluded.owner_user_id,
                visibility='private',
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
        owner_user_id: str,
        repo_identifier: str | None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        concepts: list[dict[str, Any]] = []
        async with self.db.execute(
            """SELECT * FROM agent_concepts
               WHERE visibility = 'private'
                 AND owner_user_id = ?
                 AND COALESCE(repo_identifier, '') = COALESCE(?, '')
               ORDER BY updated_at DESC, id
               LIMIT ?""",
            (owner_user_id, repo_identifier, limit),
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
                    project_key, repo_identifier, memory_level, curation_cluster_id,
                    confidence, corroboration_count,
                    contradiction_count, valid_from, valid_until,
                    superseded_by, status, retirement_reason, retired_at,
                    superseded_at, replacement_reason, replacement_kind, extraction_context,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
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
                        new_memory.memory_level,
                        new_memory.curation_cluster_id,
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

    async def add_memory_derivation(
        self,
        parent_memory_id: str,
        child_memory_id: str,
        *,
        relation: str = "summarizes",
    ) -> None:
        async with self._write_lock:
            await self.db.execute(
                """INSERT OR IGNORE INTO memory_derivations (
                    parent_memory_id, child_memory_id, relation, created_at
                ) VALUES (?, ?, ?, ?)""",
                (parent_memory_id, child_memory_id, relation, _now_iso()),
            )
            await self.db.commit()

    async def get_memory_derivation_children(
        self,
        parent_memory_id: str,
    ) -> list[MemoryDerivation]:
        results: list[MemoryDerivation] = []
        async with self.db.execute(
            """SELECT parent_memory_id, child_memory_id, relation, created_at
               FROM memory_derivations
               WHERE parent_memory_id = ?
               ORDER BY created_at, child_memory_id""",
            (parent_memory_id,),
        ) as cursor:
            async for row in cursor:
                results.append(
                    MemoryDerivation(
                        parent_memory_id=row["parent_memory_id"],
                        child_memory_id=row["child_memory_id"],
                        relation=row["relation"],
                        created_at=_parse_dt(row["created_at"]),
                    )
                )
        return results

    async def record_memory_curation_run(self, run: MemoryCurationRun) -> None:
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO memory_curation_runs (
                    id, policy_id, source_type, client, repo_identifier,
                    project_key, candidate_count, created_memory_count,
                    skipped_reason, error, started_at, completed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    policy_id=excluded.policy_id,
                    source_type=excluded.source_type,
                    client=excluded.client,
                    repo_identifier=excluded.repo_identifier,
                    project_key=excluded.project_key,
                    candidate_count=excluded.candidate_count,
                    created_memory_count=excluded.created_memory_count,
                    skipped_reason=excluded.skipped_reason,
                    error=excluded.error,
                    started_at=excluded.started_at,
                    completed_at=excluded.completed_at""",
                (
                    run.id,
                    run.policy_id,
                    run.source_type,
                    run.client,
                    run.repo_identifier,
                    run.project_key,
                    run.candidate_count,
                    run.created_memory_count,
                    run.skipped_reason,
                    run.error,
                    _utc_iso(run.started_at),
                    _utc_iso(run.completed_at) if run.completed_at else None,
                ),
            )
            await self.db.commit()

    async def get_memory_curation_run(self, run_id: str) -> MemoryCurationRun | None:
        async with self.db.execute(
            "SELECT * FROM memory_curation_runs WHERE id = ?",
            (run_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_memory_curation_run(row)

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
        else:
            await self.db.execute(
                """UPDATE memories SET
                    corroboration_count = ?, updated_at = ?
                   WHERE id = ?""",
                (total_count, now, memory_id),
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
            await self.db.execute(
                """INSERT INTO entities (canonical_name, entity_type, tags, display_name)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(canonical_name) DO UPDATE SET
                   entity_type=excluded.entity_type,
                   tags=excluded.tags,
                   display_name=excluded.display_name""",
                (canonical_name, entity_type, tags_json, display_name),
            )
            async with self.db.execute("SELECT id FROM entities WHERE canonical_name = ?", (canonical_name,)) as cursor:
                row = await cursor.fetchone()
                assert row is not None
                entity_id = int(row[0])
            await self._refresh_entity_alias_search_unlocked(entity_id)
            await self.db.commit()
            return entity_id

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
                alias_text = " ".join(
                    part
                    for part in (row["alias"] or "", row["alias_normalized"] or "")
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
        source = await self.get_entity(source_id)
        if source is None:
            raise LookupError("Source entity not found")
        target = await self.get_entity(target_id)
        if target is None:
            raise LookupError("Target entity not found")

        async with self._write_lock:
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

        return {
            "source_id": source_id,
            "source_name": source.canonical_name,
            "target_id": target_id,
            "target_name": target.canonical_name,
        }

    async def remove_entity_alias(self, *, entity_id: int, alias_normalized: str) -> bool:
        async with self._write_lock:
            result = await self.db.execute(
                "DELETE FROM entity_aliases WHERE alias_normalized = ? AND canonical_id = ?",
                (alias_normalized, entity_id),
            )
            if result.rowcount > 0:
                await self._refresh_entity_alias_search_unlocked(entity_id)
            await self.db.commit()
            return result.rowcount > 0

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
        status: str | None = None,
        project_binding: Mapping[str, Any] | None = None,
        created_by_user_id: str | None = None,
        execution_owner_user_id: str | None = None,
    ) -> None:
        """Insert or update a source row.

        `project_binding` is the structured rule the project resolver
        consults when memories are extracted from this source. `None`
        leaves the source unbound and resolves writes to `UNSORTED`.
        """
        binding_json = json.dumps(dict(project_binding)) if project_binding else None
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO sources (
                       id, type, name, config, status, project_binding,
                       created_by_user_id, execution_owner_user_id
                   )
                   VALUES (?, ?, ?, ?, COALESCE(?, 'active'), ?, ?, ?)
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
                    execution_owner_user_id,
                    status,
                ),
            )
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

    async def restore_source_snapshot(self, source: dict) -> None:
        """Restore one source row from a captured snapshot."""
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO sources
                   (id, type, name, config, status, last_sync, doc_count, project_binding,
                    created_by_user_id, execution_owner_user_id, sync_schedule_enabled,
                    sync_schedule_interval_minutes, sync_schedule_next_at,
                    sync_schedule_updated_at, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                   type=excluded.type,
                   name=excluded.name,
                   config=excluded.config,
                   status=excluded.status,
                   last_sync=excluded.last_sync,
                   doc_count=excluded.doc_count,
                   project_binding=excluded.project_binding,
                   created_by_user_id=excluded.created_by_user_id,
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
                row["sync_schedule_interval_minutes"]
                or SOURCE_SYNC_SCHEDULE_DEFAULT_INTERVAL_MINUTES
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

    async def delete_source_cascade(self, source_id: str) -> list[str]:
        """Delete a source and cascade to all documents + memories linked to those docs.

        Returns memory IDs retired because the source removal left them without
        valid support.
        """
        async with self._write_lock:
            try:
                retired_ids: list[str] = []
                doc_ids: list[str] = []
                async with self.db.execute("SELECT doc_id FROM documents WHERE source = ?", (source_id,)) as cursor:
                    async for row in cursor:
                        doc_ids.append(row[0])

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
                await self.db.execute("DELETE FROM sources WHERE id = ?", (source_id,))
                await self.db.commit()
                return list(dict.fromkeys(retired_ids))
            except Exception:
                await self.db.rollback()
                raise

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
        async with self._write_lock:
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
            await self.db.commit()

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
    ) -> SourceSyncRun:
        async with self._write_lock:
            try:
                run = await self._enqueue_source_sync_run_locked(
                    source_id=source_id,
                    workspace_id=workspace_id,
                    trigger=trigger,
                    force_full_sync=force_full_sync,
                )
                await self.db.commit()
            except Exception:
                await self.db.rollback()
                raise
        return run

    async def _enqueue_source_sync_run_locked(
        self,
        *,
        source_id: str,
        workspace_id: str = "default",
        trigger: str = "manual",
        force_full_sync: bool = False,
        now: str | None = None,
    ) -> SourceSyncRun:
        now_iso = now or _now_iso()
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
            mark_rerun = existing["status"] == "running"
            await self.db.execute(
                """UPDATE source_sync_runs
                   SET force_full_sync = CASE WHEN ? THEN 1 ELSE force_full_sync END,
                       rerun_requested = CASE WHEN ? THEN 1 ELSE rerun_requested END,
                       updated_at = ?
                   WHERE run_id = ?""",
                (
                    int(force_full_sync),
                    int(mark_rerun),
                    now_iso,
                    existing["run_id"],
                ),
            )
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
                force_full_sync, created_at, updated_at
            ) VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)""",
            (
                run_id,
                workspace_id,
                source_id,
                trigger,
                int(force_full_sync),
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
            "OR (status = 'running' AND lease_expires_at IS NOT NULL AND lease_expires_at <= ?))"
        ]
        params: list[Any] = [lease_started_iso, lease_started_iso]
        if workspace_id is not None:
            conditions.append("workspace_id = ?")
            params.append(workspace_id)

        async with self._write_lock:
            async with self.db.execute(
                "SELECT * FROM source_sync_runs WHERE "
                + " AND ".join(conditions)
                + " ORDER BY created_at LIMIT 1",
                params,
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                return None

            recovery_increment = 1 if row["status"] == "running" else 0
            await self.db.execute(
                """UPDATE source_sync_runs
                   SET status = 'running',
                       lease_owner = ?,
                       lease_expires_at = ?,
                       lease_attempt_count = lease_attempt_count + 1,
                       recovery_count = recovery_count + ?,
                       next_attempt_at = NULL,
                       started_at = COALESCE(started_at, ?),
                       updated_at = ?
                   WHERE run_id = ?""",
                (
                    worker_id,
                    lease_expires_at,
                    recovery_increment,
                    lease_started_iso,
                    lease_started_iso,
                    row["run_id"],
                ),
            )
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
        status = final_state.last_sync_status if final_state and final_state.last_sync_status else "success"
        if status not in {"success", "failed"}:
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
        failed_iso = _utc_iso(failed_at)
        status = "pending" if retryable else "failed"
        completed_at = None if retryable else failed_iso
        next_attempt_iso = _utc_iso(next_attempt_at) if retryable and next_attempt_at else None
        async with self._write_lock:
            cursor = await self.db.execute(
                """UPDATE source_sync_runs
                   SET status = ?,
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
                force_full_sync, created_at, updated_at
            ) VALUES (?, ?, ?, 'rerun', 'pending', ?, ?, ?)""",
            (
                successor_id,
                completed["workspace_id"],
                completed["source_id"],
                int(completed["force_full_sync"]),
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
    ) -> SourceSyncInput:
        now = _now_iso()
        async with self._write_lock:
            async with self.db.execute(
                """SELECT * FROM source_sync_inputs
                   WHERE workspace_id = ? AND source_id = ? AND raw_sha256 = ?""",
                (workspace_id, source_id, raw_sha256),
            ) as cursor:
                existing = await cursor.fetchone()
            if existing is not None:
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
                    json.dumps(metadata or {}, sort_keys=True),
                    now,
                ),
            )
            await self.db.commit()
            async with self.db.execute(
                "SELECT * FROM source_sync_inputs WHERE input_id = ?",
                (input_id,),
            ) as cursor:
                inserted = await cursor.fetchone()
        assert inserted is not None
        return _source_sync_input_from_row(inserted)

    async def list_source_sync_inputs(
        self,
        *,
        source_id: str,
        workspace_id: str = "default",
    ) -> list[SourceSyncInput]:
        results: list[SourceSyncInput] = []
        async with self.db.execute(
            """SELECT * FROM source_sync_inputs
               WHERE workspace_id = ? AND source_id = ?
               ORDER BY input_generation""",
            (workspace_id, source_id),
        ) as cursor:
            async for row in cursor:
                results.append(_source_sync_input_from_row(row))
        return results

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

    async def get_cross_doc_candidates(
        self,
        memory_id: str,
        entity_ids: list[int],
        doc_id: str,
        *,
        owner_user_id: str | None = None,
        visibility: str | None = None,
        project_key: str | None = None,
        excluded_source_ids: Sequence[str] = (),
        limit: int = 200,
    ) -> CandidatePage[Memory]:
        """Find active memories sharing entities with this memory but from different documents."""
        if not entity_ids:
            return CandidatePage(candidates=(), complete=True, requested_limit=limit)
        if limit < 1:
            return CandidatePage(candidates=(), complete=True, requested_limit=limit)

        placeholders = ",".join("?" for _ in entity_ids)
        visibility_clause = "AND m.visibility != 'private'"
        visibility_params: list[str] = []
        if visibility == "private" and owner_user_id:
            visibility_clause = "AND (m.visibility != 'private' OR m.owner_user_id = ?)"
            visibility_params.append(owner_user_id)
        scope_clause = ""
        scope_params: list[Any] = []
        if project_key:
            scope_clause += " AND m.project_key = ?"
            scope_params.append(project_key)
        if excluded_source_ids:
            source_placeholders = ",".join("?" for _ in excluded_source_ids)
            scope_clause += f"""
              AND (
                  NOT EXISTS (
                      SELECT 1 FROM memory_sources ms_any
                      WHERE ms_any.memory_id = m.id
                  )
                  OR EXISTS (
                      SELECT 1 FROM memory_sources ms_enabled
                      WHERE ms_enabled.memory_id = m.id
                        AND (ms_enabled.source_id IS NULL OR ms_enabled.source_id NOT IN ({source_placeholders}))
                  )
              )"""
            scope_params.extend(excluded_source_ids)
        sql = f"""
            SELECT DISTINCT m.* FROM memories m
            JOIN memory_entities me ON m.id = me.memory_id
            JOIN memory_sources ms ON m.id = ms.memory_id
            WHERE me.entity_id IN ({placeholders})
              AND ms.doc_id != ?
              AND m.id != ?
              AND m.status = 'active'
              {visibility_clause}
              {scope_clause}
            ORDER BY m.updated_at DESC, m.id
            LIMIT ?
        """
        params = [
            *entity_ids,
            doc_id,
            memory_id,
            *visibility_params,
            *scope_params,
            limit + 1,
        ]
        results: list[Memory] = []
        async with self.db.execute(sql, params) as cursor:
            async for row in cursor:
                results.append(self._row_to_memory(row))
        return CandidatePage(
            candidates=tuple(results[:limit]),
            complete=len(results) <= limit,
            requested_limit=limit,
        )

    async def record_contradiction(
        self,
        memory_id_a: str,
        memory_id_b: str,
        classification: str,
        reason: str | None = None,
    ) -> None:
        """Record a contradiction between two memories and increment their counts."""
        async with self._write_lock:
            await self.db.execute(
                """INSERT OR IGNORE INTO memory_contradictions
                   (memory_id_a, memory_id_b, classification, reason)
                   VALUES (?, ?, ?, ?)""",
                (memory_id_a, memory_id_b, classification, reason),
            )
            if classification == "contradiction":
                await self.db.execute(
                    "UPDATE memories SET contradiction_count = contradiction_count + 1 WHERE id IN (?, ?)",
                    (memory_id_a, memory_id_b),
                )
            await self.db.commit()

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
            memory_level=d.get("memory_level") or "atomic",
            curation_cluster_id=d.get("curation_cluster_id"),
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

    def _row_to_memory_curation_run(self, row) -> MemoryCurationRun:
        d = dict(row)
        started_at = _parse_dt(d["started_at"])
        if started_at is None:
            started_at = datetime.now(timezone.utc)
        return MemoryCurationRun(
            id=d["id"],
            policy_id=d["policy_id"],
            source_type=d["source_type"],
            client=d.get("client"),
            repo_identifier=d.get("repo_identifier"),
            project_key=d.get("project_key"),
            candidate_count=d["candidate_count"],
            created_memory_count=d["created_memory_count"],
            skipped_reason=d.get("skipped_reason"),
            error=d.get("error"),
            started_at=started_at,
            completed_at=_parse_dt(d.get("completed_at")),
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
