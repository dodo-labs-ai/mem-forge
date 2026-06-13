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
from datetime import datetime, timezone
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
    MemoryReview,
    MemoryReviewRelatedChallenger,
    MemorySource,
    Project,
    SHARED_PROJECT_KEY,
    SyncState,
    UNSORTED_PROJECT_KEY,
    Visibility,
    canonicalize_entity_name,
)
from memforge.memory.audit import MemoryAuditEvent
from memforge.memory.lifecycle import allowed_search_statuses, normalize_memory_status

logger = logging.getLogger(__name__)

# The three real outcomes an uploaded agent-session window can record. Knowledge
# completeness ("how much was kept vs dropped as no_output") is read from these.
AGENT_SESSION_OUTCOME_PACKAGE_CREATED = "package_created"
AGENT_SESSION_OUTCOME_NO_OUTPUT = "no_output"
AGENT_SESSION_OUTCOME_FAILED = "failed"
AGENT_SESSION_OUTCOMES = (
    AGENT_SESSION_OUTCOME_PACKAGE_CREATED,
    AGENT_SESSION_OUTCOME_NO_OUTPUT,
    AGENT_SESSION_OUTCOME_FAILED,
)
AGENT_SESSION_WINDOW_SOURCE_KIND = "generated_agent_window_summary"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _utc_iso(dt: datetime | None) -> str:
    value = dt or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s)


_VALID_VISIBILITIES = frozenset({Visibility.WORKSPACE.value, Visibility.PRIVATE.value})


def _validate_visibility(visibility: str, owner_user_id: str | None) -> None:
    """Enforce the owner/visibility invariant before any memory write."""
    if visibility not in _VALID_VISIBILITIES:
        raise ValueError(
            f"visibility must be one of {sorted(_VALID_VISIBILITIES)}, got {visibility!r}"
        )
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
    extraction_context  TEXT,
    created_at          TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at          TEXT NOT NULL DEFAULT (datetime('now')),
    CHECK (visibility IN ('private','workspace')),
    CHECK ((visibility = 'private') = (owner_user_id IS NOT NULL))
);

CREATE TABLE IF NOT EXISTS memory_sources (
    memory_id   TEXT NOT NULL REFERENCES memories(id),
    doc_id      TEXT NOT NULL REFERENCES documents(doc_id),
    source_type TEXT NOT NULL,
    excerpt     TEXT,
    support_kind TEXT NOT NULL DEFAULT 'extracted',
    added_at    TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (memory_id, doc_id)
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
    created_at      TEXT DEFAULT (datetime('now'))
);

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
-- idx_memories_access and idx_memories_owner index the visibility and owner_user_id
-- columns and are created in migration 14, which adds those columns. SCHEMA runs
-- before migrations, so an upgrading database does not have those columns here yet.
CREATE INDEX IF NOT EXISTS idx_memories_hash ON memories(content_hash);
CREATE INDEX IF NOT EXISTS idx_memory_sources_doc ON memory_sources(doc_id);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_id);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_normalized ON entity_aliases(alias_normalized);
CREATE INDEX IF NOT EXISTS idx_auth_sessions_status ON auth_sessions(status);
CREATE INDEX IF NOT EXISTS idx_agent_session_receipts_session ON agent_session_receipts(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_session_receipts_source ON agent_session_receipts(source_id);
CREATE INDEX IF NOT EXISTS idx_agent_hook_receipts_session ON agent_hook_receipts(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_hook_receipts_hook ON agent_hook_receipts(hook);

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
    (1, "Add tags column to entities, deprecate entity_type", [
        "ALTER TABLE entities ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'",
        "UPDATE entities SET tags = json_array(entity_type) WHERE tags = '[]'",
        "DROP INDEX IF EXISTS idx_entities_type",
    ]),
    (2, "Add memory_contradictions table", [
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
    ]),
    (3, "Add lean memory lifecycle metadata", [
        "ALTER TABLE memories ADD COLUMN retirement_reason TEXT",
        "ALTER TABLE memories ADD COLUMN retired_at TEXT",
        "ALTER TABLE memories ADD COLUMN superseded_at TEXT",
        "ALTER TABLE memories ADD COLUMN replacement_reason TEXT",
        "UPDATE memories SET status = 'retired' WHERE status = 'decayed'",
    ]),
    (4, "Add agent session receipt lineage", [
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
    ]),
    (5, "Add memory_reviews table for human-gated lifecycle decisions", [
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
            created_at                      TEXT NOT NULL,
            resolved_at                     TEXT
        )""",
        "CREATE INDEX IF NOT EXISTS idx_memory_reviews_status ON memory_reviews(status)",
        "CREATE INDEX IF NOT EXISTS idx_memory_reviews_incumbent ON memory_reviews(incumbent_memory_id)",
        "CREATE INDEX IF NOT EXISTS idx_memory_reviews_challenger ON memory_reviews(challenger_memory_id)",
    ]),
    (6, "Add provenance support ownership kind", [
        "ALTER TABLE memory_sources ADD COLUMN support_kind TEXT NOT NULL DEFAULT 'extracted'",
        "CREATE INDEX IF NOT EXISTS idx_memory_sources_doc_kind ON memory_sources(doc_id, support_kind)",
    ]),
    (7, "Add memory audit event ledger", [
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
    ]),
    (8, "Add shared auth sessions", [
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
    ]),
    (9, "Add agent hook lifecycle receipts", [
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
    ]),
    (10, "Add related challengers for grouped review cases", [
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
    ]),
    (11, "Add client column and index to documents table", [
        "ALTER TABLE documents ADD COLUMN client TEXT",
        "CREATE INDEX IF NOT EXISTS idx_documents_source_client ON documents(source, client)",
    ]),
    (12, "Split singleton agent-session source into per-client sources", [
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
    ]),
    (13, "Rename agent-session sources to drop 'Summaries' and re-split any singleton remnants", [
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
    ]),
    (14, "Add visibility and owner columns to memories", [
        "ALTER TABLE memories ADD COLUMN visibility TEXT NOT NULL DEFAULT 'workspace'",
        "ALTER TABLE memories ADD COLUMN owner_user_id TEXT",
        "CREATE INDEX IF NOT EXISTS idx_memories_access ON memories(status, visibility)",
        "CREATE INDEX IF NOT EXISTS idx_memories_owner ON memories(owner_user_id)",
        "DROP INDEX IF EXISTS idx_memories_scope",
    ]),
    (15, "Backfill visibility and project_key from legacy scope", [
        "UPDATE memories SET visibility = 'workspace' WHERE visibility IS NULL OR visibility = ''",
        "UPDATE memories SET owner_user_id = NULL WHERE visibility = 'workspace'",
        "UPDATE memories SET project_key = substr(scope, 9) "
        "WHERE project_key IS NULL AND scope LIKE 'project:%'",
        "UPDATE memories SET project_key = 'SHARED' "
        "WHERE project_key IS NULL AND scope = 'team'",
        "UPDATE memories SET project_key = 'UNSORTED' WHERE project_key IS NULL",
    ]),
    (16, "Backfill NULL project_key to UNSORTED and add the projects stub table", [
        # The CREATE TABLE matches SCHEMA above; running it in a migration covers
        # any database that already passed connect() before SCHEMA carried it.
        "CREATE TABLE IF NOT EXISTS projects (project_key TEXT PRIMARY KEY)",
        f"UPDATE memories SET project_key = '{UNSORTED_PROJECT_KEY}' "
        "WHERE project_key IS NULL",
    ]),
    (17, "Replace stub projects table with full schema and seed reserved rows", [
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
    ]),
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
                    legacy_scope_backfill = (
                        version == 15
                        and "no such column" in message
                        and "scope" in sql.lower()
                    )
                    if "duplicate column" in message or legacy_scope_backfill:
                        logger.debug(
                            "Migration %d: expected-absent column on this DB, skipping: %s",
                            version, sql,
                        )
                    else:
                        raise
            await self.db.execute(
                "INSERT INTO schema_migrations (version, description, applied_at) "
                "VALUES (?, ?, ?)",
                (version, description, _now_iso()),
            )
            await self.db.commit()
            logger.info("Applied migration %d: %s", version, description)

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
                    doc.doc_id, doc.source, doc.source_url, doc.title,
                    doc.space_or_project, doc.author,
                    doc.last_modified.isoformat(),
                    json.dumps(doc.labels), doc.version, doc.content_hash,
                    doc.token_count, doc.raw_content_uri, doc.raw_content_type,
                    doc.normalized_content_uri, doc.pdf_content_uri,
                    doc.last_synced.isoformat(),
                    doc.client,
                    _now_iso(),
                ),
            )
            await self.db.commit()

    async def get_document(self, doc_id: str) -> DocumentRecord | None:
        async with self.db.execute(
            "SELECT * FROM documents WHERE doc_id = ?", (doc_id,)
        ) as cursor:
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
                    doc.doc_id, doc.source, doc.source_url, doc.title,
                    doc.space_or_project, doc.author,
                    doc.last_modified.isoformat(), json.dumps(doc.labels),
                    doc.version, doc.content_hash, doc.token_count,
                    doc.raw_content_uri, doc.raw_content_type,
                    doc.normalized_content_uri, doc.pdf_content_uri,
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
        async with self.db.execute(
            "SELECT content_hash FROM documents WHERE doc_id = ?", (doc_id,)
        ) as cursor:
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

                await self.db.execute(
                    "DELETE FROM memory_sources WHERE doc_id = ?", (doc_id,)
                )
                retired_ids = await self._refresh_support_after_source_removal_unlocked(memory_ids)
                await self.db.execute(
                    "DELETE FROM document_metadata WHERE doc_id = ?", (doc_id,)
                )
                await self.db.execute(
                    "DELETE FROM document_relationships WHERE source_doc_id = ? OR target_doc_id = ?",
                    (doc_id, doc_id),
                )
                await self.db.execute(
                    "DELETE FROM changelog WHERE doc_id = ?", (doc_id,)
                )
                await self.db.execute(
                    "DELETE FROM agent_session_receipts WHERE doc_id = ?", (doc_id,)
                )
                await self.db.execute(
                    "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
                )
                await self.db.commit()
                return retired_ids
            except Exception:
                await self.db.rollback()
                raise

    async def upsert_metadata(self, meta: DocumentMetadata) -> None:
        async with self._write_lock:
            entities_json = json.dumps(
                [{"name": e.canonical_name, "tags": e.tags} for e in meta.entities]
            )
            await self.db.execute(
                """INSERT INTO document_metadata (
                    doc_id, summary, tags, entities, doc_type, complexity, enriched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    summary=excluded.summary, tags=excluded.tags,
                    entities=excluded.entities, doc_type=excluded.doc_type,
                    complexity=excluded.complexity, enriched_at=excluded.enriched_at""",
                (
                    meta.doc_id, meta.summary, json.dumps(meta.tags),
                    entities_json, meta.doc_type, meta.complexity,
                    meta.enriched_at.isoformat() if meta.enriched_at else _now_iso(),
                ),
            )
            await self.db.commit()

    async def get_metadata(self, doc_id: str) -> DocumentMetadata | None:
        async with self.db.execute(
            "SELECT * FROM document_metadata WHERE doc_id = ?", (doc_id,)
        ) as cursor:
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

    async def insert_memory(self, mem: Memory) -> str:
        """Insert a memory and its FTS5 row. Returns the memory id."""
        _validate_visibility(mem.visibility, mem.owner_user_id)
        project_key = _normalize_project_key(mem.project_key)
        async with self._write_lock:
            now = _now_iso()
            status = normalize_memory_status(mem.status)
            await self.db.execute(
                """INSERT INTO memories (
                    id, memory_type, content, content_hash, tags, visibility, owner_user_id,
                    project_key, confidence, corroboration_count,
                    contradiction_count, valid_from, valid_until,
                    superseded_by, status, retirement_reason, retired_at,
                    superseded_at, replacement_reason, extraction_context,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    mem.id, mem.memory_type, mem.content, mem.content_hash,
                    json.dumps(mem.tags), mem.visibility, mem.owner_user_id, project_key,
                    mem.confidence, mem.corroboration_count,
                    mem.contradiction_count,
                    mem.valid_from.isoformat() if mem.valid_from else None,
                    mem.valid_until.isoformat() if mem.valid_until else None,
                    mem.superseded_by, status, mem.retirement_reason,
                    mem.retired_at.isoformat() if mem.retired_at else None,
                    mem.superseded_at.isoformat() if mem.superseded_at else None,
                    mem.replacement_reason, mem.extraction_context,
                    mem.created_at.isoformat() if mem.created_at else now,
                    mem.updated_at.isoformat() if mem.updated_at else now,
                ),
            )
            # Sync FTS5
            entities_text = " ".join(mem.entity_refs)
            tags_text = " ".join(mem.tags)
            await self.db.execute(
                "INSERT INTO memories_fts (memory_id, content, entities_text, tags_text) "
                "VALUES (?, ?, ?, ?)",
                (mem.id, mem.content, entities_text, tags_text),
            )
            await self.db.commit()
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
                "INSERT INTO memories_fts (memory_id, content, entities_text, tags_text) "
                "VALUES (?, ?, ?, ?)",
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
        async with self.db.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ) as cursor:
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
               WHERE ms.doc_id = ?""" + kind_clause,
            params,
        ) as cursor:
            async for row in cursor:
                results.append(self._row_to_memory(row))
        return results

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
                "SELECT confidence, tags FROM memories WHERE id = ?", (memory_id,)
            ) as cursor:
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
                    new_content, content_hash(new_content), confidence,
                    json.dumps(tags), now, memory_id,
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
            canonical = normalize_memory_status(status)
            now = _now_iso()
            if canonical == "retired":
                await self.db.execute(
                    """UPDATE memories SET
                        status = ?, retirement_reason = COALESCE(?, retirement_reason, 'admin_hidden'),
                        retired_at = COALESCE(retired_at, ?), updated_at = ?
                       WHERE id = ?""",
                    (canonical, reason, now, now, memory_id),
                )
            elif canonical == "active":
                await self.db.execute(
                    """UPDATE memories SET
                        status = ?, retirement_reason = NULL, retired_at = NULL,
                        superseded_by = NULL, superseded_at = NULL,
                        replacement_reason = NULL, updated_at = ?
                       WHERE id = ?""",
                    (canonical, now, memory_id),
                )
            else:
                await self.db.execute(
                    "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                    (canonical, now, memory_id),
                )
            await self.db.commit()

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
                    visibility = ?, owner_user_id = ?, project_key = ?, confidence = ?,
                    corroboration_count = ?, contradiction_count = ?,
                    valid_from = ?, valid_until = ?, superseded_by = ?,
                    status = ?, retirement_reason = ?, retired_at = ?,
                    superseded_at = ?, replacement_reason = ?, extraction_context = ?,
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
                    memory.extraction_context,
                    memory.updated_at.isoformat() if memory.updated_at else None,
                    memory.id,
                ),
            )
            await self.db.execute("DELETE FROM memories_fts WHERE memory_id = ?", (memory.id,))
            if search_visible:
                await self.db.execute(
                    "INSERT INTO memories_fts (memory_id, content, entities_text, tags_text) "
                    "VALUES (?, ?, ?, ?)",
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
    ) -> str:
        """Add a supporting source and count only distinct source documents."""
        async with self._write_lock:
            async with self.db.execute(
                """SELECT excerpt, support_kind
                   FROM memory_sources
                   WHERE memory_id = ? AND doc_id = ?""",
                (memory_id, doc_id),
            ) as cursor:
                existing = await cursor.fetchone()

            if existing:
                existing_excerpt = existing["excerpt"]
                existing_kind = existing["support_kind"]
                next_kind = existing_kind
                if existing_kind == "corroborated" and support_kind == "extracted":
                    next_kind = "extracted"

                should_update_excerpt = bool(
                    excerpt
                    and excerpt != existing_excerpt
                    and (not existing_excerpt or len(excerpt) > len(existing_excerpt))
                )
                if should_update_excerpt or next_kind != existing_kind:
                    await self.db.execute(
                        """UPDATE memory_sources
                           SET source_type = ?, excerpt = ?, support_kind = ?
                           WHERE memory_id = ? AND doc_id = ?""",
                        (
                            source_type,
                            excerpt if should_update_excerpt else existing_excerpt,
                            next_kind,
                            memory_id,
                            doc_id,
                        ),
                    )
                    await self.db.commit()
                    return "updated"
                await self.db.commit()
                return "unchanged"

            cursor = await self.db.execute(
                """INSERT OR IGNORE INTO memory_sources (
                    memory_id, doc_id, source_type, excerpt, support_kind
                ) VALUES (?, ?, ?, ?, ?)""",
                (memory_id, doc_id, source_type, excerpt, support_kind),
            )
            if cursor.rowcount:
                await self.db.execute(
                    """UPDATE memories SET
                        corroboration_count = corroboration_count + 1,
                        updated_at = ?
                       WHERE id = ?""",
                    (_now_iso(), memory_id),
                )
            await self.db.commit()
            return "inserted" if cursor.rowcount else "unchanged"

    async def supersede_memory(
        self,
        old_id: str,
        new_memory: Memory,
        *,
        replacement_reason: str | None = None,
    ) -> None:
        """Mark old memory as superseded and insert the new one."""
        _validate_visibility(new_memory.visibility, new_memory.owner_user_id)
        project_key = _normalize_project_key(new_memory.project_key)
        async with self._write_lock:
            now = _now_iso()
            new_status = normalize_memory_status(new_memory.status)
            await self.db.execute(
                """INSERT INTO memories (
                    id, memory_type, content, content_hash, tags, visibility, owner_user_id,
                    project_key, confidence, corroboration_count,
                    contradiction_count, valid_from, valid_until,
                    superseded_by, status, retirement_reason, retired_at,
                    superseded_at, replacement_reason, extraction_context,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    new_memory.id, new_memory.memory_type, new_memory.content,
                    new_memory.content_hash, json.dumps(new_memory.tags),
                    new_memory.visibility, new_memory.owner_user_id, project_key,
                    new_memory.confidence, new_memory.corroboration_count,
                    new_memory.contradiction_count,
                    new_memory.valid_from.isoformat() if new_memory.valid_from else None,
                    new_memory.valid_until.isoformat() if new_memory.valid_until else None,
                    new_memory.superseded_by, new_status,
                    new_memory.retirement_reason,
                    new_memory.retired_at.isoformat() if new_memory.retired_at else None,
                    new_memory.superseded_at.isoformat() if new_memory.superseded_at else None,
                    new_memory.replacement_reason, new_memory.extraction_context,
                    new_memory.created_at.isoformat() if new_memory.created_at else now,
                    now,
                ),
            )
            await self.db.execute(
                """UPDATE memories SET
                    status = 'superseded', superseded_by = ?, valid_until = ?,
                    superseded_at = ?, replacement_reason = ?, updated_at = ?
                   WHERE id = ?""",
                (new_memory.id, now, now, replacement_reason, now, old_id),
            )
            # FTS5 for new memory
            entities_text = " ".join(new_memory.entity_refs)
            tags_text = " ".join(new_memory.tags)
            await self.db.execute(
                "INSERT INTO memories_fts (memory_id, content, entities_text, tags_text) "
                "VALUES (?, ?, ?, ?)",
                (new_memory.id, new_memory.content, entities_text, tags_text),
            )
            await self.db.commit()

    async def promote_quarantined_challenger(
        self,
        *,
        incumbent_id: str,
        challenger: Memory,
        replacement_reason: str | None = None,
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
                    valid_until = ?, replacement_reason = ?, updated_at = ?
                   WHERE id = ?""",
                (challenger.id, now, now, replacement_reason, now, incumbent_id),
            )
            await self.db.execute(
                "DELETE FROM memories_fts WHERE memory_id = ?", (incumbent_id,),
            )
            await self.db.execute(
                "DELETE FROM memories_fts WHERE memory_id = ?", (challenger.id,),
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
                "INSERT INTO memories_fts (memory_id, content, entities_text, tags_text) "
                "VALUES (?, ?, ?, ?)",
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
        """Retire active memories whose ``valid_until`` timestamp has passed."""
        async with self._write_lock:
            now = _now_iso()
            cursor = await self.db.execute(
                """UPDATE memories SET
                    status = 'retired', retirement_reason = 'expired',
                    retired_at = COALESCE(retired_at, ?), updated_at = ?
                   WHERE status = 'active'
                   AND valid_until IS NOT NULL
                   AND valid_until < ?""",
                (now, now, now),
            )
            await self.db.commit()
            return cursor.rowcount

    async def get_expired_memories(self) -> list[Memory]:
        """Return active memories whose valid_until has passed."""
        results: list[Memory] = []
        now = _now_iso()
        async with self.db.execute(
            """SELECT * FROM memories
               WHERE status = 'active'
               AND valid_until IS NOT NULL
               AND valid_until < ?""",
            (now,),
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
    ) -> None:
        async with self._write_lock:
            await self.db.execute(
                """INSERT OR IGNORE INTO memory_sources (
                    memory_id, doc_id, source_type, excerpt, support_kind
                ) VALUES (?, ?, ?, ?, ?)""",
                (memory_id, doc_id, source_type, excerpt, support_kind),
            )
            await self.db.commit()

    async def restore_memory_source_snapshot(self, source: MemorySource) -> None:
        """Restore one memory source row from a captured snapshot."""
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO memory_sources (
                    memory_id, doc_id, source_type, excerpt, support_kind, added_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(memory_id, doc_id) DO UPDATE SET
                    source_type=excluded.source_type,
                    excerpt=excluded.excerpt,
                    support_kind=excluded.support_kind,
                    added_at=excluded.added_at""",
                (
                    source.memory_id,
                    source.doc_id,
                    source.source_type,
                    source.excerpt,
                    source.support_kind,
                    source.added_at.isoformat() if source.added_at else _now_iso(),
                ),
            )
            await self.db.commit()

    async def get_memory_sources(self, memory_id: str) -> list[MemorySource]:
        results: list[MemorySource] = []
        async with self.db.execute(
            "SELECT * FROM memory_sources WHERE memory_id = ?", (memory_id,)
        ) as cursor:
            async for row in cursor:
                d = dict(row)
                results.append(MemorySource(
                    memory_id=d["memory_id"],
                    doc_id=d["doc_id"],
                    source_type=d["source_type"],
                    excerpt=d["excerpt"],
                    support_kind=d.get("support_kind", "extracted"),
                    added_at=_parse_dt(d["added_at"]),
                ))
        return results

    async def get_origin_source_pairs(
        self, memory_ids: list[str]
    ) -> dict[str, list[tuple[str, str | None, str | None]]]:
        """Return each memory's (source_type, support_kind, client) triples, ordered
        oldest-first by (added_at, doc_id), for a batch of memories in one query.
        The client value comes from documents.client and is None for non-agent-session
        sources. Memories with no sources are absent from the result."""
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
                results.append(MemorySource(
                    memory_id=d["memory_id"],
                    doc_id=d["doc_id"],
                    source_type=d["source_type"],
                    excerpt=d["excerpt"],
                    support_kind=d.get("support_kind", "corroborated"),
                    added_at=_parse_dt(d["added_at"]),
                ))
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
            if (
                writer_visibility == Visibility.PRIVATE.value
                and writer_owner_user_id is not None
            ):
                scope_clauses.append("AND m.owner_user_id = ?")
                scope_params.append(writer_owner_user_id)
            if writer_visibility == Visibility.WORKSPACE.value:
                # NULL project_key is normalized to UNSORTED at persistence
                # time; resolve the writer side the same way so the candidate
                # pool stays inside one project boundary.
                scope_clauses.append("AND m.project_key = ?")
                scope_params.append(writer_project_key or UNSORTED_PROJECT_KEY)
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
            await self.db.commit()
        async with self.db.execute(
            "SELECT id FROM entities WHERE canonical_name = ?", (canonical_name,)
        ) as cursor:
            row = await cursor.fetchone()
            assert row is not None
            return row[0]

    async def get_entity_by_canonical(self, canonical_name: str) -> Entity | None:
        async with self.db.execute(
            "SELECT * FROM entities WHERE canonical_name = ?", (canonical_name,)
        ) as cursor:
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
        async with self.db.execute(
            "SELECT * FROM entities ORDER BY canonical_name"
        ) as cursor:
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
        async with self.db.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ) as cursor:
            row = await cursor.fetchone()
            return _entity_from_row(dict(row)) if row else None

    async def count_memories_for_entity(self, entity_id: int) -> int:
        async with self.db.execute(
            "SELECT COUNT(*) FROM memory_entities WHERE entity_id = ?", (entity_id,)
        ) as cursor:
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
            await self.db.commit()

    async def get_aliases_for_entity(self, entity_id: int) -> list[EntityAlias]:
        results: list[EntityAlias] = []
        async with self.db.execute(
            "SELECT * FROM entity_aliases WHERE canonical_id = ?", (entity_id,)
        ) as cursor:
            async for row in cursor:
                d = dict(row)
                results.append(EntityAlias(
                    alias=d["alias"],
                    alias_normalized=d["alias_normalized"],
                    canonical_id=d["canonical_id"],
                    source=d["source"],
                    created_at=_parse_dt(d["created_at"]),
                ))
        return results

    async def get_all_aliases(self) -> list[tuple[str, int]]:
        """Return all (alias_normalized, canonical_id) pairs for entity detection."""
        results: list[tuple[str, int]] = []
        async with self.db.execute(
            "SELECT alias_normalized, canonical_id FROM entity_aliases"
        ) as cursor:
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
        project_binding: Mapping[str, Any] | None = None,
    ) -> None:
        """Insert or update a source row.

        `project_binding` is the structured rule the project resolver
        consults when memories are extracted from this source. `None`
        leaves the source unbound and resolves writes to `UNSORTED`.
        """
        binding_json = (
            json.dumps(dict(project_binding)) if project_binding else None
        )
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO sources (id, type, name, config, project_binding)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                   type=excluded.type,
                   name=excluded.name,
                   config=excluded.config,
                   project_binding=excluded.project_binding""",
                (id, type, name, config_json, binding_json),
            )
            await self.db.commit()

    async def get_source(self, source_id: str) -> dict | None:
        async with self.db.execute(
            "SELECT * FROM sources WHERE id = ?", (source_id,)
        ) as cursor:
            row = await cursor.fetchone()
            if not row:
                return None
            d = dict(row)
            d["config"] = json.loads(d["config"])
            d["project_binding"] = (
                json.loads(d["project_binding"]) if d.get("project_binding") else None
            )
            return d

    async def restore_source_snapshot(self, source: dict) -> None:
        """Restore one source row from a captured snapshot."""
        async with self._write_lock:
            await self.db.execute(
                """INSERT INTO sources
                   (id, type, name, config, status, last_sync, doc_count, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                   type=excluded.type,
                   name=excluded.name,
                   config=excluded.config,
                   status=excluded.status,
                   last_sync=excluded.last_sync,
                   doc_count=excluded.doc_count,
                   created_at=excluded.created_at""",
                (
                    source["id"],
                    source["type"],
                    source["name"],
                    json.dumps(source["config"]),
                    source["status"],
                    source["last_sync"],
                    source["doc_count"],
                    source["created_at"],
                ),
            )
            await self.db.commit()

    async def list_sources(self) -> list[dict]:
        results: list[dict] = []
        async with self.db.execute(
            "SELECT * FROM sources ORDER BY created_at"
        ) as cursor:
            async for row in cursor:
                d = dict(row)
                d["config"] = json.loads(d["config"])
                d["project_binding"] = (
                    json.loads(d["project_binding"]) if d.get("project_binding") else None
                )
                results.append(d)
        return results

    async def list_resolved_projects_for_source(
        self, source_id: str
    ) -> list[tuple[str, int]]:
        """Group memories from a source by their resolved `project_key`.

        Distinct from `list_source_projects`, which reports the raw
        `documents.space_or_project` field as observed at sync time. This
        view follows provenance through `memory_sources` and reads the
        resolver's verdict on each memory, so the admin can see where
        writes actually landed under the active `project_binding`.
        """
        rows: list[tuple[str, int]] = []
        async with self.db.execute(
            """
            SELECT m.project_key AS project_key,
                   COUNT(DISTINCT m.id) AS memory_count
            FROM memories m
            JOIN memory_sources ms ON ms.memory_id = m.id
            JOIN documents d ON d.doc_id = ms.doc_id
            WHERE d.source = ?
            GROUP BY m.project_key
            ORDER BY memory_count DESC, project_key ASC
            """,
            (source_id,),
        ) as cursor:
            async for row in cursor:
                key = row["project_key"] or UNSORTED_PROJECT_KEY
                rows.append((str(key), int(row["memory_count"])))
        return rows

    # ------------------------------------------------------------------
    # Projects
    # ------------------------------------------------------------------

    async def create_project(
        self, *, key: str, name: str, is_shared: bool = False
    ) -> Project:
        """Insert a project row, raising ValueError if `key` already exists."""
        proj_id = f"proj-{uuid.uuid4().hex[:12]}"
        try:
            async with self._write_lock:
                await self.db.execute(
                    "INSERT INTO projects (id, key, name, is_shared) "
                    "VALUES (?, ?, ?, ?)",
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
            "SELECT id, key, name, is_shared, created_at "
            "FROM projects WHERE id = ?",
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
        async with self.db.execute(
            "SELECT id, key, name, is_shared, created_at "
            "FROM projects ORDER BY key"
        ) as cur:
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
            raise ValueError(
                f"project {target.key!r} is reserved and cannot be deleted"
            )
        affected_ids: list[str] = []
        async with self.db.execute(
            "SELECT id FROM memories WHERE project_key = ?", (target.key,)
        ) as cur:
            async for row in cur:
                affected_ids.append(row["id"])
        return affected_ids

    async def commit_project_deletion(
        self, project_id: str, affected_ids: Sequence[str]
    ) -> None:
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
            raise ValueError(
                f"project {target.key!r} is reserved and cannot be deleted"
            )
        async with self._write_lock:
            if affected_ids:
                placeholders = ",".join("?" for _ in affected_ids)
                await self.db.execute(
                    f"UPDATE memories SET project_key = ? "
                    f"WHERE id IN ({placeholders}) AND project_key = ?",
                    (UNSORTED_PROJECT_KEY, *affected_ids, target.key),
                )
            await self.db.execute(
                "DELETE FROM projects WHERE id = ?", (project_id,)
            )
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
                async with self.db.execute(
                    "SELECT doc_id FROM documents WHERE source = ?", (source_id,)
                ) as cursor:
                    async for row in cursor:
                        doc_ids.append(row[0])

                for doc_id in doc_ids:
                    memory_ids: list[str] = []
                    async with self.db.execute(
                        "SELECT memory_id FROM memory_sources WHERE doc_id = ?",
                        (doc_id,),
                    ) as cursor:
                        async for row in cursor:
                            memory_ids.append(row[0])

                    await self.db.execute(
                        "DELETE FROM memory_sources WHERE doc_id = ?", (doc_id,)
                    )

                    retired_ids.extend(
                        await self._refresh_support_after_source_removal_unlocked(memory_ids)
                    )

                    await self.db.execute(
                        "DELETE FROM document_metadata WHERE doc_id = ?", (doc_id,)
                    )
                    await self.db.execute(
                        "DELETE FROM document_relationships WHERE source_doc_id = ? OR target_doc_id = ?",
                        (doc_id, doc_id),
                    )
                    await self.db.execute(
                        "DELETE FROM changelog WHERE doc_id = ?", (doc_id,)
                    )
                    await self.db.execute(
                        "DELETE FROM agent_session_receipts WHERE doc_id = ?", (doc_id,)
                    )
                    await self.db.execute(
                        "DELETE FROM documents WHERE doc_id = ?", (doc_id,)
                    )

                await self.db.execute(
                    "DELETE FROM agent_session_receipts WHERE source_id = ?", (source_id,)
                )
                await self.db.execute(
                    "DELETE FROM sync_state WHERE source = ?", (source_id,)
                )
                await self.db.execute(
                    "DELETE FROM sync_history WHERE source = ?", (source_id,)
                )
                await self.db.execute(
                    "DELETE FROM sources WHERE id = ?", (source_id,)
                )
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
            "SELECT * FROM agent_session_receipts WHERE doc_id = ?", (doc_id,),
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
        a package versus dropped everything as no_output. No stored verdict, no
        threshold, no background job. Explicit-document receipts and receipts
        without a recognized outcome are ignored so the fraction stays
        well-defined.

        When at least one failed receipt is present, ``latest_failure`` carries
        ``count``, ``reason`` (latest), and ``last_seen_at`` so the admin UI can
        surface an operational warning without a second query.
        """
        query = (
            "SELECT metadata, updated_at FROM agent_session_receipts "
            "WHERE source_kind = ?"
        )
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
                if outcome in counts:
                    counts[outcome] += 1
                if outcome == AGENT_SESSION_OUTCOME_FAILED:
                    seen_at = row[1]
                    if seen_at and (
                        latest_failure_seen_at is None
                        or seen_at > latest_failure_seen_at
                    ):
                        latest_failure_seen_at = seen_at
                        reason = metadata.get("reason") if isinstance(metadata, dict) else None
                        latest_failure_reason = reason if isinstance(reason, str) else None

        total = sum(counts.values())
        processed_total = (
            counts[AGENT_SESSION_OUTCOME_PACKAGE_CREATED]
            + counts[AGENT_SESSION_OUTCOME_NO_OUTPUT]
        )
        no_output_fraction = (
            counts[AGENT_SESSION_OUTCOME_NO_OUTPUT] / processed_total
            if processed_total
            else 0.0
        )
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
        async with self.db.execute(
            "SELECT * FROM sync_state WHERE source = ?", (source,)
        ) as cursor:
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
                    source, status, docs_processed, docs_updated, docs_failed,
                    memories_extracted, error_message,
                    json.dumps(failed_docs) if failed_docs else None,
                    started_at, finished_at, run_id,
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
        async with self.db.execute(
            "SELECT * FROM schedule_config WHERE id = 1"
        ) as cursor:
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
        self, memory_id: str, entity_ids: list[int], doc_id: str,
    ) -> list[Memory]:
        """Find active memories sharing entities with this memory but from different documents."""
        if not entity_ids:
            return []

        placeholders = ",".join("?" for _ in entity_ids)
        sql = f"""
            SELECT DISTINCT m.* FROM memories m
            JOIN memory_entities me ON m.id = me.memory_id
            JOIN memory_sources ms ON m.id = ms.memory_id
            WHERE me.entity_id IN ({placeholders})
              AND ms.doc_id != ?
              AND m.id != ?
              AND m.status = 'active'
            LIMIT 20
        """
        params = [*entity_ids, doc_id, memory_id]
        results: list[Memory] = []
        async with self.db.execute(sql, params) as cursor:
            async for row in cursor:
                results.append(self._row_to_memory(row))
        return results

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
                    created_at, resolved_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    review.id, review.kind, review.status,
                    review.incumbent_memory_id, review.challenger_memory_id,
                    review.reason, review.review_note, review.reviewer,
                    review.expected_incumbent_updated_at,
                    review.expected_challenger_updated_at,
                    created_at,
                    review.resolved_at.isoformat() if review.resolved_at else None,
                ),
            )
            await self.db.commit()
        return review.id

    async def get_memory_review(self, review_id: str) -> MemoryReview | None:
        async with self.db.execute(
            "SELECT * FROM memory_reviews WHERE id = ?", (review_id,),
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
                        f"Challenger {challenger_memory_id} is already attached "
                        f"to review {existing_review_id}"
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
                results.append(MemoryReviewRelatedChallenger(
                    review_id=d["review_id"],
                    challenger_memory_id=d["challenger_memory_id"],
                    reason=d["reason"],
                    created_at=_parse_dt(d["created_at"]),
                ))
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
        async with self.db.execute(
            "SELECT * FROM llm_config WHERE id = 1"
        ) as cursor:
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
        async with self.db.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ) as cursor:
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
            await self.db.execute(
                "DELETE FROM users WHERE id = ?", (user_id,)
            )
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
            confidence=d["confidence"],
            corroboration_count=d["corroboration_count"],
            contradiction_count=d["contradiction_count"],
            valid_from=_parse_dt(d.get("valid_from")),
            valid_until=_parse_dt(d.get("valid_until")),
            superseded_by=d.get("superseded_by"),
            status=d["status"],
            retirement_reason=d.get("retirement_reason"),
            retired_at=_parse_dt(d.get("retired_at")),
            superseded_at=_parse_dt(d.get("superseded_at")),
            replacement_reason=d.get("replacement_reason"),
            extraction_context=d.get("extraction_context"),
            created_at=_parse_dt(d.get("created_at")),
            updated_at=_parse_dt(d.get("updated_at")),
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
