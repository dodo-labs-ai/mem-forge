"""The agent-session capture tables of a database an earlier release made.

Releases up to schema 130 held these tables, and migration 131 drops them; the
bootstrap schema no longer creates them. A test that builds such a database
from the bootstrap schema runs ``CAPTURE_TABLES_SCHEMA`` first, so the
database holds the tables its schema version implies.
"""

from __future__ import annotations

CAPTURE_TABLES = (
    "agent_session_receipts",
    "agent_hook_receipts",
    "agent_concepts",
    "agent_claims",
    "agent_claim_citations",
)

CAPTURE_TABLES_SCHEMA = """
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

CREATE INDEX IF NOT EXISTS idx_agent_session_receipts_session ON agent_session_receipts(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_session_receipts_source ON agent_session_receipts(source_id);
CREATE INDEX IF NOT EXISTS idx_agent_hook_receipts_session ON agent_hook_receipts(session_id);
CREATE INDEX IF NOT EXISTS idx_agent_hook_receipts_hook ON agent_hook_receipts(hook);
CREATE INDEX IF NOT EXISTS idx_agent_concepts_owner_repo ON agent_concepts(owner_user_id, repo_identifier);
CREATE INDEX IF NOT EXISTS idx_agent_claims_concept ON agent_claims(concept_id);
CREATE INDEX IF NOT EXISTS idx_agent_claims_memory ON agent_claims(memory_id);
"""
