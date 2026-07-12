"""Core data models for MemForge.

All dataclasses used across the system: memories, entities, genes, documents, sync state.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Entity name canonicalization
# ---------------------------------------------------------------------------


def canonicalize_entity_name(name: str) -> str:
    """Normalize entity name: lowercase, hyphens/underscores to spaces, collapse whitespace.

    No abbreviation expansion — the alias table handles all name variants
    at runtime (self-improving via fuzzy auto-registration + LLM extraction
    + admin manual entry).
    """
    s = name.strip().lower()
    s = re.sub(r"[-_]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def slugify(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    s = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = s.strip("-")
    return s[:120] if s else "untitled"


def generate_memory_id() -> str:
    return f"mem-{uuid.uuid4().hex[:8]}"


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class MemoryType(str, Enum):
    FACT = "fact"
    DECISION = "decision"
    CONVENTION = "convention"
    PROCEDURE = "procedure"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    RETIRED = "retired"
    DECAYED = "decayed"
    PENDING_REVIEW = "pending_review"


ReplacementKind = Literal["revision", "supersession"]


class MemoryLevel(str, Enum):
    ATOMIC = "atomic"
    CONSOLIDATED = "consolidated"


class Visibility(str, Enum):
    """Per-row access tier within one datastore. 'org' is reserved (strict silos)."""

    WORKSPACE = "workspace"
    PRIVATE = "private"


# Reserved project keys. SHARED is the team-wide bucket; UNSORTED is the
# down-weighted backlog for memories with no resolvable project. Named here so
# helper code and tests never repeat the literal; the SQL migrations may inline them.
SHARED_PROJECT_KEY = "SHARED"
UNSORTED_PROJECT_KEY = "UNSORTED"


@dataclass
class Project:
    """A relevance bucket within the bound datastore.

    `is_shared=True` is the team-wide SHARED bucket: never down-weighted by the
    cross-project affinity penalty and always open for the access predicate.
    `is_shared=False` is a normal project (or the reserved UNSORTED backlog).
    """

    id: str
    key: str
    name: str
    is_shared: bool = False
    created_at: datetime | None = None


class ReconcileAction(str, Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    SUPERSEDE = "SUPERSEDE"
    DELETE = "DELETE"
    NOOP = "NOOP"


class ConfigFieldType(str, Enum):
    STRING = "string"
    SECRET = "secret"
    URL = "url"
    INTEGER = "integer"
    BOOLEAN = "boolean"
    SELECT = "select"
    MULTI_SELECT = "multi_select"
    TAG_LIST = "tag_list"


# ---------------------------------------------------------------------------
# Entity models
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    id: int
    canonical_name: str
    tags: list[str] = field(default_factory=list)  # optional soft tags: service, api, technology, etc.
    display_name: str = ""
    created_at: datetime | None = None

    @property
    def entity_type(self) -> str:
        """Deprecated: returns first tag or 'unknown'. Use .tags instead."""
        return self.tags[0] if self.tags else "unknown"


@dataclass
class EntityAlias:
    alias: str
    alias_normalized: str
    canonical_id: int
    source: str  # "exact" | "fuzzy_auto" | "llm_extracted" | "admin_manual"
    created_at: datetime | None = None


# ---------------------------------------------------------------------------
# Memory models
# ---------------------------------------------------------------------------

@dataclass
class Memory:
    id: str                          # "mem-{uuid8}"
    memory_type: str                 # fact | decision | convention | procedure
    content: str
    content_hash: str                # SHA-256 for dedup

    # Scoping
    visibility: str = Visibility.WORKSPACE.value  # team-visible by default
    owner_user_id: str | None = None              # set iff visibility is private
    project_key: str | None = None
    repo_identifier: str | None = None

    # Entity linkage
    entity_refs: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    # Confidence and lifecycle
    confidence: float = 0.7
    corroboration_count: int = 1
    contradiction_count: int = 0
    valid_from: date | None = None
    valid_until: date | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None

    # Lifecycle
    superseded_by: str | None = None
    status: str = "active"
    retirement_reason: str | None = None
    retired_at: datetime | None = None
    superseded_at: datetime | None = None
    replacement_reason: str | None = None
    replacement_kind: ReplacementKind | None = None
    extraction_context: str | None = None
    memory_level: str = MemoryLevel.ATOMIC.value
    curation_cluster_id: str | None = None


@dataclass
class RawMemory:
    """A memory candidate extracted by the LLM, before dedup/insertion."""
    content: str
    memory_type: str
    confidence: float = 0.7
    entity_refs: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    valid_from: str | None = None
    valid_until: str | None = None
    extraction_context: str | None = None
    evidence_quote: str | None = None
    evidence_anchor: str | None = None


@dataclass
class MemorySource:
    """Provenance: links a memory to a source document."""
    memory_id: str
    doc_id: str
    source_type: str
    source_id: str | None = None
    excerpt: str | None = None
    support_kind: str = "extracted"
    added_at: datetime | None = None
    source_updated_at: datetime | None = None


@dataclass
class MemoryDerivation:
    """Lineage edge from a consolidated memory to an atomic child memory."""

    parent_memory_id: str
    child_memory_id: str
    relation: str = "summarizes"
    created_at: datetime | None = None


@dataclass
class MemoryCurationRun:
    """Audit record for one Curator execution."""

    id: str
    policy_id: str
    source_type: str
    client: str | None
    repo_identifier: str | None
    project_key: str | None
    candidate_count: int
    created_memory_count: int
    skipped_reason: str | None
    error: str | None
    started_at: datetime
    completed_at: datetime | None = None


@dataclass
class AgentSessionReceipt:
    """Lineage for a client-generated agent session document."""
    doc_id: str
    source_id: str
    client: str
    session_id: str
    trigger: str
    workspace: str
    repo: str | None
    branch: str | None
    commit_sha: str | None
    history_window_kind: str
    history_window_start: str | None
    history_window_end: str | None
    submitted_at: str
    document_hash: str
    source_kind: str
    document_uri: str
    metadata: dict = field(default_factory=dict)
    updated_at: str | None = None


@dataclass
class AgentHookReceipt:
    """Lineage for a coding-agent lifecycle hook event."""

    receipt_id: str
    client: str
    session_id: str
    hook: str
    workspace: str
    repo: str | None
    branch: str | None
    commit_sha: str | None
    submitted_at: str
    metadata: dict = field(default_factory=dict)
    updated_at: str | None = None


# ---------------------------------------------------------------------------
# Document models
# ---------------------------------------------------------------------------

@dataclass
class DocRef:
    """Reference to a document discovered by a gene."""
    doc_id: str
    source: str  # source instance ID
    source_url: str
    title: str
    space_or_project: str
    last_modified: datetime
    version: str


@dataclass
class RawDocument:
    """Raw fetched content from a gene."""
    ref: DocRef
    content: bytes
    content_type: str
    author: str | None = None
    labels: list[str] = field(default_factory=list)


@dataclass
class DocumentRecord:
    """Full document record stored in the database."""
    doc_id: str
    source: str
    source_url: str
    title: str
    space_or_project: str
    author: str | None
    last_modified: datetime
    labels: list[str]
    version: str
    content_hash: str
    token_count: int | None
    raw_content_uri: str | None
    raw_content_type: str | None
    normalized_content_uri: str | None
    pdf_content_uri: str | None
    last_synced: datetime
    client: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass
class DocumentMetadata:
    """AI-enriched metadata for a document (Call 1 output)."""
    doc_id: str
    summary: str
    tags: list[str]
    entities: list[Entity]
    doc_type: str
    complexity: str
    enriched_at: datetime | None = None


@dataclass
class Relationship:
    target_doc_id: str | None
    target_title: str
    relation_type: str  # depends-on, extends, supersedes, references, related
    confidence: float


# ---------------------------------------------------------------------------
# Enrichment / extraction results
# ---------------------------------------------------------------------------

@dataclass
class RawEntityRef:
    """Entity reference as returned by the LLM enrichment call."""
    name: str
    type: str = "unknown"
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    aliases: list[str] = field(default_factory=list)  # other names the doc uses for this entity


@dataclass
class RawAliasGroup:
    """Legacy alias group format. Kept for backward compatibility."""
    canonical: str
    aliases: list[str]
    evidence: str = ""


@dataclass
class EnrichmentResult:
    """Output of Call 1 (enrichment)."""
    summary: str = ""
    tags: list[str] = field(default_factory=list)
    entities: list[RawEntityRef] = field(default_factory=list)
    relationships: list[Relationship] = field(default_factory=list)
    doc_type: str = "unknown"
    complexity: str = "medium"
    entity_aliases: list[RawAliasGroup] = field(default_factory=list)  # legacy format


@dataclass
class MemoryExtractionResult:
    """Output of Call 2 (memory extraction)."""
    memories: list[RawMemory] = field(default_factory=list)
    error_type: str | None = None
    error: str | None = None
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Gene data models
# ---------------------------------------------------------------------------

@dataclass
class ContentItem:
    """A content item discovered by a gene (the unit of sync)."""
    item_id: str  # becomes doc_id
    title: str
    source_url: str
    last_modified: datetime
    content_type: str = "text/html"
    space_or_project: str = ""
    version: str = ""
    author: str | None = None
    labels: list[str] = field(default_factory=list)
    extra: dict = field(default_factory=dict)  # source-specific metadata

    def to_doc_ref(self, source_id: str) -> DocRef:
        return DocRef(
            doc_id=self.item_id,
            source=source_id,
            source_url=self.source_url,
            title=self.title,
            space_or_project=self.space_or_project,
            last_modified=self.last_modified,
            version=self.version,
        )


@dataclass
class RawContent:
    """Raw content fetched by a gene."""
    item: ContentItem
    body: bytes
    content_type: str


@dataclass
class NormalizedContent:
    """Normalized content produced by a gene's normalizer."""
    item: ContentItem
    markdown_body: str
    source_semantics: dict = field(default_factory=dict)


SourceExecutionKind = Literal["server", "local_agent"]


@dataclass
class GeneMetadata:
    """Static metadata for a gene type."""
    name: str                            # "confluence", "jira", "teams", "outlook"
    display_name: str                    # "Microsoft Teams"
    description: str
    default_sync_interval_minutes: int
    auth_method: str                     # "pat" | "oauth2" | "api_key" | "browser_cookie"
    data_shape: str                      # "document" | "ticket" | "message" | "email"
    # Where source collection may run. Memory processing remains server-side.
    execution_kinds: tuple[SourceExecutionKind, ...] = ("server",)


@dataclass
class ConfigField:
    """A single configuration field for a gene's config schema."""
    key: str
    label: str
    field_type: ConfigFieldType
    required: bool = True
    placeholder: str = ""
    help_text: str = ""
    group: str = "general"
    order: int = 0
    default: str = ""
    options: list[str] = field(default_factory=list)  # for SELECT / MULTI_SELECT
    advanced: bool = False


@dataclass
class ConfigGroup:
    """A group of config fields displayed together in the UI."""
    key: str
    label: str
    order: int = 0


@dataclass
class GeneConfigSchema:
    """Dynamic config schema a gene declares for UI rendering.

    `project_field` names the field a `by_field` project_binding reads on
    each item this gene produces. Doc-shaped genes typically expose
    `space_or_project`; the agent-session gene exposes `repo`. The admin
    UI uses this to scope the binding editor to fields the gene actually
    populates.
    """
    groups: list[ConfigGroup] = field(default_factory=list)
    fields: list[ConfigField] = field(default_factory=list)
    project_field: str | None = None


# ---------------------------------------------------------------------------
# Sync models
# ---------------------------------------------------------------------------

@dataclass
class FailedDoc:
    doc_id: str
    title: str
    error: str


@dataclass
class SyncState:
    source: str
    last_sync_at: datetime | None = None
    last_sync_status: str | None = None
    docs_processed: int = 0
    docs_updated: int = 0
    docs_failed: int = 0
    memories_extracted: int = 0
    memories_corroborated: int = 0
    error_message: str | None = None
    failed_docs: list[FailedDoc] = field(default_factory=list)


@dataclass
class SourceSyncRun:
    run_id: str
    workspace_id: str
    source_id: str
    trigger: str
    status: str
    force_full_sync: bool = False
    input_snapshot_id: str | None = None
    rerun_input_snapshot_id: str | None = None
    coalesced: bool = False
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    lease_attempt_count: int = 0
    recovery_count: int = 0
    rerun_requested: bool = False
    next_attempt_at: datetime | None = None
    error_message: str | None = None
    progress: dict[str, object] | None = None
    progress_revision: int = 0
    progress_updated_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


@dataclass
class SourceSyncInput:
    input_id: str
    workspace_id: str
    source_id: str
    input_generation: int
    raw_uri: str
    raw_sha256: str
    raw_content_type: str
    metadata: dict[str, object] = field(default_factory=dict)
    created_at: datetime | None = None


@dataclass
class ChangelogEntry:
    id: int | None
    doc_id: str
    change_type: str  # "created" | "updated" | "deleted"
    previous_version: str | None
    current_version: str | None
    content_diff: str | None
    ai_change_summary: str | None
    detected_at: datetime
    title: str | None = None
    source: str | None = None


# ---------------------------------------------------------------------------
# Search models
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single memory search result."""
    memory_id: str
    memory_type: str
    summary: str
    confidence: float
    relevance_score: float
    tags: list[str] = field(default_factory=list)
    # Metadata
    corroborated_by: int = 1
    last_observed_at: str | None = None
    freshness: str = "current"  # current | stale | unverified
    contradiction_warning: str | None = None
    status: str = "active"
    memory_level: str = MemoryLevel.ATOMIC.value
    curation_cluster_id: str | None = None
    covered_memory_count: int = 0
    repo_identifier: str | None = None
    follow_up: dict[str, str] | None = None
    retrieval_evidence: dict[str, Any] | None = None


@dataclass
class ReconcileOperation:
    """A single reconciliation operation from the LLM."""
    action: ReconcileAction
    memory_id: str | None = None  # existing memory ID (for UPDATE/SUPERSEDE/DELETE/NOOP)
    memory: RawMemory | None = None  # new or updated memory (for ADD/UPDATE/SUPERSEDE)
    reason: str | None = None
    flag_for_review: bool = False


# ---------------------------------------------------------------------------
# Memory reviews — workbench records for human-gated reconciliation
# ---------------------------------------------------------------------------

class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    STALE = "stale"


class ReviewKind(str, Enum):
    SUPERSEDE = "supersede"


def generate_review_id() -> str:
    return f"rev-{uuid.uuid4().hex[:8]}"


def generate_deterministic_review_id(
    *,
    kind: str,
    incumbent_memory_id: str,
    challenger_memory_id: str,
    relation_run_id: str | None = None,
    evidence_unit_id: str | None = None,
    review_case: str | None = None,
) -> str:
    """Stable review id for retrying the same proposed lifecycle decision."""
    digest = content_hash(
        "\x1f".join(
            [
                kind,
                incumbent_memory_id,
                challenger_memory_id,
                relation_run_id or "",
                evidence_unit_id or "",
                review_case or "",
            ]
        )
    )[:16]
    return f"rev-{digest}"


@dataclass
class MemoryReview:
    """A pending or resolved human-review decision for a memory lifecycle change."""
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
    replacement_kind: ReplacementKind = "supersession"
    created_at: datetime | None = None
    resolved_at: datetime | None = None


@dataclass
class MemoryReviewRelatedChallenger:
    """Additional challenger memory attached to a visible review case."""
    review_id: str
    challenger_memory_id: str
    reason: str | None = None
    created_at: datetime | None = None
