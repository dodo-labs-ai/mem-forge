// API types for the MemForge admin UI

/**
 * Visibility levels carried on every memory. "private" is owner-only;
 * "workspace" is everyone in the active workspace.
 */
export type Visibility = "private" | "workspace";

export interface Memory {
  id: string;
  memory_type: "fact" | "decision" | "convention" | "procedure";
  content: string;
  content_hash: string;
  visibility: Visibility | string;
  owner_user_id: string | null;
  project_key: string | null;
  tags: string[];
  confidence: number;
  corroboration_count: number;
  contradiction_count: number;
  status: "active" | "superseded" | "retired" | "decayed" | "pending_review";
  retirement_reason: string | null;
  retired_at: string | null;
  superseded_at: string | null;
  superseded_by: string | null;
  replacement_reason: string | null;
  valid_from: string | null;
  valid_until: string | null;
  created_at: string;
  updated_at: string;
  extraction_context: string | null;
  entity_refs: string[];
  sources: MemorySource[];
  /** Source type for the leading glyph: extraction origin, else first source. */
  origin_source_type: string | null;
  /**
   * For agent_session memories, the plugin client that produced the source
   * document ("codex" or "claude-code"). Passed to SourceIcon to pick the
   * correct single brand mark.
   */
  origin_client?: string | null;
}

export type MemorySourceSupportKind = "extracted" | "corroborated";

export interface MemorySource {
  doc_id: string;
  source_type: string;
  excerpt: string | null;
  added_at: string;
  source_observed_at?: string | null;
  doc_title?: string | null;
  source_url?: string | null;
  file_uri?: string | null;
  pdf_uri?: string | null;
  support_kind?: MemorySourceSupportKind;
}

export interface Entity {
  id: number;
  canonical_name: string;
  tags: string[];
  display_name: string;
  created_at: string | null;
}

export interface EntityDetail extends Entity {
  aliases: EntityAlias[];
  linked_memory_count: number;
}

export interface EntityAlias {
  alias: string;
  alias_normalized: string;
  source: string;
  created_at: string | null;
}

export interface SyncStatus {
  status: "pending" | "running" | "recovering" | "success" | "partial" | "failed";
  run_id?: string;
  trigger?: string;
  force_full_sync?: boolean;
  next_attempt_at?: string | null;
  recovery_count?: number;
  phase?: "discovering" | "processing" | "detecting_deletions" | "complete" | string | null;
  started_at: string | null;
  finished_at: string | null;
  docs_processed?: number;
  docs_total?: number | null;
  docs_updated?: number;
  docs_failed?: number;
  memories_extracted?: number;
  docs_stored?: number;
  memories_stored?: number;
  current_title?: string | null;
  error_message: string | null;
  failed_docs?: Array<{
    doc_id: string;
    title: string;
    error: string;
  }>;
  progress?: SyncProgressSnapshot | null;
  progress_revision?: number;
  progress_updated_at?: string | null;
}

export type SyncProgressPhase =
  | "waiting_for_device"
  | "connecting"
  | "discovering"
  | "uploading"
  | "processing"
  | "reconciling";

export type SyncProgressUnit = "item" | "page" | "file" | "issue" | "message" | "conversation";

export interface SyncProgressSnapshot {
  schema_version: 1;
  phase: SyncProgressPhase;
  progress?: {
    completed: number;
    total?: number | null;
    unit: SyncProgressUnit;
  };
  source_time_range?: { start?: string; end?: string };
  counts?: { changed?: number; failed?: number; memories_created?: number };
}

/**
 * Workspace role of the viewer that fetched the source list. The backend
 * normalises any non-admin role to "member" before returning rows; "viewer"
 * is reserved for read-only roles introduced by hosted deployments.
 */
export type ViewerRole = "workspace_admin" | "member" | "viewer";

/**
 * Relationship between the viewer and a particular source. "creator" is set
 * for the user who created the source, including a workspace admin who owns
 * the row. "workspace_admin" is used when an admin manages someone else's
 * source.
 */
export type SourceViewerRelationship = ViewerRole | "creator";

export interface SourceOwnership {
  created_by_user_id: string | null;
  execution_owner_user_id: string | null;
  viewer_role: ViewerRole;
  viewer_relationship: SourceViewerRelationship;
}

/**
 * Per-row authority flags computed by the backend. The UI must render row
 * actions exclusively from these flags rather than rederiving role/creator
 * locally; backend remains authoritative on the wire.
 */
export interface SourceCapabilities {
  can_subscribe: boolean;
  can_configure: boolean;
  can_configure_connection: boolean;
  can_sync: boolean;
  can_force_resync: boolean;
  can_delete: boolean;
}

/**
 * Per-viewer subscription state. Disabling a source removes its memories
 * from this viewer's lists without affecting other workspace members.
 */
export interface SourceSubscription {
  enabled: boolean;
}

export interface SourceSyncSchedule {
  enabled: boolean;
  interval_minutes: number;
  next_run_at: string | null;
  updated_at: string | null;
}

export type SourceExecutionKind = "server" | "local_agent";

export interface SourceExecution {
  kind: SourceExecutionKind;
  operation: string | null;
  immutable_config_fields: string[];
}

export type SourceConnectionStatusReason = "authentication" | "configuration" | "identity_conflict";

export interface SourceConnectionStatus {
  state: "ready" | "action_required";
  reason: SourceConnectionStatusReason | null;
}

export interface Source {
  id: string;
  type: string;
  /** For agent_session sources, identifies the plugin client: "codex" or "claude-code". */
  client?: string | null;
  name: string;
  /**
   * Full redacted config. The backend returns `{}` for sources where the
   * viewer cannot configure (`capabilities.can_configure === false`); the
   * config dialog must not open or submit in that case.
   */
  config: Record<string, unknown>;
  status: "active" | "paused";
  last_sync: string | null;
  doc_count: number;
  memory_count?: number;
  sync?: SyncStatus | null;
  connection_status?: SourceConnectionStatus | null;
  created_at: string;
  /**
   * How memories extracted from this source are routed to projects.
   * Absent or null means the source is unbound and its memories fall to
   * the Unmapped backlog until an admin classifies them.
   */
  project_binding?: ProjectBinding | null;
  ownership?: SourceOwnership;
  capabilities?: SourceCapabilities;
  execution?: SourceExecution;
  subscription?: SourceSubscription;
  /** Convenience mirror of `subscription.enabled` for the current viewer. */
  enabled_for_me?: boolean;
  /** Personal Source List organization state for the current viewer. */
  pinned_for_me?: boolean;
  sync_schedule?: SourceSyncSchedule | null;
}

export interface SourceProject {
  project: string;
  document_count: number;
  memory_count: number;
  last_observed_at: string | null;
}

export interface SourceProjectsResponse {
  source_id: string;
  projects: SourceProject[];
}

/**
 * Wire-side project kind. "shared" is team-wide (never penalised in ranking);
 * "normal" is a regular project bucket. Storage stores this as `is_shared` 0/1.
 */
export type ProjectKind = "normal" | "shared";

export interface Project {
  id: string;
  key: string;
  name: string;
  kind: ProjectKind;
  created_at: string;
}

/**
 * How a source binds its extracted memories to projects.
 *
 * - `fixed`: every memory the source produces lands in `project_key`.
 * - `by_field`: the resolver reads `field` off each document and looks the
 *   value up in `map`; values not in `map` fall through to `default`.
 */
export type ProjectBindingMode = "fixed" | "by_field";

export interface ProjectBinding {
  mode: ProjectBindingMode;
  project_key?: string;
  field?: string;
  map?: Record<string, string>;
  default?: string;
}

/**
 * One project bucket a source has actually contributed memories to,
 * as reported by `GET /api/sources/{id}/projects/resolved`.
 */
export interface SourceResolvedProject {
  project_key: string;
  memory_count: number;
}

export interface ResolvedProjectsResponse {
  source_id: string;
  projects: SourceResolvedProject[];
}

/** A source rendered inside one project group, with that group's memory count. */
export interface GroupedSource {
  source: Source;
  memory_count: number;
}

/**
 * One section in the project-grouped sources view. `project` is null for the
 * Unmapped backlog (sources without a `project_binding`).
 */
export interface SourceProjectGroup {
  project: Project | null;
  sources: GroupedSource[];
  docCount: number;
  memoryCount: number;
}

export interface AgentSessionLatestFailure {
  count: number;
  reason: string | null;
  last_seen_at: string | null;
}

export interface AgentSessionCompleteness {
  session_id: string | null;
  source_id: string | null;
  total: number;
  processed_total: number;
  counts: Record<string, number>;
  no_output_fraction: number;
  latest_failure: AgentSessionLatestFailure | null;
}

export interface JiraAuthSession {
  provider: "jira";
  origin: string;
  status: "active" | "expired" | "missing" | "failed" | string;
  principal_id: string | null;
  principal_name: string | null;
  principal_email: string | null;
  browser: string | null;
  captured_at: string | null;
  validated_at: string | null;
  last_error: string | null;
  principal_changed?: boolean;
  sources_reset?: string[];
}

export interface SyncState {
  source: string;
  last_sync_at: string | null;
  last_sync_status: "success" | "partial" | "error" | "running" | null;
  docs_processed: number;
  docs_updated: number;
  docs_failed: number;
  memories_extracted: number;
  error_message: string | null;
}

export interface GeneMetadata {
  name: string;
  display_name: string;
  description: string;
  default_sync_interval_minutes: number;
  auth_method: string;
  data_shape: string;
  /** Where source collection may run. Memory processing remains server-side. */
  execution_kinds: SourceExecutionKind[];
}

export interface ConfigField {
  key: string;
  label: string;
  field_type: string;
  required: boolean;
  placeholder: string;
  help_text: string;
  group: string;
  order: number;
  default: string;
  options: string[];
  advanced: boolean;
}

export interface ConfigGroup {
  key: string;
  label: string;
  order: number;
}

export interface GeneConfigSchema {
  groups: ConfigGroup[];
  fields: ConfigField[];
  /**
   * Key of the document field a `by_field` project binding should read.
   * The matching `ConfigField.label` supplies the human label; absent
   * means this gene does not support `by_field` and the dialog hides it.
   */
  project_field?: string | null;
}

export interface DiscoveryPreviewItem {
  item_id: string;
  title: string;
  source_url: string;
  last_modified: string | null;
}

export interface DiscoveryPreviewResponse {
  source_type: string;
  count: number;
  truncated: boolean;
  items: DiscoveryPreviewItem[];
}

export interface GitHubRepoTreeItem {
  path: string;
  type: "tree" | "blob";
  size: number | null;
}

export interface GitHubRepoTreeResponse {
  source_type: "github_repo";
  ref: string;
  count: number;
  truncated: boolean;
  items: GitHubRepoTreeItem[];
}

export interface LocalAgentJobCreateResponse {
  job_id: string;
  status: "queued";
}

export interface LocalAgentJobCounts {
  selected?: number;
  pushed?: number;
  skipped_existing?: number;
  failed?: number;
  polls?: number;
}

export interface LocalAgentJobStatusResponse {
  job_id: string;
  source_id?: string;
  operation?: string;
  status: "queued" | "leased" | "succeeded" | "failed";
  attempt_count?: number;
  leased_until?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  finished_at?: string | null;
  result: {
    authenticated?: boolean;
    expires_in_minutes?: number | null;
    error?: string | null;
    items?: Array<{
      path?: string;
      relative_path?: string;
      type?: "tree" | "blob";
      size?: number | null;
    }>;
    favorites?: TeamsChat[];
    teams?: TeamsTeam[];
    group_chats?: TeamsChat[];
    individual_chats?: TeamsChat[];
    truncated?: boolean;
    counts?: LocalAgentJobCounts;
    progress?: SyncProgressSnapshot;
    date_from?: string | null;
    date_to?: string | null;
    messages?: number;
    conversations?: number;
    sync_started?: boolean;
  } | null;
  last_error?: string | null;
}

export interface LocalAgentDaemonStatusResponse {
  status: "online" | "offline";
  last_seen_at: string | null;
  checked_at: string;
  stale_after_seconds: number;
}

export interface LlmConfig {
  enrichment_model: string | null;
  enrichment_base_url: string | null;
  enrichment_api_key: string | null;
  enrichment_api_key_set?: boolean;
  enrichment_api_key_last4?: string | null;
  embedding_model: string | null;
  embedding_base_url: string | null;
  embedding_api_key: string | null;
  embedding_api_key_set?: boolean;
  embedding_api_key_last4?: string | null;
}

export interface LlmModelOption {
  id: string;
  label?: string | null;
}

export interface LlmConfigProbeResponse {
  ok: boolean;
  models_supported: boolean;
  models: LlmModelOption[];
  stage: "validation" | "connect" | "tls" | "timeout" | "auth" | "http" | null;
  status: number | null;
  message: string;
  latency_ms: number | null;
  suggested_base_url: string | null;
}

export interface StatBucket {
  key: string;
  count: number;
}

export interface Stats {
  total_memories: number;
  memories_by_type: StatBucket[];
  memories_by_status: StatBucket[];
  total_sources: number;
  total_entities: number;
}

// Paginated response wrapper
export interface PaginatedResponse<T> {
  data: T[];
  total: number;
  limit?: number;
  offset?: number;
}

// Teams browse types (for channel picker)
export interface TeamsChannel {
  id: string;
  displayName: string;
}

export interface TeamsTeam {
  id: string;
  displayName: string;
  channels: TeamsChannel[];
}

export interface TeamsChat {
  id: string;
  topic: string;
  lastActivity: string | null;
}

export interface TeamsBrowseData {
  favorites: TeamsChat[];
  teams: TeamsTeam[];
  group_chats: TeamsChat[];
  individual_chats: TeamsChat[];
}

export interface TeamsAuthStatus {
  authenticated: boolean;
  expires_in_minutes: number | null;
  error: string | null;
}

// Memory reviews

export type MemoryReviewStatus = "pending" | "approved" | "rejected" | "stale";
export type MemoryReviewKind = "supersede";

export interface MemoryReviewSummary {
  id: string;
  kind: MemoryReviewKind | string;
  status: MemoryReviewStatus | string;
  incumbent_memory_id: string;
  challenger_memory_id: string;
  reason: string | null;
  review_note: string | null;
  reviewer: string | null;
  expected_incumbent_updated_at: string | null;
  expected_challenger_updated_at: string | null;
  created_at: string | null;
  resolved_at: string | null;
  is_stale: boolean;
  incumbent?: MemoryReviewMemorySummary | null;
  challenger?: MemoryReviewMemorySummary | null;
}

export interface MemoryReviewMemorySummary {
  id: string;
  memory_type: Memory["memory_type"];
  content: string;
  confidence: number;
  corroboration_count: number;
  status: string;
  tags: string[];
  entity_refs: string[];
  sources: MemorySource[];
  created_at: string | null;
  updated_at: string | null;
  origin_source_type?: string | null;
  origin_client?: string | null;
}

export interface MemoryReviewDetail extends MemoryReviewSummary {
  incumbent: MemoryReviewMemorySummary | null;
  challenger: MemoryReviewMemorySummary | null;
  related_challengers: MemoryReviewMemorySummary[];
}

export interface MemoryReviewListResponse {
  data: MemoryReviewSummary[];
  total: number;
}
