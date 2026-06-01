// API types for the MemForge admin UI

export interface Memory {
  id: string;
  memory_type: "fact" | "decision" | "convention" | "procedure";
  content: string;
  content_hash: string;
  scope: string;
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
}

export type MemorySourceSupportKind = "extracted" | "corroborated";

export interface MemorySource {
  doc_id: string;
  source_type: string;
  excerpt: string | null;
  added_at: string;
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
  status: "running" | "success" | "partial" | "failed";
  phase?: "discovering" | "processing" | "detecting_deletions" | "complete" | string | null;
  started_at: string | null;
  finished_at: string | null;
  docs_processed: number;
  docs_total?: number | null;
  docs_updated: number;
  docs_failed?: number;
  memories_extracted: number;
  docs_stored?: number;
  memories_stored?: number;
  current_title?: string | null;
  error_message: string | null;
  failed_docs?: Array<{
    doc_id: string;
    title: string;
    error: string;
  }>;
}

export interface Source {
  id: string;
  type: string;
  name: string;
  config: Record<string, unknown>;
  status: "active" | "paused";
  last_sync: string | null;
  doc_count: number;
  memory_count?: number;
  sync?: SyncStatus | null;
  auth_session?: JiraAuthSession | null;
  created_at: string;
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
