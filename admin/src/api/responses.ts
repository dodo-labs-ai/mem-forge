/**
 * Response shapes for Admin API endpoints that do not declare a response
 * model yet, so the generated schema types them as unknown. `paths.ts` merges
 * them into the generated paths. When an endpoint gains a response model,
 * delete its entry here and let the generated type take over (ADR 0044).
 */

export interface SyncProgressSnapshot {
  schema_version: 1;
  phase:
    | "waiting_for_device"
    | "waiting_for_cloud"
    | "connecting"
    | "discovering"
    | "fetching"
    | "uploading"
    | "processing"
    | "recovering_derivations"
    | "reconciling";
  progress?: {
    completed: number;
    total?: number | null;
    unit: "item" | "page" | "file" | "issue" | "message" | "conversation";
  };
  source_time_range?: { start?: string; end?: string };
  counts?: { changed?: number; failed?: number; memories_created?: number };
}

export interface SyncStatus {
  created_at?: string | null;
  status: "pending" | "running" | "recovering" | "success" | "partial" | "failed";
  run_id?: string;
  trigger?: string;
  force_full_sync?: boolean;
  next_attempt_at?: string | null;
  recovery_count?: number;
  started_at: string | null;
  finished_at: string | null;
  docs_processed?: number;
  docs_total?: number | null;
  docs_failed?: number;
  memories_stored?: number;
  error_message: string | null;
  failed_docs?: Array<{ doc_id: string; title: string; error: string }>;
  progress?: SyncProgressSnapshot | null;
  progress_updated_at?: string | null;
}

export type ViewerRole = "workspace_admin" | "member" | "viewer";

export interface SourceOwnership {
  created_by_user_id: string | null;
  owner_user_id: string;
  execution_owner_user_id: string | null;
  viewer_role: ViewerRole;
  viewer_relationship: ViewerRole | "owner";
}

/** Per-row authority computed by the server; row actions render only from these flags. */
export interface SourceCapabilities {
  can_subscribe: boolean;
  can_configure: boolean;
  can_configure_connection: boolean;
  can_sync: boolean;
  can_force_resync: boolean;
  can_delete: boolean;
  can_change_access: boolean;
}

export type SourceAccessPolicy = "private" | "workspace";

export interface SourceAccessTransition {
  operation_id: string;
  source_id: string;
  previous_policy: SourceAccessPolicy;
  target_policy: SourceAccessPolicy;
  status: "queued" | "running" | "failed" | "completed" | "reverted";
  total_memories: number;
  processed_memories: number;
  error_code?: string | null;
  error_message?: string | null;
  created_at: string;
  updated_at: string;
  completed_at?: string | null;
}

export interface SourceSyncSchedule {
  enabled: boolean;
  interval_minutes: number;
  next_run_at: string | null;
  updated_at: string | null;
}

export type SourceExecutionKind = "server" | "local_agent";

export interface SourceConnectionStatus {
  state: "ready" | "action_required";
  reason: "authentication" | "configuration" | "identity_conflict" | null;
}

export interface ProjectBinding {
  mode: "fixed" | "by_field";
  project_key?: string;
  field?: string;
  map?: Record<string, string>;
  default?: string;
}

export interface Source {
  id: string;
  type: string;
  /** For agent-session sources, the plugin client: "codex" or "claude-code". */
  client?: string | null;
  name: string;
  /** Redacted config; `{}` when the viewer cannot configure the source. */
  config: Record<string, unknown>;
  status: "active" | "paused";
  access_policy: SourceAccessPolicy;
  access_state: "active" | "changing" | "orphaned_private";
  access_transition?: SourceAccessTransition | null;
  last_sync: string | null;
  doc_count: number;
  memory_count?: number;
  sync?: SyncStatus | null;
  connection_status?: SourceConnectionStatus | null;
  created_at: string;
  /** Absent or null: memories land in the Unmapped backlog. */
  project_binding?: ProjectBinding | null;
  ownership?: SourceOwnership;
  capabilities?: SourceCapabilities;
  execution?: { kind: SourceExecutionKind; operation: string | null; immutable_config_fields: string[] };
  subscription?: { enabled: boolean };
  enabled_for_me?: boolean;
  pinned_for_me?: boolean;
  sync_schedule?: SourceSyncSchedule | null;
}

export interface SourceListResponse {
  data: Source[];
}

export type SourceListSortMode = "newest" | "name" | "recently_synced";

export interface SourceListPreferences {
  sort_mode: SourceListSortMode;
}

export interface SourceSyncReceipt {
  status: string;
  run_id?: string;
  created_at?: string | null;
}

export type LocalAgentJobStatus = "queued" | "leased" | "succeeded" | "failed";

export interface LocalAgentJobCreateResponse {
  created_at?: string | null;
  job_id: string;
  status: LocalAgentJobStatus;
}

export interface LocalAgentJob {
  job_id: string;
  source_id?: string;
  operation?: string;
  status: LocalAgentJobStatus;
  attempt_count?: number;
  leased_until?: string | null;
  next_attempt_at?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
  finished_at?: string | null;
  result: {
    error?: string | null;
    progress?: SyncProgressSnapshot;
    source_sync_run_id?: string | null;
    [key: string]: unknown;
  } | null;
  last_error?: string | null;
}

export interface LocalAgentJobList {
  data: LocalAgentJob[];
}

export interface LocalAgentDaemonStatus {
  status: "online" | "offline";
  last_seen_at: string | null;
  checked_at: string;
  stale_after_seconds: number;
}
