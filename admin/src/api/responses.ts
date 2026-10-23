import type { components } from "./schema.gen";

type Schemas = components["schemas"];

export type Source = Schemas["SourceResponse"];
export type SyncStatus = Schemas["SourceSyncStatusResponse"];
export type SourceListSortMode = Schemas["SourceListPreferencesResponse"]["sort_mode"];
export type LocalAgentDaemonStatus = Schemas["LocalAgentDaemonStatusResponse"];
export type LocalAgentJobStatus = Schemas["LocalAgentJobResponse"]["status"];

/**
 * A sync progress snapshot. `waiting_for_cloud` is not sent by the server:
 * the UI shows it while a finished local sync hands its run to the server.
 */
export type SyncProgressSnapshot = Omit<Schemas["SourceSyncProgressResponse"], "phase"> & {
  phase: Schemas["SourceSyncProgressResponse"]["phase"] | "waiting_for_cloud";
};

/**
 * The server stores a local agent job's result as a free-form object that
 * each operation fills in. These are the fields the UI reads; the list route
 * is typed with this shape in `paths.ts`.
 */
export interface LocalAgentJobResult {
  error?: string | null;
  progress?: SyncProgressSnapshot;
  source_sync_run_id?: string | null;
  [key: string]: unknown;
}

export type LocalAgentJob = Omit<Schemas["LocalAgentJobResponse"], "result"> & { result: LocalAgentJobResult };

/**
 * The result each local setup job reports once it has succeeded, keyed by
 * operation. A failed setup job's result is only certain to carry `error`.
 */
export interface SetupJobResults {
  github_repo_preview_tree: Schemas["GitHubRepoPreviewTreeResult"];
  jira_auth: Schemas["JiraAuthResult"];
  local_markdown_pick_root: Schemas["LocalMarkdownPickRootResult"];
  local_markdown_preview_tree: Schemas["LocalMarkdownPreviewTreeResult"];
  teams_auth: Schemas["TeamsAuthResult"];
  teams_auth_check: Schemas["TeamsAuthCheckResult"];
  teams_browse: Schemas["TeamsBrowseResult"];
}

export type SetupJobOperation = keyof SetupJobResults;

/**
 * What a model validation observed. The server sends these as a free-form
 * object, and a failed validation is only certain to carry `latency_ms`.
 */
export interface ModelValidationFacts {
  latency_ms?: number;
  /** Generation: whether the model answered a JSON schema request or was asked for JSON as text. */
  structured_output_mode?: "native_schema" | "json_text";
  /** Generation: whether the request fell back from a JSON schema to JSON as text. */
  schema_fallback?: boolean;
  /** Generation: false when the server has no token limits for the model. */
  token_limits_known?: boolean;
  /** Generation: whether the model reads images; null when validation could not tell. */
  image_input?: boolean | null;
  /** Embedding: the length of the vectors the model returned. */
  dimension?: number;
  /** The model name the provider reported, when it reported one. */
  provider_model?: string;
}
