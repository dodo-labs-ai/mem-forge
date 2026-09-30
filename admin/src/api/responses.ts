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
