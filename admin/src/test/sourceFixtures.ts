import type { LocalAgentJob, Source } from "@/api/responses";

const FULL_CAPABILITIES = {
  can_subscribe: true,
  can_configure: true,
  can_configure_connection: true,
  can_sync: true,
  can_force_resync: true,
  can_delete: true,
  can_change_access: true,
};

/** A healthy server-synced source bound to the PAY project; override what a test needs. */
export function makeSource(overrides: Partial<Source> = {}): Source {
  return {
    id: "src-wiki",
    type: "confluence",
    client: null,
    name: "Payroll wiki",
    config: {},
    status: "active",
    access_policy: "workspace",
    access_state: "active",
    access_transition: null,
    owner_user_id: "local",
    ownership: {
      created_by_user_id: "local",
      owner_user_id: "local",
      execution_owner_user_id: null,
      viewer_role: "owner",
      viewer_relationship: "owner",
    },
    subscription: { enabled: true },
    last_sync: "2026-09-29T11:00:00Z",
    doc_count: 412,
    memory_count: 96,
    sync: {
      status: "success",
      run_id: "run-1",
      started_at: "2026-09-29T10:58:00Z",
      finished_at: "2026-09-29T11:00:00Z",
      error_message: null,
      progress: null,
    },
    connection_status: { state: "ready", reason: null },
    created_at: "2026-09-01T00:00:00Z",
    project_binding: { mode: "fixed", project_key: "PAY" },
    capabilities: FULL_CAPABILITIES,
    execution: { kind: "server", operation: null, immutable_config_fields: [] },
    enabled_for_me: true,
    pinned_for_me: false,
    sync_schedule: null,
    ...overrides,
  };
}

/** A leased local sync job with no timestamps beyond creation; override what a test needs. */
export function makeLocalAgentJob(overrides: Partial<LocalAgentJob> = {}): LocalAgentJob {
  return {
    job_id: "laj-1",
    workspace_id: "local",
    source_id: "src-teams",
    source_type: "teams",
    operation: "teams_sync",
    status: "leased",
    attempt_count: 1,
    payload: {},
    result: {},
    last_error: null,
    execution_owner_user_id: "local",
    leased_until: null,
    next_attempt_at: null,
    created_at: "2026-07-08T09:00:00Z",
    updated_at: "2026-07-08T09:00:00Z",
    finished_at: null,
    ...overrides,
  };
}
