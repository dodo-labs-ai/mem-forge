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
    failed_items: null,
    scope_transition: null,
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

type FailedItem = NonNullable<Source["failed_items"]>["items"][number];
type ScopeTransition = NonNullable<Source["scope_transition"]>;

/** A scope change from space ENG to ENG and OPS that the last sync left open; override what a test needs. */
export function makeScopeTransition(overrides: Partial<ScopeTransition> = {}): ScopeTransition {
  return {
    id: "scope-transition-1",
    status: "failed",
    previous_scope: { sync_mode: "spaces", spaces: ["ENG"] },
    target_scope: { sync_mode: "spaces", spaces: ["ENG", "OPS"] },
    created_at: "2026-09-29T09:00:00Z",
    updated_at: "2026-09-29T11:00:00Z",
    members_committed: 117,
    listed_count: 120,
    excluded_scopes_due: false,
    error: "3 listed Documents have not committed under the transition",
    failing_items: { count: 0, retrying_after_upgrade: 0, items: [] },
    restore_config: { base_url: "https://wiki.example", sync_mode: "spaces", spaces: ["ENG"] },
    removes: "at_completion",
    ...overrides,
  };
}

/** A failed item as the server's failure record lists it; override what a test needs. */
export function makeFailedItem(overrides: Partial<FailedItem> = {}): FailedItem {
  return {
    document_id: "doc-1",
    title: "Payroll FAQ",
    stage: "fetch",
    error_code: null,
    message: "403 Forbidden",
    retryable: false,
    release: "1.4.0",
    failures_in_release: 1,
    first_failed_at: "2026-09-29T10:59:00Z",
    last_failed_at: "2026-09-29T10:59:00Z",
    retrying_after_upgrade: false,
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
