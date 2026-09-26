import assert from "node:assert/strict";

import {
  presentSourceSyncActivity,
  selectSourceSyncActivity,
  sourceSyncActivityBlocksActions,
  sourceSyncActivityFromLocalJob,
  sourceSyncActivityIsActionable,
  sourceSyncActivityIsVisible,
  COMPLETED_SYNC_VISIBLE_MS,
} from "../src/views/sources/sourceSyncActivity.js";
import { teamsConversationCount } from "../src/views/sources/teamsSourceConfig.js";
import type { LocalAgentJobStatusResponse, SyncStatus } from "../src/api/types.js";

const localJob: LocalAgentJobStatusResponse = {
  job_id: "laj-1",
  operation: "teams_sync",
  status: "leased",
  result: {
    progress: {
      schema_version: 1,
      phase: "uploading",
      progress: { completed: 182, total: 194, unit: "message" },
      source_time_range: {
        start: "2026-07-08T09:00:00+00:00",
        end: "2026-07-08T09:00:00+00:00",
      },
    },
  },
  last_error: null,
};

assert.deepEqual(
  presentSourceSyncActivity(sourceSyncActivityFromLocalJob(localJob), "Microsoft Teams", "conversations"),
  {
    message: "Syncing Jul 8 messages",
    detail: "182 of 194 messages",
    completed: 182,
    total: 194,
  },
);

assert.deepEqual(
  presentSourceSyncActivity(
    {
      state: "active",
      progress: {
        schema_version: 1,
        phase: "discovering",
        progress: { completed: 86, unit: "page" },
      },
    },
    "Confluence",
    "pages",
  ),
  { message: "Finding pages", detail: "86 pages found so far" },
);

assert.deepEqual(
  presentSourceSyncActivity(
    {
      state: "active",
      progress: {
        schema_version: 1,
        phase: "fetching",
        progress: { completed: 554, total: 555, unit: "file" },
      },
    },
    "GitHub Repository",
    "files",
  ),
  {
    message: "Reading files",
    detail: "554 of 555 files",
    completed: 554,
    total: 555,
  },
);

assert.deepEqual(
  presentSourceSyncActivity(
    {
      state: "active",
      progress: {
        schema_version: 1,
        phase: "processing",
        progress: { completed: 31, total: 86, unit: "page" },
        counts: { memories_created: 104 },
      },
    },
    "Confluence",
    "pages",
  ),
  {
    message: "Creating memories from pages",
    detail: "31 of 86 pages · 104 new memories saved",
    completed: 31,
    total: 86,
  },
);

assert.deepEqual(
  presentSourceSyncActivity(
    {
      state: "active",
      progress: {
        schema_version: 1,
        phase: "recovering_derivations",
        progress: { completed: 0, total: 1, unit: "file" },
      },
    },
    "GitHub Repository",
    "files",
  ),
  {
    message: "Resuming memory creation",
    detail: "0 of 1 files · Working",
  },
);

assert.deepEqual(
  presentSourceSyncActivity(
    {
      state: "active",
      progress: {
        schema_version: 1,
        phase: "reconciling",
        progress: { completed: 0, total: 1, unit: "file" },
      },
    },
    "GitHub Repository",
    "files",
  ),
  {
    message: "Checking removed files",
    detail: "0 of 1 files",
    completed: 0,
    total: 1,
  },
);

const activeServerRun: SyncStatus = {
  status: "running",
  started_at: "2026-07-08T09:00:00+00:00",
  finished_at: null,
  error_message: null,
  progress: {
    schema_version: 1,
    phase: "processing",
    progress: { completed: 4, total: 10, unit: "page" },
  },
};
assert.equal(selectSourceSyncActivity({
  sync: activeServerRun,
  localJob,
})?.progress?.phase, "processing");

const completedLocalHandoff: LocalAgentJobStatusResponse = {
  ...localJob,
  status: "succeeded",
  result: {
    source_sync_run_id: "run-cloud-processing",
    progress: {
      schema_version: 1,
      phase: "uploading",
      progress: { completed: 0, total: 400, unit: "message" },
    },
  },
  finished_at: "2026-07-08T09:01:00Z",
};

const waitingForCloud = selectSourceSyncActivity({
  sync: {
    ...activeServerRun,
    run_id: "run-previous",
    status: "failed",
    finished_at: "2026-07-08T08:59:00Z",
  },
  localJob: completedLocalHandoff,
});
assert.deepEqual(
  presentSourceSyncActivity(waitingForCloud!, "Microsoft Teams", "conversations"),
  { message: "Processing in Cloud", detail: "Waiting for Cloud processing" },
);
assert.equal(sourceSyncActivityBlocksActions(waitingForCloud), true);

assert.equal(
  selectSourceSyncActivity({
    sync: {
      ...activeServerRun,
      run_id: "run-cloud-processing",
      status: "failed",
      finished_at: "2026-07-08T09:02:00Z",
    },
    localJob: completedLocalHandoff,
  })?.state,
  "failed",
);

assert.deepEqual(
  selectSourceSyncActivity({
    sync: {
      ...activeServerRun,
      status: "success",
      finished_at: "2026-07-08T09:00:00Z",
    },
    pending: true,
  }),
  { state: "queued" },
);

assert.equal(
  selectSourceSyncActivity({
    sync: {
      ...activeServerRun,
      status: "success",
      started_at: "2026-07-08T08:00:00Z",
      finished_at: "2026-07-08T09:00:00Z",
    },
    localJob: {
      ...localJob,
      status: "failed",
      created_at: "2026-07-08T10:00:00Z",
      updated_at: "2026-07-08T10:01:00Z",
      finished_at: "2026-07-08T10:01:00Z",
      last_error: "collection failed",
    },
  })?.state,
  "failed",
);

assert.equal(
  sourceSyncActivityIsActionable(
    {
      state: "failed",
    },
    false,
  ),
  false,
);
assert.equal(
  sourceSyncActivityIsActionable(
    {
      state: "failed",
    },
    true,
  ),
  true,
);
assert.equal(
  sourceSyncActivityIsActionable(
    {
      state: "active",
    },
    false,
  ),
  true,
);

assert.equal(
  sourceSyncActivityFromLocalJob({
    ...localJob,
    leased_until: "2000-01-01T00:00:00Z",
  }).state,
  "recovering",
);

assert.deepEqual(
  presentSourceSyncActivity(
    {
      state: "failed",
      error: {
        message: "request failed for /Users/alice/private?token=secret",
        items: [{ doc_id: "doc-1", title: "Payroll Secret", error: "token=secret" }],
      },
    },
    "Confluence",
    "pages",
  ),
  { message: "Action needed", detail: "Sync failed. Retry when ready." },
);

assert.deepEqual(
  presentSourceSyncActivity(
    sourceSyncActivityFromLocalJob({
      ...localJob,
      status: "failed",
      last_error: "local_agent_source_activity_epoch_stale",
    }),
    "GitHub Repository",
    "documents",
  ),
  {
    message: "Action needed",
    detail: "The source changed while this sync was waiting. Retry to sync its current state.",
  },
);

assert.deepEqual(
  presentSourceSyncActivity(
    {
      state: "failed",
      error: {
        message: (
          "GitHub Pages discovery reached Max Pages (200) before the configured "
          + "subtree was exhausted. Increase Max Pages or narrow the configured scope "
          + "with Subtree Root URL or Exclude URL Patterns, then retry."
        ),
      },
    },
    "GitHub Pages",
    "pages",
  ),
  {
    message: "Action needed",
    detail: (
      "GitHub Pages reached Max Pages. Increase Max Pages or narrow the configured "
      + "scope, then retry."
    ),
  },
);

assert.deepEqual(
  presentSourceSyncActivity(
    {
      state: "failed",
      error: { message: "Embedding provider unreachable: connection refused" },
    },
    "Confluence",
    "pages",
  ),
  {
    message: "Action needed",
    detail: "The embedding provider is unavailable. Check its connection, then retry.",
  },
);

assert.equal(
  teamsConversationCount({ conversation_ids: ["channel:1", "chat:2", "chat:2"] }),
  2,
);
assert.equal(
  teamsConversationCount({ channels: "A, B", group_chats: ["C"], individual_chats: ["D"] }),
  null,
);
assert.equal(teamsConversationCount({}), null);

console.log("source sync activity tests passed");

const completedSyncFinishedAt = "2026-07-08T11:02:00Z";
const completedSync = { state: "success" as const, finishedAt: completedSyncFinishedAt };
const completedSyncFinishedMs = new Date(completedSyncFinishedAt).getTime();
assert.equal(
  sourceSyncActivityIsVisible(completedSync, completedSyncFinishedMs + COMPLETED_SYNC_VISIBLE_MS),
  true,
);
assert.equal(
  sourceSyncActivityIsVisible(completedSync, completedSyncFinishedMs + COMPLETED_SYNC_VISIBLE_MS + 1),
  false,
);
assert.equal(sourceSyncActivityIsVisible({ state: "failed" }), true);
