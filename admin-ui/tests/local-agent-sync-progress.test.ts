import assert from "node:assert/strict";

import {
  presentSourceSyncActivity,
  selectSourceSyncActivity,
  sourceSyncActivityBlocksActions,
  sourceSyncActivityFromLocalJob,
  sourceSyncActivityIsActionable,
  sourceSyncActivityIsVisible,
  sourceSyncActivityPolicy,
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
      kind: "sync",
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
      kind: "sync",
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
      kind: "sync",
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

assert.deepEqual(
  selectSourceSyncActivity({
    sync: {
      ...activeServerRun,
      status: "success",
      finished_at: "2026-07-08T09:00:00Z",
    },
    pending: true,
  }),
  { kind: "sync", state: "queued" },
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

const activeMaintenance = selectSourceSyncActivity({
  sync: {
    ...activeServerRun,
    status: "partial",
    finished_at: "2026-07-08T10:00:00Z",
  },
  lifecycleMaintenance: {
    status: "running",
    created_at: "2026-07-08T11:00:00Z",
    started_at: "2026-07-08T11:01:00Z",
  },
});
assert.deepEqual(
  presentSourceSyncActivity(activeMaintenance!, "GitHub", "files"),
  { message: "Updating memories", detail: "Working" },
);
assert.equal(activeMaintenance?.kind, "memory_maintenance");
assert.equal(sourceSyncActivityBlocksActions(activeMaintenance), true);
assert.deepEqual(sourceSyncActivityPolicy(activeMaintenance!), {
  activeRowLabel: "Updating memories",
  busyActionLabel: "Updating",
  busyAriaLabel: "Memory maintenance in progress",
  canRetry: false,
});

const completedMaintenance = selectSourceSyncActivity({
  lifecycleMaintenance: {
    status: "completed",
    created_at: "2026-07-08T11:00:00Z",
    started_at: "2026-07-08T11:01:00Z",
    finished_at: "2026-07-08T11:02:00Z",
  },
});
assert.equal(
  sourceSyncActivityIsVisible(
    completedMaintenance!,
    new Date("2026-07-08T11:02:29Z").getTime(),
  ),
  true,
);
assert.equal(
  sourceSyncActivityIsVisible(
    completedMaintenance!,
    new Date("2026-07-08T11:02:31Z").getTime(),
  ),
  false,
);

const failedMaintenance = selectSourceSyncActivity({
  lifecycleMaintenance: {
    status: "failed",
    created_at: "2026-07-08T11:00:00Z",
    finished_at: "2026-07-08T11:02:00Z",
  },
});
assert.deepEqual(
  presentSourceSyncActivity(failedMaintenance!, "GitHub", "files"),
  {
    message: "Memory update needs attention",
    detail: "Memory maintenance failed. Review the maintenance details.",
  },
);
assert.equal(sourceSyncActivityBlocksActions(failedMaintenance), false);
assert.equal(sourceSyncActivityPolicy(failedMaintenance!).canRetry, false);
assert.equal(sourceSyncActivityIsActionable(failedMaintenance!, false), true);

assert.equal(
  sourceSyncActivityIsActionable(
    {
      kind: "sync",
      state: "failed",
    },
    false,
  ),
  false,
);
assert.equal(
  sourceSyncActivityIsActionable(
    {
      kind: "sync",
      state: "failed",
    },
    true,
  ),
  true,
);
assert.equal(
  sourceSyncActivityIsActionable(
    {
      kind: "sync",
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
      kind: "sync",
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
    {
      kind: "sync",
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
