import assert from "node:assert/strict";

import {
  presentSourceSyncActivity,
  selectSourceSyncActivity,
  sourceSyncActivityFromLocalJob,
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
assert.equal(selectSourceSyncActivity(activeServerRun, localJob)?.progress?.phase, "processing");

assert.deepEqual(
  selectSourceSyncActivity(
    {
      ...activeServerRun,
      status: "success",
      finished_at: "2026-07-08T09:00:00Z",
    },
    null,
    true,
  ),
  { state: "queued" },
);

assert.equal(
  selectSourceSyncActivity(
    {
      ...activeServerRun,
      status: "success",
      started_at: "2026-07-08T08:00:00Z",
      finished_at: "2026-07-08T09:00:00Z",
    },
    {
      ...localJob,
      status: "failed",
      created_at: "2026-07-08T10:00:00Z",
      updated_at: "2026-07-08T10:01:00Z",
      finished_at: "2026-07-08T10:01:00Z",
      last_error: "collection failed",
    },
  )?.state,
  "failed",
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
