import assert from "node:assert/strict";

import {
  localAgentProgressFromJob,
  localAgentProgressMessage,
  teamsConversationCount,
} from "../src/views/sources/localAgentSyncProgress.js";
import type { LocalAgentJobStatusResponse } from "../src/api/types.js";
import { currentWorkspaceId, requireCurrentWorkspaceId } from "../src/lib/workspace.js";

function job(
  status: LocalAgentJobStatusResponse["status"],
  overrides: Partial<LocalAgentJobStatusResponse> = {},
): LocalAgentJobStatusResponse {
  return {
    job_id: "laj-1",
    status,
    attempt_count: 0,
    leased_until: null,
    result: null,
    last_error: null,
    ...overrides,
  };
}

assert.deepEqual(
  localAgentProgressFromJob(job("queued"), "conversations"),
  {
    state: "queued",
    message: "Waiting for local daemon",
    detail: "Job queued",
  },
);

assert.deepEqual(
  localAgentProgressFromJob(
    job("leased", { attempt_count: 3, operation: "teams_sync" }),
    "conversations",
  ),
  {
    state: "leased",
    message: "Reading Teams messages",
    detail: "Checking recent conversations",
  },
);

assert.deepEqual(
  localAgentProgressFromJob(
    job("leased", {
      attempt_count: 3,
      operation: "teams_sync",
      result: {
        progress: {
          stage: "uploading",
          current: 7,
          total: 16,
          current_date: "2026-07-08T09:00:00+00:00",
          date_from: "2026-06-29T10:03:01+00:00",
          date_to: "2026-07-10T08:17:33+00:00",
          messages: 194,
          processed_messages: 72,
        },
      },
    }),
    "conversations",
  ),
  {
    state: "leased",
    message: "Syncing Jul 8 messages",
    detail: "72 of 194 messages",
    completed: 72,
    total: 194,
  },
);

assert.deepEqual(
  localAgentProgressFromJob(
    job("succeeded", {
      operation: "teams_sync",
      result: {
        counts: {
          selected: 22,
          pushed: 0,
          skipped_existing: 22,
          failed: 0,
          polls: 2,
        },
        messages: 194,
        date_from: "2026-06-29T10:03:01+00:00",
        date_to: "2026-07-10T08:17:33+00:00",
        sync_started: false,
      },
    }),
    "conversations",
  ),
  {
    state: "succeeded",
    message: "Up to date",
    detail: "194 messages checked · Jun 29–Jul 10",
  },
);

assert.deepEqual(
  localAgentProgressFromJob(
    job("succeeded", {
      operation: "teams_sync",
      result: {
        counts: {
          selected: 22,
          pushed: 9,
          skipped_existing: 13,
          failed: 0,
          polls: 2,
        },
        messages: 194,
        date_from: "2026-06-29T10:03:01+00:00",
        date_to: "2026-07-10T08:17:33+00:00",
        sync_started: true,
      },
    }),
    "conversations",
  ),
  {
    state: "succeeded",
    message: "Sent new Teams messages to Cloud",
    detail: "194 messages · Jun 29–Jul 10",
  },
);

assert.equal(
  localAgentProgressMessage(
    localAgentProgressFromJob(
      job("failed", {
        last_error: "Teams session expired. Connect Teams from the source wizard.",
      }),
      "conversations",
    ),
  ),
  "Action needed · Sign in to Teams in Chrome, then retry sync.",
);

assert.equal(
  teamsConversationCount({
    conversation_ids: "19:flexible-payroll@thread.v2",
  }),
  1,
);
assert.equal(
  teamsConversationCount({
    conversation_ids: ["19:a@thread.v2", "19:b@thread.v2", "19:a@thread.v2"],
  }),
  2,
);
assert.equal(
  teamsConversationCount({
    group_chats: ["19:legacy@thread.v2"],
  }),
  1,
);

Object.defineProperty(globalThis, "window", {
  configurable: true,
  value: { location: { search: "?workspace=payroll_agent" } },
});
assert.equal(currentWorkspaceId(), "payroll_agent");
assert.equal(requireCurrentWorkspaceId(), "payroll_agent");

Object.defineProperty(globalThis, "window", {
  configurable: true,
  value: { location: { search: "?workspace=%20" } },
});
assert.equal(currentWorkspaceId(), undefined);
assert.throws(
  () => requireCurrentWorkspaceId(),
  /Select a workspace before starting local sync\./,
);
