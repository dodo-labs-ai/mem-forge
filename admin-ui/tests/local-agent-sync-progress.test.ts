import assert from "node:assert/strict";

import {
  localAgentProgressFromJob,
  localAgentProgressMessage,
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
        },
      },
    }),
    "conversations",
  ),
  {
    state: "leased",
    message: "Syncing Jul 8 messages",
    detail: "7 of 16 windows · 194 messages found",
    completed: 7,
    total: 16,
  },
);

assert.deepEqual(
  localAgentProgressFromJob(
    job("succeeded", {
      result: {
        counts: {
          selected: 22,
          pushed: 0,
          skipped_existing: 22,
          failed: 0,
          polls: 2,
        },
        sync_started: false,
      },
    }),
    "conversations",
  ),
  {
    state: "succeeded",
    message: "Up to date",
    detail: "22 conversations checked · 22 unchanged",
  },
);

assert.deepEqual(
  localAgentProgressFromJob(
    job("succeeded", {
      result: {
        counts: {
          selected: 22,
          pushed: 9,
          skipped_existing: 13,
          failed: 0,
          polls: 2,
        },
        sync_started: true,
      },
    }),
    "conversations",
  ),
  {
    state: "succeeded",
    message: "Sent 9 changed conversations to Cloud",
    detail: "22 conversations checked · 13 unchanged",
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
