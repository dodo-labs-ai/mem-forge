import assert from "node:assert/strict";

import {
  localAgentProgressFromJob,
  localAgentProgressMessage,
} from "../src/views/sources/localAgentSyncProgress.js";
import type { LocalAgentJobStatusResponse } from "../src/api/types.js";

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
  localAgentProgressFromJob(job("leased", { attempt_count: 1 }), "conversations"),
  {
    state: "leased",
    message: "Local daemon is syncing conversations",
    detail: "Attempt 1",
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
