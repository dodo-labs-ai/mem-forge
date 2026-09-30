import assert from "node:assert/strict";
import {
  presentSourceSyncActivity,
  selectSourceSyncActivity,
  sourceSyncActivityFromLocalJob,
  sourceSyncActivityFromStatus,
  sourceSyncActivityIsVisible,
  sourceSyncControl,
  COMPLETED_SYNC_VISIBLE_MS,
} from "./sourceSyncActivity";
import { makeLocalAgentJob } from "@/test/sourceFixtures";
import type { LocalAgentJob as LocalAgentJobStatusResponse, SyncStatus } from "./types";
import { test } from "vitest";

test("selects and presents source sync activity", () => {
  const localJob = makeLocalAgentJob({
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
  });

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
      detail: "31 of 86 pages, 104 new memories saved",
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
      detail: "0 of 1 files, still working",
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
  assert.deepEqual(sourceSyncControl(waitingForCloud, false), { label: "Syncing", enabled: false });

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
});

const RETRY_DELAY_MS = 60 * 60_000;

test("offers Retry now for a sync waiting for its automatic retry", () => {
  const nextAttemptAt = new Date(Date.now() + RETRY_DELAY_MS).toISOString();
  const serverRun: SyncStatus = {
    status: "pending",
    run_id: "ssr-1",
    started_at: null,
    finished_at: null,
    error_message: "Rate limit exceeded",
    progress: null,
    next_attempt_at: nextAttemptAt,
  };
  assert.deepEqual(sourceSyncControl(sourceSyncActivityFromStatus(serverRun), false), {
    label: "Retry now",
    enabled: true,
    retryTarget: { execution_kind: "source_sync_run", execution_id: "ssr-1" },
  });

  const localJob = makeLocalAgentJob({ job_id: "laj-9", status: "queued", next_attempt_at: nextAttemptAt });
  assert.deepEqual(sourceSyncControl(sourceSyncActivityFromLocalJob(localJob), false), {
    label: "Retry now",
    enabled: true,
    retryTarget: { execution_kind: "local_agent_job", execution_id: "laj-9" },
  });
});

test("names the work that already covers a new sync request", () => {
  const retryTimePassed = new Date(Date.now() - RETRY_DELAY_MS).toISOString();
  assert.deepEqual(sourceSyncControl({ state: "queued" }, false), { label: "Sync queued", enabled: false });
  assert.deepEqual(
    sourceSyncControl({ state: "queued", nextAttemptAt: retryTimePassed }, false),
    { label: "Sync queued", enabled: false },
  );
  assert.deepEqual(sourceSyncControl({ state: "active" }, false), { label: "Syncing", enabled: false });
  assert.deepEqual(sourceSyncControl({ state: "recovering" }, false), { label: "Recovering", enabled: false });
});

test("offers Sync now when nothing is queued or running", () => {
  for (const activity of [undefined, { state: "success" as const }, { state: "partial" as const }, { state: "failed" as const }]) {
    assert.deepEqual(sourceSyncControl(activity, false), { label: "Sync now", enabled: true });
  }
  assert.deepEqual(sourceSyncControl(undefined, true), { label: "Starting", enabled: false });
});

test("notes a finished sync that skipped its deletion check", () => {
  const finishedSync: SyncStatus = {
    status: "success",
    run_id: "ssr-2",
    started_at: "2026-07-08T10:00:00Z",
    finished_at: "2026-07-08T10:01:00Z",
    error_message: null,
    progress: { schema_version: 1, phase: "processing", progress: { completed: 3, unit: "issue" } },
    absence_check_skipped_reason: "Jira search total changed during pagination",
  };
  assert.deepEqual(presentSourceSyncActivity(sourceSyncActivityFromStatus(finishedSync), "Jira", "issues"), {
    message: "Up to date",
    detail: "3 issues",
    notice: "Deletion check skipped: Jira search total changed during pagination",
  });
  assert.equal(
    presentSourceSyncActivity(
      sourceSyncActivityFromStatus({ ...finishedSync, absence_check_skipped_reason: null }),
      "Jira",
      "issues",
    ).notice,
    undefined,
  );
});
