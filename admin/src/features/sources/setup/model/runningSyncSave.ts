import { ApiError, HTTP_STATUS } from "@/api";
import type { Source } from "../../model/types";

/**
 * Saving a Source's settings while it syncs.
 *
 * The server refuses every change but a pause while a sync of the Source
 * runs: 409 `source_sync_running`, naming the run. The settings page then
 * asks first: stop the sync, wait until its run has ended and save, or keep
 * the edits and wait for the sync to finish.
 *
 * A worker stops at its next heartbeat. When its worker died, the run ends
 * only after its five-minute lease expires and another worker claims it, so
 * the wait polls every 2 s for up to 7 minutes.
 */
export const RUN_END_POLL_MS = 2_000;
export const RUN_END_TIMEOUT_MS = 7 * 60_000;

export const RUN_END_TIMEOUT_MESSAGE = "The sync has not stopped yet. Wait a moment, then save again.";

const ACTIVE_RUN_STATUSES = new Set(["pending", "running"]);
const ACTIVE_SYNC_STATUSES = new Set(["pending", "running", "recovering"]);

/** The server sync run of the Source in progress, which a save has to stop first; null when it is not syncing. */
export function activeSyncRunOf(source: Pick<Source, "sync"> | null | undefined): string | null {
  const sync = source?.sync;
  return sync?.run_id && ACTIVE_SYNC_STATUSES.has(sync.status) ? sync.run_id : null;
}

/** The sync run a refused save waits for, from 409 `source_sync_running`; null for any other error. */
export function runningSyncOfRefusedSave(error: unknown): string | null {
  if (!(error instanceof ApiError) || error.status !== HTTP_STATUS.conflict || error.code !== "source_sync_running") return null;
  const runId = error.detail.run_id;
  return typeof runId === "string" && runId !== "" ? runId : null;
}

/** The two server calls a stop makes: stop the Source's sync, and read where one run stands. */
export interface SyncStopCalls {
  /** `POST /sources/{id}/sync/stop`; its `run_id` is the run the stop reached. */
  stop: () => Promise<{ run_id: string | null }>;
  /** `GET /sources/{id}/sync-runs/{run_id}`. */
  readRun: (runId: string, signal?: AbortSignal) => Promise<{ status: string }>;
}

/**
 * Stops the Source's sync and resolves once the run has ended. It waits for
 * the run the stop reached, which is `runId` unless another sync started
 * meanwhile. Aborting `signal` stops the waiting, not the sync.
 */
export async function stopSyncAndWaitForRunEnd(
  calls: SyncStopCalls,
  runId: string,
  { pollMs = RUN_END_POLL_MS, timeoutMs = RUN_END_TIMEOUT_MS, signal }: { pollMs?: number; timeoutMs?: number; signal?: AbortSignal } = {},
): Promise<void> {
  const stopped = await calls.stop();
  const stoppedRunId = stopped.run_id ?? runId;
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    signal?.throwIfAborted();
    const run = await calls.readRun(stoppedRunId, signal);
    if (!ACTIVE_RUN_STATUSES.has(run.status)) return;
    if (Date.now() >= deadline) throw new Error(RUN_END_TIMEOUT_MESSAGE);
    await delay(pollMs, signal);
  }
}

function delay(ms: number, signal: AbortSignal | undefined): Promise<void> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(resolve, ms);
    signal?.addEventListener(
      "abort",
      () => {
        clearTimeout(timer);
        reject(signal.reason);
      },
      { once: true },
    );
  });
}
