import { useMutation } from "@tanstack/react-query";
import { resourceClient } from "@/api/client";

/**
 * How often, and for how long, a save that stopped a sync waits for that sync
 * to end. A worker stops within a heartbeat; when its worker died, the run
 * ends only after its five-minute lease expires and another worker claims it.
 */
const RUN_END_POLL_MS = 2_000;
const RUN_END_TIMEOUT_MS = 7 * 60_000;

const ACTIVE_RUN_STATUSES = new Set(["pending", "running"]);

/**
 * The sync run a refused save waits for: the server refuses a Source change
 * while a sync of the Source runs (409 `source_sync_running`). Null for any
 * other error.
 */
export function runningSyncOfRefusedSave(error: unknown): string | null {
  if (typeof error !== "object" || error === null) return null;
  const response = (error as { response?: { status?: number; data?: { detail?: unknown } } }).response;
  const detail = response?.data?.detail;
  if (response?.status !== 409 || typeof detail !== "object" || detail === null) return null;
  const { code, run_id: runId } = detail as { code?: unknown; run_id?: unknown };
  return code === "source_sync_running" && typeof runId === "string" ? runId : null;
}

/**
 * Stops the Source's sync and resolves once the run has ended. A running
 * server sync ends at its worker's next heartbeat, within about 30 seconds.
 */
export async function stopSyncAndWaitForRunEnd(
  sourceId: string,
  runId: string,
  { pollMs = RUN_END_POLL_MS, timeoutMs = RUN_END_TIMEOUT_MS }: { pollMs?: number; timeoutMs?: number } = {},
): Promise<void> {
  await resourceClient.post(`/sources/${sourceId}/sync/stop`);
  const deadline = Date.now() + timeoutMs;
  for (;;) {
    const run = await resourceClient.get<{ status: string }>(`/sources/${sourceId}/sync-runs/${runId}`);
    if (!ACTIVE_RUN_STATUSES.has(run.data.status)) return;
    if (Date.now() >= deadline) {
      throw new Error("The sync has not stopped yet. Wait a moment, then save again.");
    }
    await new Promise((resolve) => setTimeout(resolve, pollMs));
  }
}

/**
 * The way out of a save refused because a sync of the Source runs: stop the
 * sync, wait until it has ended, then save again. `runId` is null when the
 * last save was not refused for that reason.
 */
export function useStopRunningSyncAndSave(
  sourceId: string | undefined,
  saveError: unknown,
  saveAgain: () => void,
) {
  const runId = sourceId ? runningSyncOfRefusedSave(saveError) : null;
  const stopAndSave = useMutation({
    mutationFn: ({ source, run }: { source: string; run: string }) => stopSyncAndWaitForRunEnd(source, run),
    onSuccess: saveAgain,
  });
  return {
    runId,
    pending: stopAndSave.isPending,
    error: stopAndSave.error,
    run() {
      if (sourceId && runId) stopAndSave.mutate({ source: sourceId, run: runId });
    },
  };
}
