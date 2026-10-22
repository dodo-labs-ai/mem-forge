import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { resourceClient } from "@/api/client";
import type { Source } from "@/api/types";

/**
 * How often, and for how long, a save that stopped a sync waits for that sync
 * to end. A worker stops within a heartbeat; when its worker died, the run
 * ends only after its five-minute lease expires and another worker claims it.
 */
const RUN_END_POLL_MS = 2_000;
const RUN_END_TIMEOUT_MS = 7 * 60_000;

const ACTIVE_RUN_STATUSES = new Set(["pending", "running"]);
const ACTIVE_SYNC_STATUSES = new Set(["pending", "running", "recovering"]);

/**
 * The server sync run of the Source in progress, which a configuration save
 * has to stop first. Null when the Source is not syncing.
 */
export function activeSyncRunOf(source: Source | null | undefined): string | null {
  const sync = source?.sync;
  return sync?.run_id && ACTIVE_SYNC_STATUSES.has(sync.status) ? sync.run_id : null;
}

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
 * Saving a Source's configuration while it syncs. The server refuses any
 * change but a pause while a sync of the Source runs, so a save during a sync
 * asks first: stop the sync, wait until it has ended and save, or keep the
 * dialog open with the edits and wait for the sync to finish.
 *
 * `activeRunId` is the sync the live Sources list shows in progress. A save
 * the server refuses because a sync started meanwhile asks the same way.
 */
export function useRunningSyncSave<Payload>({
  sourceId,
  activeRunId,
  saveError,
  save,
}: {
  sourceId: string | undefined;
  activeRunId: string | null | undefined;
  saveError: unknown;
  save: (payload: Payload) => Promise<unknown>;
}) {
  const queryClient = useQueryClient();
  const [pendingSave, setPendingSave] = useState<{ runId: string; payload: Payload } | null>(null);

  const stopAndSave = useMutation({
    mutationFn: ({ source, run }: { source: string; run: string; payload: Payload }) =>
      stopSyncAndWaitForRunEnd(source, run),
    onSuccess: (_, { payload }) => {
      setPendingSave(null);
      attemptSave(payload);
    },
  });

  function confirm(runId: string, payload: Payload) {
    stopAndSave.reset();
    setPendingSave({ runId, payload });
  }

  function attemptSave(payload: Payload) {
    save(payload).catch((error: unknown) => {
      const runId = sourceId ? runningSyncOfRefusedSave(error) : null;
      if (!runId) return;
      confirm(runId, payload);
      // The Sources list learns of the sync, so the dialog keeps saying it runs until it ends.
      void queryClient.invalidateQueries({ queryKey: ["sources"] });
    });
  }

  return {
    /** A sync of the Source is running, so saving stops it first. */
    running: Boolean(sourceId && (activeRunId || pendingSave)),
    /** A save is waiting for the person to stop the sync or wait for it. */
    confirming: pendingSave !== null,
    stopping: stopAndSave.isPending,
    error: stopAndSave.error,
    /** The last save was refused because a sync runs; the confirmation answers it, so it is not an error. */
    refusedForRunningSync: runningSyncOfRefusedSave(saveError) !== null,
    save(payload: Payload) {
      if (sourceId && activeRunId) {
        confirm(activeRunId, payload);
        return;
      }
      attemptSave(payload);
    },
    stopAndSave() {
      if (sourceId && pendingSave) {
        stopAndSave.mutate({ source: sourceId, run: pendingSave.runId, payload: pendingSave.payload });
      }
    },
    wait() {
      if (stopAndSave.isPending) return;
      stopAndSave.reset();
      setPendingSave(null);
    },
  };
}
