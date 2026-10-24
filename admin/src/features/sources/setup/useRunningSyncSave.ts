import { useEffect, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { unwrap, useApi } from "@/api";
import { sourceKeys } from "../api";
import { runningSyncOfRefusedSave, stopSyncAndWaitForRunEnd, type SyncStopCalls } from "./model/runningSyncSave";

/**
 * Saves a Source's settings, asking first while a sync of the Source runs:
 * "Stop the sync and save" stops it, waits until its run has ended, then
 * saves; "Wait" saves nothing and keeps the edits.
 *
 * `activeRunId` is the sync the live Sources list shows in progress. A save
 * the server refuses because a sync started meanwhile asks the same way.
 * Leaving the page, or choosing Wait, abandons a stop's wait, so the save it
 * waited for never happens; the sync it asked to stop still stops.
 */
export function useRunningSyncSave<Payload>({
  sourceId,
  activeRunId,
  save,
}: {
  /** The edited Source; absent while a Source is created, which never waits for a sync. */
  sourceId: string | undefined;
  activeRunId: string | null;
  /** Saves the payload. A refusal other than a running sync is the caller's to report. */
  save: (payload: Payload) => Promise<unknown>;
}) {
  const api = useApi();
  const queryClient = useQueryClient();
  const [pendingSave, setPendingSave] = useState<{ runId: string; payload: Payload } | null>(null);
  const stopWait = useRef<AbortController | null>(null);

  useEffect(() => () => stopWait.current?.abort(), []);

  const stopAndSave = useMutation({
    mutationFn: ({ source, run, signal }: { source: string; run: string; payload: Payload; signal: AbortSignal }) => {
      const calls: SyncStopCalls = {
        stop: () => unwrap(api.POST("/api/v1/sources/{source_id}/sync/stop", { params: { path: { source_id: source } } })),
        readRun: (runId, abort) =>
          unwrap(
            api.GET("/api/v1/sources/{source_id}/sync-runs/{run_id}", {
              params: { path: { source_id: source, run_id: runId } },
              signal: abort,
            }),
          ),
      };
      return stopSyncAndWaitForRunEnd(calls, run, { signal });
    },
    onSuccess: (_, { payload, signal }) => {
      if (signal.aborted) return;
      setPendingSave(null);
      attemptSave(payload);
    },
  });

  function abandonStopWait() {
    stopWait.current?.abort();
    stopWait.current = null;
    stopAndSave.reset();
  }

  function confirm(runId: string, payload: Payload) {
    abandonStopWait();
    setPendingSave({ runId, payload });
  }

  function attemptSave(payload: Payload) {
    save(payload).catch((error: unknown) => {
      const runId = sourceId ? runningSyncOfRefusedSave(error) : null;
      if (!runId) return;
      confirm(runId, payload);
      // The Sources list learns of the sync, so the page keeps saying it runs until it ends.
      void queryClient.invalidateQueries({ queryKey: sourceKeys.list });
    });
  }

  return {
    /** A sync of the Source is running, so saving stops it first. */
    running: Boolean(sourceId && (activeRunId || pendingSave)),
    /** A save is waiting for the person to stop the sync or wait for it. */
    confirming: pendingSave !== null,
    stopping: stopAndSave.isPending,
    /** Why stopping the sync or waiting for its run to end failed. */
    error: stopAndSave.error,
    save(payload: Payload) {
      if (sourceId && activeRunId) {
        confirm(activeRunId, payload);
        return;
      }
      attemptSave(payload);
    },
    stopAndSave() {
      if (!sourceId || !pendingSave) return;
      abandonStopWait();
      const controller = new AbortController();
      stopWait.current = controller;
      stopAndSave.mutate({ source: sourceId, run: pendingSave.runId, payload: pendingSave.payload, signal: controller.signal });
    },
    /** Saves nothing and keeps the edits. A stop already requested still stops the sync. */
    wait() {
      abandonStopWait();
      setPendingSave(null);
    },
  };
}
