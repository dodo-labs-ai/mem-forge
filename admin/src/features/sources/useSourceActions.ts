import { toast } from "sonner";
import { LOCAL_SYNC_START_COMMAND } from "@/features/local-sync";
import { errorMessage } from "@/lib/errors";
import { useSetPaused, useSetPinned, useSetUsedInSearches, useSyncSource } from "./api";
import { V1_SOURCES_PATH } from "./constants";
import type { SourceAttentionAction } from "./model/sourceAttention";
import type { Source } from "./model/types";

/** Runs row and drawer actions, reporting the outcome in a toast. */
export function useSourceActions({ onViewDetails }: { onViewDetails: (source: Source) => void }) {
  const sync = useSyncSource();
  const setPinned = useSetPinned();
  const setUsedInSearches = useSetUsedInSearches();
  const setPaused = useSetPaused();

  function report(promise: Promise<unknown>, success: string, failure: string) {
    promise.then(
      () => toast.success(success),
      (error: unknown) => toast.error(failure, { description: errorMessage(error) }),
    );
  }

  return {
    syncPending: (sourceId: string) => sync.isPending && sync.variables?.id === sourceId,
    syncNow(source: Source) {
      report(sync.mutateAsync(source), `Sync started for ${source.name}`, `Could not start a sync for ${source.name}`);
    },
    togglePinned(source: Source) {
      const pinned = !source.pinned_for_me;
      report(
        setPinned.mutateAsync({ sourceId: source.id, pinned }),
        pinned ? `${source.name} pinned` : `${source.name} unpinned`,
        "Could not change the pin",
      );
    },
    toggleUsedInSearches(source: Source) {
      const enabled = !(source.enabled_for_me ?? true);
      report(
        setUsedInSearches.mutateAsync({ sourceId: source.id, enabled }),
        enabled ? `${source.name} is used in your searches` : `${source.name} is no longer used in your searches`,
        "Could not change search use",
      );
    },
    togglePaused(source: Source) {
      const paused = source.status !== "paused";
      report(
        setPaused.mutateAsync({ sourceId: source.id, paused }),
        paused ? `${source.name} paused` : `${source.name} resumed`,
        paused ? "Could not pause the source" : "Could not resume the source",
      );
    },
    openInV1() {
      window.location.assign(V1_SOURCES_PATH);
    },
    runAttention(action: SourceAttentionAction, source: Source) {
      switch (action) {
        case "sync_now":
        case "retry":
          this.syncNow(source);
          return;
        case "start_local_sync":
          report(navigator.clipboard.writeText(LOCAL_SYNC_START_COMMAND), "Start command copied", "Could not copy");
          return;
        case "view_details":
          onViewDetails(source);
          return;
        case "sign_in":
        case "configure":
        case "configure_scope":
        case "assign_project":
          this.openInV1();
          return;
      }
    },
  };
}

export type SourceActions = ReturnType<typeof useSourceActions>;
