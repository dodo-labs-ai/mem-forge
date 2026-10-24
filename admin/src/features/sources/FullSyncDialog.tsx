import { toast } from "sonner";
import { ConfirmDialog } from "@/patterns";
import { useFullSync } from "./api";
import { isLocalAgentBackedSource } from "./model/localAgentSources";
import type { Source } from "./model/types";

interface FullSyncDialogProps {
  /** The source to sync in full; null while the dialog is closed. */
  source: Source | null;
  onOpenChange: (open: boolean) => void;
}

/**
 * Confirms a full sync. A sync reads only what changed since the last one;
 * a full sync lists the whole source again and processes every document,
 * unchanged ones included, so it repairs content a sync left stale at the
 * cost of a first sync's model calls. Re-reading all documents is the
 * cheaper way to apply an adapter or model change, and the dialog says so
 * and where to find it, since Re-read all documents is offered only in the
 * row menu, not the drawer's.
 */
export function FullSyncDialog({ source, onOpenChange }: FullSyncDialogProps) {
  const fullSync = useFullSync();
  const origin = source && isLocalAgentBackedSource(source) ? "the device that syncs it" : "its provider";

  async function confirm() {
    await fullSync.mutateAsync(source!.id);
    toast.success(`Full sync started for ${source!.name}`);
  }

  return (
    <ConfirmDialog
      open={source !== null}
      onOpenChange={onOpenChange}
      title={`Run a full sync of ${source?.name ?? "this source"}?`}
      description={
        <>
          <span className="block">
            A full sync lists the whole source again from {origin} and processes every document with the models again,
            unchanged ones included. It takes as many model calls as the source's first sync.
          </span>
          <span className="mt-2 block">
            To process documents again after an adapter or model change, use Re-read all documents in the source's
            row menu instead. It counts the model calls before it starts.
          </span>
        </>
      }
      confirmLabel="Start full sync"
      onConfirm={confirm}
    />
  );
}
