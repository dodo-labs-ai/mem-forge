import { useEffect } from "react";
import { RefreshCw } from "lucide-react";
import { toast } from "sonner";
import { CopyableCommand } from "@/patterns";
import { Button } from "@/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/ui/dialog";
import { DAEMON_START_COMMAND, DEVICE_UPGRADE_COMMAND } from "./model/deviceCommands";
import { currentDeviceUpdate, deviceUpdateProgress, type DeviceUpdate } from "./model/deviceUpdates";
import type { Source } from "./model/types";

/** How often an open update dialog reads the device's latest report again. */
export const DEVICE_UPDATE_POLL_MS = 5_000;

interface DeviceUpdateDialogProps {
  /** The device update the dialog was opened for; null while it is closed. */
  update: DeviceUpdate | null;
  /** The current source list, whose devices say how the update stands. */
  sources: readonly Source[];
  /** Reads the source list again, and with it each device's latest report. */
  onCheckAgain: () => unknown;
  onOpenChange: (open: boolean) => void;
}

/**
 * Asks the execution owner to update MemForge on one of their devices. It
 * names the server's version, which `memforge upgrade` installs, and the
 * Sources that wait for it. While open it reads the source list again, and
 * it closes itself once the device reports a matching acquisition contract,
 * which arrives with the daemon's next heartbeat after its restart. An
 * offline device is started first, so the upgrade command waits until then.
 */
export function DeviceUpdateDialog({ update, sources, onCheckAgain, onOpenChange }: DeviceUpdateDialogProps) {
  const progress = update ? deviceUpdateProgress(update, sources) : "waiting";

  useEffect(() => {
    if (!update) return;
    const timer = setInterval(onCheckAgain, DEVICE_UPDATE_POLL_MS);
    return () => clearInterval(timer);
  }, [update, onCheckAgain]);

  useEffect(() => {
    if (!update || progress !== "updated") return;
    toast.success("Updated. Syncing again.");
    onOpenChange(false);
  }, [update, progress, onOpenChange]);

  const shown = update ? currentDeviceUpdate(update, sources) : null;

  return (
    <Dialog open={update !== null && progress !== "updated"} onOpenChange={onOpenChange}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Update MemForge on {shown?.deviceName}</DialogTitle>
          <DialogDescription>
            {progress === "offline"
              ? `${shown?.deviceName} is offline. Start the computer, or start MemForge on it, and then update it.`
              : `This computer runs MemForge ${shown?.installedVersion ?? "an older version"}. The server runs ${shown?.requiredVersion}, and these sources sync only with the same reading format:`}
          </DialogDescription>
        </DialogHeader>
        {progress === "offline" ? (
          <CopyableCommand command={DAEMON_START_COMMAND} />
        ) : (
          <div className="space-y-3">
            <ul aria-label="Sources waiting for the update" className="list-disc space-y-0.5 pl-5 text-sm">
              {shown?.sources.map((source) => <li key={source.id}>{source.name}</li>)}
            </ul>
            <CopyableCommand command={DEVICE_UPGRADE_COMMAND} />
            <p className="text-sm text-muted-foreground">
              Syncs wait until the update. Nothing is removed, and they start on their own after it.
            </p>
          </div>
        )}
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Close
          </Button>
          <Button onClick={() => onCheckAgain()}>
            <RefreshCw aria-hidden />
            Check again
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
