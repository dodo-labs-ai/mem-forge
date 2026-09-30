import { Copy, Laptop } from "lucide-react";
import { toast } from "sonner";
import { cn } from "@/lib/cn";
import { formatRelative } from "@/lib/format";
import { toneDot } from "@/patterns";
import { Button } from "@/ui/button";
import { localSyncState, useLocalSyncStatus } from "./api";

export const LOCAL_SYNC_START_COMMAND = "memforge adapter daemon run";

const LABELS = {
  checking: "Checking local sync",
  online: "Local sync online",
  offline: "Local sync offline",
} as const;

/** Sidebar card: whether sources that sync from this machine can run. */
export function LocalSyncStatus() {
  const query = useLocalSyncStatus();
  const state = localSyncState(query);
  const lastSeen = query.data?.last_seen_at;

  async function copyCommand() {
    await navigator.clipboard.writeText(LOCAL_SYNC_START_COMMAND);
    toast.success("Start command copied");
  }

  return (
    <div className="space-y-2 rounded-lg border border-sidebar-border bg-background p-2.5">
      <div className="flex items-center gap-2">
        <Laptop aria-hidden className="size-4 text-muted-foreground" />
        <span className="flex-1 truncate text-xs font-medium text-foreground">{LABELS[state]}</span>
        <span
          aria-hidden
          className={cn(
            "size-2 rounded-full",
            state === "online" ? toneDot.ok : state === "offline" ? toneDot.idle : toneDot.live,
          )}
        />
      </div>
      {state === "offline" ? (
        <div className="space-y-1.5">
          <p className="text-xs text-muted-foreground">
            {lastSeen ? `Last seen ${formatRelative(lastSeen)}. ` : ""}Start it on this computer to sync local sources.
          </p>
          <Button size="xs" variant="outline" className="w-full" onClick={copyCommand}>
            <Copy />
            Copy start command
          </Button>
        </div>
      ) : null}
    </div>
  );
}
