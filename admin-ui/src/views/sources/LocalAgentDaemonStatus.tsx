import { Circle, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { timeAgo } from "@/utils/date";
import { useLocalAgentDaemonStatus } from "./localAgentDaemonStatusQuery";

const LOCAL_AGENT_DAEMON_COMMAND = "memforge adapter daemon run";

interface LocalAgentDaemonStatusProps {
  className?: string;
}

export function LocalAgentDaemonStatus({ className }: LocalAgentDaemonStatusProps) {
  const query = useLocalAgentDaemonStatus();
  const data = query.data;
  const isOnline = data?.status === "online";
  const containerClass = [
    "flex flex-wrap items-center gap-x-3 gap-y-1 rounded-md border bg-muted/30 px-3 py-2 text-xs",
    className ?? "",
  ].filter(Boolean).join(" ");

  if (query.isPending) {
    return (
      <div className={containerClass} role="status" aria-live="polite">
        <span className="flex items-center gap-2 font-medium text-muted-foreground">
          <Loader2 className="size-3 animate-spin" />
          Checking local sync...
        </span>
      </div>
    );
  }

  if (query.isError || !data) {
    return (
      <div className={containerClass} role="status" aria-live="polite">
        <span className="flex items-center gap-2 font-medium text-muted-foreground">
          <Circle className="size-2 fill-muted-foreground/60 text-muted-foreground/60" />
          Local sync status unavailable
        </span>
      </div>
    );
  }

  const lastSeenText = data.last_seen_at ? timeAgo(data.last_seen_at) : null;
  const dotClass = isOnline
    ? "size-2 fill-emerald-500 text-emerald-500"
    : "size-2 fill-amber-500 text-amber-500";
  const label = isOnline ? "Local sync ready" : "Local sync unavailable";
  const detail = isOnline
    ? lastSeenText ? `Last seen ${lastSeenText}` : null
    : "Start local sync on this device to push local sources.";

  return (
    <div className={containerClass} role="status" aria-live="polite">
      <span className="flex items-center gap-2 font-medium text-foreground">
        <Circle className={dotClass} aria-hidden="true" />
        {label}
      </span>
      {detail && <span className="text-muted-foreground">{detail}</span>}
      {!isOnline && (
        <span className="flex items-center gap-2">
          <code className="rounded bg-background px-1.5 py-0.5 font-mono text-[11px] text-foreground">
            {LOCAL_AGENT_DAEMON_COMMAND}
          </code>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={copyDaemonCommand}
          >
            Copy
          </Button>
        </span>
      )}
    </div>
  );
}

export function LocalAgentDaemonBadge() {
  const query = useLocalAgentDaemonStatus();
  const data = query.data;

  if (query.isPending) {
    return (
      <>
        <Loader2 className="size-2.5 animate-spin text-muted-foreground" aria-hidden="true" />
        <span className="rounded-full bg-muted px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
          Checking local sync
        </span>
      </>
    );
  }

  if (query.isError || !data || data.status !== "online") {
    return (
      <>
        <Circle className="size-2 fill-amber-500 text-amber-500" aria-hidden="true" />
        <span className="rounded-full border border-amber-200 bg-amber-50 px-2.5 py-0.5 text-xs font-medium text-amber-700 dark:border-amber-900 dark:bg-amber-900/30 dark:text-amber-200">
          Local sync unavailable
        </span>
      </>
    );
  }

  return (
    <>
      <Circle className="size-2 fill-emerald-500 text-emerald-500" aria-hidden="true" />
      <span className="rounded-full bg-secondary px-2.5 py-0.5 text-xs font-medium text-secondary-foreground">
        Local sync ready
      </span>
    </>
  );
}

function copyDaemonCommand() {
  if (typeof navigator === "undefined" || !navigator.clipboard) return;
  void navigator.clipboard.writeText(LOCAL_AGENT_DAEMON_COMMAND);
}
