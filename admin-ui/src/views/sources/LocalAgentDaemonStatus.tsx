import { useQuery } from "@tanstack/react-query";
import { Circle, Loader2 } from "lucide-react";
import client from "@/api/client";
import type { LocalAgentDaemonStatusResponse } from "@/api/types";
import { Button } from "@/components/ui/button";
import { timeAgo } from "@/utils/date";

const LOCAL_AGENT_STATUS_ENDPOINT = "/api/cloud/local-agent/status";
const LOCAL_AGENT_STATUS_QUERY_KEY = ["local-agent-daemon-status"] as const;
const LOCAL_AGENT_STATUS_REFETCH_MS = 30_000;
const LOCAL_AGENT_DAEMON_COMMAND = "memforge adapter daemon run";

function useLocalAgentDaemonStatus() {
  return useQuery<LocalAgentDaemonStatusResponse>({
    queryKey: LOCAL_AGENT_STATUS_QUERY_KEY,
    queryFn: () => client.get(LOCAL_AGENT_STATUS_ENDPOINT).then((response) => response.data),
    refetchInterval: LOCAL_AGENT_STATUS_REFETCH_MS,
    refetchOnWindowFocus: true,
    staleTime: 15_000,
  });
}

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

  if (query.isPending || !data) {
    return (
      <div className={containerClass} role="status" aria-live="polite">
        <span className="flex items-center gap-2 font-medium text-muted-foreground">
          <Loader2 className="size-3 animate-spin" />
          Checking local daemon...
        </span>
      </div>
    );
  }

  if (query.isError) {
    return (
      <div className={containerClass} role="status" aria-live="polite">
        <span className="flex items-center gap-2 font-medium text-muted-foreground">
          <Circle className="size-2 fill-muted-foreground/60 text-muted-foreground/60" />
          Local daemon status unavailable
        </span>
      </div>
    );
  }

  const lastSeenText = data.last_seen_at ? timeAgo(data.last_seen_at) : null;
  const dotClass = isOnline
    ? "size-2 fill-emerald-500 text-emerald-500"
    : "size-2 fill-amber-500 text-amber-500";
  const label = isOnline ? "Local daemon online" : "Local daemon offline";
  const detail = isOnline
    ? lastSeenText ? `Last seen ${lastSeenText}` : null
    : "Start it on your machine to sync local sources.";

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

function copyDaemonCommand() {
  if (typeof navigator === "undefined" || !navigator.clipboard) return;
  void navigator.clipboard.writeText(LOCAL_AGENT_DAEMON_COMMAND);
}
