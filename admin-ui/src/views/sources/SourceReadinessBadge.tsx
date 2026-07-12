import {
  CircleAlert,
  Loader2,
  LogIn,
  MonitorCheck,
  Plug,
  TriangleAlert,
} from "lucide-react";
import type { SourceConnectionStatus } from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { useLocalAgentDaemonStatus } from "./localAgentDaemonStatusQuery";
import {
  resolveSourceReadiness,
  type LocalDaemonReadiness,
  type SourceReadiness,
} from "./sourceReadiness";

interface SourceReadinessBadgeProps {
  localExecution: boolean;
  connectionStatus?: SourceConnectionStatus | null;
}

const PRESENTATION: Record<SourceReadiness, {
  label: string;
  icon: typeof MonitorCheck;
  className: string;
}> = {
  checking_local_sync: {
    label: "Checking local sync",
    icon: Loader2,
    className: "text-muted-foreground",
  },
  local_sync_ready: {
    label: "Local sync ready",
    icon: MonitorCheck,
    className: "bg-secondary text-secondary-foreground",
  },
  local_sync_unavailable: {
    label: "Local sync unavailable",
    icon: CircleAlert,
    className: "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-900/30 dark:text-amber-200",
  },
  connection_ready: {
    label: "Connection ready",
    icon: Plug,
    className: "bg-secondary text-secondary-foreground",
  },
  sign_in_required: {
    label: "Sign in required",
    icon: LogIn,
    className: "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-900/30 dark:text-amber-200",
  },
  configuration_required: {
    label: "Finish setup",
    icon: CircleAlert,
    className: "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-900/30 dark:text-amber-200",
  },
  account_mismatch: {
    label: "Account mismatch",
    icon: TriangleAlert,
    className: "border-destructive/30 bg-destructive/10 text-destructive",
  },
};

export function SourceReadinessBadge({
  localExecution,
  connectionStatus,
}: SourceReadinessBadgeProps) {
  const query = useLocalAgentDaemonStatus(localExecution);
  const daemon: LocalDaemonReadiness | undefined = localExecution
    ? query.isPending
      ? "checking"
      : query.isError || query.data?.status !== "online"
        ? "unavailable"
        : "ready"
    : undefined;
  const readiness = resolveSourceReadiness({ localExecution, daemon, connectionStatus });
  if (readiness === null) return null;

  const presentation = PRESENTATION[readiness];
  const Icon = presentation.icon;

  return (
    <Badge variant="outline" className={cn("gap-1.5", presentation.className)}>
      <Icon
        className={cn("size-3", readiness === "checking_local_sync" && "animate-spin")}
        aria-hidden="true"
      />
      {presentation.label}
    </Badge>
  );
}
