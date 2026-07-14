import {
  CircleAlert,
  Loader2,
  LogIn,
  MonitorCheck,
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

interface SourceReadinessIndicatorProps {
  localExecution: boolean;
  connectionStatus?: SourceConnectionStatus | null;
}

const PRESENTATION: Record<SourceReadiness, {
  label: string;
  icon: typeof MonitorCheck;
  attention: boolean;
  className: string;
}> = {
  checking_local_sync: {
    label: "Checking local sync",
    icon: Loader2,
    attention: false,
    className: "text-muted-foreground",
  },
  local_sync_ready: {
    label: "Local sync ready",
    icon: MonitorCheck,
    attention: false,
    className: "text-muted-foreground",
  },
  local_sync_unavailable: {
    label: "Local sync unavailable",
    icon: CircleAlert,
    attention: true,
    className: "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-900/30 dark:text-amber-200",
  },
  sign_in_required: {
    label: "Sign in required",
    icon: LogIn,
    attention: true,
    className: "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-900/30 dark:text-amber-200",
  },
  configuration_required: {
    label: "Finish setup",
    icon: CircleAlert,
    attention: true,
    className: "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-900/30 dark:text-amber-200",
  },
  account_mismatch: {
    label: "Account mismatch",
    icon: TriangleAlert,
    attention: true,
    className: "border-destructive/30 bg-destructive/10 text-destructive",
  },
};

export function SourceReadinessIndicator({
  localExecution,
  connectionStatus,
}: SourceReadinessIndicatorProps) {
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
  const content = (
    <>
      <Icon
        className={cn("size-3", readiness === "checking_local_sync" && "animate-spin")}
        aria-hidden="true"
      />
      {presentation.label}
    </>
  );

  if (!presentation.attention) {
    return (
      <span className={cn("inline-flex items-center gap-1", presentation.className)}>
        {content}
      </span>
    );
  }

  return (
    <Badge variant="outline" className={cn("gap-1.5", presentation.className)}>
      {content}
    </Badge>
  );
}
