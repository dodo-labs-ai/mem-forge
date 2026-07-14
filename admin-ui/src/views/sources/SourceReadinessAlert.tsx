import {
  CircleAlert,
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

interface SourceReadinessAlertProps {
  localExecution: boolean;
  connectionStatus?: SourceConnectionStatus | null;
}

const ALERT_PRESENTATION: Partial<Record<SourceReadiness, {
  label: string;
  icon: typeof MonitorCheck;
  className: string;
}>> = {
  local_sync_unavailable: {
    label: "Local sync unavailable",
    icon: CircleAlert,
    className: "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-900/30 dark:text-amber-200",
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

export function SourceReadinessAlert({
  localExecution,
  connectionStatus,
}: SourceReadinessAlertProps) {
  const query = useLocalAgentDaemonStatus(localExecution);
  const daemon: LocalDaemonReadiness | undefined = localExecution
    ? query.isPending
      ? "checking"
      : query.isError || query.data?.status !== "online"
        ? "unavailable"
        : "ready"
    : undefined;
  const readiness = resolveSourceReadiness({ localExecution, daemon, connectionStatus });
  const presentation = readiness === null ? undefined : ALERT_PRESENTATION[readiness];
  if (!presentation) return null;

  const Icon = presentation.icon;
  return (
    <Badge variant="outline" className={cn("gap-1.5", presentation.className)}>
      <Icon className="size-3" aria-hidden="true" />
      {presentation.label}
    </Badge>
  );
}
