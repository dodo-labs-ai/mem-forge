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
import {
  resolveLocalSourceReadiness,
  type LocalDaemonReadiness,
  type LocalSourceReadiness,
} from "./localSourceReadiness";
import { useLocalAgentDaemonStatus } from "./localAgentDaemonStatusQuery";

interface LocalSourceReadinessBadgeProps {
  connectionStatus?: SourceConnectionStatus | null;
}

const PRESENTATION: Record<LocalSourceReadiness, {
  label: string;
  icon: typeof MonitorCheck;
  className: string;
}> = {
  checking: {
    label: "Checking local sync",
    icon: Loader2,
    className: "text-muted-foreground",
  },
  ready: {
    label: "Local sync ready",
    icon: MonitorCheck,
    className: "bg-secondary text-secondary-foreground",
  },
  unavailable: {
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

export function LocalSourceReadinessBadge({
  connectionStatus,
}: LocalSourceReadinessBadgeProps) {
  const query = useLocalAgentDaemonStatus();
  const daemon: LocalDaemonReadiness = query.isPending
    ? "checking"
    : query.isError || query.data?.status !== "online"
      ? "unavailable"
      : "ready";
  const readiness = resolveLocalSourceReadiness({ daemon, connectionStatus });
  const presentation = PRESENTATION[readiness];
  const Icon = presentation.icon;

  return (
    <Badge
      variant="outline"
      className={cn("gap-1.5", presentation.className)}
    >
      <Icon
        className={cn("size-3", readiness === "checking" && "animate-spin")}
        aria-hidden="true"
      />
      {presentation.label}
    </Badge>
  );
}
