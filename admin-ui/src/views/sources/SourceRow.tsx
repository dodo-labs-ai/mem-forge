import type { ReactNode } from "react";
import { Info, Loader2, Pause, Play, SlidersHorizontal } from "lucide-react";
import type { Source } from "@/api/types";
import { StatusDot } from "@/components/admin/StatusBadge";
import { SyncStatusBar } from "@/components/admin/SyncStatusBar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SourceIcon } from "@/components/sources/SourceIcon";
import { cn } from "@/lib/utils";
import { timeAgo } from "@/utils/date";
import { sourceActionLayout } from "./sourceActions";

/**
 * One row in the project-grouped sources view. Behaviour and DOM match the
 * flat list that preceded grouping: the row owns the source title, status,
 * counts, sync bar, and the primary configure / sync controls. The overflow
 * menu lives outside this component because its kebab affordance is shared
 * styling that other tests depend on staying in the page module.
 */

export interface SourceRowLabels {
  name: string;
  subtitle?: string;
  description?: string;
}

export function SourceRow({
  source,
  perGroupMemoryCount,
  isSyncing,
  isDeleting,
  isUpdatingStatus = false,
  canConfigure,
  isManaged,
  sourceLabel,
  itemLabel,
  authSessionLabel,
  onConfigure,
  onSync,
  onResume,
  onShowDetails,
  actionsMenu,
}: {
  source: Source;
  perGroupMemoryCount: number;
  isSyncing: boolean;
  isDeleting: boolean;
  isUpdatingStatus?: boolean;
  canConfigure: boolean;
  isManaged: boolean;
  sourceLabel: SourceRowLabels;
  itemLabel: string;
  authSessionLabel: (status: string) => string;
  onConfigure: () => void;
  onSync: () => void;
  onResume?: () => void;
  onShowDetails: () => void;
  actionsMenu: ReactNode;
}) {
  const isPaused = source.status === "paused";
  const pausedSyncHint = "Source is paused. Resume the source to sync again.";

  return (
    <div className="space-y-3 p-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <SourceIcon type={source.type} client={source.client} className="mt-0.5 size-5" />
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="truncate text-sm font-medium">{source.name}</h3>
              <StatusDot
                status={source.status}
                className={isPaused ? "bg-amber-500" : undefined}
              />
              <Badge
                variant={isPaused ? "outline" : source.status === "active" ? "secondary" : "outline"}
                className={cn(
                  isPaused && "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-900/30 dark:text-amber-200",
                )}
              >
                {source.status}
              </Badge>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {sourceLabel.name}
              {sourceLabel.subtitle ? ` · ${sourceLabel.subtitle}` : ""}
            </p>
            {source.type === "agent_session" && (
              <p className="mt-1 text-xs text-muted-foreground">
                Populated automatically by the plugin. No manual sync needed.
              </p>
            )}
            <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-sm text-muted-foreground">
              <span>
                <span className="font-medium text-foreground">{source.doc_count}</span> {itemLabel}
              </span>
              <span>
                <span className="font-medium text-foreground">{perGroupMemoryCount}</span> memories
              </span>
              <span>
                {source.sync?.status === "running"
                  ? "Syncing now"
                  : `Last synced: ${timeAgo(source.last_sync)}`}
              </span>
              {source.type === "jira" && source.auth_session && (
                <span
                  className={
                    source.auth_session.status === "active"
                      ? "text-emerald-600"
                      : "text-destructive"
                  }
                >
                  Browser session (local adapter): {authSessionLabel(source.auth_session.status)}
                </span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 sm:shrink-0">
          {isManaged && (
            <Button
              type="button"
              variant="outline"
              aria-label="View managed source details"
              disabled={isDeleting}
              onClick={onShowDetails}
            >
              <Info className="size-4" />
              <span className="hidden lg:inline">Details</span>
            </Button>
          )}
          {canConfigure && (
            <Button
              type="button"
              variant="outline"
              aria-label="Configure source"
              disabled={isDeleting}
              onClick={onConfigure}
            >
              <SlidersHorizontal className="size-4" />
              <span className="hidden lg:inline">{sourceActionLayout.primary[0].label}</span>
            </Button>
          )}
          {!isManaged && (
            <Button
              type="button"
              disabled={isSyncing || isDeleting || isPaused}
              onClick={onSync}
              title={isPaused ? pausedSyncHint : undefined}
              aria-label={
                isPaused
                  ? pausedSyncHint
                  : isSyncing
                    ? "Sync in progress"
                    : sourceActionLayout.primary[1].label
              }
            >
              {isSyncing ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Play className="size-4" />
              )}
              {isSyncing ? "Syncing" : sourceActionLayout.primary[1].label}
            </Button>
          )}
          {actionsMenu}
        </div>
      </div>

      {isPaused && !isManaged && (
        <div
          role="status"
          className="flex flex-col gap-2 rounded-md border border-amber-200 bg-amber-50/70 px-3 py-2 text-sm text-amber-900 dark:border-amber-900 dark:bg-amber-900/20 dark:text-amber-100 sm:flex-row sm:items-center sm:justify-between"
        >
          <div className="flex items-start gap-2">
            <Pause className="mt-0.5 size-4 shrink-0" />
            <span>
              Sync is paused. New {itemLabel} are not discovered, and existing memories stay in place.
            </span>
          </div>
          {onResume && (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onResume}
              disabled={isUpdatingStatus || isDeleting}
              className="border-amber-300 bg-background text-amber-900 hover:bg-amber-100 dark:border-amber-800 dark:text-amber-100 dark:hover:bg-amber-900/40"
              aria-label={`Resume sync for ${source.name}`}
            >
              {isUpdatingStatus ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Play className="size-4" />
              )}
              Resume
            </Button>
          )}
        </div>
      )}

      <SyncStatusBar sync={source.sync} itemLabel={itemLabel} onRetry={isPaused ? undefined : onSync} />
    </div>
  );
}
