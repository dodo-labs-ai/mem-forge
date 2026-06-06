import type { ReactNode } from "react";
import { Info, Loader2, Play, SlidersHorizontal } from "lucide-react";
import type { Source } from "@/api/types";
import { StatusDot } from "@/components/admin/StatusBadge";
import { SyncStatusBar } from "@/components/admin/SyncStatusBar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SourceIcon } from "@/components/sources/SourceIcon";
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
  canConfigure,
  isManaged,
  sourceLabel,
  itemLabel,
  authSessionLabel,
  onConfigure,
  onSync,
  onShowDetails,
  actionsMenu,
}: {
  source: Source;
  perGroupMemoryCount: number;
  isSyncing: boolean;
  isDeleting: boolean;
  canConfigure: boolean;
  isManaged: boolean;
  sourceLabel: SourceRowLabels;
  itemLabel: string;
  authSessionLabel: (status: string) => string;
  onConfigure: () => void;
  onSync: () => void;
  onShowDetails: () => void;
  actionsMenu: ReactNode;
}) {
  return (
    <div className="space-y-3 p-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <SourceIcon type={source.type} client={source.client} className="mt-0.5 size-5" />
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="truncate text-sm font-medium">{source.name}</h3>
              <StatusDot status={source.status} />
              <Badge variant={source.status === "active" ? "secondary" : "outline"}>
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
            <Button type="button" disabled={isSyncing || isDeleting} onClick={onSync}>
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

      <SyncStatusBar sync={source.sync} itemLabel={itemLabel} onRetry={onSync} />
    </div>
  );
}
