import type { ReactNode } from "react";
import { Popover as PopoverPrimitive } from "@base-ui/react/popover";
import { Info, Loader2, Pause, Play, SlidersHorizontal } from "lucide-react";
import type { Source, SourceCapabilities, SourceOwnership, SyncStatus } from "@/api/types";
import { StatusDot } from "@/components/admin/StatusBadge";
import { SourceSyncStatusCard } from "@/components/admin/SourceSyncStatusCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { SourceIcon } from "@/components/sources/SourceIcon";
import { cn } from "@/lib/utils";
import { formatDuration, timeAgo } from "@/utils/date";
import { sourceActionLayout } from "./sourceActions";
import { LocalAgentDaemonBadge } from "./LocalAgentDaemonStatus";
import { isLocalAgentBackedSource } from "./localAgentSources";
import type { SourceSyncActivity } from "./sourceSyncActivity";
import { teamsConversationCount } from "./teamsSourceConfig";

/**
 * One row in the project-grouped sources view. Behaviour and DOM match the
 * flat list that preceded grouping: the row owns the source title, status,
 * counts, sync bar, and the primary configure / sync controls. The overflow
 * menu lives outside this component because its kebab affordance is shared
 * styling that other tests depend on staying in the page module.
 *
 * Action visibility is driven entirely by `source.capabilities` returned by
 * the backend. When the viewer can only subscribe (a non-manager looking at
 * a shared source), the row collapses to a subscription-oriented card
 * without management controls.
 */

export interface SourceRowLabels {
  name: string;
  subtitle?: string;
  description?: string;
}

const DEFAULT_CAPABILITIES: SourceCapabilities = {
  can_subscribe: false,
  can_configure: false,
  can_configure_connection: false,
  can_sync: false,
  can_force_resync: false,
  can_delete: false,
};

export function SourceRow({
  source,
  perGroupMemoryCount,
  isSyncing,
  syncActivity,
  isDeleting,
  isUpdatingStatus = false,
  isManaged,
  sourceLabel,
  itemLabel,
  authSessionLabel,
  enabledForMe,
  isSubscriptionPending,
  onConfigure,
  onSync,
  onResume,
  onShowDetails,
  onSubscriptionChange,
  actionsMenu,
}: {
  source: Source;
  perGroupMemoryCount: number;
  isSyncing: boolean;
  syncActivity?: SourceSyncActivity;
  isDeleting: boolean;
  isUpdatingStatus?: boolean;
  isManaged: boolean;
  sourceLabel: SourceRowLabels;
  itemLabel: string;
  authSessionLabel: (status: string) => string;
  enabledForMe: boolean;
  isSubscriptionPending: boolean;
  onConfigure: () => void;
  onSync: () => void;
  onResume?: () => void;
  onShowDetails: () => void;
  onSubscriptionChange: (enabled: boolean) => void;
  actionsMenu: ReactNode;
}) {
  const isPaused = source.status === "paused";
  const capabilities = source.capabilities ?? DEFAULT_CAPABILITIES;
  const showLocalAgentStatus = !isPaused && isLocalAgentBackedSource(source) && capabilities.can_sync;
  const pausedSyncHint = "Source is paused. Resume the source to sync again.";
  const ownershipText = formatOwnership(source.ownership);
  const configuredTeamsConversations = source.type === "teams"
    ? teamsConversationCount(source.config)
    : null;
  const displayedItemCount = source.type === "teams"
    ? configuredTeamsConversations
    : source.doc_count;
  const displayedItemLabel = source.type === "teams"
    ? displayedItemCount === 1 ? "conversation" : "conversations"
    : itemLabel;
  const durableSyncLabel = activeSyncLabel(source.sync?.status);

  return (
    <div className="space-y-3 p-4">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <SourceIcon type={source.type} client={source.client} className="mt-0.5 size-5" />
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="truncate text-sm font-medium">{source.name}</h3>
              {showLocalAgentStatus ? (
                <LocalAgentDaemonBadge />
              ) : (
                <SourceLifecycleBadge status={source.status} />
              )}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {sourceLabel.name}
              {sourceLabel.subtitle ? ` · ${sourceLabel.subtitle}` : ""}
            </p>
            {ownershipText && (
              <p className="mt-1 text-xs text-muted-foreground">{ownershipText}</p>
            )}
            {source.type === "agent_session" && (
              <p className="mt-1 text-xs text-muted-foreground">
                Populated automatically by the plugin. No manual sync needed.
              </p>
            )}
            <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-sm text-muted-foreground">
              {displayedItemCount !== null && (
                <span>
                  <span className="font-medium text-foreground">{displayedItemCount}</span> {displayedItemLabel}
                </span>
              )}
              <span>
                <span className="font-medium text-foreground">{perGroupMemoryCount}</span> memories
              </span>
              <span>
                {syncActivity && ["queued", "active", "recovering"].includes(syncActivity.state)
                  ? "Syncing now"
                  : durableSyncLabel ?? <LastSyncDetails source={source} itemLabel={itemLabel} />}
              </span>
              {source.sync_schedule?.enabled && (
                <span>
                  Auto sync: {formatScheduleInterval(source.sync_schedule.interval_minutes)}
                  {source.sync_schedule.next_run_at
                    ? `, next ${formatRelativeFuture(source.sync_schedule.next_run_at)}`
                    : ""}
                </span>
              )}
              {source.type === "jira" && source.auth_session && capabilities.can_configure_connection && (
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
          {capabilities.can_subscribe && (
            <SubscriptionToggle
              sourceName={source.name}
              enabled={enabledForMe}
              pending={isSubscriptionPending}
              onChange={onSubscriptionChange}
            />
          )}
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
          {capabilities.can_configure && (
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
          {capabilities.can_sync && !isManaged && (
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

      {isPaused && capabilities.can_sync && !isManaged && (
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

      <SourceSyncStatusCard
        activity={syncActivity}
        sourceName={sourceLabel.name}
        itemLabel={itemLabel}
        onRetry={isPaused || !capabilities.can_sync ? undefined : onSync}
      />
    </div>
  );
}

function SourceLifecycleBadge({ status }: { status: Source["status"] }) {
  const isPaused = status === "paused";
  return (
    <>
      <StatusDot
        status={status}
        className={isPaused ? "bg-amber-500" : undefined}
      />
      <Badge
        variant={isPaused ? "outline" : status === "active" ? "secondary" : "outline"}
        className={cn(
          isPaused && "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-900/30 dark:text-amber-200",
        )}
      >
        {status}
      </Badge>
    </>
  );
}

function activeSyncLabel(status: SyncStatus["status"] | undefined): string | null {
  if (status === "pending") return "Waiting to sync";
  if (status === "recovering") return "Recovering sync";
  if (status === "running") return "Syncing now";
  return null;
}

function LastSyncDetails({
  source,
  itemLabel,
}: {
  source: Source;
  itemLabel: string;
}) {
  const label = `Last synced: ${timeAgo(source.last_sync)}`;
  const sync = source.sync;

  if (!sync) {
    return <span>{label}</span>;
  }

  const duration =
    sync.started_at && sync.finished_at
      ? formatDuration(sync.started_at, sync.finished_at)
      : null;
  const failedCount = sync.docs_failed ?? 0;
  const isProblemStatus = sync.status === "partial" || sync.status === "failed";

  return (
    <PopoverPrimitive.Root>
      <PopoverPrimitive.Trigger
        render={
          <button
            type="button"
            className={cn(
              "inline-flex cursor-pointer items-center gap-1 rounded-sm text-muted-foreground outline-none transition-colors hover:text-foreground",
              "focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
            )}
            aria-label={`Show last sync details for ${source.name}`}
          />
        }
      >
        <span>{label}</span>
        <Info className="size-3.5 opacity-65" />
      </PopoverPrimitive.Trigger>
      <PopoverPrimitive.Portal>
        <PopoverPrimitive.Positioner sideOffset={6} align="start">
          <PopoverPrimitive.Popup
            className={cn(
              "z-50 w-72 rounded-lg border bg-popover p-3 text-sm text-popover-foreground shadow-md outline-none",
              "data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0",
            )}
          >
            <div className="space-y-2">
              <div className="font-medium text-foreground">Last sync details</div>
              <dl className="grid grid-cols-[1fr_auto] gap-x-4 gap-y-1 text-xs">
                <dt className="text-muted-foreground">Status</dt>
                <dd className={isProblemStatus ? "font-medium text-destructive" : "font-medium text-foreground"}>
                  {sync.status}
                </dd>
                <dt className="text-muted-foreground">{capitalize(itemLabel)} checked</dt>
                <dd className="font-medium text-foreground">{sync.docs_processed ?? "-"}</dd>
                <dt className="text-muted-foreground">Updated</dt>
                <dd className="font-medium text-foreground">{sync.docs_updated ?? "-"}</dd>
                {failedCount > 0 && (
                  <>
                    <dt className="text-muted-foreground">Failed</dt>
                    <dd className="font-medium text-destructive">{failedCount}</dd>
                  </>
                )}
                {duration && (
                  <>
                    <dt className="text-muted-foreground">Duration</dt>
                    <dd className="font-medium text-foreground">{duration}</dd>
                  </>
                )}
              </dl>
              {sync.error_message && (
                <p className="rounded-md bg-destructive/10 px-2 py-1.5 text-xs text-destructive">
                  {sync.error_message}
                </p>
              )}
            </div>
          </PopoverPrimitive.Popup>
        </PopoverPrimitive.Positioner>
      </PopoverPrimitive.Portal>
    </PopoverPrimitive.Root>
  );
}

function capitalize(value: string): string {
  if (!value) return value;
  return value[0].toUpperCase() + value.slice(1);
}

function formatScheduleInterval(minutes: number): string {
  if (minutes % 1440 === 0) {
    const days = minutes / 1440;
    return days === 1 ? "daily" : `every ${days} days`;
  }
  if (minutes % 60 === 0) {
    const hours = minutes / 60;
    return hours === 1 ? "hourly" : `every ${hours} hours`;
  }
  return `every ${minutes} minutes`;
}

function formatRelativeFuture(dateString: string): string {
  const date = new Date(dateString);
  const seconds = Math.ceil((date.getTime() - Date.now()) / 1000);
  if (Number.isNaN(seconds)) return "-";
  if (seconds <= 0) return "due now";
  if (seconds < 60) return "in less than 1m";
  const minutes = Math.ceil(seconds / 60);
  if (minutes < 60) return `in ${minutes}m`;
  const hours = Math.ceil(minutes / 60);
  if (hours < 24) return `in ${hours}h`;
  const days = Math.ceil(hours / 24);
  return `in ${days}d`;
}

function SubscriptionToggle({
  sourceName,
  enabled,
  pending,
  onChange,
}: {
  sourceName: string;
  enabled: boolean;
  pending: boolean;
  onChange: (enabled: boolean) => void;
}) {
  const label = pending ? "Saving..." : enabled ? "Enabled for me" : "Disabled for me";

  return (
    <label
      className="inline-flex h-8 cursor-pointer items-center gap-2 rounded-md border bg-background px-2.5 text-xs text-muted-foreground transition-colors hover:bg-muted/50 has-data-[disabled]:cursor-not-allowed has-data-[disabled]:opacity-60"
      title={`Toggle whether memories from "${sourceName}" appear in your views`}
    >
      <Switch
        aria-label={`${enabled ? "Disable" : "Enable"} "${sourceName}" for me`}
        checked={enabled}
        disabled={pending}
        onCheckedChange={onChange}
      />
      <span>{label}</span>
    </label>
  );
}

function formatOwnership(ownership: SourceOwnership | undefined): string {
  if (!ownership) return "";
  const creator = ownership.created_by_user_id;
  if (ownership.viewer_relationship === "creator") {
    return "Created by you";
  }
  if (ownership.viewer_relationship === "workspace_admin") {
    if (!creator) return "You manage as workspace admin";
    return `Created by ${creator} · You manage as workspace admin`;
  }
  if (creator) return `Created by ${creator}`;
  return "";
}
