import type { ReactNode } from "react";
import { Popover as PopoverPrimitive } from "@base-ui/react/popover";
import { AlertCircle, Info, Loader2, Lock, Pause, Pin, Play, RefreshCw, SlidersHorizontal } from "lucide-react";
import type { Source, SourceCapabilities, SyncStatus } from "@/api/types";
import { StatusDot } from "@/components/admin/StatusBadge";
import { SourceSyncStatusCard } from "@/components/admin/SourceSyncStatusCard";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { SourceIcon } from "@/components/sources/SourceIcon";
import { cn } from "@/lib/utils";
import { formatDuration, timeAgo } from "@/utils/date";
import { sourceActionLayout } from "./sourceActions";
import { SourceReadinessAlert } from "./SourceReadinessAlert";
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
  can_change_access: false,
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
  enabledForMe,
  isSubscriptionPending,
  onConfigure,
  onSync,
  onResume,
  onShowDetails,
  onSubscriptionChange,
  actionsMenu,
  highlighted = false,
  onTogglePin,
  isPinPending = false,
  onRetryAccess,
  onRevertAccess,
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
  enabledForMe: boolean;
  isSubscriptionPending: boolean;
  onConfigure: () => void;
  onSync: () => void;
  onResume?: () => void;
  onShowDetails: () => void;
  onSubscriptionChange: (enabled: boolean) => void;
  actionsMenu: ReactNode;
  highlighted?: boolean;
  onTogglePin?: () => void;
  isPinPending?: boolean;
  onRetryAccess?: () => void;
  onRevertAccess?: () => void;
}) {
  const isPaused = source.status === "paused";
  const capabilities = source.capabilities ?? DEFAULT_CAPABILITIES;
  const localExecution = isLocalAgentBackedSource(source);
  const connectionRequiresAction = source.connection_status?.state === "action_required";
  const showReadinessAlert = !isPaused
    && capabilities.can_sync
    && (localExecution || connectionRequiresAction);
  const pausedSyncHint = "Source is paused. Resume the source to sync again.";
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
    <div
      id={`source-row-${source.id}`}
      tabIndex={-1}
      className={cn(
        "group/source-row space-y-3 p-4 transition-colors duration-700 focus:outline-none",
        highlighted && "bg-primary/5 ring-2 ring-inset ring-primary/30",
      )}
    >
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="flex min-w-0 items-start gap-3">
          <SourceIcon type={source.type} client={source.client} className="mt-0.5 size-5" />
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <h3 className="max-w-full break-words text-sm font-medium">{source.name}</h3>
              <span className="text-xs text-muted-foreground">{sourceLabel.name}</span>
              <SourceLifecycleIndicator status={source.status} />
              <SourceAccessAlertBadge source={source} />
            </div>
            <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground empty:hidden">
              <SourceAccessLabel source={source} />
              {showReadinessAlert && (
                <SourceReadinessAlert
                  localExecution={localExecution}
                  connectionStatus={source.connection_status}
                />
              )}
            </div>
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
            </div>
          </div>
        </div>

        <div className="ml-8 flex flex-wrap items-center justify-start gap-2 sm:ml-0 sm:shrink-0 sm:justify-end">
          {onTogglePin && (
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              disabled={isPinPending}
              aria-label={`${source.pinned_for_me ? "Unpin" : "Pin"} ${source.name}`}
              title={source.pinned_for_me ? "Unpin source" : "Pin source"}
              className={cn(
                "text-muted-foreground transition-opacity hover:text-foreground focus-visible:opacity-100",
                source.pinned_for_me || isPinPending
                  ? "opacity-100"
                  : "opacity-100 [@media(hover:hover)]:opacity-0 group-hover/source-row:opacity-100 group-focus-within/source-row:opacity-100",
              )}
              onClick={onTogglePin}
            >
              {isPinPending ? (
                <Loader2 className="size-4 animate-spin" aria-hidden="true" />
              ) : (
                <Pin className={cn("size-4", source.pinned_for_me && "fill-current")} />
              )}
            </Button>
          )}
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
              aria-label={`Configure ${source.name}`}
              disabled={isDeleting}
              onClick={onConfigure}
            >
              <SlidersHorizontal className="size-4" />
              <span className="hidden md:inline">{sourceActionLayout.primary.configure.label}</span>
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
                    : sourceActionLayout.primary.sync.label
              }
            >
              {isSyncing ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <RefreshCw className="size-4" />
              )}
              {isSyncing ? "Syncing" : sourceActionLayout.primary.sync.label}
            </Button>
          )}
          {actionsMenu}
        </div>
      </div>

      {isPaused && source.access_state !== "changing" && capabilities.can_sync && !isManaged && (
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

      {source.access_state === "changing" && source.access_transition && (
        <SourceAccessTransitionStatus
          source={source}
          onRetry={onRetryAccess}
          onRevert={onRevertAccess}
        />
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

function SourceAccessAlertBadge({ source }: { source: Source }) {
  const failed = source.access_transition?.status === "failed";
  if (source.access_state === "changing") {
    return (
      <Badge
        variant="outline"
        className={cn(
          "gap-1",
          failed
            ? "border-red-200 bg-red-50 text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-200"
            : "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200",
        )}
      >
        {failed ? <AlertCircle className="size-3" /> : <Loader2 className="size-3 animate-spin" />}
        {failed ? "Access change failed" : "Changing access"}
      </Badge>
    );
  }
  if (source.access_state === "orphaned_private") {
    return (
      <Badge variant="outline" className="gap-1 border-red-200 bg-red-50 text-red-700">
        <AlertCircle className="size-3" /> Owner required
      </Badge>
    );
  }
  return null;
}

function SourceAccessLabel({ source }: { source: Source }) {
  if (source.access_state === "changing" || source.access_state === "orphaned_private") {
    return null;
  }
  if (source.access_policy !== "private") return null;
  return (
    <span className="inline-flex items-center gap-1 font-medium text-foreground">
      <Lock className="size-3" aria-hidden="true" />
      Only me
    </span>
  );
}

function SourceAccessTransitionStatus({
  source,
  onRetry,
  onRevert,
}: {
  source: Source;
  onRetry?: () => void;
  onRevert?: () => void;
}) {
  const transition = source.access_transition;
  if (!transition) return null;
  const failed = transition.status === "failed";
  const total = transition.total_memories;
  const completed = Math.min(transition.processed_memories, total);
  const percentage = total > 0 ? Math.round((completed / total) * 100) : null;
  const target = transition.target_policy === "private" ? "Only me" : "workspace access";
  return (
    <div
      role="status"
      className={cn(
        "space-y-2 rounded-md border px-3 py-2 text-sm",
        failed
          ? "border-red-200 bg-red-50/70 text-red-900 dark:border-red-900 dark:bg-red-950/30 dark:text-red-100"
          : "border-amber-200 bg-amber-50/70 text-amber-900 dark:border-amber-900 dark:bg-amber-950/30 dark:text-amber-100",
      )}
    >
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="font-medium">
            {failed ? "Access change needs attention" : `Changing access to ${target}`}
          </div>
          <div className="mt-0.5 text-xs opacity-80">
            {failed
              ? transition.error_message || "Existing memories were not fully updated. The source remains owner-only."
              : total > 0
                ? `${completed} of ${total} memories updated`
                : "Preparing existing memories"}
          </div>
        </div>
        {failed && (onRetry || onRevert) && (
          <div className="flex gap-2">
            {onRevert && <Button type="button" size="sm" variant="outline" onClick={onRevert}>Revert</Button>}
            {onRetry && <Button type="button" size="sm" onClick={onRetry}>Retry</Button>}
          </div>
        )}
      </div>
      {!failed && percentage !== null && (
        <div className="h-1.5 overflow-hidden rounded-full bg-amber-200/60 dark:bg-amber-900">
          <div className="h-full rounded-full bg-amber-700 transition-all" style={{ width: `${percentage}%` }} />
        </div>
      )}
    </div>
  );
}

function SourceLifecycleIndicator({ status }: { status: Source["status"] }) {
  if (status === "active") {
    return <StatusDot status={status} />;
  }
  const isPaused = status === "paused";
  return (
    <Badge
      variant="outline"
      className={cn(
        isPaused && "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-900 dark:bg-amber-900/30 dark:text-amber-200",
      )}
    >
      {status}
    </Badge>
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
  return (
    <label
      className="inline-flex min-h-8 cursor-pointer items-center gap-1.5 px-1 text-muted-foreground transition-colors hover:text-foreground has-data-[disabled]:cursor-not-allowed has-data-[disabled]:opacity-60"
      title={`Include memories from "${sourceName}" in your searches and memory views`}
    >
      <Switch
        aria-label={`${enabled ? "Disable" : "Enable"} "${sourceName}" for me`}
        checked={enabled}
        disabled={pending}
        onCheckedChange={onChange}
      />
      {pending && <Loader2 className="size-3 animate-spin" aria-hidden="true" />}
    </label>
  );
}
