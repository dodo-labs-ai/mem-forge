import { Clock, Lock, Users } from "lucide-react";
import { formatCount, formatInterval, formatRelative, formatUntil, pluralize } from "@/lib/format";
import { StatusBadge } from "@/patterns";
import { presentSourceSyncActivity, sourceSyncActivityIsVisible } from "./model/sourceSyncActivity";
import type { SourceRow } from "./model/sourceRows";
import { SourceIcon } from "./SourceIcon";
import { sourceRunsOn } from "./sourceRunsOn";

export function SourceNameCell({ row, typeLabel }: { row: SourceRow; typeLabel: string }) {
  const { source } = row;
  const usedInSearches = source.enabled_for_me ?? true;
  return (
    <div className="flex min-w-0 items-center gap-3">
      <span className="flex size-8 shrink-0 items-center justify-center rounded-md border bg-surface-subtle">
        <SourceIcon type={source.type} client={source.client} className="size-4" />
      </span>
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <span className="truncate font-medium text-foreground">{source.name}</span>
          {source.status === "paused" ? <StatusBadge tone="idle">Paused</StatusBadge> : null}
          {!usedInSearches ? <StatusBadge tone="idle">Not in my searches</StatusBadge> : null}
        </div>
        <div className="truncate text-xs text-muted-foreground">{typeLabel}</div>
      </div>
    </div>
  );
}

export function RunsCell({ row }: { row: SourceRow }) {
  const { label, icon: Icon } = sourceRunsOn(row);
  return (
    <span className="inline-flex items-center gap-1.5 text-sm text-subtle-foreground">
      <Icon aria-hidden className="size-3.5 text-muted-foreground" />
      {label}
    </span>
  );
}

export function AccessCell({ row }: { row: SourceRow }) {
  const personal = row.source.access_policy === "private";
  const Icon = personal ? Lock : Users;
  return (
    <span className="inline-flex items-center gap-1.5 text-sm text-subtle-foreground">
      <Icon aria-hidden className="size-3.5 text-muted-foreground" />
      {personal ? "Only me" : "Workspace"}
    </span>
  );
}

export function MemoriesCell({ row }: { row: SourceRow }) {
  return (
    <div className="text-right">
      <div className="tabular-nums text-foreground">{formatCount(row.memoryCount)}</div>
      <div className="text-xs text-muted-foreground">from {pluralize(row.source.doc_count, "item")}</div>
    </div>
  );
}

function nextRunLabel(nextRunAt: string): string {
  return new Date(nextRunAt).getTime() <= Date.now() ? `was due ${formatRelative(nextRunAt)}` : `next ${formatUntil(nextRunAt)}`;
}

export function LastSyncCell({ row }: { row: SourceRow }) {
  const { source, activity } = row;
  // The group header already says a source has no project; the row keeps its sync status.
  const attention = row.attention?.reason === "unmapped" ? null : row.attention;
  const schedule = source.sync_schedule;
  const visibleActivity = activity && sourceSyncActivityIsVisible(activity) ? activity : undefined;
  const live = visibleActivity && ["queued", "active", "recovering"].includes(visibleActivity.state);
  const presentation = live ? presentSourceSyncActivity(visibleActivity, source.name, "items") : null;

  return (
    <div className="min-w-0 space-y-0.5">
      {attention ? (
        <StatusBadge tone={attention.reason === "overdue" ? "warn" : "danger"}>
          {attention.title}
        </StatusBadge>
      ) : presentation ? (
        <StatusBadge tone="live">{presentation.message}</StatusBadge>
      ) : (
        <div className="text-sm text-subtle-foreground">
          {source.last_sync ? formatRelative(source.last_sync) : "Never synced"}
        </div>
      )}
      {schedule?.enabled ? (
        <div className="flex items-center gap-1 text-xs text-muted-foreground">
          <Clock aria-hidden className="size-3" />
          {formatInterval(schedule.interval_minutes)}
          {schedule.next_run_at && source.status !== "paused" ? `, ${nextRunLabel(schedule.next_run_at)}` : ""}
        </div>
      ) : null}
    </div>
  );
}
