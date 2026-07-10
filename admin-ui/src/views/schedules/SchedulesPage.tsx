import { useQuery, useQueryClient } from "@tanstack/react-query";
import { CalendarClock, ExternalLink, RefreshCw } from "lucide-react";
import { Link } from "react-router-dom";
import { resourceClient } from "@/api/client";
import type { Source, SourceSyncSchedule } from "@/api/types";
import { AsyncBoundary } from "@/components/admin/AsyncBoundary";
import { DataSurface } from "@/components/admin/DataSurface";
import { PageHeader } from "@/components/admin/PageHeader";
import { SourceIcon } from "@/components/sources/SourceIcon";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { formatDateTime, timeAgo } from "@/utils/date";

interface SourcesResponse {
  data?: Source[];
}

function normalizeSources(payload: SourcesResponse | Source[] | undefined): Source[] {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.data)) return payload.data;
  return [];
}

const MINUTES_PER_HOUR = 60;

function formatInterval(minutes: number): string {
  if (minutes % MINUTES_PER_HOUR === 0) {
    const hours = minutes / MINUTES_PER_HOUR;
    return hours === 1 ? "Every hour" : `Every ${hours}h`;
  }
  return `Every ${minutes}m`;
}

function formatNextRun(dateString: string | null | undefined): string {
  if (!dateString) return "-";
  const date = new Date(dateString);
  const diffMs = date.getTime() - Date.now();
  if (Number.isNaN(diffMs)) return "-";
  if (diffMs < 0) return "Overdue";
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return "< 1m";
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `in ${diffMin}m`;
  const diffHrs = Math.floor(diffMin / 60);
  if (diffHrs < 24) return `in ${diffHrs}h`;
  const diffDays = Math.floor(diffHrs / 24);
  return `in ${diffDays}d`;
}

function ScheduleCell({ schedule }: { schedule: SourceSyncSchedule | null | undefined }) {
  if (!schedule) {
    return <span className="text-sm text-muted-foreground">Not scheduled</span>;
  }
  return (
    <div className="flex flex-col gap-0.5">
      <span
        className={cn(
          "text-xs font-medium",
          schedule.enabled ? "text-green-600 dark:text-green-400" : "text-muted-foreground",
        )}
      >
        {schedule.enabled ? "Enabled" : "Disabled"}
      </span>
      <span className="text-sm text-muted-foreground">{formatInterval(schedule.interval_minutes)}</span>
    </div>
  );
}

function SyncStatusDot({ status }: { status: string }) {
  const classMap: Record<string, string> = {
    running: "bg-blue-500 animate-pulse",
    success: "bg-green-500",
    partial: "bg-yellow-500",
    failed: "bg-red-500",
  };
  return (
    <span
      className={cn("inline-block size-2 shrink-0 rounded-full", classMap[status] ?? "bg-muted")}
      title={status}
    />
  );
}

function scheduleSortKey(source: Source): [number, number] {
  const schedule = source.sync_schedule;
  if (!schedule) return [2, Number.MAX_SAFE_INTEGER];
  const nextRun = schedule.next_run_at ? new Date(schedule.next_run_at).getTime() : Number.MAX_SAFE_INTEGER;
  return [schedule.enabled ? 0 : 1, Number.isNaN(nextRun) ? Number.MAX_SAFE_INTEGER : nextRun];
}

function SourceRow({ source }: { source: Source }) {
  const schedule = source.sync_schedule;
  const relationship = source.ownership?.viewer_relationship?.replace("_", " ");
  const enabledForMe = source.enabled_for_me ?? source.subscription?.enabled ?? true;

  return (
    <tr className="border-b last:border-0 hover:bg-muted/40">
      <td className="px-4 py-3">
        <div className="flex items-center gap-2">
          <SourceIcon type={source.type} client={source.client} className="size-4 shrink-0" />
          <div className="min-w-0">
            <div className="truncate font-medium">{source.name}</div>
            <div className="truncate text-xs text-muted-foreground">
              {source.type}
              {relationship ? ` · ${relationship}` : ""}
              {!enabledForMe ? " · disabled for me" : ""}
            </div>
          </div>
        </div>
      </td>

      <td className="px-4 py-3">
        <span
          className={cn(
            "inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium",
            source.status === "active"
              ? "bg-green-50 text-green-700 dark:bg-green-950/30 dark:text-green-400"
              : "bg-muted text-muted-foreground",
          )}
        >
          {source.status}
        </span>
      </td>

      <td className="px-4 py-3">
        <ScheduleCell schedule={schedule} />
      </td>

      <td className="px-4 py-3 text-sm text-muted-foreground" title={formatDateTime(schedule?.next_run_at)}>
        {schedule?.enabled ? formatNextRun(schedule.next_run_at) : "-"}
      </td>

      <td className="px-4 py-3">
        <div className="flex items-center gap-1.5">
          {source.sync?.status && <SyncStatusDot status={source.sync.status} />}
          <span className="text-sm text-muted-foreground">{timeAgo(source.last_sync)}</span>
        </div>
      </td>

      <td className="px-4 py-3 text-sm text-muted-foreground">
        {source.doc_count.toLocaleString()}
        {source.memory_count !== undefined && (
          <span className="ml-1 text-xs">/ {source.memory_count.toLocaleString()} memories</span>
        )}
      </td>

      <td className="px-4 py-3 text-right">
        <Link
          to="/sources"
          className="inline-flex items-center gap-1 text-xs text-muted-foreground transition-colors hover:text-foreground"
        >
          Manage
          <ExternalLink className="size-3" />
        </Link>
      </td>
    </tr>
  );
}

export function SchedulesPage() {
  const queryClient = useQueryClient();

  const sourcesQuery = useQuery<SourcesResponse | Source[]>({
    queryKey: ["sources"],
    queryFn: () => resourceClient.get("/sources").then((r) => r.data),
    refetchInterval: (query) => {
      const sources = normalizeSources(query.state.data);
      return sources.some((s) => s.sync?.status === "running") ? 2000 : false;
    },
  });

  const sources = normalizeSources(sourcesQuery.data);
  const enabledScheduleCount = sources.filter((source) => source.sync_schedule?.enabled).length;
  const sortedSources = [...sources].sort((a, b) => {
    const [aGroup, aNext] = scheduleSortKey(a);
    const [bGroup, bNext] = scheduleSortKey(b);
    if (aGroup !== bGroup) return aGroup - bGroup;
    if (aNext !== bNext) return aNext - bNext;
    return a.name.localeCompare(b.name);
  });

  return (
    <div className="flex flex-col gap-6 p-6">
      <PageHeader
        title="Scheduled Syncs"
        description={`${enabledScheduleCount.toLocaleString()} enabled schedules across ${sources.length.toLocaleString()} configured sources.`}
        actions={
          <Button
            variant="outline"
            size="sm"
            onClick={() => queryClient.invalidateQueries({ queryKey: ["sources"] })}
            disabled={sourcesQuery.isFetching}
          >
            <RefreshCw className={cn("size-4", sourcesQuery.isFetching && "animate-spin")} />
            Refresh
          </Button>
        }
      />

      <AsyncBoundary
        isLoading={sourcesQuery.isLoading}
        isError={sourcesQuery.isError}
        isEmpty={!sourcesQuery.isLoading && sources.length === 0}
        error={sourcesQuery.error}
        onRetry={() => queryClient.invalidateQueries({ queryKey: ["sources"] })}
        empty={
          <DataSurface>
            <div className="flex flex-col items-center justify-center gap-2 px-6 py-16 text-center">
              <CalendarClock className="mb-1 size-8 text-muted-foreground/50" />
              <p className="text-sm font-medium">No sources configured</p>
              <p className="max-w-xs text-sm text-muted-foreground">
                Add a source in{" "}
                <Link to="/sources" className="underline underline-offset-2">
                  Sources
                </Link>{" "}
                to set up scheduled syncing.
              </p>
            </div>
          </DataSurface>
        }
      >
        <DataSurface>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-sm">
              <thead>
                <tr className="border-b text-xs font-medium text-muted-foreground">
                  <th className="px-4 py-3 text-left">Source</th>
                  <th className="px-4 py-3 text-left">Status</th>
                  <th className="px-4 py-3 text-left">Schedule</th>
                  <th className="px-4 py-3 text-left">Next Run</th>
                  <th className="px-4 py-3 text-left">Last Sync</th>
                  <th className="px-4 py-3 text-left">Docs / memories</th>
                  <th className="px-4 py-3 text-right"></th>
                </tr>
              </thead>
              <tbody>
                {sortedSources.map((source) => (
                  <SourceRow key={source.id} source={source} />
                ))}
              </tbody>
            </table>
          </div>
        </DataSurface>
      </AsyncBoundary>
    </div>
  );
}
