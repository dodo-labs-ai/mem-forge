import { useEffect, useState } from "react";
import { AlertCircle, Check, Loader2, RotateCw } from "lucide-react";
import { cn } from "@/lib/utils";
import type { SourceSyncActivity } from "@/views/sources/sourceSyncActivity";
import { presentSourceSyncActivity } from "@/views/sources/sourceSyncActivity";

export function SourceSyncStatusCard({
  activity,
  sourceName,
  itemLabel,
  onRetry,
}: {
  activity: SourceSyncActivity | undefined;
  sourceName: string;
  itemLabel: string;
  onRetry?: () => void;
}) {
  const [nowMs, setNowMs] = useState(() => Date.now());
  useEffect(() => {
    if (!activity?.finishedAt) return;
    const id = window.setInterval(() => setNowMs(Date.now()), 1_000);
    return () => window.clearInterval(id);
  }, [activity?.finishedAt]);

  if (!activity) return null;
  if (activity.state === "success" && activity.finishedAt) {
    const age = nowMs - new Date(activity.finishedAt).getTime();
    if (age > 30_000) return null;
  }

  const presentation = presentSourceSyncActivity(activity, sourceName, itemLabel);
  const active = ["queued", "active", "recovering"].includes(activity.state);
  const failed = activity.state === "failed" || activity.state === "partial";
  const Icon = active ? Loader2 : failed ? AlertCircle : Check;
  const determinate = active && presentation.total != null && presentation.total > 0;
  const percentage = determinate
    ? Math.min(100, Math.round(((presentation.completed ?? 0) / presentation.total!) * 100))
    : 0;

  return (
    <div
      role={failed ? "alert" : "status"}
      aria-live={failed ? "assertive" : "polite"}
      className={cn(
        "flex flex-col gap-2 rounded-md px-3 py-2 text-sm",
        failed
          ? "bg-destructive/10 text-destructive"
          : activity.state === "success"
            ? "bg-emerald-50 text-emerald-900 dark:bg-emerald-950/30 dark:text-emerald-100"
            : "bg-muted text-muted-foreground",
      )}
    >
      <div className="flex min-w-0 items-center gap-2">
        <Icon className={cn("size-3.5 shrink-0", active && "animate-spin text-foreground")} />
        <span className={cn("shrink-0 font-medium", !failed && "text-foreground")}>{presentation.message}</span>
        {presentation.detail && <span className="min-w-0 truncate text-xs opacity-80">{presentation.detail}</span>}
        {failed && onRetry && (
          <button type="button" onClick={onRetry} className="ml-auto" aria-label="Retry sync">
            <RotateCw className="size-3.5" />
          </button>
        )}
      </div>
      {active && (
        <div
          role="progressbar"
          aria-label={`${sourceName} sync progress`}
          aria-valuemin={determinate ? 0 : undefined}
          aria-valuemax={determinate ? presentation.total : undefined}
          aria-valuenow={determinate ? Math.min(presentation.completed ?? 0, presentation.total!) : undefined}
          aria-valuetext={`${presentation.message}. ${presentation.detail}`}
          className="h-1 overflow-hidden rounded-full bg-background/80"
        >
          <div
            className={cn(
              "h-full rounded-full bg-foreground/70",
              determinate
                ? "transition-[width] duration-300"
                : "w-1/3 motion-safe:animate-pulse",
            )}
            style={determinate ? { width: `${percentage}%` } : undefined}
          />
        </div>
      )}
    </div>
  );
}
