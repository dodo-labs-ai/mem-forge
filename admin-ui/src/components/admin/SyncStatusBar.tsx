import { useEffect, useState } from "react";
import { AlertCircle, Check, Loader2, RotateCw, X } from "lucide-react";
import type { SyncStatus } from "@/api/types";
import { buildFailureDetails } from "@/components/admin/syncFailureDetails";
import { formatDuration, formatElapsed } from "@/utils/date";

export function SyncStatusBar({
  sync,
  itemLabel = "documents",
  onRetry,
}: {
  sync: SyncStatus | null | undefined;
  itemLabel?: string;
  onRetry?: () => void;
}) {
  const [nowMs, setNowMs] = useState(0);
  const [dismissedKey, setDismissedKey] = useState<string | null>(null);

  useEffect(() => {
    if (!sync?.started_at && !sync?.finished_at) return;
    const id = window.setInterval(() => setNowMs(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [sync?.started_at, sync?.finished_at]);

  useEffect(() => {
    if (sync?.status !== "success" || !sync.finished_at) return;
    const key = sync.finished_at;
    const id = window.setTimeout(() => setDismissedKey(key), 10_000);
    return () => window.clearTimeout(id);
  }, [sync?.status, sync?.finished_at]);

  if (!sync) return null;

  const currentKey = sync.finished_at ?? sync.started_at ?? sync.status;
  if (dismissedKey === currentKey) return null;

  if (sync.status === "success" && sync.finished_at) {
    if (nowMs === 0) return null;
    const age = nowMs - new Date(sync.finished_at).getTime();
    if (age > 30_000) return null;
  }

  if (sync.status === "running") {
    const startMs = sync.started_at ? new Date(sync.started_at).getTime() : 0;
    const elapsed = startMs && nowMs > startMs ? Math.floor((nowMs - startMs) / 1000) : 0;
    const progressLabel = runningProgressLabel(sync, itemLabel);

    return (
      <div role="status" aria-live="polite" className="flex items-center gap-2 rounded-md bg-muted px-3 py-2 text-sm text-muted-foreground">
        <Loader2 className="size-3.5 shrink-0 animate-spin text-foreground" />
        <span>{progressLabel}</span>
        <span className="ml-auto text-xs tabular-nums">{formatElapsed(elapsed)}</span>
      </div>
    );
  }

  if (sync.status === "pending" || sync.status === "recovering") {
    const label = sync.status === "pending" ? "Waiting for a sync worker" : "Recovering interrupted sync";
    return (
      <div role="status" aria-live="polite" className="flex items-center gap-2 rounded-md bg-muted px-3 py-2 text-sm text-muted-foreground">
        <Loader2 className="size-3.5 shrink-0 animate-spin text-foreground" />
        <span>{label}</span>
      </div>
    );
  }

  if (sync.status === "success") {
    const duration =
      sync.started_at && sync.finished_at
        ? formatDuration(sync.started_at, sync.finished_at)
        : null;

    return (
      <div role="status" aria-live="polite" className="flex items-center gap-2 rounded-md bg-muted px-3 py-2 text-sm text-muted-foreground">
        <Check className="size-3.5 shrink-0 text-emerald-600" />
        <span>
          {typeof sync.docs_processed !== "number" ? (
            "Sync completed"
          ) : sync.docs_processed === 0 ? (
            "Up to date"
          ) : (
            <>
              <span className="font-medium text-foreground">{sync.docs_processed ?? 0}</span> {itemLabel} checked
              {" · "}
              <span className="font-medium text-foreground">{sync.docs_updated ?? 0}</span> updated
            </>
          )}
          {duration && ` · ${duration}`}
        </span>
        <button
          type="button"
          onClick={() => setDismissedKey(currentKey)}
          className="ml-auto text-muted-foreground/70 hover:text-foreground"
          aria-label="Dismiss sync status"
        >
          <X className="size-3" />
        </button>
      </div>
    );
  }

  if (sync.status === "partial" || sync.status === "failed") {
    const isPartial = sync.status === "partial";
    const message = sync.error_message || (isPartial ? "Sync partially completed" : "Sync failed");
    const failureDetails = buildFailureDetails(sync);

    return (
      <div role="alert" className="flex items-start gap-2 rounded-md bg-destructive/10 px-3 py-2 text-sm text-muted-foreground">
        <AlertCircle className="mt-0.5 size-3.5 shrink-0 text-destructive" />
        <div className="min-w-0 flex-1 space-y-1">
          <div className="whitespace-normal break-words" title={message}>
            {isPartial ? (
              <>
                <span className="font-medium text-foreground">{sync.docs_processed ?? 0}</span> {itemLabel}
                {" checked"}
                {typeof sync.docs_failed === "number" && sync.docs_failed > 0 && (
                  <>
                    {" · "}
                    <span className="font-medium text-foreground">{sync.docs_failed}</span> failed
                  </>
                )}
                {" · "}
                <span className="font-medium text-foreground">Action needed</span>
                {" · "}
              </>
            ) : (
              <>
                <span className="font-medium text-foreground">Action needed</span>
                {" · "}
              </>
            )}
            {message}
          </div>
          {failureDetails && (
            <details className="text-xs">
              <summary className="cursor-pointer text-muted-foreground/80 hover:text-foreground">
                Review failed documents
              </summary>
              <div className="mt-1 space-y-1.5">
                {failureDetails.groups.map((group) => (
                  <div key={group.label}>
                    <div className="font-medium text-foreground">{group.label}</div>
                    <div>{group.help}</div>
                    <div className="mt-0.5 text-muted-foreground/90">
                      {group.items.slice(0, 4).map((item) => item.title).join(", ")}
                      {group.items.length > 4 && ` (+${group.items.length - 4} more)`}
                    </div>
                    <details className="mt-0.5">
                      <summary className="cursor-pointer text-muted-foreground/80 hover:text-foreground">
                        Technical details
                      </summary>
                      <ul className="mt-0.5 space-y-0.5">
                        {group.items.slice(0, 4).map((item) => (
                          <li key={`${group.label}:${item.title}`} className="break-words">
                            <span className="font-medium text-foreground">{item.title}:</span>{" "}
                            {item.error}
                          </li>
                        ))}
                      </ul>
                    </details>
                  </div>
                ))}
              </div>
            </details>
          )}
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-1.5">
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              className="text-muted-foreground/70 hover:text-foreground"
              aria-label="Retry sync"
            >
              <RotateCw className="size-3" />
            </button>
          )}
          <button
            type="button"
            onClick={() => setDismissedKey(currentKey)}
            className="text-muted-foreground/70 hover:text-foreground"
            aria-label="Dismiss sync status"
          >
            <X className="size-3" />
          </button>
        </div>
      </div>
    );
  }

  return (
    <div role="status" aria-live="polite" className="flex items-center gap-2 rounded-md bg-muted px-3 py-2 text-sm text-muted-foreground">
      <AlertCircle className="size-3.5 shrink-0" />
      <span>Sync status: {sync.status}</span>
    </div>
  );
}

function runningProgressLabel(sync: SyncStatus, itemLabel: string) {
  if (sync.phase === "discovering") {
    return `Discovering ${itemLabel}`;
  }

  if (sync.phase === "detecting_deletions") {
    return `Checking removed ${itemLabel}`;
  }

  if (sync.phase === "processing") {
    const total = sync.docs_total ?? 0;
    if (total > 0) {
      return (
        <>
          Processed <span className="font-medium text-foreground">{sync.docs_processed ?? 0}</span>
          {" of "}
          <span className="font-medium text-foreground">{total}</span> {itemLabel}
          {typeof sync.docs_stored === "number" && sync.docs_stored > 0 && (
            <>
              {" · "}
              <span className="font-medium text-foreground">{sync.docs_stored}</span> stored {itemLabel}
            </>
          )}
        </>
      );
    }
    return `Processing ${itemLabel}`;
  }

  return `Syncing ${itemLabel}`;
}
