import { useEffect, useState } from "react";
import { AlertCircle, Check, Loader2, RotateCw, X } from "lucide-react";
import type { SyncStatus } from "@/api/types";
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

  if (sync.status === "success") {
    const duration =
      sync.started_at && sync.finished_at
        ? formatDuration(sync.started_at, sync.finished_at)
        : null;

    return (
      <div role="status" aria-live="polite" className="flex items-center gap-2 rounded-md bg-muted px-3 py-2 text-sm text-muted-foreground">
        <Check className="size-3.5 shrink-0 text-emerald-600" />
        <span>
          {sync.docs_processed === 0 ? (
            "Up to date"
          ) : (
            <>
              <span className="font-medium text-foreground">{sync.docs_processed}</span> {itemLabel} checked
              {" · "}
              <span className="font-medium text-foreground">{sync.docs_updated}</span> updated
              {" · "}
              <span className="font-medium text-foreground">{sync.memories_extracted}</span> new memories
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
                <span className="font-medium text-foreground">{sync.docs_processed}</span> {itemLabel}
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

  return null;
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
          Processed <span className="font-medium text-foreground">{sync.docs_processed}</span>
          {" of "}
          <span className="font-medium text-foreground">{total}</span> {itemLabel}
          {typeof sync.docs_stored === "number" && sync.docs_stored > 0 && (
            <>
              {" · "}
              <span className="font-medium text-foreground">{sync.docs_stored}</span> stored {itemLabel}
            </>
          )}
          {sync.memories_extracted > 0 && (
            <>
              {" · "}
              <span className="font-medium text-foreground">{sync.memories_extracted}</span> new memories
            </>
          )}
          {typeof sync.memories_stored === "number" && sync.memories_stored > 0 && (
            <>
              {" · "}
              <span className="font-medium text-foreground">{sync.memories_stored}</span> stored memories
            </>
          )}
        </>
      );
    }
    return `Processing ${itemLabel}`;
  }

  return `Syncing ${itemLabel}`;
}

function buildFailureDetails(sync: SyncStatus) {
  const failedDocs = sync.failed_docs ?? [];
  if (failedDocs.length === 0) return null;

  const groups = new Map<string, FailureGroup>();
  for (const doc of failedDocs) {
    const key = failureCategory(doc.error);
    if (!groups.has(key)) {
      groups.set(key, failureGroup(key));
    }
    groups.get(key)?.items.push({ title: doc.title, error: doc.error });
  }

  return { groups: Array.from(groups.values()) };
}

function failureCategory(error: string) {
  const normalized = error.toLowerCase();
  if (normalized.includes("rate limit") || normalized.includes("429")) return "rate_limit";
  if (normalized.includes("pdf export") || normalized.includes("did not produce a pdf")) return "pdf_export";
  if (normalized.includes("certificate_verify_failed") || normalized.includes("certificate verify")) return "certificate";
  return "other";
}

type FailureGroup = { label: string; help: string; items: { title: string; error: string }[] };

function failureGroup(key: string): FailureGroup {
  if (key === "rate_limit") {
    return {
      label: "Rate limited by Confluence",
      help: "Confluence temporarily limited export requests. Wait a few minutes, then retry the sync.",
      items: [],
    };
  }
  if (key === "pdf_export") {
    return {
      label: "PDF export unavailable",
      help: "Confluence did not return a usable PDF for these documents.",
      items: [],
    };
  }
  if (key === "certificate") {
    return {
      label: "Certificate verification failed",
      help: "The local Python runtime could not verify the Confluence certificate chain.",
      items: [],
    };
  }
  return {
    label: "Other sync errors",
    help: "These documents failed for another reason.",
    items: [],
  };
}
