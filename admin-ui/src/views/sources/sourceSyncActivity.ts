import type {
  LocalAgentJobStatusResponse,
  SyncProgressSnapshot,
  SyncProgressUnit,
  SyncStatus,
} from "../../api/types.js";

export type SourceSyncActivityState =
  | "queued"
  | "active"
  | "recovering"
  | "success"
  | "partial"
  | "failed";

export interface SourceSyncActivity {
  state: SourceSyncActivityState;
  progress?: SyncProgressSnapshot;
  error?: {
    message?: string | null;
    items?: Array<{ doc_id: string; title: string; error: string }>;
  };
  startedAt?: string | null;
  finishedAt?: string | null;
}

export interface SourceSyncPresentation {
  message: string;
  detail: string;
  completed?: number;
  total?: number;
}

export function sourceSyncActivityFromLocalJob(job: LocalAgentJobStatusResponse): SourceSyncActivity {
  return {
    state: job.status === "queued"
      ? "queued"
      : job.status === "leased"
        ? "active"
        : job.status === "succeeded" ? "success" : "failed",
    progress: job.result?.progress,
    error: { message: job.result?.error || job.last_error },
    startedAt: job.created_at,
    finishedAt: job.finished_at,
  };
}

export function sourceSyncActivityFromStatus(sync: SyncStatus): SourceSyncActivity {
  return {
    state: sync.status === "pending"
      ? "queued"
      : sync.status === "running"
        ? "active"
        : sync.status,
    progress: sync.progress ?? undefined,
    error: { message: sync.error_message, items: sync.failed_docs },
    startedAt: sync.started_at,
    finishedAt: sync.finished_at,
  };
}

export function selectSourceSyncActivity(
  sync: SyncStatus | null | undefined,
  localJob: LocalAgentJobStatusResponse | null | undefined,
): SourceSyncActivity | undefined {
  if (sync && ["pending", "running", "recovering"].includes(sync.status)) {
    return sourceSyncActivityFromStatus(sync);
  }
  if (localJob && ["queued", "leased"].includes(localJob.status)) {
    return sourceSyncActivityFromLocalJob(localJob);
  }
  if (sync) return sourceSyncActivityFromStatus(sync);
  if (localJob) return sourceSyncActivityFromLocalJob(localJob);
  return undefined;
}

export function presentSourceSyncActivity(
  activity: SourceSyncActivity,
  sourceName: string,
  fallbackItems: string,
): SourceSyncPresentation {
  if (activity.state === "queued") {
    return { message: "Waiting to sync", detail: "Queued" };
  }
  if (activity.state === "recovering") {
    return withProgress("Recovering sync", activity.progress, fallbackItems);
  }
  if (activity.state === "failed") {
    return { message: "Action needed", detail: cleanError(activity.error?.message) };
  }
  if (activity.state === "partial") {
    return withProgress("Partially synced", activity.progress, fallbackItems);
  }
  if (activity.state === "success") {
    return withProgress("Up to date", activity.progress, fallbackItems);
  }

  const snapshot = activity.progress;
  if (!snapshot) return { message: `Syncing ${fallbackItems}`, detail: "Working" };
  const label = progressLabel(snapshot.progress?.unit, fallbackItems);
  switch (snapshot.phase) {
    case "waiting_for_device":
      return { message: "Waiting for your device", detail: "Local sync queued" };
    case "connecting":
      return { message: `Connecting to ${sourceName}`, detail: "Checking access" };
    case "discovering":
      return withProgress(`Finding ${label}`, snapshot, fallbackItems, true);
    case "uploading": {
      const date = currentSourceDate(snapshot);
      return withProgress(
        date && snapshot.progress?.unit === "message"
          ? `Syncing ${date} messages`
          : `Sending ${label} to Cloud`,
        snapshot,
        fallbackItems,
      );
    }
    case "processing":
      return withProgress(`Creating memories from ${label}`, snapshot, fallbackItems);
    case "reconciling":
      return withProgress(`Checking removed ${label}`, snapshot, fallbackItems);
  }
}

function withProgress(
  message: string,
  snapshot: SyncProgressSnapshot | undefined,
  fallbackItems: string,
  discovered = false,
): SourceSyncPresentation {
  const progress = snapshot?.progress;
  const memories = snapshot?.counts?.memories_created;
  if (!progress) return { message, detail: "Working" };
  const label = progressLabel(progress.unit, fallbackItems);
  const count = progress.total != null && progress.total > 0
    ? `${progress.completed} of ${progress.total} ${label}`
    : `${progress.completed} ${label}${discovered ? " found so far" : ""}`;
  const presentation: SourceSyncPresentation = {
    message,
    detail: [count, memories != null ? `${memories} memories found` : ""].filter(Boolean).join(" · "),
  };
  if (progress.total != null && progress.total > 0) {
    presentation.completed = progress.completed;
    presentation.total = progress.total;
  }
  return presentation;
}

function progressLabel(unit: SyncProgressUnit | undefined, fallback: string): string {
  const labels: Record<SyncProgressUnit, string> = {
    item: "items",
    page: "pages",
    file: "files",
    issue: "issues",
    message: "messages",
    conversation: "conversations",
  };
  return unit ? labels[unit] : fallback;
}

function currentSourceDate(snapshot: SyncProgressSnapshot): string {
  const start = snapshot.source_time_range?.start;
  const end = snapshot.source_time_range?.end;
  if (!start || start !== end) return "";
  const parsed = new Date(start);
  if (!Number.isFinite(parsed.getTime())) return "";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  }).format(parsed);
}

function cleanError(value: string | null | undefined): string {
  const text = value?.trim() || "Sync failed. Retry when ready.";
  const normalized = text.toLowerCase();
  if (normalized.includes("teams") && (
    normalized.includes("session expired")
    || normalized.includes("no teams session")
    || normalized.includes("tokens")
    || normalized.includes("sign in")
  )) {
    return "Sign in to Teams in Chrome, then retry sync.";
  }
  return text;
}
