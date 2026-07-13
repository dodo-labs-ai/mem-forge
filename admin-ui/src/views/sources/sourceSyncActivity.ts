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
  updatedAt?: string | null;
  finishedAt?: string | null;
}

export interface SourceSyncPresentation {
  message: string;
  detail: string;
  completed?: number;
  total?: number;
}

export function sourceSyncActivityFromLocalJob(job: LocalAgentJobStatusResponse): SourceSyncActivity {
  const leaseExpired = job.status === "leased"
    && job.leased_until != null
    && new Date(job.leased_until).getTime() <= Date.now();
  return {
    state: job.status === "queued"
      ? "queued"
      : job.status === "leased"
        ? leaseExpired ? "recovering" : "active"
        : job.status === "succeeded" ? "success" : "failed",
    progress: job.result?.progress,
    error: { message: job.result?.error || job.last_error },
    startedAt: job.created_at,
    updatedAt: job.updated_at,
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
    updatedAt: sync.progress_updated_at,
    finishedAt: sync.finished_at,
  };
}

export function selectSourceSyncActivity(
  sync: SyncStatus | null | undefined,
  localJob: LocalAgentJobStatusResponse | null | undefined,
  pending = false,
): SourceSyncActivity | undefined {
  if (sync && ["pending", "running", "recovering"].includes(sync.status)) {
    return sourceSyncActivityFromStatus(sync);
  }
  if (localJob && ["queued", "leased"].includes(localJob.status)) {
    return sourceSyncActivityFromLocalJob(localJob);
  }
  if (pending) return { state: "queued" };
  if (sync && localJob) {
    const serverActivity = sourceSyncActivityFromStatus(sync);
    const localActivity = sourceSyncActivityFromLocalJob(localJob);
    return activityTime(localActivity) > activityTime(serverActivity)
      ? localActivity
      : serverActivity;
  }
  if (sync) return sourceSyncActivityFromStatus(sync);
  if (localJob) return sourceSyncActivityFromLocalJob(localJob);
  return undefined;
}

function activityTime(activity: SourceSyncActivity): number {
  for (const value of [activity.finishedAt, activity.updatedAt, activity.startedAt]) {
    if (!value) continue;
    const parsed = new Date(value).getTime();
    if (Number.isFinite(parsed)) return parsed;
  }
  return Number.NEGATIVE_INFINITY;
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
    return { message: "Action needed", detail: safeFailureDetail(activity.error) };
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
    detail: [count, memories != null ? `${memories} new memories saved` : ""].filter(Boolean).join(" · "),
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

function safeFailureDetail(error: SourceSyncActivity["error"]): string {
  const messages = [error?.message, ...(error?.items ?? []).map((item) => item.error)]
    .filter((value): value is string => Boolean(value?.trim()));
  const normalized = messages.join(" ").toLowerCase();
  if (normalized.includes("teams") && (
    normalized.includes("session expired")
    || normalized.includes("no teams session")
    || normalized.includes("tokens")
    || normalized.includes("sign in")
  )) {
    return "Sign in to Teams in Chrome, then retry sync.";
  }
  if (normalized.includes("embedding provider unreachable")) {
    return "The embedding provider is unavailable. Check its connection, then retry.";
  }
  if (normalized.includes("llm provider unreachable") || (
    normalized.includes("litellm") && isConnectivityFailure(normalized)
  )) {
    return "The AI provider is unavailable. Check its connection, then retry.";
  }
  if (normalized.includes("rate limit") || normalized.includes("429")) {
    return "The source is temporarily rate limited. Wait a few minutes, then retry.";
  }
  if (normalized.includes("pdf export") || normalized.includes("did not produce a pdf")) {
    return "Some pages could not be exported. Check source access, then retry.";
  }
  if (normalized.includes("certificate_verify_failed") || normalized.includes("certificate verify")) {
    return "The source certificate could not be verified. Check the connection, then retry.";
  }
  return "Sync failed. Retry when ready.";
}

function isConnectivityFailure(value: string): boolean {
  return [
    "all connection attempts failed",
    "cannot connect to host",
    "connect call failed",
    "connect timeout",
    "connection refused",
    "connection timed out",
    "failed to connect",
    "network is unreachable",
  ].some((marker) => value.includes(marker));
}
