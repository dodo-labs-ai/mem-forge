import type { LocalAgentJob, SyncProgressSnapshot, SyncProgressUnit, SyncStatus } from "./types";

/** The queued work a retry starts: a local sync job, or a server run that needs no new collection. */
export type SourceSyncRetryTarget =
  | { execution_kind: "local_agent_job"; execution_id: string }
  | { execution_kind: "source_sync_run"; execution_id: string };

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
  nextAttemptAt?: string | null;
  retryTarget?: SourceSyncRetryTarget;
  /** Why a finished run removed nothing although it was due to check for deleted items. */
  absenceCheckSkippedReason?: string | null;
}

export interface SourceSyncPresentation {
  message: string;
  detail: string;
  /** Something a finished run left undone that does not make it fail. */
  notice?: string;
  configureScope?: boolean;
  completed?: number;
  total?: number;
}

/** The sync button of a source: what it says, whether it can be pressed, and what a press does. */
export interface SourceSyncControl {
  label: string;
  enabled: boolean;
  /** Set when a press starts the queued retry now instead of requesting a new sync. */
  retryTarget?: SourceSyncRetryTarget;
}

export function sourceSyncActivityFromLocalJob(job: LocalAgentJob): SourceSyncActivity {
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
    nextAttemptAt: job.next_attempt_at,
    retryTarget: job.status === "queued"
      ? { execution_kind: "local_agent_job", execution_id: job.job_id } : undefined,
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
    error: { message: sync.error_message, items: sync.failed_docs ?? undefined },
    startedAt: sync.started_at,
    updatedAt: sync.progress_updated_at,
    finishedAt: sync.finished_at,
    nextAttemptAt: sync.next_attempt_at,
    retryTarget: sync.status === "pending" && sync.run_id
      ? { execution_kind: "source_sync_run", execution_id: sync.run_id } : undefined,
    absenceCheckSkippedReason: sync.absence_check_skipped_reason,
  };
}

export function selectSourceSyncActivity({
  sync,
  localJob,
  pending = false,
}: {
  sync?: SyncStatus | null;
  localJob?: LocalAgentJob | null;
  pending?: boolean;
}): SourceSyncActivity | undefined {
  if (pending && !["pending", "running", "recovering"].includes(sync?.status ?? "")
    && !["queued", "leased"].includes(localJob?.status ?? "")) {
    return { state: "queued" };
  }
  const handedOffRunId = localJob?.status === "succeeded"
    ? localJob.result?.source_sync_run_id?.trim()
    : undefined;
  if (handedOffRunId && sync?.run_id === handedOffRunId) return sourceSyncActivityFromStatus(sync);
  if (pendingSourceSyncHandoff(localJob, sync)) {
    return {
      state: "active",
      progress: {
        schema_version: 1,
        phase: "waiting_for_cloud",
      },
      startedAt: localJob?.created_at,
      updatedAt: localJob?.updated_at,
    };
  }
  if (sync && ["pending", "running", "recovering"].includes(sync.status)) {
    return sourceSyncActivityFromStatus(sync);
  }
  if (localJob && ["queued", "leased"].includes(localJob.status)) {
    return sourceSyncActivityFromLocalJob(localJob);
  }
  if (pending) return { state: "queued" };

  const terminalActivities = [
    sync ? sourceSyncActivityFromStatus(sync) : undefined,
    localJob ? sourceSyncActivityFromLocalJob(localJob) : undefined,
  ].filter((activity): activity is SourceSyncActivity => activity != null);
  return terminalActivities.sort((left, right) => activityTime(right) - activityTime(left))[0];
}

export function pendingSourceSyncHandoff(
  job: LocalAgentJob | null | undefined,
  sync: SyncStatus | null | undefined,
): boolean {
  const runId = job?.status === "succeeded" ? job.result?.source_sync_run_id?.trim() : undefined;
  if (!runId || sync?.run_id === runId) return false;
  // A newer authoritative run can replace the receipt in the latest-run view.
  const createdAt = sync?.created_at || sync?.started_at;
  return !(createdAt && job?.finished_at && new Date(createdAt).getTime() > new Date(job.finished_at).getTime());
}

function activityTime(activity: SourceSyncActivity): number {
  for (const value of [activity.finishedAt, activity.updatedAt, activity.startedAt]) {
    if (!value) continue;
    const parsed = new Date(value).getTime();
    if (Number.isFinite(parsed)) return parsed;
  }
  return Number.NEGATIVE_INFINITY;
}

/**
 * The sync button for a source's current activity. A sync waiting for its
 * automatic retry is not running, and the server starts that same retry at
 * once on request, so the button offers Retry now. Other queued or running
 * work already covers a new request, so the button names that work and stays
 * disabled. `starting` is true while a sync or retry request is on its way.
 */
export function sourceSyncControl(activity: SourceSyncActivity | undefined, starting: boolean): SourceSyncControl {
  if (starting) return { label: "Starting", enabled: false };
  switch (activity?.state) {
    case "queued":
      return isWaitingForRetry(activity) && activity.retryTarget
        ? { label: "Retry now", enabled: true, retryTarget: activity.retryTarget }
        : { label: "Sync queued", enabled: false };
    case "active":
      return { label: "Syncing", enabled: false };
    case "recovering":
      return { label: "Recovering", enabled: false };
    default:
      return { label: "Sync now", enabled: true };
  }
}

function isWaitingForRetry(activity: SourceSyncActivity): boolean {
  return activity.state === "queued" && Boolean(activity.nextAttemptAt)
    && new Date(activity.nextAttemptAt!).getTime() > Date.now();
}

// A finished sync stays on screen briefly so its result is readable.
export const COMPLETED_SYNC_VISIBLE_MS = 30_000;

export function sourceSyncActivityIsVisible(
  activity: SourceSyncActivity,
  nowMs = Date.now(),
): boolean {
  if (activity.state !== "success" || !activity.finishedAt) return true;
  const finishedAtMs = new Date(activity.finishedAt).getTime();
  return !Number.isFinite(finishedAtMs) || nowMs - finishedAtMs <= COMPLETED_SYNC_VISIBLE_MS;
}

export function presentSourceSyncActivity(
  activity: SourceSyncActivity,
  sourceName: string,
  fallbackItems: string,
): SourceSyncPresentation {
  if (activity.state === "queued") {
    if (isWaitingForRetry(activity)) {
      return {
        message: "Waiting to retry",
        detail: `Next retry ${new Date(activity.nextAttemptAt!).toLocaleString()}`
          + (activity.error?.message ? `. ${safeFailureDetail(activity.error)}` : ""),
      };
    }
    if (activity.retryTarget?.execution_kind === "local_agent_job") {
      return { message: "Waiting for your device", detail: "Local sync queued" };
    }
    return { message: "Waiting to sync", detail: "Queued" };
  }
  if (activity.state === "recovering") {
    return withProgress("Recovering sync", activity.progress, fallbackItems);
  }
  if (activity.state === "failed") {
    const limit = repositoryFileLimit(activity.error);
    if (limit) return {
      message: "Sync scope exceeds file limit",
      detail: `Last sync matched ${limit.count} files, exceeding its configured limit of ${limit.limit}. Adjust Max Files or narrow the scope, then retry.`,
      configureScope: true,
    };
    return { message: "Action needed", detail: safeFailureDetail(activity.error) };
  }
  if (activity.state === "partial" || activity.state === "success") {
    const presentation = withProgress(
      activity.state === "partial" ? "Partially synced" : "Up to date",
      activity.progress,
      fallbackItems,
    );
    const reason = activity.absenceCheckSkippedReason?.trim();
    return reason ? { ...presentation, notice: `Deletion check skipped: ${reason}` } : presentation;
  }

  const snapshot = activity.progress;
  if (!snapshot) return { message: `Syncing ${fallbackItems}`, detail: "Working" };
  const label = progressLabel(snapshot.progress?.unit, fallbackItems);
  switch (snapshot.phase) {
    case "waiting_for_device":
      return { message: "Waiting for your device", detail: "Local sync queued" };
    case "waiting_for_cloud":
      return { message: "Processing in Cloud", detail: "Waiting for Cloud processing" };
    case "connecting":
      return { message: `Connecting to ${sourceName}`, detail: "Checking access" };
    case "discovering":
      return withProgress(`Finding ${label}`, snapshot, fallbackItems, true);
    case "fetching":
      return withProgress(`Reading ${label}`, snapshot, fallbackItems);
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
    case "recovering_derivations":
      return withIndeterminateWorkset(
        "Resuming memory creation",
        snapshot,
        fallbackItems,
      );
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
    detail: [count, memories != null ? `${memories} new memories saved` : ""].filter(Boolean).join(", "),
  };
  if (progress.total != null && progress.total > 0) {
    presentation.completed = progress.completed;
    presentation.total = progress.total;
  }
  return presentation;
}

function withIndeterminateWorkset(
  message: string,
  snapshot: SyncProgressSnapshot,
  fallbackItems: string,
): SourceSyncPresentation {
  const presentation = withProgress(message, snapshot, fallbackItems);
  return {
    message: presentation.message,
    detail: presentation.detail === "Working"
      ? presentation.detail
      : `${presentation.detail}, still working`,
  };
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

// How rate-limit failures read; a bare "429" is not matched, since identifiers in other errors contain those digits.
const RATE_LIMIT_MARKERS = ["rate limit", "ratelimit", "too many requests"];

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
  if (normalized.includes("github pages discovery") && normalized.includes("max pages")) {
    return (
      "GitHub Pages reached Max Pages. Increase Max Pages or narrow the configured "
      + "scope, then retry."
    );
  }
  if (normalized.includes("embedding provider unreachable")) {
    return "The embedding provider is unavailable. Check its connection, then retry.";
  }
  if (normalized.includes("llm provider unreachable") || (
    normalized.includes("litellm") && isConnectivityFailure(normalized)
  )) {
    return "The AI provider is unavailable. Check its connection, then retry.";
  }
  if (RATE_LIMIT_MARKERS.some((marker) => normalized.includes(marker))) {
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

function repositoryFileLimit(error: SourceSyncActivity["error"]): { count: string; limit: string } | null {
  const match = error?.message?.match(/^GitHub Repository (?:discovery|Internal network \/ VPN sync) matched ([0-9]{1,9}) files, exceeding max_files=([0-9]{1,9})$/);
  const [, count, limit] = match ?? [];
  return count && limit ? { count, limit } : null;
}
