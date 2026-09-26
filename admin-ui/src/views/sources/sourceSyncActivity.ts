import type {
  LocalAgentJobStatusResponse,
  SyncProgressSnapshot,
  SyncProgressUnit,
  SyncStatus,
} from "../../api/types.js";

export interface SourceSyncRetryTarget {
  execution_kind: "local_agent_job" | "source_sync_run";
  execution_id: string;
}

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
}

export interface SourceSyncPresentation {
  message: string;
  detail: string;
  configureScope?: boolean;
  completed?: number;
  total?: number;
}

export interface SourceSyncActivityPolicy {
  activeRowLabel: string;
  busyActionLabel: string;
  busyAriaLabel: string;
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
    error: { message: sync.error_message, items: sync.failed_docs },
    startedAt: sync.started_at,
    updatedAt: sync.progress_updated_at,
    finishedAt: sync.finished_at,
    nextAttemptAt: sync.next_attempt_at,
    retryTarget: sync.status === "pending" && sync.run_id
      ? { execution_kind: "source_sync_run", execution_id: sync.run_id } : undefined,
  };
}

export function selectSourceSyncActivity({
  sync,
  localJob,
  pending = false,
}: {
  sync?: SyncStatus | null;
  localJob?: LocalAgentJobStatusResponse | null;
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
  job: LocalAgentJobStatusResponse | null | undefined,
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

export function sourceSyncActivityBlocksActions(
  activity: SourceSyncActivity | undefined,
): boolean {
  return Boolean(activity && ["queued", "active", "recovering"].includes(activity.state));
}

export function sourceSyncActivityIsActionable(
  activity: SourceSyncActivity,
  canSync: boolean,
): boolean {
  return activity.state !== "failed" || canSync;
}

export function sourceSyncActivityPolicy(
  activity: SourceSyncActivity,
): SourceSyncActivityPolicy {
  return {
    activeRowLabel: activity.state === "queued"
      ? isWaitingForRetry(activity) ? "Waiting to retry" : "Waiting to sync"
      : "Syncing now",
    busyActionLabel: "Syncing",
    busyAriaLabel: "Sync in progress",
  };
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
          + (activity.error?.message ? ` · ${safeFailureDetail(activity.error)}` : ""),
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
    detail: [count, memories != null ? `${memories} new memories saved` : ""].filter(Boolean).join(" · "),
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
      : `${presentation.detail} · Working`,
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
  if (normalized.includes("local_agent_source_activity_epoch_stale")) {
    return "The source changed while this sync was waiting. Retry to sync its current state.";
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

function repositoryFileLimit(error: SourceSyncActivity["error"]): { count: string; limit: string } | null {
  const match = error?.message?.match(/^GitHub Repository (?:discovery|Internal network \/ VPN sync) matched ([0-9]{1,9}) files, exceeding max_files=([0-9]{1,9})$/);
  return match ? { count: match[1], limit: match[2] } : null;
}
