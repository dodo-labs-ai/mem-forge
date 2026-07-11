import type { LocalAgentJobStatusResponse } from "../../api/types.js";

export type LocalAgentProgressState = "queued" | "leased" | "succeeded" | "failed";

export interface LocalAgentSyncProgress {
  state: LocalAgentProgressState;
  message: string;
  detail: string;
  completed?: number;
  total?: number;
}

export function localAgentProgressFromJob(
  job: LocalAgentJobStatusResponse,
  itemLabel: string,
): LocalAgentSyncProgress {
  if (job.status === "queued") {
    return {
      state: "queued",
      message: "Waiting for local daemon",
      detail: "Job queued",
    };
  }

  if (job.status === "leased") {
    if (job.operation === "teams_sync") {
      return teamsProgress(job);
    }
    return {
      state: "leased",
      message: `Local daemon is syncing ${itemLabel}`,
      detail: "Working on your device",
    };
  }

  if (job.status === "failed") {
    return {
      state: "failed",
      message: "Action needed",
      detail: cleanLocalAgentJobError(job.result?.error || job.last_error || ""),
    };
  }

  const counts = job.result?.counts ?? {};
  const checked = numberValue(counts.selected);
  const pushed = numberValue(counts.pushed);
  const skipped = numberValue(counts.skipped_existing);
  const failed = numberValue(counts.failed);
  const messages = numberValue(job.result?.messages);
  const checkedDetail = checked > 0 ? `${checked} ${itemLabel} checked` : "";
  const skippedDetail = skipped > 0 ? `${skipped} unchanged` : "";
  const failedDetail = failed > 0 ? `${failed} failed` : "";
  const detail = [checkedDetail, skippedDetail, failedDetail].filter(Boolean).join(" · ");

  if (job.operation === "teams_sync") {
    const range = dateRangeDetail(job.result?.date_from, job.result?.date_to);
    const teamsDetail = [
      messages > 0 ? `${messages} messages${pushed > 0 ? "" : " checked"}` : "",
      range,
    ].filter(Boolean).join(" · ");
    return {
      state: "succeeded",
      message: pushed > 0 ? "Sent new Teams messages to Cloud" : "Up to date",
      detail: teamsDetail,
    };
  }

  if (pushed > 0) {
    return {
      state: "succeeded",
      message: `Sent ${pushed} changed ${itemLabel} to Cloud`,
      detail,
    };
  }

  return {
    state: "succeeded",
    message: "Up to date",
    detail,
  };
}

function teamsProgress(job: LocalAgentJobStatusResponse): LocalAgentSyncProgress {
  const progress = job.result?.progress;
  if (!progress || progress.stage === "connecting" || progress.stage === "reading") {
    return {
      state: "leased",
      message: progress?.stage === "connecting" ? "Connecting to Teams" : "Reading Teams messages",
      detail: progress?.stage === "connecting"
        ? "Checking your Teams session"
        : dateRangeDetail(progress?.date_from, progress?.date_to) || "Checking recent conversations",
    };
  }

  if (progress.stage === "starting_processing") {
    const sent = numberValue(progress.messages);
    const range = dateRangeDetail(progress.date_from, progress.date_to);
    return {
      state: "leased",
      message: "Starting memory extraction",
      detail: [sent > 0 ? `${sent} messages sent` : "Upload complete", range].filter(Boolean).join(" · "),
    };
  }

  const current = numberValue(progress.current);
  const total = numberValue(progress.total);
  const messages = numberValue(progress.messages);
  const processedMessages = numberValue(progress.processed_messages);
  const currentDate = formatSyncDate(progress.current_date);
  return {
    state: "leased",
    message: currentDate ? `Syncing ${currentDate} messages` : "Sending Teams messages to Cloud",
    detail: [
      processedMessages > 0 && messages > 0
        ? `${processedMessages} of ${messages} messages`
        : messages > 0 ? `${messages} messages found` : "Preparing messages",
    ].filter(Boolean).join(" · "),
    completed: processedMessages || current,
    total: messages || total,
  };
}

function dateRangeDetail(from: string | null | undefined, to: string | null | undefined): string {
  const start = formatSyncDate(from);
  const end = formatSyncDate(to);
  if (!start && !end) return "";
  if (!start || start === end) return start || end;
  return `${start}–${end}`;
}

function formatSyncDate(value: string | null | undefined): string {
  if (!value) return "";
  const parsed = new Date(value);
  if (!Number.isFinite(parsed.getTime())) return "";
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    timeZone: "UTC",
  }).format(parsed);
}

export function localAgentProgressMessage(progress: LocalAgentSyncProgress): string {
  return progress.detail ? `${progress.message} · ${progress.detail}` : progress.message;
}

export function teamsConversationCount(config: Record<string, unknown>): number | null {
  const canonical = stringList(config.conversation_ids);
  const values = canonical.length > 0
    ? canonical
    : [
        ...stringList(config.channels),
        ...stringList(config.group_chats),
        ...stringList(config.individual_chats),
      ];
  return values.length > 0 ? new Set(values).size : null;
}

function stringList(value: unknown): string[] {
  if (typeof value === "string") {
    return value.split(",").map((item) => item.trim()).filter(Boolean);
  }
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }
  return [];
}

function cleanLocalAgentJobError(value: string): string {
  const text = value.trim();
  const normalized = text.toLowerCase();
  if (
    normalized.includes("teams")
    && (
      normalized.includes("session expired")
      || normalized.includes("no teams session")
      || normalized.includes("tokens")
      || normalized.includes("sign in")
    )
  ) {
    return "Sign in to Teams in Chrome, then retry sync.";
  }
  return text || "Local daemon could not sync this source.";
}

function numberValue(value: number | null | undefined): number {
  return typeof value === "number" && Number.isFinite(value) ? Math.max(0, value) : 0;
}
