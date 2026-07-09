import type { LocalAgentJobStatusResponse } from "../../api/types.js";

export type LocalAgentProgressState = "queued" | "leased" | "succeeded" | "failed";

export interface LocalAgentSyncProgress {
  state: LocalAgentProgressState;
  message: string;
  detail: string;
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
    return {
      state: "leased",
      message: `Local daemon is syncing ${itemLabel}`,
      detail: job.attempt_count ? `Attempt ${job.attempt_count}` : "Job leased",
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
  const checkedDetail = checked > 0 ? `${checked} ${itemLabel} checked` : "";
  const skippedDetail = skipped > 0 ? `${skipped} unchanged` : "";
  const failedDetail = failed > 0 ? `${failed} failed` : "";
  const detail = [checkedDetail, skippedDetail, failedDetail].filter(Boolean).join(" · ");

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

export function localAgentProgressMessage(progress: LocalAgentSyncProgress): string {
  return progress.detail ? `${progress.message} · ${progress.detail}` : progress.message;
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
