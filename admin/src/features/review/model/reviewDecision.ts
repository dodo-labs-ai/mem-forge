import { HTTP_STATUS, isApiErrorStatus } from "@/api";
import { errorMessage } from "@/lib/errors";
import type { Tone } from "@/patterns";
import type { ReviewAction, ReviewDecision, ReviewListItem, ReviewPresentation } from "./types";

/** Where a review stands, as the queue tabs name it. */
export type ReviewState = "waiting" | "expired" | "applied" | "kept";

export function reviewState(review: Pick<ReviewListItem, "status" | "is_stale">): ReviewState {
  if (review.status === "approved") return "applied";
  if (review.status === "rejected") return "kept";
  if (review.status === "stale" || review.is_stale) return "expired";
  return "waiting";
}

/** The two choices a review offers. */
export interface DecisionOptions {
  useLatest: ReviewAction;
  keepCurrent: ReviewAction;
}

export function decisionOptions(presentation: ReviewPresentation): DecisionOptions | null {
  const useLatest = presentation.actions.find((action) => action.decision === "approve");
  const keepCurrent = presentation.actions.find((action) => action.decision === "reject");
  return useLatest && keepCurrent ? { useLatest, keepCurrent } : null;
}

const NOTE_REQUIREMENT: Record<ReviewAction["key"], { required: string; optional: string }> = {
  use_latest_state: { required: "Required to use the latest state.", optional: "Optional when using the latest state." },
  keep_current_state: {
    required: "Required to keep the current state.",
    optional: "Optional when keeping the current state.",
  },
};

/** Which choices need a note, required ones first. */
export function noteHint(options: DecisionOptions): string {
  const actions = [options.keepCurrent, options.useLatest].sort(
    (left, right) => Number(right.requires_note) - Number(left.requires_note),
  );
  return actions
    .map((action) => NOTE_REQUIREMENT[action.key][action.requires_note ? "required" : "optional"])
    .join(" ");
}

export function canSubmitDecision(action: ReviewAction, note: string): boolean {
  return !action.requires_note || note.trim() !== "";
}

/** The note sent with a decision: trimmed, and left out when empty. */
export function decisionNote(note: string): string | null {
  const trimmed = note.trim();
  return trimmed === "" ? null : trimmed;
}

/** The action a decided review recorded. */
export function recordedAction(review: Pick<ReviewListItem, "status" | "presentation">): ReviewAction | null {
  const decision: ReviewDecision | null =
    review.status === "approved" ? "approve" : review.status === "rejected" ? "reject" : null;
  return review.presentation.actions.find((action) => action.decision === decision) ?? null;
}

export interface DecidedBanner {
  title: string;
  byline: string | null;
  note: string | null;
}

/** Who decided a review, when, and why, for the banner on a decided review. */
export function decidedBanner(
  review: Pick<ReviewListItem, "status" | "presentation" | "reviewer" | "resolved_at" | "review_note">,
  formatWhen: (iso: string) => string,
): DecidedBanner {
  const action = recordedAction(review);
  const who = review.reviewer ? ` by ${review.reviewer}` : "";
  const when = review.resolved_at ? ` on ${formatWhen(review.resolved_at)}` : "";
  return {
    title: action ? `${action.label} recorded` : "Decision recorded",
    byline: who || when ? `Decided${who}${when}.` : null,
    note: review.review_note?.trim() || null,
  };
}

export interface ExpiredExplanation {
  text: string;
  /** Whether a new proposal can be made from the current state. */
  canRecheck: boolean;
}

/** Why an expired proposal cannot be applied, and whether it can be rechecked. */
export function expiredExplanation(review: Pick<ReviewListItem, "review_origin" | "refreshable">): ExpiredExplanation {
  if (review.refreshable) {
    return {
      text: "The underlying memory changed before a decision was applied. This record remains in audit history.",
      canRecheck: true,
    };
  }
  if (review.review_origin === "lifecycle") {
    return {
      text: "The underlying memory changed before a decision was applied. The next revision of its source raises this conflict again, so there is nothing to recheck here.",
      canRecheck: false,
    };
  }
  return {
    text: "The memories in this proposal changed before a decision was applied, so it can no longer be applied. This record remains in audit history.",
    canRecheck: false,
  };
}

export interface DecisionFailure {
  message: string;
  /** What the user can do next, when the server's reason alone does not say. */
  hint: string | null;
  /** The review changed on the server, so reloading shows what can be decided now. */
  reloadable: boolean;
}

export function decisionFailure(error: unknown): DecisionFailure {
  const message = errorMessage(error);
  if (isApiErrorStatus(error, HTTP_STATUS.forbidden)) {
    return { message, hint: "You can read this review, but a source manager has to decide.", reloadable: false };
  }
  if (isApiErrorStatus(error, HTTP_STATUS.conflict)) {
    return { message, hint: "Reload the review to see its current state before deciding.", reloadable: true };
  }
  return { message, hint: null, reloadable: false };
}

/**
 * Stored statuses read as the Memories pages show them. `proposed` is not
 * stored: it marks a Source change that becomes a memory only if the review
 * uses it.
 */
const MEMORY_STATUS: Record<string, { label: string; tone: Tone }> = {
  active: { label: "Active", tone: "ok" },
  pending_review: { label: "Needs review", tone: "warn" },
  proposed: { label: "Proposed", tone: "live" },
  superseded: { label: "Superseded", tone: "live" },
  retired: { label: "Retired", tone: "idle" },
  decayed: { label: "Retired", tone: "idle" },
};

/** A memory's status as a review shows it. */
export function memoryStatus(status: string): { label: string; tone: Tone } {
  return MEMORY_STATUS[status] ?? { label: status.replaceAll("_", " "), tone: "idle" };
}

const DECISION_LABEL_TONE: Record<ReviewPresentation["decision_label"], Tone> = {
  Updated: "live",
  "Support removed": "warn",
  Conflict: "danger",
};

export function decisionLabelTone(label: ReviewPresentation["decision_label"]): Tone {
  return DECISION_LABEL_TONE[label];
}
