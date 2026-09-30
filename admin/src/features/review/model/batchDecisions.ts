import type { Tone } from "@/patterns";
import { decisionNote } from "./reviewDecision";
import type { DecisionManifestItem, DecisionOutcome, DecisionResult, ReviewDecision, ReviewListItem } from "./types";

/** The most decisions the server validates or applies in one call. */
export const MAX_DECISIONS_PER_CALL = 50;

/**
 * The server's default risk. It records an agent's own assessment of a
 * decision; a person deciding in the admin UI does not give one.
 */
const PERSON_DECISION_RISK: DecisionManifestItem["risk"] = "medium";

/** Splits items into consecutive groups of at most `size`. */
export function inBatches<T>(items: readonly T[], size: number = MAX_DECISIONS_PER_CALL): T[][] {
  const batches: T[][] = [];
  for (let start = 0; start < items.length; start += size) batches.push(items.slice(start, start + size));
  return batches;
}

/** One manifest entry per review, all with the same decision and note. */
export function manifestFor(
  reviews: readonly Pick<ReviewListItem, "id" | "decision_fingerprint">[],
  decision: ReviewDecision,
  note: string,
): DecisionManifestItem[] {
  return reviews.map((review) => ({
    review_id: review.id,
    decision,
    expected_fingerprint: review.decision_fingerprint,
    note: decisionNote(note),
    risk: PERSON_DECISION_RISK,
  }));
}

interface OutcomePresentation {
  label: string;
  tone: Tone;
}

const OUTCOMES: Record<DecisionOutcome, OutcomePresentation> = {
  ready: { label: "Ready", tone: "ok" },
  applied: { label: "Applied", tone: "ok" },
  already_applied: { label: "Already decided", tone: "idle" },
  stale: { label: "Expired", tone: "warn" },
  forbidden: { label: "Not allowed", tone: "danger" },
  not_found: { label: "Not found", tone: "danger" },
  invalid: { label: "Not applied", tone: "danger" },
  failed: { label: "Not applied", tone: "danger" },
};

export function presentOutcome(outcome: DecisionOutcome): OutcomePresentation {
  return OUTCOMES[outcome];
}

/** Whether the server will apply (or has applied) this decision. */
function isGoing(outcome: DecisionOutcome): boolean {
  return outcome === "ready" || outcome === "applied";
}

/** One count per outcome label, in the order the outcomes are listed above. */
export function outcomeCounts(results: readonly DecisionResult[]): Array<OutcomePresentation & { count: number }> {
  const counts = new Map<string, OutcomePresentation & { count: number }>();
  for (const outcome of Object.keys(OUTCOMES) as DecisionOutcome[]) {
    const matching = results.filter((result) => result.outcome === outcome).length;
    if (matching === 0) continue;
    const presentation = OUTCOMES[outcome];
    const existing = counts.get(presentation.label);
    counts.set(presentation.label, { ...presentation, count: (existing?.count ?? 0) + matching });
  }
  return [...counts.values()];
}

function asSentence(text: string): string {
  return /[.!?]$/.test(text) ? text : `${text}.`;
}

/**
 * The line under one review in the batch dialog: the consequence when the
 * decision goes ahead, otherwise why it is skipped (`check`) or was skipped
 * (`done`).
 */
export function outcomeDetail(result: DecisionResult, consequence: string, phase: "check" | "done"): string {
  if (isGoing(result.outcome)) return result.consequence ?? consequence;
  const reason = asSentence(result.message ?? presentOutcome(result.outcome).label);
  if (result.outcome === "already_applied") return reason;
  return `${reason} ${phase === "check" ? "It will be skipped." : "It was skipped."}`;
}

/** The memory text that identifies a review in a batch: what the decision keeps. */
export function batchItemText(review: Pick<ReviewListItem, "incumbent" | "challenger" | "presentation">, decision: ReviewDecision) {
  const kept = decision === "approve" ? (review.challenger ?? review.incumbent) : review.incumbent;
  return kept?.content ?? review.presentation.summary;
}
