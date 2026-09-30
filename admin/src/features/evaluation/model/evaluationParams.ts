import { EVALUATION_SOURCE_PARAM } from "@/lib/paths";
import type { AssessmentLabel } from "./types";

/** The windows the page offers, in days. The server accepts 1 to 90. */
export const EVALUATION_WINDOWS = [1, 7, 30] as const;
export type EvaluationWindow = (typeof EVALUATION_WINDOWS)[number];
export const DEFAULT_WINDOW: EvaluationWindow = 1;

export const ASSESSMENT_LABELS: readonly AssessmentLabel[] = ["fail", "needs_review", "pass"];

const PARAM = {
  days: "days",
  sourceId: EVALUATION_SOURCE_PARAM,
  sourceType: "source_type",
  criterion: "criterion",
  label: "label",
} as const;

/** What the page shows, kept in the URL so a filtered view can be linked to. */
export interface EvaluationParams {
  days: EvaluationWindow;
  /** Narrows the server query to one source. */
  sourceId: string | null;
  /** Narrows the server query to one source type. */
  sourceType: string | null;
  /** Filters the loaded checks by criterion. */
  criterion: string | null;
  /** Filters the loaded checks by result. */
  label: AssessmentLabel | null;
}

function isWindow(value: number): value is EvaluationWindow {
  return (EVALUATION_WINDOWS as readonly number[]).includes(value);
}

function isLabel(value: string | null): value is AssessmentLabel {
  return value !== null && (ASSESSMENT_LABELS as readonly string[]).includes(value);
}

export function readEvaluationParams(search: URLSearchParams): EvaluationParams {
  const days = Number(search.get(PARAM.days));
  const label = search.get(PARAM.label);
  return {
    days: isWindow(days) ? days : DEFAULT_WINDOW,
    sourceId: search.get(PARAM.sourceId) || null,
    sourceType: search.get(PARAM.sourceType) || null,
    criterion: search.get(PARAM.criterion) || null,
    label: isLabel(label) ? label : null,
  };
}

/**
 * Applies a change to the current URL parameters and keeps parameters the page
 * does not own. Changing the source scope clears the check filters, because the
 * criteria on offer come from the loaded scope.
 */
export function writeEvaluationParams(search: URLSearchParams, change: Partial<EvaluationParams>): URLSearchParams {
  const next = new URLSearchParams(search);
  const scopeChanged = "sourceId" in change || "sourceType" in change;
  const merged: Partial<EvaluationParams> = scopeChanged ? { criterion: null, label: null, ...change } : change;
  for (const [key, value] of Object.entries(merged) as [keyof EvaluationParams, EvaluationParams[keyof EvaluationParams]][]) {
    const name = PARAM[key];
    if (value === null || (key === "days" && value === DEFAULT_WINDOW)) next.delete(name);
    else next.set(name, String(value));
  }
  return next;
}
