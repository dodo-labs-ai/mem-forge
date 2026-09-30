import { formatCount, formatRelative, pluralize } from "@/lib/format";
import type { Tone } from "@/patterns";
import type { CheckFilter, EvaluationView } from "./evaluationView";
import type { Assessment, AssessmentLabel, EvaluationOverview, IssueGroup, SourceEvaluationStatus, SourceHealth } from "./types";

const HOURS_PER_DAY = 24;
/** The server lists at most this many of the latest checks. */
export const LISTED_CHECK_LIMIT = 50;
const ID_HEAD_LENGTH = 8;
const ID_TAIL_LENGTH = 6;
/** IDs longer than this are shortened to their head and tail. */
const ID_SHORTEN_ABOVE = ID_HEAD_LENGTH + ID_TAIL_LENGTH + 4;

const listFormat = new Intl.ListFormat("en-US", { style: "long", type: "conjunction" });
const wholePercent = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 0 });
const precisePercent = new Intl.NumberFormat("en-US", { style: "percent", maximumFractionDigits: 1 });

/** "evidence_reference_validity" becomes "Evidence reference validity". */
export function sentenceCase(code: string): string {
  const words = code.split("_").filter(Boolean).join(" ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** The name of a check, from its criterion code. */
export const checkName = sentenceCase;

const REASON_DESCRIPTIONS: Readonly<Record<string, string>> = {
  legacy_quote_unresolved:
    "A legacy evidence quote could not be resolved, so the candidate was rejected before it could become an unsupported memory.",
  whole_block_fallback:
    "Exact evidence localization was not proven; the whole source block was retained as a conservative fallback.",
  missing_evidence_reference: "The candidate did not provide an evidence reference and was rejected.",
  unknown_evidence_block_id: "The candidate referenced a source block that was not present in the authorized projection.",
  schema_validation_failed: "The model output did not conform to the structured contract.",
};

export function reasonDescription(reasonCode: string): string {
  return REASON_DESCRIPTIONS[reasonCode] ?? sentenceCase(reasonCode);
}

export function formatCoverageRate(rate: number): string {
  return wholePercent.format(rate);
}

/** Keeps the start and end of a long ID, which is enough to tell cases apart. */
export function shortId(id: string): string {
  if (id.length <= ID_SHORTEN_ABOVE) return id;
  return `${id.slice(0, ID_HEAD_LENGTH)}…${id.slice(-ID_TAIL_LENGTH)}`;
}

/** "the last 24 hours", "the last 7 days". */
export function windowLabel(days: number): string {
  return days === 1 ? `the last ${HOURS_PER_DAY} hours` : `the last ${days} days`;
}

export function sourceTypeLabel(type: string, typeLabels: Readonly<Record<string, string>>): string {
  return typeLabels[type] ?? sentenceCase(type);
}

export interface Metric {
  label: string;
  value: string;
  detail: string;
  /** Colors the value when it needs attention. */
  tone?: Extract<Tone, "danger" | "warn">;
}

export function evaluationMetrics(overview: EvaluationOverview, view: EvaluationView): Metric[] {
  const { summary, coverage } = overview;
  return [
    {
      label: "Affected sources",
      value: `${formatCount(summary.affected_source_count)} / ${formatCount(summary.source_count)}`,
      detail: "Sources with failures, patterns to check, or coverage gaps",
      tone: summary.affected_source_count > 0 ? "danger" : undefined,
    },
    {
      label: "Live quality",
      value: view.failGroups.length > 0 ? pluralize(view.failGroups.length, "issue group") : "Healthy",
      detail:
        view.failGroups.length > 0 ? pluralize(view.failOccurrences, "failed occurrence") : "No deterministic failures",
      tone: view.failGroups.length > 0 ? "danger" : undefined,
    },
    {
      label: "Patterns to check",
      value: view.reviewOccurrences > 0 ? pluralize(view.reviewOccurrences, "occurrence") : "Clear",
      detail: view.reviewGroups.length > 0 ? pluralize(view.reviewGroups.length, "degraded pattern") : "No cases need review",
    },
    {
      label: "Evaluation coverage",
      value: formatCoverageRate(coverage.coverage_rate),
      detail: `${formatCount(coverage.assessed_occurrences)} / ${formatCount(coverage.eligible_occurrences)} eligible occurrences`,
      tone: coverage.pending_occurrences > 0 ? "warn" : undefined,
    },
  ];
}

/** One sentence per fact: how often the check failed, where, and when it last happened. */
export function issueGroupSummary(
  group: IssueGroup,
  typeLabels: Readonly<Record<string, string>>,
  now: Date = new Date(),
): string {
  const rate = precisePercent.format(group.criterion_rate);
  const checks = `${formatCount(group.occurrence_count)} of ${formatCount(group.criterion_occurrence_count)} ${checkName(group.criterion).toLowerCase()} checks (${rate}).`;
  const types = listFormat.format(group.source_types.map((type) => sourceTypeLabel(type, typeLabels)));
  const sources = `Affects ${pluralize(group.affected_source_count, "source")} (${types}).`;
  return `${checks} ${sources} Last seen ${formatRelative(group.last_seen_at, now)}.`;
}

export const RESULT_LABELS: Record<AssessmentLabel, { text: string; tone: Tone }> = {
  fail: { text: "Fail", tone: "danger" },
  needs_review: { text: "Needs review", tone: "warn" },
  pass: { text: "Pass", tone: "ok" },
};

/** The result badge of one check; a check whose evaluator errored has no label. */
export function assessmentResult(assessment: Assessment): { text: string; tone: Tone } {
  return assessment.label ? RESULT_LABELS[assessment.label] : { text: "Evaluator failed", tone: "idle" };
}

export const SOURCE_STATUS: Record<SourceEvaluationStatus, { text: string; tone: Tone }> = {
  attention: { text: "Needs attention", tone: "danger" },
  coverage_gap: { text: "Coverage gap", tone: "warn" },
  review: { text: "Pattern to check", tone: "warn" },
  healthy: { text: "Healthy", tone: "ok" },
  no_data: { text: "No recent checks", tone: "idle" },
};

/** What is wrong with one source, in full sentences. */
export function sourceHealthSummary(source: SourceHealth, now: Date = new Date()): string {
  if (source.evaluation_status === "no_data" || source.last_event_at === null) return "No events in this window.";
  const lastEvent = `Last event ${formatRelative(source.last_event_at, now)}.`;
  if (source.evaluation_status === "healthy") return `No issues. ${lastEvent}`;
  const sentences = [
    `${pluralize(source.action_issue_group_count, "issue group")}, ${pluralize(source.review_issue_group_count, "pattern")} to check.`,
  ];
  if (source.coverage.pending_occurrences > 0) {
    sentences.push(`${pluralize(source.coverage.pending_occurrences, "occurrence")} waiting for evaluation.`);
  }
  if (source.coverage.evaluator_failure_occurrences > 0) {
    sentences.push(`${pluralize(source.coverage.evaluator_failure_occurrences, "evaluator failure")}.`);
  }
  sentences.push(lastEvent);
  return sentences.join(" ");
}

export function hasCheckFilter(filter: CheckFilter): boolean {
  return filter.criterion !== null || filter.label !== null;
}
