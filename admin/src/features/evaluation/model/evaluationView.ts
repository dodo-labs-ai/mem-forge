import type { AssessmentLabel, Assessment, EvaluationOverview, IssueGroup } from "./types";

export interface CheckFilter {
  criterion: string | null;
  label: AssessmentLabel | null;
}

/** The loaded overview narrowed by the criterion and result filters. */
export interface EvaluationView {
  /** Groups of failed checks, the "Needs attention" tab. */
  failGroups: IssueGroup[];
  /** Groups of degraded checks that need a person to look, the "Patterns to check" tab. */
  reviewGroups: IssueGroup[];
  assessments: Assessment[];
  failOccurrences: number;
  reviewOccurrences: number;
}

function matches(item: { criterion: string; label: string | null }, filter: CheckFilter): boolean {
  return (!filter.criterion || item.criterion === filter.criterion) && (!filter.label || item.label === filter.label);
}

function occurrences(groups: IssueGroup[]): number {
  return groups.reduce((total, group) => total + group.occurrence_count, 0);
}

export function buildEvaluationView(overview: EvaluationOverview, filter: CheckFilter): EvaluationView {
  const groups = overview.issue_groups.filter((group) => matches(group, filter));
  const failGroups = groups.filter((group) => group.label === "fail");
  const reviewGroups = groups.filter((group) => group.label === "needs_review");
  return {
    failGroups,
    reviewGroups,
    assessments: overview.assessments.filter((assessment) => matches(assessment, filter)),
    failOccurrences: occurrences(failGroups),
    reviewOccurrences: occurrences(reviewGroups),
  };
}

/** Every criterion in the loaded scope, so the criterion filter offers only ones that can match. */
export function criteriaIn(overview: EvaluationOverview): string[] {
  const criteria = new Set([
    ...overview.issue_groups.map((group) => group.criterion),
    ...overview.assessments.map((assessment) => assessment.criterion),
  ]);
  return [...criteria].sort();
}
