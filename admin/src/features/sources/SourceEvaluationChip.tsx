import { ShieldCheck } from "lucide-react";
import { Link } from "react-router-dom";
import { pluralize } from "@/lib/format";
import { sourceEvaluationPath } from "@/lib/paths";
import { Badge } from "@/ui/badge";
import { useSourceEvaluationHealth } from "./api";
import type { Source, SourceEvaluationHealth } from "./model/types";

/** What a source's online evaluation asks of the user, or null when it asks nothing. */
function evaluationChipLabel(health: SourceEvaluationHealth | undefined): string | null {
  switch (health?.evaluation_status) {
    case "attention":
      return pluralize(health.action_issue_group_count, "issue");
    case "coverage_gap":
      return pluralize(health.coverage.pending_occurrences, "pending evaluation");
    case "review":
      return pluralize(health.review_issue_group_count, "review pattern");
    default:
      return null;
  }
}

/**
 * Links a source whose pipeline checks found issues, left evaluations
 * pending or want a review to the Evaluation page scoped to it. A healthy
 * source, or one with no evaluations, shows nothing.
 */
export function SourceEvaluationChip({ source }: { source: Source }) {
  const label = evaluationChipLabel(useSourceEvaluationHealth().get(source.id));
  if (label === null) return null;
  return (
    <Badge
      variant="outline"
      render={<Link to={sourceEvaluationPath(source.id)} aria-label={`Open the pipeline checks of ${source.name}: ${label}`} />}
      onClick={(event) => event.stopPropagation()}
    >
      <ShieldCheck aria-hidden />
      {label}
    </Badge>
  );
}
