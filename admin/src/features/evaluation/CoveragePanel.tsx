import { CircleCheck, Clock } from "lucide-react";
import { formatCount, formatRelative } from "@/lib/format";
import { Notice, StatCard } from "@/patterns";
import { formatCoverageRate } from "./model/presentation";
import type { EvaluationCoverage } from "./model/types";

/** Whether the evaluator keeps up with the pipeline. This is evaluator health, not source quality. */
export function CoveragePanel({ coverage }: { coverage: EvaluationCoverage }) {
  const keepingUp = coverage.pending_occurrences === 0 && coverage.evaluator_failure_occurrences === 0;
  const assessed = `${formatCount(coverage.assessed_occurrences)} of ${formatCount(coverage.eligible_occurrences)} eligible occurrences have a completed assessment from the current evaluator version.`;
  return (
    <div className="space-y-3.5 p-4">
      {keepingUp ? (
        <Notice tone="ok" icon={CircleCheck} title="Evaluation is keeping up">
          {assessed}
        </Notice>
      ) : (
        <Notice tone="warn" icon={Clock} title="Evaluation coverage needs attention">
          {assessed}
        </Notice>
      )}
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          label="Coverage"
          value={formatCoverageRate(coverage.coverage_rate)}
          detail={`${formatCount(coverage.assessed_occurrences)} / ${formatCount(coverage.eligible_occurrences)} assessed`}
        />
        <StatCard
          label="Pending"
          value={formatCount(coverage.pending_occurrences)}
          detail="Waiting for the evaluator"
          tone={coverage.pending_occurrences > 0 ? "warn" : undefined}
        />
        <StatCard
          label="Evaluator failures"
          value={formatCount(coverage.evaluator_failure_occurrences)}
          detail="Assessments that errored"
          tone={coverage.evaluator_failure_occurrences > 0 ? "danger" : undefined}
        />
        <StatCard
          label="Oldest pending"
          value={coverage.oldest_pending_at ? formatRelative(coverage.oldest_pending_at) : "None"}
          detail={coverage.oldest_pending_at ? "When its event occurred" : "Nothing is waiting"}
        />
      </div>
      <p className="text-xs text-muted-foreground">
        Coverage reports evaluator health, not source quality. A check counts as covered when an assessment exists for the
        same event, criterion, evaluator and evaluator version.
      </p>
    </div>
  );
}
