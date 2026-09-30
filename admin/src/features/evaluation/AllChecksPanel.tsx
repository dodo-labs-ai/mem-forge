import { ListChecks } from "lucide-react";
import { formatRelative, pluralize } from "@/lib/format";
import { EmptyState, StatusBadge } from "@/patterns";
import { assessmentResult, checkName, LISTED_CHECK_LIMIT } from "./model/presentation";
import type { Assessment } from "./model/types";

interface AllChecksPanelProps {
  assessments: Assessment[];
  /** True when the server listed as many checks as it returns, so older ones are left out. */
  limited: boolean;
  filtered: boolean;
}

/** The latest individual checks with their raw reason codes, for comparing one check against another. */
export function AllChecksPanel({ assessments, limited, filtered }: AllChecksPanelProps) {
  if (assessments.length === 0) {
    return filtered ? (
      <EmptyState icon={ListChecks} title="No checks match these filters" description="Choose another criterion or status." />
    ) : (
      <EmptyState icon={ListChecks} title="No checks in this window" description="Choose a longer window to see older checks." />
    );
  }
  return (
    <div>
      <ul className="divide-y">
        {assessments.map((assessment) => {
          const result = assessmentResult(assessment);
          return (
            <li key={assessment.assessment_id} className="flex items-center gap-3.5 px-4 py-2.5">
              <div className="min-w-0 flex-1 space-y-0.5">
                <p className="text-sm font-medium text-foreground">{checkName(assessment.criterion)}</p>
                <p className="font-mono text-xs text-muted-foreground">
                  {assessment.reason_code}
                  {assessment.occurrence_count > 1 ? (
                    <span className="font-sans">, {pluralize(assessment.occurrence_count, "occurrence")}</span>
                  ) : null}
                </p>
              </div>
              <StatusBadge tone={result.tone}>{result.text}</StatusBadge>
              <span className="w-20 shrink-0 text-right text-xs text-muted-foreground">{formatRelative(assessment.created_at)}</span>
            </li>
          );
        })}
      </ul>
      {limited ? (
        <p className="border-t px-4 py-2.5 text-xs text-muted-foreground">
          Only the latest {LISTED_CHECK_LIMIT} checks in this window are listed.
        </p>
      ) : null}
    </div>
  );
}
