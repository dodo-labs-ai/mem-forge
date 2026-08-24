import { useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2, X } from "lucide-react";
import { useSearchParams } from "react-router-dom";
import { resourceClient } from "@/api/client";
import type {
  AgentEvaluationIssueGroup,
  WorkspaceAgentEvaluationResponse,
} from "@/api/types";
import { DataSurface } from "@/components/admin/DataSurface";
import { PageHeader } from "@/components/admin/PageHeader";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { OnlineEvaluationPresentation } from "./OnlineEvaluationPresentation";

const ALLOWED_WINDOWS = new Set([1, 7, 30]);

export function OnlineEvaluationPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedDays = Number(searchParams.get("days") || "1");
  const days = ALLOWED_WINDOWS.has(requestedDays) ? requestedDays : 1;
  const sourceId = searchParams.get("source_id") || null;
  const sourceType = searchParams.get("source_type") || null;
  const criterion = searchParams.get("criterion") || null;
  const label = searchParams.get("label") || null;
  const evaluationQuery = useQuery<WorkspaceAgentEvaluationResponse>({
    queryKey: ["workspace-agent-evaluation", days, sourceId, sourceType],
    queryFn: () =>
      resourceClient
        .get("/agent-evaluations/online-overview", {
          params: {
            days,
            source_id: sourceId ?? undefined,
            source_type: sourceType ?? undefined,
          },
        })
        .then((response) => response.data),
    placeholderData: (previousData) => previousData,
  });

  const data = evaluationQuery.data;
  const criteria = useMemo(
    () => Array.from(new Set([
      ...(data?.issue_groups.map((group) => group.criterion) ?? []),
      ...(data?.assessments.map((assessment) => assessment.criterion) ?? []),
    ])).sort(),
    [data],
  );
  const sourceTypes = useMemo(
    () => data?.available_source_types ?? [],
    [data],
  );
  const presentation = useMemo(() => {
    if (!data) return null;
    const issueGroups = data.issue_groups.filter((group) =>
      matchesIssueFilters(group, criterion, label)
    );
    const assessments = data.assessments.filter((assessment) =>
      (!criterion || assessment.criterion === criterion)
      && (!label || assessment.label === label)
    );
    const failOccurrences = issueGroups
      .filter((group) => group.label === "fail")
      .reduce((total, group) => total + group.occurrence_count, 0);
    const reviewOccurrences = issueGroups
      .filter((group) => group.label === "needs_review")
      .reduce((total, group) => total + group.occurrence_count, 0);
    return {
      ...data,
      summary: {
        ...data.summary,
        action_issue_group_count: issueGroups.filter((group) => group.label === "fail").length,
        review_issue_group_count: issueGroups.filter((group) => group.label === "needs_review").length,
        label_counts: {
          ...data.summary.label_counts,
          fail: failOccurrences,
          needs_review: reviewOccurrences,
        },
      },
      issue_groups: issueGroups,
      assessments,
    };
  }, [criterion, data, label]);
  const scopedSource = sourceId ? data?.sources.find((source) => source.source_id === sourceId) : null;

  function updateParams(
    updates: Record<string, string | number | null>,
    { clearAssessmentFilters = false }: { clearAssessmentFilters?: boolean } = {},
  ) {
    const next = new URLSearchParams(searchParams);
    for (const [key, value] of Object.entries(updates)) {
      if (value === null || value === "") next.delete(key);
      else next.set(key, String(value));
    }
    if (clearAssessmentFilters) {
      next.delete("criterion");
      next.delete("label");
    }
    setSearchParams(next, { replace: true });
  }

  return (
    <div className="space-y-4">
      <PageHeader
        title="Evaluation"
        description="Detect live-traffic problems across the active workspace, then investigate representative cases."
      />

      <DataSurface>
        <div className="flex flex-wrap items-center gap-2 border-b p-3">
          <Select
            value={sourceType ?? "all"}
            onValueChange={(value) => updateParams(
              {
                source_id: null,
                source_type: value === "all" ? null : value,
              },
              { clearAssessmentFilters: true },
            )}
          >
            <SelectTrigger aria-label="Filter by Source Type" className="h-7 w-48 text-[0.8rem]">
              <SelectValue placeholder="All Source Types" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Source Types</SelectItem>
              {sourceTypes.map((type) => (
                <SelectItem key={type} value={type}>{humanize(type)}</SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select
            value={criterion ?? "all"}
            onValueChange={(value) => updateParams({ criterion: value === "all" ? null : value })}
          >
            <SelectTrigger aria-label="Filter by criterion" className="h-7 w-52 text-[0.8rem]">
              <SelectValue placeholder="All criteria" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All criteria</SelectItem>
              {criteria.map((value) => (
                <SelectItem key={value} value={value}>{humanize(value)}</SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select
            value={label ?? "all"}
            onValueChange={(value) => updateParams({ label: value === "all" ? null : value })}
          >
            <SelectTrigger aria-label="Filter by assessment status" className="h-7 w-44 text-[0.8rem]">
              <SelectValue placeholder="All statuses" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All statuses</SelectItem>
              <SelectItem value="fail">Fail</SelectItem>
              <SelectItem value="needs_review">Needs review</SelectItem>
              <SelectItem value="pass">Pass</SelectItem>
            </SelectContent>
          </Select>

          {sourceId ? (
            <Button
              type="button"
              size="sm"
              variant="secondary"
              onClick={() => updateParams(
                { source_id: null, source_type: null },
                { clearAssessmentFilters: true },
              )}
            >
              {scopedSource?.name ?? "Filtered Source"}
              <X className="size-3.5" />
            </Button>
          ) : null}
        </div>

        {evaluationQuery.isPending ? (
          <div className="flex items-center gap-2 p-6 text-sm text-muted-foreground">
            <Loader2 className="size-4 animate-spin" /> Loading workspace evaluation...
          </div>
        ) : evaluationQuery.isError || !presentation ? (
          <div className="p-6 text-sm text-destructive">
            Failed to load workspace evaluation.
          </div>
        ) : (
          <div className="p-4">
            <OnlineEvaluationPresentation
              data={presentation}
              days={days}
              onDaysChange={(value) => updateParams({ days: value })}
              scopeName={scopedSource?.name ?? "Active workspace"}
              onSelectSource={(selectedSourceId) => updateParams(
                { source_id: selectedSourceId, source_type: null },
                { clearAssessmentFilters: true },
              )}
            />
          </div>
        )}
      </DataSurface>
    </div>
  );
}

function matchesIssueFilters(
  group: AgentEvaluationIssueGroup,
  criterion: string | null,
  label: string | null,
) {
  return (!criterion || group.criterion === criterion)
    && (!label || group.label === label);
}

function humanize(value: string) {
  return value
    .split("_")
    .map((part) => part ? `${part[0].toUpperCase()}${part.slice(1)}` : part)
    .join(" ");
}
