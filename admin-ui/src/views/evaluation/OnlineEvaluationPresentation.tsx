import { useMemo, useState } from "react";
import { Check, ChevronDown, Clipboard } from "lucide-react";
import type {
  AgentAssessmentView,
  AgentAssessmentSummary,
  AgentEvaluationCoverage,
  AgentEvaluationIssueGroup,
  AgentEvaluationRepresentativeCase,
  AgentEvaluationSourceHealth,
} from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { timeAgo } from "@/utils/date";
import { buildAgentInvestigationPrompt } from "../sources/onlineEvaluationInvestigation";

type EvaluationTab = "attention" | "review" | "sources" | "coverage" | "all";

export interface OnlineEvaluationPresentationData {
  summary: AgentAssessmentSummary;
  coverage: AgentEvaluationCoverage;
  issue_groups: AgentEvaluationIssueGroup[];
  assessments: AgentAssessmentView[];
  sources?: AgentEvaluationSourceHealth[];
}

const WINDOW_OPTIONS = [
  { days: 1, label: "24h" },
  { days: 7, label: "7d" },
  { days: 30, label: "30d" },
] as const;

export function OnlineEvaluationPresentation({
  data,
  days,
  onDaysChange,
  scopeName,
  onSelectSource,
}: {
  data: OnlineEvaluationPresentationData;
  days: number;
  onDaysChange: (days: number) => void;
  scopeName?: string;
  onSelectSource?: (sourceId: string) => void;
}) {
  const [tab, setTab] = useState<EvaluationTab>("attention");
  const [expandedGroupId, setExpandedGroupId] = useState<string | null>(null);
  const [copiedEventId, setCopiedEventId] = useState<string | null>(null);
  const [copyFailedEventId, setCopyFailedEventId] = useState<string | null>(null);
  const groups = data.issue_groups;
  const attentionGroups = useMemo(
    () => groups.filter((group) => group.label === "fail"),
    [groups],
  );
  const reviewGroups = useMemo(
    () => groups.filter((group) => group.label === "needs_review"),
    [groups],
  );
  const sourceNameById = useMemo(
    () => new Map((data.sources ?? []).map((source) => [source.source_id, source.name])),
    [data.sources],
  );
  const summary = data.summary;
  const coverage = data.coverage;
  const labels = summary?.label_counts ?? {};
  const reviewOccurrences = reviewGroups.reduce(
    (total, group) => total + group.occurrence_count,
    0,
  );

  async function investigate(
    group: AgentEvaluationIssueGroup,
    evaluationCase: AgentEvaluationRepresentativeCase,
  ) {
    try {
      await navigator.clipboard.writeText(
        buildAgentInvestigationPrompt(
          sourceNameById.get(evaluationCase.source_id)
            ?? scopeName
            ?? evaluationCase.source_id,
          group,
          evaluationCase,
        ),
      );
      setCopiedEventId(evaluationCase.event_id);
      setCopyFailedEventId(null);
    } catch {
      setCopiedEventId(null);
      setCopyFailedEventId(evaluationCase.event_id);
    }
  }

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4">
      <div className="flex justify-end">
        <div className="flex rounded-lg border bg-muted/40 p-0.5" aria-label="Evaluation window">
          {WINDOW_OPTIONS.map((option) => (
            <Button
              key={option.days}
              type="button"
              size="xs"
              variant={days === option.days ? "secondary" : "ghost"}
              aria-pressed={days === option.days}
              onClick={() => {
                onDaysChange(option.days);
                setExpandedGroupId(null);
              }}
            >
              {option.label}
            </Button>
          ))}
        </div>
      </div>
      <div className={`grid gap-3 ${data.sources ? "sm:grid-cols-4" : "sm:grid-cols-3"}`}>
        {data.sources ? (
          <EvaluationMetric
            label="Affected sources"
            value={`${summary.affected_source_count ?? 0} / ${summary.source_count ?? data.sources.length}`}
            detail="Sources with failures, review cases, or coverage gaps"
            tone={(summary.affected_source_count ?? 0) ? "danger" : "primary"}
          />
        ) : null}
              <EvaluationMetric
                label="Live quality"
                value={attentionGroups.length ? `${attentionGroups.length} issue groups` : "Healthy"}
                detail={attentionGroups.length ? `${labels.fail ?? 0} failed occurrences` : "No deterministic failures"}
                tone={attentionGroups.length ? "danger" : "primary"}
              />
              <EvaluationMetric
                label="Review queue"
                value={reviewOccurrences ? `${reviewOccurrences} occurrences` : "Clear"}
                detail={reviewGroups.length ? `${reviewGroups.length} degraded patterns` : "No cases need review"}
              />
              <EvaluationMetric
                label="Evaluation coverage"
                value={formatCoverage(coverage.coverage_rate)}
                detail={`${formatCount(coverage.assessed_occurrences)} / ${formatCount(coverage.eligible_occurrences)} eligible occurrences`}
                tone={coverage.pending_occurrences ? "warning" : "primary"}
              />
      </div>

      <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap gap-1" role="tablist" aria-label="Online evaluation views">
                <EvaluationTabButton
                  active={tab === "attention"}
                  label={`Needs attention (${attentionGroups.length})`}
                  onClick={() => setTab("attention")}
                />
                <EvaluationTabButton
                  active={tab === "review"}
                  label={`Review queue (${reviewGroups.length})`}
                  onClick={() => setTab("review")}
                />
                {data.sources ? (
                  <EvaluationTabButton
                    active={tab === "sources"}
                    label={`Sources (${data.sources.length})`}
                    onClick={() => setTab("sources")}
                  />
                ) : null}
                <EvaluationTabButton
                  active={tab === "coverage"}
                  label="Coverage"
                  onClick={() => setTab("coverage")}
                />
                <EvaluationTabButton
                  active={tab === "all"}
                  label="All checks"
                  onClick={() => setTab("all")}
                />
              </div>
              <span className="text-xs text-muted-foreground">
                {formatCount(summary?.total_assessments)} assessment occurrences in {windowLabel(days)}
              </span>
      </div>

      {summary.truncated ? (
              <p className="rounded-md bg-amber-500/10 px-3 py-2 text-xs text-amber-700 dark:text-amber-300">
                This window reached the 1,000-row safety bound. Narrow the time window before drawing conclusions.
              </p>
      ) : null}

      <div className="min-h-0 flex-1 overflow-y-auto rounded-lg border">
              {tab === "attention" ? (
                <IssueGroupList
                  groups={attentionGroups}
                  emptyTitle="No live-traffic failures in this window"
                  emptyDescription="Deterministic checks are not reporting an actionable problem."
                  expandedGroupId={expandedGroupId}
                  copiedEventId={copiedEventId}
                  copyFailedEventId={copyFailedEventId}
                  onToggle={(groupId) => setExpandedGroupId((current) => current === groupId ? null : groupId)}
                  onInvestigate={investigate}
                />
              ) : null}
              {tab === "review" ? (
                <IssueGroupList
                  groups={reviewGroups}
                  emptyTitle="No representative cases need review"
                  emptyDescription="No degraded fallback patterns were selected in this window."
                  expandedGroupId={expandedGroupId}
                  copiedEventId={copiedEventId}
                  copyFailedEventId={copyFailedEventId}
                  onToggle={(groupId) => setExpandedGroupId((current) => current === groupId ? null : groupId)}
                  onInvestigate={investigate}
                />
              ) : null}
              {tab === "sources" && data.sources ? (
                <SourcesPanel sources={data.sources} onSelectSource={onSelectSource} />
              ) : null}
              {tab === "coverage" ? (
                <CoveragePanel coverage={coverage} />
              ) : null}
              {tab === "all" ? (
                <AllChecks assessments={data.assessments} />
              ) : null}
      </div>
    </div>
  );
}

function EvaluationMetric({
  label,
  value,
  detail,
  tone = "default",
}: {
  label: string;
  value: string;
  detail: string;
  tone?: "default" | "primary" | "warning" | "danger";
}) {
  const toneClass = {
    default: "",
    primary: "text-primary",
    warning: "text-amber-700 dark:text-amber-300",
    danger: "text-destructive",
  }[tone];
  return (
    <div className="rounded-lg border p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div className={`mt-1 text-xl font-semibold ${toneClass}`}>{value}</div>
      <div className="mt-1 text-xs text-muted-foreground">{detail}</div>
    </div>
  );
}

function EvaluationTabButton({
  active,
  label,
  onClick,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <Button
      type="button"
      size="sm"
      variant={active ? "secondary" : "ghost"}
      role="tab"
      aria-selected={active}
      onClick={onClick}
    >
      {label}
    </Button>
  );
}

function IssueGroupList({
  groups,
  emptyTitle,
  emptyDescription,
  expandedGroupId,
  copiedEventId,
  copyFailedEventId,
  onToggle,
  onInvestigate,
}: {
  groups: AgentEvaluationIssueGroup[];
  emptyTitle: string;
  emptyDescription: string;
  expandedGroupId: string | null;
  copiedEventId: string | null;
  copyFailedEventId: string | null;
  onToggle: (groupId: string) => void;
  onInvestigate: (
    group: AgentEvaluationIssueGroup,
    evaluationCase: AgentEvaluationRepresentativeCase,
  ) => void;
}) {
  if (!groups.length) {
    return (
      <div className="p-6 text-center">
        <div className="font-medium">{emptyTitle}</div>
        <div className="mt-1 text-sm text-muted-foreground">{emptyDescription}</div>
      </div>
    );
  }
  return (
    <div className="divide-y">
      {groups.map((group) => {
        const expanded = expandedGroupId === group.group_id;
        return (
          <div key={group.group_id}>
            <button
              type="button"
              className="grid w-full cursor-pointer gap-2 p-4 text-left hover:bg-muted/50 sm:grid-cols-[1fr_auto]"
              aria-expanded={expanded}
              onClick={() => onToggle(group.group_id)}
            >
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium">{humanize(group.criterion)}</span>
                  <Badge variant={group.label === "fail" ? "destructive" : "outline"}>
                    {group.label.replace("_", " ")}
                  </Badge>
                </div>
                <div className="mt-1 text-xs text-muted-foreground">
                  {reasonDescription(group.reason_code)}
                </div>
                <div className="mt-2 text-xs text-muted-foreground">
                  {group.occurrence_count} of {group.criterion_occurrence_count} {humanize(group.criterion).toLowerCase()} checks ({formatCoverage(group.criterion_rate)}) · {group.affected_source_count} sources · {group.source_types.map(humanize).join(", ")} · last seen {timeAgo(group.last_seen_at)}
                </div>
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                {group.representative_cases.length} examples
                <ChevronDown className={`size-4 transition-transform ${expanded ? "rotate-180" : ""}`} />
              </div>
            </button>
            {expanded ? (
              <div className="border-t bg-muted/20 p-3">
                <div className="mb-2 text-xs font-medium uppercase tracking-wide text-muted-foreground">
                  Representative cases
                </div>
                <div className="grid gap-2">
                  {group.representative_cases.map((evaluationCase) => (
                    <div key={evaluationCase.event_id} className="rounded-lg border bg-background p-3">
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div className="min-w-0">
                          <div className="text-sm font-medium">
                            {humanize(evaluationCase.criterion)} · {timeAgo(evaluationCase.occurred_at)}
                          </div>
                          <div className="mt-1 break-all text-xs text-muted-foreground">
                            Source unit {shortId(evaluationCase.source_unit_id)} · event {shortId(evaluationCase.event_id)}
                            {evaluationCase.trace_id ? ` · trace ${shortId(evaluationCase.trace_id)}` : ""}
                          </div>
                        </div>
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => onInvestigate(group, evaluationCase)}
                        >
                          {copiedEventId === evaluationCase.event_id ? <Check /> : <Clipboard />}
                          {copiedEventId === evaluationCase.event_id ? "Prompt copied" : "Investigate with agent"}
                        </Button>
                      </div>
                      <div className="mt-2 text-xs text-muted-foreground">
                        Copies a bounded, read-only diagnosis prompt with lineage and evaluator versions.
                      </div>
                      {copyFailedEventId === evaluationCase.event_id ? (
                        <div className="mt-2 text-xs text-destructive">
                          Clipboard access failed. Try again from a secure browser window.
                        </div>
                      ) : null}
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        );
      })}
    </div>
  );
}

function SourcesPanel({
  sources,
  onSelectSource,
}: {
  sources: AgentEvaluationSourceHealth[];
  onSelectSource?: (sourceId: string) => void;
}) {
  if (!sources.length) {
    return (
      <div className="p-6 text-center text-sm text-muted-foreground">
        No discoverable Sources in this scope.
      </div>
    );
  }
  return (
    <div className="divide-y">
      {sources.map((source) => {
        const body = (
          <>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-medium">{source.name}</span>
                <span className="text-xs text-muted-foreground">{humanize(source.type)}</span>
                <SourceEvaluationStatusBadge status={source.evaluation_status} />
              </div>
              <div className="mt-1 text-xs text-muted-foreground">
                {source.action_issue_group_count} attention groups · {source.review_issue_group_count} review groups
                {source.last_event_at ? ` · last event ${timeAgo(source.last_event_at)}` : " · no events in this window"}
              </div>
            </div>
            <div className="text-right text-xs text-muted-foreground">
              <div className="font-medium text-foreground">
                {source.coverage.eligible_occurrences
                  ? formatCoverage(source.coverage.coverage_rate)
                  : "No data"}
              </div>
              <div>
                {source.coverage.assessed_occurrences} / {source.coverage.eligible_occurrences} assessed
              </div>
            </div>
          </>
        );
        return onSelectSource ? (
          <button
            key={source.source_id}
            type="button"
            className="grid w-full cursor-pointer gap-3 p-4 text-left hover:bg-muted/50 sm:grid-cols-[1fr_auto] sm:items-center"
            onClick={() => onSelectSource(source.source_id)}
            aria-label={`Open evaluation for ${source.name}`}
          >
            {body}
          </button>
        ) : (
          <div
            key={source.source_id}
            className="grid gap-3 p-4 sm:grid-cols-[1fr_auto] sm:items-center"
          >
            {body}
          </div>
        );
      })}
    </div>
  );
}

function SourceEvaluationStatusBadge({
  status,
}: {
  status: AgentEvaluationSourceHealth["evaluation_status"];
}) {
  const labels = {
    attention: "Needs attention",
    coverage_gap: "Coverage gap",
    review: "Needs review",
    healthy: "Healthy",
    no_data: "No recent checks",
  } as const;
  return (
    <Badge variant={status === "attention" ? "destructive" : "outline"}>
      {labels[status]}
    </Badge>
  );
}

function CoveragePanel({
  coverage,
}: {
  coverage: AgentEvaluationCoverage;
}) {
  const healthy = coverage.pending_occurrences === 0 && coverage.evaluator_failure_occurrences === 0;
  return (
    <div className="space-y-4 p-4">
      <div className={`rounded-lg border p-4 ${healthy ? "bg-primary/5" : "bg-amber-500/10"}`}>
        <div className="font-medium">
          {healthy ? "Evaluation is keeping up" : "Evaluation coverage needs attention"}
        </div>
        <div className="mt-1 text-sm text-muted-foreground">
          {formatCount(coverage.assessed_occurrences)} of {formatCount(coverage.eligible_occurrences)} eligible occurrences have a semantically matching durable assessment.
        </div>
      </div>
      <dl className="grid gap-3 sm:grid-cols-2">
        <CoverageValue label="Coverage" value={formatCoverage(coverage.coverage_rate)} />
        <CoverageValue label="Pending" value={formatCount(coverage.pending_occurrences)} />
        <CoverageValue label="Evaluator failures" value={formatCount(coverage.evaluator_failure_occurrences)} />
        <CoverageValue
          label="Oldest pending"
          value={coverage.oldest_pending_at ? timeAgo(coverage.oldest_pending_at) : "None"}
        />
      </dl>
      <p className="text-xs text-muted-foreground">
        Coverage reports evaluator health, not source quality. Assessment schema IDs do not create user-facing gaps when the event, criterion, evaluator, and evaluator version still match.
      </p>
    </div>
  );
}

function CoverageValue({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border p-3">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="mt-1 text-lg font-medium">{value}</dd>
    </div>
  );
}

function AllChecks({ assessments }: { assessments: AgentAssessmentView[] }) {
  if (!assessments.length) {
    return <div className="p-6 text-center text-sm text-muted-foreground">No checks in this window.</div>;
  }
  return (
    <div className="divide-y">
      {assessments.map((assessment) => (
        <div
          key={assessment.assessment_id}
          className="grid gap-1 p-3 text-sm sm:grid-cols-[1fr_auto_auto] sm:items-center sm:gap-4"
        >
          <div>
            <div className="font-medium">{humanize(assessment.criterion)}</div>
            <div className="text-xs text-muted-foreground">
              {assessment.reason_code}
              {assessment.occurrence_count > 1 ? ` · ${assessment.occurrence_count} occurrences` : ""}
            </div>
          </div>
          <Badge variant={assessment.label === "fail" ? "destructive" : "outline"}>
            {assessment.label?.replace("_", " ") ?? assessment.status}
          </Badge>
          <span className="text-xs text-muted-foreground">{timeAgo(assessment.created_at)}</span>
        </div>
      ))}
    </div>
  );
}

function reasonDescription(reasonCode: string): string {
  const descriptions: Record<string, string> = {
    legacy_quote_unresolved:
      "A legacy evidence quote could not be resolved, so the candidate was rejected before it could become an unsupported Memory.",
    whole_block_fallback:
      "Exact evidence localization was not proven; the whole source block was retained as a conservative fallback.",
    missing_evidence_reference:
      "The candidate did not provide an evidence reference and was rejected.",
    unknown_evidence_block_id:
      "The candidate referenced a source block that was not present in the authorized projection.",
    schema_validation_failed:
      "The model output did not conform to the structured contract.",
  };
  return descriptions[reasonCode] ?? humanize(reasonCode);
}

function humanize(value: string): string {
  return value
    .split("_")
    .map((part) => part ? `${part[0].toUpperCase()}${part.slice(1)}` : part)
    .join(" ");
}

function shortId(value: string): string {
  return value.length > 18 ? `${value.slice(0, 8)}…${value.slice(-6)}` : value;
}

function formatCoverage(value: number | undefined): string {
  if (value === undefined) return "—";
  return `${Math.round(value * 100)}%`;
}

function formatCount(value: number | undefined): string {
  return typeof value === "number" ? value.toLocaleString() : "—";
}

function windowLabel(days: number): string {
  return days === 1 ? "the last 24 hours" : `the last ${days} days`;
}
