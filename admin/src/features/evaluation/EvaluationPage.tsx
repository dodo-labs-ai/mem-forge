import { CircleAlert, CircleCheck, LoaderCircle } from "lucide-react";
import { useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useSourceTypeLabels } from "@/features/sources";
import { errorMessage } from "@/lib/errors";
import { formatCount, pluralize } from "@/lib/format";
import { ErrorNotice, Notice, PageHeader, SegmentedControl, StatCard, type SegmentedOption } from "@/patterns";
import { Button } from "@/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/ui/tabs";
import { useEvaluationOverview } from "./api";
import { AllChecksPanel } from "./AllChecksPanel";
import { CoveragePanel } from "./CoveragePanel";
import { EvaluationFilters } from "./EvaluationFilters";
import { IssueGroupList } from "./IssueGroupList";
import {
  EVALUATION_WINDOWS,
  readEvaluationParams,
  writeEvaluationParams,
  type EvaluationParams,
  type EvaluationWindow,
} from "./model/evaluationParams";
import { buildEvaluationView, criteriaIn } from "./model/evaluationView";
import { evaluationMetrics, hasCheckFilter, LISTED_CHECK_LIMIT, windowLabel } from "./model/presentation";
import type { EvaluationOverview } from "./model/types";
import { SourcesPanel } from "./SourcesPanel";

type EvaluationTab = "attention" | "patterns" | "sources" | "coverage" | "all";

const WINDOW_LABELS: Record<EvaluationWindow, { short: string; full: string }> = {
  1: { short: "24h", full: "Last 24 hours" },
  7: { short: "7d", full: "Last 7 days" },
  30: { short: "30d", full: "Last 30 days" },
};

const WINDOW_OPTIONS: SegmentedOption<string>[] = EVALUATION_WINDOWS.map((days) => ({
  value: String(days),
  label: WINDOW_LABELS[days].short,
  accessibleLabel: WINDOW_LABELS[days].full,
}));

function TabLabel({ label, count }: { label: string; count?: number }) {
  return (
    <>
      {label}
      {count !== undefined ? (
        <>
          {" "}
          <span className="tabular-nums text-muted-foreground">{count}</span>
        </>
      ) : null}
    </>
  );
}

export function EvaluationPage() {
  const [search, setSearch] = useSearchParams();
  const params = readEvaluationParams(search);
  const overviewQuery = useEvaluationOverview(params);
  const typeLabels = useSourceTypeLabels();

  function update(change: Partial<EvaluationParams>) {
    setSearch(writeEvaluationParams(search, change), { replace: true });
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Evaluation"
        description="Automatic checks on how each source is processed: extraction, evidence, and lifecycle steps. Find pipeline problems and hand a case to a coding agent."
        actions={
          <div className="space-y-1.5">
            <p id="evaluation-window-label" className="text-xs font-medium text-muted-foreground">
              Window
            </p>
            <SegmentedControl
              aria-labelledby="evaluation-window-label"
              options={WINDOW_OPTIONS}
              value={String(params.days)}
              onValueChange={(value) => {
                const days = EVALUATION_WINDOWS.find((option) => String(option) === value);
                if (days) update({ days });
              }}
            />
          </div>
        }
      />

      {overviewQuery.isError ? (
        <div className="space-y-2">
          <ErrorNotice
            title="Could not load the evaluation"
            message={errorMessage(overviewQuery.error)}
            onRetry={() => void overviewQuery.refetch()}
          />
          {params.sourceId ? (
            <Button variant="outline" size="sm" onClick={() => update({ sourceId: null })}>
              Show all sources
            </Button>
          ) : null}
        </div>
      ) : overviewQuery.data ? (
        <EvaluationOverviewView overview={overviewQuery.data} params={params} typeLabels={typeLabels} onChange={update} />
      ) : (
        <div className="flex flex-col items-center gap-2.5 rounded-lg border bg-surface px-5 py-9 text-center">
          <LoaderCircle aria-hidden className="size-5 animate-spin text-muted-foreground" />
          <p className="font-medium text-foreground">Loading workspace evaluation…</p>
        </div>
      )}
    </div>
  );
}

interface EvaluationOverviewViewProps {
  overview: EvaluationOverview;
  params: EvaluationParams;
  typeLabels: Readonly<Record<string, string>>;
  onChange: (change: Partial<EvaluationParams>) => void;
}

function EvaluationOverviewView({ overview, params, typeLabels, onChange }: EvaluationOverviewViewProps) {
  const [tab, setTab] = useState<EvaluationTab>("attention");
  const filter = { criterion: params.criterion, label: params.label };
  const view = buildEvaluationView(overview, filter);
  const sourceNames = new Map(overview.sources.map((source) => [source.source_id, source.name]));
  const scopedSourceName = params.sourceId ? (sourceNames.get(params.sourceId) ?? null) : null;

  function selectSource(sourceId: string) {
    onChange({ sourceId, sourceType: null });
    setTab("attention");
  }

  return (
    <>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {evaluationMetrics(overview, view).map((metric) => (
          <StatCard key={metric.label} {...metric} />
        ))}
      </div>

      {overview.summary.truncated ? (
        <Notice tone="warn" icon={CircleAlert}>
          The counts cover only the latest {formatCount(overview.summary.row_limit)} events and checks in this window.
          Choose a shorter window before drawing conclusions.
        </Notice>
      ) : null}

      <section aria-label="Evaluation results" className="overflow-hidden rounded-lg border bg-surface">
        <Tabs value={tab} onValueChange={(value) => setTab(value as EvaluationTab)} className="gap-0">
          <div className="flex flex-wrap items-center gap-x-4 border-b px-4">
            <TabsList variant="line" aria-label="Evaluation views" className="h-11 gap-4">
              <TabsTrigger value="attention" className="flex-none px-0">
                <TabLabel label="Needs attention" count={view.failGroups.length} />
              </TabsTrigger>
              <TabsTrigger value="patterns" className="flex-none px-0">
                <TabLabel label="Patterns to check" count={view.reviewGroups.length} />
              </TabsTrigger>
              <TabsTrigger value="sources" className="flex-none px-0">
                <TabLabel label="Sources" count={overview.sources.length} />
              </TabsTrigger>
              <TabsTrigger value="coverage" className="flex-none px-0">
                <TabLabel label="Coverage" />
              </TabsTrigger>
              <TabsTrigger value="all" className="flex-none px-0">
                <TabLabel label="All checks" />
              </TabsTrigger>
            </TabsList>
            <p className="ml-auto py-2 text-xs text-muted-foreground">
              {pluralize(overview.summary.total_assessments, "assessment occurrence")} in {windowLabel(params.days)}
            </p>
          </div>

          <EvaluationFilters
            params={params}
            sourceTypes={overview.available_source_types}
            criteria={criteriaIn(overview)}
            typeLabels={typeLabels}
            scopedSourceName={scopedSourceName}
            onChange={onChange}
          />

          <TabsContent value="attention">
            <IssueGroupList
              groups={view.failGroups}
              sourceNames={sourceNames}
              typeLabels={typeLabels}
              empty={{
                icon: CircleCheck,
                title: "No live-traffic failures in this window",
                description: "Deterministic checks are not reporting an actionable problem.",
              }}
            />
          </TabsContent>
          <TabsContent value="patterns">
            <IssueGroupList
              groups={view.reviewGroups}
              sourceNames={sourceNames}
              typeLabels={typeLabels}
              empty={{
                icon: CircleCheck,
                title: "No representative cases need review",
                description: "No degraded fallback patterns were selected in this window.",
              }}
            />
          </TabsContent>
          <TabsContent value="sources">
            <SourcesPanel sources={overview.sources} typeLabels={typeLabels} onSelectSource={selectSource} />
          </TabsContent>
          <TabsContent value="coverage">
            <CoveragePanel coverage={overview.coverage} />
          </TabsContent>
          <TabsContent value="all">
            <AllChecksPanel
              assessments={view.assessments}
              limited={overview.assessments.length >= LISTED_CHECK_LIMIT}
              filtered={hasCheckFilter(filter)}
            />
          </TabsContent>
        </Tabs>
      </section>
    </>
  );
}
