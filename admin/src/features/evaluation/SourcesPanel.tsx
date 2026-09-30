import { ChevronRight, Database } from "lucide-react";
import { SourceIcon } from "@/features/sources";
import { formatCount } from "@/lib/format";
import { EmptyState, StatusBadge } from "@/patterns";
import { formatCoverageRate, SOURCE_STATUS, sourceHealthSummary, sourceTypeLabel } from "./model/presentation";
import type { SourceHealth } from "./model/types";

interface SourcesPanelProps {
  sources: SourceHealth[];
  typeLabels: Readonly<Record<string, string>>;
  onSelectSource: (sourceId: string) => void;
}

/** Each source's evaluation, worst first; choosing one scopes the page to it. */
export function SourcesPanel({ sources, typeLabels, onSelectSource }: SourcesPanelProps) {
  if (sources.length === 0) {
    return <EmptyState icon={Database} title="No discoverable sources" description="There are no sources you can see in this scope." />;
  }
  return (
    <ul className="divide-y">
      {sources.map((source) => {
        const status = SOURCE_STATUS[source.evaluation_status];
        const { coverage } = source;
        return (
          <li key={source.source_id}>
            <button
              type="button"
              onClick={() => onSelectSource(source.source_id)}
              aria-label={`Open evaluation for ${source.name}`}
              className="flex w-full items-center gap-4 px-4 py-3.5 text-left hover:bg-surface-subtle"
            >
              <span className="flex size-8 shrink-0 items-center justify-center rounded-md border bg-surface-subtle">
                <SourceIcon type={source.type} className="size-4" />
              </span>
              <div className="min-w-0 flex-1 space-y-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-semibold text-foreground">{source.name}</span>
                  <span className="text-xs text-muted-foreground">{sourceTypeLabel(source.type, typeLabels)}</span>
                  <StatusBadge tone={status.tone}>{status.text}</StatusBadge>
                </div>
                <p className="text-xs text-muted-foreground">{sourceHealthSummary(source)}</p>
              </div>
              <div className="shrink-0 text-right">
                <p className="font-mono text-sm font-medium text-foreground">
                  {coverage.eligible_occurrences > 0 ? formatCoverageRate(coverage.coverage_rate) : "No data"}
                </p>
                <p className="text-xs text-muted-foreground">
                  {formatCount(coverage.assessed_occurrences)} / {formatCount(coverage.eligible_occurrences)} assessed
                </p>
              </div>
              <ChevronRight aria-hidden className="size-4 shrink-0 text-muted-foreground" />
            </button>
          </li>
        );
      })}
    </ul>
  );
}
