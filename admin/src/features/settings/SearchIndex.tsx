import { AlertCircle, Loader2, RefreshCw } from "lucide-react";
import { useId } from "react";
import { isReindexInProgress } from "@/api";
import { cn } from "@/lib/cn";
import { formatCount } from "@/lib/format";
import { ErrorNotice, Notice, PropertyList, StatusBadge, type Property } from "@/patterns";
import { Button } from "@/ui/button";
import { useStartReindex } from "./api";
import {
  builtWith,
  rebuildFailure,
  rebuildLabel,
  rebuildProgress,
  rebuildTarget,
  searchIndexMarks,
  searchIndexSummary,
  startRefusalReason,
  type RebuildProgress,
} from "./model/searchIndex";
import type { ModelKindSettings, ModelProvider, VectorIndex } from "./model/types";
import { StatusLine } from "./StatusLine";

function indexProperties(vectorIndex: VectorIndex, providers: readonly ModelProvider[]): Property[] {
  const index = vectorIndex.index;
  if (index === null) return [];
  return [
    { label: "Built with", value: builtWith(vectorIndex, providers) ?? "A model that was not recorded" },
    { label: "Memories", value: formatCount(index.vector_count) },
    { label: "Vector size", value: index.dimension },
  ];
}

function ProgressBar({ progress }: { progress: RebuildProgress }) {
  const { percent, text, meanwhile } = progress;
  return (
    <div className="space-y-1.5">
      <div
        role="progressbar"
        aria-label="Rebuild of the search index"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={percent ?? undefined}
        aria-valuetext={text}
        className="h-1.5 overflow-hidden rounded-full bg-muted"
      >
        <div
          className={cn("h-full rounded-full bg-tone-live transition-[width]", percent === null && "w-1/4 animate-pulse")}
          style={percent === null ? undefined : { width: `${percent}%` }}
        />
      </div>
      <p className="text-sm text-foreground">{text}</p>
      <p className="text-xs text-muted-foreground">{meanwhile}</p>
    </div>
  );
}

interface SearchIndexProps {
  vectorIndex: VectorIndex;
  /** The embedding model in use and the saved edit, which the index is judged against. */
  embedding: ModelKindSettings;
  providers: readonly ModelProvider[];
}

/**
 * The search index of the embedding model: which model built it, how many
 * memories it holds, and whether vector search can use it. A rebuild shows
 * its progress here, and one that failed its reason. When the index does not
 * fit the model in use, or its origin is assumed, the rebuild for that model
 * starts here; the rebuild for a saved edit is the action of the form.
 */
export function SearchIndex({ vectorIndex, embedding, providers }: SearchIndexProps) {
  const rebuild = useStartReindex();
  const headingId = useId();
  const summary = searchIndexSummary(vectorIndex, embedding, providers);
  const progress = rebuildProgress(vectorIndex);
  const failure = rebuildFailure(vectorIndex);
  const properties = indexProperties(vectorIndex, providers);
  const offered = rebuildTarget(vectorIndex) === "active";
  const required = vectorIndex.status === "reindex_required";

  if (vectorIndex.status === "not_managed") return null;
  if (summary === null && properties.length === 0 && failure === null) return null;

  return (
    <div role="group" aria-labelledby={headingId} className="space-y-3 rounded-lg bg-surface-subtle p-3.5">
      <h3 id={headingId} className="text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">
        Search index
      </h3>
      {properties.length > 0 ? <PropertyList layout="row" items={properties} /> : null}
      {summary !== null || progress !== null ? (
        <div role="status" className="space-y-2.5">
          {summary !== null ? <StatusLine summary={summary} /> : null}
          {progress !== null ? <ProgressBar progress={progress} /> : null}
        </div>
      ) : null}
      {searchIndexMarks(vectorIndex).map((mark) => (
        <p key={mark.label} className="text-xs text-subtle-foreground">
          <StatusBadge tone="warn" className="mr-1.5 align-middle">
            {mark.label}
          </StatusBadge>
          {mark.text}
        </p>
      ))}
      {failure !== null ? (
        <Notice tone="danger" icon={AlertCircle} title={failure.title}>
          {failure.body}
        </Notice>
      ) : null}
      {offered ? (
        <div className="flex flex-wrap items-center gap-3">
          <Button
            type="button"
            variant={required ? "default" : "outline"}
            size="sm"
            onClick={() => rebuild.mutate()}
            disabled={rebuild.isPending}
          >
            {rebuild.isPending ? <Loader2 className="animate-spin" /> : <RefreshCw />}
            {rebuild.isPending ? "Starting the rebuild…" : rebuildLabel(vectorIndex)}
          </Button>
          <p className="min-w-0 flex-1 text-xs text-muted-foreground">
            A rebuild embeds every memory again with the model in use. While it runs, source syncs and changes to memories
            wait.
          </p>
        </div>
      ) : null}
      {rebuild.isError && !isReindexInProgress(rebuild.error) ? (
        <ErrorNotice title="Couldn't start the rebuild" message={startRefusalReason(rebuild.error)} />
      ) : null}
    </div>
  );
}
