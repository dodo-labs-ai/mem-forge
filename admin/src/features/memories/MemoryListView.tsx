import { Info } from "lucide-react";
import type { ReactNode } from "react";
import { useNavigate } from "react-router-dom";
import { pluralize } from "@/lib/format";
import { errorMessage } from "@/lib/errors";
import { DataTable, EmptyState, ErrorNotice, Pagination, SearchBox } from "@/patterns";
import { Button } from "@/ui/button";
import { useMemoryList } from "./api";
import { LIST_PAGE_SIZE, SEARCH_RESULT_LIMIT } from "./constants";
import { MEMORY_COLUMNS } from "./memoryColumns";
import { ActiveFilterChips, MemoryFiltersPopover } from "./MemoryFilters";
import {
  activeFilterKeys,
  changeFilters,
  changePageState,
  clearFilters,
  memoryListRequest,
  type FilterKey,
  type MemoriesPageState,
} from "./model/listState";
import { isHistorical, memoryStatusLabel, memoryTypeLabel } from "./model/memoryPresentation";
import type { MemoryRow } from "./model/memoryRows";
import { projectFilterOptions, projectName } from "./model/projects";
import type { Project, SourceOption } from "./model/types";

function rankedSearchNotice(state: MemoriesPageState, project: string): string {
  const scope =
    state.filters.projectScope === "project"
      ? `Search shows memories from ${project} and team-wide memories only.`
      : `Search ranks results within ${project} first, then team-wide.`;
  return `${scope} Showing the top ${SEARCH_RESULT_LIMIT} matches, so pages are not shown.`;
}

function dimHistorical(row: MemoryRow): string | undefined {
  return isHistorical(row.status) ? "opacity-60" : undefined;
}

interface MemoryListViewProps {
  state: MemoriesPageState;
  onStateChange: (state: MemoriesPageState, options?: { replace?: boolean }) => void;
  viewSwitch: ReactNode;
  projects: readonly Project[];
  sources: readonly SourceOption[];
}

export function MemoryListView({ state, onStateChange, viewSwitch, projects, sources }: MemoryListViewProps) {
  const navigate = useNavigate();
  const request = memoryListRequest(state);
  const list = useMemoryList(request);
  const page = list.page;
  const projectOptions = projectFilterOptions(projects);
  const sourceOptions = sources.map((source) => ({ value: source.id, label: source.name }));
  const hasFilters = activeFilterKeys(state.filters).length > 0;
  const clear = () => onStateChange(clearFilters(state));

  const valueLabels: Record<FilterKey, (value: string) => string> = {
    type: memoryTypeLabel,
    status: memoryStatusLabel,
    source: (id) => sources.find((source) => source.id === id)?.name ?? id,
    project: (key) => projectName(key, projects),
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {viewSwitch}
        <SearchBox
          label="Search memories"
          value={state.query}
          onCommit={(query) => onStateChange(changePageState(state, { query }), { replace: true })}
        />
        <MemoryFiltersPopover
          filters={state.filters}
          sourceOptions={sourceOptions}
          projectOptions={projectOptions}
          onChange={(change) => onStateChange(changeFilters(state, change))}
          onClear={clear}
        />
        {page ? (
          <span className="ml-auto text-sm text-muted-foreground tabular-nums">
            {page.ranked ? pluralize(page.total, "candidate") : pluralize(page.total, "memory", "memories")}
          </span>
        ) : null}
      </div>

      <ActiveFilterChips
        filters={state.filters}
        valueLabels={valueLabels}
        onRemove={(key) => onStateChange(changeFilters(state, { [key]: null }))}
        onClear={clear}
      />

      {request.kind === "search" ? (
        <p className="flex items-start gap-2 text-sm text-muted-foreground">
          <Info aria-hidden className="mt-0.5 size-4 shrink-0" />
          {rankedSearchNotice(state, projectName(request.body.active_project, projects))}
        </p>
      ) : null}

      {list.isError ? (
        <ErrorNotice
          title="Could not load memories"
          message={errorMessage(list.error)}
          onRetry={() => void list.refetch()}
        />
      ) : (
        <DataTable
          aria-label="Memories"
          columns={MEMORY_COLUMNS}
          data={page?.rows ?? []}
          getRowId={(row) => row.id}
          onRowClick={(row) => navigate(row.target)}
          rowClassName={dimHistorical}
          loading={list.isPending}
          empty={
            <EmptyState
              title="No memories found"
              description="Try changing the filters or sync a source."
              action={
                hasFilters ? (
                  <Button variant="outline" onClick={clear}>
                    Clear filters
                  </Button>
                ) : undefined
              }
            />
          }
        />
      )}

      {page && !page.ranked ? (
        <Pagination
          page={state.page}
          pageSize={LIST_PAGE_SIZE}
          total={page.total}
          itemName="memory"
          itemNamePlural="memories"
          onPageChange={(next) => onStateChange({ ...state, page: next })}
        />
      ) : null}
    </div>
  );
}
