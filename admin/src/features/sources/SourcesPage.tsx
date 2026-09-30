import { Files, Plus } from "lucide-react";
import { useMemo, useState } from "react";
import { useQueries } from "@tanstack/react-query";
import { unwrap, useApi } from "@/api";
import { localSyncState, useLocalSyncStatus } from "@/features/local-sync";
import { pluralize } from "@/lib/format";
import {
  ActionCard,
  ConfirmDialog,
  DataTable,
  EmptyState,
  ErrorNotice,
  FilterBar,
  GroupHeader,
  PageHeader,
  type DataTableGroup,
} from "@/patterns";
import { Button } from "@/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/ui/select";
import { errorMessage } from "@/lib/errors";
import {
  sourceKeys,
  useDeleteSource,
  useLocalSyncJobs,
  useProjects,
  useSetSortMode,
  useSourceListPreferences,
  useSources,
} from "./api";
import { V1_SOURCES_PATH } from "./constants";
import type { SourceListSortMode } from "./model/sourceListOrganization";
import type { LocalDaemonReadiness } from "./model/sourceReadiness";
import {
  buildSourceRowGroups,
  sourcesNeedingAttention,
  type SourceFilter,
  type SourceRow,
} from "./model/sourceRows";
import { sourceProjectBinding } from "./model/projectBinding";
import type { ResolvedBySource } from "./model/projectGrouping";
import type { Source } from "./model/types";
import { SOURCE_COLUMNS } from "./sourceColumns";
import { SourceTableContext } from "./sourceTableContext";
import { SourceDetailDrawer } from "./SourceDetailDrawer";
import { useSourceActions } from "./useSourceActions";

const SORT_LABELS: Record<SourceListSortMode, string> = {
  newest: "Newest first",
  name: "Name",
  recently_synced: "Recently synced",
};
const DEFAULT_SORT: SourceListSortMode = "newest";
/** "Needs you" cards shown above the table before the rest collapse into the filter. */
const ATTENTION_CARD_LIMIT = 3;

function useTypeLabels(): Record<string, string> {
  const api = useApi();
  const [genes] = useQueries({
    queries: [{ queryKey: ["genes"], queryFn: () => unwrap(api.GET("/api/v1/genes")) }],
  });
  return useMemo(
    () => Object.fromEntries((genes.data ?? []).map((gene) => [gene.name, gene.display_name])),
    [genes.data],
  );
}

function useResolvedProjects(sources: Source[]): ResolvedBySource {
  const api = useApi();
  const byField = sources.filter((source) => sourceProjectBinding(source)?.mode === "by_field");
  const results = useQueries({
    queries: byField.map((source) => ({
      queryKey: [...sourceKeys.all, "resolved-projects", source.id],
      queryFn: () =>
        unwrap(api.GET("/api/v1/sources/{source_id}/projects/resolved", { params: { path: { source_id: source.id } } })),
    })),
  });
  const resolved: ResolvedBySource = {};
  results.forEach((result, index) => {
    const source = byField[index];
    if (source && result.data) resolved[source.id] = result.data.projects;
  });
  return resolved;
}

export function SourcesPage() {
  const sourcesQuery = useSources();
  const { jobs } = useLocalSyncJobs();
  const projectsQuery = useProjects();
  const preferences = useSourceListPreferences();
  const setSortMode = useSetSortMode();
  const deleteSource = useDeleteSource();
  const daemonQuery = useLocalSyncStatus();
  const typeLabels = useTypeLabels();
  const resolvedBySource = useResolvedProjects(sourcesQuery.sources);

  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState<SourceFilter>("all");
  const [openRowKey, setOpenRowKey] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<Source | null>(null);

  const daemonState = localSyncState(daemonQuery);
  const daemon: LocalDaemonReadiness =
    daemonState === "online" ? "ready" : daemonState === "offline" ? "unavailable" : "checking";
  const sortMode = preferences.data?.sort_mode ?? DEFAULT_SORT;
  const projects = projectsQuery.data ?? [];

  const groups = buildSourceRowGroups({
    sources: sourcesQuery.sources,
    projects,
    resolvedBySource,
    jobs,
    daemon,
    query: search,
    filter,
    sortMode,
    typeLabels,
  });
  const attention = sourcesNeedingAttention(sourcesQuery.sources, jobs, daemon);
  const pinnedCount = sourcesQuery.sources.filter((source) => source.pinned_for_me).length;
  const rows = groups.flatMap((group) => group.rows);
  const openRow = rows.find((row) => row.key === openRowKey) ?? null;
  const openGroup = groups.find((group) => group.rows.some((row) => row.key === openRowKey));

  function viewDetails(source: Source) {
    const row = rows.find((candidate) => candidate.source.id === source.id);
    if (row) setOpenRowKey(row.key);
  }
  const actions = useSourceActions({ onViewDetails: viewDetails });

  const tableGroups: DataTableGroup<SourceRow>[] = groups.map((group) => ({
    id: group.key,
    header: (
      <GroupHeader
        title={group.title}
        description={group.description}
        meta={`${pluralize(group.rows.length, "source")}, ${pluralize(group.memoryCount, "memory", "memories")}`}
      />
    ),
    rows: group.rows,
  }));

  const noSources = !sourcesQuery.isPending && sourcesQuery.sources.length === 0;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Sources"
        description="Sources are the places MemForge reads from, such as wikis, issue trackers, repositories and coding-agent sessions. Each sync turns new and changed content into memories your agents can search."
        actions={
          <Button nativeButton={false} render={<a href={V1_SOURCES_PATH} />}>
            <Plus />
            Add source
          </Button>
        }
      />

      {sourcesQuery.isError ? (
        <ErrorNotice
          title="Could not load sources"
          message={errorMessage(sourcesQuery.error)}
          onRetry={() => void sourcesQuery.refetch()}
        />
      ) : null}

      {attention.length > 0 ? (
        <section aria-label="Needs you" className="space-y-2">
          <h2 className="text-sm font-semibold text-foreground">
            Needs you <span className="font-normal text-muted-foreground">{attention.length}</span>
          </h2>
          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {attention.slice(0, ATTENTION_CARD_LIMIT).map(({ source, attention: item }) => (
              <ActionCard
                key={source.id}
                tone={item.reason === "overdue" || item.reason === "unmapped" ? "warn" : "danger"}
                title={source.name}
                description={item.title}
                action={
                  <Button size="sm" variant="outline" onClick={() => actions.runAttention(item.action, source)}>
                    {item.actionLabel}
                  </Button>
                }
              />
            ))}
          </div>
        </section>
      ) : null}

      {noSources ? (
        <EmptyState
          icon={Files}
          title="No sources yet"
          description="Add a wiki, issue tracker, repository or folder, and MemForge will turn its content into memories."
          action={<Button nativeButton={false} render={<a href={V1_SOURCES_PATH} />}>Add source</Button>}
        />
      ) : (
        <div className="space-y-3">
          <FilterBar
            search={search}
            onSearchChange={setSearch}
            searchPlaceholder="Search sources"
            filters={[
              { value: "all", label: "All", count: sourcesQuery.sources.length },
              { value: "needs_you", label: "Needs you", count: attention.length },
              { value: "pinned", label: "Pinned", count: pinnedCount },
            ]}
            filter={filter}
            onFilterChange={setFilter}
            trailing={
              <Select
                value={sortMode}
                onValueChange={(value) => {
                  if (value) setSortMode.mutate(value as SourceListSortMode);
                }}
              >
                <SelectTrigger size="sm" aria-label="Sort sources" className="w-44">
                  <SelectValue>{(value: SourceListSortMode) => SORT_LABELS[value]}</SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {(Object.keys(SORT_LABELS) as SourceListSortMode[]).map((mode) => (
                    <SelectItem key={mode} value={mode}>
                      {SORT_LABELS[mode]}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            }
          />
          <SourceTableContext.Provider
            value={{ typeLabels, actions, onViewDetails: viewDetails, onDelete: setPendingDelete }}
          >
            <DataTable
              aria-label="Sources"
              columns={SOURCE_COLUMNS}
              data={tableGroups}
              getRowId={(row) => row.key}
              onRowClick={(row) => setOpenRowKey(row.key)}
              selectedRowId={openRowKey}
              rowClassName={(row) => (row.source.status === "paused" ? "opacity-70" : undefined)}
              loading={sourcesQuery.isPending}
              empty={<EmptyState title="No sources match" description="Try another search or filter." />}
            />
          </SourceTableContext.Provider>
        </div>
      )}

      <SourceDetailDrawer
        row={openRow}
        typeLabel={openRow ? (typeLabels[openRow.source.type] ?? openRow.source.type) : ""}
        projectName={openGroup?.title ?? ""}
        actions={actions}
        onOpenChange={(open) => {
          if (!open) setOpenRowKey(null);
        }}
      />

      <ConfirmDialog
        open={pendingDelete !== null}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(null);
        }}
        title={`Delete ${pendingDelete?.name ?? "source"}?`}
        description="This removes the source and the content it synced. Memories that no other source supports are retired and stop appearing in searches."
        confirmLabel="Delete source"
        tone="danger"
        confirmationText={pendingDelete?.name}
        onConfirm={() => deleteSource.mutateAsync(pendingDelete!.id)}
      />
    </div>
  );
}
