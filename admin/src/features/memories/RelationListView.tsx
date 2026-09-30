import type { ReactNode } from "react";
import { errorMessage } from "@/lib/errors";
import { formatCount, pluralize } from "@/lib/format";
import { DataTable, EmptyState, ErrorNotice, Pagination } from "@/patterns";
import { ToggleGroup, ToggleGroupItem } from "@/ui/toggle-group";
import { useRelationPage } from "./api";
import { LIST_PAGE_SIZE } from "./constants";
import { changePageState, type MemoriesPageState } from "./model/listState";
import { RELATION_FILTER_NAMES, RELATION_LABELS, isRelationLabel, totalRelations, type RelationCounts } from "./model/relations";
import { RELATION_COLUMNS, relationPairId } from "./relationColumns";

/** The relation filter value that shows every label. */
const ALL_RELATIONS = "all";

interface RelationListViewProps {
  state: MemoriesPageState;
  onStateChange: (state: MemoriesPageState) => void;
  viewSwitch: ReactNode;
  counts: RelationCounts | undefined;
}

export function RelationListView({ state, onStateChange, viewSwitch, counts }: RelationListViewProps) {
  const relations = useRelationPage(state.relation, state.page);
  const total = relations.data?.total;
  const filters = [
    { value: ALL_RELATIONS, label: "All", count: counts ? totalRelations(counts) : undefined },
    ...RELATION_LABELS.map((label) => ({ value: label, label: RELATION_FILTER_NAMES[label], count: counts?.[label] })),
  ];

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        {viewSwitch}
        <p className="text-sm text-muted-foreground">Relations between Memories from different documents</p>
        {total !== undefined ? (
          <span className="ml-auto text-sm text-muted-foreground tabular-nums">{pluralize(total, "relation")}</span>
        ) : null}
      </div>

      <ToggleGroup
        value={[state.relation ?? ALL_RELATIONS]}
        onValueChange={(values: string[]) => {
          const next = values[0];
          if (next !== undefined) onStateChange(changePageState(state, { relation: isRelationLabel(next) ? next : null }));
        }}
        variant="outline"
        size="sm"
        aria-label="Relation"
      >
        {filters.map((filter) => (
          <ToggleGroupItem key={filter.value} value={filter.value}>
            {filter.label}
            {filter.count !== undefined ? (
              <span className="text-muted-foreground tabular-nums">{formatCount(filter.count)}</span>
            ) : null}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>

      {relations.isError ? (
        <ErrorNotice
          title="Could not load relations"
          message={errorMessage(relations.error)}
          onRetry={() => void relations.refetch()}
        />
      ) : (
        <DataTable
          aria-label="Relations"
          columns={RELATION_COLUMNS}
          data={relations.data?.data ?? []}
          getRowId={relationPairId}
          loading={relations.isPending}
          empty={
            <EmptyState
              title="No relations found"
              description="Discovery records relations between Memories from different documents."
            />
          }
        />
      )}

      {total !== undefined ? (
        <Pagination
          page={state.page}
          pageSize={LIST_PAGE_SIZE}
          total={total}
          itemName="relation"
          onPageChange={(page) => onStateChange({ ...state, page })}
        />
      ) : null}
    </div>
  );
}
