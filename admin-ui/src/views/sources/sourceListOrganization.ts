export type SourceListSortMode = "newest" | "name" | "recently_synced";

interface OrganizableSource {
  id: string;
  name: string;
  type: string;
  created_at: string;
  last_sync: string | null;
  pinned_for_me?: boolean;
  doc_count: number;
}

interface OrganizableEntry<TSource extends OrganizableSource> {
  source: TSource;
  memory_count: number;
}

interface OrganizableGroup<TSource extends OrganizableSource> {
  project: { name: string } | null;
  sources: OrganizableEntry<TSource>[];
  docCount: number;
  memoryCount: number;
}

interface OrganizationOptions {
  query: string;
  pinnedOnly: boolean;
  sortMode: SourceListSortMode;
  typeLabels?: Readonly<Record<string, string>>;
}

function normalized(value: string): string {
  return value.trim().toLocaleLowerCase();
}

function timestamp(value: string | null): number {
  if (!value) return Number.NEGATIVE_INFINITY;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? Number.NEGATIVE_INFINITY : parsed;
}

function compareSources<TSource extends OrganizableSource>(
  left: OrganizableEntry<TSource>,
  right: OrganizableEntry<TSource>,
  sortMode: SourceListSortMode,
): number {
  const pinOrder = Number(Boolean(right.source.pinned_for_me)) - Number(Boolean(left.source.pinned_for_me));
  if (pinOrder !== 0) return pinOrder;

  let selectedOrder = 0;
  if (sortMode === "newest") {
    selectedOrder = timestamp(right.source.created_at) - timestamp(left.source.created_at);
  } else if (sortMode === "recently_synced") {
    selectedOrder = timestamp(right.source.last_sync) - timestamp(left.source.last_sync);
  }
  if (selectedOrder !== 0) return selectedOrder;

  const nameOrder = normalized(left.source.name).localeCompare(normalized(right.source.name));
  return nameOrder !== 0 ? nameOrder : left.source.id.localeCompare(right.source.id);
}

export function organizeSourceGroups<TSource extends OrganizableSource, TGroup extends OrganizableGroup<TSource>>(
  groups: readonly TGroup[],
  options: OrganizationOptions,
): TGroup[] {
  const query = normalized(options.query);
  return groups.flatMap((group) => {
    const projectMatches = query.length > 0 && normalized(group.project?.name ?? "").includes(query);
    const sources = group.sources
      .filter(({ source }) => {
        if (options.pinnedOnly && !source.pinned_for_me) return false;
        if (!query || projectMatches) return true;
        const typeLabel = options.typeLabels?.[source.type] ?? source.type;
        return [source.name, source.type, typeLabel].some((value) => normalized(value).includes(query));
      })
      .sort((left, right) => compareSources(left, right, options.sortMode));
    if (sources.length === 0) return [];
    return [{
      ...group,
      sources,
      docCount: sources.reduce((sum, entry) => sum + entry.source.doc_count, 0),
      memoryCount: sources.reduce((sum, entry) => sum + entry.memory_count, 0),
    }];
  });
}
