import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Brain, Database, Files, RefreshCw, ShieldCheck } from "lucide-react";
import client from "@/api/client";
import type {
  Memory,
  MemoryReviewListResponse,
  PaginatedResponse,
  Project,
  Source,
  Stats,
} from "@/api/types";
import { AsyncBoundary } from "@/components/admin/AsyncBoundary";
import { DataSurface } from "@/components/admin/DataSurface";
import { EmptyState } from "@/components/admin/EmptyState";
import { FilterSelect } from "@/components/admin/FilterSelect";
import { PageHeader } from "@/components/admin/PageHeader";
import { Pagination } from "@/components/admin/Pagination";
import { SearchInput } from "@/components/admin/SearchInput";
import { ConfidenceBadge, MemoryTypeBadge, StatusDot } from "@/components/admin/StatusBadge";
import { CrossProjectBanner } from "@/components/layout/CrossProjectBanner";
import { MemoryTypeIcon } from "@/components/memories/MemoryTypeIcon";
import { SourceIcon } from "@/components/sources/SourceIcon";
import { Toolbar } from "@/components/admin/Toolbar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { LIST_PAGE_SIZE } from "@/lib/constants";
import { timeAgo } from "@/utils/date";
import { useActiveProject } from "@/state/activeProject";

const TYPE_OPTIONS = [
  { value: "all", label: "All types" },
  { value: "fact", label: "Fact" },
  { value: "decision", label: "Decision" },
  { value: "convention", label: "Convention" },
  { value: "procedure", label: "Procedure" },
];

const STATUS_OPTIONS = [
  { value: "all", label: "All statuses" },
  { value: "active", label: "Active" },
  { value: "pending_review", label: "Pending review" },
  { value: "superseded", label: "Superseded" },
  { value: "retired", label: "Retired" },
];

// The narrow toggle defaults to "active project on top" (cross-project hits
// stay visible, ranked below). Flipping it to true asks the server to hard-
// narrow the workspace branch to the active project + SHARED.
const NARROW_TOGGLE_DEFAULT = false;

interface SourcesResponse {
  data?: Source[];
}

// Shape returned by POST /api/memories/search. Mirrors `SearchResult`
// (memforge.models). Only the fields the list view actually renders are
// declared here; the backend returns more (source URLs, freshness, etc.).
interface SearchHit {
  memory_id: string | null;
  memory_type: Memory["memory_type"] | null;
  summary: string;
  confidence: number;
  relevance_score: number;
  tags: string[];
  source_type: string | null;
  corroborated_by: number;
  last_observed_at: string | null;
  is_document_result: boolean;
  status: Memory["status"] | null;
}

interface SearchResponse {
  results: SearchHit[];
  total_candidates: number;
  retrieval_time_ms: number;
}

/**
 * The list view runs against two backend routes with very different shapes:
 *
 * - `GET /api/memories` -> `PaginatedResponse<Memory>` (admin/cross-project)
 * - `POST /api/memories/search` -> ranking-aware `{ results: SearchHit[] }`
 *
 * The renderer only needs a `Memory`-shaped row, so we normalize on the client.
 * That keeps the search route ranking-pure (no row hydration roundtrip) and
 * leaves the keyword GET route unchanged for the cross-project admin path.
 *
 * Visibility (predicate) and shape (adapter) are kept separate: `isMemoryHit`
 * decides whether a hit is a memory row at all; `searchHitToMemoryRow` is a
 * total adapter that callers must guard with `isMemoryHit` first.
 */
function isMemoryHit(hit: SearchHit): boolean {
  return hit.memory_id !== null && hit.memory_type !== null && !hit.is_document_result;
}

function searchHitToMemoryRow(hit: SearchHit): Memory {
  return {
    id: hit.memory_id as string,
    memory_type: hit.memory_type as Memory["memory_type"],
    content: hit.summary,
    content_hash: "",
    visibility: "workspace",
    owner_user_id: null,
    project_key: null,
    tags: hit.tags ?? [],
    confidence: hit.confidence,
    corroboration_count: hit.corroborated_by,
    contradiction_count: 0,
    status: hit.status ?? "active",
    retirement_reason: null,
    retired_at: null,
    superseded_at: null,
    superseded_by: null,
    replacement_reason: null,
    valid_from: null,
    valid_until: null,
    created_at: hit.last_observed_at ?? new Date().toISOString(),
    updated_at: hit.last_observed_at ?? new Date().toISOString(),
    extraction_context: null,
    entity_refs: [],
    sources: [],
    origin_source_type: hit.source_type,
    origin_client: null,
  };
}

function formatCount(value: number | undefined) {
  return typeof value === "number" ? value.toLocaleString() : "-";
}

function bucketCount(buckets: Stats["memories_by_status"] | undefined, key: string) {
  return buckets?.find((bucket) => bucket.key === key)?.count;
}

function OverviewCard({
  title,
  value,
  helper,
  icon: Icon,
  onClick,
}: {
  title: string;
  value: string;
  helper: string;
  icon: typeof Brain;
  onClick?: () => void;
}) {
  const interactive = Boolean(onClick);
  return (
    <Card
      size="sm"
      role={interactive ? "button" : undefined}
      tabIndex={interactive ? 0 : undefined}
      onClick={onClick}
      onKeyDown={(event) => {
        if (!interactive) return;
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          onClick?.();
        }
      }}
      className={interactive ? "cursor-pointer transition-colors hover:bg-muted/40" : undefined}
    >
      <CardHeader className="pb-0">
        <CardTitle className="flex items-center justify-between text-sm">
          <span>{title}</span>
          <Icon className="size-4 text-muted-foreground" />
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold tracking-tight">{value}</div>
        <p className="mt-1 text-xs text-muted-foreground">{helper}</p>
      </CardContent>
    </Card>
  );
}

export function MemoriesPage() {
  const [search, setSearch] = useState("");
  const [type, setType] = useState("all");
  const [status, setStatus] = useState("all");
  const [source, setSource] = useState("all");
  const [narrowToggle, setNarrowToggle] = useState(NARROW_TOGGLE_DEFAULT);
  const [page, setPage] = useState(0);
  const navigate = useNavigate();
  const { activeProjectKey, crossProjectMode } = useActiveProject();

  // Any filter change resets to the first page so the offset stays in range.
  const changeSearch = (value: string) => {
    setSearch(value);
    setPage(0);
  };
  const changeType = (value: string) => {
    setType(value);
    setPage(0);
  };
  const changeStatus = (value: string) => {
    setStatus(value);
    setPage(0);
  };
  const changeSource = (value: string) => {
    setSource(value);
    setPage(0);
  };
  const changeNarrow = (value: boolean) => {
    setNarrowToggle(value);
    setPage(0);
  };

  const sourcesQuery = useQuery<SourcesResponse | Source[]>({
    queryKey: ["sources"],
    queryFn: () => client.get("/api/sources").then((response) => response.data),
  });

  const projectsQuery = useQuery<Project[]>({
    queryKey: ["projects"],
    queryFn: () => client.get<Project[]>("/api/projects").then((response) => response.data),
  });

  const statsQuery = useQuery<Stats>({
    queryKey: ["stats"],
    queryFn: () => client.get("/api/stats").then((response) => response.data),
  });

  // The list view is a project-scoped browse by default and a ranked search
  // whenever the user types. `GET /api/memories` handles both empty-query
  // browsing (with `project=` narrowing the predicate-visible set) and the
  // explicit cross-project admin view; `POST /api/memories/search` runs only
  // when there is a query for the ranker to act on. Project-first relevance
  // weighting is meaningful only against a query, so an empty input collapses
  // the narrow-vs-project-first distinction and routes to the browse endpoint.
  // Both shapes normalize to a `Memory[]` row list before rendering.
  const hasQuery = search.trim().length > 0;
  const useSearchRoute = !crossProjectMode && activeProjectKey !== null && hasQuery;
  const memoriesEnabled = crossProjectMode || activeProjectKey !== null;
  const memoriesQuery = useQuery<PaginatedResponse<Memory>>({
    queryKey: [
      "memories",
      useSearchRoute ? "ranked" : "keyword",
      activeProjectKey,
      crossProjectMode,
      narrowToggle,
      search,
      type,
      status,
      source,
      page,
    ],
    enabled: memoriesEnabled,
    queryFn: async () => {
      if (useSearchRoute) {
        // The UI only ever asks for a project-bound view: the narrow toggle
        // hard-restricts results to the active project plus the shared bucket,
        // while the default leaves cross-project hits visible but down-weighted
        // by the ranker. Cross-project browsing flows through GET /api/memories
        // instead, so this route never needs the workspace-wide variant.
        const body = {
          query: search || "",
          memory_types: type !== "all" ? [type] : undefined,
          sources: source !== "all" ? [source] : undefined,
          status: status !== "all" ? status : undefined,
          active_project: activeProjectKey,
          scope_mode: narrowToggle ? "project" : "project-first",
          top_k: LIST_PAGE_SIZE,
        };
        const response = await client.post<SearchResponse>("/api/memories/search", body);
        const rows = response.data.results.filter(isMemoryHit).map(searchHitToMemoryRow);
        return {
          data: rows,
          total: response.data.total_candidates,
          limit: LIST_PAGE_SIZE,
          offset: 0,
        };
      }
      const response = await client.get<PaginatedResponse<Memory>>("/api/memories", {
        params: {
          search: search || undefined,
          type: type !== "all" ? type : undefined,
          status: status !== "all" ? status : undefined,
          source: source !== "all" ? source : undefined,
          project: !crossProjectMode && activeProjectKey ? activeProjectKey : undefined,
          limit: LIST_PAGE_SIZE,
          offset: page * LIST_PAGE_SIZE,
        },
      });
      return response.data;
    },
  });

  const reviewsQuery = useQuery<MemoryReviewListResponse>({
    queryKey: ["memory-reviews", "open"],
    queryFn: () =>
      client
        .get("/api/memory-reviews", { params: { status: "open", limit: 200 } })
        .then((response) => response.data),
  });

  const reviewByMemoryId = (() => {
    const map = new Map<string, string>();
    for (const review of reviewsQuery.data?.data ?? []) {
      map.set(review.challenger_memory_id, review.id);
      map.set(review.incumbent_memory_id, review.id);
    }
    return map;
  })();

  const memories = memoriesQuery.data?.data ?? [];
  const total = memoriesQuery.data?.total ?? 0;
  // The search route ranks a candidate set and trims to top_k. When the
  // candidate pool exceeds the visible page, the header advertises the wider
  // pool ("N candidates") so users know the list is a ranked window. When
  // they line up the keyword/admin route uses the same total.
  const headerCount =
    useSearchRoute && total > memories.length
      ? `${total.toLocaleString()} candidates`
      : `${memories.length.toLocaleString()} memories`;
  const sourcesData = sourcesQuery.data;
  const sourceList: Source[] = Array.isArray(sourcesData)
    ? sourcesData
    : Array.isArray(sourcesData?.data)
      ? sourcesData.data
      : [];
  const projectList = projectsQuery.data ?? [];
  const activeProject = activeProjectKey
    ? projectList.find((p) => p.key === activeProjectKey)
    : undefined;
  const activeProjectLabel = activeProject?.name ?? activeProjectKey ?? "Active project";
  const stats = statsQuery.data;
  const activeCount = bucketCount(stats?.memories_by_status, "active");
  const openReviewCount = reviewsQuery.data?.total;

  const showFirstChoiceEmpty = !crossProjectMode && activeProjectKey === null;

  return (
    <div className="space-y-4">
      <PageHeader
        title="Memories"
        description="Memory lifecycle, entities, and source coverage."
        actions={
          <Button
            type="button"
            variant="outline"
            onClick={() => {
              statsQuery.refetch();
              memoriesQuery.refetch();
              sourcesQuery.refetch();
              projectsQuery.refetch();
              reviewsQuery.refetch();
            }}
          >
            <RefreshCw className="size-4" />
            Refresh
          </Button>
        }
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <OverviewCard
          title="Total Memories"
          value={formatCount(stats?.total_memories)}
          helper={`${formatCount(activeCount)} active memories`}
          icon={Brain}
        />
        <OverviewCard
          title="Open Reviews"
          value={formatCount(openReviewCount)}
          helper="Open the review queue"
          icon={ShieldCheck}
          onClick={() => navigate("/review")}
        />
        <OverviewCard
          title="Entities"
          value={formatCount(stats?.total_entities)}
          helper="Canonical names tracked"
          icon={Database}
        />
        <OverviewCard
          title="Sources"
          value={formatCount(stats?.total_sources)}
          helper="Configured knowledge inputs"
          icon={Files}
        />
      </div>

      {crossProjectMode && <CrossProjectBanner />}

      <DataSurface>
        <div className="flex flex-col gap-3 border-b p-4 xl:flex-row xl:items-center xl:justify-between">
          <div>
            <h2 className="text-base font-semibold">Memory List</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {headerCount} in the current result set.
            </p>
          </div>
          <Toolbar className="xl:justify-end">
            <SearchInput value={search} onChange={changeSearch} placeholder="Filter memories..." />
            <FilterSelect value={type} onChange={changeType} options={TYPE_OPTIONS} label="Filter by type" />
            <FilterSelect
              value={status}
              onChange={changeStatus}
              options={STATUS_OPTIONS}
              label="Filter by status"
              className="w-full sm:w-44"
            />
            <FilterSelect
              value={source}
              onChange={changeSource}
              label="Filter by source"
              className="w-full sm:w-56"
              options={[
                { value: "all", label: "All sources" },
                ...sourceList.map((item) => ({ value: item.id, label: item.name })),
              ]}
            />
            {!crossProjectMode && activeProjectKey !== null && (
              <div
                role="group"
                aria-label="Project scope"
                className="inline-flex items-center rounded-md border bg-background p-0.5 text-sm"
              >
                <button
                  type="button"
                  onClick={() => changeNarrow(false)}
                  aria-pressed={!narrowToggle}
                  className={
                    "rounded px-3 py-1.5 transition-colors " +
                    (!narrowToggle
                      ? "bg-foreground text-background"
                      : "text-muted-foreground hover:text-foreground")
                  }
                >
                  {activeProjectLabel} on top
                </button>
                <button
                  type="button"
                  onClick={() => changeNarrow(true)}
                  aria-pressed={narrowToggle}
                  className={
                    "rounded px-3 py-1.5 transition-colors " +
                    (narrowToggle
                      ? "bg-foreground text-background"
                      : "text-muted-foreground hover:text-foreground")
                  }
                >
                  Only this project
                </button>
              </div>
            )}
          </Toolbar>
        </div>
        {showFirstChoiceEmpty ? (
          <div className="p-6">
            <EmptyState
              icon={Brain}
              title="Pick a project to start"
              description="Click the chip in the top right to choose what you're working on, or pick the cross-project view."
            />
          </div>
        ) : (
          <>
            <AsyncBoundary
              isLoading={memoriesQuery.isLoading}
              isError={memoriesQuery.isError}
              error={memoriesQuery.error}
              onRetry={() => memoriesQuery.refetch()}
              isEmpty={memories.length === 0}
              empty={
                <EmptyState
                  icon={Brain}
                  title="No memories found"
                  description="Try changing the filters or sync a source."
                />
              }
            >
              <div className="overflow-x-auto">
                <Table>
                  <TableHeader>
                    <TableRow className="hover:bg-transparent">
                      <TableHead className="w-12" />
                      <TableHead>Memory</TableHead>
                      <TableHead className="w-28">Type</TableHead>
                      <TableHead className="w-28">Confidence</TableHead>
                      <TableHead className="w-20">Sources</TableHead>
                      <TableHead className="w-24">Age</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {memories.map((memory) => {
                      const reviewId = reviewByMemoryId.get(memory.id);
                      const origin = memory.origin_source_type;
                      const target =
                        reviewId && memory.status === "pending_review"
                          ? `/review/${reviewId}`
                          : `/memories/${memory.id}`;
                      return (
                        <TableRow
                          key={memory.id}
                          className="cursor-pointer"
                          onClick={() => navigate(target)}
                        >
                          <TableCell>
                            <div className="flex items-center gap-1.5">
                              {origin ? (
                                <SourceIcon type={origin} client={memory.origin_client} className="size-4" />
                              ) : (
                                <MemoryTypeIcon type={memory.memory_type} className="size-4" />
                              )}
                              <StatusDot status={memory.status} />
                            </div>
                          </TableCell>
                          <TableCell>
                            <div className="max-w-2xl truncate text-sm font-medium">{memory.content}</div>
                            {memory.tags.length > 0 && (
                              <div className="mt-1 flex flex-wrap gap-1">
                                {memory.tags.slice(0, 3).map((tag) => (
                                  <Badge key={tag} variant="secondary" className="text-[11px]">
                                    {tag}
                                  </Badge>
                                ))}
                              </div>
                            )}
                          </TableCell>
                          <TableCell>
                            <MemoryTypeBadge type={memory.memory_type} />
                          </TableCell>
                          <TableCell>
                            <ConfidenceBadge confidence={memory.confidence} />
                          </TableCell>
                          <TableCell className="text-muted-foreground">
                            {memory.corroboration_count}
                          </TableCell>
                          <TableCell className="text-muted-foreground">
                            {timeAgo(memory.created_at)}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            </AsyncBoundary>
            {!useSearchRoute && (
              <Pagination
                page={page}
                pageSize={LIST_PAGE_SIZE}
                total={total}
                onPageChange={setPage}
                itemLabel="memories"
              />
            )}
          </>
        )}
      </DataSurface>
    </div>
  );
}
