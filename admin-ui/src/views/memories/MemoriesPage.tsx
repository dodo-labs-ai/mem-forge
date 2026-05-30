import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { Brain, Database, Files, RefreshCw, ShieldCheck } from "lucide-react";
import client from "@/api/client";
import type {
  Memory,
  MemoryReviewListResponse,
  PaginatedResponse,
  Source,
  Stats,
} from "@/api/types";
import { AsyncBoundary } from "@/components/admin/AsyncBoundary";
import { DataSurface } from "@/components/admin/DataSurface";
import { EmptyState } from "@/components/admin/EmptyState";
import { FilterSelect } from "@/components/admin/FilterSelect";
import { PageHeader } from "@/components/admin/PageHeader";
import { SearchInput } from "@/components/admin/SearchInput";
import { ConfidenceBadge, MemoryTypeBadge, StatusDot } from "@/components/admin/StatusBadge";
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

interface SourcesResponse {
  data?: Source[];
}

function formatCount(value: number | undefined) {
  return typeof value === "number" ? value.toLocaleString() : "—";
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
  const navigate = useNavigate();

  const sourcesQuery = useQuery<SourcesResponse | Source[]>({
    queryKey: ["sources"],
    queryFn: () => client.get("/api/sources").then((response) => response.data),
  });

  const statsQuery = useQuery<Stats>({
    queryKey: ["stats"],
    queryFn: () => client.get("/api/stats").then((response) => response.data),
  });

  const memoriesQuery = useQuery<PaginatedResponse<Memory>>({
    queryKey: ["memories", search, type, status, source],
    queryFn: () =>
      client
        .get("/api/memories", {
          params: {
            search: search || undefined,
            type: type !== "all" ? type : undefined,
            status: status !== "all" ? status : undefined,
            source: source !== "all" ? source : undefined,
            limit: LIST_PAGE_SIZE,
            offset: 0,
          },
        })
        .then((response) => response.data),
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
  const sourcesData = sourcesQuery.data;
  const sourceList: Source[] = Array.isArray(sourcesData)
    ? sourcesData
    : Array.isArray(sourcesData?.data)
      ? sourcesData.data
      : [];
  const stats = statsQuery.data;
  const activeCount = bucketCount(stats?.memories_by_status, "active");
  const openReviewCount = reviewsQuery.data?.total;

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

      <DataSurface>
        <div className="flex flex-col gap-3 border-b p-4 xl:flex-row xl:items-center xl:justify-between">
          <div>
            <h2 className="text-base font-semibold">Memory List</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {total.toLocaleString()} rows in the current result set.
            </p>
          </div>
          <Toolbar className="xl:justify-end">
            <SearchInput value={search} onChange={setSearch} placeholder="Filter memories..." />
            <FilterSelect value={type} onChange={setType} options={TYPE_OPTIONS} label="Filter by type" />
            <FilterSelect
              value={status}
              onChange={setStatus}
              options={STATUS_OPTIONS}
              label="Filter by status"
              className="w-full sm:w-44"
            />
            <FilterSelect
              value={source}
              onChange={setSource}
              label="Filter by source"
              className="w-full sm:w-56"
              options={[
                { value: "all", label: "All sources" },
                ...sourceList.map((item) => ({ value: item.id, label: item.name })),
              ]}
            />
          </Toolbar>
        </div>
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
                  <TableHead className="w-8" />
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
                        <StatusDot status={memory.status} />
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
      </DataSurface>
    </div>
  );
}
