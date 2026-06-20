import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { CheckCircle2, RefreshCw, ShieldCheck, Sparkles } from "lucide-react";
import client from "@/api/client";
import type {
  MemoryReviewListResponse,
  MemoryReviewSummary,
  MemorySource,
} from "@/api/types";
import { AsyncBoundary } from "@/components/admin/AsyncBoundary";
import { DataSurface } from "@/components/admin/DataSurface";
import { EmptyState } from "@/components/admin/EmptyState";
import { PageHeader } from "@/components/admin/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { timeAgo } from "@/utils/date";

const REVIEW_QUEUE_LIMIT = 100;

interface MemorySnapshot {
  id: string;
  content: string;
  confidence: number;
  corroboration_count: number;
  is_agent_session: boolean;
}

const AGENT_SESSION_SOURCE_TYPE = "agent_session";

function hasAgentSessionSource(sources: MemorySource[] | null | undefined): boolean {
  return Array.isArray(sources)
    ? sources.some((source) => source.source_type === AGENT_SESSION_SOURCE_TYPE)
    : false;
}

function useReviewQueue() {
  return useQuery<MemoryReviewListResponse>({
    queryKey: ["memory-reviews", "open", "queue"],
    queryFn: () =>
      client
        .get("/api/memory-reviews", {
          params: { status: "open", limit: REVIEW_QUEUE_LIMIT },
        })
        .then((response) => response.data),
  });
}

function useMemorySnapshots(reviews: MemoryReviewSummary[]) {
  const ids = Array.from(
    new Set(
      reviews.flatMap((review) => [review.incumbent_memory_id, review.challenger_memory_id])
    )
  );
  return useQuery<Record<string, MemorySnapshot>>({
    queryKey: ["memory-review-snapshots", ids.sort().join(",")],
    enabled: ids.length > 0,
    queryFn: async () => {
      const entries = await Promise.all(
        ids.map((id) =>
          client
            .get(`/api/memories/${id}`, { params: { include_private: "true" } })
            .then((response) => [id, response.data] as const)
            .catch(() => [id, null] as const)
        )
      );
      const snapshots: Record<string, MemorySnapshot> = {};
      for (const [id, memory] of entries) {
        if (memory) {
          snapshots[id] = {
            id,
            content: memory.content,
            confidence: memory.confidence,
            corroboration_count: memory.corroboration_count,
            is_agent_session: hasAgentSessionSource(memory.sources),
          };
        }
      }
      return snapshots;
    },
  });
}

export function ReviewQueuePage() {
  const navigate = useNavigate();
  const queueQuery = useReviewQueue();
  const reviews = queueQuery.data?.data ?? [];
  const total = queueQuery.data?.total ?? 0;
  const snapshotsQuery = useMemorySnapshots(reviews);
  const snapshots = snapshotsQuery.data ?? {};

  return (
    <div className="space-y-4">
      <PageHeader
        title="Review queue"
        description="Pending memory updates that need a human decision before they go live."
        actions={
          <Button
            type="button"
            variant="outline"
            onClick={() => {
              queueQuery.refetch();
              snapshotsQuery.refetch();
            }}
          >
            <RefreshCw className="size-4" />
            Refresh
          </Button>
        }
      />

      <DataSurface>
        <div className="flex flex-col gap-3 border-b p-4 xl:flex-row xl:items-center xl:justify-between">
          <div>
            <h2 className="text-base font-semibold">Pending decisions</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              {total.toLocaleString()} reviews waiting. Open one to compare and decide.
            </p>
          </div>
        </div>
        <AsyncBoundary
          isLoading={queueQuery.isLoading}
          isError={queueQuery.isError}
          error={queueQuery.error}
          onRetry={() => queueQuery.refetch()}
          isEmpty={reviews.length === 0}
          empty={
            <EmptyState
              icon={CheckCircle2}
              title="All clear"
              description="No memory reviews are pending. New ones appear here when sync flags a risky update."
            />
          }
        >
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead className="w-8" />
                  <TableHead>Proposed update</TableHead>
                  <TableHead>Current memory</TableHead>
                  <TableHead className="w-40">Reason</TableHead>
                  <TableHead className="w-32">Age</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {reviews.map((review) => {
                  const challenger = snapshots[review.challenger_memory_id];
                  const incumbent = snapshots[review.incumbent_memory_id];
                  return (
                    <TableRow
                      key={review.id}
                      className="cursor-pointer"
                      onClick={() => navigate(`/review/${review.id}`)}
                    >
                      <TableCell>
                        <ShieldCheck className="size-4 text-amber-500" />
                      </TableCell>
                      <TableCell>
                        <div className="flex max-w-xl items-start gap-2">
                          <div className="min-w-0 flex-1 truncate text-sm font-medium">
                            {challenger?.content ?? "Loading..."}
                          </div>
                          {challenger?.is_agent_session && (
                            <Badge
                              variant="outline"
                              className="shrink-0 gap-1 text-[10px]"
                              title="Generated agent-session summary"
                            >
                              <Sparkles className="size-3" />
                              agent-session
                            </Badge>
                          )}
                        </div>
                        {challenger && (
                          <div className="mt-1 text-xs text-muted-foreground">
                            confidence {challenger.confidence.toFixed(2)} ·{" "}
                            {challenger.corroboration_count} source
                            {challenger.corroboration_count === 1 ? "" : "s"}
                          </div>
                        )}
                      </TableCell>
                      <TableCell>
                        <div className="max-w-xl truncate text-sm text-muted-foreground">
                          {incumbent?.content ?? "Loading..."}
                        </div>
                        {incumbent && (
                          <div className="mt-1 text-xs text-muted-foreground">
                            confidence {incumbent.confidence.toFixed(2)} ·{" "}
                            {incumbent.corroboration_count} source
                            {incumbent.corroboration_count === 1 ? "" : "s"}
                          </div>
                        )}
                      </TableCell>
                      <TableCell>
                        {review.is_stale ? (
                          <Badge variant="secondary" className="text-[11px]">
                            Stale
                          </Badge>
                        ) : (
                          <span className="text-sm text-muted-foreground">
                            {review.reason ?? "—"}
                          </span>
                        )}
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {timeAgo(review.created_at)}
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
