import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { CheckCircle2, RefreshCw, ShieldCheck, Sparkles } from "lucide-react";
import client from "@/api/client";
import type {
  MemoryReviewListResponse,
  MemoryReviewMemorySummary,
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

const AGENT_SESSION_SOURCE_TYPE = "agent_session";

function MissingSnapshotLabel() {
  return <span className="italic text-muted-foreground">Unavailable</span>;
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

function isAgentSessionMemory(memory: MemoryReviewMemorySummary | null | undefined): boolean {
  return memory?.origin_source_type === AGENT_SESSION_SOURCE_TYPE;
}

export function ReviewQueuePage() {
  const navigate = useNavigate();
  const queueQuery = useReviewQueue();
  const reviews = queueQuery.data?.data ?? [];
  const total = queueQuery.data?.total ?? 0;

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
                  const challenger = review.challenger;
                  const incumbent = review.incumbent;
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
                            {challenger?.content ?? <MissingSnapshotLabel />}
                          </div>
                          {isAgentSessionMemory(challenger) && (
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
                          {incumbent?.content ?? <MissingSnapshotLabel />}
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
