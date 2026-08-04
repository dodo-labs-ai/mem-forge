import { useQuery } from "@tanstack/react-query";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ArrowRight, CheckCircle2, RefreshCw } from "lucide-react";
import { resourceClient } from "@/api/client";
import type { MemoryReviewListResponse } from "@/api/types";
import { AsyncBoundary } from "@/components/admin/AsyncBoundary";
import { DataSurface } from "@/components/admin/DataSurface";
import { EmptyState } from "@/components/admin/EmptyState";
import { PageHeader } from "@/components/admin/PageHeader";
import { Pagination } from "@/components/admin/Pagination";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { REVIEW_QUEUE_PAGE_SIZE } from "@/lib/constants";
import { timeAgo } from "@/utils/date";

function useReviewQueue(page: number) {
  return useQuery<MemoryReviewListResponse>({
    queryKey: ["memory-reviews", "open", "queue", page],
    queryFn: () =>
      resourceClient
        .get("/memory-reviews", {
          params: {
            status: "open",
            limit: REVIEW_QUEUE_PAGE_SIZE,
            offset: page * REVIEW_QUEUE_PAGE_SIZE,
          },
        })
        .then((response) => response.data),
  });
}

function preview(value: string | null | undefined): string {
  if (!value) return "Not available in this proposal";
  return value.length > 120 ? `${value.slice(0, 117)}…` : value;
}

function pageFromSearchParams(searchParams: URLSearchParams): number {
  const value = Number.parseInt(searchParams.get("page") ?? "1", 10);
  return Number.isFinite(value) && value > 0 ? value - 1 : 0;
}

export function ReviewQueuePage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const page = pageFromSearchParams(searchParams);
  const queueQuery = useReviewQueue(page);
  const reviews = queueQuery.data?.data ?? [];
  const total = queueQuery.data?.total ?? 0;

  const setPage = (nextPage: number) => {
    const next = new URLSearchParams(searchParams);
    if (nextPage === 0) {
      next.delete("page");
    } else {
      next.set("page", String(nextPage + 1));
    }
    setSearchParams(next);
  };

  return (
    <div className="space-y-4">
      <PageHeader
        title="Review queue"
        description="Decide which memory state MemForge should use. Open a Review for evidence and technical details."
        actions={
          <Button type="button" variant="outline" onClick={() => queueQuery.refetch()}>
            <RefreshCw className="size-4" />
            Refresh list
          </Button>
        }
      />

      <DataSurface>
        <div className="border-b px-4 py-3">
          <div className="text-sm font-medium">Decisions waiting</div>
          <div className="text-xs text-muted-foreground">
            {total.toLocaleString()} {total === 1 ? "review needs" : "reviews need"} your input
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
              description="No current memory decisions need your attention."
            />
          }
        >
          <ul className="divide-y divide-border">
            {reviews.map((review) => (
              <li key={review.id}>
                <button
                  type="button"
                  onClick={() => navigate(`/review/${review.id}`)}
                  className="group flex w-full items-start gap-4 px-4 py-3 text-left transition-colors hover:bg-accent/40"
                >
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      <Badge variant="secondary">
                        {review.source_name ??
                          (review.review_origin === "lifecycle" ? "Source sync" : "Memory update")}
                      </Badge>
                      <span>{timeAgo(review.created_at)}</span>
                    </div>
                    <h2 className="mt-1.5 text-sm font-semibold">
                      {review.presentation.summary}
                    </h2>
                    <div className="mt-2 grid gap-x-6 gap-y-1 lg:grid-cols-2">
                      <p className="min-w-0 truncate text-sm text-muted-foreground">
                        <span className="mr-2 text-[11px] font-medium uppercase tracking-wide">
                          {review.presentation.current_label}
                        </span>
                        {preview(review.incumbent?.content)}
                      </p>
                      <p className="min-w-0 truncate text-sm text-muted-foreground">
                        <span className="mr-2 text-[11px] font-medium uppercase tracking-wide">
                          {review.presentation.proposed_label}
                        </span>
                        {preview(review.challenger?.content)}
                      </p>
                    </div>
                  </div>
                  <span className="mt-1 flex shrink-0 items-center gap-1 text-sm font-medium">
                    Review
                    <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
                  </span>
                </button>
              </li>
            ))}
          </ul>
        </AsyncBoundary>

        <Pagination
          page={page}
          pageSize={REVIEW_QUEUE_PAGE_SIZE}
          total={total}
          onPageChange={setPage}
          itemLabel="reviews"
        />
      </DataSurface>
    </div>
  );
}
