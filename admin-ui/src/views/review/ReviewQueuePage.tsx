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
import {
  buildReviewChangeCue,
  type ReviewCueExcerpt,
} from "@/views/review/reviewQueueCue";

type ReviewQueueFilter = "all" | "lifecycle" | "cross_source_conflict" | "supersede";

function useReviewQueue(page: number, filter: ReviewQueueFilter) {
  return useQuery<MemoryReviewListResponse>({
    queryKey: ["memory-reviews", "open", "queue", page, filter],
    queryFn: () =>
      resourceClient
        .get("/memory-reviews", {
          params: {
            status: "open",
            origin: filter === "lifecycle" ? "lifecycle" : undefined,
            kind:
              filter === "cross_source_conflict" || filter === "supersede"
                ? filter
                : undefined,
            limit: REVIEW_QUEUE_PAGE_SIZE,
            offset: page * REVIEW_QUEUE_PAGE_SIZE,
          },
        })
        .then((response) => response.data),
  });
}

function CueExcerpt({ value, emptyText }: { value: ReviewCueExcerpt | null; emptyText: string }) {
  if (!value) return <span>{emptyText}</span>;
  return (
    <span>
      {value.before}
      {value.before && value.changed ? " " : null}
      {value.changed ? (
        <mark className="bg-amber-100/80 px-0.5 text-foreground dark:bg-amber-900/40">
          {value.changed}
        </mark>
      ) : null}
      {value.changed && value.after ? " " : null}
      {value.after}
    </span>
  );
}

function pageFromSearchParams(searchParams: URLSearchParams): number {
  const value = Number.parseInt(searchParams.get("page") ?? "1", 10);
  return Number.isFinite(value) && value > 0 ? value - 1 : 0;
}

export function ReviewQueuePage() {
  const navigate = useNavigate();
  const [searchParams, setSearchParams] = useSearchParams();
  const page = pageFromSearchParams(searchParams);
  const rawFilter = searchParams.get("filter");
  const filter: ReviewQueueFilter =
    rawFilter === "lifecycle" ||
    rawFilter === "cross_source_conflict" ||
    rawFilter === "supersede"
      ? rawFilter
      : "all";
  const queueQuery = useReviewQueue(page, filter);
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

  const setFilter = (nextFilter: ReviewQueueFilter) => {
    const next = new URLSearchParams(searchParams);
    next.delete("page");
    if (nextFilter === "all") next.delete("filter");
    else next.set("filter", nextFilter);
    setSearchParams(next);
  };

  return (
    <div className="space-y-4">
      <PageHeader
        title="Review queue"
        description="Resolve lifecycle proposals and conflict findings. Open a Review for evidence, consequences, and technical details."
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <label className="sr-only" htmlFor="review-queue-filter">
              Filter reviews
            </label>
            <select
              id="review-queue-filter"
              value={filter}
              onChange={(event) => setFilter(event.target.value as ReviewQueueFilter)}
              className="h-9 rounded-md border bg-background px-3 text-sm"
            >
              <option value="all">All decisions</option>
              <option value="lifecycle">Source lifecycle</option>
              <option value="cross_source_conflict">Cross-source conflicts</option>
              <option value="supersede">Memory updates</option>
            </select>
            <Button type="button" variant="outline" onClick={() => queueQuery.refetch()}>
              <RefreshCw className="size-4" />
              Refresh list
            </Button>
          </div>
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
            {reviews.map((review) => {
              const cue = buildReviewChangeCue(
                review.incumbent?.content,
                review.challenger?.content,
              );
              return <li key={review.id}>
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
                      <Badge variant="outline">{review.presentation.decision_label}</Badge>
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
                        <CueExcerpt value={cue.current} emptyText="Current memory unavailable" />
                      </p>
                      <p className="min-w-0 truncate text-sm text-muted-foreground">
                        <span className="mr-2 text-[11px] font-medium uppercase tracking-wide">
                          {review.presentation.proposed_label}
                        </span>
                        <CueExcerpt
                          value={cue.proposed}
                          emptyText={review.presentation.proposed_empty_text}
                        />
                      </p>
                    </div>
                  </div>
                  <span className="mt-1 flex shrink-0 items-center text-muted-foreground">
                    <span className="sr-only">Open review</span>
                    <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5 group-hover:text-foreground" />
                  </span>
                </button>
              </li>
            })}
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
