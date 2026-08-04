import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { ArrowRight, CheckCircle2, RefreshCw, ShieldCheck } from "lucide-react";
import { resourceClient } from "@/api/client";
import type { MemoryReviewListResponse } from "@/api/types";
import { AsyncBoundary } from "@/components/admin/AsyncBoundary";
import { EmptyState } from "@/components/admin/EmptyState";
import { PageHeader } from "@/components/admin/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { timeAgo } from "@/utils/date";

const REVIEW_QUEUE_LIMIT = 100;

function useReviewQueue() {
  return useQuery<MemoryReviewListResponse>({
    queryKey: ["memory-reviews", "open", "queue"],
    queryFn: () =>
      resourceClient
        .get("/memory-reviews", {
          params: { status: "open", limit: REVIEW_QUEUE_LIMIT },
        })
        .then((response) => response.data),
  });
}

function preview(value: string | null | undefined): string {
  if (!value) return "Not available in this proposal";
  return value.length > 180 ? `${value.slice(0, 177)}…` : value;
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
        description="Decide which memory state MemForge should use. Technical details stay available when you need them."
        actions={
          <Button type="button" variant="outline" onClick={() => queueQuery.refetch()}>
            <RefreshCw className="size-4" />
            Refresh list
          </Button>
        }
      />

      <div className="flex items-center justify-between rounded-lg border bg-muted/20 px-4 py-3">
        <div>
          <div className="text-sm font-medium">Decisions waiting</div>
          <div className="text-xs text-muted-foreground">
            {total.toLocaleString()} {total === 1 ? "review needs" : "reviews need"} your input
          </div>
        </div>
        <ShieldCheck className="size-5 text-amber-500" />
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
        <div className="space-y-3">
          {reviews.map((review) => (
            <Card key={review.id} className="transition-colors hover:border-foreground/20">
              <CardContent className="p-5">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                  <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                      <Badge variant="secondary">
                        {review.source_name ?? (review.review_origin === "lifecycle" ? "Source sync" : "Memory update")}
                      </Badge>
                      <span>{timeAgo(review.created_at)}</span>
                    </div>
                    <h2 className="mt-3 text-base font-semibold">{review.presentation.summary}</h2>
                    <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
                      {review.presentation.why_human}
                    </p>
                    <div className="mt-4 grid gap-3 md:grid-cols-2">
                      <div className="rounded-md border bg-muted/20 p-3">
                        <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                          {review.presentation.current_label}
                        </div>
                        <p className="mt-1 text-sm leading-relaxed">
                          {preview(review.incumbent?.content)}
                        </p>
                      </div>
                      <div className="rounded-md border bg-muted/20 p-3">
                        <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                          {review.presentation.proposed_label}
                        </div>
                        <p className="mt-1 text-sm leading-relaxed">
                          {preview(review.challenger?.content)}
                        </p>
                      </div>
                    </div>
                  </div>
                  <Button className="shrink-0" onClick={() => navigate(`/review/${review.id}`)}>
                    Review decision <ArrowRight className="size-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </AsyncBoundary>
    </div>
  );
}
