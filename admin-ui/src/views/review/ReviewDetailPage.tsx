import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  ChevronDown,
  ExternalLink,
  Loader2,
  ShieldCheck,
} from "lucide-react";
import { resourceClient } from "@/api/client";
import type {
  MemoryReviewActionPresentation,
  MemoryReviewDetail,
  MemoryReviewMemorySummary,
} from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatDateTime, timeAgo } from "@/utils/date";

function extractError(error: unknown): string | null {
  if (!error) return null;
  if (typeof error === "object" && error !== null && "response" in error) {
    const response = (error as { response?: { data?: { detail?: unknown; error?: string } } }).response;
    const detail = response?.data?.detail;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object" && "message" in detail) {
      return String((detail as { message: unknown }).message);
    }
    if (response?.data?.error) return response.data.error;
  }
  return error instanceof Error ? error.message : null;
}

function StateCard({
  label,
  memory,
  emptyCopy,
}: {
  label: string;
  memory: MemoryReviewMemorySummary | null;
  emptyCopy: string;
}) {
  return (
    <div className="rounded-lg border bg-muted/20 p-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {label}
        </div>
        {memory?.status && <Badge variant="outline">{memory.status.replaceAll("_", " ")}</Badge>}
      </div>
      <p className="mt-3 whitespace-pre-wrap text-sm leading-6">
        {memory?.content || emptyCopy}
      </p>
      {memory?.sources?.length ? (
        <div className="mt-4 space-y-2 border-t pt-3">
          {memory.sources.slice(0, 3).map((source, index) => (
            <div key={`${source.doc_id}-${index}`} className="text-xs text-muted-foreground">
              <div className="flex items-center gap-2">
                <span className="font-medium text-foreground">
                  {source.doc_title ?? source.doc_id}
                </span>
                <Badge variant="secondary" className="text-[10px]">
                  {source.source_type}
                </Badge>
                {source.source_url && (
                  <a
                    href={source.source_url}
                    target="_blank"
                    rel="noreferrer"
                    className="inline-flex items-center gap-1 hover:text-foreground"
                  >
                    Open source <ExternalLink className="size-3" />
                  </a>
                )}
              </div>
              {source.excerpt && (
                <blockquote className="mt-1 border-l-2 pl-2 leading-relaxed">
                  {source.excerpt}
                </blockquote>
              )}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function ActionExplanation({ action }: { action: MemoryReviewActionPresentation }) {
  return (
    <div className="rounded-md border p-3">
      <div className="text-sm font-medium">{action.label}</div>
      <p className="mt-1 text-xs leading-relaxed text-muted-foreground">{action.consequence}</p>
    </div>
  );
}

export function ReviewDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [note, setNote] = useState("");

  const detailQuery = useQuery<MemoryReviewDetail>({
    queryKey: ["memory-review", id],
    queryFn: () => resourceClient.get(`/memory-reviews/${id}`).then((response) => response.data),
    enabled: Boolean(id),
  });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["memory-review", id] });
    queryClient.invalidateQueries({ queryKey: ["memory-reviews"] });
    queryClient.invalidateQueries({ queryKey: ["pending-reviews"] });
    queryClient.invalidateQueries({ queryKey: ["stats"] });
    queryClient.invalidateQueries({ queryKey: ["memories"] });
  };

  const useLatestMutation = useMutation({
    mutationFn: () =>
      resourceClient
        .post(`/memory-reviews/${id}/approve`, {
          expected_fingerprint: detailQuery.data?.decision_fingerprint,
          note: note.trim() || null,
        })
        .then((response) => response.data),
    onSuccess: (data: MemoryReviewDetail) => {
      queryClient.setQueryData(["memory-review", id], data);
      invalidate();
    },
    onError: invalidate,
  });

  const keepCurrentMutation = useMutation({
    mutationFn: () =>
      resourceClient
        .post(`/memory-reviews/${id}/reject`, {
          expected_fingerprint: detailQuery.data?.decision_fingerprint,
          note: note.trim(),
        })
        .then((response) => response.data),
    onSuccess: (data: MemoryReviewDetail) => {
      queryClient.setQueryData(["memory-review", id], data);
      invalidate();
    },
    onError: invalidate,
  });

  if (detailQuery.isLoading) {
    return (
      <div className="flex items-center justify-center gap-2 px-6 py-16 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" /> Loading review...
      </div>
    );
  }

  if (detailQuery.isError || !detailQuery.data) {
    return (
      <div className="flex flex-col items-center justify-center rounded-xl border bg-card px-6 py-12 text-center">
        <AlertCircle className="mb-3 size-6 text-destructive" />
        <h1 className="text-sm font-medium">Unable to load review</h1>
        <p className="mt-1 max-w-sm text-sm text-muted-foreground">
          {detailQuery.error instanceof Error ? detailQuery.error.message : "The review request failed."}
        </p>
        <Button className="mt-4" variant="outline" size="sm" onClick={() => detailQuery.refetch()}>
          Retry
        </Button>
      </div>
    );
  }

  const review = detailQuery.data;
  const pending = review.status === "pending" && !review.is_stale;
  const expired = review.status === "stale" || review.is_stale;
  const resolved = review.status === "approved" || review.status === "rejected";
  const approveAction = review.presentation.actions.find((action) => action.decision === "approve");
  const rejectAction = review.presentation.actions.find((action) => action.decision === "reject");
  const resolvedAction = review.presentation.actions.find(
    (action) => action.decision === (review.status === "approved" ? "approve" : "reject"),
  );
  const mutationPending = useLatestMutation.isPending || keepCurrentMutation.isPending;
  const decisionError = extractError(useLatestMutation.error) ?? extractError(keepCurrentMutation.error);

  return (
    <div className="space-y-6">
      <Button variant="ghost" size="sm" onClick={() => navigate("/review")} className="-ml-2">
        <ArrowLeft className="mr-1 size-4" /> Back to review queue
      </Button>

      <div>
        <div className="flex flex-wrap items-center gap-2">
          <ShieldCheck className="size-5 text-amber-500" />
          <h1 className="text-xl font-semibold tracking-tight">Memory decision</h1>
          {review.source_name && <Badge variant="secondary">{review.source_name}</Badge>}
        </div>
        <h2 className="mt-4 text-lg font-semibold">{review.presentation.summary}</h2>
        <p className="mt-1 max-w-3xl text-sm leading-relaxed text-muted-foreground">
          {review.presentation.why_human}
        </p>
      </div>

      {expired && (
        <div className="rounded-lg border border-amber-300 bg-amber-50 p-4 text-sm text-amber-950">
          <div className="font-medium">This proposal expired.</div>
          <p className="mt-1 text-amber-900/80">
            The underlying memory changed before a decision was applied. This record remains in
            audit history; a new Review will be created from current source state if a decision is
            still needed.
          </p>
        </div>
      )}

      {resolved && (
        <div className="flex items-start gap-3 rounded-lg border border-emerald-300 bg-emerald-50 p-4 text-sm text-emerald-950">
          <CheckCircle2 className="mt-0.5 size-5" />
          <div>
            <div className="font-medium">
              {resolvedAction ? `${resolvedAction.label} recorded` : "Decision recorded"}
            </div>
            {review.review_note && <p className="mt-1 text-emerald-900/80">{review.review_note}</p>}
          </div>
        </div>
      )}

      <div className="grid gap-4 lg:grid-cols-2">
        <StateCard
          label={review.presentation.current_label}
          memory={review.incumbent}
          emptyCopy="The current memory snapshot is unavailable."
        />
        <StateCard
          label={review.presentation.proposed_label}
          memory={review.challenger}
          emptyCopy={review.presentation.proposed_empty_text}
        />
      </div>

      {pending && approveAction && rejectAction && (
        <Card className="border-2">
          <CardHeader>
            <CardTitle className="text-base">Choose what MemForge should do</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 md:grid-cols-2">
              <ActionExplanation action={approveAction} />
              <ActionExplanation action={rejectAction} />
            </div>
            <div>
              <label htmlFor="review-note" className="text-sm font-medium">
                Decision note
              </label>
              <textarea
                id="review-note"
                value={note}
                onChange={(event) => setNote(event.target.value)}
                rows={3}
                placeholder={`Required for ${rejectAction.label}; optional for ${approveAction.label}.`}
                className="mt-2 w-full rounded-md border bg-background p-3 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </div>
            {decisionError && (
              <div className="rounded-md border border-destructive/40 bg-destructive/5 p-3 text-sm text-destructive">
                {decisionError}
              </div>
            )}
            <div className="flex flex-wrap justify-end gap-2">
              <Button
                variant="outline"
                disabled={mutationPending || (rejectAction.requires_note && note.trim().length === 0)}
                onClick={() => keepCurrentMutation.mutate()}
              >
                {rejectAction.label}
              </Button>
              <Button disabled={mutationPending} onClick={() => useLatestMutation.mutate()}>
                {approveAction.label}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      <details className="group rounded-lg border bg-muted/10 p-4">
        <summary className="flex cursor-pointer list-none items-center justify-between text-sm font-medium">
          View technical details
          <ChevronDown className="size-4 transition-transform group-open:rotate-180" />
        </summary>
        <dl className="mt-4 grid gap-3 text-xs text-muted-foreground md:grid-cols-2">
          <div><dt className="font-medium text-foreground">Review ID</dt><dd className="font-mono">{review.id}</dd></div>
          <div><dt className="font-medium text-foreground">Review mechanism</dt><dd>{review.review_origin}</dd></div>
          <div><dt className="font-medium text-foreground">Current memory ID</dt><dd className="font-mono">{review.incumbent_memory_id}</dd></div>
          <div><dt className="font-medium text-foreground">Proposed memory ID</dt><dd className="font-mono">{review.challenger_memory_id ?? "none"}</dd></div>
          <div><dt className="font-medium text-foreground">Opened</dt><dd title={formatDateTime(review.created_at)}>{timeAgo(review.created_at)}</dd></div>
          <div><dt className="font-medium text-foreground">Status</dt><dd>{review.status}</dd></div>
          {review.presentation.technical_reason && (
            <div className="md:col-span-2"><dt className="font-medium text-foreground">Planner reason</dt><dd className="mt-1 font-mono break-words">{review.presentation.technical_reason}</dd></div>
          )}
        </dl>
      </details>
    </div>
  );
}
