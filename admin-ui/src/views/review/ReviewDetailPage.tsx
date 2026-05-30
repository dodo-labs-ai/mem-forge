import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate, useParams } from "react-router-dom";
import {
  AlertCircle,
  ArrowLeft,
  CheckCircle2,
  ExternalLink,
  Loader2,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  XCircle,
} from "lucide-react";
import client from "@/api/client";
import type {
  MemorySource,
  MemoryReviewDetail,
  MemoryReviewMemorySummary,
} from "@/api/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import { formatDateTime, timeAgo } from "@/utils/date";
import {
  type DiffSegment,
  type DiffSide,
  diffWords,
  segmentsForSide,
} from "@/utils/diff";

const DECISION_STATUS_LABEL: Record<string, string> = {
  pending: "Pending decision",
  approved: "Approved",
  rejected: "Rejected",
  stale: "Stale - needs refresh",
};

interface AttributeRowProps {
  label: string;
  current: string;
  proposed: string;
  changed: boolean;
}

function AttributeRow({ label, current, proposed, changed }: AttributeRowProps) {
  return (
    <div
      className={cn(
        "grid grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)_minmax(0,1fr)_auto] items-center gap-2 border-b py-2 text-sm last:border-b-0",
        !changed && "text-muted-foreground",
      )}
    >
      <div className="text-muted-foreground">{label}</div>
      <div className={cn(changed ? "text-foreground" : "text-muted-foreground")}>{current}</div>
      <div className={cn(changed ? "font-medium text-foreground" : "text-muted-foreground")}>
        {proposed}
      </div>
      <div className="justify-self-end">
        {changed ? (
          <Badge variant="secondary" className="text-[10px]">
            Changed
          </Badge>
        ) : (
          <span className="text-muted-foreground">-</span>
        )}
      </div>
    </div>
  );
}

function HighlightedText({
  segments,
  side,
}: {
  segments: DiffSegment[];
  side: DiffSide;
}) {
  const visibleSegments = segmentsForSide(segments, side);
  return (
    <>
      {visibleSegments.map((segment, index) => {
        if (segment.op === "equal") {
          return <span key={index}>{segment.text}</span>;
        }
        const isAdded = segment.op === "added";
        return (
          <span
            key={index}
            className={cn(
              "rounded-sm px-0.5",
              isAdded
                ? "bg-emerald-100 text-emerald-900 dark:bg-emerald-900/40 dark:text-emerald-100"
                : "bg-rose-100 text-rose-900 line-through dark:bg-rose-900/40 dark:text-rose-100",
            )}
          >
            {segment.text}
          </span>
        );
      })}
    </>
  );
}

function SplitDiffBlock({
  before,
  after,
  beforeLabel = "Current",
  afterLabel = "Proposed",
}: {
  before: string;
  after: string;
  beforeLabel?: string;
  afterLabel?: string;
}) {
  const segments = diffWords(before, after);
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <div className="rounded-md border bg-muted/30 p-3">
        <div className="mb-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          {beforeLabel}
        </div>
        <p className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-foreground">
          <HighlightedText segments={segments} side="before" />
        </p>
      </div>
      <div className="rounded-md border bg-muted/30 p-3">
        <div className="mb-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          {afterLabel}
        </div>
        <p className="whitespace-pre-wrap font-mono text-sm leading-relaxed text-foreground">
          <HighlightedText segments={segments} side="after" />
        </p>
      </div>
    </div>
  );
}

function RelativeTime({ value }: { value: string | null | undefined }) {
  return <time title={formatDateTime(value)}>{timeAgo(value)}</time>;
}

function statusLabel(status: string | null | undefined): string {
  if (!status) return "Unknown";
  return status.replace(/_/g, " ");
}

function sourceCountLabel(count: number): string {
  return `${count} source ${count === 1 ? "record" : "records"}`;
}

const AGENT_SESSION_SOURCE_TYPE = "agent_session";

function hasAgentSessionSource(sources: MemorySource[] | null | undefined): boolean {
  return Array.isArray(sources)
    ? sources.some((source) => source.source_type === AGENT_SESSION_SOURCE_TYPE)
    : false;
}

function SourceEvidenceList({ sources }: { sources: MemorySource[] }) {
  if (sources.length === 0) {
    return (
      <div className="rounded-md border border-dashed bg-muted/20 p-3 text-sm text-muted-foreground">
        No source records attached.
      </div>
    );
  }

  return (
    <div className="space-y-2">
      {sources.map((source, idx) => (
        <div key={`${source.doc_id}-${idx}`} className="rounded-md border p-3">
          <div className="flex flex-wrap items-center gap-2 text-sm">
            <Badge variant="secondary" className="text-[11px]">
              {source.source_type}
            </Badge>
            <span className="min-w-0 flex-1 truncate font-medium">
              {source.doc_title ?? source.doc_id}
            </span>
            <span className="text-xs text-muted-foreground">
              <RelativeTime value={source.added_at} />
            </span>
            {source.source_url && (
              <a
                href={source.source_url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
              >
                Source <ExternalLink className="size-3.5" />
              </a>
            )}
            {(source.pdf_uri || source.file_uri) && (
              <a
                href={(source.pdf_uri ?? source.file_uri) || undefined}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground"
              >
                File <ExternalLink className="size-3.5" />
              </a>
            )}
          </div>
          {source.excerpt && (
            <p className="mt-2 border-l-2 border-border pl-3 text-sm text-muted-foreground">
              {source.excerpt}
            </p>
          )}
        </div>
      ))}
    </div>
  );
}

function EvidencePanel({
  incumbent,
  challenger,
}: {
  incumbent: MemoryReviewMemorySummary | null;
  challenger: MemoryReviewMemorySummary | null;
}) {
  const currentSources = incumbent?.sources ?? [];
  const proposedSources = challenger?.sources ?? [];
  const hasAnySources = currentSources.length > 0 || proposedSources.length > 0;

  if (!hasAnySources) {
    return (
      <div className="rounded-md border border-dashed bg-muted/20 p-3 text-sm text-muted-foreground">
        No source records are attached to either memory in this review snapshot.
      </div>
    );
  }

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <div>
        <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
          <span className="font-medium uppercase tracking-wide">Current evidence</span>
          <span>{sourceCountLabel(currentSources.length)}</span>
        </div>
        <SourceEvidenceList sources={currentSources} />
      </div>
      <div>
        <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
          <span className="font-medium uppercase tracking-wide">Proposed evidence</span>
          <span>{sourceCountLabel(proposedSources.length)}</span>
        </div>
        <SourceEvidenceList sources={proposedSources} />
      </div>
    </div>
  );
}

function RelatedChallengersPanel({
  challengers,
}: {
  challengers: MemoryReviewMemorySummary[];
}) {
  if (challengers.length === 0) return null;

  return (
    <div className="rounded-md border bg-muted/20 p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
          Related challengers
        </div>
        <Badge variant="secondary" className="text-[10px]">
          {challengers.length}
        </Badge>
      </div>
      <div className="space-y-2">
        {challengers.map((memory) => (
          <div key={memory.id} className="rounded-md border bg-background p-3">
            <div className="mb-1 flex flex-wrap items-center gap-2">
              <Badge variant="outline" className="text-[10px] capitalize">
                {statusLabel(memory.status)}
              </Badge>
              <span className="font-mono text-xs text-muted-foreground">{memory.id}</span>
            </div>
            <p className="text-sm leading-relaxed text-foreground">{memory.content}</p>
            {memory.sources.length > 0 && (
              <p className="mt-2 text-xs text-muted-foreground">
                {sourceCountLabel(memory.sources.length)}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function diffField(label: string, current: unknown, proposed: unknown): AttributeRowProps {
  const left = formatField(current);
  const right = formatField(proposed);
  return { label, current: left, proposed: right, changed: left !== right };
}

function formatField(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (Array.isArray(value)) return value.length ? value.join(", ") : "—";
  if (typeof value === "number") return value.toString();
  return String(value);
}

interface OutcomeStateProps {
  title: string;
  memory: MemoryReviewMemorySummary | null;
  description: string;
}

function OutcomeState({ title, memory, description }: OutcomeStateProps) {
  return (
    <div className="rounded-md border bg-background p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="text-sm font-medium">{title}</div>
        <Badge variant="outline" className="capitalize">
          {statusLabel(memory?.status)}
        </Badge>
      </div>
      {memory?.id && <div className="mt-1 font-mono text-xs text-muted-foreground">{memory.id}</div>}
      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">{description}</p>
    </div>
  );
}

function ResolutionBanner({
  review,
  incumbent,
  challenger,
  onBackToQueue,
}: {
  review: MemoryReviewDetail;
  incumbent: MemoryReviewMemorySummary | null;
  challenger: MemoryReviewMemorySummary | null;
  onBackToQueue: () => void;
}) {
  const approved = review.status === "approved";
  const headline = approved ? "Review approved" : "Review rejected";
  const outcomeCopy = approved
    ? "The proposed memory is now active and available for retrieval. The previous memory has been superseded."
    : "The proposed memory was retired. The current memory remains active and available for retrieval.";
  const Icon = approved ? CheckCircle2 : XCircle;

  return (
    <Card
      className={cn(
        "border-2",
        approved
          ? "border-emerald-300 bg-emerald-50/60 dark:border-emerald-900/70 dark:bg-emerald-950/20"
          : "border-destructive/40 bg-destructive/5",
      )}
    >
      <CardContent className="p-6">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
          <div className="flex items-start gap-4">
            <div
              className={cn(
                "flex size-10 shrink-0 items-center justify-center rounded-full",
                approved
                  ? "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200"
                  : "bg-destructive/10 text-destructive",
              )}
            >
              <Icon className="size-5" />
            </div>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-lg font-semibold tracking-tight">{headline}</h2>
                <Badge
                  variant="outline"
                  className={cn(
                    "text-[11px]",
                    approved
                      ? "border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-200"
                      : "border-destructive/40 bg-destructive/5 text-destructive",
                  )}
                >
                  {approved ? "Approved" : "Rejected"}
                </Badge>
              </div>
              <p className="mt-1 max-w-2xl text-sm text-muted-foreground">{outcomeCopy}</p>
              <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-muted-foreground">
                <div className="flex items-center gap-1">
                  <dt className="font-medium uppercase tracking-wide">Resolved</dt>
                  <dd>
                    {review.resolved_at ? <RelativeTime value={review.resolved_at} /> : "—"}
                  </dd>
                </div>
                {review.reviewer && (
                  <div className="flex items-center gap-1">
                    <dt className="font-medium uppercase tracking-wide">Reviewer</dt>
                    <dd>{review.reviewer}</dd>
                  </div>
                )}
              </dl>
              {review.review_note && (
                <div className="mt-3 max-w-2xl rounded-md border bg-background/70 p-3 text-sm">
                  <div className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                    Reviewer note
                  </div>
                  <p className="mt-1 leading-relaxed">{review.review_note}</p>
                </div>
              )}
            </div>
          </div>
          <Button onClick={onBackToQueue} className="gap-2 self-start lg:self-auto">
            <ArrowLeft className="size-4" />
            Back to queue
          </Button>
        </div>

        <div className="mt-6 grid gap-3 sm:grid-cols-2">
          <OutcomeState
            title="Proposed memory"
            memory={challenger}
            description={
              approved
                ? "Promoted from pending review and now active in retrieval."
                : "Rejected by this review and removed from active retrieval."
            }
          />
          <OutcomeState
            title="Previous memory"
            memory={incumbent}
            description={
              approved
                ? "Superseded by the approved replacement."
                : "Kept as the active memory because the replacement was rejected."
            }
          />
        </div>
      </CardContent>
    </Card>
  );
}

export function ReviewDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [note, setNote] = useState("");
  const [confirmedReview, setConfirmedReview] = useState(false);

  const detailQuery = useQuery<MemoryReviewDetail>({
    queryKey: ["memory-review", id],
    queryFn: () => client.get(`/api/memory-reviews/${id}`).then((response) => response.data),
    enabled: Boolean(id),
  });

  const invalidate = () => {
    queryClient.invalidateQueries({ queryKey: ["memory-review", id] });
    queryClient.invalidateQueries({ queryKey: ["memory-reviews"] });
    queryClient.invalidateQueries({ queryKey: ["pending-reviews"] });
    queryClient.invalidateQueries({ queryKey: ["stats"] });
    queryClient.invalidateQueries({ queryKey: ["memories"] });
  };

  const approveMutation = useMutation({
    mutationFn: () =>
      client.post(`/api/memory-reviews/${id}/approve`, {
        note: note.trim() || null,
      }).then((response) => response.data),
    onSuccess: (data: MemoryReviewDetail) => {
      queryClient.setQueryData(["memory-review", id], data);
      invalidate();
    },
  });

  const rejectMutation = useMutation({
    mutationFn: () =>
      client.post(`/api/memory-reviews/${id}/reject`, {
        note: note.trim(),
      }).then((response) => response.data),
    onSuccess: (data: MemoryReviewDetail) => {
      queryClient.setQueryData(["memory-review", id], data);
      invalidate();
    },
  });

  const refreshMutation = useMutation({
    mutationFn: () => client.post(`/api/memory-reviews/${id}/refresh`),
    onSuccess: invalidate,
  });

  if (detailQuery.isLoading) {
    return (
      <div className="flex items-center justify-center gap-2 px-6 py-16 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" />
        Loading review...
      </div>
    );
  }

  if (detailQuery.isError || !detailQuery.data) {
    return (
      <div className="flex flex-col items-center justify-center rounded-xl border bg-card px-6 py-12 text-center">
        <AlertCircle className="mb-3 size-6 text-destructive" />
        <h1 className="text-sm font-medium">Unable to load review</h1>
        <p className="mt-1 max-w-sm text-sm text-muted-foreground">
          {detailQuery.error instanceof Error
            ? detailQuery.error.message
            : "The review request failed."}
        </p>
        <Button className="mt-4" variant="outline" size="sm" onClick={() => detailQuery.refetch()}>
          Retry
        </Button>
      </div>
    );
  }

  const review = detailQuery.data;
  const incumbent = review.incumbent;
  const challenger = review.challenger;
  const relatedChallengers = review.related_challengers ?? [];
  const decisionPending = review.status === "pending" && !review.is_stale;
  const decisionResolved = review.status === "approved" || review.status === "rejected";
  const isStale = review.status === "stale" || review.is_stale;

  const attributeRows: AttributeRowProps[] = [];
  if (incumbent && challenger) {
    attributeRows.push(diffField("Confidence", incumbent.confidence, challenger.confidence));
    for (const row of [
      diffField("Tags", incumbent.tags, challenger.tags),
      diffField("Entities", incumbent.entity_refs, challenger.entity_refs),
      diffField("Memory type", incumbent.memory_type, challenger.memory_type),
    ]) {
      if (row.changed) attributeRows.push(row);
    }
  }

  const approveDisabled =
    !decisionPending ||
    !confirmedReview ||
    approveMutation.isPending ||
    rejectMutation.isPending;
  const rejectDisabled =
    !decisionPending ||
    note.trim().length === 0 ||
    approveMutation.isPending ||
    rejectMutation.isPending;

  const headingTitle = decisionResolved
    ? review.status === "approved"
      ? "Approved review"
      : "Rejected review"
    : "Review proposed replacement";
  const diffCardTitle = decisionResolved ? "What changed" : "What's changing";
  const beforeDiffLabel = review.status === "approved" ? "Previous" : "Current";
  const afterDiffLabel =
    review.status === "approved"
      ? "Approved"
      : review.status === "rejected"
        ? "Rejected proposal"
        : "Proposed";

  const showAgentSessionNotice =
    hasAgentSessionSource(challenger?.sources) &&
    !hasAgentSessionSource(incumbent?.sources);

  return (
    <div className="space-y-6">
      <Button variant="ghost" size="sm" onClick={() => navigate("/review")} className="-ml-2">
        <ArrowLeft className="mr-1 size-4" /> Back to review queue
      </Button>

      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            {review.status === "approved" && (
              <CheckCircle2 className="size-5 text-emerald-600 dark:text-emerald-400" />
            )}
            {review.status === "rejected" && <XCircle className="size-5 text-destructive" />}
            <h1 className="text-xl font-semibold tracking-tight">{headingTitle}</h1>
          </div>
          <p className="mt-1 flex flex-wrap items-center gap-x-2 gap-y-1 text-sm text-muted-foreground">
            {DECISION_STATUS_LABEL[review.status] ?? review.status}
            <span aria-hidden="true">·</span>
            <span className="font-mono text-xs">{review.id}</span>
            <span aria-hidden="true">·</span>
            <span className="capitalize">{review.kind}</span>
            {review.created_at && (
              <>
                <span aria-hidden="true">·</span>
                <span>
                  opened <RelativeTime value={review.created_at} />
                </span>
              </>
            )}
            {decisionResolved && review.resolved_at && (
              <>
                <span aria-hidden="true">·</span>
                <span>
                  resolved <RelativeTime value={review.resolved_at} />
                </span>
              </>
            )}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {isStale && !decisionResolved && (
            <Badge variant="outline" className="border-amber-300 bg-amber-50 text-[11px] text-amber-800">
              Stale
            </Badge>
          )}
        </div>
      </div>

      {decisionResolved && (
        <ResolutionBanner
          review={review}
          incumbent={incumbent}
          challenger={challenger}
          onBackToQueue={() => navigate("/review")}
        />
      )}

      <div
        className={cn(
          "grid gap-6 xl:items-start",
          !decisionResolved && "xl:grid-cols-[minmax(0,1fr)_360px]",
        )}
      >
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">{diffCardTitle}</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {showAgentSessionNotice && (
                <div className="flex items-start gap-2 rounded-md border border-sky-200 bg-sky-50 p-3 text-sm text-sky-900 dark:border-sky-900/60 dark:bg-sky-950/30 dark:text-sky-100">
                  <Sparkles className="mt-0.5 size-4 shrink-0" />
                  <p className="leading-relaxed">
                    Generated agent-session summary. Compare evidence before replacing the current
                    memory.
                  </p>
                </div>
              )}
              {incumbent && challenger ? (
                <SplitDiffBlock
                  before={incumbent.content}
                  after={challenger.content}
                  beforeLabel={beforeDiffLabel}
                  afterLabel={afterDiffLabel}
                />
              ) : (
                <p className="text-sm text-muted-foreground">
                  Cannot render diff: one side of the review is unavailable.
                </p>
              )}

              {review.reason && (
                <div className="rounded-md border bg-muted/40 p-3 text-sm text-muted-foreground">
                  <span className="font-medium text-foreground">Reason: </span>
                  {review.reason}
                </div>
              )}

              <RelatedChallengersPanel challengers={relatedChallengers} />

              {attributeRows.length > 0 && (
                <div className="rounded-md border">
                  <div className="grid grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)_minmax(0,1fr)_auto] gap-2 border-b bg-muted/40 px-3 py-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                    <div>Attribute</div>
                    <div>{beforeDiffLabel}</div>
                    <div>{afterDiffLabel}</div>
                    <div className="justify-self-end">Delta</div>
                  </div>
                  <div className="px-3">
                    {attributeRows.map((row) => (
                      <AttributeRow key={row.label} {...row} />
                    ))}
                  </div>
                </div>
              )}

              <div>
                <div className="mb-2 text-[11px] font-medium uppercase tracking-wide text-muted-foreground">
                  Evidence
                </div>
                <EvidencePanel incumbent={incumbent} challenger={challenger} />
              </div>
            </CardContent>
          </Card>
        </div>

        {!decisionResolved && (
          <Card className="border-2 xl:sticky xl:top-20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-sm">
                <ShieldCheck className="size-4 text-amber-500" />
                Decision
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {isStale && (
                <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
                  <AlertCircle className="mt-0.5 size-4 shrink-0" />
                  <div className="flex-1">
                    <div className="font-medium">This review is stale.</div>
                    <p className="mt-1 text-xs text-amber-900/80">
                      One of the underlying memories changed after the review was filed. Refresh
                      to re-pin expectations against the current state, then decide again.
                    </p>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    className="shrink-0"
                    disabled={refreshMutation.isPending}
                    onClick={() => refreshMutation.mutate()}
                  >
                    <RefreshCw className="size-4" />
                    Refresh
                  </Button>
                </div>
              )}

              <label className="block text-sm font-medium" htmlFor="review-note">
                Note
              </label>
              <textarea
                id="review-note"
                value={note}
                onChange={(event) => setNote(event.target.value)}
                placeholder="Why are you approving or rejecting?"
                rows={4}
                disabled={!decisionPending}
                className="w-full rounded-md border bg-background p-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring disabled:cursor-not-allowed disabled:opacity-60"
              />
              <p className="-mt-2 text-xs text-muted-foreground">
                Required to reject; optional when approving.
              </p>

              <label className="flex items-start gap-2 text-sm text-muted-foreground">
                <input
                  type="checkbox"
                  checked={confirmedReview}
                  disabled={!decisionPending}
                  onChange={(event) => setConfirmedReview(event.target.checked)}
                  className="mt-0.5"
                />
                <span>I reviewed both memories and the proposed change is correct.</span>
              </label>
              {decisionPending && !confirmedReview && (
                <p className="text-xs text-muted-foreground">Confirm review to enable approval.</p>
              )}

              {(approveMutation.isError || rejectMutation.isError) && (
                <div className="rounded-md border border-destructive/40 bg-destructive/5 p-2 text-sm text-destructive">
                  {extractError(approveMutation.error) ?? extractError(rejectMutation.error)}
                </div>
              )}

              <div className="flex flex-wrap items-center justify-end gap-2">
                <Button
                  variant="outline"
                  disabled={rejectDisabled}
                  onClick={() => rejectMutation.mutate()}
                >
                  <XCircle className="size-4" />
                  Reject
                </Button>
                <Button
                  disabled={approveDisabled}
                  title={
                    approveDisabled && decisionPending ? "Confirm review to enable" : undefined
                  }
                  className={cn(approveDisabled && "disabled:opacity-45")}
                  onClick={() => approveMutation.mutate()}
                >
                  <CheckCircle2 className="size-4" />
                  Approve
                </Button>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}

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
  if (error instanceof Error) return error.message;
  return null;
}
