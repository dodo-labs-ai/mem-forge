import { AlertCircle, Loader2, RefreshCw, ShieldCheck } from "lucide-react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import { useSourceTypeLabels } from "@/features/sources";
import { errorMessage } from "@/lib/errors";
import { REVIEW_PATH, reviewPath } from "@/lib/paths";
import { BackLink, EmptyState } from "@/patterns";
import { Button } from "@/ui/button";
import { AlsoAffected } from "./AlsoAffected";
import { useRecheckReview, useReviewDetail } from "./api";
import { DecisionPanel } from "./DecisionPanel";
import { MemoryStateCard } from "./MemoryStateCard";
import { decisionOptions, reviewState } from "./model/reviewDecision";
import { reviewSourceLabel } from "./model/reviewQueue";
import type { ReviewDetail } from "./model/types";
import { DecidedBanner, ExpiredBanner } from "./ReviewBanners";
import { queueSearchFrom, type ReviewLinkState } from "./reviewLinkState";
import { TechnicalDetails } from "./TechnicalDetails";

function BackToQueue({ search }: { search: string }) {
  return <BackLink label="Review queue" render={<Link to={{ pathname: REVIEW_PATH, search }} />} />;
}

function ReviewView({ review, queueSearch, onReload }: { review: ReviewDetail; queueSearch: string; onReload: () => void }) {
  const navigate = useNavigate();
  const typeLabels = useSourceTypeLabels();
  const recheck = useRecheckReview(review);
  const state = reviewState(review);
  const options = decisionOptions(review.presentation);
  const { presentation } = review;

  function recheckReview() {
    recheck.mutate(undefined, {
      onSuccess: (rechecked) =>
        navigate(reviewPath(rechecked.id), { state: { queueSearch } satisfies ReviewLinkState }),
    });
  }

  return (
    <div className="space-y-5">
      <div className="space-y-3">
        <BackToQueue search={queueSearch} />
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <ShieldCheck aria-hidden className="size-4 text-tone-warn" />
          <span className="font-medium text-muted-foreground">Memory decision</span>
          <span className="text-subtle-foreground">{reviewSourceLabel(review)}</span>
        </div>
        <h1 className="text-[22px] leading-snug font-semibold tracking-tight text-foreground">{presentation.summary}</h1>
        <p className="max-w-3xl text-sm leading-relaxed text-muted-foreground">
          {presentation.why_human}
          {state === "waiting" ? " The current memory stays active and searchable until you decide." : null}
        </p>
      </div>

      {state === "expired" ? (
        <ExpiredBanner review={review} onRecheck={recheckReview} rechecking={recheck.isPending} recheckError={recheck.error} />
      ) : null}
      {state === "applied" || state === "kept" ? <DecidedBanner review={review} /> : null}

      <div className="grid gap-4 lg:grid-cols-2">
        <MemoryStateCard
          label={presentation.current_label}
          memory={review.incumbent}
          emptyText="The current memory is unavailable."
          typeLabels={typeLabels}
        />
        <MemoryStateCard
          label={presentation.proposed_label}
          memory={review.challenger}
          emptyText={presentation.proposed_empty_text}
          typeLabels={typeLabels}
          emphasized={state === "waiting"}
        />
      </div>

      {state === "waiting" && options ? <DecisionPanel review={review} options={options} onReload={onReload} /> : null}

      <AlsoAffected memories={review.related_challengers ?? []} />

      <TechnicalDetails review={review} />
    </div>
  );
}

export function ReviewDetailPage() {
  const { reviewId = "" } = useParams();
  const queueSearch = queueSearchFrom(useLocation().state);
  const detail = useReviewDetail(reviewId);

  if (detail.isPending) {
    return (
      <div className="flex items-center justify-center gap-2 py-16 text-sm text-muted-foreground" role="status">
        <Loader2 aria-hidden className="size-4 animate-spin" />
        Loading review…
      </div>
    );
  }

  if (detail.isError) {
    return (
      <div className="space-y-4">
        <BackToQueue search={queueSearch} />
        <div role="alert" className="rounded-lg border bg-surface">
          <EmptyState
            icon={AlertCircle}
            title="Unable to load review"
            description={`${errorMessage(detail.error)} The review may be gone, or it involves a source you cannot access.`}
            action={
              <Button variant="outline" onClick={() => void detail.refetch()}>
                <RefreshCw />
                Retry
              </Button>
            }
          />
        </div>
      </div>
    );
  }

  return <ReviewView review={detail.data} queueSearch={queueSearch} onReload={() => void detail.refetch()} />;
}
