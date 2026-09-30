import { CheckCircle2, Clock, RefreshCw } from "lucide-react";
import { errorMessage } from "@/lib/errors";
import { formatDateTime } from "@/lib/format";
import { cn } from "@/lib/cn";
import { Notice } from "@/patterns";
import { Button } from "@/ui/button";
import { decidedBanner, expiredExplanation } from "./model/reviewDecision";
import type { ReviewDetail } from "./model/types";

interface ExpiredBannerProps {
  review: ReviewDetail;
  onRecheck: () => void;
  rechecking: boolean;
  recheckError: unknown;
}

/** Why an expired proposal can no longer be applied, with a recheck when the server offers one. */
export function ExpiredBanner({ review, onRecheck, rechecking, recheckError }: ExpiredBannerProps) {
  const explanation = expiredExplanation(review);
  return (
    <Notice tone="warn" icon={Clock} title="This proposal expired.">
      <p>{explanation.text}</p>
      {explanation.canRecheck ? (
        <div className="flex flex-wrap items-center gap-2.5">
          <Button size="sm" variant="outline" onClick={onRecheck} disabled={rechecking || !review.can_decide}>
            <RefreshCw aria-hidden className={cn(rechecking && "animate-spin")} />
            Recheck current state
          </Button>
          <span className="text-xs text-muted-foreground">
            {review.can_decide
              ? "Creates a new review from the current state if a decision is still needed."
              : "Only a source manager can recheck it."}
          </span>
        </div>
      ) : null}
      {recheckError ? (
        <p role="alert" className="text-tone-danger-foreground">
          {errorMessage(recheckError)}
        </p>
      ) : null}
    </Notice>
  );
}

/** Who decided a review, when, and the note they left. */
export function DecidedBanner({ review }: { review: ReviewDetail }) {
  const banner = decidedBanner(review, formatDateTime);
  return (
    <Notice tone="ok" icon={CheckCircle2} title={banner.title}>
      {banner.byline ? <p>{banner.byline}</p> : null}
      {banner.note ? <p className="italic">“{banner.note}”</p> : null}
    </Notice>
  );
}
