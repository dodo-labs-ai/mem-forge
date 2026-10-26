import { CheckCircle2, Clock } from "lucide-react";
import { formatDateTime } from "@/lib/format";
import { Notice } from "@/patterns";
import { decidedBanner, EXPIRED_EXPLANATION } from "./model/reviewDecision";
import type { ReviewDetail } from "./model/types";

/** Why an expired proposal can no longer be applied. */
export function ExpiredBanner() {
  return (
    <Notice tone="warn" icon={Clock} title="This proposal expired.">
      <p>{EXPIRED_EXPLANATION}</p>
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
