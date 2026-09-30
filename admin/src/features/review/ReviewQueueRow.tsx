import { ArrowRight, Check, Lock } from "lucide-react";
import { useId } from "react";
import { Link } from "react-router-dom";
import { SourceIcon } from "@/features/sources";
import { reviewPath } from "@/lib/paths";
import { StatusBadge } from "@/patterns";
import { Checkbox } from "@/ui/checkbox";
import { ChangeExcerpt } from "./ChangeExcerpt";
import { buildChangeCue } from "./model/changeCue";
import { decisionLabelTone, recordedAction, reviewState } from "./model/reviewDecision";
import { reviewOrigin, reviewSourceLabel, rowTimeLabel } from "./model/reviewQueue";
import type { ReviewListItem } from "./model/types";
import type { ReviewLinkState } from "./reviewLinkState";

/** Why a row cannot be picked for a bulk decision; the review page says the same at more length. */
const VIEW_ONLY_REASON = "View only: only the source owner or a workspace admin can decide it";

interface ReviewQueueRowProps {
  review: ReviewListItem;
  /** The queue view the row is listed in, so the review page links back to it. */
  queueSearch: string;
  /** Set when the queue offers bulk decisions; a review the caller cannot decide shows why it cannot be picked. */
  selection?: { selected: boolean; onSelectedChange: (selected: boolean) => void };
}

function Comparison({ review }: { review: ReviewListItem }) {
  const { presentation } = review;
  const cue = buildChangeCue(review.incumbent?.content, review.challenger?.content);
  return (
    <div className="grid grid-cols-2 gap-6">
      <div className="min-w-0 space-y-1">
        <p className="text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">{presentation.current_label}</p>
        <p className="truncate text-sm">
          <ChangeExcerpt excerpt={cue.current} emptyText="The current memory is unavailable." />
        </p>
      </div>
      <div className="min-w-0 space-y-1">
        <p className="text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">{presentation.proposed_label}</p>
        <p className="truncate text-sm">
          <ChangeExcerpt excerpt={cue.proposed} emptyText={presentation.proposed_empty_text} />
        </p>
      </div>
    </div>
  );
}

function RecordedDecision({ review }: { review: ReviewListItem }) {
  const action = recordedAction(review);
  return (
    <div className="flex items-center gap-2">
      <StatusBadge tone="ok" className="rounded-md">
        <Check aria-hidden />
        {action?.label ?? "Decision recorded"}
      </StatusBadge>
      {review.reviewer ? <span className="text-xs text-muted-foreground">by {review.reviewer}</span> : null}
    </div>
  );
}

/** One review in the queue: where it came from, the question, and what would change. */
export function ReviewQueueRow({ review, queueSearch, selection }: ReviewQueueRowProps) {
  const state = reviewState(review);
  const decided = state === "applied" || state === "kept";
  const time = rowTimeLabel(review);
  const origin = reviewOrigin(review);
  const viewOnlyId = useId();
  const viewOnly = selection !== undefined && !review.can_decide;
  return (
    <li className="relative flex items-start gap-3.5 border-b px-4.5 py-4 last:border-b-0 hover:bg-surface-subtle has-[a:focus-visible]:bg-surface-subtle">
      {selection ? (
        <Checkbox
          className="relative z-10 mt-0.5"
          checked={selection.selected}
          disabled={viewOnly}
          onCheckedChange={(checked) => selection.onSelectedChange(checked === true)}
          aria-label={`Select review: ${review.incumbent?.content ?? review.presentation.summary}`}
          aria-describedby={viewOnly ? viewOnlyId : undefined}
        />
      ) : null}
      <div className="min-w-0 flex-1 space-y-2">
        <div className="flex flex-wrap items-center gap-2.5 text-xs">
          <span className="inline-flex items-center gap-1.5 font-medium text-subtle-foreground">
            {origin ? <SourceIcon type={origin.sourceType} client={origin.client} className="size-3.5" /> : null}
            {reviewSourceLabel(review)}
          </span>
          <StatusBadge tone={decisionLabelTone(review.presentation.decision_label)}>
            {review.presentation.decision_label}
          </StatusBadge>
          {time ? <span className="text-muted-foreground">{time}</span> : null}
          {viewOnly ? (
            <span id={viewOnlyId} className="inline-flex items-center gap-1 text-muted-foreground">
              <Lock aria-hidden className="size-3" />
              {VIEW_ONLY_REASON}
            </span>
          ) : null}
        </div>
        <Link
          to={reviewPath(review.id)}
          state={{ queueSearch } satisfies ReviewLinkState}
          className="block font-semibold text-foreground outline-none after:absolute after:inset-0 focus-visible:underline"
        >
          {review.presentation.summary}
        </Link>
        {decided ? <RecordedDecision review={review} /> : <Comparison review={review} />}
      </div>
      <ArrowRight aria-hidden className="mt-7 size-4 shrink-0 text-muted-foreground" />
    </li>
  );
}
