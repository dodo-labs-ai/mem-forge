import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { useApi } from "@/api";
import { errorMessage } from "@/lib/errors";
import { pluralize } from "@/lib/format";
import { StatusBadge } from "@/patterns";
import { Button } from "@/ui/button";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/ui/dialog";
import { Label } from "@/ui/label";
import { Textarea } from "@/ui/textarea";
import { checkDecisions, reviewKeys, useApplyDecisions } from "./api";
import {
  batchItemText,
  manifestFor,
  outcomeCounts,
  outcomeDetail,
  presentOutcome,
} from "./model/batchDecisions";
import type { DecisionResult, ReviewAction, ReviewDecision, ReviewListItem } from "./model/types";

interface BatchDecisionDialogProps {
  decision: ReviewDecision;
  reviews: ReviewListItem[];
  onOpenChange: (open: boolean) => void;
  /** Runs once the decisions were sent, whatever each outcome was. */
  onApplied: () => void;
}

const NOTE_FIELD_ID = "batch-decision-note";
/** The line under a review whose decision got no result because its request failed. */
const NO_RESULT = "No result came back for this review. Check it in the queue.";

function actionFor(review: ReviewListItem, decision: ReviewDecision): ReviewAction | undefined {
  return review.presentation.actions.find((action) => action.decision === decision);
}

function OutcomeCounts({ results }: { results: DecisionResult[] }) {
  return (
    <div className="flex flex-wrap gap-2">
      {outcomeCounts(results).map((count) => (
        <StatusBadge key={count.label} tone={count.tone} className="rounded-md">
          {count.count} {count.label.toLowerCase()}
        </StatusBadge>
      ))}
    </div>
  );
}

/**
 * Confirms one decision for several reviews. MemForge first checks each
 * review against its current state, then applies the ones that are still
 * ready and reports why it skipped the others.
 */
export function BatchDecisionDialog({ decision, reviews, onOpenChange, onApplied }: BatchDecisionDialogProps) {
  const api = useApi();
  const apply = useApplyDecisions();
  const [note, setNote] = useState("");
  // The note the last check ran with; the check reruns when the user leaves the note field.
  const [checkedNote, setCheckedNote] = useState("");

  const first = reviews[0];
  const label = (first && actionFor(first, decision)?.label) ?? "Decide";
  const noteRequired = reviews.some((review) => actionFor(review, decision)?.requires_note);
  const canCheck = !noteRequired || checkedNote.trim() !== "";

  const check = useQuery({
    queryKey: reviewKeys.check(decision, checkedNote.trim(), reviews.map((review) => review.id)),
    queryFn: () => checkDecisions(api, manifestFor(reviews, decision, checkedNote)),
    enabled: canCheck && apply.isIdle,
    staleTime: 0,
    gcTime: 0,
  });

  const applied = apply.data?.results;
  const results = applied ?? (check.data && checkedNote === note ? check.data : undefined);
  const resultById = new Map(results?.map((result) => [result.review_id, result]));
  const readyCount = check.data?.filter((result) => result.outcome === "ready").length;
  const noteMissing = noteRequired && note.trim() === "";

  function close() {
    if (apply.isPending) return;
    if (applied) onApplied();
    onOpenChange(false);
  }

  return (
    <Dialog open onOpenChange={(open) => (open ? undefined : close())}>
      <DialogContent className="gap-0 p-0 sm:max-w-xl">
        <DialogHeader className="gap-1.5 p-5 pb-0">
          <DialogTitle>
            {applied
              ? `${pluralize(applied.filter((result) => result.outcome === "applied").length, "decision")} applied`
              : `${label} for ${pluralize(reviews.length, "review")}?`}
          </DialogTitle>
          <DialogDescription>
            {applied
              ? "Each review below shows what happened to it."
              : "MemForge checks each review against its current state. Only ready reviews are applied."}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-3 px-5 pt-4 pb-5">
          {results ? <OutcomeCounts results={results} /> : null}
          {!canCheck ? (
            <p className="text-sm text-muted-foreground">Add a note to check these reviews.</p>
          ) : check.isError && !applied ? (
            <p role="alert" className="text-sm text-tone-danger-foreground">
              Could not check these reviews. {errorMessage(check.error)}
            </p>
          ) : null}

          <ul aria-label="Reviews in this decision" className="max-h-72 divide-y overflow-y-auto border-y">
            {reviews.map((review) => {
              const result = resultById.get(review.id);
              const consequence = actionFor(review, decision)?.consequence ?? "";
              return (
                <li key={review.id} className="flex items-start gap-3 py-2.5">
                  <div className="min-w-0 flex-1 space-y-0.5">
                    <p className="text-sm font-medium text-foreground">{batchItemText(review, decision)}</p>
                    <p className="text-xs text-muted-foreground">
                      {result ? outcomeDetail(result, consequence, applied ? "done" : "check") : applied ? NO_RESULT : consequence}
                    </p>
                  </div>
                  {result ? (
                    <StatusBadge tone={presentOutcome(result.outcome).tone} className="rounded-md">
                      {presentOutcome(result.outcome).label}
                    </StatusBadge>
                  ) : null}
                </li>
              );
            })}
          </ul>

          {applied ? null : (
            <div className="space-y-1.5">
              <Label htmlFor={NOTE_FIELD_ID}>Note for all</Label>
              <Textarea
                id={NOTE_FIELD_ID}
                rows={2}
                value={note}
                onChange={(event) => setNote(event.target.value)}
                onBlur={() => setCheckedNote(note)}
                placeholder={noteRequired ? "Required. Saved with each decision." : "Optional. Saved with each decision."}
                aria-required={noteRequired}
              />
            </div>
          )}

          {apply.isError ? (
            <p role="alert" className="text-sm text-tone-danger-foreground">
              {errorMessage(apply.error)}
            </p>
          ) : apply.data?.error ? (
            <p role="alert" className="text-sm text-tone-danger-foreground">
              Could not send every decision. {errorMessage(apply.data.error)}
            </p>
          ) : null}
        </div>

        <DialogFooter className="m-0 rounded-b-xl border-t bg-surface-subtle px-5 py-3.5">
          {applied ? (
            <Button onClick={close}>Done</Button>
          ) : (
            <>
              <Button variant="outline" onClick={close} disabled={apply.isPending}>
                Cancel
              </Button>
              <Button
                disabled={noteMissing || apply.isPending || readyCount === 0}
                onClick={() => apply.mutate(manifestFor(reviews, decision, note))}
              >
                {readyCount !== undefined && checkedNote === note
                  ? `Apply ${pluralize(readyCount, "decision")}`
                  : "Apply decisions"}
              </Button>
            </>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
