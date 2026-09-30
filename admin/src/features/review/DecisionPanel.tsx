import { Lock } from "lucide-react";
import { useState } from "react";
import { Button } from "@/ui/button";
import { Label } from "@/ui/label";
import { Textarea } from "@/ui/textarea";
import { useDecideReview } from "./api";
import {
  canSubmitDecision,
  decisionFailure,
  noteHint,
  type DecisionOptions,
} from "./model/reviewDecision";
import type { ReviewAction, ReviewDetail } from "./model/types";

const NOTE_FIELD_ID = "review-decision-note";
const NOTE_HINT_ID = "review-decision-note-hint";

interface OptionCardProps {
  action: ReviewAction;
  primary: boolean;
  disabled: boolean;
  onChoose: () => void;
}

function OptionCard({ action, primary, disabled, onChoose }: OptionCardProps) {
  return (
    <div className="flex flex-1 flex-col gap-2.5 rounded-lg border p-4">
      <h3 className="text-sm font-semibold text-foreground">{action.label}</h3>
      <p className="flex-1 text-sm leading-relaxed text-muted-foreground">{action.consequence}</p>
      <Button
        className="self-start"
        variant={primary ? "default" : "outline"}
        disabled={disabled}
        aria-describedby={action.requires_note ? NOTE_HINT_ID : undefined}
        onClick={onChoose}
      >
        {action.label}
      </Button>
    </div>
  );
}

interface DecisionPanelProps {
  review: ReviewDetail;
  options: DecisionOptions;
  /** Reloads the review after the server reported that it changed. */
  onReload: () => void;
}

/**
 * The two choices of a waiting review, each with its consequence next to its
 * own button. A failed decision keeps the comparison and the typed note on
 * screen and says what to do next.
 */
export function DecisionPanel({ review, options, onReload }: DecisionPanelProps) {
  const decide = useDecideReview(review);
  const [note, setNote] = useState("");
  const failure = decide.isError ? decisionFailure(decide.error) : null;
  const viewOnly = !review.can_decide;

  function choose(action: ReviewAction) {
    decide.mutate({ decision: action.decision, note });
  }

  return (
    <section aria-labelledby="review-decision-title" className="space-y-4 rounded-xl border bg-surface p-5">
      <h2 id="review-decision-title" className="text-base font-semibold text-foreground">
        Choose what MemForge should do
      </h2>

      {viewOnly ? (
        <div role="note" className="flex items-start gap-2.5 rounded-lg bg-muted px-3.5 py-3 text-sm text-subtle-foreground">
          <Lock aria-hidden className="mt-0.5 size-4 shrink-0" />
          <p>
            View only. Only the source owner or a workspace admin can decide this review, because the decision changes
            memories of a source you do not manage.
          </p>
        </div>
      ) : null}

      <div className="flex flex-col gap-3 sm:flex-row">
        <OptionCard
          action={options.useLatest}
          primary
          disabled={viewOnly || decide.isPending}
          onChoose={() => choose(options.useLatest)}
        />
        <OptionCard
          action={options.keepCurrent}
          primary={false}
          disabled={viewOnly || decide.isPending || !canSubmitDecision(options.keepCurrent, note)}
          onChoose={() => choose(options.keepCurrent)}
        />
      </div>

      {viewOnly ? null : (
        <div className="space-y-1.5">
          <Label htmlFor={NOTE_FIELD_ID}>Decision note</Label>
          <Textarea
            id={NOTE_FIELD_ID}
            rows={3}
            value={note}
            onChange={(event) => setNote(event.target.value)}
            disabled={decide.isPending}
            aria-describedby={NOTE_HINT_ID}
            placeholder="Why you chose this. Saved with the decision in audit history."
          />
          <p id={NOTE_HINT_ID} className="text-xs text-muted-foreground">
            {noteHint(options)}
          </p>
        </div>
      )}

      {failure ? (
        <div role="alert" className="space-y-2 rounded-lg border border-tone-danger/30 bg-tone-danger-soft p-3 text-sm">
          <p className="font-medium text-tone-danger-foreground">{failure.message}</p>
          {failure.hint ? <p className="text-tone-danger-foreground/90">{failure.hint}</p> : null}
          {failure.reloadable ? (
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                decide.reset();
                onReload();
              }}
            >
              Reload review
            </Button>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}
