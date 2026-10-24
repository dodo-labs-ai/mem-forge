import { CheckIcon } from "lucide-react";
import { useRef, type KeyboardEvent } from "react";
import { cn } from "@/lib/cn";

export interface RailStep {
  key: string;
  label: string;
  /** Every required input of the step is given. */
  complete?: boolean;
  /** How many inputs of the step need correcting. */
  errorCount?: number;
  /** One quiet line under a completed step, such as the site it connects to; a long one is cut short, whole in its tooltip. */
  summary?: string;
}

interface StepRailProps {
  steps: readonly RailStep[];
  current: string;
  onSelect: (key: string) => void;
  "aria-label": string;
  className?: string;
}

function issueText(count: number): string {
  return count === 1 ? "1 issue" : `${count} issues`;
}

/** The step's name for assistive technology: its label, then its issues or that it is complete. */
function stepName(step: RailStep, errors: number): string {
  if (errors > 0) return `${step.label}, ${issueText(errors)}`;
  return step.complete ? `${step.label}, complete` : step.label;
}

/**
 * The steps of a multi-step form, any of which can be opened. Each step shows
 * whether it is complete and how many of its inputs need correcting. Only the
 * current step is in the tab order; the arrow keys, Home and End move between
 * steps, and Enter or Space opens the focused one.
 */
export function StepRail({ steps, current, onSelect, "aria-label": label, className }: StepRailProps) {
  const buttons = useRef<Array<HTMLButtonElement | null>>([]);

  function onKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    const last = steps.length - 1;
    const target =
      event.key === "ArrowDown" || event.key === "ArrowRight"
        ? Math.min(index + 1, last)
        : event.key === "ArrowUp" || event.key === "ArrowLeft"
          ? Math.max(index - 1, 0)
          : event.key === "Home"
            ? 0
            : event.key === "End"
              ? last
              : null;
    if (target === null) return;
    event.preventDefault();
    buttons.current[target]?.focus();
  }

  return (
    <nav aria-label={label} className={cn("min-w-0", className)}>
      <ol className="space-y-0.5">
        {steps.map((step, index) => {
          const isCurrent = step.key === current;
          const errors = step.errorCount ?? 0;
          return (
            <li key={step.key}>
              <button
                ref={(element) => {
                  buttons.current[index] = element;
                }}
                type="button"
                tabIndex={isCurrent ? 0 : -1}
                aria-current={isCurrent ? "step" : undefined}
                aria-label={stepName(step, errors)}
                onClick={() => onSelect(step.key)}
                onKeyDown={(event) => onKeyDown(event, index)}
                className={cn(
                  "flex w-full items-start gap-2.5 rounded-md px-2.5 py-2 text-left outline-none transition-colors hover:bg-muted/60 focus-visible:ring-3 focus-visible:ring-ring/50",
                  isCurrent && "bg-muted",
                )}
              >
                <span
                  aria-hidden
                  className={cn(
                    "mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full border text-xs tabular-nums",
                    errors > 0
                      ? "border-tone-danger-foreground text-tone-danger-foreground"
                      : step.complete
                        ? "border-primary bg-primary text-primary-foreground"
                        : "border-input text-muted-foreground",
                  )}
                >
                  {step.complete && errors === 0 ? <CheckIcon className="size-3" /> : index + 1}
                </span>
                <span className="min-w-0 flex-1">
                  <span className={cn("block text-sm", isCurrent ? "font-medium text-foreground" : "text-subtle-foreground")}>
                    {step.label}
                  </span>
                  {errors > 0 ? (
                    <span className="block text-xs text-tone-danger-foreground">{issueText(errors)}</span>
                  ) : step.complete && step.summary ? (
                    <span className="block truncate text-xs text-muted-foreground" title={step.summary}>
                      {step.summary}
                    </span>
                  ) : null}
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
