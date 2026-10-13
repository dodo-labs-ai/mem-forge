import { AlertCircle, AlertTriangle, CircleCheck } from "lucide-react";
import { cn } from "@/lib/cn";
import { toneText } from "@/patterns";
import type { ValidationSummary } from "./model/modelStatus";

const ICONS = { ok: CircleCheck, warn: AlertTriangle, danger: AlertCircle, live: CircleCheck, idle: AlertTriangle } satisfies Record<
  ValidationSummary["tone"],
  unknown
>;

/** The latest validation of a model: what it found, and what follows from it. */
export function ValidationLine({ summary }: { summary: ValidationSummary }) {
  const Icon = ICONS[summary.tone];
  return (
    <div className="flex items-start gap-2 text-sm">
      <Icon aria-hidden className={cn("mt-0.5 size-4 shrink-0", toneText[summary.tone])} />
      <p className="min-w-0">
        <span className={cn("font-medium", toneText[summary.tone])}>{summary.headline}</span>
        {summary.detail ? <span className="text-subtle-foreground"> {summary.detail}</span> : null}
      </p>
    </div>
  );
}
