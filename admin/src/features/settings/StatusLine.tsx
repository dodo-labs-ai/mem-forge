import { AlertCircle, AlertTriangle, CircleCheck, RefreshCw } from "lucide-react";
import { cn } from "@/lib/cn";
import { toneText } from "@/patterns";
import type { StatusSummary } from "./model/types";

const ICONS = { ok: CircleCheck, warn: AlertTriangle, danger: AlertCircle, live: RefreshCw, idle: AlertTriangle } satisfies Record<
  StatusSummary["tone"],
  unknown
>;

/** One state in a line: what was found, and what follows from it. */
export function StatusLine({ summary }: { summary: StatusSummary }) {
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
