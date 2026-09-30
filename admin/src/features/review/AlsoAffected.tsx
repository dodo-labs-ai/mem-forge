import { Link } from "react-router-dom";
import { memoryPath } from "@/lib/paths";
import { StatusBadge } from "@/patterns";
import { memoryStatus } from "./model/reviewDecision";
import type { ReviewMemory } from "./model/types";

/** Other memories the same decision changes, linked to their memory pages. */
export function AlsoAffected({ memories }: { memories: ReviewMemory[] }) {
  if (memories.length === 0) return null;
  return (
    <section aria-labelledby="review-also-affected" className="space-y-1.5 rounded-xl border bg-surface p-4">
      <div className="flex flex-wrap items-baseline gap-2.5">
        <h2 id="review-also-affected" className="text-sm font-semibold text-foreground">
          Also affected
        </h2>
        <p className="text-xs text-muted-foreground">Other memories this decision touches</p>
      </div>
      <ul>
        {memories.map((memory) => {
          const status = memoryStatus(memory.status);
          return (
            <li key={memory.id} className="flex items-center gap-3 border-t py-2.5">
              <StatusBadge tone={status.tone} variant="dot" className="shrink-0">
                {status.label}
              </StatusBadge>
              <Link to={memoryPath(memory.id)} className="min-w-0 truncate text-sm text-foreground hover:underline">
                {memory.content}
              </Link>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
