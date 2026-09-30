import { ExternalLink } from "lucide-react";
import { cn } from "@/lib/cn";
import { StatusBadge } from "@/patterns";
import { memoryStatus } from "./model/reviewDecision";
import type { ReviewEvidenceGroup, ReviewMemory } from "./model/types";

/** Evidence groups shown per memory; the memory page lists the rest. */
const EVIDENCE_SHOWN = 3;

interface MemoryStateCardProps {
  label: string;
  memory: ReviewMemory | null | undefined;
  emptyText: string;
  typeLabels: Readonly<Record<string, string>>;
  /** Marks the side a waiting decision would switch to. */
  emphasized?: boolean;
}

function EvidenceGroup({ group, typeLabels }: { group: ReviewEvidenceGroup; typeLabels: Readonly<Record<string, string>> }) {
  const excerpt = group.items.find((item) => item.role === "primary")?.excerpt;
  const sourceUrl = group.document?.source_url;
  return (
    <div className="space-y-1.5 border-t pt-2.5">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-medium text-foreground">
          {group.document?.title ?? group.document?.doc_id ?? group.doc_id ?? group.source_id}
        </span>
        <StatusBadge tone="idle">{typeLabels[group.source_type] ?? group.source_type}</StatusBadge>
        {sourceUrl ? (
          <a
            href={sourceUrl}
            target="_blank"
            rel="noreferrer"
            className="ml-auto inline-flex items-center gap-1 text-xs text-subtle-foreground hover:text-foreground"
          >
            Open source
            <ExternalLink aria-hidden className="size-3" />
          </a>
        ) : null}
      </div>
      {excerpt ? <blockquote className="border-l-2 pl-2.5 text-xs leading-relaxed text-subtle-foreground">{excerpt}</blockquote> : null}
    </div>
  );
}

/** One side of a review: the memory as it is, or as the proposal would make it, with its evidence. */
export function MemoryStateCard({ label, memory, emptyText, typeLabels, emphasized = false }: MemoryStateCardProps) {
  const status = memory ? memoryStatus(memory.status) : null;
  const evidence = memory?.evidence.slice(0, EVIDENCE_SHOWN) ?? [];
  return (
    <section
      aria-label={label}
      className={cn("flex flex-col gap-3.5 rounded-xl border bg-surface p-4.5", emphasized && "border-foreground")}
    >
      <div className="flex items-center gap-2">
        <h2 className="text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">{label}</h2>
        {status ? (
          <StatusBadge tone={status.tone} className="ml-auto">
            {status.label}
          </StatusBadge>
        ) : null}
      </div>
      {memory ? (
        <p className="text-[15px] leading-relaxed whitespace-pre-wrap text-foreground">{memory.content}</p>
      ) : (
        <p className="text-sm text-muted-foreground">{emptyText}</p>
      )}
      {evidence.length > 0 ? (
        <div className="space-y-2.5">
          <h3 className="text-xs font-semibold text-muted-foreground">Evidence</h3>
          {evidence.map((group, index) => (
            <EvidenceGroup key={group.evidence_unit_id ?? `${group.source_id}:${group.doc_id}:${index}`} group={group} typeLabels={typeLabels} />
          ))}
        </div>
      ) : null}
    </section>
  );
}
