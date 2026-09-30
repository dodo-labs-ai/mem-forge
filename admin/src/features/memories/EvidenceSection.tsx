import { ExternalLink, Paperclip } from "lucide-react";
import { SourceIcon } from "@/features/sources";
import { formatDateTime, pluralize } from "@/lib/format";
import { StatusBadge } from "@/patterns";
import { buttonVariants } from "@/ui/button";
import { DetailCard } from "./DetailCard";
import type { EvidenceGroup, EvidenceItem } from "./model/types";

const GROUP_KIND_LABELS: Record<EvidenceGroup["kind"], string> = { evidence_unit: "Evidence unit", document: "Document" };
const ITEM_ROLE_LABELS: Record<EvidenceItem["role"], string> = { primary: "Primary", required: "Required", context: "Context" };

/** "Text", "Text, context only" or "Artifact": what the item is and whether it supports the memory. */
function itemKindText(item: EvidenceItem): string {
  const kind = item.kind === "artifact" ? "Artifact" : "Text";
  return item.support_contribution ? kind : `${kind}, context only`;
}

function EvidenceItemBlock({ item }: { item: EvidenceItem }) {
  return (
    <div className="flex flex-col gap-2 rounded-md bg-surface-subtle p-3">
      <div className="flex items-center gap-2">
        <StatusBadge tone={item.role === "primary" ? "live" : "idle"}>{ITEM_ROLE_LABELS[item.role]}</StatusBadge>
        <span className="text-xs text-muted-foreground">{itemKindText(item)}</span>
      </div>
      {item.excerpt ? (
        <blockquote className="border-l-2 border-border-strong pl-3 text-sm text-subtle-foreground">{item.excerpt}</blockquote>
      ) : null}
      {item.artifact ? (
        <a
          href={item.artifact.url}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-sm text-foreground underline-offset-2 hover:underline"
        >
          <Paperclip aria-hidden className="size-3.5" />
          {item.artifact.filename}
        </a>
      ) : null}
    </div>
  );
}

function EvidenceGroupCard({ group, typeLabel }: { group: EvidenceGroup; typeLabel: string }) {
  const document = group.document;
  const title = document?.title ?? document?.doc_id ?? group.doc_id ?? group.source_id;
  return (
    <article className="flex flex-col gap-3 rounded-lg border p-3.5">
      <div className="flex items-start gap-2.5">
        <SourceIcon type={group.source_type} className="mt-0.5 size-4" />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1">
            <h3 className="truncate text-sm font-medium text-foreground">{title}</h3>
            {document?.source_url ? (
              <a
                href={document.source_url}
                target="_blank"
                rel="noopener noreferrer"
                aria-label={`Open ${title} in its source`}
                className={buttonVariants({ size: "icon-xs", variant: "ghost" })}
              >
                <ExternalLink />
              </a>
            ) : null}
          </div>
          <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
            <span>{typeLabel}</span>
            <span>{GROUP_KIND_LABELS[group.kind]}</span>
            {group.current ? null : <StatusBadge tone="idle">Historical</StatusBadge>}
          </div>
        </div>
      </div>
      <div className="flex flex-col gap-2">
        {group.items.map((item, index) => (
          <EvidenceItemBlock key={item.evidence_reference_id ?? `${item.role}-${index}`} item={item} />
        ))}
      </div>
      {document?.content_url || document?.pdf_url || document?.source_updated_at ? (
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs">
          {document.content_url ? (
            <a href={document.content_url} target="_blank" rel="noopener noreferrer" className="font-medium hover:underline">
              Open content
            </a>
          ) : null}
          {document.pdf_url ? (
            <a href={document.pdf_url} target="_blank" rel="noopener noreferrer" className="font-medium hover:underline">
              Open PDF
            </a>
          ) : null}
          {document.source_updated_at ? (
            <span className="ml-auto text-muted-foreground">Source updated {formatDateTime(document.source_updated_at)}</span>
          ) : null}
        </div>
      ) : null}
    </article>
  );
}

interface EvidenceSectionProps {
  evidence: readonly EvidenceGroup[];
  typeLabels: Readonly<Record<string, string>>;
}

export function EvidenceSection({ evidence, typeLabels }: EvidenceSectionProps) {
  return (
    <DetailCard title="Evidence" meta={pluralize(evidence.length, "document")}>
      {evidence.length === 0 ? (
        <p className="text-sm text-muted-foreground">No readable evidence recorded.</p>
      ) : (
        <div className="flex flex-col gap-3">
          {evidence.map((group, index) => (
            <EvidenceGroupCard
              key={group.evidence_unit_id ?? `${group.source_id}-${group.doc_id}-${index}`}
              group={group}
              typeLabel={typeLabels[group.source_type] ?? group.source_type}
            />
          ))}
        </div>
      )}
    </DetailCard>
  );
}
