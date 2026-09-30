import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";
import { formatDateTime } from "@/lib/format";
import { PropertyList, type Property } from "@/patterns";
import type { ReviewDetail } from "./model/types";

const PANEL_ID = "review-technical-details";

function Mono({ children }: { children: string }) {
  return <span className="font-mono text-xs break-all">{children}</span>;
}

/** Identifiers and planner output for support and audits, collapsed by default. */
export function TechnicalDetails({ review }: { review: ReviewDetail }) {
  const [open, setOpen] = useState(false);
  const reason = review.presentation.technical_reason;
  const items: Property[] = [
    { label: "Review ID", value: <Mono>{review.id}</Mono> },
    { label: "Review mechanism", value: review.review_origin },
    { label: "Current memory ID", value: <Mono>{review.incumbent_memory_id}</Mono> },
    { label: "Proposed memory ID", value: review.challenger_memory_id ? <Mono>{review.challenger_memory_id}</Mono> : "None" },
    { label: "Opened", value: review.created_at ? formatDateTime(review.created_at) : "Unknown" },
    { label: "Status", value: review.status },
    ...(reason ? [{ label: "Planner reason", value: <Mono>{reason}</Mono> }] : []),
  ];
  return (
    <div className="border-t pt-3.5">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={PANEL_ID}
        onClick={() => setOpen((current) => !current)}
        className="flex items-center gap-1.5 text-sm font-medium text-subtle-foreground hover:text-foreground"
      >
        {open ? <ChevronDown aria-hidden className="size-3.5" /> : <ChevronRight aria-hidden className="size-3.5" />}
        Technical details
      </button>
      {open ? (
        <div id={PANEL_ID} className="pt-3.5">
          <PropertyList items={items} />
        </div>
      ) : null}
    </div>
  );
}
