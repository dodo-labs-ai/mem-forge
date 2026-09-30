import { formatRelative } from "@/lib/format";
import { shortId } from "./model/presentation";
import type { EvaluationCase } from "./model/types";

/** The IDs that locate one case, each with its own label. */
export function CaseIdentifiers({ evaluationCase }: { evaluationCase: EvaluationCase }) {
  const items = [
    { label: "Occurred", value: formatRelative(evaluationCase.occurred_at), mono: false },
    { label: "Source unit", value: shortId(evaluationCase.source_unit_id), mono: true },
    { label: "Event", value: shortId(evaluationCase.event_id), mono: true },
    ...(evaluationCase.trace_id ? [{ label: "Trace", value: shortId(evaluationCase.trace_id), mono: true }] : []),
  ];
  return (
    <dl className="flex flex-wrap gap-x-5 gap-y-1 text-xs">
      {items.map((item) => (
        <div key={item.label} className="flex min-w-0 gap-1.5">
          <dt className="text-muted-foreground">{item.label}</dt>
          <dd className={item.mono ? "font-mono text-subtle-foreground" : "text-subtle-foreground"}>{item.value}</dd>
        </div>
      ))}
    </dl>
  );
}
