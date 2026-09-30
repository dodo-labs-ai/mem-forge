import { Link } from "react-router-dom";
import { memoryPath } from "@/lib/paths";
import { RelationLabelBadge, SourceLabel } from "./MemoryBadges";
import { useMemoryTable } from "./memoryTableContext";
import { sourceName } from "./model/memoryRows";
import { orderedPairMemories } from "./model/relations";
import type { RelatedMemory, RelationPair } from "./model/types";

const SIDE_LABELS = { newer: "Newer", older: "Older" } as const;
const DECIDED_BY_LABELS: Record<RelationPair["decided_by"], string> = { classifier: "Classifier", review: "Reviewer" };

interface RelatedMemorySummaryProps {
  memory: RelatedMemory;
  typeLabels: Readonly<Record<string, string>>;
}

/** A related Memory's statement, linked to its detail, with its sources and when they recorded it. */
export function RelatedMemorySummary({ memory, typeLabels }: RelatedMemorySummaryProps) {
  return (
    <div className="flex min-w-0 flex-col gap-1 whitespace-normal">
      <Link to={memoryPath(memory.memory_id)} className="text-sm font-medium text-foreground hover:underline">
        {memory.summary}
      </Link>
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        {(memory.sources ?? []).map((source) => (
          <SourceLabel key={source.source_id} source={source} name={sourceName(source, typeLabels)} />
        ))}
        {memory.evidence_time ? (
          <span className="text-xs text-muted-foreground">Recorded {memory.evidence_time}</span>
        ) : null}
      </div>
    </div>
  );
}

function PairSide({ pair, index }: { pair: RelationPair; index: number }) {
  const { typeLabels } = useMemoryTable();
  const entry = orderedPairMemories(pair)[index];
  if (!entry) return null;
  return (
    <div className="flex flex-col gap-1">
      {entry.side ? (
        <span className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">{SIDE_LABELS[entry.side]}</span>
      ) : null}
      <RelatedMemorySummary memory={entry.memory} typeLabels={typeLabels} />
    </div>
  );
}

export function RelationLabelCell({ pair }: { pair: RelationPair }) {
  return <RelationLabelBadge label={pair.label} />;
}

export function FirstMemoryCell({ pair }: { pair: RelationPair }) {
  return <PairSide pair={pair} index={0} />;
}

export function SecondMemoryCell({ pair }: { pair: RelationPair }) {
  return <PairSide pair={pair} index={1} />;
}

export function DecidedByCell({ pair }: { pair: RelationPair }) {
  return <span className="text-muted-foreground">{DECIDED_BY_LABELS[pair.decided_by]}</span>;
}
