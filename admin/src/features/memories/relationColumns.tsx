import { dataColumns } from "@/patterns";
import { DecidedByCell, FirstMemoryCell, RelationLabelCell, SecondMemoryCell } from "./RelationCells";
import type { RelationPair } from "./model/types";

/** Module-level so the table keeps row identity across renders. */
const column = dataColumns<RelationPair>();

export const RELATION_COLUMNS = [
  column.display({
    id: "relation",
    header: "Relation",
    cell: ({ row }) => <RelationLabelCell pair={row.original} />,
    meta: { className: "w-36 align-top" },
  }),
  column.display({
    id: "memory",
    header: "Memory",
    cell: ({ row }) => <FirstMemoryCell pair={row.original} />,
    meta: { className: "w-[40%] align-top" },
  }),
  column.display({
    id: "related",
    header: "Related memory",
    cell: ({ row }) => <SecondMemoryCell pair={row.original} />,
    meta: { className: "w-[40%] align-top" },
  }),
  column.display({
    id: "decided",
    header: "Decided by",
    cell: ({ row }) => <DecidedByCell pair={row.original} />,
    meta: { className: "w-28 align-top" },
  }),
];

export function relationPairId(pair: RelationPair): string {
  return [pair.label, ...pair.memories.map((memory) => memory.memory_id)].join(":");
}
