import { dataColumns } from "@/patterns";
import { AgeCell, MemoryCell, StatusCell, SupportCell, TypeCell } from "./MemoryCells";
import type { MemoryRow } from "./model/memoryRows";

/** Module-level so the table keeps row identity across renders; cells read the page through MemoryTableContext. */
const column = dataColumns<MemoryRow>();

export const MEMORY_COLUMNS = [
  column.display({ id: "memory", header: "Memory", cell: ({ row }) => <MemoryCell row={row.original} /> }),
  column.display({
    id: "type",
    header: "Type",
    cell: ({ row }) => <TypeCell row={row.original} />,
    meta: { className: "w-32" },
  }),
  column.display({
    id: "status",
    header: "Status",
    cell: ({ row }) => <StatusCell row={row.original} />,
    meta: { className: "w-36" },
  }),
  column.display({
    id: "sources",
    header: () => <div className="text-right">Sources</div>,
    cell: ({ row }) => <SupportCell row={row.original} />,
    meta: { className: "w-24" },
  }),
  column.display({
    id: "age",
    header: () => <div className="text-right">Age</div>,
    cell: ({ row }) => <AgeCell row={row.original} />,
    meta: { className: "w-28" },
  }),
];
