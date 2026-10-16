import { dataColumns } from "@/patterns";
import type { SourceRow } from "./model/sourceRows";
import { AccessCell, LastSyncCell, MemoriesCell, RunsCell } from "./SourceCells";
import { ActionsCell, NameCell } from "./SourceTableCells";

/** Module-level so the table keeps row identity across renders; cells read the page through SourceTableContext. */
const column = dataColumns<SourceRow>();

const SHARED_WORKSPACE_COLUMNS = [
  column.display({
    id: "source",
    header: "Source",
    cell: ({ row }) => <NameCell row={row.original} />,
    meta: { className: "w-[34%]" },
  }),
  column.display({ id: "runs", header: "Runs on", cell: ({ row }) => <RunsCell row={row.original} /> }),
  column.display({ id: "access", header: "Access", cell: ({ row }) => <AccessCell row={row.original} /> }),
  column.display({
    id: "memories",
    header: () => <div className="text-right">Memories</div>,
    cell: ({ row }) => <MemoriesCell row={row.original} />,
  }),
  column.display({ id: "sync", header: "Last sync", cell: ({ row }) => <LastSyncCell row={row.original} /> }),
  column.display({
    id: "actions",
    header: () => <span className="sr-only">Actions</span>,
    cell: ({ row }) => <ActionsCell row={row.original} />,
    meta: { className: "w-[1%] whitespace-nowrap" },
  }),
];

/** A workspace of one has nobody to share a Source with, so it leaves out who can use each one. */
const PERSONAL_WORKSPACE_COLUMNS = SHARED_WORKSPACE_COLUMNS.filter((sourceColumn) => sourceColumn.id !== "access");

export function sourceColumns(sharedWorkspace: boolean) {
  return sharedWorkspace ? SHARED_WORKSPACE_COLUMNS : PERSONAL_WORKSPACE_COLUMNS;
}
