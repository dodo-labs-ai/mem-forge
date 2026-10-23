import { dataColumns } from "@/patterns";
import type { Entity } from "./model/types";
import { EntityFirstSeenCell, EntityNameCell } from "./EntityTableCells";

/** Module-level so the table keeps row identity across renders. */
const column = dataColumns<Entity>();

export const ENTITY_COLUMNS = [
  column.display({ id: "entity", header: "Entity", cell: ({ row }) => <EntityNameCell entity={row.original} /> }),
  column.display({
    id: "first-seen",
    header: () => <div className="text-right">First seen</div>,
    cell: ({ row }) => <EntityFirstSeenCell entity={row.original} />,
    meta: { className: "w-36" },
  }),
];
