import { dataColumns } from "@/patterns";
import type { Project } from "./model/types";
import {
  ProjectActionsCell,
  ProjectCreatedCell,
  ProjectMemoriesCell,
  ProjectNameCell,
  ProjectSourcesCell,
} from "./ProjectTableCells";

/** Module-level so the table keeps row identity across renders; cells read the page through ProjectTableContext. */
const column = dataColumns<Project>();

export const PROJECT_COLUMNS = [
  column.display({ id: "project", header: "Project", cell: ({ row }) => <ProjectNameCell project={row.original} /> }),
  column.display({
    id: "sources",
    header: () => <div className="text-right">Sources</div>,
    cell: ({ row }) => <ProjectSourcesCell project={row.original} />,
    meta: { className: "w-28" },
  }),
  column.display({
    id: "memories",
    header: () => <div className="text-right">Memories</div>,
    cell: ({ row }) => <ProjectMemoriesCell project={row.original} />,
    meta: { className: "w-32" },
  }),
  column.display({
    id: "created",
    header: () => <div className="text-right">Created</div>,
    cell: ({ row }) => <ProjectCreatedCell project={row.original} />,
    meta: { className: "w-36" },
  }),
  column.display({
    id: "actions",
    header: () => <span className="sr-only">Actions</span>,
    cell: ({ row }) => <ProjectActionsCell project={row.original} />,
    meta: { className: "w-[1%] whitespace-nowrap" },
  }),
];
