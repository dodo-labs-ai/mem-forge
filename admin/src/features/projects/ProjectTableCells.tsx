import { Link } from "react-router-dom";
import { isReservedProjectKey } from "@/api";
import { formatCount, formatRelative } from "@/lib/format";
import { projectPath } from "@/lib/paths";
import { StatusBadge } from "@/patterns";
import { builtInRole } from "./model/projectList";
import type { Project } from "./model/types";
import { ProjectRowActions } from "./ProjectRowActions";
import { useProjectTable } from "./projectTableContext";

const NO_VALUE = "—";

export function ProjectNameCell({ project }: { project: Project }) {
  const role = builtInRole(project.key);
  return (
    <div className="flex min-w-0 flex-col gap-0.5">
      <span className="flex items-center gap-2">
        <Link
          to={projectPath(project.key)}
          onClick={(event) => event.stopPropagation()}
          className="truncate font-medium text-foreground hover:underline"
        >
          {project.name}
        </Link>
        {role ? <StatusBadge tone="idle">{role}</StatusBadge> : null}
      </span>
      <span className="truncate font-mono text-xs text-muted-foreground">{project.key}</span>
    </div>
  );
}

export function ProjectSourcesCell({ project }: { project: Project }) {
  const { sourceCounts } = useProjectTable();
  const count = sourceCounts?.get(project.key) ?? 0;
  return (
    <div className="text-right font-mono text-sm text-subtle-foreground tabular-nums">
      {sourceCounts === null ? NO_VALUE : formatCount(count)}
    </div>
  );
}

export function ProjectMemoriesCell({ project }: { project: Project }) {
  return (
    <div className="text-right font-mono text-sm font-medium text-foreground tabular-nums">
      {formatCount(project.memory_count)}
    </div>
  );
}

export function ProjectCreatedCell({ project }: { project: Project }) {
  return (
    <div className="text-right text-sm text-muted-foreground">
      {project.created_at ? formatRelative(project.created_at) : NO_VALUE}
    </div>
  );
}

export function ProjectActionsCell({ project }: { project: Project }) {
  const { canManage, onEdit, onDelete } = useProjectTable();
  // Built-in projects cannot be renamed or deleted, so they have no menu.
  if (isReservedProjectKey(project.key)) return null;
  return <ProjectRowActions project={project} changes={canManage ? { onEdit, onDelete } : null} />;
}
