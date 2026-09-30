import { ArrowLeft, ArrowRight, ChevronRight, Files, Pencil, Search, Trash2 } from "lucide-react";
import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { isReservedProjectKey } from "@/api";
import { SourceIcon, useProjectSources, useSourceTypeLabels } from "@/features/sources";
import { cn } from "@/lib/cn";
import { errorMessage } from "@/lib/errors";
import { formatCount, pluralize } from "@/lib/format";
import { PROJECTS_PATH, SOURCES_PATH, projectMemoriesPath } from "@/lib/paths";
import { BackLink, EmptyState, ErrorNotice, StatCard, StatusBadge } from "@/patterns";
import { Button, buttonVariants } from "@/ui/button";
import { Skeleton } from "@/ui/skeleton";
import { useProjectList } from "./api";
import { DeleteProjectDialog } from "./DeleteProjectDialog";
import type { DeletedProjectState } from "./model/projectDeletion";
import { builtInRole } from "./model/projectList";
import { projectSourceRows, type ProjectSourceRow } from "./model/projectSources";
import type { Project } from "./model/types";
import { EditProjectDialog } from "./ProjectFormDialogs";

const NO_VALUE = "—";
const BINDING_LABELS: Record<ProjectSourceRow["binding"], string> = { direct: "Direct", by_field: "By field" };

function BackToProjects() {
  return <BackLink label="Projects" render={<Link to={PROJECTS_PATH} />} />;
}

function SourceRowLink({ row, typeLabel }: { row: ProjectSourceRow; typeLabel: string }) {
  return (
    <li>
      <Link to={SOURCES_PATH} className="flex items-center gap-3 px-4 py-3 hover:bg-muted/50">
        <span className="flex size-8 shrink-0 items-center justify-center rounded-md border bg-surface-subtle">
          <SourceIcon type={row.source.type} client={row.source.client} className="size-4" />
        </span>
        <span className="flex min-w-0 flex-1 flex-col gap-0.5">
          <span className="flex items-center gap-2">
            <span className="truncate font-medium text-foreground">{row.source.name}</span>
            <StatusBadge tone={row.binding === "by_field" ? "live" : "idle"}>{BINDING_LABELS[row.binding]}</StatusBadge>
          </span>
          <span className="truncate text-xs text-muted-foreground">{typeLabel}</span>
        </span>
        <span className="shrink-0 font-mono text-sm text-subtle-foreground tabular-nums">
          {pluralize(row.memoryCount, "memory", "memories")}
        </span>
        <ChevronRight aria-hidden className="size-4 shrink-0 text-muted-foreground" />
      </Link>
    </li>
  );
}

function ProjectSources({ rows, loading, error, onRetry }: {
  rows: ProjectSourceRow[];
  loading: boolean;
  error: unknown;
  onRetry: () => void;
}) {
  const typeLabels = useSourceTypeLabels();
  const goToSources = (
    <Link to={SOURCES_PATH} className={cn(buttonVariants({ size: "sm", variant: "outline" }))}>
      <ArrowRight />
      Go to Sources
    </Link>
  );
  return (
    <section aria-labelledby="project-sources-title" className="overflow-hidden rounded-lg border bg-surface">
      <div className="flex items-center gap-3 border-b px-4 py-3">
        <div className="flex-1 space-y-0.5">
          <h2 id="project-sources-title" className="text-sm font-semibold text-foreground">
            Sources sending memories here
          </h2>
          <p className="text-xs text-muted-foreground">
            Change where a source sends its memories in that source’s Configure dialog.
          </p>
        </div>
        {rows.length > 0 ? goToSources : null}
      </div>
      {error ? (
        <div className="p-4">
          <ErrorNotice title="Could not load sources" message={errorMessage(error)} onRetry={onRetry} />
        </div>
      ) : loading ? (
        <div className="p-4">
          <Skeleton className="h-6 w-full" />
        </div>
      ) : rows.length === 0 ? (
        <EmptyState
          icon={Files}
          title="No sources send memories here"
          description="Open a source’s Configure dialog and pick this project."
          action={goToSources}
        />
      ) : (
        <ul className="divide-y">
          {rows.map((row) => (
            <SourceRowLink key={row.key} row={row} typeLabel={typeLabels[row.source.type] ?? row.source.type} />
          ))}
        </ul>
      )}
    </section>
  );
}

function ProjectDetail({ project, canManage }: { project: Project; canManage: boolean }) {
  const navigate = useNavigate();
  const sourcesQuery = useProjectSources();
  const [editing, setEditing] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const rows = projectSourceRows(sourcesQuery.byProject.get(project.key) ?? []);
  const builtIn = isReservedProjectKey(project.key);
  const changeable = canManage && !builtIn;
  const role = builtInRole(project.key);

  return (
    <div className="space-y-6">
      <BackToProjects />
      <header className="flex flex-wrap items-end gap-2">
        <div className="min-w-0 flex-1 space-y-1.5">
          <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight text-foreground">
            <span className="truncate">{project.name}</span>
            {role ? <StatusBadge tone="idle">{role}</StatusBadge> : null}
          </h1>
          <p className="flex items-center gap-2 text-xs">
            <span className="text-muted-foreground">Code</span>
            <span className="font-mono text-subtle-foreground">{project.key}</span>
          </p>
        </div>
        <Link to={projectMemoriesPath(project.key)} className={cn(buttonVariants({ variant: "outline" }))}>
          <ArrowRight />
          View memories
        </Link>
        {changeable ? (
          <>
            <Button variant="outline" onClick={() => setEditing(true)}>
              <Pencil />
              Edit
            </Button>
            <Button variant="outline" className="text-tone-danger-foreground" onClick={() => setDeleting(true)}>
              <Trash2 />
              Delete
            </Button>
          </>
        ) : null}
      </header>
      {builtIn ? (
        <p className="text-sm text-muted-foreground">Built-in projects cannot be renamed or deleted.</p>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2">
        <StatCard label="Memories" value={formatCount(project.memory_count)} detail="Active memories you can see" />
        <StatCard
          label="Sources"
          value={sourcesQuery.isError ? NO_VALUE : sourcesQuery.isPending ? undefined : formatCount(rows.length)}
          detail="Direct and field-based bindings"
        />
      </div>

      <ProjectSources
        rows={rows}
        loading={sourcesQuery.isPending}
        error={sourcesQuery.isError ? sourcesQuery.error : null}
        onRetry={() => void sourcesQuery.refetch()}
      />

      {changeable ? (
        <>
          <EditProjectDialog project={editing ? project : null} onOpenChange={setEditing} />
          <DeleteProjectDialog
            project={deleting ? project : null}
            onOpenChange={setDeleting}
            onDeleted={(notice) => {
              const state: DeletedProjectState = { deletedNotice: notice };
              navigate(PROJECTS_PATH, { state });
            }}
          />
        </>
      ) : null}
    </div>
  );
}

export function ProjectDetailPage() {
  const { key = "" } = useParams<{ key: string }>();
  const projectsQuery = useProjectList();
  const project = projectsQuery.projects.find((candidate) => candidate.key === key);

  if (projectsQuery.isPending) {
    return (
      <div className="space-y-6">
        <BackToProjects />
        <Skeleton className="h-8 w-64" />
        <Skeleton className="h-24 w-full" />
      </div>
    );
  }
  if (projectsQuery.isError) {
    return (
      <div className="space-y-6">
        <BackToProjects />
        <ErrorNotice
          title="Could not load the project"
          message={errorMessage(projectsQuery.error)}
          onRetry={() => void projectsQuery.refetch()}
        />
      </div>
    );
  }
  if (!project) {
    return (
      <EmptyState
        icon={Search}
        title="Project not found"
        description="It may have been deleted, or the link is out of date."
        action={
          <Link to={PROJECTS_PATH} className={cn(buttonVariants({ variant: "outline" }))}>
            <ArrowLeft />
            Back to Projects
          </Link>
        }
      />
    );
  }
  return <ProjectDetail project={project} canManage={projectsQuery.canManage} />;
}
