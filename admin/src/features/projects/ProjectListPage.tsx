import { FolderKanban, Info, Plus, X } from "lucide-react";
import { useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useProjectSources } from "@/features/sources";
import { errorMessage } from "@/lib/errors";
import { projectPath } from "@/lib/paths";
import { DataTable, EmptyState, ErrorNotice, GroupHeader, Notice, PageHeader, type DataTableGroup } from "@/patterns";
import { Button } from "@/ui/button";
import { useProjectList } from "./api";
import { DeleteProjectDialog } from "./DeleteProjectDialog";
import { deletedNoticeFrom } from "./model/projectDeletion";
import { splitProjects } from "./model/projectList";
import type { Project } from "./model/types";
import { PROJECT_COLUMNS } from "./projectColumns";
import { CreateProjectDialog, EditProjectDialog } from "./ProjectFormDialogs";
import { ProjectTableContext } from "./projectTableContext";

const OWN_GROUP_ID = "own";
const BUILT_IN_GROUP_ID = "built-in";

export function ProjectListPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const projectsQuery = useProjectList();
  const sourcesQuery = useProjectSources();
  const [createOpen, setCreateOpen] = useState(false);
  const [editing, setEditing] = useState<Project | null>(null);
  const [deleting, setDeleting] = useState<Project | null>(null);
  const [notice, setNotice] = useState<string | null>(() => deletedNoticeFrom(location.state));
  const { canManage } = projectsQuery;

  const { own, builtIn } = splitProjects(projectsQuery.projects);
  const sourceCounts =
    sourcesQuery.isPending || sourcesQuery.isError
      ? null
      : new Map([...sourcesQuery.byProject].map(([key, sources]) => [key, sources.length]));

  const groups: DataTableGroup<Project>[] = [
    ...(own.length > 0 ? [{ id: OWN_GROUP_ID, header: null, rows: own }] : []),
    ...(builtIn.length > 0 ? [{ id: BUILT_IN_GROUP_ID, header: <GroupHeader title="Built-in" />, rows: builtIn }] : []),
  ];
  const showTable = projectsQuery.isPending || groups.length > 0;

  function dismissNotice() {
    setNotice(null);
    // The notice may have come with the navigation; drop it so a reload does not show it again.
    if (location.state !== null) navigate(location.pathname, { replace: true, state: null });
  }

  const newProjectButton = canManage ? (
    <Button onClick={() => setCreateOpen(true)}>
      <Plus />
      New project
    </Button>
  ) : null;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Projects"
        description="Projects group memories by what they are about. Each source sends its memories to a project, and agents working in a matching repository see that project first."
        actions={newProjectButton}
      />

      {notice ? (
        <Notice
          tone="idle"
          icon={Info}
          action={
            <Button size="icon-sm" variant="ghost" aria-label="Dismiss" onClick={dismissNotice}>
              <X />
            </Button>
          }
        >
          {notice}
        </Notice>
      ) : null}

      {projectsQuery.isError ? (
        <ErrorNotice
          title="Could not load projects"
          message={errorMessage(projectsQuery.error)}
          onRetry={() => void projectsQuery.refetch()}
        />
      ) : null}

      {projectsQuery.isSuccess && own.length === 0 ? (
        <div className="rounded-lg border bg-surface">
          <EmptyState
            icon={FolderKanban}
            title="No projects yet"
            description={
              canManage
                ? "Create one for a product or codebase, then bind sources to it."
                : "A workspace admin creates projects for products or codebases."
            }
            action={newProjectButton}
          />
        </div>
      ) : null}

      {showTable ? (
        <div className="space-y-2">
          <ProjectTableContext.Provider value={{ sourceCounts, canManage, onEdit: setEditing, onDelete: setDeleting }}>
            <DataTable
              aria-label="Projects"
              columns={PROJECT_COLUMNS}
              data={groups}
              getRowId={(project) => project.id}
              onRowClick={(project) => navigate(projectPath(project.key))}
              loading={projectsQuery.isPending}
            />
          </ProjectTableContext.Provider>
          {builtIn.length > 0 ? (
            <p className="text-xs text-muted-foreground">
              Shared memories rank with every project. Unsorted holds memories from sources without a project.
              Built-in projects cannot be renamed or deleted.
            </p>
          ) : null}
        </div>
      ) : null}

      {canManage ? (
        <>
          <CreateProjectDialog open={createOpen} onOpenChange={setCreateOpen} />
          <EditProjectDialog
            project={editing}
            onOpenChange={(open) => {
              if (!open) setEditing(null);
            }}
          />
          <DeleteProjectDialog
            project={deleting}
            onOpenChange={(open) => {
              if (!open) setDeleting(null);
            }}
            onDeleted={setNotice}
          />
        </>
      ) : null}
    </div>
  );
}
