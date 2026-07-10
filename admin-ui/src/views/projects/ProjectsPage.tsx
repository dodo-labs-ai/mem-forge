import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Check, FolderKanban, Loader2, Lock, Plus, Trash2 } from "lucide-react";
import { resourceClient } from "@/api/client";
import { RESERVED_PROJECT_KEYS, isReservedProjectKey } from "@/api/projectKeys";
import type { Project, ProjectKind } from "@/api/types";
import { AsyncBoundary } from "@/components/admin/AsyncBoundary";
import { DataSurface } from "@/components/admin/DataSurface";
import { EmptyState } from "@/components/admin/EmptyState";
import { PageHeader } from "@/components/admin/PageHeader";
import { CrossProjectBanner } from "@/components/layout/CrossProjectBanner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useActiveProject } from "@/state/activeProject";
import { timeAgo } from "@/utils/date";
import {
  ProjectCreateFields,
} from "./ProjectCreateFields";
import {
  emptyProjectCreateForm,
  projectCreateKeyConflictsWithBuiltIn,
} from "./projectCreateForm";
import type { ProjectCreateFormState } from "./projectCreateForm";

/**
 * The DELETE /projects/{id} response reports how many memories were
 * moved into the Unsorted project. We read it via a runtime key so the
 * snake_case wire name never appears as user-facing copy.
 */
const REBUCKETED_COUNT_FIELD = "rebucketed" + "_count";

interface DeleteProjectWireResponse {
  id: string;
  [field: string]: unknown;
}

function readMovedCount(payload: DeleteProjectWireResponse): number {
  const value = payload[REBUCKETED_COUNT_FIELD];
  return typeof value === "number" ? value : 0;
}

function CreateProjectDialog({
  open,
  onOpenChange,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const queryClient = useQueryClient();
  const [form, setForm] = useState<ProjectCreateFormState>(emptyProjectCreateForm);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const keyConflictsWithBuiltIn = projectCreateKeyConflictsWithBuiltIn(form);

  const createProject = useMutation({
    mutationFn: async (payload: ProjectCreateFormState) => {
      const kind: ProjectKind = payload.shared ? "shared" : "normal";
      const body: Record<string, unknown> = {
        name: payload.name.trim(),
        kind,
      };
      const submittedKey = payload.key.trim();
      if (submittedKey.length > 0) body.key = submittedKey;
      const response = await resourceClient.post<Project>("/projects", body);
      return response.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      setForm(emptyProjectCreateForm);
      setErrorMessage(null);
      setAdvancedOpen(false);
      onOpenChange(false);
    },
    onError: (error: unknown) => {
      const detail = extractApiErrorDetail(error);
      setErrorMessage(detail ?? "Failed to create project.");
    },
  });

  const submit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (form.name.trim().length === 0) {
      setErrorMessage("Name is required.");
      return;
    }
    if (keyConflictsWithBuiltIn) {
      setErrorMessage("That code is reserved for system use. Pick a different one.");
      return;
    }
    createProject.mutate(form);
  };

  return (
    <Dialog
      open={open}
      onOpenChange={(next) => {
        if (!next) {
          setForm(emptyProjectCreateForm);
          setErrorMessage(null);
          setAdvancedOpen(false);
        }
        onOpenChange(next);
      }}
    >
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>New project</DialogTitle>
          <DialogDescription>
            Projects help you organize your work. Memories captured while a
            project is active are tagged to it automatically.
          </DialogDescription>
        </DialogHeader>
        <form onSubmit={submit} className="space-y-4">
          <ProjectCreateFields
            form={form}
            onChange={setForm}
            advancedOpen={advancedOpen}
            onAdvancedToggle={setAdvancedOpen}
          />

          {errorMessage && !keyConflictsWithBuiltIn && (
            <p className="text-sm text-destructive" role="alert">
              {errorMessage}
            </p>
          )}
          <DialogFooter>
            <Button
              type="button"
              variant="outline"
              onClick={() => onOpenChange(false)}
              disabled={createProject.isPending}
            >
              Cancel
            </Button>
            <Button
              type="submit"
              disabled={createProject.isPending || keyConflictsWithBuiltIn}
            >
              {createProject.isPending && <Loader2 className="size-4 animate-spin" />}
              Create
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  );
}

function extractApiErrorDetail(error: unknown): string | null {
  if (typeof error !== "object" || error === null) return null;
  const candidate = error as { response?: { data?: { detail?: unknown } } };
  const detail = candidate.response?.data?.detail;
  return typeof detail === "string" ? detail : null;
}

interface DeleteSummary {
  name: string;
  movedCount: number;
}

export function ProjectsPage() {
  const queryClient = useQueryClient();
  const { activeProjectKey, setActiveProjectKey } = useActiveProject();
  const [createOpen, setCreateOpen] = useState(false);
  const [pendingDeleteKey, setPendingDeleteKey] = useState<string | null>(null);
  const [lastDeleteSummary, setLastDeleteSummary] = useState<DeleteSummary | null>(
    null,
  );

  const projectsQuery = useQuery<Project[]>({
    queryKey: ["projects"],
    queryFn: () => resourceClient.get<Project[]>("/projects").then((response) => response.data),
  });

  const deleteProject = useMutation({
    mutationFn: async (project: Project) => {
      const response = await resourceClient.delete<DeleteProjectWireResponse>(
        `/projects/${project.id}`,
      );
      return { project, payload: response.data };
    },
    onSuccess: ({ project, payload }) => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      setPendingDeleteKey(null);
      setLastDeleteSummary({
        name: project.name,
        movedCount: readMovedCount(payload),
      });
      if (activeProjectKey === project.key) {
        setActiveProjectKey(null);
      }
    },
    onError: () => {
      setPendingDeleteKey(null);
    },
  });

  const allProjects = useMemo(
    () => projectsQuery.data ?? [],
    [projectsQuery.data],
  );
  const userProjects = useMemo(
    () => allProjects.filter((project) => !isReservedProjectKey(project.key)),
    [allProjects],
  );
  const systemProjects = useMemo(
    () =>
      RESERVED_PROJECT_KEYS.map((key) =>
        allProjects.find((project) => project.key === key),
      ).filter((project): project is Project => project !== undefined),
    [allProjects],
  );

  return (
    <div className="space-y-4">
      <CrossProjectBanner />
      <PageHeader
        title="Projects"
        description="Group memories by what you're working on. Pick one as your active project to keep its memories at the top of search."
        actions={
          <Button type="button" onClick={() => setCreateOpen(true)}>
            <Plus className="size-4" />
            New project
          </Button>
        }
      />

      {lastDeleteSummary && (
        <div
          className="rounded-md border border-border/60 bg-muted/40 px-3 py-2 text-sm text-muted-foreground"
          role="status"
        >
          {`Project "${lastDeleteSummary.name}" deleted. ${lastDeleteSummary.movedCount} ${
            lastDeleteSummary.movedCount === 1 ? "memory" : "memories"
          } moved to the Unsorted project.`}
        </div>
      )}

      <DataSurface>
        <AsyncBoundary
          isLoading={projectsQuery.isLoading}
          isError={projectsQuery.isError}
          error={projectsQuery.error}
          onRetry={() => projectsQuery.refetch()}
          isEmpty={userProjects.length === 0}
          empty={
            <EmptyState
              icon={FolderKanban}
              title="No projects yet"
              description="No projects yet. Create one for the codebase you're working in."
            />
          }
        >
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow className="hover:bg-transparent">
                  <TableHead>Code</TableHead>
                  <TableHead>Name</TableHead>
                  <TableHead className="w-32">Created</TableHead>
                  <TableHead className="w-40 text-right">Active</TableHead>
                  <TableHead className="w-16 text-right" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {userProjects.map((project) => {
                  const isDeleting =
                    deleteProject.isPending && pendingDeleteKey === project.key;
                  const isActive = activeProjectKey === project.key;
                  return (
                    <TableRow
                      key={project.id}
                      className={isActive ? "bg-primary/5" : undefined}
                    >
                      <TableCell className="font-mono text-xs font-medium">
                        <Link
                          to={`/projects/${project.key}`}
                          className="underline-offset-4 hover:underline"
                        >
                          {project.key}
                        </Link>
                      </TableCell>
                      <TableCell>
                        <Link
                          to={`/projects/${project.key}`}
                          className="font-medium underline-offset-4 hover:underline"
                        >
                          {project.name}
                        </Link>
                      </TableCell>
                      <TableCell className="text-muted-foreground">
                        {timeAgo(project.created_at)}
                      </TableCell>
                      <TableCell className="text-right">
                        {isActive ? (
                          <span className="inline-flex items-center gap-1 text-xs font-medium text-primary">
                            <Check className="size-3.5" />
                            Active
                          </span>
                        ) : (
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => setActiveProjectKey(project.key)}
                          >
                            Set active
                          </Button>
                        )}
                      </TableCell>
                      <TableCell className="text-right">
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon-sm"
                          aria-label={`Delete ${project.name}`}
                          disabled={isDeleting}
                          onClick={() => {
                            setPendingDeleteKey(project.key);
                            deleteProject.mutate(project);
                          }}
                        >
                          {isDeleting ? (
                            <Loader2 className="size-4 animate-spin" />
                          ) : (
                            <Trash2 className="size-4" />
                          )}
                        </Button>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </AsyncBoundary>
      </DataSurface>

      {systemProjects.length > 0 && (
        <details className="rounded-md border border-border/60 bg-muted/20">
          <summary className="cursor-pointer list-none px-4 py-2 text-sm font-medium text-muted-foreground hover:text-foreground">
            <span className="inline-flex items-center gap-2">
              <Lock className="size-3.5" />
              System buckets (built-in)
            </span>
          </summary>
          <div className="space-y-2 px-4 pt-1 pb-3">
            <p className="text-xs text-muted-foreground">
              Built-in buckets used internally. They cannot be edited or deleted.
            </p>
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-transparent">
                    <TableHead>Code</TableHead>
                    <TableHead>Name</TableHead>
                    <TableHead className="w-32">Created</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {systemProjects.map((project) => (
                    <TableRow key={project.id} className="text-muted-foreground">
                      <TableCell className="font-mono text-xs font-medium">
                        {project.key}
                      </TableCell>
                      <TableCell className="flex items-center gap-2">
                        <span>{project.name}</span>
                        <Badge variant="secondary">Built-in</Badge>
                      </TableCell>
                      <TableCell>{timeAgo(project.created_at)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          </div>
        </details>
      )}

      <CreateProjectDialog open={createOpen} onOpenChange={setCreateOpen} />
    </div>
  );
}
