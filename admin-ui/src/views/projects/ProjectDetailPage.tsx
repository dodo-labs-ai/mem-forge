/**
 * Detail view for a single project.
 *
 * Resolves the `:key` URL segment against the cached `["projects"]` list (no
 * single-project endpoint exists on the wire today) and renders three things:
 *   - the project's identity (name, code, kind) plus a Set active control
 *   - the sources whose memories land in this project
 *   - aggregate counts for what's stored under this project key
 *
 * Bound-source membership matches the rules used by the Sources page grouping:
 *   - `fixed` bindings count when their `project_key` equals this project
 *   - `by_field` bindings count when the resolver has observed at least one
 *     memory in this project; if the resolver has not run yet, the source's
 *     `default` fallback is used so the row still appears for an admin to act
 *     on
 *   - unbound sources are never shown here (they appear in Unmapped)
 */
import { useMemo, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  ArrowLeft,
  Check,
  FolderKanban,
  FolderTree,
  Loader2,
  Pencil,
  Trash2,
} from "lucide-react";
import { resourceClient } from "@/api/client";
import type {
  Project,
  ResolvedProjectsResponse,
  Source,
  SourceResolvedProject,
} from "@/api/types";
import { AsyncBoundary } from "@/components/admin/AsyncBoundary";
import { DataSurface } from "@/components/admin/DataSurface";
import { EmptyState } from "@/components/admin/EmptyState";
import { PageHeader } from "@/components/admin/PageHeader";
import { StatusDot } from "@/components/admin/StatusBadge";
import { SourceIcon } from "@/components/sources/SourceIcon";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { isReservedProjectKey } from "@/api/projectKeys";
import { useActiveProject } from "@/state/activeProject";
import { isManagedSourceId, isManagedSourceType } from "../sources/managedSources";
import { ProjectEditDialog } from "./ProjectEditDialog";

interface SourcesResponse {
  data?: Source[];
}

function normalizeSources(payload: SourcesResponse | Source[] | undefined): Source[] {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.data)) return payload.data;
  return [];
}

interface DeleteProjectWireResponse {
  id: string;
}

interface BoundSourceRow {
  source: Source;
  bindingMode: "fixed" | "by_field";
  bindingLabel: string;
  memoryCount: number;
}

const FIXED_BINDING_LABEL = "Direct";
const BY_FIELD_BINDING_LABEL = "By field";

export function ProjectDetailPage() {
  const params = useParams<{ key: string }>();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const { activeProjectKey, setActiveProjectKey } = useActiveProject();
  const projectKey = params.key ?? "";
  const [editOpen, setEditOpen] = useState(false);
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);

  const projectsQuery = useQuery<Project[]>({
    queryKey: ["projects"],
    queryFn: () =>
      resourceClient.get<Project[]>("/projects").then((response) => response.data),
  });

  const sourcesQuery = useQuery<SourcesResponse | Source[]>({
    queryKey: ["sources"],
    queryFn: () =>
      resourceClient.get("/sources").then((response) => response.data),
  });

  const project = useMemo(() => {
    const projects = projectsQuery.data ?? [];
    return projects.find((candidate) => candidate.key === projectKey) ?? null;
  }, [projectsQuery.data, projectKey]);

  const sources = useMemo(
    () => normalizeSources(sourcesQuery.data),
    [sourcesQuery.data],
  );

  // Sources whose membership in this project hinges on the resolver. Managed
  // (push) sources are excluded because their resolver wiring is owned by the
  // plugin, not by the admin UI.
  const sourcesNeedingResolve = useMemo(
    () =>
      sources.filter(
        (source) =>
          source.project_binding?.mode === "by_field" &&
          !isManagedSourceType(source.type) &&
          !isManagedSourceId(source.id),
      ),
    [sources],
  );

  const resolvedQueries = useQueries({
    queries: sourcesNeedingResolve.map((source) => ({
      queryKey: ["resolvedProjects", source.id],
      queryFn: () =>
        resourceClient
          .get<ResolvedProjectsResponse>(
            `/sources/${source.id}/projects/resolved`,
          )
          .then((response) => response.data),
    })),
  });

  const resolvedBySource = useMemo(() => {
    const map: Record<string, SourceResolvedProject[]> = {};
    sourcesNeedingResolve.forEach((source, index) => {
      const data = resolvedQueries[index]?.data;
      if (data?.projects) {
        map[source.id] = data.projects;
      }
    });
    return map;
  }, [resolvedQueries, sourcesNeedingResolve]);

  const boundRows = useMemo<BoundSourceRow[]>(() => {
    if (!project) return [];
    const rows: BoundSourceRow[] = [];
    for (const source of sources) {
      const binding = source.project_binding;
      if (!binding) continue;
      if (binding.mode === "fixed") {
        if (binding.project_key === project.key) {
          rows.push({
            source,
            bindingMode: "fixed",
            bindingLabel: FIXED_BINDING_LABEL,
            memoryCount: source.memory_count ?? 0,
          });
        }
        continue;
      }
      if (binding.mode === "by_field") {
        const observed = resolvedBySource[source.id];
        if (observed && observed.length > 0) {
          const match = observed.find((row) => row.project_key === project.key);
          if (match) {
            rows.push({
              source,
              bindingMode: "by_field",
              bindingLabel: BY_FIELD_BINDING_LABEL,
              memoryCount: match.memory_count,
            });
          }
          continue;
        }
        // Resolver has not reported yet; the source falls under its default.
        if ((binding.default ?? "") === project.key) {
          rows.push({
            source,
            bindingMode: "by_field",
            bindingLabel: BY_FIELD_BINDING_LABEL,
            memoryCount: 0,
          });
        }
      }
    }
    return rows;
  }, [project, sources, resolvedBySource]);

  const totalProjectMemoryCount = useMemo(
    () => boundRows.reduce((sum, row) => sum + row.memoryCount, 0),
    [boundRows],
  );

  const deleteProject = useMutation({
    mutationFn: async (target: Project) => {
      const response = await resourceClient.delete<DeleteProjectWireResponse>(
        `/projects/${target.id}`,
      );
      return { project: target, payload: response.data };
    },
    onSuccess: ({ project: deleted }) => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
      if (activeProjectKey === deleted.key) {
        setActiveProjectKey(null);
      }
      navigate("/projects");
    },
  });

  const isLoading = projectsQuery.isLoading;
  const isError = projectsQuery.isError;

  if (isLoading || isError) {
    return (
      <AsyncBoundary
        isLoading={isLoading}
        isError={isError}
        error={projectsQuery.error}
        onRetry={() => projectsQuery.refetch()}
        empty={null}
      >
        {null}
      </AsyncBoundary>
    );
  }

  if (!project) {
    return (
      <div className="space-y-4">
        <EmptyState
          icon={FolderKanban}
          title="Project not found"
          description={`No project matches "${projectKey}". It may have been deleted or you may have followed a stale link.`}
        />
        <div className="flex justify-center">
          <Button
            type="button"
            variant="outline"
            onClick={() => navigate("/projects")}
          >
            <ArrowLeft className="size-4" />
            Back to Project setup
          </Button>
        </div>
      </div>
    );
  }

  const isReserved = isReservedProjectKey(project.key);
  const isShared = project.kind === "shared";
  const isActive = activeProjectKey === project.key;
  const isDeleting = deleteProject.isPending;

  return (
    <div className="space-y-4">
      <div>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="-ml-2"
          onClick={() => navigate("/projects")}
        >
          <ArrowLeft className="size-4" />
          Back to Project setup
        </Button>
      </div>

      <PageHeader
        title={project.name}
        description={`Code: ${project.key}`}
        actions={
          <div className="flex items-center gap-2">
            {isShared && <Badge variant="secondary">team-wide</Badge>}
            {isReserved && <Badge variant="outline">Built-in</Badge>}
            {!isReserved &&
              (isActive ? (
                <Button type="button" variant="outline" disabled>
                  <Check className="size-4" />
                  Active
                </Button>
              ) : (
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setActiveProjectKey(project.key)}
                >
                  Set active
                </Button>
              ))}
            {!isReserved && (
              <>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setEditOpen(true)}
                >
                  <Pencil className="size-4" />
                  Edit
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setConfirmDeleteOpen(true)}
                  disabled={isDeleting}
                >
                  {isDeleting ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Trash2 className="size-4" />
                  )}
                  Delete
                </Button>
              </>
            )}
          </div>
        }
      />

      <div className="grid gap-4 md:grid-cols-2">
        <DataSurface>
          <div className="border-b p-4">
            <h2 className="text-base font-semibold">Memory count</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              Total memories that resolved into this project.
            </p>
          </div>
          <div className="p-4">
            <div className="text-3xl font-semibold">
              {totalProjectMemoryCount.toLocaleString()}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Summed across all sources bound to this project.
            </p>
          </div>
        </DataSurface>

        <DataSurface>
          <div className="border-b p-4">
            <h2 className="text-base font-semibold">Sources bound</h2>
            <p className="mt-1 text-sm text-muted-foreground">
              How many sources route memories here.
            </p>
          </div>
          <div className="p-4">
            <div className="text-3xl font-semibold">
              {boundRows.length.toLocaleString()}
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              Includes both direct and field-based bindings.
            </p>
          </div>
        </DataSurface>
      </div>

      <DataSurface>
        <div className="border-b p-4">
          <h2 className="text-base font-semibold">Bound sources</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Sources whose extracted memories land in this project.
          </p>
        </div>
        <AsyncBoundary
          isLoading={sourcesQuery.isLoading}
          isError={sourcesQuery.isError}
          error={sourcesQuery.error}
          onRetry={() => sourcesQuery.refetch()}
          isEmpty={boundRows.length === 0}
          empty={
            <div className="flex flex-col items-center justify-center px-6 py-12 text-center">
              <div className="mb-3 grid size-10 place-items-center rounded-full bg-muted text-muted-foreground">
                <FolderTree className="size-5" />
              </div>
              <h3 className="text-sm font-medium">No sources bound yet</h3>
              <p className="mt-1 max-w-sm text-sm text-muted-foreground">
                Open a source's configuration and pick this project to start
                routing its memories here.
              </p>
              <Button
                type="button"
                className="mt-4"
                variant="outline"
                size="sm"
                onClick={() => navigate("/sources")}
              >
                Go to Sources
              </Button>
            </div>
          }
        >
          <ul className="divide-y">
            {boundRows.map((row) => (
              <li
                key={row.source.id}
                className="flex flex-col gap-2 p-4 sm:flex-row sm:items-center sm:justify-between"
              >
                <div className="flex min-w-0 items-start gap-3">
                  <SourceIcon
                    type={row.source.type}
                    client={row.source.client}
                    className="mt-0.5 size-5"
                  />
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <Link
                        to="/sources"
                        className="truncate text-sm font-medium hover:underline"
                      >
                        {row.source.name}
                      </Link>
                      <StatusDot status={row.source.status} />
                      <Badge variant="outline" className="text-[11px]">
                        {row.bindingLabel}
                      </Badge>
                    </div>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {row.source.type}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-4 text-sm text-muted-foreground sm:shrink-0">
                  <span>
                    <span className="font-medium text-foreground">
                      {row.memoryCount.toLocaleString()}
                    </span>{" "}
                    memories
                  </span>
                </div>
              </li>
            ))}
          </ul>
        </AsyncBoundary>
      </DataSurface>

      {!isReserved && (
        <ProjectEditDialog
          project={project}
          open={editOpen}
          onOpenChange={setEditOpen}
        />
      )}

      {confirmDeleteOpen && (
        <DeleteConfirmDialog
          project={project}
          memoryCount={totalProjectMemoryCount}
          isDeleting={isDeleting}
          onCancel={() => setConfirmDeleteOpen(false)}
          onConfirm={() => {
            setConfirmDeleteOpen(false);
            deleteProject.mutate(project);
          }}
        />
      )}
    </div>
  );
}

function DeleteConfirmDialog({
  project,
  memoryCount,
  isDeleting,
  onCancel,
  onConfirm,
}: {
  project: Project;
  memoryCount: number;
  isDeleting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}) {
  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 p-4"
      role="dialog"
      aria-modal="true"
    >
      <div className="w-full max-w-md rounded-xl bg-card p-6 shadow-lg ring-1 ring-foreground/10">
        <h2 className="text-base font-semibold">Delete project?</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          {memoryCount > 0
            ? `Project "${project.name}" will be deleted and ${memoryCount.toLocaleString()} ${
                memoryCount === 1 ? "memory" : "memories"
              } will move to the Unsorted project.`
            : `Project "${project.name}" will be deleted. No memories are linked to it.`}
        </p>
        <div className="mt-4 flex justify-end gap-2">
          <Button
            type="button"
            variant="outline"
            onClick={onCancel}
            disabled={isDeleting}
          >
            Cancel
          </Button>
          <Button
            type="button"
            variant="destructive"
            onClick={onConfirm}
            disabled={isDeleting}
          >
            {isDeleting && <Loader2 className="size-4 animate-spin" />}
            Delete project
          </Button>
        </div>
      </div>
    </div>
  );
}
