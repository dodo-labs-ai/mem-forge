/**
 * Project binding controls for the Source dialog.
 */
import { useMemo, useState } from "react";
import type { FormEvent } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Loader2, Plus, Trash2 } from "lucide-react";
import client from "@/api/client";
import {
  UNSORTED_PROJECT_KEY,
  isReservedProjectKey,
} from "@/api/projectKeys";
import type {
  GeneConfigSchema,
  Project,
  ProjectBinding,
  ProjectKind,
  SourceProjectsResponse,
} from "@/api/types";
import { Button } from "@/components/ui/button";
import {
  ProjectCreateFields,
} from "@/views/projects/ProjectCreateFields";
import {
  emptyProjectCreateForm,
  projectCreateKeyConflictsWithBuiltIn,
} from "@/views/projects/projectCreateForm";
import type { ProjectCreateFormState } from "@/views/projects/projectCreateForm";

const FIXED_MODE = "fixed" as const;
const BY_FIELD_MODE = "by_field" as const;

type BindingMode = typeof FIXED_MODE | typeof BY_FIELD_MODE;

interface MapRow {
  id: string;
  value: string;
  projectKey: string;
}

function makeRowId(): string {
  return crypto.randomUUID();
}

function toMapRows(map: Record<string, string> | undefined): MapRow[] {
  if (!map) return [];
  return Object.entries(map).map(([value, projectKey]) => ({
    id: makeRowId(),
    value,
    projectKey,
  }));
}

function rowsToMap(rows: MapRow[]): Record<string, string> {
  return rows.reduce<Record<string, string>>((acc, row) => {
    const trimmedValue = row.value.trim();
    if (trimmedValue.length === 0) return acc;
    if (row.projectKey.trim().length === 0) return acc;
    acc[trimmedValue] = row.projectKey;
    return acc;
  }, {});
}

function pickInitialMode(
  binding: ProjectBinding | null,
  byFieldEnabled: boolean,
): BindingMode {
  if (binding?.mode === BY_FIELD_MODE && byFieldEnabled) return BY_FIELD_MODE;
  return FIXED_MODE;
}

function buildBinding(
  mode: BindingMode,
  fixedKey: string,
  defaultKey: string,
  mapRows: MapRow[],
  projectFieldKey: string | null,
): ProjectBinding {
  if (mode === FIXED_MODE) {
    return { mode: FIXED_MODE, project_key: fixedKey };
  }
  return {
    mode: BY_FIELD_MODE,
    field: projectFieldKey ?? "",
    map: rowsToMap(mapRows),
    default: defaultKey,
  };
}

function fieldLabelFromSchema(
  schema: GeneConfigSchema,
  fieldKey: string | null | undefined,
): string {
  if (!fieldKey) return "";
  const match = schema.fields.find((field) => field.key === fieldKey);
  return match?.label ?? fieldKey;
}

export function ProjectBindingFields({
  schema,
  sourceId,
  value,
  onChange,
}: {
  schema: GeneConfigSchema;
  sourceId: string | null;
  value: ProjectBinding | null;
  onChange: (next: ProjectBinding | null) => void;
}) {
  const queryClient = useQueryClient();
  const projectFieldKey = schema.project_field ?? null;
  const byFieldEnabled = Boolean(projectFieldKey);

  const projectsQuery = useQuery<Project[]>({
    queryKey: ["projects"],
    queryFn: () =>
      client.get<Project[]>("/api/projects").then((response) => response.data),
  });
  const projects = useMemo(() => projectsQuery.data ?? [], [projectsQuery.data]);
  const userProjects = useMemo(
    () => projects.filter((project) => !isReservedProjectKey(project.key)),
    [projects],
  );

  const observedQuery = useQuery<SourceProjectsResponse>({
    queryKey: ["source-projects", sourceId],
    queryFn: () => {
      if (!sourceId) throw new Error("sourceId required");
      return client
        .get(`/api/sources/${sourceId}/projects`)
        .then((response) => response.data);
    },
    enabled: Boolean(sourceId) && byFieldEnabled,
  });

  const [mode, setMode] = useState<BindingMode>(() =>
    pickInitialMode(value, byFieldEnabled),
  );
  const [fixedKey, setFixedKey] = useState<string>(
    value?.mode === FIXED_MODE ? value.project_key ?? "" : "",
  );
  const [defaultKey, setDefaultKey] = useState<string>(() => {
    if (value?.mode === BY_FIELD_MODE && value.default) return value.default;
    return UNSORTED_PROJECT_KEY;
  });
  const [mapRows, setMapRows] = useState<MapRow[]>(() =>
    value?.mode === BY_FIELD_MODE ? toMapRows(value.map) : [],
  );
  const [createOpen, setCreateOpen] = useState(false);
  const [createTarget, setCreateTarget] = useState<
    | { kind: "fixed" }
    | { kind: "default" }
    | { kind: "row"; rowId: string }
    | null
  >(null);

  // Keep parent form state in sync with the latest local project binding.
  const emit = (
    nextMode: BindingMode = mode,
    nextFixedKey: string = fixedKey,
    nextDefaultKey: string = defaultKey,
    nextMapRows: MapRow[] = mapRows,
  ) => {
    onChange(
      buildBinding(
        nextMode,
        nextFixedKey,
        nextDefaultKey,
        nextMapRows,
        projectFieldKey,
      ),
    );
  };

  const chooseMode = (nextMode: BindingMode) => {
    setMode(nextMode);
    emit(nextMode);
  };

  const chooseUnmapped = () => {
    setFixedKey("");
    setDefaultKey(UNSORTED_PROJECT_KEY);
    setMapRows([]);
    onChange(null);
  };

  const chooseFixedKey = (nextFixedKey: string) => {
    setFixedKey(nextFixedKey);
    emit(mode, nextFixedKey);
  };

  const chooseDefaultKey = (nextDefaultKey: string) => {
    setDefaultKey(nextDefaultKey);
    emit(mode, fixedKey, nextDefaultKey);
  };

  const replaceMapRows = (nextMapRows: MapRow[]) => {
    setMapRows(nextMapRows);
    emit(mode, fixedKey, defaultKey, nextMapRows);
  };

  const addRow = () => {
    replaceMapRows([
      ...mapRows,
      { id: makeRowId(), value: "", projectKey: "" },
    ]);
  };

  const removeRow = (rowId: string) => {
    replaceMapRows(mapRows.filter((row) => row.id !== rowId));
  };

  const updateRow = (rowId: string, patch: Partial<MapRow>) => {
    replaceMapRows(
      mapRows.map((row) => (row.id === rowId ? { ...row, ...patch } : row)),
    );
  };

  const seedFromObserved = () => {
    const observed = observedQuery.data?.projects ?? [];
    if (observed.length === 0) return;
    const existingValues = new Set(mapRows.map((row) => row.value.trim()));
    const additions = observed
      .filter((row) => !existingValues.has(row.project))
      .map((row) => ({
        id: makeRowId(),
        value: row.project,
        projectKey: "",
      }));
    if (additions.length > 0) {
      replaceMapRows([...mapRows, ...additions]);
    }
  };

  const openCreate = (
    target:
      | { kind: "fixed" }
      | { kind: "default" }
      | { kind: "row"; rowId: string },
  ) => {
    setCreateTarget(target);
    setCreateOpen(true);
  };

  const handleProjectCreated = (project: Project) => {
    if (!createTarget) return;
    if (createTarget.kind === "fixed") {
      chooseFixedKey(project.key);
    } else if (createTarget.kind === "default") {
      chooseDefaultKey(project.key);
    } else {
      updateRow(createTarget.rowId, { projectKey: project.key });
    }
    setCreateTarget(null);
    queryClient.invalidateQueries({ queryKey: ["projects"] });
  };

  return (
    <section className="space-y-3">
      <header className="space-y-1">
        <h3 className="text-sm font-semibold">Where does this source land?</h3>
        <p className="text-xs text-muted-foreground">
          Leave it unmapped to keep memories searchable in Unsorted, or bind it
          to a project now.
        </p>
      </header>

      <div className="flex flex-wrap items-center gap-2">
        <div className="inline-flex rounded-md border border-border bg-muted/30 p-0.5 text-xs">
          <ModeButton
            active={value === null}
            onClick={chooseUnmapped}
            label="Unmapped"
          />
          <ModeButton
            active={value !== null && mode === FIXED_MODE}
            onClick={() => chooseMode(FIXED_MODE)}
            label="One project"
          />
          {byFieldEnabled ? (
            <ModeButton
              active={value !== null && mode === BY_FIELD_MODE}
              onClick={() => chooseMode(BY_FIELD_MODE)}
              label="Map by field"
            />
          ) : null}
        </div>
      </div>

      {value === null ? (
        <div className="rounded-md border border-border/60 bg-muted/20 px-3 py-2 text-xs text-muted-foreground">
          This source is not assigned to a project yet. New memories land in
          Unsorted and remain visible in the all-project memory view.
        </div>
      ) : mode === FIXED_MODE ? (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <ProjectPicker
              projects={userProjects}
              value={fixedKey}
              onChange={chooseFixedKey}
              loading={projectsQuery.isLoading}
              placeholder="Pick a project"
              includeShared
            />
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => openCreate({ kind: "fixed" })}
            >
              <Plus className="size-3.5" />
              New
            </Button>
          </div>
        </div>
      ) : (
        <div className="space-y-3">
          <div className="rounded-md border border-border/60 bg-muted/20 px-3 py-2 text-xs">
            <span className="font-medium text-foreground">
              Maps {fieldLabelFromSchema(schema, projectFieldKey)}
            </span>
            <p className="mt-1 text-muted-foreground">
              Each document's {fieldLabelFromSchema(schema, projectFieldKey)} is
              looked up below. Values not in the table fall through to the
              default project.
            </p>
          </div>

          <div className="space-y-1.5">
            <span className="text-xs font-medium text-muted-foreground">
              Default project
              <span className="text-destructive"> *</span>
            </span>
            <div className="flex items-center gap-2">
              <ProjectPicker
                projects={userProjects}
                value={defaultKey}
                onChange={chooseDefaultKey}
                loading={projectsQuery.isLoading}
                placeholder="Pick a default"
                includeShared
                includeUnsorted
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => openCreate({ kind: "default" })}
              >
                <Plus className="size-3.5" />
                New
              </Button>
            </div>
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-xs font-medium text-muted-foreground">
                Value to project map
              </span>
              <div className="flex gap-2">
                {sourceId && (observedQuery.data?.projects.length ?? 0) > 0 && (
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={seedFromObserved}
                  >
                    Seed from observed
                  </Button>
                )}
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  onClick={addRow}
                >
                  <Plus className="size-3.5" />
                  Add row
                </Button>
              </div>
            </div>
            {mapRows.length === 0 ? (
              <p className="text-xs text-muted-foreground">
                No mappings yet. Unmapped values fall through to the default
                project.
              </p>
            ) : (
              <ul className="space-y-1.5">
                {mapRows.map((row) => (
                  <li key={row.id} className="flex items-center gap-2">
                    <input
                      type="text"
                      className="h-8 flex-1 rounded-md border border-input bg-background px-2.5 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
                      value={row.value}
                      placeholder={fieldLabelFromSchema(schema, projectFieldKey) || "Value"}
                      onChange={(event) =>
                        updateRow(row.id, { value: event.target.value })
                      }
                    />
                    <span className="text-xs text-muted-foreground">to</span>
                    <ProjectPicker
                      projects={userProjects}
                      value={row.projectKey}
                      onChange={(next) =>
                        updateRow(row.id, { projectKey: next })
                      }
                      loading={projectsQuery.isLoading}
                      placeholder="Project"
                      includeShared
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="icon-sm"
                      aria-label="Remove row"
                      onClick={() => removeRow(row.id)}
                    >
                      <Trash2 className="size-4" />
                    </Button>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}

      <InlineCreateProjectDialog
        open={createOpen}
        onOpenChange={(next) => {
          setCreateOpen(next);
          if (!next) setCreateTarget(null);
        }}
        onCreated={handleProjectCreated}
      />
    </section>
  );
}

function ModeButton({
  active,
  onClick,
  label,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={
        active
          ? "rounded-sm bg-background px-2.5 py-1 font-medium text-foreground shadow-xs"
          : "rounded-sm px-2.5 py-1 text-muted-foreground hover:text-foreground"
      }
    >
      {label}
    </button>
  );
}

function ProjectPicker({
  projects,
  value,
  onChange,
  loading,
  placeholder,
  includeShared = false,
  includeUnsorted = false,
}: {
  projects: Project[];
  value: string;
  onChange: (next: string) => void;
  loading: boolean;
  placeholder: string;
  includeShared?: boolean;
  includeUnsorted?: boolean;
}) {
  const sharedProjects = useMemo(
    () => (includeShared ? projects.filter((project) => project.kind === "shared") : []),
    [projects, includeShared],
  );

  const items = useMemo(() => {
    const list: Project[] = [];
    list.push(...sharedProjects);
    list.push(
      ...projects.filter(
        (project) => !sharedProjects.includes(project) && !isReservedProjectKey(project.key),
      ),
    );
    if (includeUnsorted) {
      const unsorted = projects.find(
        (project) => project.key === UNSORTED_PROJECT_KEY,
      );
      if (unsorted) list.push(unsorted);
    }
    return list;
  }, [projects, sharedProjects, includeUnsorted]);

  return (
    <select
      value={value}
      onChange={(event) => onChange(event.target.value)}
      disabled={loading}
      className="h-8 min-w-0 flex-1 rounded-md border border-input bg-background px-3 text-sm text-foreground shadow-xs outline-none transition-colors focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:opacity-50"
    >
      <option value="" disabled>
        {loading ? "Loading projects..." : placeholder}
      </option>
      {items.map((project) => (
        <option key={project.id} value={project.key}>
          {project.name}
          {project.kind === "shared" ? " (team-wide)" : ""}
          {project.key === UNSORTED_PROJECT_KEY ? " (built-in)" : ""}
        </option>
      ))}
    </select>
  );
}

function InlineCreateProjectDialog({
  open,
  onOpenChange,
  onCreated,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCreated: (project: Project) => void;
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
      const response = await client.post<Project>("/api/projects", body);
      return response.data;
    },
    onSuccess: (project) => {
      queryClient.invalidateQueries({ queryKey: ["projects"] });
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      onCreated(project);
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

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4"
      role="dialog"
      aria-modal="true"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onOpenChange(false);
      }}
    >
      <div className="w-full max-w-md rounded-xl border bg-background p-4 shadow-xl">
        <h4 className="text-base font-semibold">New project</h4>
        <p className="mt-1 text-xs text-muted-foreground">
          Memories from this source will land here.
        </p>
        <form onSubmit={submit} className="mt-3 space-y-4">
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
          <div className="flex justify-end gap-2">
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
          </div>
        </form>
      </div>
    </div>
  );
}

function extractApiErrorDetail(error: unknown): string | null {
  if (typeof error !== "object" || error === null) return null;
  const candidate = error as { response?: { data?: { detail?: unknown } } };
  const detail = candidate.response?.data?.detail;
  return typeof detail === "string" ? detail : null;
}
