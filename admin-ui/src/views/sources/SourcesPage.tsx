import { type CSSProperties, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { Files, Info, Loader2, LockKeyhole, MoreHorizontal, Pause, Pin, Play, Plus, RefreshCw, Search, Trash2, X } from "lucide-react";
import { resourceClient } from "@/api/client";
import { createLocalAgentJob, getCurrentLocalAgentJobs, getLocalAgentJob } from "@/api/localAgentJobs";
import type {
  AgentSessionCompleteness,
  GeneMetadata,
  LocalAgentJobCreateResponse,
  LocalAgentJobStatusResponse,
  Project,
  ResolvedProjectsResponse,
  Source,
  SourceCapabilities,
  SourceProjectsResponse,
} from "@/api/types";
import { AsyncBoundary } from "@/components/admin/AsyncBoundary";
import { DataSurface } from "@/components/admin/DataSurface";
import { EmptyState } from "@/components/admin/EmptyState";
import { PageHeader } from "@/components/admin/PageHeader";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { timeAgo } from "@/utils/date";
import { SourceIcon } from "@/components/sources/SourceIcon";
import { SourceSetupDialog } from "./SourceSetupDialog";
import { SourceAccessChangeDialog } from "./SourceAccessChangeDialog";
import { LocalAgentDaemonStatus } from "./LocalAgentDaemonStatus";
import { isManagedSourceId, isManagedSourceType, userConfigurableGenes } from "./managedSources";
import { getSourceActionEndpoint, getSourceMenuStyle, sourceActionLayout } from "./sourceActions";
import { ProjectGroup } from "./ProjectGroup";
import {
  PROJECT_GROUPS_DEFAULT_EXPANDED,
  groupSourcesByProject,
  projectGroupKey,
  type ResolvedBySource,
} from "./projectGrouping";
import { SourceRow } from "./SourceRow";
import {
  selectSourceSyncActivity,
  sourceSyncActivityBlocksActions,
} from "./sourceSyncActivity";
import { organizeSourceGroups, type SourceListSortMode } from "./sourceListOrganization";
import { localAgentSyncOperation } from "./localAgentSources";
import {
  presentSourceConnection,
  type SourceConnectionMode,
} from "./sourceConnectionPresentation";

const SOURCE_LABELS: Record<string, { name: string; subtitle: string; description: string }> = {
  // Per-client agent-session sources returned by the split backend.
  "src-agent-sessions-codex": { name: "Codex Session", subtitle: "Managed source", description: "Coding-agent session summaries from the Codex plugin" },
  "src-agent-sessions-claude-code": { name: "Claude Code Session", subtitle: "Managed source", description: "Coding-agent session summaries from the Claude Code plugin" },
  // Legacy / fallback entry used when the backend returns the singleton type.
  agent_session: { name: "Agent Session", subtitle: "Managed source", description: "Generated coding-agent session summaries" },
  confluence: { name: "Confluence", subtitle: "Knowledge source", description: "Wiki pages and documentation" },
  github_pages: { name: "GitHub Pages", subtitle: "Documentation source", description: "Published documentation pages" },
  jira: { name: "Jira", subtitle: "Work tracking source", description: "Tickets, decisions, and work items" },
  local_markdown: { name: "Local Repository", subtitle: "Local folder source", description: "Sync files through your local daemon." },
  teams: { name: "Microsoft Teams", subtitle: "Conversation source", description: "Channel messages, group chats, and direct messages" },
};

const SOURCE_ITEM_LABELS: Record<string, string> = {
  "src-agent-sessions-codex": "summaries",
  "src-agent-sessions-claude-code": "summaries",
  agent_session: "summaries",
  confluence: "pages",
  github_pages: "documents",
  jira: "issues",
  local_markdown: "files",
  teams: "conversations",
};

interface SourcesResponse {
  data?: Source[];
}

interface SourceListPreferencesResponse {
  sort_mode: SourceListSortMode;
}

function normalizeSources(payload: SourcesResponse | Source[] | undefined): Source[] {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.data)) return payload.data;
  return [];
}

function localAgentJobPayload(forceFullSync = false): Record<string, unknown> {
  return {
    force_full_sync: forceFullSync,
  };
}

const LOCAL_AGENT_SYNC_POLL_ATTEMPTS = 1_800;
const LOCAL_AGENT_SYNC_POLL_INTERVAL_MS = 2_000;
const LOCAL_AGENT_WAITING_MESSAGE = "Waiting for local daemon to sync this source.";
const LOCAL_AGENT_TIMEOUT_MESSAGE =
  "Local daemon did not pick up this job. Start it with `memforge adapter daemon run` and try again.";
const LOCAL_AGENT_CONFIGURE_FOLDER_MESSAGE = "Configure a folder path before syncing this local source.";
const LOCAL_AGENT_SYNC_FAILED_MESSAGE = "Local daemon could not sync this source.";
const LOCAL_AGENT_TERMINAL_PROGRESS_RETENTION_MS = 30_000;

function safeSourceErrorMessage(error: unknown): string | null {
  if (!(error instanceof Error)) return null;
  if (
    error.message === LOCAL_AGENT_WAITING_MESSAGE
    || error.message === LOCAL_AGENT_TIMEOUT_MESSAGE
    || error.message === LOCAL_AGENT_CONFIGURE_FOLDER_MESSAGE
    || error.message === LOCAL_AGENT_SYNC_FAILED_MESSAGE
    || error.message.startsWith(`${LOCAL_AGENT_SYNC_FAILED_MESSAGE} `)
  ) {
    return error.message;
  }
  return null;
}

function localAgentJobErrorMessage(status: LocalAgentJobStatusResponse): string {
  const result = status.result as { error?: unknown } | null;
  const detail = typeof result?.error === "string" && result.error.trim()
    ? result.error.trim()
    : status.last_error?.trim();
  if (!detail) return LOCAL_AGENT_SYNC_FAILED_MESSAGE;
  return `${LOCAL_AGENT_SYNC_FAILED_MESSAGE} ${cleanLocalAgentJobError(detail)}`;
}

function cleanLocalAgentJobError(value: string): string {
  const text = value.trim();
  const normalized = text.toLowerCase();
  if (normalized.includes("teams") && (
    normalized.includes("session expired")
    || normalized.includes("no teams session")
    || normalized.includes("tokens")
    || normalized.includes("sign in")
  )) {
    return "Sign in to Teams in Chrome, then retry sync.";
  }
  return text;
}

async function createLocalAgentSyncJob(
  source: Source,
  options: {
    onStatus?: (status: LocalAgentJobStatusResponse) => void;
    forceFullSync?: boolean;
  },
): Promise<LocalAgentJobCreateResponse | null> {
  const operation = localAgentSyncOperation(source);
  if (!operation) {
    if (source.type === "local_markdown") {
      throw new Error(LOCAL_AGENT_CONFIGURE_FOLDER_MESSAGE);
    }
    return null;
  }
  const created = await createLocalAgentJob({
    sourceId: source.id,
    sourceType: source.type,
    operation,
    payload: localAgentJobPayload(options.forceFullSync),
  });
  options.onStatus?.({
    job_id: created.job_id,
    operation,
    status: "queued",
    result: null,
    last_error: null,
  });
  const status = await pollLocalAgentSyncJob(created.job_id, options);
  if (status.status === "failed") {
    if (status.last_error) {
      console.warn("Local daemon sync failed", status.last_error);
    }
    throw new Error(localAgentJobErrorMessage(status));
  }
  return created;
}

async function pollLocalAgentSyncJob(
  jobId: string,
  options: {
    onStatus?: (status: LocalAgentJobStatusResponse) => void;
  },
): Promise<LocalAgentJobStatusResponse> {
  for (let attempt = 0; attempt < LOCAL_AGENT_SYNC_POLL_ATTEMPTS; attempt += 1) {
    const status = await getLocalAgentJob(jobId);
    options.onStatus?.(status);
    if (status.status === "succeeded" || status.status === "failed") {
      return status;
    }
    await new Promise((resolve) => window.setTimeout(resolve, LOCAL_AGENT_SYNC_POLL_INTERVAL_MS));
  }
  throw new Error(LOCAL_AGENT_TIMEOUT_MESSAGE);
}

function sourceItemLabel(source: Source): string {
  return SOURCE_ITEM_LABELS[source.id] ?? SOURCE_ITEM_LABELS[source.type] ?? "documents";
}

export function SourcesPage() {
  const queryClient = useQueryClient();
  const [addOpen, setAddOpen] = useState(false);
  const [configDialog, setConfigDialog] = useState<{
    sourceType: string | null;
    source?: Source | null;
    initialFocus?: { step: "project" };
  }>({ sourceType: null, source: null });
  const [detailsSource, setDetailsSource] = useState<Source | null>(null);
  const [accessSource, setAccessSource] = useState<Source | null>(null);
  const [openMenuSourceId, setOpenMenuSourceId] = useState<string | null>(null);
  const [sourcePendingDelete, setSourcePendingDelete] = useState<Source | null>(null);
  const [pendingSyncIds, setPendingSyncIds] = useState<Set<string>>(new Set());
  const [localAgentJobBySource, setLocalAgentJobBySource] = useState<
    Record<string, LocalAgentJobStatusResponse | undefined>
  >({});
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(() => new Set());
  const [pendingSubscriptionIds, setPendingSubscriptionIds] = useState<Set<string>>(
    () => new Set(),
  );
  const [authorityMessage, setAuthorityMessage] = useState<string | null>(null);
  const [sourceSearch, setSourceSearch] = useState("");
  const [pinnedOnly, setPinnedOnly] = useState(false);
  const [newSourceId, setNewSourceId] = useState<string | null>(null);

  const handleAuthorityError = (error: unknown, fallback: string) => {
    if (isForbiddenError(error)) {
      setAuthorityMessage(
        "You no longer have permission to manage this source. The list has been refreshed.",
      );
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      return true;
    }
    setAuthorityMessage(safeSourceErrorMessage(error) ?? fallback);
    return false;
  };

  const setLocalAgentJob = (sourceId: string, job: LocalAgentJobStatusResponse) => {
    setLocalAgentJobBySource((current) => ({ ...current, [sourceId]: job }));
  };

  const clearLocalAgentJob = (sourceId: string) => {
    setLocalAgentJobBySource((current) => {
      if (!current[sourceId]) return current;
      const next = { ...current };
      delete next[sourceId];
      return next;
    });
  };

  const genesQuery = useQuery<GeneMetadata[]>({
    queryKey: ["genes"],
    queryFn: () => resourceClient.get("/genes").then((response) => response.data),
  });

  const sourcesQuery = useQuery<SourcesResponse | Source[]>({
    queryKey: ["sources"],
    queryFn: () => resourceClient.get("/sources").then((response) => response.data),
    refetchInterval: (query) => {
      const sources = normalizeSources(query.state.data);
      const terminal = new Set(["success", "partial", "failed"]);
      return sources.some((source) => {
        if (source.access_state === "changing" && source.access_transition?.status !== "failed") {
          return true;
        }
        if (["queued", "running"].includes(source.lifecycle_maintenance?.status ?? "")) {
          return true;
        }
        const status = source.sync?.status;
        return Boolean(status && !terminal.has(status));
      }) ? 2000 : false;
    },
  });

  const sourceListPreferencesQuery = useQuery<SourceListPreferencesResponse>({
    queryKey: ["source-list-preferences"],
    queryFn: () => resourceClient.get("/source-list/preferences").then((response) => response.data),
  });

  const setSourceListSort = useMutation({
    mutationFn: (sortMode: SourceListSortMode) =>
      resourceClient.put("/source-list/preferences", { sort_mode: sortMode }),
    onSuccess: (_data, sortMode) => {
      queryClient.setQueryData(["source-list-preferences"], { sort_mode: sortMode });
    },
    onError: (error) => handleAuthorityError(error, "Failed to save Source List sorting."),
  });

  const setSourcePin = useMutation({
    mutationFn: ({ sourceId, pinned }: { sourceId: string; pinned: boolean }) =>
      pinned
        ? resourceClient.put(`/sources/${sourceId}/pin`)
        : resourceClient.delete(`/sources/${sourceId}/pin`),
    onError: (error) => handleAuthorityError(error, "Failed to update the pinned source."),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["sources"] }),
  });

  const currentLocalJobsQuery = useQuery<LocalAgentJobStatusResponse[]>({
    queryKey: ["currentLocalAgentJobs"],
    queryFn: getCurrentLocalAgentJobs,
    refetchInterval: (query) => query.state.data?.some((job) => ["queued", "leased"].includes(job.status)) ? 2_000 : false,
  });

  const currentLocalJobBySource = Object.fromEntries(
    (currentLocalJobsQuery.data ?? [])
      .filter((job): job is LocalAgentJobStatusResponse & { source_id: string } => Boolean(job.source_id))
      .map((job) => [job.source_id, job]),
  );

  const projectsQuery = useQuery<Project[]>({
    queryKey: ["projects"],
    queryFn: () => resourceClient.get("/projects").then((response) => response.data),
  });

  const syncSource = useMutation({
    mutationFn: async ({ source, forceFullSync = false }: { source: Source; forceFullSync?: boolean }) => {
      const sourceId = source.id;
      setPendingSyncIds((current) => new Set(current).add(sourceId));
      const localAgentJob = await createLocalAgentSyncJob(source, {
        onStatus: (status) => setLocalAgentJob(sourceId, status),
        forceFullSync,
      });
      if (localAgentJob) {
        return { data: localAgentJob };
      }
      return resourceClient.post(`/sources/${sourceId}/sync`, { force_full_sync: forceFullSync });
    },
    onError: (error, variables) => {
      setPendingSyncIds((current) => {
        const next = new Set(current);
        next.delete(variables.source.id);
        return next;
      });
      if (localAgentSyncOperation(variables.source)) {
        setLocalAgentJob(variables.source.id, {
          job_id: `failed:${variables.source.id}`,
          status: "failed",
          result: { error: safeSourceErrorMessage(error) ?? LOCAL_AGENT_SYNC_FAILED_MESSAGE },
          last_error: null,
        });
      }
      handleAuthorityError(error, "Failed to start sync.");
    },
    onSettled: async (_data, _error, variables) => {
      if (!_error && localAgentSyncOperation(variables.source)) {
        window.setTimeout(
          () => clearLocalAgentJob(variables.source.id),
          LOCAL_AGENT_TERMINAL_PROGRESS_RETENTION_MS,
        );
      } else if (!localAgentSyncOperation(variables.source)) {
        clearLocalAgentJob(variables.source.id);
      }
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["sources"] }),
        queryClient.invalidateQueries({ queryKey: ["currentLocalAgentJobs"] }),
        queryClient.invalidateQueries({ queryKey: ["stats"] }),
      ]);
      setPendingSyncIds((current) => {
        const next = new Set(current);
        next.delete(variables.source.id);
        return next;
      });
    },
  });

  const deleteSource = useMutation({
    mutationFn: (sourceId: string) => resourceClient.delete(getSourceActionEndpoint(sourceId, "delete")),
    onError: (error) => handleAuthorityError(error, "Failed to delete source."),
    onSuccess: () => {
      setSourcePendingDelete(null);
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["currentLocalAgentJobs"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
    },
  });

  const forceResyncSource = useMutation({
    mutationFn: async (source: Source) => {
      setPendingSyncIds((current) => new Set(current).add(source.id));
      const localAgentJob = await createLocalAgentSyncJob(source, {
        onStatus: (status) => setLocalAgentJob(source.id, status),
        forceFullSync: true,
      });
      if (localAgentJob) {
        return { data: localAgentJob };
      }
      return resourceClient.post(getSourceActionEndpoint(source.id, "force-resync"));
    },
    onError: (error, source) => {
      setPendingSyncIds((current) => {
        const next = new Set(current);
        next.delete(source.id);
        return next;
      });
      if (localAgentSyncOperation(source)) {
        setLocalAgentJob(source.id, {
          job_id: `failed:${source.id}`,
          status: "failed",
          result: { error: safeSourceErrorMessage(error) ?? LOCAL_AGENT_SYNC_FAILED_MESSAGE },
          last_error: null,
        });
      }
      handleAuthorityError(error, "Failed to start refresh.");
    },
    onSettled: async (_data, _error, source) => {
      if (!_error && localAgentSyncOperation(source)) {
        window.setTimeout(
          () => clearLocalAgentJob(source.id),
          LOCAL_AGENT_TERMINAL_PROGRESS_RETENTION_MS,
        );
      } else if (!localAgentSyncOperation(source)) {
        clearLocalAgentJob(source.id);
      }
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["sources"] }),
        queryClient.invalidateQueries({ queryKey: ["stats"] }),
      ]);
      setPendingSyncIds((current) => {
        const next = new Set(current);
        next.delete(source.id);
        return next;
      });
    },
  });

  const setSourceStatus = useMutation({
    mutationFn: ({ sourceId, status }: { sourceId: string; status: Source["status"] }) =>
      resourceClient.put(`/sources/${sourceId}`, { status }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const setSubscription = useMutation({
    mutationFn: ({ sourceId, enabled }: { sourceId: string; enabled: boolean }) => {
      setPendingSubscriptionIds((current) => new Set(current).add(sourceId));
      return resourceClient.put(`/sources/${sourceId}/subscription`, { enabled });
    },
    onError: (error) => handleAuthorityError(error, "Failed to update subscription."),
    onSettled: (_data, _error, variables) => {
      setPendingSubscriptionIds((current) => {
        const next = new Set(current);
        next.delete(variables.sourceId);
        return next;
      });
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
    },
  });

  const retrySourceAccess = useMutation({
    mutationFn: (source: Source) => {
      const operationId = source.access_transition?.operation_id;
      if (!operationId) throw new Error("No access operation is available to retry.");
      return resourceClient.post(`/sources/${source.id}/access-transitions/${operationId}/retry`);
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["sources"] }),
  });

  const revertSourceAccess = useMutation({
    mutationFn: (source: Source) => {
      const operationId = source.access_transition?.operation_id;
      if (!operationId) throw new Error("No access operation is available to revert.");
      return resourceClient.post(`/sources/${source.id}/access-transitions/${operationId}/revert`);
    },
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["sources"] }),
  });

  const sources = normalizeSources(sourcesQuery.data);
  const genes = genesQuery.data ?? [];
  const geneByName = new Map(genes.map((gene) => [gene.name, gene]));
  const totalDocs = sources.reduce((sum, source) => sum + source.doc_count, 0);
  const totalMemories = sources.reduce((sum, source) => sum + (source.memory_count ?? 0), 0);
  const projects = projectsQuery.data ?? [];

  // Sources whose project assignment depends on per-document field values
  // need the resolver result to know which groups they appear in. Sources
  // pinned to a single project (or unbound, or managed) don't.
  const sourcesNeedingResolve = sources.filter(
    (source) =>
      source.project_binding?.mode === "by_field" &&
      !isManagedSourceType(source.type) &&
      !isManagedSourceId(source.id),
  );

  const resolvedQueries = useQueries({
    queries: sourcesNeedingResolve.map((source) => ({
      queryKey: ["resolvedProjects", source.id],
      queryFn: () =>
        resourceClient.get<ResolvedProjectsResponse>(`/sources/${source.id}/projects/resolved`)
          .then((response) => response.data),
    })),
  });

  const resolvedBySource: ResolvedBySource = {};
  sourcesNeedingResolve.forEach((source, index) => {
    const data = resolvedQueries[index]?.data;
    if (data?.projects) {
      resolvedBySource[source.id] = data.projects;
    }
  });

  const sourceTypeLabels = Object.fromEntries(
    Object.entries(SOURCE_LABELS).map(([type, label]) => [type, label.name]),
  );
  const baseGroups = groupSourcesByProject(sources, projects, resolvedBySource);
  const groups = organizeSourceGroups(
    baseGroups,
    {
      query: sourceSearch,
      pinnedOnly,
      sortMode: sourceListPreferencesQuery.data?.sort_mode ?? "newest",
      typeLabels: sourceTypeLabels,
    },
  );
  const visibleSourceIds = new Set(groups.flatMap((group) => group.sources.map(({ source }) => source.id)));
  const visibleSourceCount = visibleSourceIds.size;
  const pinnedSourceCount = sources.filter((source) => source.pinned_for_me).length;
  const hasOrganizationFilter = sourceSearch.trim().length > 0 || pinnedOnly;
  const newSourceGroup = newSourceId
    ? groups.find((group) => group.sources.some(({ source }) => source.id === newSourceId))
    : undefined;
  const newSourceGroupKey = newSourceGroup ? projectGroupKey(newSourceGroup) : null;

  useEffect(() => {
    if (!newSourceId || !sources.some((source) => source.id === newSourceId)) return;
    const frame = window.requestAnimationFrame(() => {
      if (newSourceGroupKey && collapsedGroups.has(newSourceGroupKey)) {
        setCollapsedGroups((current) => {
          const next = new Set(current);
          next.delete(newSourceGroupKey);
          return next;
        });
        return;
      }
      const row = document.getElementById(`source-row-${newSourceId}`);
      row?.scrollIntoView({ behavior: "smooth", block: "center" });
      row?.focus({ preventScroll: true });
    });
    const timeout = window.setTimeout(() => setNewSourceId(null), 3_000);
    return () => {
      window.cancelAnimationFrame(frame);
      window.clearTimeout(timeout);
    };
  }, [collapsedGroups, newSourceGroupKey, newSourceId, sources]);

  const allInUnmapped =
    sources.length > 0 && baseGroups.length === 1 && baseGroups[0].project === null;

  const isGroupExpanded = (group: typeof groups[number]) => {
    const key = projectGroupKey(group);
    return PROJECT_GROUPS_DEFAULT_EXPANDED ? !collapsedGroups.has(key) : collapsedGroups.has(key);
  };

  const toggleGroup = (group: typeof groups[number]) => {
    const key = projectGroupKey(group);
    setCollapsedGroups((current) => {
      const next = new Set(current);
      if (next.has(key)) {
        next.delete(key);
      } else {
        next.add(key);
      }
      return next;
    });
  };

  return (
    <div className="space-y-4">
      <PageHeader
        title="Sources"
        description="Configure ingestion sources and run syncs."
        actions={
          <Button type="button" onClick={() => setAddOpen(true)}>
            <Plus className="size-4" />
            Add Source
          </Button>
        }
      />

      <div className="grid gap-4 md:grid-cols-3">
        <StatCard label="Configured sources" value={sources.length.toLocaleString()} />
        <StatCard label="Source items synced" value={totalDocs.toLocaleString()} />
        <StatCard label="Memories extracted" value={totalMemories.toLocaleString()} />
      </div>

      <DataSurface>
        <div className="border-b p-4">
          <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <h2 className="text-base font-semibold">Source List</h2>
              <p className="mt-1 text-sm text-muted-foreground">
                {hasOrganizationFilter
                  ? `${visibleSourceCount.toLocaleString()} of ${sources.length.toLocaleString()} sources`
                  : `${sources.length.toLocaleString()} configured ingestion sources.`}
              </p>
            </div>
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <label className="relative min-w-0 sm:w-64">
                <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <input
                  type="search"
                  value={sourceSearch}
                  onChange={(event) => setSourceSearch(event.target.value)}
                  placeholder="Search sources"
                  aria-label="Search sources"
                  className="h-7 w-full rounded-[min(var(--radius-md),12px)] border bg-background pl-8 pr-2.5 text-[0.8rem] outline-none focus-visible:ring-2 focus-visible:ring-ring"
                />
              </label>
              <Select<SourceListSortMode>
                value={sourceListPreferencesQuery.data?.sort_mode ?? "newest"}
                disabled={sourceListPreferencesQuery.isLoading || setSourceListSort.isPending}
                onValueChange={(value) => value && setSourceListSort.mutate(value)}
              >
                <SelectTrigger aria-label="Sort sources" className="h-7 w-full text-[0.8rem] sm:w-44">
                  <SelectValue>
                    {sourceListPreferencesQuery.data?.sort_mode === "name"
                      ? "Name"
                      : sourceListPreferencesQuery.data?.sort_mode === "recently_synced"
                        ? "Recently synced"
                        : "Newest added"}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="newest">Newest added</SelectItem>
                  <SelectItem value="name">Name</SelectItem>
                  <SelectItem value="recently_synced">Recently synced</SelectItem>
                </SelectContent>
              </Select>
              <Button
                type="button"
                variant={pinnedOnly ? "secondary" : "outline"}
                size="sm"
                aria-pressed={pinnedOnly}
                onClick={() => setPinnedOnly((current) => !current)}
              >
                <Pin className="size-4" />
                Pinned {pinnedSourceCount}
              </Button>
            </div>
          </div>
        </div>
        {authorityMessage && (
          <div
            role="status"
            aria-live="polite"
            className="flex items-start justify-between gap-3 border-b bg-amber-50/60 px-4 py-2 text-sm text-amber-900 dark:bg-amber-900/20 dark:text-amber-100"
          >
            <span>{authorityMessage}</span>
            <Button
              type="button"
              size="icon-sm"
              variant="ghost"
              aria-label="Dismiss notice"
              onClick={() => setAuthorityMessage(null)}
            >
              <X className="size-4" />
            </Button>
          </div>
        )}
        <AsyncBoundary
          isLoading={sourcesQuery.isLoading}
          isError={sourcesQuery.isError}
          error={sourcesQuery.error}
          onRetry={() => sourcesQuery.refetch()}
          isEmpty={sources.length === 0}
          empty={
            <EmptyState
              icon={Files}
              title="No sources connected"
              description="Connect a source to start extracting memories."
            />
          }
        >
          <div>
            {sources.length > 0 && groups.length === 0 && (
              <div className="flex flex-col items-center justify-center px-6 py-16 text-center">
                <div className="mb-3 grid size-10 place-items-center rounded-full bg-muted text-muted-foreground">
                  <Search className="size-5" />
                </div>
                <h3 className="text-sm font-medium">No matching sources</h3>
                <p className="mt-1 max-w-sm text-sm text-muted-foreground">
                  Search by source name, type, or project.
                </p>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="mt-4"
                  onClick={() => {
                    setSourceSearch("");
                    setPinnedOnly(false);
                  }}
                >
                  Show all sources
                </Button>
              </div>
            )}
            {allInUnmapped && (
              <div className="border-b bg-amber-50/60 px-4 py-3 text-sm text-amber-900 dark:bg-amber-900/20 dark:text-amber-100">
                None of your sources are bound to a project yet. Open Configure on any source
                below to pick where its memories should land.
              </div>
            )}
            {groups.map((group) => {
              const expanded = isGroupExpanded(group);
              const groupKey = projectGroupKey(group);
              const isUnmappedGroup = group.project === null;
              return (
                <ProjectGroup
                  key={groupKey}
                  group={group}
                  expanded={expanded}
                  onToggle={() => toggleGroup(group)}
                >
                  {group.sources.map(({ source, memory_count }) => {
                    const localAgentJob = localAgentJobBySource[source.id] ?? currentLocalJobBySource[source.id];
                    const syncActivity = selectSourceSyncActivity({
                      sync: source.sync,
                      localJob: localAgentJob,
                      lifecycleMaintenance: source.lifecycle_maintenance,
                      pending: pendingSyncIds.has(source.id),
                    });
                    const isSourceBusy = sourceSyncActivityBlocksActions(syncActivity);
                    const isDeleting = deleteSource.isPending && sourcePendingDelete?.id === source.id;
                    const isUpdatingStatus =
                      setSourceStatus.isPending &&
                      setSourceStatus.variables?.sourceId === source.id;
                    const capabilities: SourceCapabilities = source.capabilities ?? {
                      can_subscribe: false,
                      can_configure: false,
                      can_configure_connection: false,
                      can_sync: false,
                      can_force_resync: false,
                      can_delete: false,
                      can_change_access: false,
                    };
                    const isManaged = isManagedSourceType(source.type) || isManagedSourceId(source.id);
                    const gene = geneByName.get(source.type);
                    const sourceLabel = SOURCE_LABELS[source.id] ?? SOURCE_LABELS[source.type] ?? {
                      name: gene?.display_name ?? source.type,
                      subtitle: gene?.data_shape ?? "",
                    };
                    const itemLabel =
                      sourceItemLabel(source);
                    const enabledForMe = source.enabled_for_me ?? source.subscription?.enabled ?? true;

                    return (
                      <SourceRow
                        key={`${groupKey}:${source.id}`}
                        source={source}
                        perGroupMemoryCount={memory_count}
                        syncActivity={syncActivity}
                        isDeleting={isDeleting}
                        isUpdatingStatus={isUpdatingStatus}
                        isManaged={isManaged}
                        sourceLabel={sourceLabel}
                        itemLabel={itemLabel}
                        enabledForMe={enabledForMe}
                        isSubscriptionPending={pendingSubscriptionIds.has(source.id)}
                        onConfigure={() => {
                          if (!capabilities.can_configure) return;
                          setConfigDialog({
                            sourceType: source.type,
                            source,
                            initialFocus: isUnmappedGroup ? { step: "project" } : undefined,
                          });
                        }}
                        onSync={() => {
                          if (!capabilities.can_sync || source.status === "paused") return;
                          syncSource.mutate({ source });
                        }}
                        onResume={() =>
                          setSourceStatus.mutate({ sourceId: source.id, status: "active" })
                        }
                        onShowDetails={() => setDetailsSource(source)}
                        onSubscriptionChange={(enabled) => {
                          if (!capabilities.can_subscribe) return;
                          setSubscription.mutate({ sourceId: source.id, enabled });
                        }}
                        actionsMenu={
                          <SourceActionsMenu
                              source={source}
                              capabilities={capabilities}
                              open={openMenuSourceId === source.id}
                              onOpenChange={(open) =>
                                setOpenMenuSourceId(open ? source.id : null)
                              }
                              onDelete={() => {
                                setOpenMenuSourceId(null);
                                setSourcePendingDelete(source);
                              }}
                              onForceResync={() => {
                                setOpenMenuSourceId(null);
                                forceResyncSource.mutate(source);
                              }}
                              onToggleStatus={() => {
                                setOpenMenuSourceId(null);
                                setSourceStatus.mutate({
                                  sourceId: source.id,
                                  status: source.status === "paused" ? "active" : "paused",
                                });
                              }}
                              onChangeAccess={() => {
                                setOpenMenuSourceId(null);
                                setAccessSource(source);
                              }}
                              disableMutatingActions={isSourceBusy || isDeleting}
                              disableForceResync={isSourceBusy || isDeleting || source.status === "paused"}
                              disableToggleStatus={isSourceBusy || isDeleting || isUpdatingStatus}
                              isUpdatingStatus={isUpdatingStatus}
                          />
                        }
                        highlighted={newSourceId === source.id}
                        onTogglePin={() => setSourcePin.mutate({
                          sourceId: source.id,
                          pinned: !source.pinned_for_me,
                        })}
                        isPinPending={
                          setSourcePin.isPending && setSourcePin.variables?.sourceId === source.id
                        }
                        onRetryAccess={
                          source.ownership?.viewer_relationship === "owner"
                          && source.access_transition?.status === "failed"
                            ? () => retrySourceAccess.mutate(source)
                            : undefined
                        }
                        onRevertAccess={
                          source.ownership?.viewer_relationship === "owner"
                          && source.access_transition?.status === "failed"
                            ? () => revertSourceAccess.mutate(source)
                            : undefined
                        }
                      />
                    );
                  })}
                </ProjectGroup>
              );
            })}
          </div>
        </AsyncBoundary>
      </DataSurface>

      <AddSourceDialog
        open={addOpen}
        onOpenChange={setAddOpen}
        genes={genes}
        isLoading={genesQuery.isLoading}
        onConfigureSelected={(sourceType) => {
          setAddOpen(false);
          setConfigDialog({ sourceType, source: null });
        }}
      />

      <SourceSetupDialog
        open={Boolean(configDialog.sourceType)}
        onOpenChange={(open) => {
          if (!open) setConfigDialog({ sourceType: null, source: null });
        }}
        sourceType={configDialog.sourceType}
        source={configDialog.source}
        initialFocus={configDialog.initialFocus}
        onSaved={(sourceId) => {
          if (!configDialog.source) {
            setSourceSearch("");
            setPinnedOnly(false);
            setNewSourceId(sourceId);
          }
        }}
        onRequestAccessChange={setAccessSource}
      />

      <SourceAccessChangeDialog
        source={accessSource}
        onOpenChange={(open) => {
          if (!open) setAccessSource(null);
        }}
      />

      <AgentSessionDetailsDialog
        source={detailsSource}
        onOpenChange={(open) => {
          if (!open) setDetailsSource(null);
        }}
      />

      <DeleteSourceDialog
        source={sourcePendingDelete}
        isDeleting={deleteSource.isPending}
        error={deleteSource.error}
        onOpenChange={(open) => {
          if (!open && !deleteSource.isPending) setSourcePendingDelete(null);
        }}
        onConfirm={() => {
          if (sourcePendingDelete) deleteSource.mutate(sourcePendingDelete.id);
        }}
      />
    </div>
  );
}

function SourceActionsMenu({
  source,
  capabilities,
  open,
  onOpenChange,
  onDelete,
  onForceResync,
  onToggleStatus,
  onChangeAccess,
  disableMutatingActions,
  disableForceResync,
  disableToggleStatus,
  isUpdatingStatus,
}: {
  source: Source;
  capabilities: SourceCapabilities;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDelete: () => void;
  onForceResync: () => void;
  onToggleStatus: () => void;
  onChangeAccess: () => void;
  disableMutatingActions: boolean;
  disableForceResync: boolean;
  disableToggleStatus: boolean;
  isUpdatingStatus: boolean;
}) {
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const [menuStyle, setMenuStyle] = useState<CSSProperties>({});
  const toggleStatus = sourceActionLayout.menu.find((action) => action.id === "toggle-status");
  const forceResync = sourceActionLayout.menu.find((action) => action.id === "force-resync");
  const deleteAction = sourceActionLayout.menu.find((action) => action.id === "delete");
  const isPaused = source.status === "paused";
  const canToggleStatus = capabilities.can_configure;
  const canForceResync = capabilities.can_force_resync;
  const canDelete = capabilities.can_delete;
  const canChangeAccess = capabilities.can_change_access && source.access_state === "active";
  const toggleStatusLabel = isPaused ? "Resume source" : "Pause source";
  const ToggleStatusIcon = isPaused ? Play : Pause;
  const forceResyncDisabledHint = isPaused
    ? "Resume the source first to refresh."
    : undefined;

  useLayoutEffect(() => {
    if (!open || !triggerRef.current || !menuRef.current) return;
    const rect = triggerRef.current.getBoundingClientRect();
    setMenuStyle(getSourceMenuStyle({
      triggerRight: rect.right,
      triggerTop: rect.top,
      triggerBottom: rect.bottom,
      viewportWidth: window.innerWidth,
      viewportHeight: window.innerHeight,
      menuHeight: menuRef.current.offsetHeight,
    }));
  }, [open]);

  useEffect(() => {
    if (!open) return;

    function onPointerDown(event: PointerEvent) {
      const target = event.target;
      if (!(target instanceof Node)) return;
      if (triggerRef.current?.contains(target) || menuRef.current?.contains(target)) return;
      onOpenChange(false);
    }

    function onKeyDown(event: KeyboardEvent) {
      if (event.key === "Escape") onOpenChange(false);
    }

    document.addEventListener("pointerdown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("pointerdown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [open, onOpenChange]);

  if (!canToggleStatus && !canForceResync && !canChangeAccess && !canDelete) {
    return null;
  }

  return (
    <div className="relative">
      <Button
        type="button"
        variant="ghost"
        size="icon"
        aria-label={`More actions for ${source.name}`}
        aria-haspopup="menu"
        aria-expanded={open}
        ref={triggerRef}
        onClick={() => {
          if (!open && triggerRef.current) {
            const rect = triggerRef.current.getBoundingClientRect();
            setMenuStyle(getSourceMenuStyle({
              triggerRight: rect.right,
              triggerTop: rect.top,
              triggerBottom: rect.bottom,
              viewportWidth: window.innerWidth,
              viewportHeight: window.innerHeight,
              menuHeight: canDelete ? 224 : 160,
            }));
          }
          onOpenChange(!open);
        }}
      >
        <MoreHorizontal className="size-4" />
      </Button>
      {open && typeof document !== "undefined" && createPortal(
        <div
          ref={menuRef}
          role="menu"
          aria-label={`More actions for ${source.name}`}
          className="z-50 rounded-lg border bg-popover p-1 text-popover-foreground shadow-lg"
          style={menuStyle}
        >
          {canChangeAccess && (
            <button
              type="button"
              role="menuitem"
              disabled={disableMutatingActions}
              className="flex w-full cursor-pointer items-start gap-3 rounded-md px-3 py-2 text-left text-sm hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
              onClick={onChangeAccess}
            >
              <LockKeyhole className="mt-0.5 size-4" />
              <span>
                <span className="block font-medium text-foreground">Change access</span>
                <span className="mt-0.5 block text-xs">Choose only you or everyone in the workspace.</span>
              </span>
            </button>
          )}
          {canToggleStatus && (
            <button
              type="button"
              role="menuitem"
              disabled={disableToggleStatus}
              className="flex w-full cursor-pointer items-start gap-3 rounded-md px-3 py-2 text-left text-sm hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
              onClick={onToggleStatus}
            >
              {isUpdatingStatus ? (
                <Loader2 className="mt-0.5 size-4 animate-spin" />
              ) : (
                <ToggleStatusIcon className="mt-0.5 size-4" />
              )}
              <span>
                <span className="block font-medium text-foreground">{toggleStatusLabel}</span>
                <span className="mt-0.5 block text-xs">{toggleStatus?.description}</span>
              </span>
            </button>
          )}
          {canForceResync && (
            <button
              type="button"
              role="menuitem"
              disabled={disableForceResync}
              title={forceResyncDisabledHint}
              className="flex w-full cursor-pointer items-start gap-3 rounded-md px-3 py-2 text-left text-sm hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
              onClick={onForceResync}
            >
              <RefreshCw className="mt-0.5 size-4" />
              <span>
                <span className="block font-medium text-foreground">{forceResync?.label}</span>
                <span className="mt-0.5 block text-xs">
                  {isPaused
                    ? "Resume the source first to look for new, changed, or removed documents."
                    : forceResync?.description}
                </span>
              </span>
            </button>
          )}
          {canDelete && (
            <>
              {(canToggleStatus || canForceResync || canChangeAccess) && <div className="my-1 h-px bg-border" />}
              <button
                type="button"
                role="menuitem"
                disabled={disableMutatingActions}
                className="flex w-full cursor-pointer items-start gap-3 rounded-md px-3 py-2 text-left text-sm text-destructive hover:bg-destructive/10 disabled:cursor-not-allowed disabled:opacity-60"
                onClick={onDelete}
              >
                <Trash2 className="mt-0.5 size-4" />
                <span>
                  <span className="block font-medium">{deleteAction?.label}</span>
                  <span className="mt-0.5 block text-xs text-destructive/80">{deleteAction?.description}</span>
                </span>
              </button>
            </>
          )}
        </div>,
        document.body,
      )}
    </div>
  );
}

function AgentSessionDetailsDialog({
  source,
  onOpenChange,
}: {
  source: Source | null;
  onOpenChange: (open: boolean) => void;
}) {
  const open = Boolean(source);
  const projectsQuery = useQuery<SourceProjectsResponse>({
    queryKey: ["source-projects", source?.id],
    queryFn: () => {
      if (!source) throw new Error("source is required");
      return resourceClient.get(`/sources/${source.id}/projects`).then((response) => response.data);
    },
    enabled: open && (source?.type === "agent_session" || (source ? isManagedSourceId(source.id) : false)),
  });
  const completenessQuery = useQuery<AgentSessionCompleteness>({
    queryKey: ["agent-session-completeness", source?.id],
    queryFn: () => {
      if (!source) throw new Error("source is required");
      return resourceClient
        .get("/agent-sessions/completeness", { params: { source_id: source.id } })
        .then((response) => response.data);
    },
    enabled: open && (source?.type === "agent_session" || (source ? isManagedSourceId(source.id) : false)),
  });

  if (!source) return null;

  const counts = completenessQuery.data?.counts ?? {};
  const projects = projectsQuery.data?.projects ?? [];
  const memoriesCreated = source.memory_count ?? 0;
  const keptSummaries = counts.package_created ?? 0;
  const skippedLowSignal = counts.no_output ?? 0;
  const failedCount = counts.failed ?? 0;
  const latestFailure = completenessQuery.data?.latest_failure ?? null;
  const totalProcessed = completenessQuery.data?.processed_total ?? keptSummaries + skippedLowSignal;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>Agent Session</DialogTitle>
          <DialogDescription>
            Automatic memory source populated by coding-agent plugins.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3 sm:grid-cols-2">
          <DetailMetric
            label="Memories created"
            value={formatCount(memoriesCreated)}
            tone="primary"
          />
          <DetailMetric
            label="Kept summaries"
            value={formatCount(keptSummaries)}
            tone="primary"
          />
        </div>

        <p className="text-xs text-muted-foreground">
          Most agent-session windows are conversational or temporary. MemForge keeps
          only windows likely to help future work and skips the rest as low-signal.
        </p>

        <details className="rounded-lg border bg-muted/30 text-xs">
          <summary className="cursor-pointer select-none px-3 py-2 text-muted-foreground hover:text-foreground">
            Operational details
          </summary>
          <dl className="grid grid-cols-2 gap-x-4 gap-y-1 px-3 pb-3 text-muted-foreground">
            <dt>Windows processed</dt>
            <dd className="text-right font-medium text-foreground">{formatCount(totalProcessed)}</dd>
            <dt>Kept summaries</dt>
            <dd className="text-right font-medium text-foreground">{formatCount(keptSummaries)}</dd>
            <dt>Skipped low-signal</dt>
            <dd className="text-right font-medium text-foreground">{formatCount(skippedLowSignal)}</dd>
            <dt>Needs retry</dt>
            <dd className="text-right font-medium text-foreground">{formatCount(failedCount)}</dd>
            {latestFailure?.last_seen_at && (
              <>
                <dt>Last retry seen</dt>
                <dd className="text-right font-medium text-foreground">
                  {timeAgo(latestFailure.last_seen_at)}
                </dd>
              </>
            )}
          </dl>
        </details>

        <div>
          <div className="flex items-center justify-between gap-3">
            <h3 className="text-sm font-medium">Projects with extracted memory</h3>
            <span className="text-xs text-muted-foreground">
              {memoriesCreated} memories
            </span>
          </div>

          <div className="mt-2 overflow-hidden rounded-lg border">
            {projectsQuery.isPending ? (
              <div className="flex items-center gap-2 p-3 text-sm text-muted-foreground">
                <Loader2 className="size-4 animate-spin" />
                Loading projects...
              </div>
            ) : projectsQuery.isError ? (
              <div className="p-3 text-sm text-destructive">
                Failed to load project summary.
              </div>
            ) : projects.length === 0 ? (
              <div className="p-3 text-sm text-muted-foreground">
                No projects observed yet.
              </div>
            ) : (
              <div className="divide-y">
                {projects.map((project) => (
                  <div key={project.project} className="grid gap-2 p-3 text-sm sm:grid-cols-[1fr_auto_auto_auto] sm:items-center">
                    <div className="min-w-0 font-medium">{project.project}</div>
                    <div className="text-muted-foreground">
                      <span className="font-medium text-foreground">{project.document_count}</span> summaries
                    </div>
                    <div className="text-muted-foreground">
                      <span className="font-medium text-foreground">{project.memory_count}</span> memories
                    </div>
                    <div className="text-xs text-muted-foreground">
                      {timeAgo(project.last_observed_at)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        <DialogFooter showCloseButton />
      </DialogContent>
    </Dialog>
  );
}

function DetailMetric({
  label,
  value,
  tone = "default",
}: {
  label: string;
  value: string;
  tone?: "default" | "primary";
}) {
  return (
    <div className="rounded-lg border p-3">
      <div className="text-xs text-muted-foreground">{label}</div>
      <div
        className={
          tone === "primary"
            ? "mt-1 text-2xl font-semibold tracking-tight text-foreground"
            : "mt-1 text-lg font-semibold"
        }
      >
        {value}
      </div>
    </div>
  );
}

function formatCount(value: number | undefined): string {
  return (value ?? 0).toLocaleString();
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl bg-card p-4 ring-1 ring-foreground/10">
      <div className="text-sm font-medium">{label}</div>
      <div className="mt-2 text-2xl font-bold tracking-tight">{value}</div>
    </div>
  );
}

function AddSourceDialog({
  open,
  onOpenChange,
  genes,
  isLoading,
  onConfigureSelected,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  genes: GeneMetadata[];
  isLoading: boolean;
  onConfigureSelected: (sourceType: string) => void;
}) {
  const configurableGenes = userConfigurableGenes(genes);
  const [agentSetupClient, setAgentSetupClient] = useState<AgentSessionClient | null>(null);

  const renderConfigurableGene = (gene: (typeof configurableGenes)[number]) => {
    const source = SOURCE_LABELS[gene.name] ?? {
      name: gene.display_name,
      subtitle: gene.data_shape,
      description: gene.description,
    };
    const connection = presentSourceConnection(gene);
    return (
      <div key={gene.name} className="flex min-h-36 flex-col rounded-lg border p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="flex min-w-0 items-start gap-3">
            <SourceIcon type={gene.name} className="mt-0.5 size-6" />
            <div className="min-w-0">
              <div className="text-sm font-medium">{source.name}</div>
              <div className="mt-1 text-xs text-muted-foreground">{source.description}</div>
            </div>
          </div>
          <SourceConnectionBadge mode={connection.mode} label={connection.label} />
        </div>
        <div className="mt-auto flex flex-wrap gap-2 pt-4">
          <Button
            type="button"
            size="sm"
            onClick={() => onConfigureSelected(gene.name)}
          >
            Set up
          </Button>
        </div>
      </div>
    );
  };

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-2xl">
          <DialogHeader>
            <DialogTitle>Connect a new source</DialogTitle>
            <DialogDescription>
              Choose how MemForge reaches the source, then configure what to sync.
            </DialogDescription>
          </DialogHeader>

          {isLoading && (
            <div className="flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              Loading source types...
            </div>
          )}

          {!isLoading && (
            <div className="space-y-6">
              <section className="space-y-3">
                <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                  <div>
                    <h3 className="text-sm font-semibold text-foreground">Sources</h3>
                    <p className="mt-0.5 text-xs text-muted-foreground">
                      Select a source to configure its connection and sync scope.
                    </p>
                  </div>
                  <LocalAgentDaemonStatus />
                </div>
                <div className="grid gap-3 sm:grid-cols-2">
                  {configurableGenes.map(renderConfigurableGene)}
                </div>
              </section>

              <section className="space-y-3 border-t pt-5">
                <div>
                  <h3 className="text-sm font-semibold text-foreground">Coding agent integrations</h3>
                  <p className="mt-0.5 text-xs text-muted-foreground">
                    Installed plugins add these sources automatically.
                  </p>
                </div>
                <div className="grid gap-3 sm:grid-cols-2">
                  {AGENT_SESSION_CARDS.map((card) => (
                    <div key={card.client} className="flex min-h-36 flex-col rounded-lg border p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex min-w-0 items-start gap-3">
                          <SourceIcon
                            type="agent_session"
                            client={card.client}
                            className="mt-0.5 size-6"
                          />
                          <div className="min-w-0">
                            <div className="text-sm font-medium">{card.title}</div>
                            <div className="mt-1 text-xs text-muted-foreground">
                              {card.description}
                            </div>
                          </div>
                        </div>
                        <Badge
                          variant="outline"
                          className="border-amber-200 bg-amber-50 text-[11px] text-amber-700 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-200"
                        >
                          Plugin
                        </Badge>
                      </div>
                      <div className="mt-auto pt-4">
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={() => setAgentSetupClient(card.client)}
                        >
                          <Info className="size-3.5" />
                          View setup
                        </Button>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            </div>
          )}
        </DialogContent>
      </Dialog>

      <AgentSessionSetupDialog
        client={agentSetupClient}
        onOpenChange={(open) => !open && setAgentSetupClient(null)}
      />
    </>
  );
}

const MEMFORGE_MARKETPLACE_REPO = "shno-labs/mem-forge";
const MEMFORGE_PLUGIN_NAME = "memory@memforge";

const CODEX_PLUGIN_COMMANDS = [
  `codex plugin marketplace add ${MEMFORGE_MARKETPLACE_REPO}`,
  `codex plugin add ${MEMFORGE_PLUGIN_NAME}`,
] as const;

const CLAUDE_CODE_PLUGIN_COMMANDS = [
  `/plugin marketplace add ${MEMFORGE_MARKETPLACE_REPO}`,
  `/plugin install ${MEMFORGE_PLUGIN_NAME}`,
] as const;

type AgentSessionClient = "codex" | "claude-code";

const AGENT_SESSION_CARDS: ReadonlyArray<{
  client: AgentSessionClient;
  title: string;
  description: string;
}> = [
  {
    client: "codex",
    title: "Codex Session",
    description:
      "Save Codex coding sessions as memory.",
  },
  {
    client: "claude-code",
    title: "Claude Code Session",
    description:
      "Save Claude Code coding sessions as memory.",
  },
];

const AGENT_SESSION_SETUP_BY_CLIENT: Record<
  AgentSessionClient,
  { agentName: string; commands: readonly string[]; runFrom: string }
> = {
  codex: {
    agentName: "Codex",
    commands: CODEX_PLUGIN_COMMANDS,
    runFrom: "Run these from any terminal where the codex CLI is on PATH:",
  },
  "claude-code": {
    agentName: "Claude Code",
    commands: CLAUDE_CODE_PLUGIN_COMMANDS,
    runFrom: "Run these inside an active Claude Code session:",
  },
};

function AgentSessionSetupDialog({
  client,
  onOpenChange,
}: {
  client: AgentSessionClient | null;
  onOpenChange: (open: boolean) => void;
}) {
  const setup = client ? AGENT_SESSION_SETUP_BY_CLIENT[client] : null;
  return (
    <Dialog open={Boolean(client)} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{setup ? `${setup.agentName} session setup` : "Agent session setup"}</DialogTitle>
          <DialogDescription>
            MemForge is a SaaS service and cannot read local agent transcripts. The plugin runs on
            your machine, captures session windows, and pushes them to MemForge after each session.
            No further configuration in this UI is needed; the source appears automatically after
            the first upload.
          </DialogDescription>
        </DialogHeader>

        {setup && (
          <div className="space-y-5">
            <AgentPluginInstructions
              agentName={setup.agentName}
              runFrom={setup.runFrom}
              commands={setup.commands}
            />

            <p className="text-xs text-muted-foreground">
              After installing, set{" "}
              <code className="rounded bg-muted px-1 font-mono text-[11px]">MEMFORGE_API_URL</code>{" "}
              and optionally{" "}
              <code className="rounded bg-muted px-1 font-mono text-[11px]">MEMFORGE_API_TOKEN</code>{" "}
              when the plugin should reach a non-default MemForge instance. See{" "}
              <span className="font-medium text-foreground">
                docs/design/agent-session-saas-plugin-flow.md
              </span>{" "}
              in the MemForge repository for the full design.
            </p>
          </div>
        )}

        <DialogFooter>
          <Button type="button" onClick={() => onOpenChange(false)}>
            Got it
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function AgentPluginInstructions({
  agentName,
  runFrom,
  commands,
}: {
  agentName: string;
  runFrom: string;
  commands: readonly string[];
}) {
  return (
    <div className="space-y-2">
      <div className="text-sm font-medium">{agentName}</div>
      <p className="text-xs text-muted-foreground">{runFrom}</p>
      <div className="space-y-1.5">
        {commands.map((command) => (
          <CopyableCommand key={command} command={command} />
        ))}
      </div>
    </div>
  );
}

function SourceConnectionBadge({
  mode,
  label,
}: {
  mode: SourceConnectionMode;
  label: string;
}) {
  const tone = mode === "device"
    ? "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-800 dark:bg-emerald-950/50 dark:text-emerald-200"
    : mode === "choice"
      ? "border-violet-200 bg-violet-50 text-violet-700 dark:border-violet-800 dark:bg-violet-950/50 dark:text-violet-200"
      : "border-sky-200 bg-sky-50 text-sky-700 dark:border-sky-800 dark:bg-sky-950/50 dark:text-sky-200";
  return (
    <Badge variant="outline" className={`text-[11px] ${tone}`}>
      {label}
    </Badge>
  );
}

function CopyableCommand({ command }: { command: string }) {
  const copy = () => {
    if (typeof navigator !== "undefined" && navigator.clipboard) {
      void navigator.clipboard.writeText(command);
    }
  };
  return (
    <div className="flex items-center gap-2 rounded-md border bg-muted/30 p-2">
      <code className="flex-1 break-all font-mono text-[11px] text-foreground">{command}</code>
      <Button type="button" variant="outline" size="sm" onClick={copy}>
        Copy
      </Button>
    </div>
  );
}

function DeleteSourceDialog({
  source,
  isDeleting,
  error,
  onOpenChange,
  onConfirm,
}: {
  source: Source | null;
  isDeleting: boolean;
  error: unknown;
  onOpenChange: (open: boolean) => void;
  onConfirm: () => void;
}) {
  const itemLabel = source ? SOURCE_ITEM_LABELS[source.type] ?? "documents" : "documents";
  const memoryCount = source?.memory_count ?? 0;
  const ownership = source?.ownership;
  const creatorLabel = (() => {
    if (!ownership) return null;
    if (ownership.viewer_relationship === "owner") return "Created by you";
    if (ownership.created_by_user_id) return `Created by ${ownership.created_by_user_id}`;
    return null;
  })();
  return (
    <Dialog open={Boolean(source)} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Delete source?</DialogTitle>
          <DialogDescription>
            This removes the source, its synced {itemLabel}, and this source's provenance links. Memories with no
            remaining source support will be retired and removed from search.
          </DialogDescription>
        </DialogHeader>

        {source && (
          <div className="rounded-lg border bg-muted/30 p-3 text-sm">
            <div className="font-medium">{source.name}</div>
            <div className="mt-1 text-muted-foreground">
              {source.doc_count.toLocaleString()} {itemLabel} · {memoryCount.toLocaleString()} memories
            </div>
            {creatorLabel && (
              <div className="mt-1 text-xs text-muted-foreground">{creatorLabel}</div>
            )}
          </div>
        )}

        {Boolean(error) && (
          <div className="rounded-lg bg-destructive/10 p-3 text-sm text-destructive">
            Delete failed. Check the API logs and try again.
          </div>
        )}

        <DialogFooter>
          <Button type="button" variant="outline" disabled={isDeleting} onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button type="button" variant="destructive" disabled={isDeleting} onClick={onConfirm}>
            {isDeleting ? <Loader2 className="size-4 animate-spin" /> : <Trash2 className="size-4" />}
            Delete Source
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

function isForbiddenError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  const status = (error as { response?: { status?: number }; status?: number }).response?.status
    ?? (error as { status?: number }).status;
  return status === 403;
}
