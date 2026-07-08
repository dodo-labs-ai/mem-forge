import { type CSSProperties, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { Files, Info, Loader2, MoreHorizontal, Pause, Play, Plus, RefreshCw, Trash2, X } from "lucide-react";
import client from "@/api/client";
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
import { Button } from "@/components/ui/button";
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
import { SourceConfigDialog } from "./SourceConfigDialog";
import { LocalAgentDaemonStatus } from "./LocalAgentDaemonStatus";
import { isManagedSourceId, isManagedSourceType, isPushBasedSourceType, userConfigurableGenes } from "./managedSources";
import { getSourceActionEndpoint, getSourceMenuStyle, sourceActionLayout } from "./sourceActions";
import { ProjectGroup } from "./ProjectGroup";
import {
  PROJECT_GROUPS_DEFAULT_EXPANDED,
  groupSourcesByProject,
  projectGroupKey,
  type ResolvedBySource,
} from "./projectGrouping";
import { SourceRow } from "./SourceRow";
import { TeamsSourceWizard } from "./TeamsSourceWizard";

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

function normalizeSources(payload: SourcesResponse | Source[] | undefined): Source[] {
  if (Array.isArray(payload)) return payload;
  if (Array.isArray(payload?.data)) return payload.data;
  return [];
}

function isInternalGitHubSource(source: Source): boolean {
  return source.type === "github_repo" && String(source.config.connection_mode ?? "cloud_pull") === "local_push";
}

function localAgentSyncOperation(source: Source): string | null {
  if (isInternalGitHubSource(source)) return "github_repo_sync";
  if (source.type === "local_markdown" && String(source.config.root ?? "").trim().length > 0) {
    return "local_markdown_sync";
  }
  if (source.type === "jira" && String(source.config.sync_mode ?? "cloud") === "local_agent") return "jira_sync";
  return null;
}

const LOCAL_AGENT_SYNC_POLL_ATTEMPTS = 1_800;
const LOCAL_AGENT_SYNC_POLL_INTERVAL_MS = 2_000;
const LOCAL_AGENT_WAITING_MESSAGE = "Waiting for local daemon to sync this source.";
const LOCAL_AGENT_TIMEOUT_MESSAGE =
  "Local daemon did not pick up this job. Start it with `memforge adapter daemon run` and try again.";
const LOCAL_AGENT_CONFIGURE_FOLDER_MESSAGE = "Configure a folder path before syncing this local source.";
const LOCAL_AGENT_SYNC_FAILED_MESSAGE = "Local daemon could not sync this source.";

function safeSourceErrorMessage(error: unknown): string | null {
  if (!(error instanceof Error)) return null;
  if (
    error.message === LOCAL_AGENT_WAITING_MESSAGE
    || error.message === LOCAL_AGENT_TIMEOUT_MESSAGE
    || error.message === LOCAL_AGENT_CONFIGURE_FOLDER_MESSAGE
    || error.message === LOCAL_AGENT_SYNC_FAILED_MESSAGE
  ) {
    return error.message;
  }
  return null;
}

async function createLocalAgentSyncJob(source: Source): Promise<LocalAgentJobCreateResponse | null> {
  const operation = localAgentSyncOperation(source);
  if (!operation) {
    if (source.type === "local_markdown") {
      throw new Error(LOCAL_AGENT_CONFIGURE_FOLDER_MESSAGE);
    }
    return null;
  }
  const response = await client.post<LocalAgentJobCreateResponse>("/api/cloud/local-agent/jobs", {
    source_id: source.id,
    source_type: source.type,
    operation,
    payload: {
      ...source.config,
      process_now: true,
    },
  });
  const status = await pollLocalAgentSyncJob(response.data.job_id);
  if (status.status === "failed") {
    if (status.last_error) {
      console.warn("Local daemon sync failed", status.last_error);
    }
    throw new Error(LOCAL_AGENT_SYNC_FAILED_MESSAGE);
  }
  return response.data;
}

async function pollLocalAgentSyncJob(jobId: string): Promise<LocalAgentJobStatusResponse> {
  for (let attempt = 0; attempt < LOCAL_AGENT_SYNC_POLL_ATTEMPTS; attempt += 1) {
    const response = await client.get<LocalAgentJobStatusResponse>(`/api/cloud/local-agent/jobs/${jobId}`);
    if (response.data.status === "succeeded" || response.data.status === "failed") {
      return response.data;
    }
    await new Promise((resolve) => window.setTimeout(resolve, LOCAL_AGENT_SYNC_POLL_INTERVAL_MS));
  }
  throw new Error(LOCAL_AGENT_TIMEOUT_MESSAGE);
}

export function SourcesPage() {
  const queryClient = useQueryClient();
  const [addOpen, setAddOpen] = useState(false);
  const [teamsWizardOpen, setTeamsWizardOpen] = useState(false);
  const [configDialog, setConfigDialog] = useState<{
    sourceType: string | null;
    source?: Source | null;
    initialFocus?: { step: "project" };
  }>({ sourceType: null, source: null });
  const [detailsSource, setDetailsSource] = useState<Source | null>(null);
  const [openMenuSourceId, setOpenMenuSourceId] = useState<string | null>(null);
  const [sourcePendingDelete, setSourcePendingDelete] = useState<Source | null>(null);
  const [pendingSyncIds, setPendingSyncIds] = useState<Set<string>>(new Set());
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(() => new Set());
  const [pendingSubscriptionIds, setPendingSubscriptionIds] = useState<Set<string>>(
    () => new Set(),
  );
  const [authorityMessage, setAuthorityMessage] = useState<string | null>(null);

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

  const genesQuery = useQuery<GeneMetadata[]>({
    queryKey: ["genes"],
    queryFn: () => client.get("/api/genes").then((response) => response.data),
  });

  const sourcesQuery = useQuery<SourcesResponse | Source[]>({
    queryKey: ["sources"],
    queryFn: () => client.get("/api/sources").then((response) => response.data),
    refetchInterval: (query) => {
      const sources = normalizeSources(query.state.data);
      return sources.some((source) => source.sync?.status === "running") ? 2000 : false;
    },
  });

  const projectsQuery = useQuery<Project[]>({
    queryKey: ["projects"],
    queryFn: () => client.get("/api/projects").then((response) => response.data),
  });

  const syncSource = useMutation({
    mutationFn: async ({ source, forceFullSync = false }: { source: Source; forceFullSync?: boolean }) => {
      const sourceId = source.id;
      setPendingSyncIds((current) => new Set(current).add(sourceId));
      if (localAgentSyncOperation(source)) {
        setAuthorityMessage(LOCAL_AGENT_WAITING_MESSAGE);
      }
      const localAgentJob = await createLocalAgentSyncJob(source);
      if (localAgentJob) {
        return { data: localAgentJob };
      }
      return client.post(`/api/sources/${sourceId}/sync`, { force_full_sync: forceFullSync });
    },
    onError: (error) => handleAuthorityError(error, "Failed to start sync."),
    onSettled: (_data, _error, variables) => {
      if (!_error) {
        setAuthorityMessage((current) => current === LOCAL_AGENT_WAITING_MESSAGE ? null : current);
      }
      setPendingSyncIds((current) => {
        const next = new Set(current);
        next.delete(variables.source.id);
        return next;
      });
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const deleteSource = useMutation({
    mutationFn: (sourceId: string) => client.delete(getSourceActionEndpoint(sourceId, "delete")),
    onError: (error) => handleAuthorityError(error, "Failed to delete source."),
    onSuccess: () => {
      setSourcePendingDelete(null);
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
    },
  });

  const forceResyncSource = useMutation({
    mutationFn: async (source: Source) => {
      setPendingSyncIds((current) => new Set(current).add(source.id));
      if (localAgentSyncOperation(source)) {
        setAuthorityMessage(LOCAL_AGENT_WAITING_MESSAGE);
      }
      const localAgentJob = await createLocalAgentSyncJob(source);
      if (localAgentJob) {
        return { data: localAgentJob };
      }
      return client.post(getSourceActionEndpoint(source.id, "force-resync"));
    },
    onError: (error) => handleAuthorityError(error, "Failed to start refresh."),
    onSettled: (_data, _error, source) => {
      if (!_error) {
        setAuthorityMessage((current) => current === LOCAL_AGENT_WAITING_MESSAGE ? null : current);
      }
      setPendingSyncIds((current) => {
        const next = new Set(current);
        next.delete(source.id);
        return next;
      });
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const setSourceStatus = useMutation({
    mutationFn: ({ sourceId, status }: { sourceId: string; status: Source["status"] }) =>
      client.put(`/api/sources/${sourceId}`, { status }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const setSubscription = useMutation({
    mutationFn: ({ sourceId, enabled }: { sourceId: string; enabled: boolean }) => {
      setPendingSubscriptionIds((current) => new Set(current).add(sourceId));
      return client.put(`/api/sources/${sourceId}/subscription`, { enabled });
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
        client
          .get<ResolvedProjectsResponse>(`/api/sources/${source.id}/projects/resolved`)
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

  const groups = groupSourcesByProject(sources, projects, resolvedBySource);

  const allInUnmapped =
    sources.length > 0 && groups.length === 1 && groups[0].project === null;

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
          <h2 className="text-base font-semibold">Source List</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            {sources.length.toLocaleString()} configured ingestion sources.
          </p>
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
                    const isSyncing = source.sync?.status === "running" || pendingSyncIds.has(source.id);
                    const isDeleting = deleteSource.isPending && sourcePendingDelete?.id === source.id;
                    const isUpdatingStatus =
                      setSourceStatus.isPending &&
                      setSourceStatus.variables?.sourceId === source.id;
                    const capabilities: SourceCapabilities = source.capabilities ?? {
                      can_subscribe: false,
                      can_configure: false,
                      can_sync: false,
                      can_force_resync: false,
                      can_delete: false,
                    };
                    const isManaged = isManagedSourceType(source.type) || isManagedSourceId(source.id);
                    const gene = geneByName.get(source.type);
                    const sourceLabel = SOURCE_LABELS[source.id] ?? SOURCE_LABELS[source.type] ?? {
                      name: gene?.display_name ?? source.type,
                      subtitle: gene?.data_shape ?? "",
                    };
                    const itemLabel =
                      SOURCE_ITEM_LABELS[source.id] ??
                      SOURCE_ITEM_LABELS[source.type] ??
                      "documents";
                    const showActionsMenu =
                      capabilities.can_configure ||
                      capabilities.can_force_resync ||
                      capabilities.can_delete;
                    const enabledForMe = source.enabled_for_me ?? source.subscription?.enabled ?? true;

                    return (
                      <SourceRow
                        key={`${groupKey}:${source.id}`}
                        source={source}
                        perGroupMemoryCount={memory_count}
                        isSyncing={isSyncing}
                        isDeleting={isDeleting}
                        isUpdatingStatus={isUpdatingStatus}
                        isManaged={isManaged}
                        sourceLabel={sourceLabel}
                        itemLabel={itemLabel}
                        authSessionLabel={authSessionLabel}
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
                          showActionsMenu ? (
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
                              disableForceResync={isSyncing || isDeleting || source.status === "paused"}
                              disableToggleStatus={isSyncing || isDeleting || isUpdatingStatus}
                              isUpdatingStatus={isUpdatingStatus}
                            />
                          ) : null
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
        onTeamsSelected={() => {
          setAddOpen(false);
          setTeamsWizardOpen(true);
        }}
        onConfigureSelected={(sourceType) => {
          setAddOpen(false);
          setConfigDialog({ sourceType, source: null });
        }}
      />

      <SourceConfigDialog
        open={Boolean(configDialog.sourceType)}
        onOpenChange={(open) => {
          if (!open) setConfigDialog({ sourceType: null, source: null });
        }}
        sourceType={configDialog.sourceType}
        source={configDialog.source}
        initialFocus={configDialog.initialFocus}
      />

      <AgentSessionDetailsDialog
        source={detailsSource}
        onOpenChange={(open) => {
          if (!open) setDetailsSource(null);
        }}
      />

      <TeamsSourceWizard
        open={teamsWizardOpen}
        onOpenChange={setTeamsWizardOpen}
        onCreated={() => {
          setTeamsWizardOpen(false);
          queryClient.invalidateQueries({ queryKey: ["sources"] });
          queryClient.invalidateQueries({ queryKey: ["stats"] });
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
              menuHeight: canDelete ? 288 : 180,
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
              {(canToggleStatus || canForceResync) && <div className="my-1 h-px bg-border" />}
              <button
                type="button"
                role="menuitem"
                className="flex w-full cursor-pointer items-start gap-3 rounded-md px-3 py-2 text-left text-sm text-destructive hover:bg-destructive/10"
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
      return client.get(`/api/sources/${source.id}/projects`).then((response) => response.data);
    },
    enabled: open && (source?.type === "agent_session" || (source ? isManagedSourceId(source.id) : false)),
  });
  const completenessQuery = useQuery<AgentSessionCompleteness>({
    queryKey: ["agent-session-completeness", source?.id],
    queryFn: () => {
      if (!source) throw new Error("source is required");
      return client
        .get("/api/agent-sessions/completeness", { params: { source_id: source.id } })
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

function authSessionLabel(status: string) {
  if (status === "active") return "active";
  if (status === "expired") return "expired";
  if (status === "failed") return "failed";
  return "missing";
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
  onTeamsSelected,
  onConfigureSelected,
}: {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  genes: GeneMetadata[];
  isLoading: boolean;
  onTeamsSelected: () => void;
  onConfigureSelected: (sourceType: string) => void;
}) {
  const configurableGenes = userConfigurableGenes(genes);
  const remoteGenes = configurableGenes.filter((gene) => !isPushBasedSourceType(gene.name));
  const pushGenes = configurableGenes.filter((gene) => isPushBasedSourceType(gene.name));
  const [agentSetupClient, setAgentSetupClient] = useState<AgentSessionClient | null>(null);

  const renderConfigurableGene = (gene: (typeof configurableGenes)[number]) => {
    const source = SOURCE_LABELS[gene.name] ?? {
      name: gene.display_name,
      subtitle: gene.data_shape,
      description: gene.description,
    };
    const isTeams = gene.name === "teams";
    return (
      <div key={gene.name} className="rounded-lg border p-4">
        <div className="flex items-start gap-3">
          <SourceIcon type={gene.name} className="mt-0.5 size-6" />
          <div className="min-w-0">
            <div className="text-sm font-medium">{source.name}</div>
            <div className="mt-1 text-xs text-muted-foreground">{source.description}</div>
          </div>
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          <Button
            type="button"
            size="sm"
            onClick={() => onConfigureSelected(gene.name)}
          >
            Configure
          </Button>
          {isTeams && (
            <Button type="button" size="sm" variant="outline" onClick={onTeamsSelected}>
              Browse Teams
            </Button>
          )}
        </div>
      </div>
    );
  };

  return (
    <>
      <Dialog open={open} onOpenChange={onOpenChange}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle>Connect a new source</DialogTitle>
            <DialogDescription>
              Configure source connection and sync scope before creating it.
            </DialogDescription>
          </DialogHeader>

          {isLoading && (
            <div className="flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              Loading source types...
            </div>
          )}

          {!isLoading && (
            <div className="space-y-5">
              {remoteGenes.length > 0 && (
                <section className="space-y-3">
                  <SectionDivider label="Pull from a remote service" />
                  <div className="grid gap-3 sm:grid-cols-2">
                    {remoteGenes.map(renderConfigurableGene)}
                  </div>
                </section>
              )}

              <section className="space-y-3">
                <SectionDivider label="Push from your local device" />
                <LocalAgentDaemonStatus />
                <div className="grid gap-3 sm:grid-cols-2">
                  {pushGenes.map(renderConfigurableGene)}
                  {AGENT_SESSION_CARDS.map((card) => (
                    <div key={card.client} className="rounded-lg border p-4">
                      <div className="flex items-start gap-3">
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
                      <div className="mt-4">
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
              when the plugin should reach a non-default MemForge instance. Hosted
              multi-workspace deployments also need{" "}
              <code className="rounded bg-muted px-1 font-mono text-[11px]">
                MEMFORGE_WORKSPACE_ID
              </code>. See{" "}
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

function SectionDivider({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3">
      <div className="h-px flex-1 bg-border" />
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      <div className="h-px flex-1 bg-border" />
    </div>
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
    if (ownership.viewer_relationship === "creator") return "Created by you";
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
