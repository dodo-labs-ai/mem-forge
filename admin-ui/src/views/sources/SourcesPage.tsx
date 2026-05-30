import { type CSSProperties, useEffect, useLayoutEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Files, Loader2, MoreHorizontal, Play, Plus, RefreshCw, SlidersHorizontal, Trash2 } from "lucide-react";
import client from "@/api/client";
import type { GeneMetadata, Source } from "@/api/types";
import { AsyncBoundary } from "@/components/admin/AsyncBoundary";
import { DataSurface } from "@/components/admin/DataSurface";
import { EmptyState } from "@/components/admin/EmptyState";
import { PageHeader } from "@/components/admin/PageHeader";
import { StatusDot } from "@/components/admin/StatusBadge";
import { SyncStatusBar } from "@/components/admin/SyncStatusBar";
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
import { timeAgo } from "@/utils/date";
import { SourceConfigDialog } from "./SourceConfigDialog";
import { getSourceActionEndpoint, getSourceMenuStyle, sourceActionLayout } from "./sourceActions";
import { TeamsSourceWizard } from "./TeamsSourceWizard";

const SOURCE_LABELS: Record<string, { name: string; subtitle: string; description: string }> = {
  confluence: { name: "Confluence", subtitle: "Knowledge source", description: "Wiki pages and documentation" },
  github_pages: { name: "GitHub Pages", subtitle: "Documentation source", description: "Published documentation pages" },
  jira: { name: "Jira", subtitle: "Work tracking source", description: "Tickets, decisions, and work items" },
  teams: { name: "Microsoft Teams", subtitle: "Conversation source", description: "Channel messages, group chats, and direct messages" },
};

const SOURCE_DOTS: Record<string, string> = {
  confluence: "bg-blue-500",
  github_pages: "bg-slate-500",
  jira: "bg-emerald-500",
  teams: "bg-violet-500",
  outlook: "bg-orange-500",
};

const SOURCE_ITEM_LABELS: Record<string, string> = {
  confluence: "pages",
  github_pages: "documents",
  jira: "issues",
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

export function SourcesPage() {
  const queryClient = useQueryClient();
  const [addOpen, setAddOpen] = useState(false);
  const [teamsWizardOpen, setTeamsWizardOpen] = useState(false);
  const [configDialog, setConfigDialog] = useState<{
    sourceType: string | null;
    source?: Source | null;
  }>({ sourceType: null, source: null });
  const [openMenuSourceId, setOpenMenuSourceId] = useState<string | null>(null);
  const [sourcePendingDelete, setSourcePendingDelete] = useState<Source | null>(null);
  const [pendingSyncIds, setPendingSyncIds] = useState<Set<string>>(new Set());

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

  const syncSource = useMutation({
    mutationFn: ({ sourceId, forceFullSync = false }: { sourceId: string; forceFullSync?: boolean }) => {
      setPendingSyncIds((current) => new Set(current).add(sourceId));
      return client.post(`/api/sources/${sourceId}/sync`, { force_full_sync: forceFullSync });
    },
    onSettled: (_data, _error, variables) => {
      setPendingSyncIds((current) => {
        const next = new Set(current);
        next.delete(variables.sourceId);
        return next;
      });
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const deleteSource = useMutation({
    mutationFn: (sourceId: string) => client.delete(getSourceActionEndpoint(sourceId, "delete")),
    onSuccess: () => {
      setSourcePendingDelete(null);
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
      queryClient.invalidateQueries({ queryKey: ["memories"] });
    },
  });

  const forceResyncSource = useMutation({
    mutationFn: (sourceId: string) => {
      setPendingSyncIds((current) => new Set(current).add(sourceId));
      return client.post(getSourceActionEndpoint(sourceId, "force-resync"));
    },
    onSettled: (_data, _error, sourceId) => {
      setPendingSyncIds((current) => {
        const next = new Set(current);
        next.delete(sourceId);
        return next;
      });
      queryClient.invalidateQueries({ queryKey: ["sources"] });
      queryClient.invalidateQueries({ queryKey: ["stats"] });
    },
  });

  const sources = normalizeSources(sourcesQuery.data);
  const genes = genesQuery.data ?? [];
  const geneByName = new Map(genes.map((gene) => [gene.name, gene]));
  const totalDocs = sources.reduce((sum, source) => sum + source.doc_count, 0);
  const totalMemories = sources.reduce((sum, source) => sum + (source.memory_count ?? 0), 0);

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
          <div className="divide-y">
            {sources.map((source) => {
              const isSyncing = source.sync?.status === "running" || pendingSyncIds.has(source.id);
              const isDeleting = deleteSource.isPending && sourcePendingDelete?.id === source.id;
              const gene = geneByName.get(source.type);
              const sourceLabel = SOURCE_LABELS[source.type] ?? {
                name: gene?.display_name ?? source.type,
                subtitle: gene?.data_shape ?? "",
              };
              const itemLabel = SOURCE_ITEM_LABELS[source.type] ?? "documents";

              return (
                <div key={source.id} className="space-y-3 p-4">
                  <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                    <div className="flex min-w-0 items-start gap-3">
                      <span className={`mt-1 size-2.5 shrink-0 rounded-full ${SOURCE_DOTS[source.type] ?? "bg-muted-foreground"}`} />
                      <div className="min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="truncate text-sm font-medium">{source.name}</h3>
                          <StatusDot status={source.status} />
                          <Badge variant={source.status === "active" ? "secondary" : "outline"}>
                            {source.status}
                          </Badge>
                        </div>
                        <p className="mt-1 text-xs text-muted-foreground">
                          {sourceLabel?.name ?? source.type}
                          {sourceLabel?.subtitle ? ` · ${sourceLabel.subtitle}` : ""}
                        </p>
                        <div className="mt-3 flex flex-wrap gap-x-5 gap-y-1 text-sm text-muted-foreground">
                          <span>
                            <span className="font-medium text-foreground">{source.doc_count}</span> {itemLabel}
                          </span>
                          <span>
                            <span className="font-medium text-foreground">{source.memory_count ?? 0}</span> memories
                          </span>
                          <span>{source.sync?.status === "running" ? "Syncing now" : `Last synced: ${timeAgo(source.last_sync)}`}</span>
                          {source.type === "jira" && source.auth_session && (
                            <span className={source.auth_session.status === "active" ? "text-emerald-600" : "text-destructive"}>
                              Browser session: {authSessionLabel(source.auth_session.status)}
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    <div className="flex items-center justify-end gap-2 sm:shrink-0">
                      <Button
                        type="button"
                        variant="outline"
                        aria-label="Configure source"
                        disabled={isDeleting}
                        onClick={() => setConfigDialog({ sourceType: source.type, source })}
                      >
                        <SlidersHorizontal className="size-4" />
                        <span className="hidden lg:inline">{sourceActionLayout.primary[0].label}</span>
                      </Button>
                      <Button
                        type="button"
                        disabled={isSyncing || isDeleting}
                        onClick={() => syncSource.mutate({ sourceId: source.id })}
                      >
                        {isSyncing ? (
                          <Loader2 className="size-4 animate-spin" />
                        ) : (
                          <Play className="size-4" />
                        )}
                        {isSyncing ? "Syncing" : sourceActionLayout.primary[1].label}
                      </Button>
                      <SourceActionsMenu
                        source={source}
                        open={openMenuSourceId === source.id}
                        onOpenChange={(open) => setOpenMenuSourceId(open ? source.id : null)}
                        onDelete={() => {
                          setOpenMenuSourceId(null);
                          setSourcePendingDelete(source);
                        }}
                        onForceResync={() => {
                          setOpenMenuSourceId(null);
                          forceResyncSource.mutate(source.id);
                        }}
                        disableForceResync={isSyncing || isDeleting}
                      />
                    </div>
                  </div>

                  <SyncStatusBar
                    sync={source.sync}
                    itemLabel={itemLabel}
                    onRetry={() => syncSource.mutate({ sourceId: source.id })}
                  />
                </div>
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
  open,
  onOpenChange,
  onDelete,
  onForceResync,
  disableForceResync,
}: {
  source: Source;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onDelete: () => void;
  onForceResync: () => void;
  disableForceResync: boolean;
}) {
  const triggerRef = useRef<HTMLButtonElement | null>(null);
  const menuRef = useRef<HTMLDivElement | null>(null);
  const [menuStyle, setMenuStyle] = useState<CSSProperties>({});
  const forceResync = sourceActionLayout.menu.find((action) => action.id === "force-resync");
  const deleteAction = sourceActionLayout.menu.find((action) => action.id === "delete");

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
              menuHeight: 224,
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
          <button
            type="button"
            role="menuitem"
            disabled={disableForceResync}
            className="flex w-full cursor-pointer items-start gap-3 rounded-md px-3 py-2 text-left text-sm hover:bg-muted disabled:cursor-not-allowed disabled:opacity-60"
            onClick={onForceResync}
          >
            <RefreshCw className="mt-0.5 size-4" />
            <span>
              <span className="block font-medium text-foreground">{forceResync?.label}</span>
              <span className="mt-0.5 block text-xs">{forceResync?.description}</span>
            </span>
          </button>
          <div className="my-1 h-px bg-border" />
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
        </div>,
        document.body,
      )}
    </div>
  );
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
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Connect a new source</DialogTitle>
          <DialogDescription>
            Configure source connection and sync scope before creating it.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-3 sm:grid-cols-2">
          {isLoading && (
            <div className="col-span-full flex items-center justify-center gap-2 py-8 text-sm text-muted-foreground">
              <Loader2 className="size-4 animate-spin" />
              Loading source types...
            </div>
          )}
          {!isLoading && genes.map((gene) => {
            const source = SOURCE_LABELS[gene.name] ?? {
              name: gene.display_name,
              subtitle: gene.data_shape,
              description: gene.description,
            };
            const isTeams = gene.name === "teams";
            return (
              <div key={gene.name} className="rounded-lg border p-4">
                <div className="flex items-start gap-3">
                  <span className={`mt-1 size-3 shrink-0 rounded-full ${SOURCE_DOTS[gene.name] ?? "bg-muted-foreground"}`} />
                  <div className="min-w-0">
                    <div className="text-sm font-medium">{source.name}</div>
                    <div className="mt-1 text-xs text-muted-foreground">{source.description}</div>
                  </div>
                </div>
                <div className="mt-4 flex flex-wrap gap-2">
                  <Button
                    type="button"
                    size="sm"
                    variant={isTeams ? "outline" : "default"}
                    onClick={() => onConfigureSelected(gene.name)}
                  >
                    Configure
                  </Button>
                  {isTeams && (
                    <Button type="button" size="sm" onClick={onTeamsSelected}>
                      Browse Teams
                    </Button>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </DialogContent>
    </Dialog>
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
