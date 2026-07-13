import { useEffect, useMemo, useRef, useState } from "react";
import {
  AlertCircle,
  ChevronDown,
  ChevronRight,
  ChevronUp,
  FileText,
  Folder,
  FolderOpen,
  FolderTree,
  Loader2,
  RefreshCw,
  Search,
  X,
} from "lucide-react";
import { resourceClient } from "@/api/client";
import { createLocalAgentJob, getLocalAgentJob } from "@/api/localAgentJobs";
import type { GitHubRepoTreeResponse, LocalAgentJobStatusResponse } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import {
  normalizeRepoPickerPath,
  repoEffectiveFiles,
  repoPickerItemsFromFilePaths,
  repoPickerSelectionState,
  repoPickerTreeRows,
  repoScopeSummary,
  type RepoPickerItem,
  type RepoPickerSelectionState,
  type RepoPickerTreeRow,
  updateRepoPathSelection,
} from "./githubRepoFolderPickerUtils";

type ConfigValue = string | number | boolean | string[] | null;
type ScopeMode = "exclude" | "include";

const LOCAL_SCAN_LIMIT = 2_000;
const LOCAL_AGENT_POLL_ATTEMPTS = 180;
const LOCAL_AGENT_POLL_INTERVAL_MS = 1_000;
const EXCLUSION_SUGGESTION_PATTERN = /(^|\/)(archive|archived|deprecated|obsolete|outdated)(\/|$)/i;

export function GitHubRepoFolderPicker({
  connectionMode,
  sourceId,
  config,
  includePaths,
  excludePaths,
  onIncludePathsChange,
  onExcludePathsChange,
}: {
  connectionMode: string;
  sourceId?: string;
  config: Record<string, ConfigValue | undefined>;
  includePaths: string[];
  excludePaths: string[];
  onIncludePathsChange: (paths: string[]) => void;
  onExcludePathsChange: (paths: string[]) => void;
}) {
  const [items, setItems] = useState<RepoPickerItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [editorMode, setEditorMode] = useState<ScopeMode | null>(null);
  const [advancedOpen, setAdvancedOpen] = useState(includePaths.length > 0);
  const [query, setQuery] = useState("");
  const [expandedPaths, setExpandedPaths] = useState<Set<string>>(new Set());
  const [previewOpen, setPreviewOpen] = useState(false);
  const isLocalPush = connectionMode === "local_push";
  const selectedPaths = editorMode === "include" ? includePaths : excludePaths;
  const visibleTreeRows = useMemo(
    () => repoPickerTreeRows(items, expandedPaths, query),
    [expandedPaths, items, query],
  );
  const suggestedExclusions = useMemo(
    () => items
      .filter((item) => item.type === "tree" && EXCLUSION_SUGGESTION_PATTERN.test(item.path))
      .map((item) => item.path)
      .filter((path) => !excludePaths.includes(path))
      .slice(0, 5),
    [excludePaths, items],
  );
  const effectiveFiles = useMemo(
    () => repoEffectiveFiles(items, includePaths, excludePaths),
    [excludePaths, includePaths, items],
  );
  const scopeSummary = useMemo(
    () => repoScopeSummary(items, includePaths, excludePaths),
    [excludePaths, includePaths, items],
  );

  const applyTreeItems = (nextItems: RepoPickerItem[]) => {
    setItems(nextItems);
    setExpandedPaths(new Set(
      repoPickerTreeRows(nextItems, new Set(), "")
        .filter((row) => row.depth === 0 && row.item.type === "tree")
        .map((row) => row.item.path),
    ));
    setPreviewOpen(false);
  };

  const browseTree = async () => {
    setLoading(true);
    setMessage(null);
    const repoUrl = typeof config.repo_url === "string" ? config.repo_url.trim() : "";
    if (!repoUrl.startsWith("https://")) {
      setMessage("Enter a valid HTTPS Repository URL before browsing.");
      setLoading(false);
      return;
    }
    try {
      if (isLocalPush) {
        const created = await createLocalAgentJob({
          sourceType: "github_repo",
          operation: "github_repo_preview_tree",
          payload: { ...config, limit: LOCAL_SCAN_LIMIT },
        });
        const status = await pollLocalAgentJob(created.job_id);
        if (status.status === "failed") {
          setMessage(status.last_error || "Local sync could not load the remote repository tree.");
          return;
        }
        const paths = localAgentJobPaths(status);
        applyTreeItems(repoPickerItemsFromFilePaths(paths));
        if (status.result?.truncated || paths.length >= LOCAL_SCAN_LIMIT) {
          setMessage("Repository tree is large. Search for a folder or file to narrow the list.");
        }
      } else {
        const response = await resourceClient.post<GitHubRepoTreeResponse>("/genes/github_repo/browse-tree", {
          ...(sourceId ? { source_id: sourceId } : {}),
          config,
          limit: LOCAL_SCAN_LIMIT,
        });
        applyTreeItems(response.data.items);
        if (response.data.truncated) {
          setMessage("Repository tree is large. Search for a folder or file to narrow the list.");
        }
      }
    } catch (error) {
      setMessage(extractMessage(error) || (
        isLocalPush
          ? "Could not reach local sync. Check the daemon, VPN, and gh login."
          : "Could not load the repository tree."
      ));
    } finally {
      setLoading(false);
    }
  };

  const openEditor = (mode: ScopeMode) => {
    setEditorMode(mode);
    setQuery("");
    if (items.length === 0 && !loading) void browseTree();
  };

  const updateSelection = (path: string, selected: boolean) => {
    if (editorMode === "include") {
      onIncludePathsChange(updateRepoPathSelection(includePaths, path, selected));
    } else {
      onExcludePathsChange(updateRepoPathSelection(excludePaths, path, selected));
    }
  };

  const toggleExpanded = (path: string) => {
    setExpandedPaths((current) => {
      const next = new Set(current);
      if (next.has(path)) next.delete(path);
      else next.add(path);
      return next;
    });
  };

  return (
    <div className="space-y-3 rounded-xl border p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <FolderTree className="size-4 text-muted-foreground" />
            <p className="text-sm font-medium">
              {includePaths.length === 0
                ? "Sync all supported files in this repository"
                : "Sync only selected folders and files"}
            </p>
          </div>
          <p className="mt-1 text-xs text-muted-foreground">
            {excludePaths.length === 0
              ? "Nothing is excluded. You can leave outdated or archived areas out."
              : `${excludePaths.length} path${excludePaths.length === 1 ? " is" : "s are"} excluded.`}
          </p>
        </div>
        <Button type="button" variant="outline" size="sm" onClick={() => openEditor("exclude")}>
          Choose exclusions
        </Button>
      </div>

      <PathChips tone="exclude" paths={excludePaths} label="Excluded" onRemove={(path) => (
        onExcludePathsChange(updateRepoPathSelection(excludePaths, path, false))
      )} />

      <button
        type="button"
        className="flex w-full items-center justify-between border-t pt-3 text-left text-sm font-medium"
        onClick={() => setAdvancedOpen((open) => !open)}
      >
        <span>Sync only selected folders instead</span>
        {advancedOpen ? <ChevronUp className="size-4" /> : <ChevronDown className="size-4" />}
      </button>

      {advancedOpen && (
        <div className="rounded-lg bg-muted/50 p-3">
          <p className="text-xs text-muted-foreground">
            Use this when most of the repository is out of scope. Exclusions still win inside selected folders.
          </p>
          <div className="mt-3 flex flex-wrap gap-2">
            <Button type="button" variant="outline" size="sm" onClick={() => openEditor("include")}>
              Choose included folders
            </Button>
            {includePaths.length > 0 && (
              <Button type="button" variant="ghost" size="sm" onClick={() => onIncludePathsChange([])}>
                Reset to whole repository
              </Button>
            )}
          </div>
          <PathChips tone="include" paths={includePaths} label="Included" onRemove={(path) => (
            onIncludePathsChange(updateRepoPathSelection(includePaths, path, false))
          )} />
        </div>
      )}

      {editorMode && (
        <div className="rounded-lg border bg-background p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <p className="text-sm font-medium">
                {editorMode === "exclude" ? "Choose folders or files to exclude" : "Choose folders or files to include"}
              </p>
              <p className="text-xs text-muted-foreground">
                {isLocalPush ? "Loaded from the remote repository through local sync." : "Loaded from the remote repository by MemForge Cloud."}
              </p>
            </div>
            <div className="flex gap-2">
              <Button type="button" variant="outline" size="sm" onClick={() => void browseTree()} disabled={loading}>
                {loading ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
                Refresh
              </Button>
              <Button type="button" variant="ghost" size="sm" onClick={() => setEditorMode(null)}>Done</Button>
            </div>
          </div>

          {message && (
            <div className="mt-3 flex items-start gap-2 rounded-md bg-muted p-2 text-xs text-muted-foreground">
              <AlertCircle className="mt-0.5 size-3 shrink-0" />
              <span>{message}</span>
            </div>
          )}

          {items.length > 0 && (
            <>
              {editorMode === "exclude" && suggestedExclusions.length > 0 && (
                <div className="mt-3">
                  <p className="text-xs font-medium text-muted-foreground">Suggested exclusions</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {suggestedExclusions.map((path) => (
                      <button
                        key={path}
                        type="button"
                        className="rounded-full border px-2.5 py-1 font-mono text-xs hover:bg-muted"
                        onClick={() => onExcludePathsChange(updateRepoPathSelection(excludePaths, path, true))}
                      >
                        + {path}
                      </button>
                    ))}
                  </div>
                  <p className="mt-1 text-[11px] text-muted-foreground">Suggestions are never applied automatically.</p>
                </div>
              )}

              <div className="relative mt-3">
                <Search className="absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                <Input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search repository paths" className="pl-8" />
              </div>
              <div role="tree" aria-label="Repository paths" className="mt-3 max-h-72 overflow-y-auto rounded-md border">
                {visibleTreeRows.map((row) => (
                  <RepositoryTreeRow
                    key={`${row.item.type}:${row.item.path}`}
                    row={row}
                    mode={editorMode}
                    selectionState={repoPickerSelectionState(row.item.path, selectedPaths)}
                    expanded={query.trim().length > 0 || expandedPaths.has(row.item.path)}
                    onToggle={() => toggleExpanded(row.item.path)}
                    onSelected={(selected) => updateSelection(row.item.path, selected)}
                  />
                ))}
                {visibleTreeRows.length === 0 && (
                  <p className="px-3 py-6 text-center text-sm text-muted-foreground">No matching paths.</p>
                )}
              </div>
            </>
          )}
        </div>
      )}

      {items.length > 0 && (
        <div className="rounded-xl border border-emerald-200 bg-emerald-50/60 p-3.5 dark:border-emerald-900/70 dark:bg-emerald-950/20">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <p className="font-semibold text-emerald-950 dark:text-emerald-100">{scopeSummary.readyLabel}</p>
              <p className="mt-0.5 text-xs text-emerald-800/80 dark:text-emerald-200/80">{scopeSummary.detailLabel}</p>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              aria-expanded={previewOpen}
              onClick={() => setPreviewOpen((open) => !open)}
            >
              {previewOpen ? "Hide preview" : "Preview files"}
            </Button>
          </div>
          {previewOpen && (
            <div className="mt-3 max-h-48 overflow-y-auto rounded-lg border border-emerald-200 bg-background" aria-label="Files ready to sync">
              {effectiveFiles.map((item) => (
                <div key={item.path} className="flex items-center gap-2 border-b px-3 py-2 text-xs last:border-b-0">
                  <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                  <span className="min-w-0 truncate font-mono" title={item.path}>{item.path}</span>
                </div>
              ))}
              {effectiveFiles.length === 0 && (
                <p className="px-3 py-5 text-center text-sm text-muted-foreground">No supported files match this scope.</p>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function RepositoryTreeRow({
  row,
  mode,
  selectionState,
  expanded,
  onToggle,
  onSelected,
}: {
  row: RepoPickerTreeRow;
  mode: ScopeMode;
  selectionState: RepoPickerSelectionState;
  expanded: boolean;
  onToggle: () => void;
  onSelected: (selected: boolean) => void;
}) {
  const isFolder = row.item.type === "tree";
  const isCovered = selectionState === "selected" || selectionState === "inherited";
  const statusLabel = mode === "exclude" ? "Excluded" : "Included";
  return (
    <div
      role="treeitem"
      aria-level={row.depth + 1}
      aria-expanded={isFolder && row.hasChildren ? expanded : undefined}
      className={cn(
        "flex min-w-0 items-center gap-2 border-b py-2 pr-2.5 text-sm last:border-b-0",
        isCovered && (mode === "exclude" ? "bg-red-50/60 dark:bg-red-950/20" : "bg-emerald-50/60 dark:bg-emerald-950/20"),
      )}
      style={{ paddingLeft: `${10 + row.depth * 22}px` }}
    >
      {isFolder && row.hasChildren ? (
        <button
          type="button"
          className="grid size-5 shrink-0 place-items-center rounded hover:bg-muted"
          aria-label={`${expanded ? "Collapse" : "Expand"} ${row.item.path}`}
          aria-expanded={expanded}
          onClick={onToggle}
        >
          {expanded ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
        </button>
      ) : <span className="size-5 shrink-0" />}
      <TreeSelectionCheckbox
        path={row.item.path}
        mode={mode}
        state={selectionState}
        onSelected={onSelected}
      />
      {isFolder ? (
        expanded
          ? <FolderOpen className="size-4 shrink-0 text-muted-foreground" />
          : <Folder className="size-4 shrink-0 text-muted-foreground" />
      ) : (
        <FileText className="size-4 shrink-0 text-muted-foreground" />
      )}
      <div className="min-w-0 flex-1">
        <p className="truncate font-mono text-xs font-medium" title={row.item.path}>{row.name}</p>
        <p className="text-[11px] text-muted-foreground">
          {isFolder ? `${row.fileCount} file${row.fileCount === 1 ? "" : "s"}` : "File"}
        </p>
      </div>
      {isCovered && (
        <span
          className={cn(
            "shrink-0 rounded-full px-2 py-0.5 text-[11px] font-semibold",
            mode === "exclude"
              ? "bg-red-100 text-red-700 dark:bg-red-950/60 dark:text-red-200"
              : "bg-emerald-100 text-emerald-700 dark:bg-emerald-950/60 dark:text-emerald-200",
          )}
          title={selectionState === "inherited" ? `${statusLabel} by a selected parent folder` : undefined}
        >
          {statusLabel}
        </span>
      )}
      {selectionState === "partial" && (
        <span className="shrink-0 rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-semibold text-amber-800 dark:bg-amber-950/60 dark:text-amber-200">
          Partial
        </span>
      )}
    </div>
  );
}

function TreeSelectionCheckbox({
  path,
  mode,
  state,
  onSelected,
}: {
  path: string;
  mode: ScopeMode;
  state: RepoPickerSelectionState;
  onSelected: (selected: boolean) => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  const inherited = state === "inherited";
  useEffect(() => {
    if (ref.current) ref.current.indeterminate = state === "partial";
  }, [state]);
  return (
    <input
      ref={ref}
      type="checkbox"
      className={cn("size-4 shrink-0", mode === "exclude" ? "accent-red-600" : "accent-emerald-600")}
      aria-label={path}
      aria-checked={state === "partial" ? "mixed" : state === "selected" || inherited}
      checked={state === "selected" || inherited}
      disabled={inherited}
      title={inherited ? "Selected by a parent folder" : undefined}
      onChange={(event) => onSelected(event.target.checked)}
    />
  );
}

function PathChips({
  paths,
  label,
  tone,
  onRemove,
}: {
  paths: string[];
  label: string;
  tone: ScopeMode;
  onRemove: (path: string) => void;
}) {
  if (paths.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs text-muted-foreground">{label}</span>
      {paths.map((path) => (
        <button
          key={path}
          type="button"
          className={cn(
            "flex max-w-full items-center gap-1 rounded-full border px-2.5 py-1 font-mono text-xs",
            tone === "exclude"
              ? "border-red-200 bg-red-50 text-red-700 hover:bg-red-100 dark:border-red-900/70 dark:bg-red-950/40 dark:text-red-200"
              : "border-emerald-200 bg-emerald-50 text-emerald-700 hover:bg-emerald-100 dark:border-emerald-900/70 dark:bg-emerald-950/40 dark:text-emerald-200",
          )}
          onClick={() => onRemove(path)}
          title={`Remove ${path}`}
        >
          <span className="truncate">{path}</span>
          <X className="size-3 shrink-0" />
        </button>
      ))}
    </div>
  );
}

async function pollLocalAgentJob(jobId: string): Promise<LocalAgentJobStatusResponse> {
  for (let attempt = 0; attempt < LOCAL_AGENT_POLL_ATTEMPTS; attempt += 1) {
    const status = await getLocalAgentJob(jobId);
    if (status.status === "succeeded" || status.status === "failed") return status;
    await wait(LOCAL_AGENT_POLL_INTERVAL_MS);
  }
  throw new Error("Timed out waiting for local sync.");
}

function localAgentJobPaths(status: LocalAgentJobStatusResponse): string[] {
  return (status.result?.items ?? [])
    .map((item) => normalizeRepoPickerPath(item.relative_path ?? item.path ?? ""))
    .filter((path) => path.length > 0);
}

function wait(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function extractMessage(error: unknown): string {
  if (typeof error === "object" && error && "response" in error) {
    const response = (error as { response?: { data?: { detail?: unknown } } }).response;
    if (typeof response?.data?.detail === "string") return response.data.detail;
  }
  return error instanceof Error ? error.message : "";
}
