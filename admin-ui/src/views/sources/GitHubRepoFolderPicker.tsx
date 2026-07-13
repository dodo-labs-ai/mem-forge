import { useMemo, useState } from "react";
import {
  AlertCircle,
  ChevronDown,
  ChevronUp,
  FileText,
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
import {
  normalizeRepoPickerPath,
  repoEffectiveFileCount,
  repoPickerItemsFromFilePaths,
  type RepoPickerItem,
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
  const isLocalPush = connectionMode === "local_push";
  const selectedPaths = editorMode === "include" ? includePaths : excludePaths;
  const filteredItems = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    return needle ? items.filter((item) => item.path.toLocaleLowerCase().includes(needle)) : items;
  }, [items, query]);
  const suggestedExclusions = useMemo(
    () => items
      .filter((item) => item.type === "tree" && EXCLUSION_SUGGESTION_PATTERN.test(item.path))
      .map((item) => item.path)
      .filter((path) => !excludePaths.includes(path))
      .slice(0, 5),
    [excludePaths, items],
  );
  const effectiveFileCount = useMemo(
    () => repoEffectiveFileCount(items, includePaths, excludePaths),
    [excludePaths, includePaths, items],
  );

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
        setItems(repoPickerItemsFromFilePaths(paths));
        if (status.result?.truncated || paths.length >= LOCAL_SCAN_LIMIT) {
          setMessage("Repository tree is large. Search for a folder or file to narrow the list.");
        }
      } else {
        const response = await resourceClient.post<GitHubRepoTreeResponse>("/genes/github_repo/browse-tree", {
          ...(sourceId ? { source_id: sourceId } : {}),
          config,
          limit: LOCAL_SCAN_LIMIT,
        });
        setItems(response.data.items);
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

      <PathChips paths={excludePaths} label="Excluded" onRemove={(path) => (
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
          <PathChips paths={includePaths} label="Included" onRemove={(path) => (
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
              <div className="mt-3 max-h-64 overflow-y-auto rounded-md border">
                {filteredItems.map((item) => (
                  <label key={`${item.type}:${item.path}`} className="flex min-w-0 items-center gap-2 border-b px-2.5 py-2 text-sm last:border-b-0">
                    <input
                      type="checkbox"
                      className="size-4 shrink-0"
                      checked={selectedPaths.includes(item.path)}
                      onChange={(event) => updateSelection(item.path, event.target.checked)}
                    />
                    {item.type === "tree" ? (
                      <FolderTree className="size-3.5 shrink-0 text-muted-foreground" />
                    ) : (
                      <FileText className="size-3.5 shrink-0 text-muted-foreground" />
                    )}
                    <span className="min-w-0 truncate font-mono text-xs">{item.path}</span>
                  </label>
                ))}
                {filteredItems.length === 0 && (
                  <p className="px-3 py-6 text-center text-sm text-muted-foreground">No matching paths.</p>
                )}
              </div>
              <p className="mt-2 text-xs text-muted-foreground">
                {effectiveFileCount} supported file{effectiveFileCount === 1 ? "" : "s"} currently in scope.
              </p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

function PathChips({ paths, label, onRemove }: { paths: string[]; label: string; onRemove: (path: string) => void }) {
  if (paths.length === 0) return null;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs text-muted-foreground">{label}</span>
      {paths.map((path) => (
        <button
          key={path}
          type="button"
          className="flex max-w-full items-center gap-1 rounded-full border bg-muted/60 px-2.5 py-1 font-mono text-xs"
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
