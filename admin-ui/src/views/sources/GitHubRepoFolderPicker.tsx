import { useMemo, useState } from "react";
import { AlertCircle, FileText, FolderTree, Loader2, RefreshCw } from "lucide-react";
import client from "@/api/client";
import type { GitHubRepoTreeResponse, LocalAgentJobCreateResponse, LocalAgentJobStatusResponse } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  normalizeRepoPickerPath,
  repoPickerItemsFromFilePaths,
  type RepoPickerItem,
  updateRepoPathSelection,
} from "./githubRepoFolderPickerUtils";

type ConfigValue = string | number | boolean | string[] | null;

const LOCAL_SCAN_LIMIT = 2_000;
const LOCAL_AGENT_POLL_ATTEMPTS = 180;
const LOCAL_AGENT_POLL_INTERVAL_MS = 1_000;

export function GitHubRepoFolderPicker({
  connectionMode,
  config,
  value,
  onChange,
}: {
  connectionMode: string;
  config: Record<string, ConfigValue | undefined>;
  value: string[];
  onChange: (paths: string[]) => void;
}) {
  const [items, setItems] = useState<RepoPickerItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [manualPath, setManualPath] = useState("");
  const selected = useMemo(() => new Set(value.map(normalizeRepoPickerPath)), [value]);
  const isLocalPush = connectionMode === "local_push";

  const browseCloudTree = async () => {
    setLoading(true);
    setMessage(null);
    try {
      const response = await client.post<GitHubRepoTreeResponse>("/api/genes/github_repo/browse-tree", {
        config,
        limit: LOCAL_SCAN_LIMIT,
      });
      setItems(response.data.items);
      if (response.data.truncated) {
        setMessage("Repository tree is large. Narrow the selection before syncing.");
      }
    } catch (error) {
      setMessage(extractMessage(error) || "Could not load repository tree.");
    } finally {
      setLoading(false);
    }
  };

  const browseLocalAgentTree = async () => {
    setLoading(true);
    setMessage(null);
    try {
      const created = await client.post<LocalAgentJobCreateResponse>("/api/cloud/local-agent/jobs", {
        source_type: "github_repo",
        operation: "github_repo_preview_tree",
        payload: {
          ...config,
          limit: LOCAL_SCAN_LIMIT,
        },
      });
      const status = await pollLocalAgentJob(created.data.job_id);
      if (status.status === "failed") {
        setMessage(status.last_error || "Local daemon could not load repository folders.");
        return;
      }
      const paths = localAgentJobPaths(status);
      setItems(repoPickerItemsFromFilePaths(paths));
      if (status.result?.truncated || paths.length >= LOCAL_SCAN_LIMIT) {
        setMessage("Repository tree is large. Narrow the selection before syncing.");
      }
    } catch (error) {
      setMessage(extractMessage(error) || "Could not reach the local daemon. Start memforge adapter daemon run.");
    } finally {
      setLoading(false);
    }
  };

  const addManualPath = () => {
    const normalized = normalizeRepoPickerPath(manualPath);
    if (!normalized) return;
    onChange(updateRepoPathSelection(value, normalized, true));
    setManualPath("");
  };

  return (
    <div className="rounded-lg border p-3">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex min-w-0 items-center gap-2">
          <FolderTree className="size-4 text-muted-foreground" />
          <span className="text-sm font-medium">Folders and files</span>
        </div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={isLocalPush ? browseLocalAgentTree : browseCloudTree}
          disabled={loading}
        >
          {loading ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
          {isLocalPush ? "Fetch folders" : "Browse repo"}
        </Button>
      </div>

      {isLocalPush && (
        <p className="mt-2 text-xs text-muted-foreground">
          Start the local daemon, then fetch folders from the repository.
        </p>
      )}

      {message && (
        <div className="mt-3 flex items-start gap-2 rounded-md bg-muted p-2 text-xs text-muted-foreground">
          <AlertCircle className="mt-0.5 size-3 shrink-0" />
          <span>{message}</span>
        </div>
      )}

      <SelectedPaths paths={value} onRemove={(path) => onChange(updateRepoPathSelection(value, path, false))} />

      {items.length > 0 && (
        <div className="mt-3 max-h-56 overflow-y-auto rounded-md border">
          {items.map((item) => (
            <label key={`${item.type}:${item.path}`} className="flex min-w-0 items-center gap-2 border-b px-2 py-1.5 text-sm last:border-b-0">
              <input
                type="checkbox"
                className="size-4 shrink-0"
                checked={selected.has(item.path)}
                onChange={(event) => onChange(updateRepoPathSelection(value, item.path, event.target.checked))}
              />
              {item.type === "tree" ? (
                <FolderTree className="size-3.5 shrink-0 text-muted-foreground" />
              ) : (
                <FileText className="size-3.5 shrink-0 text-muted-foreground" />
              )}
              <span className="min-w-0 truncate font-mono text-xs">{item.path}</span>
            </label>
          ))}
        </div>
      )}

      <details className="mt-3">
        <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
          Type a path instead
        </summary>
        <div className="mt-2 flex gap-2">
          <Input
            value={manualPath}
            onChange={(event) => setManualPath(event.target.value)}
            placeholder="repo/folder or repo/file.md"
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.preventDefault();
                addManualPath();
              }
            }}
          />
          <Button type="button" variant="outline" size="sm" onClick={addManualPath}>
            Add
          </Button>
        </div>
      </details>
    </div>
  );
}

function SelectedPaths({ paths, onRemove }: { paths: string[]; onRemove: (path: string) => void }) {
  if (paths.length === 0) {
    return <p className="mt-2 text-xs text-muted-foreground">No scope selected. Sync uses the whole repository.</p>;
  }
  return (
    <div className="mt-3 flex flex-wrap gap-2">
      {paths.map((path) => (
        <button
          key={path}
          type="button"
          className="max-w-full truncate rounded-md border bg-muted px-2 py-1 font-mono text-xs"
          onClick={() => onRemove(path)}
          title="Remove"
        >
          {path}
        </button>
      ))}
    </div>
  );
}

async function pollLocalAgentJob(jobId: string): Promise<LocalAgentJobStatusResponse> {
  for (let attempt = 0; attempt < LOCAL_AGENT_POLL_ATTEMPTS; attempt += 1) {
    const response = await client.get<LocalAgentJobStatusResponse>(`/api/cloud/local-agent/jobs/${jobId}`);
    if (response.data.status === "succeeded" || response.data.status === "failed") {
      return response.data;
    }
    await wait(LOCAL_AGENT_POLL_INTERVAL_MS);
  }
  throw new Error("Timed out waiting for the local daemon.");
}

function localAgentJobPaths(status: LocalAgentJobStatusResponse): string[] {
  const items = status.result?.items ?? [];
  return items
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
