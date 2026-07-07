import { useMemo, useState } from "react";
import { AlertCircle, FileText, FolderTree, Loader2, RefreshCw } from "lucide-react";
import client from "@/api/client";
import type { GitHubRepoTreeResponse } from "@/api/types";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  normalizeRepoPickerPath,
  repoPickerItemsFromFilePaths,
  type RepoPickerItem,
  updateRepoPathSelection,
} from "./githubRepoFolderPickerUtils";

type ConfigValue = string | number | boolean | string[] | null;

interface BrowserFileHandle {
  kind: "file";
  name: string;
}

interface BrowserDirectoryHandle {
  kind: "directory";
  name: string;
  values(): AsyncIterable<BrowserFileHandle | BrowserDirectoryHandle>;
}

interface BrowserWindow extends Window {
  showDirectoryPicker?: () => Promise<BrowserDirectoryHandle>;
}

const LOCAL_SCAN_LIMIT = 2_000;
const SKIPPED_DIRECTORIES = new Set([".git", "node_modules", ".venv", "venv", "dist", "build", ".next"]);

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
  const canPickLocalDirectory =
    typeof window !== "undefined" && typeof (window as BrowserWindow).showDirectoryPicker === "function";

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

  const browseLocalClone = async () => {
    if (!canPickLocalDirectory) {
      setMessage("Local folder selection is not available in this browser.");
      return;
    }
    setLoading(true);
    setMessage(null);
    try {
      const root = await (window as BrowserWindow).showDirectoryPicker!();
      const paths = await scanLocalDirectory(root);
      setItems(repoPickerItemsFromFilePaths(paths));
      if (paths.length >= LOCAL_SCAN_LIMIT) {
        setMessage("Local scan reached the preview limit. Choose a top-level folder or add a precise path.");
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setMessage(extractMessage(error) || "Could not scan local folder.");
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
          onClick={isLocalPush ? browseLocalClone : browseCloudTree}
          disabled={loading}
        >
          {loading ? <Loader2 className="size-3.5 animate-spin" /> : <RefreshCw className="size-3.5" />}
          {isLocalPush ? "Choose clone" : "Browse repo"}
        </Button>
      </div>

      {isLocalPush && (
        <p className="mt-2 text-xs text-muted-foreground">
          Pick folders from your clone. The sync command still needs the clone path.
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

      <details className="mt-3" open={isLocalPush && !canPickLocalDirectory}>
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

async function scanLocalDirectory(root: BrowserDirectoryHandle): Promise<string[]> {
  const paths: string[] = [];
  await scanDirectory(root, "", paths);
  return paths;
}

async function scanDirectory(handle: BrowserDirectoryHandle, prefix: string, paths: string[]): Promise<void> {
  if (paths.length >= LOCAL_SCAN_LIMIT) return;
  for await (const child of handle.values()) {
    if (paths.length >= LOCAL_SCAN_LIMIT) return;
    const path = prefix ? `${prefix}/${child.name}` : child.name;
    if (child.kind === "directory") {
      if (SKIPPED_DIRECTORIES.has(child.name)) continue;
      await scanDirectory(child, path, paths);
    } else {
      paths.push(path);
    }
  }
}

function extractMessage(error: unknown): string {
  if (typeof error === "object" && error && "response" in error) {
    const response = (error as { response?: { data?: { detail?: unknown } } }).response;
    if (typeof response?.data?.detail === "string") return response.data.detail;
  }
  return error instanceof Error ? error.message : "";
}
