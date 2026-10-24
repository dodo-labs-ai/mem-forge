import { ChevronDown, ChevronRight, FileText, Folder, FolderOpen, FolderTree, Loader2, RefreshCw, Search } from "lucide-react";
import { useMemo, useState } from "react";
import { cn } from "@/lib/cn";
import { errorMessage } from "@/lib/errors";
import { pluralize } from "@/lib/format";
import { Button } from "@/ui/button";
import { Checkbox } from "@/ui/checkbox";
import { Input } from "@/ui/input";
import { TagInput } from "@/ui/tag-input";
import { useRepoTree } from "../githubRepoTree";
import {
  normalizeRepoPath,
  repoScopeSummary,
  repoSelectionState,
  repoTreeRows,
  suggestedExclusions,
  topLevelFolders,
  updateRepoPathSelection,
  type RepoTreeRow,
} from "../model/githubRepoPaths";
import type { ConfigControlProps } from "../widgets";

type PathMode = "include" | "exclude";

function listOf(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

/**
 * Chooses a GitHub repository Source's `include_paths` or `exclude_paths`
 * from the repository's tree. The paths show as chips, which can also be
 * typed; "Browse repository" lists the folders and files to tick. The
 * exclude picker suggests outdated-looking folders, never applying them by
 * itself, and counts the files both lists leave to sync.
 */
export default function GitHubRepoPathsPicker({
  field,
  id,
  labelId,
  describedBy,
  value,
  onChange,
  onBlur,
  invalid,
  readOnly,
  config,
  sourceId,
}: ConfigControlProps) {
  const mode: PathMode = field.key === "exclude_paths" ? "exclude" : "include";
  const paths = Array.isArray(value) ? value : [];
  const [browsing, setBrowsing] = useState(false);
  const [query, setQuery] = useState("");
  const [expanded, setExpanded] = useState<ReadonlySet<string> | null>(null);
  const tree = useRepoTree(config, sourceId, browsing);
  const items = useMemo(() => tree.data?.items ?? [], [tree.data]);
  const openFolders = expanded ?? topLevelFolders(items);
  const rows = useMemo(() => repoTreeRows(items, openFolders, query), [items, openFolders, query]);
  const includePaths = mode === "include" ? paths : listOf(config.include_paths);
  const excludePaths = mode === "exclude" ? paths : listOf(config.exclude_paths);
  const suggestions = mode === "exclude" ? suggestedExclusions(items, paths) : [];
  const repoUrl = typeof config.repo_url === "string" ? config.repo_url.trim() : "";

  function change(next: string[]) {
    onChange(next.map(normalizeRepoPath).filter(Boolean));
    onBlur?.();
  }

  function toggleFolder(path: string) {
    const next = new Set(openFolders);
    if (next.has(path)) next.delete(path);
    else next.add(path);
    setExpanded(next);
  }

  return (
    <div className="space-y-3">
      <TagInput
        id={id}
        aria-labelledby={labelId}
        aria-describedby={describedBy}
        aria-invalid={invalid}
        value={paths}
        placeholder={field.placeholder}
        disabled={readOnly}
        onValueChange={change}
        onBlur={onBlur}
      />
      {readOnly ? null : !browsing ? (
        <Button type="button" size="sm" variant="outline" disabled={!repoUrl} onClick={() => setBrowsing(true)}>
          <FolderTree aria-hidden />
          Browse repository
        </Button>
      ) : (
        <div className="space-y-3 rounded-lg border p-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <p className="text-sm font-medium">{mode === "exclude" ? "Choose folders or files to leave out" : "Choose folders or files to sync"}</p>
            <div className="flex gap-2">
              <Button type="button" size="sm" variant="outline" disabled={tree.isFetching} onClick={() => void tree.refetch()}>
                {tree.isFetching ? <Loader2 aria-hidden className="animate-spin" /> : <RefreshCw aria-hidden />}
                Refresh
              </Button>
              <Button type="button" size="sm" variant="ghost" onClick={() => setBrowsing(false)}>
                Done
              </Button>
            </div>
          </div>
          {tree.isPending ? (
            <p className="flex items-center gap-2 py-4 text-sm text-muted-foreground">
              <Loader2 aria-hidden className="size-4 animate-spin" />
              Listing the repository…
            </p>
          ) : tree.isError ? (
            <p role="alert" className="text-sm text-tone-danger-foreground">
              Could not list the repository. {errorMessage(tree.error)}
            </p>
          ) : (
            <>
              {tree.data.truncated ? (
                <p className="text-xs text-muted-foreground">The repository is large, so only part of it is listed. Search to find a path.</p>
              ) : null}
              {suggestions.length > 0 ? (
                <div className="space-y-1.5">
                  <p className="text-xs font-medium text-muted-foreground">Suggested to leave out</p>
                  <div className="flex flex-wrap gap-1.5">
                    {suggestions.map((path) => (
                      <Button key={path} type="button" size="xs" variant="outline" className="font-mono" onClick={() => change(updateRepoPathSelection(paths, path, true))}>
                        + {path}
                      </Button>
                    ))}
                  </div>
                </div>
              ) : null}
              <div className="relative">
                <Search aria-hidden className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
                <Input aria-label="Search repository paths" className="pl-8" placeholder="Search repository paths" value={query} onChange={(event) => setQuery(event.target.value)} />
              </div>
              <div role="tree" aria-label="Repository paths" className="max-h-72 overflow-y-auto rounded-md border">
                {rows.length === 0 ? (
                  <p className="px-3 py-6 text-center text-sm text-muted-foreground">No matching paths.</p>
                ) : (
                  rows.map((row) => (
                    <TreeRow
                      key={`${row.item.type}:${row.item.path}`}
                      row={row}
                      mode={mode}
                      paths={paths}
                      expanded={query.trim() !== "" || openFolders.has(row.item.path)}
                      onToggleFolder={() => toggleFolder(row.item.path)}
                      onSelected={(selected) => change(updateRepoPathSelection(paths, row.item.path, selected))}
                    />
                  ))
                )}
              </div>
            </>
          )}
        </div>
      )}
      {mode === "exclude" && items.length > 0 ? <ScopeSummary {...repoScopeSummary(items, includePaths, excludePaths)} /> : null}
    </div>
  );
}

function ScopeSummary({ readyLabel, detailLabel }: { readyLabel: string; detailLabel: string }) {
  return (
    <div role="status" className="rounded-lg bg-tone-ok-soft px-3 py-2 text-sm text-tone-ok-foreground">
      <p className="font-medium">{readyLabel}</p>
      <p className="text-xs">{detailLabel}</p>
    </div>
  );
}

interface TreeRowProps {
  row: RepoTreeRow;
  mode: PathMode;
  paths: readonly string[];
  expanded: boolean;
  onToggleFolder: () => void;
  onSelected: (selected: boolean) => void;
}

function TreeRow({ row, mode, paths, expanded, onToggleFolder, onSelected }: TreeRowProps) {
  const state = repoSelectionState(row.item.path, paths);
  const folder = row.item.type === "tree";
  const covered = state === "selected" || state === "inherited";
  const FolderIcon = expanded ? FolderOpen : Folder;
  return (
    <div
      role="treeitem"
      aria-level={row.depth + 1}
      aria-expanded={folder && row.hasChildren ? expanded : undefined}
      aria-selected={covered}
      className={cn(
        "flex min-w-0 items-center gap-2 border-b py-1.5 pr-2.5 text-sm last:border-b-0",
        covered && (mode === "exclude" ? "bg-tone-danger-soft" : "bg-tone-ok-soft"),
      )}
      style={{ paddingLeft: `${8 + row.depth * 20}px` }}
    >
      {folder && row.hasChildren ? (
        <button
          type="button"
          className="grid size-5 shrink-0 place-items-center rounded hover:bg-muted"
          aria-label={`${expanded ? "Collapse" : "Expand"} ${row.item.path}`}
          onClick={onToggleFolder}
        >
          {expanded ? <ChevronDown aria-hidden className="size-3.5" /> : <ChevronRight aria-hidden className="size-3.5" />}
        </button>
      ) : (
        <span className="size-5 shrink-0" />
      )}
      <Checkbox
        aria-label={row.item.path}
        checked={covered}
        indeterminate={state === "partial"}
        disabled={state === "inherited"}
        onCheckedChange={(checked) => onSelected(checked)}
      />
      {folder ? <FolderIcon aria-hidden className="size-4 shrink-0 text-muted-foreground" /> : <FileText aria-hidden className="size-4 shrink-0 text-muted-foreground" />}
      <span className="min-w-0 flex-1 truncate font-mono text-xs" title={row.item.path}>
        {row.name}
      </span>
      <span className="shrink-0 text-xs text-muted-foreground">{folder ? pluralize(row.fileCount, "file") : null}</span>
    </div>
  );
}
