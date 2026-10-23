import { pluralize } from "@/lib/format";

/**
 * Repository paths for a GitHub repository Source's folder picker.
 *
 * A Source reads the files under its `include_paths` (the whole repository
 * when there are none), minus everything under its `exclude_paths`, so an
 * exclusion wins inside an included folder. A path names a folder or a file,
 * relative to the repository root, with `/` between its parts.
 *
 * Example: with `include_paths` `["docs"]` and `exclude_paths`
 * `["docs/archive"]`, `docs/guide.md` is read and `docs/archive/old.md` and
 * `README.md` are not.
 */

/** One folder (`tree`) or file (`blob`) of the repository. */
export interface RepoItem {
  path: string;
  type: "tree" | "blob";
  size?: number | null;
}

/** A row of the picker's tree: a folder or file, how deep it sits, and how many files it holds. */
export interface RepoTreeRow {
  item: RepoItem;
  name: string;
  depth: number;
  fileCount: number;
  hasChildren: boolean;
}

/**
 * How a selection covers a path: chosen itself, covered by a chosen folder
 * above it, holding a chosen path below it, or none of these.
 */
export type RepoSelectionState = "selected" | "inherited" | "partial" | "unselected";

interface RepoTreeNode {
  item: RepoItem;
  children: RepoTreeNode[];
}

/** Folders whose names suggest leaving them out, offered as exclusions and never applied by themselves. */
const EXCLUSION_SUGGESTION = /(^|\/)(archive|archived|deprecated|obsolete|outdated)(\/|$)/i;
const MAX_SUGGESTIONS = 5;

/** A path as the Source stores it: forward slashes, no leading or trailing slash. */
export function normalizeRepoPath(value: string): string {
  return value.trim().replace(/\\/g, "/").replace(/^\/+/, "").replace(/\/+$/, "");
}

/**
 * The selection with `path` chosen or cleared. Choosing a folder drops the
 * chosen paths inside it, and choosing a path a chosen folder already covers
 * changes nothing. The result is sorted.
 */
export function updateRepoPathSelection(current: readonly string[], path: string, selected: boolean): string[] {
  const normalized = normalizeRepoPath(path);
  if (!normalized) return [...current];
  const next = new Set(current.map(normalizeRepoPath).filter(Boolean));
  if (selected) {
    for (const existing of next) {
      if (existing === normalized || existing.startsWith(`${normalized}/`)) next.delete(existing);
    }
    if (![...next].some((existing) => normalized.startsWith(`${existing}/`))) next.add(normalized);
  } else {
    next.delete(normalized);
  }
  return [...next].sort((a, b) => a.localeCompare(b));
}

/** Whether a chosen path is `path` or a folder above it. */
export function pathIsCoveredBySelection(path: string, selections: readonly string[]): boolean {
  const normalized = normalizeRepoPath(path);
  return selections.some((selection) => {
    const scope = normalizeRepoPath(selection);
    return normalized === scope || normalized.startsWith(`${scope}/`);
  });
}

export function repoSelectionState(path: string, selections: readonly string[]): RepoSelectionState {
  const normalized = normalizeRepoPath(path);
  const scopes = selections.map(normalizeRepoPath).filter(Boolean);
  if (scopes.includes(normalized)) return "selected";
  if (scopes.some((scope) => normalized.startsWith(`${scope}/`))) return "inherited";
  if (scopes.some((scope) => scope.startsWith(`${normalized}/`))) return "partial";
  return "unselected";
}

/** The files the Source would read: under an included path (any, when none is), and under no excluded path. */
export function repoEffectiveFiles(items: readonly RepoItem[], includePaths: readonly string[], excludePaths: readonly string[]): RepoItem[] {
  return items.filter(
    (item) =>
      item.type === "blob" &&
      (includePaths.length === 0 || pathIsCoveredBySelection(item.path, includePaths)) &&
      !pathIsCoveredBySelection(item.path, excludePaths),
  );
}

/** "4 files ready to sync", and what the paths leave out. */
export function repoScopeSummary(
  items: readonly RepoItem[],
  includePaths: readonly string[],
  excludePaths: readonly string[],
): { readyLabel: string; detailLabel: string } {
  const total = items.filter((item) => item.type === "blob").length;
  const ready = repoEffectiveFiles(items, includePaths, excludePaths).length;
  const left = total - ready;
  const exclusions = pluralize(excludePaths.length, "exclusion");
  let detailLabel: string;
  if (includePaths.length > 0) detailLabel = `${pluralize(left, "file")} outside the selected paths · ${exclusions}`;
  else if (left > 0) detailLabel = `${pluralize(left, "file")} left out by ${exclusions}`;
  else detailLabel = total === 1 ? "The one file is included." : `All ${pluralize(total, "file")} are included.`;
  return { readyLabel: `${pluralize(ready, "file")} ready to sync`, detailLabel };
}

/** Folders that look outdated and are not excluded yet, at most five. */
export function suggestedExclusions(items: readonly RepoItem[], excludePaths: readonly string[]): string[] {
  return items
    .filter((item) => item.type === "tree" && EXCLUSION_SUGGESTION.test(item.path) && !pathIsCoveredBySelection(item.path, excludePaths))
    .map((item) => item.path)
    .slice(0, MAX_SUGGESTIONS);
}

/** The folders and files of a list of file paths, as a local sync preview reports them: folders first, each sorted. */
export function repoItemsFromFilePaths(paths: readonly string[]): RepoItem[] {
  const folders = new Set<string>();
  const files = new Set<string>();
  for (const raw of paths) {
    const path = normalizeRepoPath(raw);
    if (!path) continue;
    files.add(path);
    const parts = path.split("/").slice(0, -1);
    for (let index = 1; index <= parts.length; index += 1) folders.add(parts.slice(0, index).join("/"));
  }
  return [
    ...[...folders].sort((a, b) => a.localeCompare(b)).map((path) => ({ path, type: "tree" as const })),
    ...[...files].sort((a, b) => a.localeCompare(b)).map((path) => ({ path, type: "blob" as const })),
  ];
}

/**
 * The rows the tree shows: top-level folders and files, and the children of
 * each expanded folder, folders before files. A search shows every path that
 * contains it, with its whole chain of folders above it.
 */
export function repoTreeRows(items: readonly RepoItem[], expandedPaths: ReadonlySet<string>, query: string): RepoTreeRow[] {
  const needle = query.trim().toLocaleLowerCase();
  const rows: RepoTreeRow[] = [];

  const visit = (node: RepoTreeNode, depth: number) => {
    const children = needle ? node.children.filter((child) => treeContains(child, needle)) : node.children;
    if (needle && !node.item.path.toLocaleLowerCase().includes(needle) && children.length === 0) return;
    rows.push({
      item: node.item,
      name: pathName(node.item.path),
      depth,
      fileCount: fileCount(node),
      hasChildren: node.children.length > 0,
    });
    if (needle || expandedPaths.has(node.item.path)) for (const child of children) visit(child, depth + 1);
  };

  for (const root of buildTree(items)) visit(root, 0);
  return rows;
}

/** The top-level folders, which the tree opens expanded. */
export function topLevelFolders(items: readonly RepoItem[]): Set<string> {
  return new Set(
    repoTreeRows(items, new Set(), "")
      .filter((row) => row.item.type === "tree")
      .map((row) => row.item.path),
  );
}

function buildTree(items: readonly RepoItem[]): RepoTreeNode[] {
  const byPath = new Map<string, RepoItem>();
  for (const item of items) {
    const path = normalizeRepoPath(item.path);
    if (!path) continue;
    byPath.set(path, { ...item, path });
    const parts = path.split("/");
    const folderParts = item.type === "tree" ? parts : parts.slice(0, -1);
    for (let index = 1; index <= folderParts.length; index += 1) {
      const folder = folderParts.slice(0, index).join("/");
      if (!byPath.has(folder)) byPath.set(folder, { path: folder, type: "tree" });
    }
  }
  const nodes = new Map<string, RepoTreeNode>();
  for (const item of byPath.values()) nodes.set(item.path, { item, children: [] });
  const roots: RepoTreeNode[] = [];
  for (const node of nodes.values()) {
    const parent = nodes.get(parentPath(node.item.path));
    if (parent) parent.children.push(node);
    else roots.push(node);
  }
  for (const node of nodes.values()) node.children.sort(compareNodes);
  return roots.sort(compareNodes);
}

function treeContains(node: RepoTreeNode, needle: string): boolean {
  return node.item.path.toLocaleLowerCase().includes(needle) || node.children.some((child) => treeContains(child, needle));
}

function fileCount(node: RepoTreeNode): number {
  if (node.item.type === "blob") return 1;
  return node.children.reduce((total, child) => total + fileCount(child), 0);
}

function parentPath(path: string): string {
  const separator = path.lastIndexOf("/");
  return separator < 0 ? "" : path.slice(0, separator);
}

function pathName(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1);
}

function compareNodes(left: RepoTreeNode, right: RepoTreeNode): number {
  if (left.item.type !== right.item.type) return left.item.type === "tree" ? -1 : 1;
  return left.item.path.localeCompare(right.item.path);
}
