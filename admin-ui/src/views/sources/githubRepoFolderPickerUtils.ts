export interface RepoPickerItem {
  path: string;
  type: "tree" | "blob";
  size: number | null;
}

export interface RepoPickerTreeRow {
  item: RepoPickerItem;
  name: string;
  depth: number;
  fileCount: number;
  hasChildren: boolean;
}

export type RepoPickerSelectionState = "selected" | "inherited" | "partial" | "unselected";

export interface RepoScopeSummary {
  readyCount: number;
  totalCount: number;
  filteredCount: number;
  readyLabel: string;
  detailLabel: string;
}

interface RepoPickerTreeNode {
  item: RepoPickerItem;
  children: RepoPickerTreeNode[];
}

export function normalizeRepoPickerPath(value: string): string {
  return value.trim().replace(/\\/g, "/").replace(/^\/+/, "").replace(/\/+$/, "");
}

export function updateRepoPathSelection(current: string[], path: string, selected: boolean): string[] {
  const normalized = normalizeRepoPickerPath(path);
  if (!normalized) return [...current];
  const next = new Set(current.map(normalizeRepoPickerPath).filter(Boolean));
  if (selected) {
    for (const existing of next) {
      if (existing === normalized || existing.startsWith(`${normalized}/`)) next.delete(existing);
    }
    if (![...next].some((existing) => normalized.startsWith(`${existing}/`))) next.add(normalized);
  }
  else next.delete(normalized);
  return [...next].sort((a, b) => a.localeCompare(b));
}

export function pathIsCoveredBySelection(path: string, selections: string[]): boolean {
  const normalized = normalizeRepoPickerPath(path);
  return selections.some((selection) => {
    const scope = normalizeRepoPickerPath(selection);
    return normalized === scope || normalized.startsWith(`${scope}/`);
  });
}

export function repoPickerSelectionState(path: string, selections: string[]): RepoPickerSelectionState {
  const normalized = normalizeRepoPickerPath(path);
  const scopes = selections.map(normalizeRepoPickerPath).filter(Boolean);
  if (scopes.includes(normalized)) return "selected";
  if (scopes.some((scope) => normalized.startsWith(`${scope}/`))) return "inherited";
  if (scopes.some((scope) => scope.startsWith(`${normalized}/`))) return "partial";
  return "unselected";
}

export function repoEffectiveFileCount(
  items: RepoPickerItem[],
  includePaths: string[],
  excludePaths: string[],
): number {
  return repoEffectiveFiles(items, includePaths, excludePaths).length;
}

export function repoEffectiveFiles(
  items: RepoPickerItem[],
  includePaths: string[],
  excludePaths: string[],
): RepoPickerItem[] {
  return items.filter((item) => item.type === "blob"
    && (includePaths.length === 0 || pathIsCoveredBySelection(item.path, includePaths))
    && !pathIsCoveredBySelection(item.path, excludePaths));
}

export function repoScopeSummary(
  items: RepoPickerItem[],
  includePaths: string[],
  excludePaths: string[],
): RepoScopeSummary {
  const totalCount = items.filter((item) => item.type === "blob").length;
  const readyCount = repoEffectiveFileCount(items, includePaths, excludePaths);
  const filteredCount = totalCount - readyCount;
  const exclusionLabel = `${excludePaths.length} confirmed exclusion${excludePaths.length === 1 ? "" : "s"}`;
  let detailLabel: string;
  if (includePaths.length > 0) {
    detailLabel = `${filteredCount} file${filteredCount === 1 ? "" : "s"} outside the effective scope · ${exclusionLabel}`;
  } else if (filteredCount > 0) {
    detailLabel = `${filteredCount} file${filteredCount === 1 ? "" : "s"} filtered out by ${exclusionLabel}`;
  } else {
    detailLabel = `All ${totalCount} eligible file${totalCount === 1 ? " is" : "s are"} included.`;
  }
  return {
    readyCount,
    totalCount,
    filteredCount,
    readyLabel: `${readyCount} file${readyCount === 1 ? "" : "s"} ready to sync`,
    detailLabel,
  };
}

export function repoPickerItemsFromFilePaths(paths: string[]): RepoPickerItem[] {
  const folders = new Set<string>();
  const blobs = new Set<string>();
  for (const rawPath of paths) {
    const path = normalizeRepoPickerPath(rawPath);
    if (!path) continue;
    blobs.add(path);
    const parts = path.split("/").slice(0, -1);
    for (let index = 1; index <= parts.length; index += 1) {
      folders.add(parts.slice(0, index).join("/"));
    }
  }
  return [
    ...[...folders].sort((a, b) => a.localeCompare(b)).map((path) => ({ path, type: "tree" as const, size: null })),
    ...[...blobs].sort((a, b) => a.localeCompare(b)).map((path) => ({ path, type: "blob" as const, size: null })),
  ];
}

export function repoPickerTreeRows(
  items: RepoPickerItem[],
  expandedPaths: ReadonlySet<string>,
  query: string,
): RepoPickerTreeRow[] {
  const roots = buildRepoPickerTree(items);
  const needle = query.trim().toLocaleLowerCase();
  const rows: RepoPickerTreeRow[] = [];

  const visit = (node: RepoPickerTreeNode, depth: number): boolean => {
    const matchingChildren = needle
      ? node.children.filter((child) => treeContainsQuery(child, needle))
      : node.children;
    const matches = !needle
      || node.item.path.toLocaleLowerCase().includes(needle)
      || matchingChildren.length > 0;
    if (!matches) return false;

    rows.push({
      item: node.item,
      name: repoPickerPathName(node.item.path),
      depth,
      fileCount: treeFileCount(node),
      hasChildren: node.children.length > 0,
    });
    if (needle || expandedPaths.has(node.item.path)) {
      for (const child of matchingChildren) visit(child, depth + 1);
    }
    return true;
  };

  for (const root of roots) visit(root, 0);
  return rows;
}

function buildRepoPickerTree(items: RepoPickerItem[]): RepoPickerTreeNode[] {
  const itemByPath = new Map<string, RepoPickerItem>();
  for (const item of items) {
    const path = normalizeRepoPickerPath(item.path);
    if (!path) continue;
    itemByPath.set(path, { ...item, path });
    const parts = path.split("/");
    const folderParts = item.type === "tree" ? parts : parts.slice(0, -1);
    for (let index = 1; index <= folderParts.length; index += 1) {
      const folderPath = folderParts.slice(0, index).join("/");
      if (!itemByPath.has(folderPath)) {
        itemByPath.set(folderPath, { path: folderPath, type: "tree", size: null });
      }
    }
  }

  const nodeByPath = new Map<string, RepoPickerTreeNode>();
  for (const item of itemByPath.values()) nodeByPath.set(item.path, { item, children: [] });

  const roots: RepoPickerTreeNode[] = [];
  for (const node of nodeByPath.values()) {
    const parent = nodeByPath.get(repoPickerParentPath(node.item.path));
    if (parent) parent.children.push(node);
    else roots.push(node);
  }
  for (const node of nodeByPath.values()) node.children.sort(compareTreeNodes);
  return roots.sort(compareTreeNodes);
}

function treeContainsQuery(node: RepoPickerTreeNode, needle: string): boolean {
  return node.item.path.toLocaleLowerCase().includes(needle)
    || node.children.some((child) => treeContainsQuery(child, needle));
}

function treeFileCount(node: RepoPickerTreeNode): number {
  if (node.item.type === "blob") return 1;
  return node.children.reduce((total, child) => total + treeFileCount(child), 0);
}

function repoPickerParentPath(path: string): string {
  const separator = path.lastIndexOf("/");
  return separator < 0 ? "" : path.slice(0, separator);
}

function repoPickerPathName(path: string): string {
  const separator = path.lastIndexOf("/");
  return separator < 0 ? path : path.slice(separator + 1);
}

function compareTreeNodes(left: RepoPickerTreeNode, right: RepoPickerTreeNode): number {
  if (left.item.type !== right.item.type) return left.item.type === "tree" ? -1 : 1;
  return left.item.path.localeCompare(right.item.path);
}
