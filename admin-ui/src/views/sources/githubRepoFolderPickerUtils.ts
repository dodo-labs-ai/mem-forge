export interface RepoPickerItem {
  path: string;
  type: "tree" | "blob";
  size: number | null;
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

export function repoEffectiveFileCount(
  items: RepoPickerItem[],
  includePaths: string[],
  excludePaths: string[],
): number {
  return items.filter((item) => item.type === "blob"
    && (includePaths.length === 0 || pathIsCoveredBySelection(item.path, includePaths))
    && !pathIsCoveredBySelection(item.path, excludePaths)).length;
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
