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
  if (selected) next.add(normalized);
  else next.delete(normalized);
  return [...next].sort((a, b) => a.localeCompare(b));
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
