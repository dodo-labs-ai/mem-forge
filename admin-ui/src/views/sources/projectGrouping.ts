/**
 * Group sources into project buckets for the Sources page.
 *
 * Ordering rule (deterministic):
 *   1. SHARED bucket (the team-wide project, if any sources land there)
 *   2. User projects, alphabetical by display name
 *   3. Unsorted project (system catch-all, see UNSORTED_PROJECT_KEY)
 *   4. Unmapped (sources with no `project_binding` at all)
 *
 * Membership rule:
 *   - `fixed` bindings appear in exactly one group: their `project_key`.
 *   - `by_field` bindings fan out: one row per project_key the resolver
 *     observed for that source. If the resolver has not run yet, the
 *     source falls under its binding's `default`.
 *   - Unbound sources land in the null Unmapped group.
 *
 * Per-group `memory_count` reflects only the memories that resolved
 * INTO that group's project, not the source's total.
 */
import {
  SHARED_PROJECT_KEY,
  UNSORTED_PROJECT_KEY,
  isReservedProjectKey,
} from "@/api/projectKeys";
import type {
  GroupedSource,
  Project,
  Source,
  SourceProjectGroup,
  SourceResolvedProject,
} from "@/api/types";

export const PROJECT_GROUPS_DEFAULT_EXPANDED = true;

/**
 * Sentinel that keys the React-side collapsed-set when a group has no
 * project at all (the Unmapped backlog). The wire never carries this
 * token; it lives only inside the page's local state.
 */
export const UNMAPPED_GROUP_KEY = "__unmapped__";

export function projectGroupKey(group: SourceProjectGroup): string {
  return group.project ? group.project.key : UNMAPPED_GROUP_KEY;
}

export type ResolvedBySource = Record<string, SourceResolvedProject[]>;

// An empty or whitespace-only project_key on a binding means the admin has not
// chosen a project yet; that source belongs in the Unmapped (null) group, NOT
// in the Unsorted project (which is a real reserved project).
function isUsableKey(key: string | undefined | null): key is string {
  return typeof key === "string" && key.trim().length > 0;
}

export function groupSourcesByProject(
  sources: readonly Source[],
  projects: readonly Project[],
  resolvedBySource: ResolvedBySource,
): SourceProjectGroup[] {
  const projectByKey = new Map<string, Project>();
  for (const project of projects) {
    projectByKey.set(project.key, project);
  }

  // Map<projectKey | null, GroupedSource[]>; null is the Unmapped bucket.
  const buckets = new Map<string | null, GroupedSource[]>();

  function pushInto(key: string | null, entry: GroupedSource): void {
    const existing = buckets.get(key);
    if (existing) {
      existing.push(entry);
    } else {
      buckets.set(key, [entry]);
    }
  }

  for (const source of sources) {
    const binding = source.project_binding ?? null;
    if (!binding) {
      pushInto(null, { source, memory_count: source.memory_count ?? 0 });
      continue;
    }

    if (binding.mode === "fixed") {
      // No project chosen yet: fall through to Unmapped, not UNSORTED.
      const key: string | null = isUsableKey(binding.project_key)
        ? binding.project_key.trim()
        : null;
      pushInto(key, { source, memory_count: source.memory_count ?? 0 });
      continue;
    }

    // by_field: fan out across observed projects, or fall back to default.
    const observed = resolvedBySource[source.id] ?? [];
    const usableObserved = observed.filter((row) => isUsableKey(row.project_key));
    if (usableObserved.length === 0) {
      // Neither resolver rows nor a usable default; treat as Unmapped.
      const fallback: string | null = isUsableKey(binding.default)
        ? binding.default.trim()
        : null;
      pushInto(fallback, { source, memory_count: source.memory_count ?? 0 });
      continue;
    }
    for (const row of usableObserved) {
      pushInto(row.project_key, { source, memory_count: row.memory_count });
    }
  }

  const groups: SourceProjectGroup[] = [];
  for (const [key, entries] of buckets.entries()) {
    const project = key === null ? null : projectByKey.get(key) ?? null;
    if (key !== null && project === null) {
      // Unknown project_key referenced by a binding or resolver row;
      // surface it as a synthetic group so admins can find and fix it.
      groups.push({
        project: {
          id: `missing-${key}`,
          key,
          name: key,
          kind: isReservedProjectKey(key) ? "shared" : "normal",
          created_at: "",
        },
        sources: entries,
        docCount: sumDocs(entries),
        memoryCount: sumMemories(entries),
      });
      continue;
    }
    groups.push({
      project,
      sources: entries,
      docCount: sumDocs(entries),
      memoryCount: sumMemories(entries),
    });
  }

  groups.sort(compareGroups);
  return groups;
}

function sumDocs(entries: readonly GroupedSource[]): number {
  let total = 0;
  for (const entry of entries) {
    total += entry.source.doc_count ?? 0;
  }
  return total;
}

function sumMemories(entries: readonly GroupedSource[]): number {
  let total = 0;
  for (const entry of entries) {
    total += entry.memory_count;
  }
  return total;
}

const ORDER_SHARED = 0;
const ORDER_USER = 1;
const ORDER_UNSORTED = 2;
const ORDER_UNMAPPED = 3;

function rankFor(group: SourceProjectGroup): number {
  if (group.project === null) {
    return ORDER_UNMAPPED;
  }
  if (group.project.key === SHARED_PROJECT_KEY) {
    return ORDER_SHARED;
  }
  if (group.project.key === UNSORTED_PROJECT_KEY) {
    return ORDER_UNSORTED;
  }
  return ORDER_USER;
}

function compareGroups(a: SourceProjectGroup, b: SourceProjectGroup): number {
  const ra = rankFor(a);
  const rb = rankFor(b);
  if (ra !== rb) {
    return ra - rb;
  }
  if (ra === ORDER_USER && a.project && b.project) {
    return a.project.name.localeCompare(b.project.name);
  }
  return 0;
}
