import { isLocalAgentBackedSource } from "./localAgentSources";
import { SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY } from "@/api";
import { groupSourcesByProject, projectGroupKey, type ResolvedBySource } from "./projectGrouping";
import { sourceAttention, type SourceAttention } from "./sourceAttention";
import { organizeSourceGroups, type SourceListSortMode } from "./sourceListOrganization";
import { resolveSourceReadiness, type LocalDaemonReadiness, type SourceReadiness } from "./sourceReadiness";
import { selectSourceSyncActivity, type SourceSyncActivity } from "./sourceSyncActivity";
import type { LocalAgentJob, Project, Source, SourceProjectGroup } from "./types";

export type SourceFilter = "all" | "needs_you" | "pinned";

export interface SourceRow {
  /** Unique per row: a source bound by field can appear in several project groups. */
  key: string;
  source: Source;
  /** Memories this source resolved into the row's project. */
  memoryCount: number;
  activity: SourceSyncActivity | undefined;
  readiness: SourceReadiness | null;
  attention: SourceAttention | null;
}

export interface SourceRowGroup {
  key: string;
  title: string;
  description: string | null;
  memoryCount: number;
  rows: SourceRow[];
}

interface BuildInput {
  sources: readonly Source[];
  projects: readonly Project[];
  resolvedBySource: ResolvedBySource;
  jobs: readonly LocalAgentJob[];
  daemon: LocalDaemonReadiness;
  query: string;
  filter: SourceFilter;
  sortMode: SourceListSortMode;
  typeLabels: Readonly<Record<string, string>>;
  now?: Date;
}

const GROUP_DESCRIPTIONS: Record<string, string> = {
  [SHARED_PROJECT_KEY]: "Team-wide knowledge, ranked the same in every project.",
  [UNSORTED_PROJECT_KEY]: "Memories that matched none of the source's project rules.",
};
const UNMAPPED_TITLE = "No project";
const UNMAPPED_DESCRIPTION = "Sources without a project. Their new memories are not assigned to one.";

function latestJobBySource(jobs: readonly LocalAgentJob[]): Map<string, LocalAgentJob> {
  const latest = new Map<string, LocalAgentJob>();
  for (const job of jobs) {
    if (!job.source_id) continue;
    const current = latest.get(job.source_id);
    if (!current || (job.created_at ?? "") > (current.created_at ?? "")) latest.set(job.source_id, job);
  }
  return latest;
}

/** Everything a row shows about one source, computed once per source. */
export function describeSource(
  source: Source,
  localJob: LocalAgentJob | undefined,
  daemon: LocalDaemonReadiness,
  now: Date,
): Omit<SourceRow, "key" | "memoryCount"> {
  const activity = selectSourceSyncActivity({ sync: source.sync, localJob });
  const readiness = resolveSourceReadiness({
    localExecution: isLocalAgentBackedSource(source),
    daemon,
    connectionStatus: source.connection_status,
  });
  return { source, activity, readiness, attention: sourceAttention({ source, readiness, activity, now }) };
}

function groupTitle(group: SourceProjectGroup): string {
  return group.project?.name ?? UNMAPPED_TITLE;
}

function groupDescription(group: SourceProjectGroup): string | null {
  if (group.project === null) return UNMAPPED_DESCRIPTION;
  return GROUP_DESCRIPTIONS[group.project.key] ?? null;
}

/** Groups, filters and sorts sources for the Sources table. */
export function buildSourceRowGroups(input: BuildInput): SourceRowGroup[] {
  const now = input.now ?? new Date();
  const jobs = latestJobBySource(input.jobs);
  const described = new Map(
    input.sources.map((source) => [source.id, describeSource(source, jobs.get(source.id), input.daemon, now)]),
  );
  const matchesFilter = (source: Source) =>
    input.filter !== "needs_you" || described.get(source.id)?.attention != null;

  const grouped = groupSourcesByProject(input.sources, input.projects, input.resolvedBySource).map((group) => ({
    ...group,
    sources: group.sources.filter(({ source }) => matchesFilter(source)),
  }));
  const organized = organizeSourceGroups(grouped, {
    query: input.query,
    pinnedOnly: input.filter === "pinned",
    sortMode: input.sortMode,
    typeLabels: input.typeLabels,
  });

  return organized.map((group) => {
    const key = projectGroupKey(group);
    return {
      key,
      title: groupTitle(group),
      description: groupDescription(group),
      memoryCount: group.memoryCount,
      rows: group.sources.map(({ source, memory_count }) => ({
        key: `${key}:${source.id}`,
        memoryCount: memory_count,
        ...described.get(source.id)!,
      })),
    };
  });
}

/** Sources that need the user, one entry per source. */
export function sourcesNeedingAttention(
  sources: readonly Source[],
  jobs: readonly LocalAgentJob[],
  daemon: LocalDaemonReadiness,
  now: Date = new Date(),
): Array<{ source: Source; attention: SourceAttention; activity: SourceSyncActivity | undefined }> {
  const latest = latestJobBySource(jobs);
  return sources.flatMap((source) => {
    const described = describeSource(source, latest.get(source.id), daemon, now);
    return described.attention ? [{ source, attention: described.attention, activity: described.activity }] : [];
  });
}
