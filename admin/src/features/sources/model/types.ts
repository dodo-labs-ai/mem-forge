import type { LocalAgentJob, Source, SyncProgressSnapshot, SyncStatus } from "@/api/responses";
import type { components } from "@/api/schema.gen";

export type { LocalAgentJob, Source, SyncProgressSnapshot, SyncStatus };
export type SourceConnectionStatus = NonNullable<Source["connection_status"]>;
export type SyncProgressUnit = NonNullable<SyncProgressSnapshot["progress"]>["unit"];

/** The project fields the Sources page groups by. */
export type Project = Pick<components["schemas"]["ProjectResponse"], "id" | "key" | "name">;

export interface SourceResolvedProject {
  project_key: string;
  memory_count: number;
}

/** A source inside one project group, with the memories it resolved into that project. */
export interface GroupedSource {
  source: Source;
  memory_count: number;
}

/** One project section of the Sources page; `project` is null for the Unmapped backlog. */
export interface SourceProjectGroup {
  project: Project | null;
  sources: GroupedSource[];
  docCount: number;
  memoryCount: number;
}
