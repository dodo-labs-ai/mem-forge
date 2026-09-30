import type { ProjectSource } from "@/features/sources";

export interface ProjectSourceRow {
  key: string;
  source: ProjectSource["source"];
  /** How the source is bound: every memory goes here, or memories are routed by a document field. */
  binding: "direct" | "by_field";
  /** Memories this source sent to the project. */
  memoryCount: number;
}

/** The sources that send memories to a project, the largest contributors first. */
export function projectSourceRows(sources: readonly ProjectSource[]): ProjectSourceRow[] {
  return sources
    .map((entry) => ({
      key: entry.source.id,
      source: entry.source,
      binding: entry.binding.mode === "fixed" ? ("direct" as const) : ("by_field" as const),
      memoryCount: entry.memory_count,
    }))
    .sort((a, b) => b.memoryCount - a.memoryCount || a.source.name.localeCompare(b.source.name));
}
