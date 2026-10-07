import { memoryPath, reviewPath } from "@/lib/paths";
import type { Memory, MemoryRelation, MemorySearchHit, MemorySourceRef } from "./types";

export type RelationHint =
  | { kind: "conflicts"; count: number; target: string }
  | { kind: "updated"; target: string };

/** Where a Memory belongs and who reads it. */
export interface MemoryContext {
  projectKey: string | null;
  isPrivate: boolean;
  sources: MemorySourceRef[];
}

/** One row of the Memories list, from the admin listing or from ranked search. */
export interface MemoryRow {
  id: string;
  content: string;
  memoryType: string;
  status: string;
  /** Null for a ranked search result, which does not report project, access or sources. */
  context: MemoryContext | null;
  /** How many sources currently support the Memory. */
  supportCount: number;
  /** When the Memory was created, or last observed for a ranked search result. */
  seenAt: string | null;
  relationHints: RelationHint[];
  /** Where opening the row goes: the review a listed Memory waits for, else its detail. */
  target: string;
}

/** Visible only to its owner, as memories from an "Only me" source are. */
export const PRIVATE_VISIBILITY = "private";

/** The relations a reader must weigh: conflicts, and a newer Memory that updates this one. */
export function relationHints(memoryId: string, relations: readonly MemoryRelation[]): RelationHint[] {
  const hints: RelationHint[] = [];
  const conflicts = relations.filter((relation) => relation.label === "contradicts").length;
  if (conflicts > 0) hints.push({ kind: "conflicts", count: conflicts, target: memoryPath(memoryId) });
  const newer = relations.find((relation) => relation.label === "updates" && relation.role === "older");
  if (newer) hints.push({ kind: "updated", target: memoryPath(newer.counterpart.memory_id) });
  return hints;
}

export function memoryRow(memory: Memory): MemoryRow {
  const relations = memory.relations ?? [];
  return {
    id: memory.id,
    content: memory.content,
    memoryType: memory.memory_type,
    status: memory.status,
    context: {
      projectKey: memory.project_key ?? null,
      isPrivate: memory.visibility === PRIVATE_VISIBILITY,
      sources: memory.sources ?? [],
    },
    supportCount: memory.corroboration_count,
    seenAt: memory.created_at ?? null,
    relationHints: relationHints(memory.id, relations),
    target: memory.open_review_id ? reviewPath(memory.open_review_id) : memoryPath(memory.id),
  };
}

/** A ranked search result opens its detail, which links to any review the Memory waits for. */
export function searchHitRow(hit: MemorySearchHit): MemoryRow {
  return {
    id: hit.memory_id,
    content: hit.summary,
    memoryType: hit.memory_type,
    status: hit.status,
    context: null,
    supportCount: hit.corroborated_by,
    seenAt: hit.last_observed_at ?? null,
    relationHints: relationHints(hit.memory_id, hit.relations ?? []),
    target: memoryPath(hit.memory_id),
  };
}

/** The first source a row names, and how many more it has. */
export function sourceSummary(sources: readonly MemorySourceRef[]): { first: MemorySourceRef; more: number } | null {
  const [first, ...rest] = sources;
  return first ? { first, more: rest.length } : null;
}

/** Sources MemForge itself records: people add and correct Memories without a configured source. */
const VIRTUAL_SOURCE_NAMES: Readonly<Record<string, string>> = {
  user_memory: "Added manually",
  user_correction: "Correction",
};

/** A source's name, falling back to its type's display name for sources without one. */
export function sourceName(source: MemorySourceRef, typeLabels: Readonly<Record<string, string>>): string {
  return source.name ?? VIRTUAL_SOURCE_NAMES[source.source_type] ?? typeLabels[source.source_type] ?? source.source_type;
}
