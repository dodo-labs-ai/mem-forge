import { formatCount, pluralize } from "@/lib/format";
import type { Tone } from "@/patterns";
import type { MemoryRelation, RelatedMemory, RelationLabel, RelationPair } from "./types";

/** Relation labels in reading order: what a reader must weigh first comes first. */
export const RELATION_LABELS: readonly RelationLabel[] = ["contradicts", "updates", "equivalent"];

export const RELATION_LABEL_NAMES: Record<RelationLabel, string> = {
  contradicts: "Conflict",
  updates: "Update",
  equivalent: "Same knowledge",
};

/** The relation filter's option names, which count relations of each label. */
export const RELATION_FILTER_NAMES: Record<RelationLabel, string> = {
  contradicts: "Conflicts",
  updates: "Updates",
  equivalent: "Same knowledge",
};

export const RELATION_LABEL_TONES: Record<RelationLabel, Tone> = {
  contradicts: "danger",
  updates: "live",
  equivalent: "idle",
};

/** The dismissal wording for each label: a person says the relation does not hold. */
export const RELATION_DISMISS_ACTIONS: Record<RelationLabel, string> = {
  contradicts: "Not a conflict",
  updates: "Not an update",
  equivalent: "Not the same",
};

const RELATION_DISMISS_TITLES: Record<RelationLabel, string> = {
  contradicts: "Mark as not a conflict?",
  updates: "Mark as not an update?",
  equivalent: "Mark as not the same?",
};

const RELATION_DISMISS_EFFECTS: Record<RelationLabel, string> = {
  contradicts: "a conflict",
  updates: "an update",
  equivalent: "the same knowledge",
};

export function isRelationLabel(value: string | null): value is RelationLabel {
  return value !== null && (RELATION_LABELS as readonly string[]).includes(value);
}

/** How the relation reads from this Memory's side. */
export function relationSentence(relation: Pick<MemoryRelation, "label" | "role">): string {
  if (relation.label === "contradicts") return "Conflicts with";
  if (relation.label === "updates") return relation.role === "older" ? "Updated by the newer" : "Updates the older";
  return "States the same as";
}

export function dismissalTitle(label: RelationLabel): string {
  return RELATION_DISMISS_TITLES[label];
}

export function dismissalDescription(label: RelationLabel): string {
  return `Relation discovery will stop showing these two memories as ${RELATION_DISMISS_EFFECTS[label]}. You can undo this from either memory.`;
}

/** "Not a conflict, Not the same": the labels a person dismissed for one pair. */
export function dismissedLabelsText(labels: readonly RelationLabel[]): string {
  return labels.map((label) => RELATION_DISMISS_ACTIONS[label]).join(", ");
}

export type RelationCounts = Record<RelationLabel, number>;

export function totalRelations(counts: RelationCounts): number {
  return RELATION_LABELS.reduce((sum, label) => sum + counts[label], 0);
}

/** "Also 31 updates, 58 same knowledge": the other labels, beside the conflict count. */
export function otherRelationsSummary(counts: RelationCounts): string {
  return `Also ${pluralize(counts.updates, "update")}, ${formatCount(counts.equivalent)} same knowledge`;
}

/** The pair's Memories with the newer one of an update first. */
export function orderedPairMemories(pair: RelationPair): { memory: RelatedMemory; side: "newer" | "older" | null }[] {
  if (pair.label !== "updates" || !pair.newer_memory_id) {
    return pair.memories.map((memory) => ({ memory, side: null }));
  }
  const newer = pair.memories.filter((memory) => memory.memory_id === pair.newer_memory_id);
  const older = pair.memories.filter((memory) => memory.memory_id !== pair.newer_memory_id);
  return [
    ...newer.map((memory) => ({ memory, side: "newer" as const })),
    ...older.map((memory) => ({ memory, side: "older" as const })),
  ];
}
