import type { MemoryRelation, RelatedMemory, RelationLabel } from "@/api/types";

export const RELATION_LABEL_NAMES: Record<RelationLabel, string> = {
  contradicts: "Conflict",
  updates: "Update",
  equivalent: "Same knowledge",
};

/** The dismissal wording for each label: a person says the relation does not hold. */
export const RELATION_DISMISS_ACTIONS: Record<RelationLabel, string> = {
  contradicts: "Not a conflict",
  updates: "Not an update",
  equivalent: "Not the same",
};

export function relationSentence(relation: Pick<MemoryRelation, "label" | "role">): string {
  if (relation.label === "contradicts") return "Conflicts with";
  if (relation.label === "updates") {
    return relation.role === "older" ? "Updated by the newer" : "Updates the older";
  }
  return "States the same as";
}

export function relatedMemorySources(memory: RelatedMemory): string {
  return memory.sources.map((source) => source.name ?? source.source_type).join(", ");
}

export function relationRevisionDate(memory: RelatedMemory): string | null {
  if (!memory.revision_at) return null;
  const revised = new Date(memory.revision_at);
  return Number.isNaN(revised.getTime()) ? null : revised.toLocaleDateString();
}
