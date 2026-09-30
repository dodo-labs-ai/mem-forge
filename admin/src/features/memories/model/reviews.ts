import type { ReviewListItem } from "./types";

/**
 * The open review each Memory takes part in. A Memory waits for review as the
 * proposed replacement of a correction or as the Memory a source change
 * affects, so both sides of a review point to it; the first review wins.
 */
export function reviewIdByMemoryId(reviews: readonly ReviewListItem[]): ReadonlyMap<string, string> {
  const byMemory = new Map<string, string>();
  for (const review of reviews) {
    for (const memoryId of [review.challenger_memory_id, review.incumbent_memory_id]) {
      if (memoryId && !byMemory.has(memoryId)) byMemory.set(memoryId, review.id);
    }
  }
  return byMemory;
}
