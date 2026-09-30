import { describe, expect, test } from "vitest";
import { makeRelatedMemory, makeRelationPair } from "@/test/memoryFixtures";
import {
  dismissalDescription,
  dismissedLabelsText,
  orderedPairMemories,
  otherRelationsSummary,
  relationSentence,
  totalRelations,
} from "./relations";

describe("relations", () => {
  test("read from the side the memory is on", () => {
    expect(relationSentence({ label: "updates", role: "older" })).toBe("Updated by the newer");
    expect(relationSentence({ label: "updates", role: "newer" })).toBe("Updates the older");
    expect(relationSentence({ label: "contradicts", role: "peer" })).toBe("Conflicts with");
    expect(relationSentence({ label: "equivalent", role: "peer" })).toBe("States the same as");
  });

  test("summarize the labels beside the conflict count", () => {
    const counts = { contradicts: 12, updates: 31, equivalent: 1_058 };
    expect(totalRelations(counts)).toBe(1_101);
    expect(otherRelationsSummary(counts)).toBe("Also 31 updates, 1,058 same knowledge");
    expect(otherRelationsSummary({ contradicts: 0, updates: 1, equivalent: 0 })).toBe("Also 1 update, 0 same knowledge");
  });

  test("describe a dismissal", () => {
    expect(dismissalDescription("updates")).toBe(
      "Relation discovery will stop showing these two memories as an update. You can undo this from either memory.",
    );
    expect(dismissedLabelsText(["contradicts", "equivalent"])).toBe("Not a conflict, Not the same");
  });

  test("put the newer memory of an update first", () => {
    const older = makeRelatedMemory({ memory_id: "mem-old" });
    const newer = makeRelatedMemory({ memory_id: "mem-new" });
    const pair = makeRelationPair({ label: "updates", newer_memory_id: "mem-new", memories: [older, newer] });
    expect(orderedPairMemories(pair)).toEqual([
      { memory: newer, side: "newer" },
      { memory: older, side: "older" },
    ]);
    expect(orderedPairMemories(makeRelationPair()).map((entry) => entry.side)).toEqual([null, null]);
  });
});
