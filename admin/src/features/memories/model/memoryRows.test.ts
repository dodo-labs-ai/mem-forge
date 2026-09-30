import { describe, expect, test } from "vitest";
import { makeMemory, makeRelatedMemory, makeRelation } from "@/test/memoryFixtures";
import { makeReviewListItem } from "@/test/reviewFixtures";
import { memoryRow, relationHints, searchHitRow, sourceName, sourceSummary } from "./memoryRows";
import { reviewIdByMemoryId } from "./reviews";

const NO_REVIEWS = new Map<string, string>();

describe("relation hints", () => {
  test("count conflicts and point an updated memory to the newer one", () => {
    const relations = [
      makeRelation(),
      makeRelation({ counterpart: makeRelatedMemory({ memory_id: "mem-b" }) }),
      makeRelation({ label: "updates", role: "older", counterpart: makeRelatedMemory({ memory_id: "mem-newer" }) }),
    ];
    expect(relationHints("mem-a", relations)).toEqual([
      { kind: "conflicts", count: 2, target: "/memories/mem-a" },
      { kind: "updated", target: "/memories/mem-newer" },
    ]);
  });

  test("need nothing for the newer side of an update or the same knowledge", () => {
    const relations = [makeRelation({ label: "updates", role: "newer" }), makeRelation({ label: "equivalent" })];
    expect(relationHints("mem-a", relations)).toEqual([]);
  });
});

describe("memory rows", () => {
  test("carry project, access and sources from the admin listing", () => {
    const row = memoryRow(makeMemory({ visibility: "private" }), NO_REVIEWS);
    expect(row.context).toEqual({
      projectKey: "PAY",
      isPrivate: true,
      sources: [{ source_id: "src-handbook", source_type: "confluence", name: "Payroll handbook" }],
    });
    expect(row.target).toBe("/memories/mem-cutoff");
  });

  test("open the pending review of a memory that waits for one", () => {
    const reviews = reviewIdByMemoryId([makeReviewListItem({ id: "rev-9", challenger_memory_id: "mem-cutoff" })]);
    expect(memoryRow(makeMemory({ status: "pending_review" }), reviews).target).toBe("/review/rev-9");
    expect(memoryRow(makeMemory({ status: "active" }), reviews).target).toBe("/memories/mem-cutoff");
  });

  test("from ranked search have no context", () => {
    const row = searchHitRow(
      {
        memory_id: "mem-hit",
        memory_type: "procedure",
        summary: "Clear the blocker hint first.",
        relevance_score: 0.9,
        corroborated_by: 1,
        last_observed_at: "2026-09-29T08:00:00Z",
        freshness: "current",
        status: "active",
        relations: [makeRelation()],
      },
      NO_REVIEWS,
    );
    expect(row).toMatchObject({ id: "mem-hit", context: null, supportCount: 1, seenAt: "2026-09-29T08:00:00Z" });
    expect(row.relationHints).toEqual([{ kind: "conflicts", count: 1, target: "/memories/mem-hit" }]);
  });
});

describe("reviews by memory", () => {
  test("point both sides of a review to it and keep the first review", () => {
    const byMemory = reviewIdByMemoryId([
      makeReviewListItem({ id: "rev-1", incumbent_memory_id: "mem-a", challenger_memory_id: "mem-b" }),
      makeReviewListItem({ id: "rev-2", incumbent_memory_id: "mem-b", challenger_memory_id: null }),
    ]);
    expect(Object.fromEntries(byMemory)).toEqual({ "mem-a": "rev-1", "mem-b": "rev-1" });
  });
});

describe("sources", () => {
  test("summarize as the first source and how many more", () => {
    const sources = makeMemory().sources!;
    expect(sourceSummary([...sources, ...sources])).toEqual({ first: sources[0], more: 1 });
    expect(sourceSummary([])).toBeNull();
  });

  test("fall back to a name for sources MemForge records and to the type's display name", () => {
    const labels = { jira: "Jira" };
    expect(sourceName({ source_id: "user_memory", source_type: "user_memory", name: null }, labels)).toBe("Added manually");
    expect(sourceName({ source_id: "src-j", source_type: "jira", name: null }, labels)).toBe("Jira");
    expect(sourceName({ source_id: "src-x", source_type: "custom", name: null }, labels)).toBe("custom");
  });
});
