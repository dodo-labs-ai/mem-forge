import { describe, expect, test } from "vitest";
import { makeMemory } from "@/test/memoryFixtures";
import { lifecycleDetail, statusBanner } from "./lifecycle";

describe("lifecycle detail", () => {
  test("translates a recorded retirement reason and keeps its code", () => {
    const memory = makeMemory({ status: "retired", retirement_reason: "source_deleted", retired_at: "2026-09-12T08:03:00Z" });
    expect(lifecycleDetail(memory)).toEqual({
      status: "Retired",
      reason: "Source document removed from current indexed source",
      technicalReason: "source_deleted",
      occurred: { label: "Retired", at: "2026-09-12T08:03:00Z" },
    });
  });

  test("shows a person's own retirement reason as written", () => {
    const memory = makeMemory({ status: "retired", retirement_reason: "No longer true after the workflow change" });
    const detail = lifecycleDetail(memory);
    expect(detail?.reason).toBe("No longer true after the workflow change");
    expect(detail).not.toHaveProperty("technicalReason");
  });

  test("names the memory that replaced a superseded one", () => {
    const memory = makeMemory({
      status: "superseded",
      replacement_reason: "Replaced by a newer decision",
      superseded_at: "2026-09-26T11:40:00Z",
      superseded_by: "mem-new",
    });
    expect(lifecycleDetail(memory)).toEqual({
      status: "Superseded",
      reason: "Replaced by a newer decision",
      occurred: { label: "Superseded", at: "2026-09-26T11:40:00Z" },
      replacedBy: "mem-new",
    });
  });

  test("reads a decayed memory as retired", () => {
    expect(lifecycleDetail(makeMemory({ status: "decayed" }))?.status).toBe("Retired");
  });

  test("appears for an active memory only when lifecycle metadata is left over", () => {
    expect(lifecycleDetail(makeMemory())).toBeNull();
    expect(lifecycleDetail(makeMemory({ replacement_reason: "Restored after review" }))?.reason).toBe("Restored after review");
  });
});

describe("status banner", () => {
  test("explains what each non-active status means for search", () => {
    expect(statusBanner(makeMemory())).toBeNull();
    expect(
      statusBanner(makeMemory({ status: "superseded", superseded_at: "2026-09-26T11:40:00Z", superseded_by: "mem-new" })),
    ).toEqual({
      tone: "live",
      title: "Superseded.",
      message: "A newer memory replaced this one on Sep 26, 2026. Superseded memories are kept for history and left out of search.",
      newerMemoryId: "mem-new",
    });
    expect(statusBanner(makeMemory({ status: "retired", retirement_reason: "source_deleted" }))?.message).toBe(
      "Source document removed from current indexed source. Retired memories are left out of search.",
    );
    expect(statusBanner(makeMemory({ status: "retired" }))?.message).toBe("Retired memories are left out of search.");
    expect(statusBanner(makeMemory({ status: "pending_review" }))).toMatchObject({ tone: "warn", title: "Needs review." });
  });
});
