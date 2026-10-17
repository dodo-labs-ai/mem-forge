import { describe, expect, test } from "vitest";
import { makeFailedItem, makeSource } from "@/test/sourceFixtures";
import { failedItemStageLabel, sourceFailedItems } from "./sourceFailedItems";

const recorded = (items = [makeFailedItem()], retrying_after_upgrade = 0, count = items.length) => ({
  count,
  retrying_after_upgrade,
  items,
});

describe("sourceFailedItems", () => {
  test("lists each recorded item with the step that failed and why", () => {
    const source = makeSource({
      failed_items: recorded([
        makeFailedItem(),
        makeFailedItem({ document_id: "doc-2", title: null, stage: "derivation", message: "Model output was invalid" }),
      ]),
    });

    expect(sourceFailedItems(source, { state: "partial" })).toEqual({
      count: 2,
      retryingAfterUpgrade: 0,
      items: [
        { documentId: "doc-1", title: "Payroll FAQ", stage: "Reading", reason: "403 Forbidden", retryingAfterUpgrade: false },
        {
          documentId: "doc-2",
          title: "doc-2",
          stage: "Creating memories",
          reason: "Model output was invalid",
          retryingAfterUpgrade: false,
        },
      ],
    });
  });

  test("items an earlier version failed are retrying only while a sync is on its way", () => {
    const source = makeSource({
      failed_items: recorded([makeFailedItem({ retrying_after_upgrade: true }), makeFailedItem({ document_id: "doc-2" })], 1),
    });

    for (const state of ["queued", "active", "recovering"] as const) {
      const failed = sourceFailedItems(source, { state });
      expect(failed?.retryingAfterUpgrade).toBe(1);
      expect(failed?.items.map((item) => item.retryingAfterUpgrade)).toEqual([true, false]);
    }
    const settled = sourceFailedItems(source, { state: "success" });
    expect(settled?.retryingAfterUpgrade).toBe(0);
    expect(settled?.items.every((item) => !item.retryingAfterUpgrade)).toBe(true);
  });

  test("the recorded count covers items beyond the listed ones", () => {
    expect(sourceFailedItems(makeSource({ failed_items: recorded([makeFailedItem()], 0, 140) }), undefined)?.count).toBe(140);
  });

  test("the failure record is the authority over the last sync", () => {
    const source = makeSource({ failed_items: recorded([], 0, 0) });
    const activity = { state: "partial" as const, error: { items: [{ doc_id: "doc-9", title: "Old", error: "boom" }] } };

    expect(sourceFailedItems(source, activity)).toBeNull();
  });

  test("a source without a failure record shows the items its last sync could not read", () => {
    const activity = { state: "partial" as const, error: { items: [{ doc_id: "doc-9", title: "", error: "boom" }] } };

    expect(sourceFailedItems(makeSource(), activity)).toEqual({
      count: 1,
      retryingAfterUpgrade: 0,
      items: [{ documentId: "doc-9", title: "doc-9", stage: null, reason: "boom", retryingAfterUpgrade: false }],
    });
    expect(sourceFailedItems(makeSource(), { state: "success" })).toBeNull();
  });

  test("an unknown step reads as words", () => {
    expect(failedItemStageLabel("plan_construction")).toBe("Plan construction");
    expect(failedItemStageLabel("lifecycle_commit")).toBe("Saving memories");
  });
});
