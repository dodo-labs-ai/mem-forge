import { expect, test } from "vitest";
import { makeReviewListItem } from "@/test/reviewFixtures";
import {
  MAX_DECISIONS_PER_CALL,
  batchItemText,
  inBatches,
  manifestFor,
  outcomeCounts,
  outcomeDetail,
} from "./batchDecisions";
import type { DecisionResult } from "./types";

function result(outcome: DecisionResult["outcome"], overrides: Partial<DecisionResult> = {}): DecisionResult {
  return { review_id: "review-1", decision: "approve", outcome, ...overrides };
}

test("sends at most the server's limit per call", () => {
  const ids = Array.from({ length: MAX_DECISIONS_PER_CALL * 2 + 1 }, (_, index) => index);
  expect(inBatches(ids).map((batch) => batch.length)).toEqual([MAX_DECISIONS_PER_CALL, MAX_DECISIONS_PER_CALL, 1]);
  expect(inBatches([])).toEqual([]);
});

test("builds one manifest entry per review with its fingerprint and the shared note", () => {
  const reviews = [makeReviewListItem({ id: "a", decision_fingerprint: "fp-a" }), makeReviewListItem({ id: "b", decision_fingerprint: "fp-b" })];
  expect(manifestFor(reviews, "reject", "  Still correct  ")).toEqual([
    { review_id: "a", decision: "reject", expected_fingerprint: "fp-a", note: "Still correct", risk: "medium" },
    { review_id: "b", decision: "reject", expected_fingerprint: "fp-b", note: "Still correct", risk: "medium" },
  ]);
  expect(manifestFor(reviews, "approve", "")[0]?.note).toBeNull();
});

test("counts outcomes under their labels", () => {
  const counts = outcomeCounts([result("ready"), result("ready"), result("stale"), result("invalid"), result("failed")]);
  expect(counts.map(({ label, count }) => [label, count])).toEqual([
    ["Ready", 2],
    ["Expired", 1],
    ["Not applied", 2],
  ]);
});

test("explains what happens to each review", () => {
  expect(outcomeDetail(result("ready"), "Use the proposed state.", "check")).toBe("Use the proposed state.");
  expect(outcomeDetail(result("stale", { message: "Review participants changed after analysis" }), "", "check")).toBe(
    "Review participants changed after analysis. It will be skipped.",
  );
  expect(outcomeDetail(result("forbidden"), "", "done")).toBe("Not allowed. It was skipped.");
  expect(outcomeDetail(result("already_applied"), "", "done")).toBe("Already decided.");
});

test("identifies a review by the memory the decision keeps", () => {
  const review = makeReviewListItem();
  expect(batchItemText(review, "approve")).toBe(review.challenger?.content);
  expect(batchItemText(review, "reject")).toBe(review.incumbent?.content);
  expect(batchItemText(makeReviewListItem({ challenger: null }), "approve")).toBe(review.incumbent?.content);
});
