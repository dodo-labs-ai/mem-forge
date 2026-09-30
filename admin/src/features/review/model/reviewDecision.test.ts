import { describe, expect, test } from "vitest";
import { ApiError } from "@/api/errors";
import { makeReviewDetail } from "@/test/reviewFixtures";
import {
  canSubmitDecision,
  decidedBanner,
  decisionFailure,
  decisionNote,
  decisionOptions,
  expiredExplanation,
  noteHint,
  reviewState,
} from "./reviewDecision";

const formatWhen = (iso: string) => `at ${iso}`;

describe("reviewState", () => {
  test("maps statuses to the queue tabs", () => {
    expect(reviewState({ status: "pending", is_stale: false })).toBe("waiting");
    expect(reviewState({ status: "pending", is_stale: true })).toBe("expired");
    expect(reviewState({ status: "stale", is_stale: true })).toBe("expired");
    expect(reviewState({ status: "approved", is_stale: false })).toBe("applied");
    expect(reviewState({ status: "rejected", is_stale: false })).toBe("kept");
  });
});

describe("decision options", () => {
  const options = decisionOptions(makeReviewDetail().presentation)!;

  test("pairs the approve and reject actions", () => {
    expect(options.useLatest.label).toBe("Use latest state");
    expect(options.keepCurrent.label).toBe("Keep current state");
  });

  test("keeping the current state waits for a note", () => {
    expect(canSubmitDecision(options.keepCurrent, "  ")).toBe(false);
    expect(canSubmitDecision(options.keepCurrent, "Different environments")).toBe(true);
    expect(canSubmitDecision(options.useLatest, "")).toBe(true);
  });

  test("names the required note first", () => {
    expect(noteHint(options)).toBe("Required to keep the current state. Optional when using the latest state.");
  });

  test("sends a trimmed note, or none", () => {
    expect(decisionNote("  why  ")).toBe("why");
    expect(decisionNote("   ")).toBeNull();
  });
});

describe("decidedBanner", () => {
  test("says who decided, when, and why", () => {
    const review = makeReviewDetail({
      status: "approved",
      reviewer: "alex.stone",
      resolved_at: "2026-09-29T10:04:00Z",
      review_note: "SFPAY-173288 is the spec.",
    });
    expect(decidedBanner(review, formatWhen)).toEqual({
      title: "Use latest state recorded",
      byline: "Decided by alex.stone on at 2026-09-29T10:04:00Z.",
      note: "SFPAY-173288 is the spec.",
    });
  });

  test("leaves out what the server did not record", () => {
    const review = makeReviewDetail({ status: "rejected", reviewer: null, resolved_at: null, review_note: " " });
    expect(decidedBanner(review, formatWhen)).toEqual({ title: "Keep current state recorded", byline: null, note: null });
  });
});

describe("expiredExplanation", () => {
  test("offers a recheck only when the server can refresh the proposal", () => {
    expect(expiredExplanation({ review_origin: "lifecycle", refreshable: true }).canRecheck).toBe(true);
    const raisedAgain = expiredExplanation({ review_origin: "lifecycle", refreshable: false });
    expect(raisedAgain.canRecheck).toBe(false);
    expect(raisedAgain.text).toMatch(/next revision of its source/);
    expect(expiredExplanation({ review_origin: "memory", refreshable: false }).text).toMatch(/audit history/);
  });
});

describe("decisionFailure", () => {
  test("explains a missing permission", () => {
    const failure = decisionFailure(new ApiError(403, "Only the source owner or a workspace admin can manage this source."));
    expect(failure).toMatchObject({ reloadable: false, hint: "You can read this review, but a source manager has to decide." });
  });

  test("offers a reload when the review changed", () => {
    const failure = decisionFailure(new ApiError(409, "Review participants changed"));
    expect(failure).toMatchObject({ message: "Review participants changed", reloadable: true });
  });

  test("passes other errors through", () => {
    expect(decisionFailure(new Error("Network down"))).toEqual({ message: "Network down", hint: null, reloadable: false });
  });
});
