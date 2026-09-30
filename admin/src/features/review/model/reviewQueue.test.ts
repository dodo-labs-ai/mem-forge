import { expect, test } from "vitest";
import { makeReviewMemory } from "@/test/reviewFixtures";
import {
  REVIEW_QUEUE_PAGE_SIZE,
  lastQueuePage,
  queueQuery,
  queueSummary,
  queueViewFromSearch,
  reviewOrigin,
  reviewSourceLabel,
  rowTimeLabel,
  searchForQueueView,
} from "./reviewQueue";

test("reads the queue view from the URL and falls back to waiting reviews", () => {
  expect(queueViewFromSearch(new URLSearchParams())).toEqual({ tab: "waiting", kind: "all", page: 1 });
  expect(queueViewFromSearch(new URLSearchParams("tab=kept&kind=memory&page=3"))).toEqual({
    tab: "kept",
    kind: "memory",
    page: 3,
  });
  expect(queueViewFromSearch(new URLSearchParams("tab=bogus&page=-2"))).toEqual({ tab: "waiting", kind: "all", page: 1 });
});

test("writes only the values that differ from the default", () => {
  expect(searchForQueueView({ tab: "waiting", kind: "all", page: 1 }).toString()).toBe("");
  expect(searchForQueueView({ tab: "expired", kind: "lifecycle", page: 2 }).toString()).toBe("tab=expired&kind=lifecycle&page=2");
});

test("the last page holds the remaining reviews, and an empty queue has one page", () => {
  expect(lastQueuePage(0)).toBe(1);
  expect(lastQueuePage(REVIEW_QUEUE_PAGE_SIZE)).toBe(1);
  expect(lastQueuePage(REVIEW_QUEUE_PAGE_SIZE + 1)).toBe(2);
});

test("maps each tab and kind to the list endpoint's filters", () => {
  expect(queueQuery({ tab: "waiting", kind: "all", page: 1 })).toEqual({
    status: "open",
    origin: undefined,
    limit: REVIEW_QUEUE_PAGE_SIZE,
    offset: 0,
  });
  expect(queueQuery({ tab: "expired", kind: "lifecycle", page: 2 })).toMatchObject({
    status: "stale",
    origin: "lifecycle",
    offset: REVIEW_QUEUE_PAGE_SIZE,
  });
  expect(queueQuery({ tab: "applied", kind: "memory", page: 1 }).status).toBe("approved");
  expect(queueQuery({ tab: "kept", kind: "memory", page: 1 }).status).toBe("rejected");
});

test("describes the tab in full sentences", () => {
  expect(queueSummary("waiting", 1).description).toBe(
    "1 review needs your input. Current memories stay active and searchable until you decide.",
  );
});

test("names the source, or the kind of decision when no single source is known", () => {
  expect(reviewSourceLabel({ source_name: "SFPAY board", review_origin: "lifecycle" })).toBe("SFPAY board");
  expect(reviewSourceLabel({ source_name: null, review_origin: "memory" })).toBe("Memory update");
});

test("dates each row by the moment that matters for its tab", () => {
  const now = new Date("2026-09-29T12:00:00Z");
  const opened = { created_at: "2026-09-29T09:00:00Z", resolved_at: null, is_stale: false };
  expect(rowTimeLabel({ ...opened, status: "pending" }, now)).toBe("3 h ago");
  expect(rowTimeLabel({ ...opened, status: "pending", is_stale: true }, now)).toBe("Opened 3 h ago");
  expect(rowTimeLabel({ ...opened, status: "stale", resolved_at: "2026-09-28T12:00:00Z" }, now)).toBe("Expired 1 day ago");
  expect(rowTimeLabel({ ...opened, status: "approved", resolved_at: "2026-09-29T11:00:00Z" }, now)).toBe("Decided 1 h ago");
});

test("marks a row with the proposal's origin, or the current memory's when nothing replaces it", () => {
  const codex = makeReviewMemory({ origin_source_type: "user_memory", origin_client: "codex" });
  const jira = makeReviewMemory({ origin_source_type: "jira", origin_client: null });
  expect(reviewOrigin({ incumbent: jira, challenger: codex })).toEqual({ sourceType: "user_memory", client: "codex" });
  expect(reviewOrigin({ incumbent: jira, challenger: null })).toEqual({ sourceType: "jira", client: null });
  expect(reviewOrigin({ incumbent: makeReviewMemory(), challenger: null })).toBeNull();
});
