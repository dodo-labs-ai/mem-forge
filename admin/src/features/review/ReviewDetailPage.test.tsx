import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import { errorResponse, type Route as FakeRoute } from "@/test/fakeApi";
import { renderRoutes } from "@/test/renderRoutes";
import { makeReviewDetail, makeReviewMemory } from "@/test/reviewFixtures";
import type { ReviewDetail } from "./model/types";
import { ReviewPage } from "./ReviewPage";

afterEach(() => {
  vi.unstubAllGlobals();
});

function renderReview(review: ReviewDetail, routes: Record<string, FakeRoute> = {}) {
  const api: Record<string, FakeRoute> = {
    [`GET /api/v1/memory-reviews/${review.id}`]: () => review,
    "GET /api/v1/genes": () => [{ name: "jira", display_name: "Jira" }],
    ...routes,
  };
  return renderRoutes(
    [{ path: "/review/*", element: <ReviewPage /> }, { path: "*", element: null }],
    { path: `/review/${review.id}`, api },
  );
}

async function bodyOf(calls: Request[], path: string) {
  const call = calls.find((request) => request.method === "POST" && new URL(request.url).pathname === path);
  return call ? call.clone().json() : undefined;
}

test("keeping the current state waits for a note and sends the fingerprint", async () => {
  const review = makeReviewDetail();
  const { calls } = renderReview(review, {
    "POST /api/v1/memory-reviews/review-1/reject": () => ({ ...review, status: "rejected" }),
  });
  const user = userEvent.setup();

  expect(await screen.findByRole("heading", { level: 1, name: review.presentation.summary })).toBeInTheDocument();
  const keep = screen.getByRole("button", { name: "Keep current state" });
  expect(keep).toBeDisabled();
  expect(screen.getByText("Keep the current memory active and discard this proposal.")).toBeInTheDocument();

  await user.type(screen.getByLabelText("Decision note"), "Different payroll environments");
  expect(keep).toBeEnabled();
  await user.click(keep);

  await vi.waitFor(async () =>
    expect(await bodyOf(calls, "/api/v1/memory-reviews/review-1/reject")).toEqual({
      expected_fingerprint: "review-decision-v1:current",
      note: "Different payroll environments",
    }),
  );
});

test("using the latest state needs no note and shows the recorded decision", async () => {
  const review = makeReviewDetail();
  const decided = { ...review, status: "approved", reviewer: "alex.stone", resolved_at: "2026-09-29T10:04:00Z" };
  let stored = review;
  const { calls } = renderReview(review, {
    "GET /api/v1/memory-reviews/review-1": () => stored,
    "POST /api/v1/memory-reviews/review-1/approve": () => (stored = decided),
  });
  const user = userEvent.setup();

  await user.click(await screen.findByRole("button", { name: "Use latest state" }));

  expect(await screen.findByText("Use latest state recorded")).toBeInTheDocument();
  expect(screen.getByText(/^Decided by alex\.stone on /)).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Use latest state" })).not.toBeInTheDocument();
  expect(await bodyOf(calls, "/api/v1/memory-reviews/review-1/approve")).toEqual({
    expected_fingerprint: "review-decision-v1:current",
    note: null,
  });
});

test("a failed decision keeps the comparison and the typed note, and offers a reload", async () => {
  const review = makeReviewDetail();
  renderReview(review, {
    "POST /api/v1/memory-reviews/review-1/approve": () =>
      errorResponse(409, { error: "stale", message: "Review participants changed" }),
  });
  const user = userEvent.setup();

  await user.type(await screen.findByLabelText("Decision note"), "Checked with payroll");
  await user.click(screen.getByRole("button", { name: "Use latest state" }));

  const alert = await screen.findByRole("alert");
  expect(within(alert).getByText("Review participants changed")).toBeInTheDocument();
  expect(within(alert).getByRole("button", { name: "Reload review" })).toBeInTheDocument();
  expect(screen.getByText(review.incumbent!.content)).toBeInTheDocument();
  expect(screen.getByText(review.challenger!.content)).toBeInTheDocument();
  expect(screen.getByLabelText("Decision note")).toHaveValue("Checked with payroll");
});

test("a caller who cannot decide sees the review as view only", async () => {
  renderReview(makeReviewDetail({ can_decide: false }));

  expect(await screen.findByText(/^View only\./)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Use latest state" })).toBeDisabled();
  expect(screen.getByRole("button", { name: "Keep current state" })).toBeDisabled();
  expect(screen.queryByLabelText("Decision note")).not.toBeInTheDocument();
});

test("a decided review shows who decided, when and why, and no decision panel", async () => {
  renderReview(
    makeReviewDetail({
      status: "rejected",
      reviewer: "alex.stone",
      resolved_at: "2026-09-28T17:40:00Z",
      review_note: "The handbook is still correct for 2026.",
    }),
  );

  expect(await screen.findByText("Keep current state recorded")).toBeInTheDocument();
  expect(screen.getByText("“The handbook is still correct for 2026.”")).toBeInTheDocument();
  expect(screen.queryByRole("heading", { name: "Choose what MemForge should do" })).not.toBeInTheDocument();
});

test("rechecking an expired proposal opens the new review", async () => {
  const expired = makeReviewDetail({ status: "stale", is_stale: true, refreshable: true, decision_fingerprint: "fp-stale" });
  const rechecked = makeReviewDetail({ id: "review-2", decision_fingerprint: "fp-new" });
  const { calls } = renderReview(expired, {
    "POST /api/v1/memory-reviews/review-1/refresh": () => rechecked,
    "GET /api/v1/memory-reviews/review-2": () => rechecked,
  });
  const user = userEvent.setup();

  expect(await screen.findByText("This proposal expired.")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Recheck current state" }));

  expect(await screen.findByRole("heading", { name: "Choose what MemForge should do" })).toBeInTheDocument();
  expect(await bodyOf(calls, "/api/v1/memory-reviews/review-1/refresh")).toEqual({ expected_fingerprint: "fp-stale" });
});

test("lists the other memories a decision touches, linked to their pages", async () => {
  renderReview(
    makeReviewDetail({
      related_challengers: [
        makeReviewMemory({ id: "mem-related", content: "On-demand requests after the next pay date are rejected." }),
      ],
    }),
  );

  const section = await screen.findByRole("region", { name: "Also affected" });
  expect(within(section).getByRole("link", { name: "On-demand requests after the next pay date are rejected." })).toHaveAttribute(
    "href",
    "/memories/mem-related",
  );
});
