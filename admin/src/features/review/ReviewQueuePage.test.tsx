import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import type { Route as FakeRoute } from "@/test/fakeApi";
import { renderRoutes } from "@/test/renderRoutes";
import { makeReviewListItem, makeReviewMemory } from "@/test/reviewFixtures";
import { REVIEW_QUEUE_PAGE_SIZE } from "./model/reviewQueue";
import type { DecisionManifestItem, DecisionResult, ReviewListItem } from "./model/types";
import { ReviewPage } from "./ReviewPage";

const RECEIPT = "review-manifest-v1:receipt";

afterEach(() => {
  vi.unstubAllGlobals();
});

const FIVE = makeReviewListItem({
  id: "review-five",
  decision_fingerprint: "fp-five",
  source_name: "mem-inception",
  incumbent: makeReviewMemory({ content: "The admin API exposes five endpoints." }),
  challenger: makeReviewMemory({ id: "mem-six", content: "The admin API exposes six endpoints.", status: "pending_review" }),
});
const WIKI = makeReviewListItem({ id: "review-wiki", decision_fingerprint: "fp-wiki", source_name: "Engineering wiki" });

function renderQueue(reviews: ReviewListItem[], routes: Record<string, FakeRoute> = {}, path = "/review") {
  const api: Record<string, FakeRoute> = {
    "GET /api/v1/memory-reviews": () => ({ data: reviews, total: reviews.length, limit: 25, offset: 0 }),
    ...routes,
  };
  return renderRoutes(
    [{ path: "/review/*", element: <ReviewPage /> }, { path: "*", element: null }],
    { path, api },
  );
}

function lastListQuery(calls: Request[]): URLSearchParams {
  const listCalls = calls.filter((request) => new URL(request.url).pathname === "/api/v1/memory-reviews");
  return new URL(listCalls.at(-1)!.url).searchParams;
}

test("shows waiting reviews with the words each decision changes", async () => {
  renderQueue([FIVE]);

  const list = await screen.findByRole("list", { name: "Reviews" });
  expect(within(list).getByRole("link", { name: FIVE.presentation.summary })).toHaveAttribute("href", "/review/review-five");
  expect(within(list).getByText("five").tagName).toBe("MARK");
  expect(within(list).getByText("six").tagName).toBe("MARK");
  expect(screen.getByText(/^1 review needs your input\./)).toBeInTheDocument();
});

test("tabs and the kind control filter the list endpoint", async () => {
  const { calls } = renderQueue([FIVE]);
  const user = userEvent.setup();
  await screen.findByRole("list", { name: "Reviews" });
  expect(lastListQuery(calls).get("status")).toBe("open");

  await user.click(screen.getByRole("tab", { name: "Expired" }));
  await vi.waitFor(() => expect(lastListQuery(calls).get("status")).toBe("stale"));

  await user.click(screen.getByRole("button", { name: "Source lifecycle" }));
  await vi.waitFor(() => expect(lastListQuery(calls).get("origin")).toBe("lifecycle"));
  expect(lastListQuery(calls).get("status")).toBe("stale");
  expect(lastListQuery(calls).get("offset")).toBe("0");
});

test("a bulk decision is checked first, then applied with the check's receipt", async () => {
  const validated: DecisionManifestItem[][] = [];
  const applied: unknown[] = [];
  const outcome = (item: DecisionManifestItem, mode: "check" | "apply"): DecisionResult =>
    item.review_id === "review-wiki"
      ? { review_id: item.review_id, decision: item.decision, outcome: "stale", message: "Review participants changed after analysis" }
      : { review_id: item.review_id, decision: item.decision, outcome: mode === "check" ? "ready" : "applied" };
  const { calls } = renderQueue([FIVE, WIKI], {
    "POST /api/v1/memory-reviews/decisions/validate": async (request) => {
      const { decisions } = (await request.json()) as { decisions: DecisionManifestItem[] };
      validated.push(decisions);
      return { mode: "validate", results: decisions.map((item) => outcome(item, "check")), validation_receipt: RECEIPT };
    },
    "POST /api/v1/memory-reviews/decisions/apply": async (request) => {
      const body = (await request.json()) as { decisions: DecisionManifestItem[]; validation_receipt: string };
      applied.push(body);
      return { mode: "apply", results: body.decisions.map((item) => outcome(item, "apply")) };
    },
  });
  const user = userEvent.setup();

  await user.click(await screen.findByRole("checkbox", { name: "Select every review on this page that you can decide" }));
  const bar = screen.getByRole("region", { name: "Bulk decision" });
  expect(within(bar).getByText("2 selected")).toBeInTheDocument();
  await user.click(within(bar).getByRole("button", { name: "Use latest state" }));

  const dialog = await screen.findByRole("dialog");
  expect(await within(dialog).findByText("Review participants changed after analysis. It will be skipped.")).toBeInTheDocument();
  expect(validated[0]?.map((item) => [item.review_id, item.expected_fingerprint, item.decision])).toEqual([
    ["review-five", "fp-five", "approve"],
    ["review-wiki", "fp-wiki", "approve"],
  ]);

  await user.click(within(dialog).getByRole("button", { name: "Apply 1 decision" }));

  expect(await within(dialog).findByText("1 decision applied")).toBeInTheDocument();
  expect(applied).toEqual([{ decisions: validated[1], validation_receipt: RECEIPT }]);
  const order = calls
    .filter((request) => request.method === "POST")
    .map((request) => new URL(request.url).pathname.split("/").at(-1));
  expect(order).toEqual(["validate", "validate", "apply"]);
});

test("a review the caller cannot decide cannot be picked for a bulk decision, and says why", async () => {
  const viewOnly = makeReviewListItem({
    id: "review-view-only",
    can_decide: false,
    incumbent: makeReviewMemory({ content: "Payroll runs on the last working day." }),
  });
  renderQueue([FIVE, viewOnly]);
  const user = userEvent.setup();

  const locked = await screen.findByRole("checkbox", { name: `Select review: ${viewOnly.incumbent!.content}` });
  expect(locked).toHaveAttribute("aria-disabled", "true");
  expect(locked).toHaveAccessibleDescription("View only: only the source owner or a workspace admin can decide it");

  await user.click(screen.getByRole("checkbox", { name: "Select every review on this page that you can decide" }));

  expect(within(screen.getByRole("region", { name: "Bulk decision" })).getByText("1 selected")).toBeInTheDocument();
  expect(screen.getByRole("checkbox", { name: `Select review: ${FIVE.incumbent!.content}` })).toBeChecked();
  expect(locked).not.toBeChecked();
});

test("select all is off when no review on the page can be decided", async () => {
  renderQueue([makeReviewListItem({ can_decide: false })]);

  expect(
    await screen.findByRole("checkbox", { name: "Select every review on this page that you can decide" }),
  ).toHaveAttribute("aria-disabled", "true");
});

test("keeping the current state in bulk waits for a note before checking", async () => {
  const { calls } = renderQueue([FIVE], {
    "POST /api/v1/memory-reviews/decisions/validate": async (request) => {
      const { decisions } = (await request.json()) as { decisions: DecisionManifestItem[] };
      return {
        mode: "validate",
        results: decisions.map((item) => ({ review_id: item.review_id, decision: item.decision, outcome: "ready" })),
        validation_receipt: RECEIPT,
      };
    },
  });
  const user = userEvent.setup();

  await user.click(await screen.findByRole("checkbox", { name: `Select review: ${FIVE.incumbent!.content}` }));
  await user.click(within(screen.getByRole("region", { name: "Bulk decision" })).getByRole("button", { name: "Keep current state" }));

  const dialog = await screen.findByRole("dialog");
  expect(within(dialog).getByText("Add a note to check these reviews.")).toBeInTheDocument();
  expect(within(dialog).getByRole("button", { name: "Apply decisions" })).toBeDisabled();
  expect(calls.some((request) => request.method === "POST")).toBe(false);

  await user.type(within(dialog).getByLabelText("Note for all"), "Still correct");
  await user.tab();

  expect(await within(dialog).findByRole("button", { name: "Apply 1 decision" })).toBeEnabled();
});

test("a page past the last one moves to the last page that has reviews", async () => {
  const total = REVIEW_QUEUE_PAGE_SIZE + 1;
  const { router } = renderQueue(
    [],
    {
      "GET /api/v1/memory-reviews": (request) => {
        const offset = Number(new URL(request.url).searchParams.get("offset"));
        return { data: offset < total ? [FIVE] : [], total, limit: REVIEW_QUEUE_PAGE_SIZE, offset };
      },
    },
    "/review?page=3",
  );

  expect(await screen.findByRole("list", { name: "Reviews" })).toBeInTheDocument();
  expect(router.state.location.search).toBe("?page=2");
  expect(screen.queryByText("All clear")).not.toBeInTheDocument();
});
