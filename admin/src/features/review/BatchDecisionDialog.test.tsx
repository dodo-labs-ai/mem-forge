import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import { errorResponse } from "@/test/fakeApi";
import { renderRoutes } from "@/test/renderRoutes";
import { makeReviewListItem } from "@/test/reviewFixtures";
import { BatchDecisionDialog } from "./BatchDecisionDialog";
import { MAX_DECISIONS_PER_CALL } from "./model/batchDecisions";
import type { DecisionManifestItem } from "./model/types";

const RECEIPT = "review-manifest-v1:receipt";
const HTTP_SERVER_ERROR = 500;

afterEach(() => {
  vi.unstubAllGlobals();
});

function decided(outcome: "ready" | "applied") {
  return async (request: Request) => {
    const { decisions } = (await request.json()) as { decisions: DecisionManifestItem[] };
    return {
      mode: outcome === "ready" ? "validate" : "apply",
      results: decisions.map((item) => ({ review_id: item.review_id, decision: item.decision, outcome })),
      validation_receipt: RECEIPT,
    };
  };
}

test("a failed later batch keeps the results of the batches already applied", async () => {
  const reviews = Array.from({ length: MAX_DECISIONS_PER_CALL + 1 }, (_, index) =>
    makeReviewListItem({ id: `review-${index}`, decision_fingerprint: `fp-${index}` }),
  );
  const applyDecisions = decided("applied");
  let applyCalls = 0;
  renderRoutes(
    [
      {
        path: "*",
        element: <BatchDecisionDialog decision="approve" reviews={reviews} onOpenChange={() => {}} onApplied={() => {}} />,
      },
    ],
    {
      path: "/review",
      api: {
        "POST /api/v1/memory-reviews/decisions/validate": decided("ready"),
        "POST /api/v1/memory-reviews/decisions/apply": (request) => {
          applyCalls += 1;
          return applyCalls === 1 ? applyDecisions(request) : errorResponse(HTTP_SERVER_ERROR, "Review store unavailable");
        },
      },
    },
  );
  const user = userEvent.setup();

  const dialog = await screen.findByRole("dialog");
  await user.click(await within(dialog).findByRole("button", { name: `Apply ${reviews.length} decisions` }));

  expect(await within(dialog).findByText(`${MAX_DECISIONS_PER_CALL} decisions applied`)).toBeInTheDocument();
  expect(within(dialog).getByRole("alert")).toHaveTextContent("Could not send every decision. Review store unavailable");
  expect(within(dialog).getByText("No result came back for this review. Check it in the queue.")).toBeInTheDocument();
});
