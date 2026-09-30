import { expect, test, type Page } from "@playwright/test";

const ACTIONS = [
  {
    key: "use_latest_state",
    decision: "approve",
    label: "Use latest state",
    consequence: "Use the proposed state going forward and keep the previous state in audit history.",
    requires_note: false,
  },
  {
    key: "keep_current_state",
    decision: "reject",
    label: "Keep current state",
    consequence: "Keep the current memory active and discard this proposal.",
    requires_note: true,
  },
];

const REVIEW = {
  id: "review-1",
  kind: "lifecycle",
  status: "pending",
  review_origin: "lifecycle",
  source_id: "src-jira",
  source_name: "SFPAY board",
  incumbent_memory_id: "mem-current",
  challenger_memory_id: "mem-proposed",
  reason: "candidate_supersede_vs_audit_keep",
  review_note: null,
  reviewer: null,
  replacement_kind: "supersession",
  created_at: new Date().toISOString(),
  resolved_at: null,
  is_stale: false,
  refreshable: false,
  decision_fingerprint: "review-decision-v1:current",
  can_decide: true,
  presentation: {
    decision_label: "Updated",
    summary: "Use the proposed source state or keep the current memory?",
    why_human: "Applying the proposal would change active memory state and requires your decision.",
    current_label: "Current memory",
    proposed_label: "Proposed memory",
    proposed_empty_text: "No replacement memory; this proposal removes source support.",
    actions: ACTIONS,
    technical_reason: "candidate_supersede_vs_audit_keep",
  },
  incumbent: {
    id: "mem-current",
    memory_type: "fact",
    content: "On-demand pay dates can be any date in the open period.",
    corroboration_count: 1,
    status: "active",
    entity_refs: [],
    evidence: [],
  },
  challenger: {
    id: "mem-proposed",
    memory_type: "fact",
    content: "On-demand pay dates must fall before the next regular pay date.",
    corroboration_count: 1,
    status: "pending_review",
    entity_refs: [],
    evidence: [],
  },
};

const RELATED = {
  id: "mem-related",
  memory_type: "fact",
  content: "On-demand payment requests are rejected when the pay date is after the next regular pay date.",
  corroboration_count: 1,
  status: "active",
  entity_refs: [],
  evidence: [],
};

/** Serves the queue and the review's detail; the queue row reports the detail's `can_decide`. */
async function stubApi(page: Page, detail: Record<string, unknown>) {
  const responses: Record<string, unknown> = {
    "/api/v1/memory-reviews": { data: [{ ...REVIEW, can_decide: detail.can_decide }], total: 1, limit: 25, offset: 0 },
    "/api/v1/memory-reviews/review-1": detail,
    "/api/v1/genes": [{ name: "jira", display_name: "Jira" }],
    "/api/cloud/local-agent/status": { status: "offline", last_seen_at: null, checked_at: "", stale_after_seconds: 60 },
  };
  await page.route("**/api/**", (route) => {
    const body = responses[new URL(route.request().url()).pathname];
    return body === undefined ? route.fulfill({ status: 404, json: { detail: "Not stubbed" } }) : route.fulfill({ json: body });
  });
}

test("the Review queue opens a review from the sidebar", async ({ page }) => {
  await stubApi(page, { ...REVIEW, related_challengers: [RELATED], can_decide: true });

  await page.goto("/v2/review");

  const nav = page.getByRole("navigation", { name: "Main" });
  await expect(nav.getByRole("link", { name: "Review" })).toHaveAttribute("aria-current", "page");
  await expect(page.getByRole("heading", { name: "Review queue" })).toBeVisible();
  await expect(page.getByText("1 review needs your input.", { exact: false })).toBeVisible();

  await page.getByRole("link", { name: REVIEW.presentation.summary }).click();

  await expect(page).toHaveURL(/\/v2\/review\/review-1$/);
  await expect(page.getByRole("heading", { level: 1, name: REVIEW.presentation.summary })).toBeVisible();
  await expect(page.getByRole("button", { name: "Keep current state" })).toBeDisabled();
  await expect(page.getByRole("region", { name: "Also affected" }).getByRole("link")).toHaveAttribute(
    "href",
    "/v2/memories/mem-related",
  );

  await page.getByRole("link", { name: "Review queue" }).click();
  await expect(page).toHaveURL(/\/v2\/review$/);
});

test("a review the caller cannot decide is view only", async ({ page }) => {
  await stubApi(page, { ...REVIEW, related_challengers: [], can_decide: false });

  await page.goto("/v2/review/review-1");

  await expect(page.getByText("View only.", { exact: false })).toBeVisible();
  await expect(page.getByRole("button", { name: "Use latest state" })).toBeDisabled();
});

test("a review the caller cannot decide cannot be picked for a bulk decision", async ({ page }) => {
  await stubApi(page, { ...REVIEW, related_challengers: [], can_decide: false });

  await page.goto("/v2/review");

  const row = page.getByRole("list", { name: "Reviews" }).getByRole("listitem");
  await expect(row.getByRole("checkbox")).toHaveAttribute("aria-disabled", "true");
  await expect(row.getByText("View only: only the source owner or a workspace admin can decide it")).toBeVisible();
  await expect(page.getByRole("checkbox", { name: "Select every review on this page that you can decide" })).toHaveAttribute(
    "aria-disabled",
    "true",
  );
});
