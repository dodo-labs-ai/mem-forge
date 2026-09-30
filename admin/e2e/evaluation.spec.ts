import { expect, test } from "@playwright/test";
import { makeOverview } from "../src/test/evaluationFixtures";

const RESPONSES: Record<string, unknown> = {
  "/api/v1/agent-evaluations/online-overview": makeOverview(),
  "/api/v1/genes": [
    { name: "confluence", display_name: "Confluence" },
    { name: "github_pages", display_name: "GitHub Pages" },
    { name: "github_repo", display_name: "GitHub Repository" },
    { name: "jira", display_name: "Jira" },
  ],
  "/api/cloud/local-agent/status": { status: "online", last_seen_at: null, checked_at: "", stale_after_seconds: 60 },
};

test("the Evaluation page renders in the shell and scopes to a source", async ({ page }) => {
  await page.route("**/api/**", (route) => {
    const body = RESPONSES[new URL(route.request().url()).pathname];
    return body === undefined ? route.fulfill({ status: 404, json: { detail: "Not stubbed" } }) : route.fulfill({ json: body });
  });

  await page.goto("/v2/evaluation");

  await expect(page.getByRole("heading", { name: "Evaluation" })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Evaluation" })).toHaveAttribute(
    "aria-current",
    "page",
  );
  await expect(page.getByText("2 issue groups")).toBeVisible();
  await expect(page.getByRole("tab", { name: "Patterns to check 1" })).toBeVisible();

  await page.getByRole("tab", { name: "Sources 3" }).click();
  await page.getByRole("button", { name: "Open evaluation for mem-inception" }).click();
  await expect(page).toHaveURL(/\/v2\/evaluation\?source_id=src-repo$/);
  await expect(page.getByRole("button", { name: "Show all sources" })).toBeVisible();
});
