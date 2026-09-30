import { expect, test } from "@playwright/test";

const SOURCE = {
  id: "src-wiki",
  type: "confluence",
  name: "Payroll wiki",
  config: {},
  status: "active",
  access_policy: "workspace",
  access_state: "active",
  last_sync: new Date().toISOString(),
  doc_count: 412,
  memory_count: 96,
  sync: { status: "success", started_at: null, finished_at: null, error_message: null },
  connection_status: { state: "ready", reason: null },
  created_at: "2026-09-01T00:00:00Z",
  project_binding: { mode: "fixed", project_key: "PAY" },
  capabilities: { can_sync: true, can_subscribe: true, can_configure: true, can_delete: true },
  execution: { kind: "server", operation: null, immutable_config_fields: [] },
  enabled_for_me: true,
  pinned_for_me: false,
};

const RESPONSES: Record<string, unknown> = {
  "/api/v1/sources": { data: [SOURCE] },
  "/api/v1/projects": [{ id: "p1", key: "PAY", name: "Payroll", kind: "normal" }],
  "/api/v1/genes": [{ name: "confluence", display_name: "Confluence" }],
  "/api/v1/source-list/preferences": { sort_mode: "name" },
  "/api/cloud/local-agent/status": { status: "offline", last_seen_at: null, checked_at: "", stale_after_seconds: 60 },
  "/api/cloud/local-agent/jobs/current": { data: [] },
};

test("the Sources page renders in the shell", async ({ page }) => {
  await page.route("**/api/**", (route) => {
    const body = RESPONSES[new URL(route.request().url()).pathname];
    return body === undefined ? route.fulfill({ status: 404, json: { detail: "Not stubbed" } }) : route.fulfill({ json: body });
  });

  await page.goto("/v2/");

  await expect(page).toHaveURL(/\/v2\/sources$/);
  await expect(page.getByRole("heading", { name: "Sources" })).toBeVisible();
  await expect(page.getByRole("table", { name: "Sources" }).getByText("Payroll wiki")).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Sources" })).toHaveAttribute("aria-current", "page");
  await expect(page.getByText("Local sync offline")).toBeVisible();
});
