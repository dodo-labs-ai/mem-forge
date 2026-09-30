import { expect, test, type Page } from "@playwright/test";

const SOURCE = {
  id: "src-jira",
  type: "jira",
  name: "SFPAY board",
  config: {},
  status: "active",
  access_policy: "workspace",
  access_state: "active",
  last_sync: new Date().toISOString(),
  doc_count: 120,
  memory_count: 611,
  sync: { status: "success", started_at: null, finished_at: null, error_message: null },
  connection_status: { state: "ready", reason: null },
  created_at: "2026-09-01T00:00:00Z",
  project_binding: { mode: "fixed", project_key: "PAYROLL_V2" },
  capabilities: { can_sync: true, can_subscribe: true, can_configure: true, can_delete: true },
  execution: { kind: "server", operation: null, immutable_config_fields: [] },
  enabled_for_me: true,
  pinned_for_me: false,
};

const PROJECTS = [
  { id: "p-shared", key: "SHARED", name: "Shared", kind: "shared", memory_count: 1441, created_at: "2026-06-01T00:00:00Z" },
  { id: "p-unsorted", key: "UNSORTED", name: "Unsorted", kind: "normal", memory_count: 40, created_at: "2026-06-01T00:00:00Z" },
  { id: "p-payroll", key: "PAYROLL_V2", name: "payroll-v2", kind: "normal", memory_count: 708, created_at: "2026-07-02T00:00:00Z" },
  { id: "p-memforge", key: "MEMFORGE", name: "memforge", kind: "normal", memory_count: 1388, created_at: "2026-06-04T00:00:00Z" },
];

const RESPONSES: Record<string, unknown> = {
  "GET /api/v1/projects": { data: PROJECTS, can_manage: true },
  "GET /api/v1/projects/p-payroll/deletion-impact": { memory_count: 708, source_count: 1 },
  "GET /api/v1/sources": { data: [SOURCE] },
  "GET /api/v1/genes": [{ name: "jira", display_name: "Jira" }],
  "GET /api/cloud/local-agent/status": { status: "online", last_seen_at: null, checked_at: "", stale_after_seconds: 60 },
  "DELETE /api/v1/projects/p-payroll": {
    id: "p-payroll",
    rebucketed_count: 708,
    rebucketed_memory_ids: [],
    released_source_count: 1,
  },
};

async function stubApi(page: Page) {
  await page.route("**/api/**", (route) => {
    const request = route.request();
    const body = RESPONSES[`${request.method()} ${new URL(request.url()).pathname}`];
    return body === undefined ? route.fulfill({ status: 404, json: { detail: "Not stubbed" } }) : route.fulfill({ json: body });
  });
}

test("the Projects page lists projects with their sources and memories", async ({ page }) => {
  await stubApi(page);
  await page.goto("/v2/projects");

  await expect(page.getByRole("heading", { name: "Projects" })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Projects" })).toHaveAttribute("aria-current", "page");
  const payroll = page.getByRole("row").filter({ has: page.getByRole("link", { name: "payroll-v2" }) });
  await expect(payroll.getByRole("cell").nth(1)).toHaveText("1");
  await expect(payroll.getByRole("cell").nth(2)).toHaveText("708");
  const shared = page.getByRole("row").filter({ has: page.getByRole("link", { name: "Shared" }) });
  await expect(shared.getByText("Team-wide")).toBeVisible();
  await expect(shared.getByRole("button")).toHaveCount(0);

  await page.getByRole("button", { name: "Actions for payroll-v2" }).click();
  await page.getByRole("menuitem", { name: "Delete…" }).click();
  const dialog = page.getByRole("dialog", { name: "Delete payroll-v2?" });
  await expect(
    dialog.getByText("708 memories move to the Unsorted project. 1 source will stop writing to this project."),
  ).toBeVisible();
  await dialog.getByRole("button", { name: "Continue" }).click();
  await dialog.getByLabel(/to confirm/).fill("PAYROLL_V2");
  await dialog.getByRole("button", { name: "Delete project" }).click();
  await expect(
    page.getByText(
      "Project “payroll-v2” deleted. 708 memories moved to the Unsorted project. 1 source stopped writing to this project.",
    ),
  ).toBeVisible();
});

test("a project's detail page shows its sources and links to its memories", async ({ page }) => {
  await stubApi(page);
  await page.goto("/v2/projects/PAYROLL_V2");

  await expect(page.getByRole("heading", { name: "payroll-v2" })).toBeVisible();
  await expect(page.getByRole("link", { name: "View memories" })).toHaveAttribute("href", "/v2/memories?project=PAYROLL_V2");
  const sources = page.getByRole("region", { name: "Sources sending memories here" });
  await expect(sources.getByText("SFPAY board")).toBeVisible();
  await expect(sources.getByText("611 memories")).toBeVisible();
  await expect(page).toHaveTitle(/^Projects/);
});
