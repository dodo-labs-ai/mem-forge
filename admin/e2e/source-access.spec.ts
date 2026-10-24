import { expect, test } from "@playwright/test";
import { appPath, appUrl } from "./support/appPath";

/** The harness page that renders the app in a workspace other people use (see `vite.config.ts`). */
function sharedWorkspacePath(page: string): string {
  return `${appPath("e2e/harness/shared-workspace.html")}?page=${encodeURIComponent(page)}`;
}

const SOURCE = {
  id: "src-wiki",
  type: "confluence",
  name: "Payroll wiki",
  config: {},
  status: "active",
  access_policy: "private",
  access_state: "active",
  access_transition: null,
  ownership: {
    created_by_user_id: "u1",
    owner_user_id: "u1",
    execution_owner_user_id: null,
    viewer_role: "owner",
    viewer_relationship: "owner",
  },
  last_sync: new Date().toISOString(),
  doc_count: 412,
  memory_count: 96,
  sync: { status: "success", started_at: null, finished_at: null, error_message: null },
  connection_status: { state: "ready", reason: null },
  created_at: "2026-09-01T00:00:00Z",
  project_binding: { mode: "fixed", project_key: "PAY" },
  capabilities: { can_sync: true, can_subscribe: true, can_configure: true, can_delete: true, can_change_access: true },
  execution: { kind: "server", operation: null, immutable_config_fields: [] },
  enabled_for_me: true,
  pinned_for_me: false,
};

const TRANSITION = {
  operation_id: "sat-1",
  source_id: "src-wiki",
  previous_policy: "private",
  target_policy: "workspace",
  status: "running",
  total_memories: 4,
  processed_memories: 1,
  error_code: null,
  error_message: null,
  created_at: "2026-10-24T09:00:00Z",
  updated_at: "2026-10-24T09:00:00Z",
  completed_at: null,
};

const RESPONSES: Record<string, unknown> = {
  "/api/v1/projects": { data: [{ id: "p1", key: "PAY", name: "Payroll", kind: "normal" }], can_manage: true },
  "/api/v1/genes": [{ name: "confluence", display_name: "Confluence" }],
  "/api/v1/source-list/preferences": { sort_mode: "name" },
  "/api/cloud/local-agent/status": { status: "online", last_seen_at: null, checked_at: "", stale_after_seconds: 60 },
  "/api/cloud/local-agent/jobs/current": { data: [] },
};

test("a private source is shared with the workspace and the page follows the change until it is applied", async ({ page }) => {
  const starts: { body: unknown; key: string | null }[] = [];
  let reads = 0;
  await page.route("**/api/**", (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === "POST" && path === "/api/v1/sources/src-wiki/access-transitions") {
      starts.push({ body: request.postDataJSON(), key: request.headers()["idempotency-key"] ?? null });
      return route.fulfill({ status: 202, json: { ...TRANSITION, status: "queued", total_memories: 0, processed_memories: 0 } });
    }
    if (path === "/api/v1/sources") {
      // Once the change starts, the list shows it running for two reads, then applied.
      const source =
        starts.length === 0
          ? SOURCE
          : reads++ < 2
            ? { ...SOURCE, access_state: "changing", access_transition: TRANSITION }
            : { ...SOURCE, access_policy: "workspace" };
      return route.fulfill({ json: { data: [source] } });
    }
    const body = RESPONSES[path];
    return body === undefined ? route.fulfill({ status: 404, json: { detail: "Not stubbed" } }) : route.fulfill({ json: body });
  });

  await page.goto(sharedWorkspacePath("sources"));
  await expect(page).toHaveURL(appUrl("sources"));

  const table = page.getByRole("table", { name: "Sources" });
  const row = table.getByRole("row").filter({ hasText: "Payroll wiki" });
  await expect(row.getByText("Only me")).toBeVisible();
  await row.getByRole("button", { name: "More actions for Payroll wiki" }).click();
  await page.getByRole("menuitem", { name: "Change who can use this source" }).click();

  const dialog = page.getByRole("dialog", { name: "Change who can use Payroll wiki" });
  await expect(dialog.getByRole("radio", { name: "Everyone in this workspace" })).toBeChecked();
  await dialog.getByRole("button", { name: "Change access" }).click();
  await expect(dialog).toBeHidden();
  expect(starts).toEqual([{ body: { target_policy: "workspace" }, key: expect.any(String) }]);

  await expect(row.getByText("Changing access")).toBeVisible();
  await row.getByText("Payroll wiki").click();
  const drawer = page.getByRole("dialog", { name: "Payroll wiki" });
  await expect(drawer.getByText("Changing access to Everyone in this workspace")).toBeVisible();
  await expect(drawer.getByText("1 of 4 memories updated")).toBeVisible();

  // The open drawer hides the table from assistive technology, so the drawer's own Access line is read.
  await expect(drawer.getByText("Workspace", { exact: true })).toBeVisible({ timeout: 10_000 });
  await expect(drawer.getByText("Changing access to Everyone in this workspace")).toBeHidden();
});

test("a failed access change offers retry and revert to the owner", async ({ page }) => {
  const settled: string[] = [];
  const failed = { ...TRANSITION, status: "failed", error_message: "Search index offline" };
  await page.route("**/api/**", (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === "POST" && path.startsWith("/api/v1/sources/src-wiki/access-transitions/sat-1/")) {
      settled.push(path.split("/").at(-1)!);
      return route.fulfill({ json: { operation_id: "sat-1", status: "queued" } });
    }
    if (path === "/api/v1/sources/src-wiki/access-transition") {
      return route.fulfill({ json: { transition: settled.length > 0 ? { ...TRANSITION, status: "completed" } : failed } });
    }
    if (path === "/api/v1/sources") {
      const source =
        settled.length > 0
          ? { ...SOURCE, access_policy: "private" }
          : { ...SOURCE, status: "paused", access_state: "changing", access_transition: failed };
      return route.fulfill({ json: { data: [source] } });
    }
    const body = RESPONSES[path];
    return body === undefined ? route.fulfill({ status: 404, json: { detail: "Not stubbed" } }) : route.fulfill({ json: body });
  });

  await page.goto(sharedWorkspacePath("sources"));

  const row = page.getByRole("table", { name: "Sources" }).getByRole("row").filter({ hasText: "Payroll wiki" });
  await expect(row.getByText("Access change failed")).toBeVisible();
  await row.getByText("Payroll wiki").click();
  const drawer = page.getByRole("dialog", { name: "Payroll wiki" });
  await expect(drawer.getByText("The change to Everyone in this workspace failed")).toBeVisible();
  await expect(drawer.getByText("Search index offline")).toBeVisible();
  await expect(drawer.getByRole("button", { name: "Retry" })).toBeVisible();

  await drawer.getByRole("button", { name: "Revert to Only me" }).click();
  await expect(drawer.getByText("The change to Everyone in this workspace failed")).toBeHidden();
  await expect(drawer.getByText("Only me", { exact: true })).toBeVisible();
  expect(settled).toEqual(["revert"]);
});
