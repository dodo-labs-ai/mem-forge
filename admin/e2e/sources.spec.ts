import { expect, test } from "@playwright/test";
import { appPath, appUrl } from "./support/appPath";

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
  "/api/v1/projects": { data: [{ id: "p1", key: "PAY", name: "Payroll", kind: "normal" }], can_manage: true },
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

  await page.goto(appPath());

  await expect(page).toHaveURL(appUrl("sources"));
  await expect(page.getByRole("heading", { name: "Sources" })).toBeVisible();
  await expect(page.getByRole("table", { name: "Sources" }).getByText("Payroll wiki")).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Sources" })).toHaveAttribute("aria-current", "page");
  await expect(page.getByText("Local sync offline")).toBeVisible();
});

test("a source's scope change in progress shows in its drawer and restores the previous scope", async ({ page }) => {
  const restoreConfig = { base_url: "https://wiki.example", sync_mode: "spaces", spaces: ["ENG"] };
  const source = {
    ...SOURCE,
    scope_transition: {
      id: "scope-transition-1",
      status: "failed",
      previous_scope: { sync_mode: "spaces", spaces: ["ENG"] },
      target_scope: { sync_mode: "spaces", spaces: ["ENG", "OPS"] },
      created_at: "2026-09-29T09:00:00Z",
      updated_at: "2026-09-29T11:00:00Z",
      members_committed: 117,
      listed_count: 120,
      excluded_scopes_due: false,
      error: null,
      failing_items: { count: 0, retrying_after_upgrade: 0, items: [] },
      restore_config: restoreConfig,
      removes: "at_completion",
    },
  };
  const saved: unknown[] = [];
  await page.route("**/api/**", (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (request.method() === "POST" && path === "/api/v1/sources/src-wiki/scope-change-preview") {
      return route.fulfill({
        json: { outcome: "returns_to_previous_scope", open_transition: true, removes: "at_completion", refusal: null },
      });
    }
    if (request.method() === "PUT" && path === "/api/v1/sources/src-wiki") {
      saved.push(request.postDataJSON());
      return route.fulfill({ json: { ok: true, id: "src-wiki" } });
    }
    const body = path === "/api/v1/sources" ? { data: [source] } : RESPONSES[path];
    return body === undefined ? route.fulfill({ status: 404, json: { detail: "Not stubbed" } }) : route.fulfill({ json: body });
  });

  await page.goto(appPath("sources"));
  await page.getByRole("table", { name: "Sources" }).getByText("Payroll wiki").click();
  const drawer = page.getByRole("dialog", { name: "Payroll wiki" });
  await expect(drawer.getByRole("heading", { name: "Scope change in progress" })).toBeVisible();
  await expect(drawer.getByText("117 of 120 listed items synced under the new scope")).toBeVisible();

  await drawer.getByRole("button", { name: "Restore the previous scope" }).click();
  await page.getByRole("dialog", { name: "Restore the previous scope?" }).getByRole("button", { name: "Restore" }).click();

  await expect.poll(() => saved).toEqual([{ config: restoreConfig, expected_scope_transition_id: "scope-transition-1" }]);
});

test("an agent-session source's project rows keep each name clear of its counts", async ({ page }) => {
  const sessions = {
    ...SOURCE,
    id: "src-agent-sessions-codex",
    type: "agent_session",
    client: "codex",
    name: "Codex Session",
    sync: null,
    capabilities: { can_sync: false, can_subscribe: true, can_configure: false, can_delete: false },
  };
  const responses: Record<string, unknown> = {
    ...RESPONSES,
    "/api/v1/sources": { data: [sessions] },
    "/api/v1/genes": [{ name: "agent_session", display_name: "Agent sessions" }],
    "/api/v1/agent-sessions/completeness": {
      session_id: null,
      source_id: sessions.id,
      total: 400,
      processed_total: 400,
      counts: { knowledge_patched: 182, no_output: 218, failed: 0 },
      no_output_fraction: 0.545,
      latest_failure: null,
    },
    [`/api/v1/sources/${sessions.id}/projects`]: {
      source_id: sessions.id,
      projects: [{ project: "UNSORTED", document_count: 182, memory_count: 1204, last_observed_at: new Date().toISOString() }],
    },
  };
  await page.route("**/api/**", (route) => {
    const body = responses[new URL(route.request().url()).pathname];
    return body === undefined ? route.fulfill({ status: 404, json: { detail: "Not stubbed" } }) : route.fulfill({ json: body });
  });
  await page.setViewportSize({ width: 800, height: 900 });

  await page.goto(appPath("sources"));
  await page.getByRole("table", { name: "Sources" }).getByText("Codex Session").click();
  const row = page.getByRole("dialog", { name: "Codex Session" }).getByRole("list", { name: "Projects with memories from sessions" }).getByRole("listitem");
  const name = row.getByText("UNSORTED");
  const nameBox = (await name.boundingBox())!;
  const summariesBox = (await row.getByText("182 summaries").boundingBox())!;

  // The name's text fits its own box, and that box ends before the counts start or sits on a line above them.
  expect(await name.evaluate((element: { scrollWidth: number; clientWidth: number }) => element.scrollWidth - element.clientWidth)).toBe(0);
  expect(nameBox.x + nameBox.width <= summariesBox.x || nameBox.y + nameBox.height <= summariesBox.y).toBe(true);
});
