import { readFileSync } from "node:fs";
import { expect, test, type Page, type Route } from "@playwright/test";
import { appPath, appUrl } from "./support/appPath";

/** The server's real `GET /genes` and config-schema answers, as the unit tests read them. */
const FORMS = JSON.parse(readFileSync(new URL("../src/test/fixtures/sourceTypeForms.json", import.meta.url), "utf8")) as {
  genes: unknown[];
  schemas: Record<string, unknown>;
};

const CONFLUENCE_CONFIG = { base_url: "https://wiki.example.com", sync_mode: "space", spaces: ["PAY"], pat_configured: true };

const SOURCE = {
  id: "src-wiki",
  type: "confluence",
  name: "Payroll wiki",
  config: CONFLUENCE_CONFIG,
  status: "active",
  access_policy: "workspace",
  access_state: "active",
  last_sync: new Date().toISOString(),
  doc_count: 412,
  memory_count: 96,
  sync: { status: "success", run_id: "run-1", started_at: null, finished_at: null, error_message: null },
  connection_status: { state: "ready", reason: null },
  created_at: "2026-09-01T00:00:00Z",
  project_binding: { mode: "fixed", project_key: "PAY" },
  capabilities: { can_sync: true, can_subscribe: true, can_configure: true, can_configure_connection: true, can_delete: true },
  execution: { kind: "server", operation: null, immutable_config_fields: [] },
  enabled_for_me: true,
  pinned_for_me: false,
  sync_schedule: null,
};

const RESPONSES: Record<string, unknown> = {
  "GET /api/v1/genes": FORMS.genes,
  "GET /api/v1/genes/confluence/config-schema": FORMS.schemas.confluence,
  "GET /api/v1/projects": { data: [{ id: "p1", key: "PAY", name: "Payroll", kind: "normal", memory_count: 1 }], can_manage: true },
  "GET /api/v1/source-list/preferences": { sort_mode: "name" },
  "GET /api/cloud/local-agent/status": { status: "online", last_seen_at: null, checked_at: "", stale_after_seconds: 60 },
  "GET /api/cloud/local-agent/jobs/current": { data: [] },
  "POST /api/v1/genes/confluence/config-check": { derived: {}, facts: [], issues: [] },
};

/** Stubs the API: `routes` answers first, then the shared responses; anything else is a 404 the test would notice. */
async function stubApi(page: Page, routes: Record<string, (route: Route) => unknown>) {
  await page.route("**/api/**", (route) => {
    const request = route.request();
    const key = `${request.method()} ${new URL(request.url()).pathname}`;
    const handler = routes[key];
    if (handler) return handler(route);
    const body = RESPONSES[key];
    return body === undefined ? route.fulfill({ status: 404, json: { detail: "Not stubbed" } }) : route.fulfill({ json: body });
  });
}

test("a Confluence source is created from the type picker", async ({ page }) => {
  const created: unknown[] = [];
  await stubApi(page, {
    "GET /api/v1/sources": (route) => route.fulfill({ json: { data: created.length > 0 ? [SOURCE] : [] } }),
    "POST /api/v1/sources": (route) => {
      created.push(route.request().postDataJSON());
      return route.fulfill({ json: { id: "src-wiki", name: "Payroll wiki", type: "confluence", access_policy: "private" } });
    },
  });

  await page.goto(appPath("sources"));
  await page.getByRole("button", { name: "Add source" }).first().click();
  await expect(page).toHaveURL(appUrl("sources/new"));
  await page.getByRole("list", { name: "Source types" }).getByRole("link", { name: /Confluence/ }).click();
  await expect(page).toHaveURL(appUrl("sources/new/confluence"));

  await page.getByLabel("Source name").fill("Payroll wiki");
  await page.getByRole("button", { name: "Next" }).click();
  await page.getByLabel("Wiki URL").fill("https://wiki.example.com");
  await page.getByLabel("Personal Access Token").fill("token-1");
  await page.getByRole("button", { name: "Next" }).click();
  await page.getByLabel("Spaces to Sync").fill("PAY");
  await page.getByLabel("Spaces to Sync").press("Enter");
  await page.getByRole("button", { name: "Create source" }).click();

  await expect.poll(() => created).toEqual([
    {
      name: "Payroll wiki",
      type: "confluence",
      access_policy: "private",
      config: { base_url: "https://wiki.example.com", pat: "token-1", spaces: ["PAY"] },
      project_binding: null,
      sync_schedule: { enabled: false, interval_minutes: 1440 },
    },
  ]);
  await expect(page).toHaveURL(appUrl("sources"));
});

test("a scope edit saved during a running sync stops the sync first", async ({ page }) => {
  const saved: unknown[] = [];
  const stopped: string[] = [];
  const running = {
    ...SOURCE,
    sync: { status: "running", run_id: "ssr-1", started_at: new Date().toISOString(), finished_at: null, error_message: null },
  };
  await stubApi(page, {
    "GET /api/v1/sources": (route) => route.fulfill({ json: { data: [running] } }),
    "POST /api/v1/sources/src-wiki/scope-change-preview": (route) =>
      route.fulfill({ json: { outcome: "opens_transition", open_transition: false, refusal: null } }),
    "POST /api/v1/sources/src-wiki/sync/stop": (route) => {
      stopped.push("src-wiki");
      return route.fulfill({ json: { source_id: "src-wiki", outcome: "stopping", run_id: "ssr-1", local_agent_job_ids: [] } });
    },
    "GET /api/v1/sources/src-wiki/sync-runs/ssr-1": (route) => route.fulfill({ json: { run_id: "ssr-1", status: "failed" } }),
    "PUT /api/v1/sources/src-wiki": (route) => {
      saved.push(route.request().postDataJSON());
      return route.fulfill({ json: { ok: true, id: "src-wiki" } });
    },
  });

  await page.goto(appPath("sources/src-wiki/settings?step=scope"));
  await expect(page.getByText("A sync of this source is running")).toBeVisible();
  await page.getByLabel("Spaces to Sync").fill("OPS");
  await page.getByLabel("Spaces to Sync").press("Enter");
  await page.getByRole("button", { name: "Save changes" }).click();
  const dialog = page.getByRole("dialog", { name: "Stop the sync and save?" });
  await dialog.getByRole("button", { name: "Stop the sync and save" }).click();

  await expect.poll(() => saved).toEqual([
    {
      name: "Payroll wiki",
      sync_schedule: { enabled: false, interval_minutes: 1440 },
      project_binding: { mode: "fixed", project_key: "PAY" },
      config: { base_url: "https://wiki.example.com", sync_mode: "space", spaces: ["PAY", "OPS"] },
    },
  ]);
  expect(stopped).toEqual(["src-wiki"]);
  await expect(page).toHaveURL(appUrl("sources"));
});

/** Stubs local setup jobs: each job is named for its operation and completes with what `results` gives it. */
function setupJobs(results: Record<string, unknown>, created: unknown[]): Record<string, (route: Route) => unknown> {
  const routes: Record<string, (route: Route) => unknown> = {
    "POST /api/cloud/local-agent/jobs": (route) => {
      const body = route.request().postDataJSON() as { operation: string };
      created.push(body);
      return route.fulfill({ status: 201, json: { job_id: `laj-${body.operation}`, status: "queued" } });
    },
  };
  for (const [operation, result] of Object.entries(results)) {
    routes[`GET /api/cloud/local-agent/jobs/laj-${operation}`] = (route) =>
      route.fulfill({ json: { job_id: `laj-${operation}`, status: "succeeded", last_error: null, result: { operation, error: null, ...(result as object) } } });
  }
  return routes;
}

test("a Teams source is created with the conversations its session lists", async ({ page }) => {
  const created: unknown[] = [];
  const jobs: unknown[] = [];
  await stubApi(page, {
    "GET /api/v1/sources": (route) => route.fulfill({ json: { data: [] } }),
    "GET /api/v1/genes/teams/config-schema": (route) => route.fulfill({ json: FORMS.schemas.teams }),
    "POST /api/v1/genes/teams/config-check": (route) => route.fulfill({ json: { derived: {}, facts: [], issues: [] } }),
    "POST /api/v1/sources": (route) => {
      created.push(route.request().postDataJSON());
      return route.fulfill({ json: { id: "src-teams", name: "Team chats", type: "teams", access_policy: "private" } });
    },
    ...setupJobs(
      {
        teams_auth_check: { region: "emea", authenticated: true, expires_in_minutes: 50 },
        teams_browse: {
          region: "emea",
          favorites: [],
          teams: [{ id: "t-eng", displayName: "Engineering", channels: [{ id: "19:arch@thread.tacv2", displayName: "Architecture" }] }],
          group_chats: [{ id: "19:planning@thread.v2", topic: "Planning chat", lastActivity: null }],
          individual_chats: [],
        },
      },
      jobs,
    ),
  });

  await page.goto(appPath("sources/new/teams"));
  await page.getByLabel("Source name").fill("Team chats");
  await page.getByRole("button", { name: "Next" }).click();
  await expect(page.getByText("Teams session active")).toBeVisible();
  await page.getByRole("button", { name: "Next" }).click();
  await page.getByRole("list", { name: "Teams conversations" }).getByRole("button", { name: /Planning chat/ }).click();
  await expect(page.getByRole("list", { name: "Selected conversations" })).toContainText("Planning chat");
  await page.getByRole("button", { name: "Create source" }).click();

  await expect.poll(() => created).toEqual([
    {
      name: "Team chats",
      type: "teams",
      access_policy: "private",
      config: { conversation_ids: ["19:planning@thread.v2"] },
      project_binding: null,
      sync_schedule: { enabled: false, interval_minutes: 60 },
    },
  ]);
  expect(jobs).toEqual([
    { source_id: "", source_type: "teams", operation: "teams_auth_check", payload: { region: "emea" } },
    { source_id: "", source_type: "teams", operation: "teams_browse", payload: { region: "emea" } },
  ]);
  await expect(page).toHaveURL(appUrl("sources"));
});

test("a GitHub repository source syncs the folder chosen from its tree", async ({ page }) => {
  const created: unknown[] = [];
  await stubApi(page, {
    "GET /api/v1/sources": (route) => route.fulfill({ json: { data: [] } }),
    "GET /api/v1/genes/github_repo/config-schema": (route) => route.fulfill({ json: FORMS.schemas.github_repo }),
    "POST /api/v1/genes/github_repo/config-check": (route) => route.fulfill({ json: { derived: {}, facts: [], issues: [] } }),
    "POST /api/v1/genes/github_repo/browse-tree": (route) =>
      route.fulfill({
        json: {
          source_type: "github_repo",
          ref: "main",
          count: 3,
          truncated: false,
          items: [
            { path: "docs", type: "tree" },
            { path: "docs/guide.md", type: "blob", size: 10 },
            { path: "README.md", type: "blob", size: 10 },
          ],
        },
      }),
    "POST /api/v1/sources": (route) => {
      created.push(route.request().postDataJSON());
      return route.fulfill({ json: { id: "src-repo", name: "Docs repo", type: "github_repo", access_policy: "private" } });
    },
  });

  await page.goto(appPath("sources/new/github_repo"));
  await page.getByLabel("Source name").fill("Docs repo");
  await page.getByRole("button", { name: "Next" }).click();
  await page.getByLabel("Repository URL").fill("https://github.com/acme/docs");
  await page.getByRole("button", { name: "Next" }).click();
  await page.getByRole("button", { name: "Browse repository" }).first().click();
  await page.getByRole("tree", { name: "Repository paths" }).getByRole("checkbox", { name: "docs", exact: true }).click();
  await page.getByRole("button", { name: "Create source" }).click();

  await expect.poll(() => created).toEqual([
    {
      name: "Docs repo",
      type: "github_repo",
      access_policy: "private",
      config: { repo_url: "https://github.com/acme/docs", include_paths: ["docs"] },
      project_binding: null,
      sync_schedule: { enabled: false, interval_minutes: 1440 },
    },
  ]);
  await expect(page).toHaveURL(appUrl("sources"));
});
