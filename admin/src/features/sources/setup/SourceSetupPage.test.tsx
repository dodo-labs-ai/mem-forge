import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLocation } from "react-router-dom";
import { afterEach, beforeAll, expect, test, vi } from "vitest";
import type { Source } from "@/api";
import { errorResponse, type Route } from "@/test/fakeApi";
import { renderRoutes } from "@/test/renderRoutes";
import { setupJobRoutes } from "@/test/setupJobs";
import { makeScopeTransition, makeSource } from "@/test/sourceFixtures";
import { SOURCE_TYPE_FORMS, sourceTypeSchema } from "@/test/sourceTypeForms";
import { NewSourcePage, SourceSettingsPage } from "./SourceSetupPage";

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

// The New project dialog loads with the Projects feature when it opens; loading it first keeps that out of each test's wait.
beforeAll(async () => {
  await import("@/features/projects");
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

const PAYROLL = { id: "p-pay", key: "PAY", name: "Payroll", kind: "normal", memory_count: 4, created_at: null };
const SHARED = { id: "p-shared", key: "SHARED", name: "Shared", kind: "shared", memory_count: 1, created_at: null };
const NO_ISSUES = { derived: {}, facts: [], issues: [] };
const CREATED = { id: "src-new", name: "Payroll wiki", type: "confluence", access_policy: "private" };
const CONFLUENCE_CONFIG = { base_url: "https://wiki.example.com", sync_mode: "space", spaces: ["PAY"], pat_configured: true };

function CurrentLocation() {
  const location = useLocation();
  return <output aria-label="Location">{`${location.pathname}${location.search}`}</output>;
}

function renderSetup(path: string, { sources = [], routes = {} }: { sources?: Source[]; routes?: Record<string, Route> } = {}) {
  const schemas = Object.fromEntries(
    Object.keys(SOURCE_TYPE_FORMS.schemas).map((type) => [`GET /api/v1/genes/${type}/config-schema`, () => sourceTypeSchema(type)]),
  );
  return renderRoutes(
    [
      { path: "/sources/new/:type", element: <NewSourcePage /> },
      { path: "/sources/:sourceId/settings", element: <SourceSettingsPage /> },
      { path: "*", element: <CurrentLocation /> },
    ],
    {
      path,
      api: {
        "GET /api/v1/genes": () => SOURCE_TYPE_FORMS.genes,
        ...schemas,
        "GET /api/v1/projects": () => ({ data: [SHARED, PAYROLL], can_manage: true }),
        "GET /api/v1/sources": () => ({ data: sources }),
        "POST /api/v1/genes/confluence/config-check": () => NO_ISSUES,
        "POST /api/v1/genes/jira/config-check": () => NO_ISSUES,
        ...routes,
      },
    },
  );
}

function requestsTo(calls: Request[], method: string, path: string): Request[] {
  return calls.filter((request) => request.method === method && new URL(request.url).pathname === path);
}

async function bodyOf(calls: Request[], method: string, path: string): Promise<unknown> {
  await waitFor(() => expect(requestsTo(calls, method, path)).not.toHaveLength(0));
  return requestsTo(calls, method, path).at(-1)!.clone().json();
}

async function next() {
  await userEvent.click(screen.getByRole("button", { name: "Next" }));
}

async function choose(trigger: HTMLElement, option: string) {
  await userEvent.click(trigger);
  await userEvent.click(await screen.findByRole("option", { name: option }));
}

/** Fills a new Confluence source up to its project step: name, site, token and one space. */
async function fillConfluence() {
  await userEvent.type(await screen.findByLabelText("Source name"), "Payroll wiki");
  await next();
  await userEvent.type(await screen.findByLabelText("Wiki URL"), "https://wiki.example.com");
  await userEvent.type(screen.getByLabelText("Personal Access Token"), "token-1");
  await next();
  await userEvent.type(await screen.findByLabelText("Spaces to Sync"), "PAY{Enter}");
}

function running(source: Source): Source {
  return {
    ...source,
    sync: { status: "running", run_id: "ssr-1", started_at: "2026-10-24T08:00:00Z", finished_at: null, error_message: null, progress: null },
  };
}

test("creates a Confluence source with project and schedule", async () => {
  const { calls } = renderSetup("/sources/new/confluence", { routes: { "POST /api/v1/sources": () => CREATED } });

  expect(await screen.findByRole("heading", { name: "Add Confluence" })).toBeInTheDocument();
  await fillConfluence();
  await next();
  await choose(await screen.findByRole("combobox", { name: "Save memories to" }), "Payroll");
  await next();
  await userEvent.click(await screen.findByRole("switch", { name: "Sync automatically" }));
  await choose(screen.getByRole("combobox", { name: "Frequency" }), "Hourly");
  await userEvent.click(screen.getByRole("button", { name: "Create source" }));

  expect(await bodyOf(calls, "POST", "/api/v1/sources")).toEqual({
    name: "Payroll wiki",
    type: "confluence",
    access_policy: "private",
    config: { base_url: "https://wiki.example.com", pat: "token-1", spaces: ["PAY"] },
    project_binding: { mode: "fixed", project_key: "PAY" },
    sync_schedule: { enabled: true, interval_minutes: 60 },
  });
  expect(await screen.findByRole("status", { name: "Location" })).toHaveTextContent(/^\/sources$/);
});

test("inline project creation selects the new project", async () => {
  let projects = [SHARED, PAYROLL];
  const payments = { id: "p-new", key: "PAYMENTS", name: "Payments", kind: "normal", created_at: null };
  renderSetup("/sources/new/confluence?step=project", {
    routes: {
      "GET /api/v1/projects": () => ({ data: projects, can_manage: true }),
      "POST /api/v1/projects": () => {
        projects = [...projects, { ...payments, memory_count: 0 }];
        return payments;
      },
    },
  });

  await userEvent.click(await screen.findByRole("button", { name: "New project" }));
  const dialog = await screen.findByRole("dialog", { name: "New project" });
  await userEvent.type(within(dialog).getByLabelText(/Name/), "Payments");
  await userEvent.click(within(dialog).getByRole("button", { name: "Create" }));

  await waitFor(() => expect(screen.getByRole("combobox", { name: "Save memories to" })).toHaveTextContent("Payments"));
  expect(screen.queryByRole("dialog", { name: "New project" })).not.toBeInTheDocument();
});

test("Check setup lists the first five items", async () => {
  const items = Array.from({ length: 5 }, (_, index) => ({
    item_id: `page-${index}`,
    title: `Payroll page ${index + 1}`,
    source_url: `https://wiki.example.com/pages/${index}`,
  }));
  const { calls } = renderSetup("/sources/new/confluence", {
    routes: {
      "POST /api/v1/genes/confluence/preview-discovery": () => ({ source_type: "confluence", count: 5, items, truncated: true }),
    },
  });

  await fillConfluence();
  await userEvent.click(screen.getByRole("button", { name: "Check setup" }));

  const found = await screen.findByRole("list", { name: "First items found" });
  expect(within(found).getAllByRole("link").map((link) => link.textContent)).toEqual(items.map((item) => item.title));
  expect(screen.getByText("The first 5 items found")).toBeInTheDocument();
  expect(await bodyOf(calls, "POST", "/api/v1/genes/confluence/preview-discovery")).toEqual({
    config: { base_url: "https://wiki.example.com", pat: "token-1", spaces: ["PAY"] },
    limit: 5,
    source_id: null,
  });
});

test("Check setup for a local folder lists the files local sync finds there", async () => {
  const created: Parameters<typeof setupJobRoutes>[1] = [];
  renderSetup("/sources/new/local_markdown?step=connection", {
    routes: {
      "POST /api/v1/genes/local_markdown/config-check": () => NO_ISSUES,
      ...setupJobRoutes(
        {
          local_markdown_preview_tree: {
            status: "succeeded",
            result: {
              operation: "local_markdown_preview_tree",
              counts: { included: 12, ignored: 3, too_large: 0, invalid_utf8: 0, unreadable: 0 },
              items: [
                { relative_path: "payroll/runbook.md", title: "Payroll runbook" },
                { relative_path: "notes.md", title: "" },
              ],
            },
          },
        },
        created,
      ),
    },
  });

  await userEvent.type(await screen.findByLabelText("Folder Path"), "/Users/me/notes");
  await next();
  await userEvent.click(await screen.findByRole("button", { name: "Check setup" }));

  const found = await screen.findByRole("list", { name: "First items found" });
  expect(within(found).getAllByRole("listitem").map((item) => item.textContent)).toEqual(["Payroll runbook", "notes.md"]);
  expect(screen.getByText("The first 2 items found")).toBeInTheDocument();
  expect(created).toEqual([
    {
      source_id: "",
      source_type: "local_markdown",
      operation: "local_markdown_preview_tree",
      payload: { root: "/Users/me/notes", vault_id: "preview", limit: 5 },
    },
  ]);
});

test("a pasted Confluence page URL fills the page tree root and switches an unchosen sync scope to page tree", async () => {
  const pageUrl = "https://wiki.example.com/pages/viewpage.action?pageId=12345";
  const { calls } = renderSetup("/sources/new/confluence?step=connection", {
    routes: {
      "POST /api/v1/genes/confluence/config-check": () => ({
        derived: { base_url: "https://wiki.example.com", page_tree_root: "12345", sync_mode: "page_tree" },
        facts: [{ label: "Detected site", value: "wiki.example.com" }],
        issues: [],
      }),
    },
  });

  await userEvent.type(await screen.findByLabelText("Wiki URL"), pageUrl);
  await userEvent.tab();
  await waitFor(() => expect(requestsTo(calls, "POST", "/api/v1/genes/confluence/config-check")).not.toHaveLength(0));
  expect(screen.getByLabelText("Wiki URL")).toHaveValue(pageUrl);
  await next();

  expect(await screen.findByRole("radio", { name: "This page tree" })).toBeChecked();
  expect(within(screen.getByRole("region", { name: "Detected settings" })).getByText("wiki.example.com")).toBeInTheDocument();
  expect(screen.getByLabelText("Page Tree Root")).toHaveValue("12345");
  expect(screen.getByLabelText("Page Tree Root")).toHaveAccessibleDescription(expect.stringContaining("Filled from the URL"));
  expect(screen.queryByLabelText("Spaces to Sync")).not.toBeInTheDocument();
  expect(await bodyOf(calls, "POST", "/api/v1/genes/confluence/config-check")).toEqual({ config: { base_url: pageUrl }, source_id: null });
});

test("a sync scope the user chose is not overwritten by a derived value", async () => {
  renderSetup("/sources/new/confluence?step=scope", {
    routes: {
      "POST /api/v1/genes/confluence/config-check": () => ({ derived: { sync_mode: "page_tree" }, facts: [], issues: [] }),
    },
  });

  await userEvent.click(await screen.findByRole("radio", { name: "Whole space" }));

  await waitFor(() => expect(screen.getByRole("radio", { name: "Whole space" })).toBeChecked());
  expect(screen.getByLabelText("Spaces to Sync")).toBeInTheDocument();
  expect(screen.queryByLabelText("Page Tree Root")).not.toBeInTheDocument();
});

test("an untouched mode field is sent absent", async () => {
  const { calls } = renderSetup("/sources/new/jira", {
    routes: { "POST /api/v1/sources": () => ({ ...CREATED, type: "jira" }) },
  });

  await userEvent.type(await screen.findByLabelText("Source name"), "Payroll Jira");
  await next();
  expect(screen.getByRole("radio", { name: "Browser session" })).toBeChecked();
  await userEvent.type(screen.getByLabelText("Jira Base URL"), "https://jira.example.com");
  await next();
  await userEvent.type(await screen.findByLabelText("Projects to Sync"), "PAY{Enter}");
  await userEvent.click(screen.getByRole("button", { name: "Create source" }));

  const body = (await bodyOf(calls, "POST", "/api/v1/sources")) as { config: Record<string, unknown> };
  expect(body.config).toEqual({ base_url: "https://jira.example.com", projects: ["PAY"] });
});

test("a stored Confluence source with a root and a blank sync scope opens on the page tree step", async () => {
  const source = makeSource({ config: { base_url: "https://wiki.example.com", page_tree_root: "12345", pat_configured: true } });
  const { calls } = renderSetup("/sources/src-wiki/settings?step=scope", {
    sources: [source],
    routes: {
      "POST /api/v1/genes/confluence/config-check": () => ({ derived: { sync_mode: "page_tree" }, facts: [], issues: [] }),
    },
  });

  expect(await screen.findByRole("heading", { name: "Payroll wiki settings" })).toBeInTheDocument();
  expect(await screen.findByRole("radio", { name: "This page tree" })).toBeChecked();
  expect(screen.getByLabelText("Page Tree Root")).toHaveValue("12345");
  expect(await bodyOf(calls, "POST", "/api/v1/genes/confluence/config-check")).toEqual({
    config: { base_url: "https://wiki.example.com", page_tree_root: "12345" },
    source_id: "src-wiki",
  });
});

test("a config_invalid refusal shows each field's message", async () => {
  renderSetup("/sources/new/confluence", {
    routes: {
      "POST /api/v1/sources": () =>
        errorResponse(400, {
          code: "config_invalid",
          message: "Wiki URL must use HTTPS. The site did not answer.",
          fields: [
            { key: "base_url", message: "Wiki URL must use HTTPS." },
            { key: null, message: "The site did not answer." },
          ],
        }),
    },
  });

  await fillConfluence();
  await userEvent.click(screen.getByRole("button", { name: "Create source" }));

  expect(await screen.findByText("Wiki URL must use HTTPS.")).toBeInTheDocument();
  expect(screen.getByLabelText("Wiki URL")).toHaveAccessibleDescription(expect.stringContaining("Wiki URL must use HTTPS."));
  expect(screen.getByText("The site did not answer.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Confluence site, 1 issue" })).toHaveAttribute("aria-current", "step");
});

test("editing a scope field previews the transition and save sends expected_scope_transition_id", async () => {
  const source = makeSource({ config: CONFLUENCE_CONFIG, scope_transition: makeScopeTransition() });
  const { calls } = renderSetup("/sources/src-wiki/settings?step=scope", {
    sources: [source],
    routes: {
      "POST /api/v1/sources/src-wiki/scope-change-preview": () => ({ outcome: "supersedes_transition", open_transition: true, refusal: null }),
      "PUT /api/v1/sources/src-wiki": () => ({ ok: true, id: "src-wiki" }),
    },
  });

  await userEvent.type(await screen.findByLabelText("Spaces to Sync"), "OPS{Enter}");

  expect(await screen.findByText("This replaces the scope change in progress")).toBeInTheDocument();
  expect(await bodyOf(calls, "POST", "/api/v1/sources/src-wiki/scope-change-preview")).toEqual({
    config: { base_url: "https://wiki.example.com", sync_mode: "space", spaces: ["PAY", "OPS"] },
  });
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
  expect(await bodyOf(calls, "PUT", "/api/v1/sources/src-wiki")).toEqual({
    name: "Payroll wiki",
    sync_schedule: { enabled: false, interval_minutes: 1440 },
    project_binding: { mode: "fixed", project_key: "PAY" },
    config: { base_url: "https://wiki.example.com", sync_mode: "space", spaces: ["PAY", "OPS"] },
    expected_scope_transition_id: "scope-transition-1",
  });
});

test("a refused scope change explains why and blocks save", async () => {
  const source = makeSource({ config: CONFLUENCE_CONFIG });
  renderSetup("/sources/src-wiki/settings?step=connection", {
    sources: [source],
    routes: {
      "POST /api/v1/sources/src-wiki/scope-change-preview": () => ({
        outcome: "refused",
        open_transition: false,
        refusal: { code: "source_provider_namespace_immutable", message: null, mode: null },
      }),
    },
  });

  await screen.findByLabelText("Wiki URL");
  await next();
  await userEvent.type(await screen.findByLabelText("Spaces to Sync"), "OPS{Enter}");

  expect(await screen.findByText("This change cannot be saved")).toBeInTheDocument();
  expect(screen.getByText(/points the source at another site or project/)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
});

test("saving during a running sync stops it, waits for the run to end, then saves", async () => {
  const { calls } = renderSetup("/sources/src-wiki/settings", {
    sources: [running(makeSource({ config: CONFLUENCE_CONFIG }))],
    routes: {
      "POST /api/v1/sources/src-wiki/sync/stop": () => ({ source_id: "src-wiki", outcome: "stopping", run_id: "ssr-1", local_agent_job_ids: [] }),
      "GET /api/v1/sources/src-wiki/sync-runs/ssr-1": () => ({ run_id: "ssr-1", status: "failed" }),
      "PUT /api/v1/sources/src-wiki": () => ({ ok: true, id: "src-wiki" }),
    },
  });

  expect(await screen.findByText("A sync of this source is running")).toBeInTheDocument();
  const name = await screen.findByLabelText("Source name");
  await userEvent.clear(name);
  await userEvent.type(name, "Payroll handbook");
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
  const dialog = await screen.findByRole("dialog", { name: "Stop the sync and save?" });
  expect(requestsTo(calls, "PUT", "/api/v1/sources/src-wiki")).toHaveLength(0);

  await userEvent.click(within(dialog).getByRole("button", { name: "Stop the sync and save" }));

  const body = (await bodyOf(calls, "PUT", "/api/v1/sources/src-wiki")) as { name: string };
  expect(body.name).toBe("Payroll handbook");
  expect(requestsTo(calls, "POST", "/api/v1/sources/src-wiki/sync/stop")).toHaveLength(1);
  expect(requestsTo(calls, "GET", "/api/v1/sources/src-wiki/sync-runs/ssr-1")).toHaveLength(1);
  expect(await screen.findByRole("status", { name: "Location" })).toHaveTextContent(/^\/sources$/);
});

test("Wait keeps the edit and does not stop the sync", async () => {
  const { calls } = renderSetup("/sources/src-wiki/settings", {
    sources: [running(makeSource({ config: CONFLUENCE_CONFIG }))],
  });

  const name = await screen.findByLabelText("Source name");
  await userEvent.clear(name);
  await userEvent.type(name, "Payroll handbook");
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
  const dialog = await screen.findByRole("dialog", { name: "Stop the sync and save?" });
  await userEvent.click(within(dialog).getByRole("button", { name: "Wait" }));

  await waitFor(() => expect(screen.queryByRole("dialog", { name: "Stop the sync and save?" })).not.toBeInTheDocument());
  expect(screen.getByLabelText("Source name")).toHaveValue("Payroll handbook");
  expect(requestsTo(calls, "POST", "/api/v1/sources/src-wiki/sync/stop")).toHaveLength(0);
  expect(requestsTo(calls, "PUT", "/api/v1/sources/src-wiki")).toHaveLength(0);
});

test("fixed-after-create fields are read-only", async () => {
  const source = makeSource({
    id: "src-jira",
    type: "jira",
    name: "Payroll Jira",
    config: { base_url: "https://jira.example.com", sync_mode: "local_agent", auth_mode: "browser_cookie", projects: ["PAY"] },
    execution: { kind: "local_agent", operation: "jira_sync", immutable_config_fields: ["sync_mode"] },
  });
  renderSetup("/sources/src-jira/settings?step=connection", { sources: [source] });

  const local = await screen.findByRole("radio", { name: "Local sync" });
  expect(local).toBeChecked();
  expect(local).toHaveAttribute("aria-disabled", "true");
  expect(screen.getByRole("radio", { name: "Cloud" })).toHaveAttribute("aria-disabled", "true");
  const facts = screen.getByRole("region", { name: "Detected settings" });
  expect(within(facts).getByText("Sync Location")).toBeInTheDocument();
  expect(within(facts).getByText("Local sync")).toBeInTheDocument();
  expect(within(facts).getByText("Fixed when the source was created.")).toBeInTheDocument();
});

test("a blank secret keeps the stored token", async () => {
  const { calls } = renderSetup("/sources/src-wiki/settings?step=connection", {
    sources: [makeSource({ config: CONFLUENCE_CONFIG })],
    routes: { "PUT /api/v1/sources/src-wiki": () => ({ ok: true, id: "src-wiki" }) },
  });

  const token = await screen.findByLabelText("Personal Access Token");
  expect(token).toHaveValue("");
  expect(token).toHaveAccessibleDescription(expect.stringContaining("A value is stored. Leave this blank to keep it."));
  await userEvent.clear(screen.getByLabelText("Wiki URL"));
  await userEvent.type(screen.getByLabelText("Wiki URL"), "https://docs.example.com");
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

  const body = (await bodyOf(calls, "PUT", "/api/v1/sources/src-wiki")) as { config: Record<string, unknown> };
  expect(body.config).toEqual({ base_url: "https://docs.example.com", sync_mode: "space", spaces: ["PAY"] });
});

test("leaving with unsaved changes asks first", async () => {
  renderSetup("/sources/new/confluence");

  await userEvent.type(await screen.findByLabelText("Source name"), "Payroll wiki");
  await userEvent.click(screen.getByText("Cancel", { selector: "a" }));
  const dialog = await screen.findByRole("dialog", { name: "Leave without saving?" });
  await userEvent.click(within(dialog).getByRole("button", { name: "Keep editing" }));
  expect(screen.getByLabelText("Source name")).toHaveValue("Payroll wiki");

  await userEvent.click(screen.getByText("Cancel", { selector: "a" }));
  await userEvent.click(within(await screen.findByRole("dialog", { name: "Leave without saving?" })).getByRole("button", { name: "Leave without saving" }));
  expect(await screen.findByRole("status", { name: "Location" })).toHaveTextContent(/^\/sources$/);
});
