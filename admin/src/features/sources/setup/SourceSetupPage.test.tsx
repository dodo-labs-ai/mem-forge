import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLocation } from "react-router-dom";
import { afterEach, beforeAll, expect, test, vi } from "vitest";
import type { Source } from "@/api";
import { errorResponse, type Route } from "@/test/fakeApi";
import { renderRoutes } from "@/test/renderRoutes";
import { setupJobRoutes } from "@/test/setupJobs";
import { makeDevice, makeLocalSyncStatus, makeScopeTransition, makeSource } from "@/test/sourceFixtures";
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
    executor: "server",
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
    executor: "server",
  });
});

test("a new local folder opens its folder picker on the computer chosen in Where it runs", async () => {
  const created: Parameters<typeof setupJobRoutes>[1] = [];
  renderSetup("/sources/new/local_markdown?step=connection", {
    routes: {
      "POST /api/v1/genes/local_markdown/config-check": () => NO_ISSUES,
      ...setupJobRoutes({ local_markdown_pick_root: { status: "succeeded", result: { operation: "local_markdown_pick_root", root: "/Users/alex/notes" } } }, created),
      "GET /api/cloud/local-agent/status": () =>
        makeLocalSyncStatus([makeDevice(), makeDevice({ device_id: "dev-desktop", name: "Alex's desktop" })]),
    },
  });

  await userEvent.click(await screen.findByRole("radio", { name: /Alex's desktop/ }));
  await userEvent.click(await screen.findByRole("button", { name: "Choose folder" }));

  await waitFor(() => expect(screen.getByLabelText("Folder Path")).toHaveValue("/Users/alex/notes"));
  expect(created).toEqual([
    {
      source_id: "",
      source_type: "local_markdown",
      operation: "local_markdown_pick_root",
      payload: { title: "Choose folder to sync", initial_directory: null },
      device_id: "dev-desktop",
    },
  ]);
  expect(screen.queryByRole("button", { name: "Check setup" })).not.toBeInTheDocument();
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
  expect(await bodyOf(calls, "POST", "/api/v1/genes/confluence/config-check")).toEqual({ config: { base_url: pageUrl }, source_id: null, executor: "server" });
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

test("a radio change and a switch toggle check the configuration they commit", async () => {
  const { calls } = renderSetup("/sources/new/confluence?step=scope");
  const checks = () => requestsTo(calls, "POST", "/api/v1/genes/confluence/config-check");

  await userEvent.click(await screen.findByRole("radio", { name: "This page tree" }));
  await waitFor(() => expect(checks()).toHaveLength(1));
  expect(await checks()[0]!.clone().json()).toEqual({ config: { sync_mode: "page_tree" }, source_id: null, executor: "server" });

  await userEvent.click(screen.getByRole("switch", { name: "Include Child Pages" }));
  await waitFor(() => expect(checks()).toHaveLength(2));
  expect(await checks()[1]!.clone().json()).toEqual({ config: { sync_mode: "page_tree", include_children: false }, source_id: null, executor: "server" });
});

test("an untouched mode field is sent absent", async () => {
  const { calls } = renderSetup("/sources/new/jira", {
    routes: { "POST /api/v1/sources": () => ({ ...CREATED, type: "jira" }) },
  });

  await userEvent.type(await screen.findByLabelText("Source name"), "Payroll Jira");
  await next();
  expect(screen.getByRole("radio", { name: "Personal access token" })).toBeChecked();
  await userEvent.type(screen.getByLabelText("Jira Base URL"), "https://jira.example.com");
  await userEvent.type(screen.getByLabelText("Personal Access Token"), "jira-pat");
  await next();
  await userEvent.type(await screen.findByLabelText("Projects to Sync"), "PAY{Enter}");
  await userEvent.click(screen.getByRole("button", { name: "Create source" }));

  const body = (await bodyOf(calls, "POST", "/api/v1/sources")) as { config: Record<string, unknown> };
  expect(body.config).toEqual({ base_url: "https://jira.example.com", pat: "jira-pat", projects: ["PAY"] });
});

test("a new Jira source can choose a browser session instead of a token", async () => {
  const { calls } = renderSetup("/sources/new/jira", {
    routes: { "POST /api/v1/sources": () => ({ ...CREATED, type: "jira" }) },
  });

  await userEvent.type(await screen.findByLabelText("Source name"), "Payroll Jira");
  await next();
  await userEvent.click(screen.getByRole("radio", { name: "Browser session" }));
  expect(screen.queryByLabelText("Personal Access Token")).not.toBeInTheDocument();
  await userEvent.type(screen.getByLabelText("Jira Base URL"), "https://jira.example.com");
  await next();
  await userEvent.type(await screen.findByLabelText("Projects to Sync"), "PAY{Enter}");
  await userEvent.click(screen.getByRole("button", { name: "Create source" }));

  const body = (await bodyOf(calls, "POST", "/api/v1/sources")) as { config: Record<string, unknown> };
  expect(body.config).toEqual({ base_url: "https://jira.example.com", auth_mode: "browser_cookie", projects: ["PAY"] });
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
    executor: "server",
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

/** A Source that runs on the viewer's device, which no device report names yet. */
function deviceSource(credential: "secret" | "browser_session", overrides: Partial<Source> = {}): Source {
  return makeSource({
    config: { base_url: "https://wiki.example.com", sync_mode: "space", spaces: ["PAY"] },
    connection_status: { state: "action_required", reason: "authentication" },
    ownership: { ...makeSource().ownership, execution_owner_user_id: "local", viewer_is_execution_owner: true },
    execution: {
      kind: "local_agent",
      operation: "document_acquisition",
      device: { id: "dev-1", name: "Alex's MacBook", state: "online", not_ready: [], installed_version: "0.2.0b7", required_version: "0.2.0b7", credential },
    },
    ...overrides,
  });
}

/**
 * Config-check for a stored device Source, answering as the server does for
 * the executor the request names: on its device the server holds no token, so
 * a typed one is refused; on the server the token is required.
 */
async function deviceSourceConfigCheck(request: Request) {
  const { config, executor } = (await request.clone().json()) as { config: Record<string, unknown>; executor?: string };
  if ((executor ?? "device") === "device") {
    return config.pat ? { ...NO_ISSUES, issues: [{ key: "pat", message: "The token is stored on the device that runs this source, not on the server." }] } : NO_ISSUES;
  }
  return config.pat ? NO_ISSUES : { ...NO_ISSUES, issues: [{ key: "pat", message: "Personal Access Token is required." }] };
}

async function chooseMyComputer() {
  await userEvent.click(await screen.findByRole("radio", { name: /My computer/ }));
}

test("Where it runs on My computer creates the source without its secret once a device is ready", async () => {
  const { calls } = renderSetup("/sources/new/confluence", {
    routes: { "POST /api/v1/sources": () => CREATED, "GET /api/cloud/local-agent/status": () => makeLocalSyncStatus() },
  });

  await userEvent.type(await screen.findByLabelText("Source name"), "Team wiki");
  await next();
  expect(screen.getByRole("radio", { name: /Server/ })).toBeChecked();
  await chooseMyComputer();
  expect(screen.queryByLabelText("Personal Access Token")).not.toBeInTheDocument();
  // A single computer is chosen already.
  const computer = await screen.findByRole("radio", { name: /Alex's MacBook/ });
  expect(computer).toBeChecked();
  expect(within(screen.getByRole("region", { name: "Your computers" })).getByText(/Online · MemForge 0.2.0b7/)).toBeInTheDocument();
  await userEvent.type(screen.getByLabelText("Wiki URL"), "https://wiki.example.com");
  await next();
  await userEvent.type(await screen.findByLabelText("Spaces to Sync"), "PAY{Enter}");
  await userEvent.click(screen.getByRole("button", { name: "Create source" }));

  expect(await bodyOf(calls, "POST", "/api/v1/sources")).toMatchObject({
    executor: "device",
    execution_device_id: "dev-1",
    config: { base_url: "https://wiki.example.com", spaces: ["PAY"] },
  });
  const created = (await bodyOf(calls, "POST", "/api/v1/sources")) as { config: Record<string, unknown> };
  expect(created.config).not.toHaveProperty("pat");
  expect(requestsTo(calls, "POST", "/api/v1/genes/confluence/preview-discovery")).toHaveLength(0);
});

test.each([
  ["no device", [], "Set up MemForge on your computer", "memforge setup"],
  ["an offline device chosen", [makeDevice({ status: "offline" })], "Start MemForge on that computer", "memforge daemon start"],
  ["a device that needs an update chosen", [makeDevice({ acquisition_contracts: { confluence: 1 } })], "Update MemForge on that computer", "memforge upgrade"],
])("Where it runs disables save with %s and shows the step that makes it ready", async (_, devices, step, command) => {
  renderSetup("/sources/new/confluence?step=connection", {
    routes: { "GET /api/cloud/local-agent/status": () => makeLocalSyncStatus(devices) },
  });

  await chooseMyComputer();
  expect(await screen.findByText(step)).toBeInTheDocument();
  expect(screen.getByText(command)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Create source" })).toBeDisabled();
  await userEvent.click(screen.getByRole("radio", { name: /Server/ }));
  expect(screen.getByRole("button", { name: "Create source" })).toBeEnabled();
  expect(screen.getByLabelText("Personal Access Token")).toBeInTheDocument();
});

test("a server source moves to My computer through the executor change", async () => {
  const source = makeSource({ config: CONFLUENCE_CONFIG });
  const { calls } = renderSetup("/sources/src-wiki/settings?step=connection", {
    sources: [source],
    routes: {
      "GET /api/cloud/local-agent/status": () => makeLocalSyncStatus(),
      "PUT /api/v1/sources/src-wiki/executor": () => ({ executor: "device", cancelled_jobs: 0 }),
    },
  });

  await chooseMyComputer();
  expect(await screen.findByRole("radio", { name: /Alex's MacBook/ })).toBeChecked();
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

  expect(await bodyOf(calls, "PUT", "/api/v1/sources/src-wiki/executor")).toEqual({ executor: "device", execution_device_id: "dev-1" });
  expect(requestsTo(calls, "PUT", "/api/v1/sources/src-wiki")).toHaveLength(0);
});

test("a device source moves to the server with the secret the form requires", async () => {
  const source = deviceSource("secret");
  const { calls } = renderSetup("/sources/src-wiki/settings?step=connection", {
    sources: [source],
    routes: {
      "POST /api/v1/genes/confluence/config-check": deviceSourceConfigCheck,
      "PUT /api/v1/sources/src-wiki/executor": () => ({ executor: "server", cancelled_jobs: 0 }),
    },
  });

  await userEvent.click(await screen.findByRole("radio", { name: /Server/ }));
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
  expect(await screen.findByText("Personal Access Token is required.")).toBeInTheDocument();
  expect(requestsTo(calls, "PUT", "/api/v1/sources/src-wiki/executor")).toHaveLength(0);

  await userEvent.type(screen.getByLabelText("Personal Access Token"), "token-2");
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

  expect(await bodyOf(calls, "POST", "/api/v1/genes/confluence/config-check")).toMatchObject({ source_id: "src-wiki", executor: "server" });
  expect(await bodyOf(calls, "PUT", "/api/v1/sources/src-wiki/executor")).toEqual({ executor: "server", secrets: { pat: "token-2" } });
});

test("a device source moving to the server checks its setup through the server with the typed token", async () => {
  const { calls } = renderSetup("/sources/src-wiki/settings?step=connection", {
    sources: [deviceSource("secret")],
    routes: {
      "POST /api/v1/genes/confluence/config-check": deviceSourceConfigCheck,
      "POST /api/v1/genes/confluence/preview-discovery": () => ({ source_type: "confluence", count: 0, items: [], truncated: false }),
    },
  });

  await userEvent.click(await screen.findByRole("radio", { name: /Server/ }));
  await userEvent.type(screen.getByLabelText("Personal Access Token"), "token-2");
  await next();
  await userEvent.click(await screen.findByRole("button", { name: "Check setup" }));

  expect(await bodyOf(calls, "POST", "/api/v1/genes/confluence/preview-discovery")).toMatchObject({
    source_id: "src-wiki",
    executor: "server",
    config: { pat: "token-2" },
  });
});

test("a device source's token is stored on its computer with a copyable command, never in the form", async () => {
  renderSetup("/sources/src-wiki/settings?step=connection", { sources: [deviceSource("secret")] });

  const step = await screen.findByRole("region", { name: "Your computer" });
  expect(within(step).getByText("Sign in required")).toBeInTheDocument();
  expect(within(step).getByText("memforge source credentials set src-wiki")).toBeInTheDocument();
  expect(within(step).getByRole("button", { name: "Copy memforge source credentials set src-wiki" })).toBeInTheDocument();
  expect(screen.queryByLabelText("Personal Access Token")).not.toBeInTheDocument();
});

test("a device source with a browser session signs in through a credential check on its computer", async () => {
  const created: Parameters<typeof setupJobRoutes>[1] = [];
  const source = deviceSource("browser_session", {
    type: "jira",
    config: { base_url: "https://jira.example.com", sync_mode: "cloud", auth_mode: "browser_cookie", projects: ["ACME"] },
  });
  renderSetup("/sources/src-wiki/settings?step=connection", {
    sources: [source],
    routes: setupJobRoutes({ credential_check: { status: "succeeded", result: { ok: true } } }, created),
  });

  const step = await screen.findByRole("region", { name: "Your computer" });
  expect(screen.queryByRole("region", { name: "Jira browser session" })).not.toBeInTheDocument();
  await userEvent.click(within(step).getByRole("button", { name: "Sign in" }));
  await waitFor(() => expect(created).toHaveLength(1));
  expect(created[0]).toMatchObject({ operation: "credential_check", source_id: "src-wiki", device_id: "dev-1" });
});

test("a choice of two computers waits for the person to choose one", async () => {
  renderSetup("/sources/new/confluence?step=connection", {
    routes: {
      "GET /api/cloud/local-agent/status": () =>
        makeLocalSyncStatus([makeDevice(), makeDevice({ device_id: "dev-desktop", name: "Alex's desktop" })]),
    },
  });

  await chooseMyComputer();
  expect(await screen.findByText("Choose the computer that runs this source")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Create source" })).toBeDisabled();
  await userEvent.click(screen.getByRole("radio", { name: /Alex's desktop/ }));
  expect(screen.queryByText("Choose the computer that runs this source")).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Create source" })).toBeEnabled();
});

test("a device source whose computer is offline moves to another of the person's computers", async () => {
  const source = deviceSource("secret", {
    connection_status: { state: "action_required", reason: "authentication" },
  });
  source.execution!.device!.state = "offline";
  const { calls } = renderSetup("/sources/src-wiki/settings?step=connection", {
    sources: [source],
    routes: {
      "POST /api/v1/genes/confluence/config-check": deviceSourceConfigCheck,
      "GET /api/cloud/local-agent/status": () =>
        makeLocalSyncStatus([makeDevice({ status: "offline" }), makeDevice({ device_id: "dev-desktop", name: "Alex's desktop" })]),
      "PUT /api/v1/sources/src-wiki/executor": () => ({ executor: "device", cancelled_jobs: 0 }),
    },
  });

  const step = await screen.findByRole("region", { name: "Your computer" });
  expect(within(step).getByText("Alex's MacBook is offline.")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
  await userEvent.click(within(step).getByRole("button", { name: "Move" }));
  // The current computer is chosen, and offline: save waits until another is chosen.
  expect(await screen.findByRole("radio", { name: /Alex's MacBook/ })).toBeChecked();
  expect(screen.getByText("Start MemForge on that computer")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("radio", { name: /Alex's desktop/ }));
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

  expect(await bodyOf(calls, "PUT", "/api/v1/sources/src-wiki/executor")).toEqual({
    executor: "device",
    execution_device_id: "dev-desktop",
  });
});

test("a device source whose computer is online changes to another of the person's computers", async () => {
  const source = deviceSource("secret", { connection_status: { state: "ready", reason: null } });
  const { calls } = renderSetup("/sources/src-wiki/settings?step=connection", {
    sources: [source],
    routes: {
      "POST /api/v1/genes/confluence/config-check": deviceSourceConfigCheck,
      "GET /api/cloud/local-agent/status": () =>
        makeLocalSyncStatus([makeDevice(), makeDevice({ device_id: "dev-desktop", name: "Alex's desktop" })]),
      "PUT /api/v1/sources/src-wiki/executor": () => ({ executor: "device", cancelled_jobs: 0 }),
    },
  });

  const step = await screen.findByRole("region", { name: "Your computer" });
  expect(within(step).queryByRole("button", { name: "Move" })).not.toBeInTheDocument();
  await userEvent.click(within(step).getByRole("button", { name: "Change computer" }));
  expect(await screen.findByRole("radio", { name: /Alex's MacBook/ })).toBeChecked();
  await userEvent.click(screen.getByRole("radio", { name: /Alex's desktop/ }));
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

  expect(await bodyOf(calls, "PUT", "/api/v1/sources/src-wiki/executor")).toEqual({
    executor: "device",
    execution_device_id: "dev-desktop",
  });
});

test("another member takes over a device source by choosing their own computer", async () => {
  const source = deviceSource("secret", {
    connection_status: { state: "ready", reason: null },
    ownership: { ...makeSource().ownership, execution_owner_user_id: "alex", viewer_is_execution_owner: false },
  });
  source.execution!.device!.name = null;
  const { calls } = renderSetup("/sources/src-wiki/settings?step=connection", {
    sources: [source],
    routes: {
      "POST /api/v1/genes/confluence/config-check": deviceSourceConfigCheck,
      "GET /api/cloud/local-agent/status": () => makeLocalSyncStatus([makeDevice({ device_id: "dev-sam", name: "Sam's laptop" })]),
      "PUT /api/v1/sources/src-wiki/executor": () => ({ executor: "device", cancelled_jobs: 0 }),
    },
  });

  const step = await screen.findByRole("region", { name: "Your computer" });
  expect(step).toHaveTextContent("Runs on its owner's computer");
  await userEvent.click(within(step).getByRole("button", { name: "Change computer" }));
  // Their single computer is chosen already.
  expect(await screen.findByRole("radio", { name: /Sam's laptop/ })).toBeChecked();
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

  expect(await bodyOf(calls, "PUT", "/api/v1/sources/src-wiki/executor")).toEqual({ executor: "device", execution_device_id: "dev-sam" });
});

test("a device source without a computer shows Finish setup by choosing one", async () => {
  const source = deviceSource("secret", { connection_status: { state: "action_required", reason: "configuration" } });
  source.execution!.device = null;
  const { calls } = renderSetup("/sources/src-wiki/settings?step=connection", {
    sources: [source],
    routes: {
      "POST /api/v1/genes/confluence/config-check": deviceSourceConfigCheck,
      "GET /api/cloud/local-agent/status": () => makeLocalSyncStatus(),
      "PUT /api/v1/sources/src-wiki/executor": () => ({ executor: "device", cancelled_jobs: 0 }),
    },
  });

  expect(await screen.findByRole("radio", { name: /My computer/ })).toBeChecked();
  expect(await screen.findByRole("radio", { name: /Alex's MacBook/ })).toBeChecked();
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

  expect(await bodyOf(calls, "PUT", "/api/v1/sources/src-wiki/executor")).toEqual({ executor: "device", execution_device_id: "dev-1" });
});

test("a device source whose folder is missing shows Finish setup with the folder and the computer", async () => {
  const source = makeSource({
    id: "src-notes",
    type: "local_markdown",
    config: { root: "/Users/alex/notes" },
    connection_status: { state: "action_required", reason: "configuration" },
    ownership: { ...makeSource().ownership, execution_owner_user_id: "local", viewer_is_execution_owner: true },
    execution: {
      kind: "local_agent",
      operation: "document_acquisition",
      device: {
        id: "dev-1",
        name: "Alex's MacBook",
        state: "online",
        not_ready: [{ kind: "location", field: "root", path: "/Users/alex/notes" }],
        installed_version: "0.2.0b7",
        required_version: "0.2.0b7",
        credential: "secret",
      },
    },
  });
  renderSetup("/sources/src-notes/settings?step=connection", {
    sources: [source],
    routes: { "POST /api/v1/genes/local_markdown/config-check": () => NO_ISSUES },
  });

  const step = await screen.findByRole("region", { name: "Your computer" });
  expect(within(step).getByText("Finish setup")).toBeInTheDocument();
  expect(step).toHaveTextContent("The folder /Users/alex/notes is missing on Alex's MacBook.");
});

test("an offline computer the person no longer uses is removed after they confirm", async () => {
  const { calls } = renderSetup("/sources/new/confluence?step=connection", {
    routes: {
      "GET /api/cloud/local-agent/status": () =>
        makeLocalSyncStatus([makeDevice(), makeDevice({ device_id: "dev-old", name: "Old laptop", status: "offline" })]),
      "DELETE /api/cloud/local-agent/devices/dev-old": () => ({ device_id: "dev-old", source_ids: ["src-wiki"] }),
    },
  });

  await chooseMyComputer();
  await userEvent.click(await screen.findByRole("button", { name: "Remove Old laptop" }));
  expect(screen.getByText(/Sources it runs show Finish setup/)).toBeInTheDocument();
  expect(requestsTo(calls, "DELETE", "/api/cloud/local-agent/devices/dev-old")).toHaveLength(0);
  await userEvent.click(screen.getByRole("button", { name: "Remove" }));

  await waitFor(() => expect(requestsTo(calls, "DELETE", "/api/cloud/local-agent/devices/dev-old")).toHaveLength(1));
});
