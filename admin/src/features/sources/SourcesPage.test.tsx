import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLocation } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { toast } from "sonner";
import { STANDALONE_TARGET, type WorkspaceTarget } from "@/api";
import { errorResponse, type Route } from "@/test/fakeApi";
import { makeOverview, makeSourceHealth } from "@/test/evaluationFixtures";
import { renderRoutes } from "@/test/renderRoutes";
import { makeFailedItem, makeLocalAgentJob, makeScopeTransition, makeSource } from "@/test/sourceFixtures";
import { sourceTypeSchema } from "@/test/sourceTypeForms";
import { SourcesPage } from "./SourcesPage";

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

const RETRY_DELAY_MS = 60 * 60_000;

function CurrentLocation() {
  const location = useLocation();
  return <output aria-label="Location">{`${location.pathname}${location.search}`}</output>;
}

function renderPage(
  sources = [makeSource(), makeSource({ id: "src-jira", type: "jira", name: "Payroll Jira", project_binding: null })],
  routes: Record<string, Route> = {},
  workspace?: WorkspaceTarget,
) {
  const api: Record<string, Route> = {
    "GET /api/v1/sources": () => ({ data: sources }),
    "GET /api/v1/projects": () => ({ data: [{ id: "p1", key: "PAY", name: "Payroll", kind: "normal" }], can_manage: true }),
    "GET /api/v1/genes": () => [
      { name: "confluence", display_name: "Confluence" },
      { name: "jira", display_name: "Jira" },
    ],
    "GET /api/v1/source-list/preferences": () => ({ sort_mode: "name" }),
    "GET /api/cloud/local-agent/status": () => ({ status: "online", last_seen_at: null, checked_at: "", stale_after_seconds: 60 }),
    "GET /api/cloud/local-agent/jobs/current": () => ({ data: [] }),
    "GET /api/v1/agent-evaluations/online-overview": () => makeOverview({ sources: [] }),
    "POST /api/v1/sources/src-wiki/sync": () => ({ status: "pending", run_id: "run-2" }),
    ...routes,
  };
  return renderRoutes([{ path: "/sources", element: <SourcesPage /> }, { path: "*", element: <CurrentLocation /> }], {
    path: "/sources",
    api,
    workspace,
  });
}

test("lists sources by project and surfaces the ones that need the user", async () => {
  renderPage();

  const table = await screen.findByRole("table", { name: "Sources" });
  expect(await within(table).findByText("Payroll wiki")).toBeInTheDocument();
  expect(within(table).getByText("Payroll")).toBeInTheDocument();
  expect(within(table).getByText("No project")).toBeInTheDocument();
  expect(within(table).getByText("Payroll Jira")).toBeInTheDocument();

  const needsYou = screen.getByRole("region", { name: "Needs you" });
  expect(within(needsYou).getByText("Payroll Jira")).toBeInTheDocument();
  expect(within(needsYou).getByRole("button", { name: "Choose a project" })).toBeInTheDocument();
});

test("a workspace of one leaves out who can use each source", async () => {
  renderPage([makeSource({ access_policy: "private" })]);

  const table = await screen.findByRole("table", { name: "Sources" });
  expect(within(table).queryByRole("columnheader", { name: "Access" })).not.toBeInTheDocument();
  expect(within(table).queryByText("Only me")).not.toBeInTheDocument();
  await userEvent.click(await within(table).findByText("Payroll wiki"));
  const drawer = await screen.findByRole("dialog", { name: "Payroll wiki" });
  expect(within(drawer).queryByText("Access")).not.toBeInTheDocument();
});

test("a shared workspace shows who can use each source", async () => {
  renderPage([makeSource({ access_policy: "private" })], {}, { ...STANDALONE_TARGET, shared: true });

  const table = await screen.findByRole("table", { name: "Sources" });
  expect(within(table).getByRole("columnheader", { name: "Access" })).toBeInTheDocument();
  await userEvent.click(await within(table).findByText("Payroll wiki"));
  const drawer = await screen.findByRole("dialog", { name: "Payroll wiki" });
  expect(within(drawer).getByText("Access")).toBeInTheDocument();
  expect(within(drawer).getByText("Only me")).toBeInTheDocument();
});

test.each([
  [{ count: 3, plugin_version: "0.2.0", last_seen_at: "2026-09-29T11:00:00Z" }, true],
  [null, false],
])("an agent-session source warns about session lines its plugin did not recognize (%o)", async (lines, warns) => {
  const source = makeSource({
    id: "src-claude",
    type: "agent_session",
    client: "claude-code",
    name: "Claude Code Session",
    unrecognized_session_lines: lines,
  });
  renderPage([source], {
    "GET /api/v1/agent-sessions/completeness": () => ({
      session_id: null,
      source_id: "src-claude",
      total: 0,
      processed_total: 0,
      counts: { knowledge_patched: 0, no_output: 0, failed: 0 },
      no_output_fraction: 0,
      latest_failure: null,
    }),
    "GET /api/v1/sources/src-claude/projects": () => ({ source_id: "src-claude", projects: [] }),
  });

  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByText("Claude Code Session"));
  const drawer = await screen.findByRole("dialog", { name: "Claude Code Session" });
  const warning = within(drawer).queryByText("3 session lines from Claude Code were not recognized");
  if (warns) {
    expect(warning).toBeInTheDocument();
    expect(within(drawer).getByText("Update the MemForge plugin so new sessions are read in full.")).toBeInTheDocument();
  } else {
    expect(warning).not.toBeInTheDocument();
    expect(within(drawer).queryByText(/not recognized/)).not.toBeInTheDocument();
  }
});

test("an agent-session source's drawer shows when a session window was last processed, not a sync status it cannot have", async () => {
  const source = makeSource({
    id: "src-claude",
    type: "agent_session",
    client: "claude-code",
    name: "Claude Code Session",
    sync: null,
    sync_schedule: null,
    last_sync: "2026-10-24T05:00:00Z",
  });
  renderPage([source], {
    "GET /api/v1/agent-sessions/completeness": () => ({
      session_id: null,
      source_id: "src-claude",
      total: 0,
      processed_total: 0,
      counts: { knowledge_patched: 0, no_output: 0, failed: 0 },
      no_output_fraction: 0,
      latest_failure: null,
    }),
    "GET /api/v1/sources/src-claude/projects": () => ({ source_id: "src-claude", projects: [] }),
  });

  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByText("Claude Code Session"));
  const drawer = await screen.findByRole("dialog", { name: "Claude Code Session" });
  expect(await within(drawer).findByText("Last window processed")).toBeInTheDocument();
  expect(within(drawer).queryByRole("heading", { name: "Sync" })).not.toBeInTheDocument();
  for (const text of ["Not synced yet", "Status", "Last sync", "Schedule", "Manual only"]) {
    expect(within(drawer).queryByText(text)).not.toBeInTheDocument();
  }
});

test("a synced source's drawer keeps its sync status, last sync and schedule", async () => {
  renderPage([makeSource()]);

  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByText("Payroll wiki"));
  const drawer = await screen.findByRole("dialog", { name: "Payroll wiki" });
  expect(within(drawer).getByRole("heading", { name: "Sync" })).toBeInTheDocument();
  for (const text of ["Status", "Last sync", "Schedule"]) {
    expect(within(drawer).getByText(text)).toBeInTheDocument();
  }
  expect(within(drawer).queryByText("Last window processed")).not.toBeInTheDocument();
});

test("Add source opens the type picker in the app", async () => {
  renderPage();
  await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(screen.getByRole("button", { name: "Add source" }));
  expect(await screen.findByRole("status", { name: "Location" })).toHaveTextContent(/^\/sources\/new$/);
});

const REPO_OVER_LIMIT = makeSource({
  id: "src-repo",
  type: "github_repo",
  name: "Payroll repo",
  sync: {
    status: "failed",
    run_id: "run-9",
    started_at: "2026-09-29T10:58:00Z",
    finished_at: "2026-09-29T11:00:00Z",
    error_message: "GitHub Repository discovery matched 600 files, exceeding max_files=500",
    progress: null,
  },
});

test.each([
  ["Configure in the menu", undefined, "/sources/src-wiki/settings"],
  ["Choose a project", undefined, "/sources/src-jira/settings?step=project"],
  ["Configure", [makeSource({ connection_status: { state: "action_required", reason: "configuration" } })], "/sources/src-wiki/settings?step=connection"],
  ["Configure file scope", [REPO_OVER_LIMIT], "/sources/src-repo/settings?step=scope"],
])("%s opens the source's settings in the app", async (entry, sources, target) => {
  renderPage(sources, { "GET /api/v1/genes/github_repo/config-schema": () => sourceTypeSchema("github_repo") });

  const table = await screen.findByRole("table", { name: "Sources" });
  if (entry === "Configure in the menu") {
    await userEvent.click(await within(table).findByRole("button", { name: "More actions for Payroll wiki" }));
    await userEvent.click(await screen.findByRole("menuitem", { name: /^Configure$/ }));
  } else {
    await userEvent.click(within(await screen.findByRole("region", { name: "Needs you" })).getByRole("button", { name: entry }));
  }

  expect(await screen.findByRole("status", { name: "Location" })).toHaveTextContent(target);
});

test("Sync now starts a sync for that source", async () => {
  const { calls } = renderPage([makeSource()]);
  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByRole("button", { name: "Sync now" }));

  await vi.waitFor(() => expect(calls.some((request) => request.method === "POST")).toBe(true));
  const syncCall = calls.find((request) => request.method === "POST")!;
  expect(new URL(syncCall.url).pathname).toBe("/api/v1/sources/src-wiki/sync");
  expect(await syncCall.clone().json()).toEqual({ dry_run: false, force_full_sync: false });
});

test.each(["row", "card", "drawer"])("Jira Sign in from the %s starts authentication without leaving V2", async (entry) => {
  const source = makeSource({
    id: "src-jira",
    type: "jira",
    name: "Payroll Jira",
    config: { base_url: "https://jira.example.invalid", auth_mode: "browser_cookie" },
    connection_status: { state: "action_required", reason: "authentication" },
  });
  const { calls, router } = renderPage([source], {
    "POST /api/cloud/local-agent/jobs": () => ({ job_id: "laj-auth", status: "queued" }),
    "GET /api/cloud/local-agent/jobs/laj-auth": () => ({ job_id: "laj-auth", status: "succeeded", result: {} }),
  });
  const table = await screen.findByRole("table", { name: "Sources" });
  if (entry === "drawer") await userEvent.click(await within(table).findByText("Payroll Jira"));
  const surface = entry === "row" ? table : entry === "card"
    ? await screen.findByRole("region", { name: "Needs you" })
    : await screen.findByRole("dialog", { name: "Payroll Jira" });
  await userEvent.click(await within(surface).findByRole("button", { name: "Sign in" }));

  await vi.waitFor(() => expect(calls.some((request) => request.method === "POST")).toBe(true));
  const authCall = calls.find((request) => request.method === "POST")!;
  expect(new URL(authCall.url).pathname).toBe("/api/cloud/local-agent/jobs");
  expect(await authCall.clone().json()).toMatchObject({
    source_id: "src-jira",
    source_type: "jira",
    operation: "jira_auth",
    payload: { base_url: "https://jira.example.invalid", auth_mode: "browser_cookie" },
  });
  expect(router.state.location.pathname).toBe("/sources");
  await vi.waitFor(() => expect(toast.success).toHaveBeenCalledWith("Signed in to Payroll Jira"));
});

function expiredJira() {
  return makeSource({
    id: "src-jira", type: "jira", name: "Payroll Jira",
    config: { base_url: "https://jira.example.invalid", auth_mode: "browser_cookie" },
    connection_status: { state: "action_required", reason: "authentication" },
  });
}

test("waits for durable Jira authentication and refreshes readiness in the selected Cloud workspace", async () => {
  let complete!: (value: unknown) => void;
  let signedIn = false;
  const { calls } = renderPage([], {
    "GET /api/cloud/workspaces/team_alpha/v1/sources": () => ({ data: [signedIn
      ? { ...expiredJira(), connection_status: { state: "ready", reason: null } } : expiredJira()] }),
    "GET /api/cloud/workspaces/team_alpha/v1/projects": () => ({ data: [], can_manage: true }),
    "GET /api/cloud/workspaces/team_alpha/v1/genes": () => [{ name: "jira", display_name: "Jira" }],
    "GET /api/cloud/workspaces/team_alpha/v1/source-list/preferences": () => ({ sort_mode: "name" }),
    "GET /api/cloud/workspaces/team_alpha/local-agent/status": () => ({ status: "online" }),
    "GET /api/cloud/workspaces/team_alpha/local-agent/jobs/current": () => ({ data: [] }),
    "POST /api/cloud/workspaces/team_alpha/local-agent/jobs": () => ({ job_id: "laj-auth", status: "queued" }),
    "GET /api/cloud/local-agent/jobs/laj-auth": () => new Promise((resolve) => { complete = resolve; }),
  }, {
    resourceBaseUrl: "/api/cloud/workspaces/team_alpha/v1",
    localAgentBaseUrl: "/api/cloud/workspaces/team_alpha/local-agent",
    workspaceId: "team_alpha",
    shared: true,
  });
  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByRole("button", { name: "Sign in" }));
  expect(await within(table).findByRole("button", { name: "Waiting for sign-in…" })).toBeDisabled();
  expect(screen.getAllByRole("button", { name: "Waiting for sign-in…" }).every((button) => button.hasAttribute("disabled"))).toBe(true);
  expect(toast.success).not.toHaveBeenCalled();
  await vi.waitFor(() => expect(complete).toBeTypeOf("function"));
  signedIn = true;
  complete({ job_id: "laj-auth", status: "succeeded", result: {} });
  await vi.waitFor(() => expect(toast.success).toHaveBeenCalledWith("Signed in to Payroll Jira"));
  await within(table).findByRole("button", { name: "Sync now" });
  expect(calls.filter((request) => request.method === "POST")).toHaveLength(1);
});

test.each(["daemon", "job", "permission", "timeout"])("reports %s failure without claiming Jira sign-in succeeded", async (failure) => {
  const { calls } = renderPage([expiredJira()], {
    "GET /api/cloud/local-agent/status": () => ({ status: failure === "daemon" ? "offline" : "online" }),
    "POST /api/cloud/local-agent/jobs": () => failure === "permission"
      ? errorResponse(403, "local_agent_job_forbidden") : { job_id: "laj-auth", status: "queued" },
    "GET /api/cloud/local-agent/jobs/laj-auth": () => {
      if (failure === "timeout") throw new DOMException("Timed out", "TimeoutError");
      return { job_id: "laj-auth", status: "failed", result: { error: "principal_changed" } };
    },
  });
  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByRole("button", { name: "Sign in" }));
  await vi.waitFor(() => expect(toast.error).toHaveBeenCalledWith("Jira sign-in failed", {
    description: failure === "daemon" ? "Start local sync on your computer, then try signing in again."
      : failure === "timeout" ? "Still waiting for Jira sign-in. Check the browser on your computer, then try again."
      : failure === "job" ? "Signed in with a different Jira account. Use the account this source was set up with, then try again."
        : "You do not have permission to renew this connection. Ask a workspace admin for help.",
  }));
  expect(toast.success).not.toHaveBeenCalled();
  if (failure === "daemon") expect(calls.some((request) => request.method === "POST")).toBe(false);
  expect(within(table).getByRole("button", { name: "Sign in" })).toBeEnabled();
});

test("retrying a failed status request resumes the admitted sign-in job", async () => {
  let statusReads = 0;
  const { calls } = renderPage([expiredJira()], {
    "POST /api/cloud/local-agent/jobs": () => ({ job_id: "laj-auth", status: "queued" }),
    "GET /api/cloud/local-agent/jobs/laj-auth": () => ++statusReads === 1
      ? errorResponse(503, "Service unavailable") : { job_id: "laj-auth", status: "succeeded", result: {} },
  });
  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByRole("button", { name: "Sign in" }));
  await vi.waitFor(() => expect(toast.error).toHaveBeenCalled());
  await userEvent.click(within(table).getByRole("button", { name: "Sign in" }));
  await vi.waitFor(() => expect(toast.success).toHaveBeenCalled());
  expect(calls.filter((request) => request.method === "POST")).toHaveLength(1);
  expect(statusReads).toBe(2);
});

function unmappedSource(index: number) {
  return makeSource({ id: `src-jira-${index}`, type: "jira", name: `Jira ${index}`, project_binding: null });
}

test("View all shows every source that needs the user when the cards cannot", async () => {
  const scrollIntoView = vi.fn();
  Element.prototype.scrollIntoView = scrollIntoView;
  renderPage([makeSource(), ...[1, 2, 3, 4].map(unmappedSource)]);

  const needsYou = await screen.findByRole("region", { name: "Needs you" });
  expect(within(needsYou).getAllByRole("button", { name: "Choose a project" })).toHaveLength(3);
  await userEvent.type(screen.getByRole("searchbox", { name: "Search sources" }), "Payroll");
  await userEvent.click(within(needsYou).getByRole("button", { name: "View all 4" }));

  const table = screen.getByRole("table", { name: "Sources" });
  expect(screen.getByRole("searchbox", { name: "Search sources" })).toHaveValue("");
  for (const index of [1, 2, 3, 4]) expect(within(table).getByText(`Jira ${index}`)).toBeInTheDocument();
  expect(within(table).queryByText("Payroll wiki")).not.toBeInTheDocument();
  expect(scrollIntoView).toHaveBeenCalled();
});

test("does not offer View all when every source that needs the user has a card", async () => {
  renderPage([makeSource(), ...[1, 2, 3].map(unmappedSource)]);
  const needsYou = await screen.findByRole("region", { name: "Needs you" });
  expect(within(needsYou).queryByRole("button", { name: /View all/ })).not.toBeInTheDocument();
});

test("Retry now starts the waiting server run instead of a new sync", async () => {
  const waiting = makeSource({
    sync: {
      status: "pending",
      run_id: "ssr-waiting",
      started_at: null,
      finished_at: null,
      error_message: "Rate limit exceeded",
      progress: null,
      next_attempt_at: new Date(Date.now() + RETRY_DELAY_MS).toISOString(),
    },
  });
  const { calls } = renderPage([waiting]);
  const table = await screen.findByRole("table", { name: "Sources" });
  expect(await within(table).findByText("Waiting to retry")).toBeInTheDocument();
  await userEvent.click(within(table).getByRole("button", { name: "Retry now" }));

  await vi.waitFor(() => expect(calls.some((request) => request.method === "POST")).toBe(true));
  const retryCall = calls.find((request) => request.method === "POST")!;
  expect(new URL(retryCall.url).pathname).toBe("/api/v1/sources/src-wiki/sync");
  expect(await retryCall.clone().json()).toEqual({
    dry_run: false,
    force_full_sync: false,
    retry_target: { execution_kind: "source_sync_run", execution_id: "ssr-waiting" },
  });
});

test("Retry now starts the waiting local sync job on the user's device", async () => {
  const teams = makeSource({
    id: "src-teams",
    type: "teams",
    name: "Team chats",
    sync: null,
    execution: { kind: "local_agent", operation: "teams_sync", immutable_config_fields: [] },
  });
  const job = makeLocalAgentJob({
    job_id: "laj-waiting",
    status: "queued",
    next_attempt_at: new Date(Date.now() + RETRY_DELAY_MS).toISOString(),
  });
  const { calls } = renderPage([teams], {
    "GET /api/cloud/local-agent/jobs/current": () => ({ data: [job] }),
    "POST /api/cloud/local-agent/jobs": () => ({ job_id: "laj-waiting", status: "queued", coalesced: true }),
  });
  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByRole("button", { name: "Retry now" }));

  await vi.waitFor(() => expect(calls.some((request) => request.method === "POST")).toBe(true));
  const retryCall = calls.find((request) => request.method === "POST")!;
  expect(new URL(retryCall.url).pathname).toBe("/api/cloud/local-agent/jobs");
  expect(await retryCall.clone().json()).toMatchObject({ source_id: "src-teams", retry_job_id: "laj-waiting" });
});

test("a queued sync shows as queued, not as syncing", async () => {
  const queued = makeSource({
    sync: { status: "pending", run_id: "ssr-queued", started_at: null, finished_at: null, error_message: null, progress: null },
  });
  renderPage([queued]);
  const table = await screen.findByRole("table", { name: "Sources" });
  expect(await within(table).findByRole("button", { name: "Sync queued" })).toBeDisabled();
});

test("Pipeline checks opens the source on the Evaluation page", async () => {
  const { router } = renderPage([makeSource()]);
  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByRole("button", { name: "More actions for Payroll wiki" }));
  await userEvent.click(await screen.findByRole("menuitem", { name: /Pipeline checks/ }));

  expect(router.state.location.pathname).toBe("/evaluation");
  expect(router.state.location.search).toBe("?source_id=src-wiki");
});

test("the drawer lists each failed item with its step and reason, and Sync is its retry", async () => {
  const source = makeSource({
    sync: {
      status: "partial", run_id: "run-1", started_at: null, finished_at: "2026-09-29T11:00:00Z", error_message: null, progress: null,
    },
    failed_items: {
      count: 2,
      retrying_after_upgrade: 0,
      items: [
        makeFailedItem(),
        makeFailedItem({ document_id: "doc-2", title: "Leave policy", stage: "derivation", message: "Model output was invalid" }),
      ],
    },
  });
  renderPage([source]);

  const needsYou = await screen.findByRole("region", { name: "Needs you" });
  expect(within(needsYou).getByText("Some items failed (2)")).toBeInTheDocument();
  await userEvent.click(within(needsYou).getByRole("button", { name: "View details" }));

  const drawer = await screen.findByRole("dialog", { name: "Payroll wiki" });
  const section = within(drawer).getByRole("heading", { name: "Some items failed (2)" }).closest("section")!;
  expect(within(section).getByText("Payroll FAQ")).toBeInTheDocument();
  expect(within(section).getByText("Reading:")).toBeInTheDocument();
  expect(within(section).getByText("403 Forbidden")).toBeInTheDocument();
  expect(within(section).getByText("Creating memories:")).toBeInTheDocument();
  expect(within(section).getByText(/Syncing this source retries every failed item/)).toBeInTheDocument();
  expect(within(drawer).queryByRole("button", { name: /^Retry/ })).not.toBeInTheDocument();
  expect(within(drawer).getByRole("button", { name: "Sync now" })).toBeEnabled();
});

test("a sync after an upgrade shows that it retries the items an earlier version failed", async () => {
  const source = makeSource({
    sync: { status: "running", run_id: "ssr-release", started_at: "2026-09-29T11:05:00Z", finished_at: null, error_message: null, progress: null },
    failed_items: {
      count: 3,
      retrying_after_upgrade: 2,
      items: [
        makeFailedItem({ retrying_after_upgrade: true }),
        makeFailedItem({ document_id: "doc-2", title: "Leave policy", retrying_after_upgrade: true }),
        makeFailedItem({ document_id: "doc-3", title: "Holiday calendar", release: "1.5.0" }),
      ],
    },
  });
  renderPage([source]);

  const table = await screen.findByRole("table", { name: "Sources" });
  expect(await within(table).findByText("Retrying 2 items after upgrade")).toBeInTheDocument();
  expect(screen.queryByRole("region", { name: "Needs you" })).not.toBeInTheDocument();

  await userEvent.click(within(table).getByText("Payroll wiki"));
  const drawer = await screen.findByRole("dialog", { name: "Payroll wiki" });
  const notice = within(drawer).getByText("Retrying 2 items after upgrade").closest("[role=status]");
  expect(notice).toHaveTextContent("MemForge was upgraded");
  expect(within(drawer).getAllByText("Retrying")).toHaveLength(2);
});

function reprocessPosts(calls: Request[]) {
  return Promise.all(
    calls
      .filter((request) => request.method === "POST" && new URL(request.url).pathname.endsWith("/reprocess"))
      .map((request) => request.clone().json()),
  );
}

async function openReRead() {
  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByRole("button", { name: "More actions for Payroll wiki" }));
  await userEvent.click(await screen.findByRole("menuitem", { name: /Re-read all documents/ }));
}

const QUEUED = {
  ok: true,
  message: "Reprocess enqueued",
  source_id: "src-wiki",
  run_id: "ssr-1",
  status: "pending",
  created_at: "2026-10-20T09:00:00Z",
  document_ids: [],
};

test("Re-read all documents says how many documents and model calls it takes, then re-reads the ones it can", async () => {
  const { calls } = renderPage([makeSource()], {
    "POST /api/v1/sources/src-wiki/reprocess": async (request: Request) =>
      (await request.clone().json()).dry_run
        ? {
            source_id: "src-wiki",
            units: [
              { document_id: "doc-1", available: true },
              { document_id: "doc-2", available: true },
              { document_id: "doc-3", available: false, reason: "stored_content_empty" },
            ],
            available_unit_count: 2,
            unavailable_unit_count: 1,
            estimated_model_calls: 940,
          }
        : QUEUED,
  });
  await openReRead();

  const dialog = await screen.findByRole("dialog", { name: "Re-read all 2 documents?" });
  expect(dialog).toHaveTextContent("Every document in Payroll wiki is processed again with the current models.");
  expect(dialog).toHaveTextContent("about 940 model calls");
  expect(dialog).toHaveTextContent("1 document can't be re-read from what was stored and stay as they are.");
  await userEvent.click(within(dialog).getByRole("button", { name: "Re-read documents" }));

  await vi.waitFor(() => expect(toast.success).toHaveBeenCalledWith("Re-reading 2 documents in Payroll wiki"));
  expect(await reprocessPosts(calls)).toEqual([
    { all_documents: true, dry_run: true },
    { document_ids: ["doc-1", "doc-2"], all_documents: false, dry_run: false },
  ]);
});

test("Re-read all documents still re-reads every document when the documents cannot be counted", async () => {
  const { calls } = renderPage([makeSource()], {
    "POST /api/v1/sources/src-wiki/reprocess": async (request: Request) =>
      (await request.clone().json()).dry_run ? errorResponse(504, "Gateway timeout") : QUEUED,
  });
  await openReRead();

  const dialog = await screen.findByRole("dialog", { name: "Re-read all documents?" });
  expect(await within(dialog).findByText(/^Couldn't count the documents and model calls/)).toBeInTheDocument();
  expect(dialog).toHaveTextContent("Any that can't be re-read from what was stored show as failed items.");
  await userEvent.click(within(dialog).getByRole("button", { name: "Re-read documents" }));

  await vi.waitFor(() => expect(toast.success).toHaveBeenCalledWith("Re-reading Payroll wiki"));
  expect((await reprocessPosts(calls)).at(-1)).toEqual({ all_documents: true, dry_run: false });
});

test.each([
  ["source_has_no_documents", "This source has no documents yet. Sync it first."],
  ["source_sync_not_supported", "source_sync_not_supported"],
])("Re-read all documents shows the refusal %s and offers no re-read", async (code, shown) => {
  renderPage([makeSource()], {
    "POST /api/v1/sources/src-wiki/reprocess": () => errorResponse(409, code),
  });
  await openReRead();

  const dialog = await screen.findByRole("dialog", { name: "Re-read all documents?" });
  expect(await within(dialog).findByText(shown)).toBeInTheDocument();
  expect(within(dialog).getByRole("button", { name: "Re-read documents" })).toBeDisabled();
});

test("Stop sync stops the running sync of a source, from its row and from its drawer", async () => {
  const running = makeSource({
    sync: { status: "running", run_id: "ssr-run", started_at: "2026-10-23T08:00:00Z", finished_at: null, error_message: null, progress: null },
  });
  const { calls } = renderPage([running], {
    "POST /api/v1/sources/src-wiki/sync/stop": () => ({
      source_id: "src-wiki", outcome: "stopping", run_id: "ssr-run", local_agent_job_ids: [],
    }),
  });
  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByRole("button", { name: "Stop sync" }));

  await vi.waitFor(() => expect(toast.success).toHaveBeenCalledWith("Stopping the sync of Payroll wiki"));
  const stopCall = calls.find((request) => request.method === "POST")!;
  expect(new URL(stopCall.url).pathname).toBe("/api/v1/sources/src-wiki/sync/stop");

  await userEvent.click(within(table).getByText("Payroll wiki"));
  const drawer = await screen.findByRole("dialog", { name: "Payroll wiki" });
  expect(within(drawer).getByRole("button", { name: "Stop sync" })).toBeEnabled();
  expect(within(drawer).getByRole("button", { name: "Syncing" })).toBeDisabled();
});

test("Configure stays available during a sync, beside Pause, for someone who can configure the source", async () => {
  const running = makeSource({
    sync: { status: "running", run_id: "ssr-run", started_at: "2026-10-23T08:00:00Z", finished_at: null, error_message: null, progress: null },
  });
  renderPage([running]);
  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByRole("button", { name: "More actions for Payroll wiki" }));

  for (const name of [/^Configure$/, /Pause syncing/]) {
    const item = await screen.findByRole("menuitem", { name });
    expect(item).not.toHaveAttribute("aria-disabled", "true");
    expect(item).not.toHaveAttribute("data-disabled");
  }
});

test("someone who can pause a source but not sync it can still stop its sync", async () => {
  const base = makeSource();
  const running = makeSource({
    capabilities: { ...base.capabilities!, can_sync: false, can_configure: true },
    sync: { status: "running", run_id: "ssr-run", started_at: "2026-10-23T08:00:00Z", finished_at: null, error_message: null, progress: null },
  });
  renderPage([running], {});
  const table = await screen.findByRole("table", { name: "Sources" });

  expect(await within(table).findByRole("button", { name: "Stop sync" })).toBeEnabled();
});

test("a sync being stopped reads Stopping and a stopped one reads Stopped, not failed", async () => {
  const stopping = makeSource({
    sync: {
      status: "running", run_id: "ssr-run", started_at: "2026-10-23T08:00:00Z", finished_at: null, error_message: null,
      progress: null, stop_requested_at: "2026-10-23T08:01:00Z",
    },
  });
  const stopped = makeSource({
    id: "src-stopped",
    name: "Stopped wiki",
    sync: {
      status: "failed", run_id: "ssr-old", started_at: "2026-10-23T07:00:00Z", finished_at: "2026-10-23T07:01:00Z",
      error_message: "cancelled_by_user", progress: null,
    },
  });
  renderPage([stopping, stopped]);

  const table = await screen.findByRole("table", { name: "Sources" });
  expect(await within(table).findByRole("button", { name: "Stopping" })).toBeDisabled();
  expect(within(table).queryByRole("button", { name: "Stop sync" })).not.toBeInTheDocument();
  expect(within(table).getByText("Stopped")).toBeInTheDocument();
  expect(screen.queryByRole("region", { name: "Needs you" })).not.toBeInTheDocument();
});

const CONFLUENCE_SCHEMA = {
  fields: [{ key: "spaces", label: "Spaces", field_type: "text", options: [], default: "", help_text: "", group: "content", order: 1, advanced: false }],
  groups: [],
};

async function openScopeChange(source = makeSource({ scope_transition: makeScopeTransition() }), routes: Record<string, Route> = {}) {
  const page = renderPage([source], {
    "GET /api/v1/genes/confluence/config-schema": () => CONFLUENCE_SCHEMA,
    "POST /api/v1/sources/src-wiki/scope-change-preview": () => ({
      outcome: "returns_to_previous_scope",
      open_transition: true,
      removes: source.scope_transition?.removes ?? "at_completion",
      refusal: null,
    }),
    ...routes,
  });
  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByText("Payroll wiki"));
  const drawer = await screen.findByRole("dialog", { name: "Payroll wiki" });
  const section = within(drawer).getByRole("heading", { name: "Scope change in progress" }).closest("section")!;
  return { ...page, drawer, section };
}

test("the drawer shows the scope change in progress: what it changes, how far it is, and what keeps it open", async () => {
  const { section } = await openScopeChange(
    makeSource({
      scope_transition: makeScopeTransition({
        failing_items: { count: 1, retrying_after_upgrade: 0, items: [makeFailedItem({ document_id: "page-3", title: "OPS runbook" })] },
      }),
    }),
  );

  expect(within(section).getByText("The last sync left it open")).toBeInTheDocument();
  expect(await within(section).findByText("Spaces")).toBeInTheDocument();
  expect(within(section).getByText("ENG")).toBeInTheDocument();
  expect(within(section).getByText("ENG, OPS")).toBeInTheDocument();
  expect(within(section).getByText("117 of 120 listed items synced under the new scope")).toBeInTheDocument();
  expect(within(section).getByText("1 failed item not yet synced under the change")).toBeInTheDocument();
  expect(within(section).getByText("OPS runbook")).toBeInTheDocument();
  expect(within(section).getByText("403 Forbidden")).toBeInTheDocument();
});

test("Restore the previous scope confirms, then saves the previous scope through the update route", async () => {
  const transition = makeScopeTransition();
  const { section, calls } = await openScopeChange(makeSource({ scope_transition: transition }), {
    "PUT /api/v1/sources/src-wiki": () => ({ ok: true, id: "src-wiki" }),
  });

  await userEvent.click(within(section).getByRole("button", { name: "Restore the previous scope" }));
  const confirm = await screen.findByRole("dialog", { name: "Restore the previous scope?" });
  expect(await within(confirm).findByText(/replaces the scope change in progress with a change back/)).toBeInTheDocument();
  expect(within(confirm).getByText(/Nothing the previous scope synced is removed/)).toBeInTheDocument();
  const preview = calls.find((request) => new URL(request.url).pathname.endsWith("/scope-change-preview"))!;
  expect(await preview.clone().json()).toEqual({ config: transition.restore_config });
  await userEvent.click(within(confirm).getByRole("button", { name: "Restore" }));

  await vi.waitFor(() => expect(calls.some((request) => request.method === "PUT")).toBe(true));
  const save = calls.find((request) => request.method === "PUT")!;
  expect(new URL(save.url).pathname).toBe("/api/v1/sources/src-wiki");
  expect(await save.clone().json()).toEqual({
    config: transition.restore_config,
    expected_scope_transition_id: transition.id,
  });
  await vi.waitFor(() => expect(toast.success).toHaveBeenCalledWith("Payroll wiki returns to its previous scope"));
});

test.each([
  [{ code: "source_sync_running", run_id: "run-9", message: "A sync of this Source is running." }, "A sync of this source is running. Stop the sync, then try again."],
  ["projection_scope_transition_stale", "The scope change in progress changed while you were looking at it. Reload the source, then try again."],
])("a refused restore explains the refusal and keeps the dialog open (%o)", async (detail, message) => {
  const { section } = await openScopeChange(undefined, {
    "PUT /api/v1/sources/src-wiki": () => errorResponse(409, detail),
  });

  await userEvent.click(within(section).getByRole("button", { name: "Restore the previous scope" }));
  const confirm = await screen.findByRole("dialog", { name: "Restore the previous scope?" });
  const restore = within(confirm).getByRole("button", { name: "Restore" });
  await vi.waitFor(() => expect(restore).toBeEnabled());
  await userEvent.click(restore);

  expect(await within(confirm).findByText(message)).toBeInTheDocument();
  expect(toast.success).not.toHaveBeenCalled();
});

test("a change back to the previous scope reads as returning there, with no restore of its own", async () => {
  const { section } = await openScopeChange(
    makeSource({
      scope_transition: makeScopeTransition({
        target_scope: { sync_mode: "spaces", spaces: ["ENG"] },
        restore_config: null,
        listed_count: null,
      }),
    }),
  );

  expect(within(section).getByText("Returning to the previous scope")).toBeInTheDocument();
  expect(within(section).getByText("The next sync lists what the previous scope holds.")).toBeInTheDocument();
  expect(within(section).queryByRole("button", { name: "Restore the previous scope" })).not.toBeInTheDocument();
});

test("a conversation source's change says its next sync removes the deselected conversations, and so does its restore", async () => {
  const { section } = await openScopeChange(
    makeSource({
      scope_transition: makeScopeTransition({
        status: "pending",
        previous_scope: { conversation_ids: ["19:a@thread.v2"] },
        target_scope: { conversation_ids: ["19:a@thread.v2", "19:c@thread.v2"] },
        members_committed: 0,
        listed_count: null,
        error: null,
        restore_config: { conversation_ids: ["19:a@thread.v2"], initial_history_days: 30 },
        removes: "next_sync",
      }),
    }),
  );

  expect(within(section).getByText("The next sync removes the conversations the new selection leaves out.")).toBeInTheDocument();
  expect(within(section).getByText("The change completes once every selected conversation is synced.")).toBeInTheDocument();
  expect(within(section).queryByText(/Nothing is removed/)).not.toBeInTheDocument();
  await userEvent.click(within(section).getByRole("button", { name: "Restore the previous scope" }));
  const confirm = await screen.findByRole("dialog", { name: "Restore the previous scope?" });
  expect(
    await within(confirm).findByText(/The next sync removes the conversations the new selection leaves out\. The change completes/),
  ).toBeInTheDocument();
  expect(within(confirm).queryByText(/Nothing the previous scope synced is removed/)).not.toBeInTheDocument();
});

test("a viewer who cannot read the configuration sees the progress, but neither scope nor a way back", async () => {
  const { section } = await openScopeChange(
    makeSource({
      scope_transition: makeScopeTransition({ previous_scope: null, target_scope: null, restore_config: null }),
    }),
  );

  expect(within(section).getByText("The last sync left it open")).toBeInTheDocument();
  expect(within(section).getByText("117 of 120 listed items synced under the new scope")).toBeInTheDocument();
  expect(within(section).queryByText("ENG")).not.toBeInTheDocument();
  expect(within(section).queryByText("ENG, OPS")).not.toBeInTheDocument();
  expect(within(section).queryByRole("button", { name: "Restore the previous scope" })).not.toBeInTheDocument();
});

test("a source with issues links to its evaluation", async () => {
  const { calls, router } = renderPage(undefined, {
    "GET /api/v1/agent-evaluations/online-overview": () =>
      makeOverview({
        sources: [
          makeSourceHealth({ source_id: "src-wiki", name: "Payroll wiki", evaluation_status: "attention", action_issue_group_count: 2 }),
          makeSourceHealth({ source_id: "src-jira", name: "Payroll Jira", evaluation_status: "healthy", action_issue_group_count: 0 }),
        ],
      }),
  });

  const table = await screen.findByRole("table", { name: "Sources" });
  const chip = await within(table).findByRole("link", { name: "Open the pipeline checks of Payroll wiki: 2 issues" });
  expect(within(table).queryByRole("link", { name: /pipeline checks of Payroll Jira/ })).not.toBeInTheDocument();
  const overview = new URL(calls.find((request) => request.url.includes("/online-overview"))!.url);
  expect(overview.searchParams.get("days")).toBe("1");

  await userEvent.click(chip);

  expect(router.state.location.pathname + router.state.location.search).toBe("/evaluation?source_id=src-wiki");
});

test("the drawer shows the evaluation chip and offers Full sync", async () => {
  renderPage([makeSource()], {
    "GET /api/v1/agent-evaluations/online-overview": () =>
      makeOverview({
        sources: [makeSourceHealth({ source_id: "src-wiki", evaluation_status: "review", action_issue_group_count: 0, review_issue_group_count: 1 })],
      }),
  });

  await userEvent.click(await within(await screen.findByRole("table", { name: "Sources" })).findByText("Payroll wiki"));
  const drawer = await screen.findByRole("dialog", { name: "Payroll wiki" });
  expect(await within(drawer).findByRole("link", { name: "Open the pipeline checks of Payroll wiki: 1 review pattern" })).toHaveAttribute(
    "href",
    "/evaluation?source_id=src-wiki",
  );

  await userEvent.click(within(drawer).getByRole("button", { name: "More actions for Payroll wiki" }));
  await userEvent.click(await screen.findByRole("menuitem", { name: /Full sync/ }));

  expect(await screen.findByRole("dialog", { name: "Run a full sync of Payroll wiki?" })).toBeInTheDocument();
});
