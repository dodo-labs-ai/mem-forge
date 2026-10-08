import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import { toast } from "sonner";
import type { WorkspaceTarget } from "@/api";
import { errorResponse, type Route } from "@/test/fakeApi";
import { renderRoutes } from "@/test/renderRoutes";
import { makeLocalAgentJob, makeSource } from "@/test/sourceFixtures";
import { SourcesPage } from "./SourcesPage";

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

const RETRY_DELAY_MS = 60 * 60_000;

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
    "POST /api/v1/sources/src-wiki/sync": () => ({ status: "pending", run_id: "run-2" }),
    ...routes,
  };
  return renderRoutes([{ path: "/sources", element: <SourcesPage /> }, { path: "*", element: null }], {
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

test("Sync now starts a sync for that source", async () => {
  const { calls } = renderPage([makeSource()]);
  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByRole("button", { name: "Sync now" }));

  await vi.waitFor(() => expect(calls.some((request) => request.method === "POST")).toBe(true));
  const syncCall = calls.find((request) => request.method === "POST")!;
  expect(new URL(syncCall.url).pathname).toBe("/api/v1/sources/src-wiki/sync");
  expect(await syncCall.clone().json()).toEqual({ force_full_sync: false });
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
    "GET /api/cloud/workspaces/mount_tai/v1/sources": () => ({ data: [signedIn
      ? { ...expiredJira(), connection_status: { state: "ready", reason: null } } : expiredJira()] }),
    "GET /api/cloud/workspaces/mount_tai/v1/projects": () => ({ data: [], can_manage: true }),
    "GET /api/cloud/workspaces/mount_tai/v1/genes": () => [{ name: "jira", display_name: "Jira" }],
    "GET /api/cloud/workspaces/mount_tai/v1/source-list/preferences": () => ({ sort_mode: "name" }),
    "GET /api/cloud/workspaces/mount_tai/local-agent/status": () => ({ status: "online" }),
    "GET /api/cloud/workspaces/mount_tai/local-agent/jobs/current": () => ({ data: [] }),
    "POST /api/cloud/workspaces/mount_tai/local-agent/jobs": () => ({ job_id: "laj-auth", status: "queued" }),
    "GET /api/cloud/local-agent/jobs/laj-auth": () => new Promise((resolve) => { complete = resolve; }),
  }, {
    resourceBaseUrl: "/api/cloud/workspaces/mount_tai/v1",
    localAgentBaseUrl: "/api/cloud/workspaces/mount_tai/local-agent",
    workspaceId: "mount_tai",
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
