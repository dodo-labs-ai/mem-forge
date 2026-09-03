import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { resourceClient } from "@/api/client";
import { createLocalAgentJob, getCurrentLocalAgentJobs, getLocalAgentJob } from "@/api/localAgentJobs";
import type { LocalAgentJobStatusResponse, Source } from "@/api/types";
import { SourcesPage } from "./SourcesPage";

vi.mock("@/api/client", () => ({
  currentLocalAgentBaseUrl: () => "/api/cloud/workspaces/ws-a/local-agent",
  resourceClient: { get: vi.fn(), post: vi.fn(), put: vi.fn() },
}));
vi.mock("@/api/localAgentJobs", () => ({
  createLocalAgentJob: vi.fn(), getCurrentLocalAgentJobs: vi.fn(), getLocalAgentJob: vi.fn(),
  getLocalAgentDaemonStatus: async () => ({ state: "online", last_seen_at: null }),
}));
vi.mock("./SourceSetupDialog", () => ({ SourceSetupDialog: () => null }));
vi.mock("./SourceAccessChangeDialog", () => ({ SourceAccessChangeDialog: () => null }));
vi.mock("./LocalAgentDaemonStatus", () => ({ LocalAgentDaemonStatus: () => null }));

const workspace = "/api/cloud/workspaces/ws-a/local-agent";
const source: Source = {
  id: "src-local", name: "Local fixture", type: "local_markdown", status: "active",
  config: { root: "/fixture" }, access_policy: "workspace", memory_count: 1, doc_count: 1,
  access_state: "active", last_sync: null, created_at: "2026-09-01T00:00:00Z",
  capabilities: { can_sync: true, can_configure: true, can_delete: true, can_configure_connection: true,
    can_subscribe: false, can_change_access: false, can_force_resync: true },
  execution: { kind: "local_agent", operation: "local_markdown_sync", immutable_config_fields: [] },
  sync: { run_id: "old-run", status: "success", started_at: "2026-09-01T00:00:00Z", error_message: null,
    finished_at: "2026-09-01T00:00:00Z" },
};
let queryClient: QueryClient;
beforeEach(() => {
  vi.mocked(getLocalAgentJob).mockImplementation(async (id) => ({ job_id: id, status: "queued", result: null, last_error: null }));
});
afterEach(() => { cleanup(); queryClient?.clear(); vi.resetAllMocks(); });

function mount() {
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><MemoryRouter><SourcesPage /></MemoryRouter></QueryClientProvider>);
}

function otherResponse(path: string) {
  if (path === "/source-list/preferences") return { sort_mode: "newest" };
  if (path === "/agent-evaluations/online-overview") return { sources: [] };
  return [];
}

it("accepts without waiting for daemon completion and keeps refreshing through Cloud handoff", async () => {
  let jobs: LocalAgentJobStatusResponse[] = [];
  let handoffReads = 0;
  vi.mocked(getCurrentLocalAgentJobs).mockImplementation(async () => jobs);
  vi.mocked(resourceClient.get).mockImplementation(async (path) => {
    if (path !== "/sources") return { data: otherResponse(path) };
    let sync = source.sync;
    if (jobs[0]?.status === "succeeded" && ++handoffReads >= 2) {
      sync = { ...source.sync!, run_id: "cloud-run", status: handoffReads >= 3 ? "success" : "pending" };
    }
    return { data: [{ ...source, sync }] };
  });
  vi.mocked(createLocalAgentJob).mockImplementation(async () => {
    jobs = [{ job_id: "local-job", source_id: source.id, status: "queued", operation: "local_markdown_sync",
      next_attempt_at: "2099-01-01T00:00:00Z", result: null, last_error: "VPN unavailable" }];
    return { ok: true, job_id: "local-job", status: "queued" };
  });
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Sync" }));
  await screen.findAllByRole("button", { name: "Retry now" });
  expect(createLocalAgentJob).toHaveBeenCalledTimes(1);
  expect(screen.getByRole("button", { name: "Configure Local fixture" }).hasAttribute("disabled")).toBe(true);
  expect(queryClient.isMutating()).toBe(0);
  jobs = [{ ...jobs[0], status: "succeeded", result: { source_sync_run_id: "cloud-run" }, finished_at: "2026-09-03T00:00:00Z" }];
  await waitFor(() => {
    const current = queryClient.getQueryData<Source[]>(["sources", workspace]);
    expect(current?.[0].sync?.run_id).toBe("cloud-run");
    expect(current?.[0].sync?.status).toBe("success");
  }, { timeout: 10000 });
  expect(handoffReads).toBeGreaterThanOrEqual(3);
  expect(createLocalAgentJob).toHaveBeenCalledTimes(1);
}, 15000);

it("retries the displayed Cloud run without starting another collection", async () => {
  vi.mocked(getCurrentLocalAgentJobs).mockResolvedValue([]);
  vi.mocked(resourceClient.get).mockImplementation(async (path) => ({ data: path === "/sources"
    ? [{ ...source, sync: { ...source.sync, run_id: "cloud-retry", status: "pending", next_attempt_at: "2099-01-01T00:00:00Z" } }]
    : otherResponse(path) }));
  vi.mocked(resourceClient.post).mockResolvedValue({ data: { run_id: "cloud-retry", status: "pending" } });
  mount();
  fireEvent.click((await screen.findAllByRole("button", { name: "Retry now" }))[0]);
  await waitFor(() => expect(resourceClient.post).toHaveBeenCalledWith("/sources/src-local/sync", {
    retry_target: { execution_kind: "source_sync_run", execution_id: "cloud-retry" },
  }));
  expect(createLocalAgentJob).not.toHaveBeenCalled();
});

it("shows a query error without fabricating a failed local job", async () => {
  vi.mocked(getCurrentLocalAgentJobs).mockRejectedValue(new Error("offline"));
  vi.mocked(resourceClient.get).mockImplementation(async (path) => ({ data: path === "/sources" ? [source] : otherResponse(path) }));
  mount();
  await screen.findByText(/Unable to refresh sync status/);
  expect(screen.queryByText("Sync failed")).toBeNull();
  expect(queryClient.getQueryData(["currentLocalAgentJobs", workspace])).toBeUndefined();
});

it("retains admitted work across a failed first refresh and recovers without user refresh", async () => {
  let admitted = false;
  let failRefresh = true;
  const job: LocalAgentJobStatusResponse = { job_id: "admitted-job", source_id: source.id, operation: "local_markdown_sync",
    status: "queued", result: null, last_error: null, next_attempt_at: "2099-01-01T00:00:00Z" };
  vi.mocked(getCurrentLocalAgentJobs).mockImplementation(async () => {
    if (!admitted) return [];
    if (failRefresh) throw new Error("temporary status outage");
    return [job];
  });
  vi.mocked(resourceClient.get).mockImplementation(async (path) => ({ data: path === "/sources" ? [source] : otherResponse(path) }));
  vi.mocked(createLocalAgentJob).mockImplementation(async () => {
    admitted = true;
    return { ok: true, job_id: job.job_id, status: "queued" };
  });
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Sync" }));
  await screen.findByText(/Unable to refresh sync status/);
  expect(queryClient.isMutating()).toBe(0);
  expect(screen.getByRole("button", { name: "Configure Local fixture" }).hasAttribute("disabled")).toBe(true);
  expect(screen.queryByText("Sync failed")).toBeNull();
  failRefresh = false;
  await waitFor(() => expect(screen.getAllByRole("button", { name: "Retry now" }).length).toBeGreaterThan(0), { timeout: 5000 });
  expect(createLocalAgentJob).toHaveBeenCalledTimes(1);
});

it("clears an admission when the first refresh already shows a newer terminal run", async () => {
  let admitted = false;
  vi.mocked(getCurrentLocalAgentJobs).mockResolvedValue([]);
  vi.mocked(resourceClient.get).mockImplementation(async (path) => ({ data: path === "/sources"
    ? [{ ...source, type: "github_repo", execution: { kind: "server", operation: null, immutable_config_fields: [] },
      sync: admitted ? { ...source.sync, run_id: "r2", status: "success", created_at: "2026-09-03T00:00:02Z" } : source.sync }]
    : otherResponse(path) }));
  vi.mocked(resourceClient.post).mockImplementation(async () => {
    admitted = true;
    return { data: { run_id: "r1", status: "pending", created_at: "2026-09-03T00:00:01Z" } };
  });
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Sync" }));
  await waitFor(() => expect(queryClient.getQueryData<Source[]>(["sources", workspace])?.[0].sync?.run_id).toBe("r2"));
  await waitFor(() => expect(screen.getByRole("button", { name: "Configure Local fixture" }).hasAttribute("disabled")).toBe(false));
  expect(screen.queryByText("Waiting to sync")).toBeNull();
});

it("does not retain a stale exact retry receipt after the server returns terminal", async () => {
  let retried = false;
  vi.mocked(getCurrentLocalAgentJobs).mockImplementation(async () => [{
    job_id: retried ? "new-job" : "old-job", source_id: source.id, operation: "local_markdown_sync",
    status: retried ? "succeeded" : "queued", result: null, last_error: null,
    created_at: "2026-09-03T00:00:00Z", finished_at: retried ? "2026-09-03T00:01:00Z" : null,
  }]);
  vi.mocked(resourceClient.get).mockImplementation(async (path) => ({ data: path === "/sources" ? [source] : otherResponse(path) }));
  vi.mocked(createLocalAgentJob).mockImplementation(async () => {
    retried = true;
    return { job_id: "old-job", status: "succeeded" };
  });
  mount();
  fireEvent.click((await screen.findAllByRole("button", { name: "Retry now" }))[0]);
  await waitFor(() => expect(screen.getByRole("button", { name: "Configure Local fixture" }).hasAttribute("disabled")).toBe(false));
  expect(screen.queryByText("Waiting to sync")).toBeNull();
});

it("resolves equal-timestamp local successors by exact ID instead of guessing order", async () => {
  let retried = false;
  const created_at = "2026-09-03T00:00:00Z";
  vi.mocked(getCurrentLocalAgentJobs).mockImplementation(async () => [{
    job_id: retried ? "j2" : "j1", source_id: source.id, operation: "local_markdown_sync",
    status: retried ? "succeeded" : "queued", result: null, last_error: null, created_at,
  }]);
  vi.mocked(resourceClient.get).mockImplementation(async (path) => ({ data: path === "/sources" ? [source] : otherResponse(path) }));
  vi.mocked(createLocalAgentJob).mockImplementation(async () => {
    retried = true;
    return { job_id: "j1", status: "queued", created_at };
  });
  vi.mocked(getLocalAgentJob).mockResolvedValue({ job_id: "j1", status: "succeeded", result: null, last_error: null });
  mount();
  fireEvent.click((await screen.findAllByRole("button", { name: "Retry now" }))[0]);
  await waitFor(() => expect(getLocalAgentJob).toHaveBeenCalledWith("j1"));
  await waitFor(() => expect(screen.getByRole("button", { name: "Configure Local fixture" }).hasAttribute("disabled")).toBe(false));
  expect(createLocalAgentJob).toHaveBeenCalledTimes(1);
});

it("resolves equal-timestamp server successors through a read-only exact-run request", async () => {
  let retried = false;
  const created_at = "2026-09-03T00:00:00Z";
  vi.mocked(getCurrentLocalAgentJobs).mockResolvedValue([]);
  vi.mocked(resourceClient.get).mockImplementation(async (path) => ({ data: path === "/sources"
    ? [{ ...source, sync: { ...source.sync, run_id: retried ? "r2" : "r1", status: retried ? "success" : "pending", created_at } }]
    : path.endsWith("/sync-runs/r1") ? { run_id: "r1", status: "succeeded" } : otherResponse(path) }));
  vi.mocked(resourceClient.post).mockImplementation(async () => {
    retried = true;
    return { data: { run_id: "r1", status: "pending", created_at } };
  });
  mount();
  fireEvent.click((await screen.findAllByRole("button", { name: "Retry now" }))[0]);
  await waitFor(() => expect(resourceClient.get).toHaveBeenCalledWith("/sources/src-local/sync-runs/r1"));
  await waitFor(() => expect(screen.getByRole("button", { name: "Configure Local fixture" }).hasAttribute("disabled")).toBe(false));
  expect(resourceClient.post).toHaveBeenCalledTimes(1);
  expect(createLocalAgentJob).not.toHaveBeenCalled();
});

it("shows exact-record refresh failures without releasing the admitted-work guard", async () => {
  vi.mocked(getCurrentLocalAgentJobs).mockResolvedValue([]);
  vi.mocked(getLocalAgentJob).mockRejectedValue(new Error("exact status unavailable"));
  vi.mocked(resourceClient.get).mockImplementation(async (path) => ({ data: path === "/sources" ? [source] : otherResponse(path) }));
  vi.mocked(createLocalAgentJob).mockResolvedValue({ job_id: "admitted-job", status: "queued" });
  mount();
  fireEvent.click(await screen.findByRole("button", { name: "Sync" }));
  await screen.findByText(/Unable to refresh.*sync status/);
  expect(getLocalAgentJob).toHaveBeenCalledWith("admitted-job");
  expect(screen.getByRole("button", { name: "Configure Local fixture" }).hasAttribute("disabled")).toBe(true);
  expect(screen.queryByText("Sync failed")).toBeNull();
  expect(createLocalAgentJob).toHaveBeenCalledTimes(1);
});
