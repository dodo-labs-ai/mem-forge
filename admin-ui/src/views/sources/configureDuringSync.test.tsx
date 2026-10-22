import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { createWorkspaceApiController, resourceClient } from "@/api/client";
import { getCurrentLocalAgentJobs } from "@/api/localAgentJobs";
import type { Source } from "@/api/types";
import { SchemaSourceSetup } from "./SchemaSourceSetup";
import { SourcesPage } from "./SourcesPage";

vi.mock("@/api/localAgentJobs", () => ({
  createLocalAgentJob: vi.fn(), getCurrentLocalAgentJobs: vi.fn(), getLocalAgentJob: vi.fn(),
  getLocalAgentDaemonStatus: async () => ({ state: "online", last_seen_at: null }),
}));
vi.mock("./LocalAgentDaemonStatus", () => ({ LocalAgentDaemonStatus: () => null }));

const idleSource: Source = {
  id: "src-pages", name: "Team pages", type: "github_pages", status: "active",
  config: {}, access_policy: "private", memory_count: 1, doc_count: 1,
  access_state: "active", last_sync: null, created_at: "2026-09-01T00:00:00Z",
  capabilities: { can_sync: true, can_configure: true, can_delete: true, can_configure_connection: true,
    can_subscribe: true, can_change_access: true, can_force_resync: true },
  execution: { kind: "server", operation: null, immutable_config_fields: [] },
  sync: { run_id: "run-0", status: "success", started_at: "2026-10-22T00:00:00Z", finished_at: "2026-10-22T00:01:00Z", error_message: null },
};
const syncingSource: Source = {
  ...idleSource,
  sync: { run_id: "run-1", status: "running", started_at: "2026-10-23T00:00:00Z", finished_at: null, error_message: null },
};
const refusal = {
  response: {
    status: 409,
    data: { detail: { code: "source_sync_running", run_id: "run-2", message: "A sync of this Source is running." } },
  },
};

let queryClient: QueryClient;
const workspaceApi = createWorkspaceApiController({ clear: () => queryClient?.clear() });

beforeEach(() => {
  workspaceApi.setTarget({ resourceBaseUrl: "/api/v1", localAgentBaseUrl: "/api/cloud/local-agent", workspaceId: "local", shared: false });
  vi.mocked(getCurrentLocalAgentJobs).mockResolvedValue([]);
  vi.spyOn(resourceClient, "get").mockImplementation(async (path: string) => {
    if (path === "/sources") return { data: [syncingSource] };
    if (path === "/source-list/preferences") return { data: { sort_mode: "newest" } };
    if (path === "/agent-evaluations/online-overview") return { data: { sources: [] } };
    if (path === "/genes/github_pages/config-schema") return { data: { groups: [], fields: [] } };
    if (path.startsWith("/sources/src-pages/sync-runs/")) return { data: { status: "failed" } };
    return { data: [] };
  });
  vi.spyOn(resourceClient, "post").mockResolvedValue({ data: { outcome: "stopping", run_id: "run-1" } });
  vi.spyOn(resourceClient, "put").mockResolvedValue({ data: {} });
});

afterEach(() => {
  cleanup();
  queryClient?.clear();
  workspaceApi.setTarget(null);
  vi.restoreAllMocks();
});

function mount(element: ReactNode) {
  queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  render(<QueryClientProvider client={queryClient}><MemoryRouter>{element}</MemoryRouter></QueryClientProvider>);
}

async function openConfigureDuringSync() {
  mount(<SourcesPage />);
  const configure = await screen.findByRole("button", { name: "Configure Team pages" });
  expect(configure.hasAttribute("disabled")).toBe(false);
  fireEvent.click(configure);
  // An unmapped Source opens on its project section.
  fireEvent.click(await screen.findByRole("button", { name: /1\. Basics/ }));
  return screen.findByPlaceholderText("Source name");
}

function stopCalls() {
  return vi.mocked(resourceClient.post).mock.calls.filter(([path]) => path === "/sources/src-pages/sync/stop");
}

it("keeps Configure open during a sync and says that saving stops it", async () => {
  await openConfigureDuringSync();

  expect(screen.getByText("A sync is running. Saving will stop it first.")).toBeTruthy();
});

it("asks before saving during a sync, then stops the sync, waits for it to end and saves", async () => {
  const name = await openConfigureDuringSync();
  fireEvent.change(name, { target: { value: "Renamed pages" } });

  fireEvent.click(screen.getByRole("button", { name: "Save Changes" }));
  const confirmation = await screen.findByRole("dialog", { name: "Stop the sync and save?" });
  expect(resourceClient.put).not.toHaveBeenCalled();

  fireEvent.click(within(confirmation).getByRole("button", { name: "Stop sync and save" }));

  await waitFor(() => expect(resourceClient.put).toHaveBeenCalledWith(
    "/sources/src-pages",
    expect.objectContaining({ name: "Renamed pages" }),
  ));
  expect(stopCalls()).toHaveLength(1);
  expect(resourceClient.get).toHaveBeenCalledWith("/sources/src-pages/sync-runs/run-1");
  expect(vi.mocked(resourceClient.post).mock.invocationCallOrder[0])
    .toBeLessThan(vi.mocked(resourceClient.put).mock.invocationCallOrder[0]);
});

it("waits for the sync to finish with the edits kept and the sync left running", async () => {
  const name = await openConfigureDuringSync();
  fireEvent.change(name, { target: { value: "Renamed pages" } });

  fireEvent.click(screen.getByRole("button", { name: "Save Changes" }));
  const confirmation = await screen.findByRole("dialog", { name: "Stop the sync and save?" });
  fireEvent.click(within(confirmation).getByRole("button", { name: "Wait for the sync to finish" }));

  await waitFor(() => expect(screen.queryByRole("dialog", { name: "Stop the sync and save?" })).toBeNull());
  expect((screen.getByPlaceholderText("Source name") as HTMLInputElement).value).toBe("Renamed pages");
  expect(stopCalls()).toHaveLength(0);
  expect(resourceClient.put).not.toHaveBeenCalled();
});

it("asks the same way when the server refuses a save because a sync started meanwhile", async () => {
  vi.mocked(resourceClient.put).mockRejectedValueOnce(refusal).mockResolvedValueOnce({ data: {} });
  mount(<SchemaSourceSetup open onOpenChange={() => {}} sourceType="github_pages" source={idleSource} />);
  await screen.findByPlaceholderText("Source name");
  expect(screen.queryByText("A sync is running. Saving will stop it first.")).toBeNull();

  fireEvent.click(screen.getByRole("button", { name: "Save Changes" }));
  const confirmation = await screen.findByRole("dialog", { name: "Stop the sync and save?" });
  expect(screen.getByText("A sync is running. Saving will stop it first.")).toBeTruthy();
  fireEvent.click(within(confirmation).getByRole("button", { name: "Stop sync and save" }));

  await waitFor(() => expect(resourceClient.put).toHaveBeenCalledTimes(2));
  expect(stopCalls()).toHaveLength(1);
  expect(resourceClient.get).toHaveBeenCalledWith("/sources/src-pages/sync-runs/run-2");
});
