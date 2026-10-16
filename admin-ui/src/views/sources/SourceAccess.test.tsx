import type { ReactNode } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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

const privateSource: Source = {
  id: "src-wiki", name: "Payroll wiki", type: "confluence", status: "active",
  config: {}, access_policy: "private", memory_count: 1, doc_count: 1,
  access_state: "active", last_sync: null, created_at: "2026-09-01T00:00:00Z",
  capabilities: { can_sync: true, can_configure: true, can_delete: true, can_configure_connection: true,
    can_subscribe: true, can_change_access: true, can_force_resync: true },
  sync: null,
};

let queryClient: QueryClient;
const workspaceApi = createWorkspaceApiController({ clear: () => queryClient?.clear() });

function useWorkspace(shared: boolean) {
  workspaceApi.setTarget({ resourceBaseUrl: "/api/v1", localAgentBaseUrl: "/api/cloud/local-agent", workspaceId: "local", shared });
}

beforeEach(() => {
  vi.mocked(getCurrentLocalAgentJobs).mockResolvedValue([]);
  vi.spyOn(resourceClient, "get").mockImplementation(async (path: string) => {
    if (path === "/sources") return { data: [privateSource] };
    if (path === "/source-list/preferences") return { data: { sort_mode: "newest" } };
    if (path === "/agent-evaluations/online-overview") return { data: { sources: [] } };
    if (path === "/genes/github_pages/config-schema") return { data: { groups: [], fields: [] } };
    return { data: [] };
  });
  vi.spyOn(resourceClient, "post").mockResolvedValue({ data: { id: "src-new" } });
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

async function openSourceMenu() {
  await screen.findByText("Payroll wiki");
  fireEvent.click(screen.getByRole("button", { name: /More actions for Payroll wiki/ }));
}

it("a workspace of one shows no access label and no Change access action", async () => {
  useWorkspace(false);
  mount(<SourcesPage />);

  await openSourceMenu();

  expect(screen.queryByText("Only me")).toBeNull();
  expect(screen.queryByText("Change access")).toBeNull();
});

it("a shared workspace labels private sources and offers Change access", async () => {
  useWorkspace(true);
  mount(<SourcesPage />);

  await openSourceMenu();

  expect(screen.getByText("Only me")).toBeTruthy();
  expect(screen.getByText("Change access")).toBeTruthy();
});

it("a workspace of one creates its sources private without asking who can use them", async () => {
  useWorkspace(false);
  mount(<SchemaSourceSetup open onOpenChange={() => {}} sourceType="github_pages" />);

  fireEvent.change(await screen.findByPlaceholderText("Source name"), { target: { value: "Team pages" } });
  expect(screen.queryByText(/Who can use this source\?/)).toBeNull();
  fireEvent.click(screen.getByRole("button", { name: "Create Source" }));

  await waitFor(() => expect(resourceClient.post).toHaveBeenCalledWith(
    "/sources",
    expect.objectContaining({ type: "github_pages", access_policy: "private", name: "Team pages" }),
  ));
});

it("a shared workspace asks who can use a new source", async () => {
  useWorkspace(true);
  mount(<SchemaSourceSetup open onOpenChange={() => {}} sourceType="github_pages" />);

  expect(await screen.findByText(/Who can use this source\?/)).toBeTruthy();
});
