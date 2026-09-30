import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { ApiProvider, createApiClient, createWorkspaceController } from "@/api";
import { fakeFetch } from "@/test/fakeApi";
import { makeSource } from "@/test/sourceFixtures";
import { TooltipProvider } from "@/ui/tooltip";
import { SourcesPage } from "./SourcesPage";

const ORIGIN = "http://admin.test";

afterEach(() => {
  vi.unstubAllGlobals();
});

function renderPage(sources = [makeSource(), makeSource({ id: "src-jira", type: "jira", name: "Payroll Jira", project_binding: null })]) {
  const fetch = fakeFetch({
    "GET /api/v1/sources": () => ({ data: sources }),
    "GET /api/v1/projects": () => [{ id: "p1", key: "PAY", name: "Payroll", kind: "normal" }],
    "GET /api/v1/genes": () => [
      { name: "confluence", display_name: "Confluence" },
      { name: "jira", display_name: "Jira" },
    ],
    "GET /api/v1/source-list/preferences": () => ({ sort_mode: "name" }),
    "GET /api/cloud/local-agent/status": () => ({ status: "online", last_seen_at: null, checked_at: "", stale_after_seconds: 60 }),
    "GET /api/cloud/local-agent/jobs/current": () => ({ data: [] }),
    "POST /api/v1/sources/src-wiki/sync": () => ({ status: "pending", run_id: "run-2" }),
  });
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const workspace = createWorkspaceController(() => queryClient.clear());
  render(
    <QueryClientProvider client={queryClient}>
      <ApiProvider api={createApiClient(workspace, ORIGIN)} workspace={workspace}>
        <TooltipProvider>
          <MemoryRouter>
            <SourcesPage />
          </MemoryRouter>
        </TooltipProvider>
      </ApiProvider>
    </QueryClientProvider>,
  );
  return fetch;
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
