import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { ApiProvider, createApiClient, createWorkspaceController } from "@/api";
import { fakeFetch } from "@/test/fakeApi";
import type { AdminExtension } from "../extension/contract";
import { ExtensionProvider } from "../extension/ExtensionProvider";
import { Sidebar } from "./Sidebar";

const ORIGIN = "http://admin.test";

afterEach(() => {
  vi.unstubAllGlobals();
});

function renderSidebar(extension: AdminExtension) {
  fakeFetch({
    "GET /api/cloud/local-agent/status": () => ({ status: "online", last_seen_at: null, checked_at: "", stale_after_seconds: 60 }),
    "GET /api/cloud/local-agent/jobs/current": () => ({ data: [] }),
  });
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const workspace = createWorkspaceController(() => queryClient.clear());
  render(
    <ExtensionProvider extension={extension}>
      <QueryClientProvider client={queryClient}>
        <ApiProvider api={createApiClient(workspace, ORIGIN)} workspace={workspace}>
          <MemoryRouter>
            <Sidebar />
          </MemoryRouter>
        </ApiProvider>
      </QueryClientProvider>
    </ExtensionProvider>,
  );
}

function settingsRedirect(visible: boolean): AdminExtension {
  return { id: "cloud", reservedRouteRedirects: [{ from: "settings", to: "/cloud/settings", visibleWhen: () => visible }] };
}

test("a redirected product item opens the extension page", () => {
  renderSidebar(settingsRedirect(true));

  const nav = screen.getByRole("navigation", { name: "Main" });
  expect(within(nav).getByRole("link", { name: "Settings" })).toHaveAttribute("href", "/cloud/settings");
});

test("a redirected product item is hidden from users who cannot open its target", () => {
  renderSidebar(settingsRedirect(false));

  const nav = screen.getByRole("navigation", { name: "Main" });
  expect(within(nav).queryByRole("link", { name: "Settings" })).not.toBeInTheDocument();
});
