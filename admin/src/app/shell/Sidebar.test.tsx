import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { ApiProvider, createApiClient, createWorkspaceController, reviewQueryKeys } from "@/api";
import { fakeFetch } from "@/test/fakeApi";
import type { AdminExtension } from "../extension/contract";
import { ExtensionProvider } from "../extension/ExtensionProvider";
import { Sidebar } from "./Sidebar";

const ORIGIN = "http://admin.test";

afterEach(() => {
  vi.unstubAllGlobals();
});

function renderSidebar(extension?: AdminExtension, openReviews = 0) {
  const { calls } = fakeFetch({
    "GET /api/cloud/local-agent/status": () => ({ status: "online", last_seen_at: null, checked_at: "", stale_after_seconds: 60 }),
    "GET /api/cloud/local-agent/jobs/current": () => ({ data: [] }),
    "GET /api/v1/memory-reviews": () => ({ data: [], total: openReviews, limit: 1, offset: 0 }),
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
  return { calls, queryClient };
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

test("shows the open review count and caps at 99+", async () => {
  const { calls } = renderSidebar(undefined, 120);

  const review = within(screen.getByRole("navigation", { name: "Main" })).getByRole("link", { name: /Review/ });
  expect(await within(review).findByText("99+")).toBeInTheDocument();
  const request = new URL(calls.find((call) => call.url.includes("/memory-reviews"))!.url);
  expect(request.searchParams.get("status")).toBe("open");
  expect(request.searchParams.get("limit")).toBe("1");
});

test("shows a count up to 99 as it is", async () => {
  renderSidebar(undefined, 7);

  const review = within(screen.getByRole("navigation", { name: "Main" })).getByRole("link", { name: /Review/ });
  expect(await within(review).findByText("7")).toBeInTheDocument();
});

test("hides at zero", async () => {
  const { queryClient } = renderSidebar(undefined, 0);

  await waitFor(() => expect(queryClient.getQueryState(reviewQueryKeys.openCount)?.status).toBe("success"));
  const review = within(screen.getByRole("navigation", { name: "Main" })).getByRole("link", { name: /Review/ });
  expect(review).toHaveTextContent(/^Review$/);
});
