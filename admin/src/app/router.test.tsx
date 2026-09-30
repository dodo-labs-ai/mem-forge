import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { RouterProvider, createMemoryRouter } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { ApiProvider, createApiClient, createWorkspaceController } from "@/api";
import { fakeFetch } from "@/test/fakeApi";
import { APP_BASE_PATH } from "./basePath";
import type { AdminExtension } from "./extension/contract";
import { ExtensionProvider } from "./extension/ExtensionProvider";
import { buildRoutes } from "./router";

const ORIGIN = "http://admin.test";
const BASENAME = APP_BASE_PATH.replace(/\/$/, "");

const CLOUD_EXTENSION: AdminExtension = {
  id: "cloud",
  routes: [{ path: "cloud/settings", element: <h1>Cloud settings</h1> }],
  reservedRouteRedirects: [{ from: "settings", to: "/cloud/settings" }],
};

afterEach(() => {
  vi.unstubAllGlobals();
});

function openApp(path: string, extension?: AdminExtension) {
  fakeFetch({
    "GET /api/cloud/local-agent/status": () => ({ status: "online", last_seen_at: null, checked_at: "", stale_after_seconds: 60 }),
    "GET /api/cloud/local-agent/jobs/current": () => ({ data: [] }),
    "GET /api/v1/llm-config": () => ({ writable: true, enrichment_api_key_set: false, embedding_api_key_set: false }),
  });
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const workspace = createWorkspaceController(() => queryClient.clear());
  const router = createMemoryRouter(buildRoutes(extension), { basename: BASENAME, initialEntries: [`${BASENAME}${path}`] });
  render(
    <ExtensionProvider extension={extension}>
      <QueryClientProvider client={queryClient}>
        <ApiProvider api={createApiClient(workspace, ORIGIN)} workspace={workspace}>
          <RouterProvider router={router} />
        </ApiProvider>
      </QueryClientProvider>
    </ExtensionProvider>,
  );
  return router;
}

test("the standalone app serves the Settings page", async () => {
  const router = openApp("/settings");

  expect(await screen.findByRole("heading", { name: "Settings" })).toBeInTheDocument();
  expect(router.state.location.pathname).toBe(`${BASENAME}/settings`);
});

test("an extension that redirects Settings opens its own page instead", async () => {
  const router = openApp("/settings", CLOUD_EXTENSION);

  expect(await screen.findByRole("heading", { name: "Cloud settings" })).toBeInTheDocument();
  expect(router.state.location.pathname).toBe(`${BASENAME}/cloud/settings`);
});
