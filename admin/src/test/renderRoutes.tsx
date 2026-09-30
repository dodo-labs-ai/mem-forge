import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import { RouterProvider, createMemoryRouter, type RouteObject } from "react-router-dom";
import { ApiProvider, createApiClient, createWorkspaceController, type WorkspaceTarget } from "@/api";
import { TooltipProvider } from "@/ui/tooltip";
import { fakeFetch, type Route as FakeRoute } from "./fakeApi";

const ORIGIN = "http://admin.test";

interface RenderRoutesOptions {
  /** Where the router starts, such as "/memories?project=PAY". */
  path: string;
  /** The fake API the page reads; see `fakeFetch`. */
  api: Record<string, FakeRoute>;
  /** The selected workspace; the standalone target when left out. */
  workspace?: WorkspaceTarget;
  /** How long a query's data stays fresh by default, as the app sets it; every read refetches when left out. */
  queryStaleMs?: number;
}

/**
 * Renders `routes` at `path` inside the providers the app gives every page,
 * against a fake API. Returns the requests the page made and the router.
 */
export function renderRoutes(
  routes: RouteObject[],
  { path, api, workspace: target, queryStaleMs }: RenderRoutesOptions,
) {
  const fetch = fakeFetch(api);
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: queryStaleMs }, mutations: { retry: false } },
  });
  const workspace = createWorkspaceController(() => queryClient.clear(), target);
  const router = createMemoryRouter(routes, { initialEntries: [path] });
  render(
    <QueryClientProvider client={queryClient}>
      <ApiProvider api={createApiClient(workspace, ORIGIN)} workspace={workspace}>
        <TooltipProvider>
          <RouterProvider router={router} />
        </TooltipProvider>
      </ApiProvider>
    </QueryClientProvider>,
  );
  return { ...fetch, router };
}
