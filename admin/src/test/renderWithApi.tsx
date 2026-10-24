import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render } from "@testing-library/react";
import type { ReactElement } from "react";
import { ApiProvider, createApiClient, createWorkspaceController } from "@/api";
import { TooltipProvider } from "@/ui/tooltip";
import { fakeFetch, type Route } from "./fakeApi";

/**
 * Renders a component that reads the API, such as a setup widget, inside the
 * providers the app gives it, against a fake API (see `fakeFetch`), with no
 * router. Returns the requests it made.
 */
export function renderWithApi(ui: ReactElement, api: Record<string, Route>) {
  const fetch = fakeFetch(api);
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } });
  const workspace = createWorkspaceController(() => queryClient.clear());
  const view = render(
    <QueryClientProvider client={queryClient}>
      <ApiProvider api={createApiClient(workspace, "http://admin.test")} workspace={workspace}>
        <TooltipProvider>{ui}</TooltipProvider>
      </ApiProvider>
    </QueryClientProvider>,
  );
  return { ...fetch, ...view, queryClient };
}
