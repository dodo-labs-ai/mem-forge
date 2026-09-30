import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";
import { RouterProvider, createBrowserRouter } from "react-router-dom";
import { ApiProvider } from "@/api/ApiProvider";
import { createApiClient } from "@/api/client";
import { createWorkspaceController } from "@/api/workspace";
import { Toaster } from "@/ui/sonner";
import { TooltipProvider } from "@/ui/tooltip";
import { APP_BASE_PATH } from "./basePath";
import type { AdminExtension } from "./extension/contract";
import { ExtensionProvider } from "./extension/ExtensionProvider";
import { buildRoutes } from "./router";

const QUERY_STALE_MS = 30_000;

function PassThrough({ children }: { children: ReactNode }) {
  return <>{children}</>;
}

function createServices(extension: AdminExtension | undefined) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { staleTime: QUERY_STALE_MS, retry: 1 } },
  });
  // Data cached for one workspace must never show under another.
  const workspace = createWorkspaceController(() => queryClient.clear());
  const api = createApiClient(workspace);
  const router = createBrowserRouter(buildRoutes(extension), { basename: APP_BASE_PATH.replace(/\/$/, "") });
  return { queryClient, workspace, api, router };
}

/** The admin UI, optionally with one extension such as MemForge Cloud. */
export function AdminApp({ extension }: { extension?: AdminExtension }) {
  const [{ queryClient, workspace, api, router }] = useState(() => createServices(extension));
  const Wrapper = extension?.Wrapper ?? PassThrough;
  return (
    <ExtensionProvider extension={extension}>
      <QueryClientProvider client={queryClient}>
        <ApiProvider api={api} workspace={workspace}>
          <Wrapper workspace={workspace}>
            <TooltipProvider>
              <RouterProvider router={router} />
              <Toaster position="bottom-right" />
            </TooltipProvider>
          </Wrapper>
        </ApiProvider>
      </QueryClientProvider>
    </ExtensionProvider>
  );
}
