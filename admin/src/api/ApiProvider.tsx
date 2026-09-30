import { createContext, useContext, useSyncExternalStore, type ReactNode } from "react";
import type { ApiClient } from "./client";
import type { WorkspaceController, WorkspaceTarget } from "./workspace";

interface ApiContextValue {
  api: ApiClient;
  workspace: WorkspaceController;
}

const ApiContext = createContext<ApiContextValue | null>(null);

export function ApiProvider({ api, workspace, children }: ApiContextValue & { children: ReactNode }) {
  return <ApiContext.Provider value={{ api, workspace }}>{children}</ApiContext.Provider>;
}

function useApiContext(): ApiContextValue {
  const value = useContext(ApiContext);
  if (value === null) throw new Error("useApi must be used inside <ApiProvider>.");
  return value;
}

/** The typed Admin API client. */
export function useApi(): ApiClient {
  return useApiContext().api;
}

/** The current workspace target; re-renders when the Cloud extension switches workspace. */
export function useWorkspaceTarget(): WorkspaceTarget | null {
  const { workspace } = useApiContext();
  return useSyncExternalStore(workspace.subscribe, workspace.current, workspace.current);
}
