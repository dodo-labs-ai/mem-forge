import { useQuery } from "@tanstack/react-query";
import { unwrap, useApi } from "@/api";

const STATUS_REFETCH_MS = 30_000;
const STATUS_STALE_MS = 15_000;

export const localSyncKeys = {
  status: ["local-sync", "status"] as const,
  jobs: ["local-sync", "jobs"] as const,
};

/** Whether the local sync daemon on the user's machine is reachable. */
export function useLocalSyncStatus() {
  const api = useApi();
  return useQuery({
    queryKey: localSyncKeys.status,
    queryFn: () => unwrap(api.GET("/api/cloud/local-agent/status")),
    refetchInterval: STATUS_REFETCH_MS,
    staleTime: STATUS_STALE_MS,
  });
}

export type LocalSyncState = "checking" | "online" | "offline";

export function localSyncState(query: ReturnType<typeof useLocalSyncStatus>): LocalSyncState {
  if (query.isPending) return "checking";
  if (query.isError) return "offline";
  return query.data.status === "online" ? "online" : "offline";
}
