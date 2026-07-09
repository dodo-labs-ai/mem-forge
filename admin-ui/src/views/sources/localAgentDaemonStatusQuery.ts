import { useQuery } from "@tanstack/react-query";
import client from "@/api/client";
import type { LocalAgentDaemonStatusResponse } from "@/api/types";

const LOCAL_AGENT_STATUS_ENDPOINT = "/api/cloud/local-agent/status";
const LOCAL_AGENT_STATUS_QUERY_KEY = ["local-agent-daemon-status"] as const;
const LOCAL_AGENT_STATUS_REFETCH_MS = 30_000;

export function useLocalAgentDaemonStatus() {
  return useQuery<LocalAgentDaemonStatusResponse>({
    queryKey: LOCAL_AGENT_STATUS_QUERY_KEY,
    queryFn: () => client.get(LOCAL_AGENT_STATUS_ENDPOINT).then((response) => response.data),
    refetchInterval: LOCAL_AGENT_STATUS_REFETCH_MS,
    refetchOnWindowFocus: true,
    staleTime: 15_000,
  });
}
