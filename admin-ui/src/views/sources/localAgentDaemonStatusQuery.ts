import { useQuery } from "@tanstack/react-query";
import { getLocalAgentDaemonStatus } from "@/api/localAgentJobs";
import type { LocalAgentDaemonStatusResponse } from "@/api/types";

const LOCAL_AGENT_STATUS_QUERY_KEY = ["local-agent-daemon-status"] as const;
const LOCAL_AGENT_STATUS_REFETCH_MS = 30_000;

export function useLocalAgentDaemonStatus() {
  return useQuery<LocalAgentDaemonStatusResponse>({
    queryKey: LOCAL_AGENT_STATUS_QUERY_KEY,
    queryFn: getLocalAgentDaemonStatus,
    refetchInterval: LOCAL_AGENT_STATUS_REFETCH_MS,
    refetchOnWindowFocus: true,
    staleTime: 15_000,
  });
}
