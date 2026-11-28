import { LOCAL_AGENT_HOST_BASE_URL, hostClient, resourceClient } from "./client";
import type {
  LocalAgentDaemonStatusResponse,
  LocalAgentJobCreateResponse,
  LocalAgentJobStatusResponse,
} from "./types";

interface CreateLocalAgentJobInput {
  sourceId?: string;
  sourceType: string;
  operation: string;
  payload?: Record<string, unknown>;
  retryJobId?: string;
}

export async function createLocalAgentJob({
  sourceId = "",
  sourceType,
  operation,
  payload = {},
  retryJobId,
}: CreateLocalAgentJobInput): Promise<LocalAgentJobCreateResponse> {
  const response = await resourceClient.post<LocalAgentJobCreateResponse>("/local-agent/jobs", {
    source_id: sourceId,
    source_type: sourceType,
    operation,
    payload,
    ...(retryJobId ? { retry_job_id: retryJobId } : {}),
  });
  return response.data;
}

export async function getLocalAgentJob(jobId: string): Promise<LocalAgentJobStatusResponse> {
  const response = await hostClient.get<LocalAgentJobStatusResponse>(
    `${LOCAL_AGENT_HOST_BASE_URL}/jobs/${encodeURIComponent(jobId)}`,
  );
  return response.data;
}

export async function getCurrentLocalAgentJobs(): Promise<LocalAgentJobStatusResponse[]> {
  const response = await resourceClient.get<{ data: LocalAgentJobStatusResponse[] }>("/local-agent/jobs/current");
  return response.data.data;
}

export async function getLocalAgentDaemonStatus(): Promise<LocalAgentDaemonStatusResponse> {
  const response = await hostClient.get<LocalAgentDaemonStatusResponse>(`${LOCAL_AGENT_HOST_BASE_URL}/status`);
  return response.data;
}
