import { currentLocalAgentBaseUrl, hostClient } from "./client";
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
}

const LOCAL_AGENT_JOB_CONTROL_BASE_URL = "/api/cloud/local-agent";

export async function createLocalAgentJob({
  sourceId = "",
  sourceType,
  operation,
  payload = {},
}: CreateLocalAgentJobInput): Promise<LocalAgentJobCreateResponse> {
  const response = await hostClient.post<LocalAgentJobCreateResponse>(localAgentUrl("/jobs"), {
    source_id: sourceId,
    source_type: sourceType,
    operation,
    payload,
  });
  return response.data;
}

export async function getLocalAgentJob(jobId: string): Promise<LocalAgentJobStatusResponse> {
  const response = await hostClient.get<LocalAgentJobStatusResponse>(
    `${LOCAL_AGENT_JOB_CONTROL_BASE_URL}/jobs/${encodeURIComponent(jobId)}`,
  );
  return response.data;
}

export async function getLocalAgentDaemonStatus(): Promise<LocalAgentDaemonStatusResponse> {
  const response = await hostClient.get<LocalAgentDaemonStatusResponse>(localAgentUrl("/status"));
  return response.data;
}

function localAgentUrl(path: `/${string}`): string {
  return `${currentLocalAgentBaseUrl()}${path}`;
}
