import client from "./client";
import type { LocalAgentJobCreateResponse } from "./types";
import { requireCurrentWorkspaceId } from "@/lib/workspace";

interface CreateLocalAgentJobInput {
  sourceId?: string;
  sourceType: string;
  operation: string;
  payload?: Record<string, unknown>;
}

export async function createLocalAgentJob({
  sourceId = "",
  sourceType,
  operation,
  payload = {},
}: CreateLocalAgentJobInput): Promise<LocalAgentJobCreateResponse> {
  const response = await client.post<LocalAgentJobCreateResponse>("/api/cloud/local-agent/jobs", {
    workspace_id: requireCurrentWorkspaceId(),
    source_id: sourceId,
    source_type: sourceType,
    operation,
    payload,
  });
  return response.data;
}
