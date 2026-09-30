import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { unwrap, useApi } from "@/api";
import type { LlmConfigUpdate, LlmProbeRequest } from "./model/types";

export const settingsKeys = {
  llmConfig: ["settings", "llm-config"] as const,
};

export function useLlmConfig() {
  const api = useApi();
  return useQuery({
    queryKey: settingsKeys.llmConfig,
    queryFn: () => unwrap(api.GET("/api/v1/llm-config")),
  });
}

/** Saves the changed endpoints; the server answers with the stored result. */
export function useSaveLlmConfig() {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (body: LlmConfigUpdate) => unwrap(api.PUT("/api/v1/llm-config", { body })),
    onSuccess: (config) => queryClient.setQueryData(settingsKeys.llmConfig, config),
  });
}

/** Tests an endpoint from the API server and lists its models when it can. */
export function useProbeEndpoint() {
  const api = useApi();
  return useMutation({
    mutationFn: (body: LlmProbeRequest) => unwrap(api.POST("/api/v1/llm-config/probe", { body })),
  });
}
