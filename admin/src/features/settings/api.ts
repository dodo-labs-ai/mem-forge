import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { modelSettingsQueryKeys, unwrap, useApi } from "@/api";
import type {
  ModelCandidateRequest,
  ModelDiscoveryRequest,
  ModelKind,
  ModelKindSettings,
  ModelSettings,
} from "./model/types";

/** The providers a model can be reached through, with the fields each needs. */
export function useModelProviders() {
  const api = useApi();
  return useQuery({
    queryKey: modelSettingsQueryKeys.providers,
    queryFn: () => unwrap(api.GET("/api/v1/model-settings/providers")),
    select: (list) => list.providers,
    staleTime: Infinity,
  });
}

/**
 * Puts the settings of one kind, as a write answered them, into the settings
 * every page reads. A write to the embedding model also has the settings read
 * again: the search index is judged against that model, and the answer to the
 * write does not carry it.
 */
function useStoreKindSettings(kind: ModelKind) {
  const queryClient = useQueryClient();
  return (settings: ModelKindSettings) => {
    queryClient.setQueryData<ModelSettings>(modelSettingsQueryKeys.settings, (current) =>
      current ? { ...current, [kind]: settings } : current,
    );
    if (kind === "embedding") void queryClient.invalidateQueries({ queryKey: modelSettingsQueryKeys.settings });
  };
}

/** Saves an edit to a model. The model in use is unchanged until the edit passes validation. */
export function useSaveCandidate(kind: ModelKind) {
  const api = useApi();
  const store = useStoreKindSettings(kind);
  return useMutation({
    mutationFn: (body: ModelCandidateRequest) =>
      unwrap(api.PUT("/api/v1/model-settings/{kind}/candidate", { params: { path: { kind } }, body })),
    onSuccess: store,
  });
}

/** Drops the saved edit of a model, leaving the model in use as it is. */
export function useDiscardCandidate(kind: ModelKind) {
  const api = useApi();
  const store = useStoreKindSettings(kind);
  return useMutation({
    mutationFn: () => unwrap(api.DELETE("/api/v1/model-settings/{kind}/candidate", { params: { path: { kind } } })),
    onSuccess: store,
  });
}

/**
 * Validates the saved edit with one real request, which can take up to a
 * minute. An edit that passes becomes the model in use, except an embedding
 * model whose vectors cannot be compared with the stored ones: it stays the
 * saved edit until the search index is rebuilt for it. When the request
 * itself fails the settings are read again, since the server may have changed
 * them.
 */
export function useValidateCandidate(kind: ModelKind) {
  const api = useApi();
  const queryClient = useQueryClient();
  const store = useStoreKindSettings(kind);
  return useMutation({
    mutationFn: () =>
      unwrap(api.POST("/api/v1/model-settings/{kind}/candidate/validate", { params: { path: { kind } } })),
    onSuccess: (outcome) => store(outcome.settings),
    onError: () => queryClient.invalidateQueries({ queryKey: modelSettingsQueryKeys.settings }),
  });
}

/** Validates the model in use again. It stays in use whatever the result. */
export function useValidateActive(kind: ModelKind) {
  const api = useApi();
  const store = useStoreKindSettings(kind);
  return useMutation({
    mutationFn: () => unwrap(api.POST("/api/v1/model-settings/{kind}/active/validate", { params: { path: { kind } } })),
    onSuccess: (outcome) => store(outcome.settings),
  });
}

/** Asks an endpoint, from the API server, which models it offers. */
export function useDiscoverModels(kind: ModelKind) {
  const api = useApi();
  return useMutation({
    mutationFn: (body: ModelDiscoveryRequest) =>
      unwrap(api.POST("/api/v1/model-settings/{kind}/discover-models", { params: { path: { kind } }, body })),
  });
}

/**
 * Starts a rebuild of the search index. The server answers at once and the
 * rebuild runs beside it; the settings every page reads then follow it until
 * it ends. A start that is refused reads the settings again, since the server
 * may have validated the embedding model first, or a rebuild may already be
 * running.
 */
export function useStartReindex() {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => unwrap(api.POST("/api/v1/model-settings/vector-index/reindex")),
    onSuccess: async (vectorIndex) => {
      // A read that began before the rebuild would put the index back as it was.
      await queryClient.cancelQueries({ queryKey: modelSettingsQueryKeys.settings });
      queryClient.setQueryData<ModelSettings>(modelSettingsQueryKeys.settings, (current) =>
        current ? { ...current, vector_index: vectorIndex } : current,
      );
    },
    onError: () => queryClient.invalidateQueries({ queryKey: modelSettingsQueryKeys.settings }),
  });
}
