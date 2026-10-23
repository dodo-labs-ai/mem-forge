import { keepPreviousData, skipToken, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { unwrap, useApi } from "@/api";
import { memoryKeys } from "@/features/memories";
import { entityPath } from "@/lib/paths";

export const entityKeys = {
  all: ["entities"] as const,
  list: (search: string, limit: number, offset: number) => ["entities", "list", search, limit, offset] as const,
  detail: (entityId: number) => ["entities", "detail", entityId] as const,
};

interface EntityListRequest {
  search: string;
  limit: number;
  offset: number;
}

/** One page of the entities the caller can discover, narrowed to names matching `search` when it is not blank. */
export function useEntityList({ search, limit, offset }: EntityListRequest) {
  const api = useApi();
  const query = search.trim();
  return useQuery({
    queryKey: entityKeys.list(query, limit, offset),
    queryFn: () =>
      unwrap(api.GET("/api/v1/entities", { params: { query: { search: query || undefined, limit, offset } } })),
    placeholderData: keepPreviousData,
  });
}

/** One entity with its aliases, linked memory count and whether the caller may curate it; idle while `entityId` is null. */
export function useEntity(entityId: number | null) {
  const api = useApi();
  return useQuery({
    queryKey: entityKeys.detail(entityId ?? -1),
    queryFn:
      entityId === null
        ? skipToken
        : () => unwrap(api.GET("/api/v1/entities/{entity_id}", { params: { path: { entity_id: entityId } } })),
  });
}

/** An alias change touches only the entity's own detail. */
function useAliasMutation<TVariables>(entityId: number, run: (variables: TVariables) => Promise<unknown>) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: run,
    onSettled: () => queryClient.invalidateQueries({ queryKey: entityKeys.detail(entityId) }),
  });
}

export function useAddEntityAlias(entityId: number) {
  const api = useApi();
  return useAliasMutation(entityId, (alias: string) =>
    unwrap(api.POST("/api/v1/entities/{entity_id}/aliases", { params: { path: { entity_id: entityId } }, body: { alias } })),
  );
}

export function useRemoveEntityAlias(entityId: number) {
  const api = useApi();
  return useAliasMutation(entityId, (alias: string) =>
    unwrap(api.DELETE("/api/v1/entities/{entity_id}/aliases/{alias}", { params: { path: { entity_id: entityId, alias } } })),
  );
}

/**
 * Merges `sourceId` into a target and opens the target. The source entity is
 * gone afterwards, so its detail is dropped rather than refetched; the
 * target's detail, the lists and the memories that name either entity are
 * read again.
 */
export function useMergeEntity(sourceId: number) {
  const api = useApi();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  return useMutation({
    mutationFn: (targetId: number) =>
      unwrap(api.POST("/api/v1/entities/merge", { body: { source_id: sourceId, target_id: targetId } })),
    onSuccess: async ({ merged }) => {
      navigate(entityPath(merged.target_id));
      queryClient.removeQueries({ queryKey: entityKeys.detail(sourceId) });
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: entityKeys.all }),
        queryClient.invalidateQueries({ queryKey: memoryKeys.all }),
      ]);
    },
  });
}
