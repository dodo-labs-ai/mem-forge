import { keepPreviousData, useMutation, useQueries, useQuery, useQueryClient, type QueryKey } from "@tanstack/react-query";
import { useMemo } from "react";
import { projectQueryKeys, unwrap, useApi, type components } from "@/api";
import { reviewKeys } from "@/features/review";
import { LIST_PAGE_SIZE } from "./constants";
import { pageOffset, type MemoryListRequest } from "./model/listState";
import { memoryRow, searchHitRow, type MemoryRow } from "./model/memoryRows";
import { RELATION_LABELS, type RelationCounts } from "./model/relations";
import type { CorrectionKind, DismissedRelation, MemoryDetail, MemoryRelation, RelationLabel } from "./model/types";

export const memoryKeys = {
  all: ["memories"] as const,
  list: (request: MemoryListRequest) => ["memories", "list", request] as const,
  stats: ["memories", "stats"] as const,
  openReviewCount: ["memories", "open-review-count"] as const,
  relations: ["memories", "relations"] as const,
  relationPage: (label: RelationLabel | null, page: number) => ["memories", "relations", label, page] as const,
  relationCount: (label: RelationLabel) => ["memories", "relations", label, "count"] as const,
  detail: (memoryId: string) => ["memories", "detail", memoryId] as const,
};

type Schemas = components["schemas"];

/** One row is enough to read a list's total. */
const COUNT_ONLY_LIMIT = 1;

/** How many reviews wait for a decision. */
export function useOpenReviewCount() {
  const api = useApi();
  const query = useQuery({
    queryKey: memoryKeys.openReviewCount,
    queryFn: () =>
      unwrap(api.GET("/api/v1/memory-reviews", { params: { query: { status: "open", limit: COUNT_ONLY_LIMIT } } })),
  });
  return { ...query, total: query.data?.total };
}

export interface MemoryPage {
  rows: MemoryRow[];
  /** Every matching Memory for the listing; the ranked candidate count for a search. */
  total: number;
  ranked: boolean;
}

type MemoryListData =
  | { kind: "search"; page: Schemas["MemorySearchResponse"] }
  | { kind: "list"; page: Schemas["MemoryListResponse"] };

export function useMemoryList(request: MemoryListRequest) {
  const api = useApi();
  const query = useQuery({
    queryKey: memoryKeys.list(request),
    queryFn: async (): Promise<MemoryListData> =>
      request.kind === "search"
        ? { kind: "search", page: await unwrap(api.POST("/api/v1/memories/search", { body: request.body })) }
        : { kind: "list", page: await unwrap(api.GET("/api/v1/memories", { params: { query: request.query } })) },
    placeholderData: keepPreviousData,
  });
  const page = useMemo<MemoryPage | undefined>(() => {
    const data = query.data;
    if (!data) return undefined;
    if (data.kind === "search") {
      return {
        rows: data.page.results.map(searchHitRow),
        total: data.page.total_candidates,
        ranked: true,
      };
    }
    return { rows: data.page.data.map(memoryRow), total: data.page.total, ranked: false };
  }, [query.data]);
  return { ...query, page };
}

export function useMemoryStats() {
  const api = useApi();
  return useQuery({ queryKey: memoryKeys.stats, queryFn: () => unwrap(api.GET("/api/v1/memories/stats")) });
}

/** The number of current relations per label, one small request each. */
export function useRelationCounts(): { counts: RelationCounts | undefined; isError: boolean } {
  const api = useApi();
  const results = useQueries({
    queries: RELATION_LABELS.map((label) => ({
      queryKey: memoryKeys.relationCount(label),
      queryFn: () =>
        unwrap(api.GET("/api/v1/memories/relations", { params: { query: { label, limit: COUNT_ONLY_LIMIT } } })),
    })),
  });
  const totals = results.map((result) => result.data?.total);
  const complete = totals.every((total) => total !== undefined);
  const counts = complete
    ? (Object.fromEntries(RELATION_LABELS.map((label, index) => [label, totals[index]!])) as RelationCounts)
    : undefined;
  return { counts, isError: results.some((result) => result.isError) };
}

export function useRelationPage(label: RelationLabel | null, page: number) {
  const api = useApi();
  return useQuery({
    queryKey: memoryKeys.relationPage(label, page),
    queryFn: () =>
      unwrap(
        api.GET("/api/v1/memories/relations", {
          params: { query: { ...(label ? { label } : {}), limit: LIST_PAGE_SIZE, offset: pageOffset(page) } },
        }),
      ),
    placeholderData: keepPreviousData,
  });
}

/** Refetches everything the Memories pages show. */
export function useRefreshMemories() {
  const queryClient = useQueryClient();
  return () => queryClient.invalidateQueries({ queryKey: memoryKeys.all });
}

export function useMemory(memoryId: string) {
  const api = useApi();
  return useQuery({
    queryKey: memoryKeys.detail(memoryId),
    queryFn: () =>
      unwrap(api.GET("/api/v1/memories/{memory_id}", { params: { path: { memory_id: memoryId } } })),
  });
}

/**
 * Retiring or correcting a Memory can open or close a review, and changes the
 * active Memories each project counts.
 */
const LIFECYCLE_AFFECTED: readonly QueryKey[] = [memoryKeys.all, reviewKeys.all, projectQueryKeys.all];
/** A relation is shown only on the Memories pages. */
const RELATION_AFFECTED: readonly QueryKey[] = [memoryKeys.all];

/** Runs one memory change, then refetches every cached query in `affected`. */
function useMemoryMutation<TVariables, TResult>(
  affected: readonly QueryKey[],
  run: (variables: TVariables) => Promise<TResult>,
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: run,
    onSettled: () => Promise.all(affected.map((queryKey) => queryClient.invalidateQueries({ queryKey }))),
  });
}

export function useRetireMemory(memory: MemoryDetail) {
  const api = useApi();
  return useMemoryMutation(LIFECYCLE_AFFECTED, (reason: string) =>
    unwrap(
      api.POST("/api/v1/memories/{memory_id}/retire", {
        params: { path: { memory_id: memory.id } },
        body: { reason, expected_content_hash: memory.content_hash },
      }),
    ),
  );
}

export interface CorrectionInput {
  replacementContent: string;
  replacementKind: CorrectionKind;
  provenance: string;
  reason: string;
}

export function useProposeCorrection(memory: MemoryDetail) {
  const api = useApi();
  return useMemoryMutation(LIFECYCLE_AFFECTED, (input: CorrectionInput) =>
    unwrap(
      api.POST("/api/v1/memories/{memory_id}/corrections/propose", {
        params: { path: { memory_id: memory.id } },
        body: {
          replacement_content: input.replacementContent,
          replacement_kind: input.replacementKind,
          provenance: input.provenance,
          reason: input.reason,
          expected_content_hash: memory.content_hash,
        },
      }),
    ),
  );
}

export function useDismissRelation(memory: MemoryDetail) {
  const api = useApi();
  return useMemoryMutation(RELATION_AFFECTED, ({ relation, note }: { relation: MemoryRelation; note: string }) =>
    unwrap(
      api.POST("/api/v1/memories/{memory_id}/relations/{counterpart_memory_id}/dismissal", {
        params: { path: { memory_id: memory.id, counterpart_memory_id: relation.counterpart.memory_id } },
        body: {
          label: relation.label,
          expected_content_hash: memory.content_hash,
          counterpart_expected_content_hash: relation.counterpart.content_hash,
          ...(note.trim() !== "" ? { note: note.trim() } : {}),
        },
      }),
    ),
  );
}

export function useRestoreRelation(memory: MemoryDetail) {
  const api = useApi();
  return useMemoryMutation(RELATION_AFFECTED, (dismissal: DismissedRelation) =>
    unwrap(
      api.DELETE("/api/v1/memories/{memory_id}/relations/{counterpart_memory_id}/dismissal", {
        params: { path: { memory_id: memory.id, counterpart_memory_id: dismissal.counterpart.memory_id } },
      }),
    ),
  );
}
