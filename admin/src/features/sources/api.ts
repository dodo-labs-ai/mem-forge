import { useMemo } from "react";
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { unwrap, useApi, type Source, type SourceListSortMode } from "@/api";
import { localSyncKeys } from "@/features/local-sync";
import { ACTIVE_POLL_MS } from "./constants";
import type { SourceSyncRetryTarget } from "./model/sourceSyncActivity";
import { sourceProjectBinding } from "./model/projectBinding";
import { sourcesByProjectKey, type ResolvedBySource } from "./model/projectGrouping";
import type { LocalAgentJob } from "./model/types";

export const sourceKeys = {
  all: ["sources"] as const,
  list: ["sources", "list"] as const,
  imageDocuments: ["sources", "image-documents"] as const,
  preferences: ["sources", "preferences"] as const,
  types: ["genes"] as const,
};

const EMPTY_SOURCES: Source[] = [];
const EMPTY_JOBS: LocalAgentJob[] = [];
const ACTIVE_SYNC_STATES = new Set(["pending", "running", "recovering"]);
const ACTIVE_JOB_STATES = new Set(["queued", "leased"]);

function sourcesAreChanging(sources: Source[]): boolean {
  return sources.some(
    (source) =>
      ACTIVE_SYNC_STATES.has(source.sync?.status ?? "") ||
      (source.access_state === "changing" && source.access_transition?.status !== "failed"),
  );
}

export function useSources() {
  const api = useApi();
  const query = useQuery({
    queryKey: sourceKeys.list,
    queryFn: () => unwrap(api.GET("/api/v1/sources")),
    select: (response) => response.data,
    refetchInterval: (current) =>
      sourcesAreChanging(current.state.data?.data ?? EMPTY_SOURCES) ? ACTIVE_POLL_MS : false,
  });
  return { ...query, sources: query.data ?? EMPTY_SOURCES };
}

/** Local sync jobs in flight, which carry progress for sources that sync from a device. */
export function useLocalSyncJobs() {
  const api = useApi();
  const query = useQuery({
    queryKey: localSyncKeys.jobs,
    queryFn: () => unwrap(api.GET("/api/cloud/local-agent/jobs/current")),
    select: (response) => response.data,
    refetchInterval: (current) =>
      (current.state.data?.data ?? EMPTY_JOBS).some((job) => ACTIVE_JOB_STATES.has(job.status)) ? ACTIVE_POLL_MS : false,
  });
  return { ...query, jobs: query.data ?? EMPTY_JOBS };
}

/** Display names of source types, such as "Confluence" for `confluence`. */
export function useSourceTypeLabels(): Record<string, string> {
  const api = useApi();
  const genes = useQuery({ queryKey: sourceKeys.types, queryFn: () => unwrap(api.GET("/api/v1/genes")) });
  return useMemo(
    () => Object.fromEntries((genes.data ?? []).map((gene) => [gene.name, gene.display_name])),
    [genes.data],
  );
}

/** The projects each field-bound source has sent memories to, as the server resolved them. */
export function useResolvedProjects(sources: Source[]): ResolvedBySource {
  const api = useApi();
  const byField = sources.filter((source) => sourceProjectBinding(source)?.mode === "by_field");
  const results = useQueries({
    queries: byField.map((source) => ({
      queryKey: [...sourceKeys.all, "resolved-projects", source.id],
      queryFn: () =>
        unwrap(api.GET("/api/v1/sources/{source_id}/projects/resolved", { params: { path: { source_id: source.id } } })),
    })),
  });
  const resolved: ResolvedBySource = {};
  results.forEach((result, index) => {
    const source = byField[index];
    if (source && result.data) resolved[source.id] = result.data.projects;
  });
  return resolved;
}

/** The sources that send memories to each project, keyed by project key. */
export function useProjectSources() {
  const sourcesQuery = useSources();
  const resolved = useResolvedProjects(sourcesQuery.sources);
  return { ...sourcesQuery, byProject: sourcesByProjectKey(sourcesQuery.sources, resolved) };
}

export function useSourceListPreferences() {
  const api = useApi();
  return useQuery({
    queryKey: sourceKeys.preferences,
    queryFn: () => unwrap(api.GET("/api/v1/source-list/preferences")),
  });
}

export function useSetSortMode() {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (sortMode: SourceListSortMode) =>
      unwrap(api.PUT("/api/v1/source-list/preferences", { body: { sort_mode: sortMode } })),
    onSuccess: (preferences) => queryClient.setQueryData(sourceKeys.preferences, preferences),
  });
}

function useSourceMutation<TVariables>(run: (variables: TVariables) => Promise<unknown>) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: run,
    onSettled: () => Promise.all([
      queryClient.invalidateQueries({ queryKey: sourceKeys.all }),
      queryClient.invalidateQueries({ queryKey: localSyncKeys.jobs }),
    ]),
  });
}

/**
 * Reads how many current documents have an image a model can read, in the
 * sources the user can re-read. It is read when asked, not kept fresh.
 */
export function useFetchImageDocuments() {
  const api = useApi();
  const queryClient = useQueryClient();
  return () =>
    queryClient.fetchQuery({
      queryKey: sourceKeys.imageDocuments,
      queryFn: () => unwrap(api.GET("/api/v1/sources/image-documents")),
      staleTime: 0,
    });
}

/** Queues a re-read of the documents with an image, one run per source. */
export function useReReadImageDocuments() {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => unwrap(api.POST("/api/v1/sources/image-documents/reprocess")),
    onSettled: () => queryClient.invalidateQueries({ queryKey: sourceKeys.all }),
  });
}

/** What re-reading every document of a source would read, and the model calls it would take. */
export function useReReadPreview(sourceId: string | null) {
  const api = useApi();
  return useQuery({
    // Kept apart from the source keys, which every source write refreshes.
    queryKey: ["source-re-read-preview", sourceId] as const,
    queryFn: async () => {
      const preview = await unwrap(
        api.POST("/api/v1/sources/{source_id}/reprocess", {
          params: { path: { source_id: sourceId! } },
          body: { all_documents: true, dry_run: true },
        }),
      );
      if (!("units" in preview)) throw new Error("The server queued a re-read instead of previewing it.");
      return preview;
    },
    enabled: sourceId !== null,
    staleTime: 0,
    gcTime: 0,
  });
}

/**
 * Queues a re-read of a source with the current models: of the documents
 * named, or, with none named, of every current document.
 */
export function useReReadSource() {
  const api = useApi();
  return useSourceMutation(({ sourceId, documentIds }: { sourceId: string; documentIds: string[] | null }) =>
    unwrap(
      api.POST("/api/v1/sources/{source_id}/reprocess", {
        params: { path: { source_id: sourceId } },
        body:
          documentIds === null
            ? { all_documents: true, dry_run: false }
            : { document_ids: documentIds, all_documents: false, dry_run: false },
      }),
    ),
  );
}

export function useSetPinned() {
  const api = useApi();
  return useSourceMutation(({ sourceId, pinned }: { sourceId: string; pinned: boolean }) => {
    const params = { params: { path: { source_id: sourceId } } };
    return unwrap(pinned ? api.PUT("/api/v1/sources/{source_id}/pin", params) : api.DELETE("/api/v1/sources/{source_id}/pin", params));
  });
}

export function useSetUsedInSearches() {
  const api = useApi();
  return useSourceMutation(({ sourceId, enabled }: { sourceId: string; enabled: boolean }) =>
    unwrap(
      api.PUT("/api/v1/sources/{source_id}/subscription", {
        params: { path: { source_id: sourceId } },
        body: { enabled },
      }),
    ),
  );
}

export function useSetPaused() {
  const api = useApi();
  return useSourceMutation(({ sourceId, paused }: { sourceId: string; paused: boolean }) =>
    unwrap(
      api.PUT("/api/v1/sources/{source_id}", {
        params: { path: { source_id: sourceId } },
        body: { status: paused ? "paused" : "active" },
      }),
    ),
  );
}

/**
 * Starts a sync on the server, or queues it for local sync when the source
 * runs on a device. With a retry target it starts that queued retry now
 * instead, so a waiting server run is not collected again from the device.
 */
export function useSyncSource() {
  const api = useApi();
  return useSourceMutation(({ source, retryTarget }: { source: Source; retryTarget?: SourceSyncRetryTarget }) => {
    const operation = source.execution?.kind === "local_agent" ? source.execution.operation : null;
    if (operation && retryTarget?.execution_kind !== "source_sync_run") {
      return unwrap(
        api.POST("/api/cloud/local-agent/jobs", {
          body: {
            source_id: source.id,
            source_type: source.type,
            operation,
            payload: { force_full_sync: false },
            retry_job_id: retryTarget?.execution_id,
          },
        }),
      );
    }
    return unwrap(
      api.POST("/api/v1/sources/{source_id}/sync", {
        params: { path: { source_id: source.id } },
        body: {
          force_full_sync: false,
          retry_target: retryTarget?.execution_kind === "source_sync_run" ? retryTarget : undefined,
        },
      }),
    );
  });
}

export function useDeleteSource() {
  const api = useApi();
  return useSourceMutation((sourceId: string) =>
    unwrap(api.DELETE("/api/v1/sources/{source_id}", { params: { path: { source_id: sourceId } } })),
  );
}
