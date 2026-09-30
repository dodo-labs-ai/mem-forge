import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { unwrap, useApi, type Source, type SourceListSortMode } from "@/api";
import { localSyncKeys } from "@/features/local-sync";
import { ACTIVE_POLL_MS } from "./constants";
import type { LocalAgentJob } from "./model/types";

export const sourceKeys = {
  all: ["sources"] as const,
  list: ["sources", "list"] as const,
  preferences: ["sources", "preferences"] as const,
  projects: ["projects", "list"] as const,
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

export function useProjects() {
  const api = useApi();
  return useQuery({
    queryKey: sourceKeys.projects,
    queryFn: () => unwrap(api.GET("/api/v1/projects")),
  });
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

/** Starts a sync on the server, or queues it for local sync when the source runs on a device. */
export function useSyncSource() {
  const api = useApi();
  return useSourceMutation((source: Source) => {
    const operation = source.execution?.kind === "local_agent" ? source.execution.operation : null;
    if (operation) {
      return unwrap(
        api.POST("/api/cloud/local-agent/jobs", {
          body: {
            source_id: source.id,
            source_type: source.type,
            operation,
            payload: { force_full_sync: false },
          },
        }),
      );
    }
    return unwrap(
      api.POST("/api/v1/sources/{source_id}/sync", {
        params: { path: { source_id: source.id } },
        body: { force_full_sync: false },
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
