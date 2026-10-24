import { useMemo } from "react";
import { queryOptions, useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { unwrap, useApi, type ApiClient, type Source, type SourceListSortMode } from "@/api";
import { localSyncKeys } from "@/features/local-sync";
import { ACTIVE_POLL_MS } from "./constants";
import type { AccessPolicy } from "./model/sourceAccess";
import type { SourceSyncRetryTarget } from "./model/sourceSyncActivity";
import { sourceProjectBinding } from "./model/projectBinding";
import { sourcesByProjectKey, type ResolvedBySource } from "./model/projectGrouping";
import type { LocalAgentJob, SourceEvaluationHealth } from "./model/types";

export const sourceKeys = {
  all: ["sources"] as const,
  list: ["sources", "list"] as const,
  imageDocuments: ["sources", "image-documents"] as const,
  preferences: ["sources", "preferences"] as const,
  types: ["genes"] as const,
  configSchema: (sourceType: string) => ["genes", sourceType, "config-schema"] as const,
};

/** The days of online evaluation the rows and the drawer summarize. */
const EVALUATION_HEALTH_DAYS = 1;
/** How long that evaluation is reused before it is read again. */
const EVALUATION_HEALTH_STALE_MS = 60_000;
const EMPTY_HEALTH: ReadonlyMap<string, SourceEvaluationHealth> = new Map();

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

/** The registered source types, with their display names, where they run and whether the service manages them. */
export function useSourceTypes() {
  const api = useApi();
  return useQuery({ queryKey: sourceKeys.types, queryFn: () => unwrap(api.GET("/api/v1/genes")) });
}

/** Display names of source types, such as "Confluence" for `confluence`. */
export function useSourceTypeLabels(): Record<string, string> {
  const genes = useSourceTypes();
  return useMemo(
    () => Object.fromEntries((genes.data ?? []).map((gene) => [gene.name, gene.display_name])),
    [genes.data],
  );
}

/** A source type's form description, which does not change while the app runs. */
export function configSchemaQuery(api: ApiClient, sourceType: string) {
  return queryOptions({
    queryKey: sourceKeys.configSchema(sourceType),
    queryFn: () => unwrap(api.GET("/api/v1/genes/{name}/config-schema", { params: { path: { name: sourceType } } })),
    staleTime: Infinity,
  });
}

/** The labels a source type's configuration gives its fields, keyed by field, such as "Spaces" for `spaces`. */
export function useSourceConfigLabels(sourceType: string | null): Record<string, string> {
  const api = useApi();
  const schema = useQuery({ ...configSchemaQuery(api, sourceType ?? ""), enabled: sourceType !== null });
  return useMemo(
    () => Object.fromEntries((schema.data?.fields ?? []).map((field) => [field.key, field.label])),
    [schema.data],
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

// One reference for every caller, so the map is built once per response.
const healthBySource = (overview: { sources: readonly SourceEvaluationHealth[] }) =>
  new Map(overview.sources.map((health) => [health.source_id, health]));

/**
 * The online evaluation of each source the viewer can discover over the last
 * day, by source id. A source without an entry, or one the evaluation cannot
 * be read for, shows nothing.
 */
export function useSourceEvaluationHealth(): ReadonlyMap<string, SourceEvaluationHealth> {
  const api = useApi();
  const query = useQuery({
    // Kept apart from the source keys, which every source write refreshes.
    queryKey: ["source-evaluation-health", EVALUATION_HEALTH_DAYS] as const,
    queryFn: () =>
      unwrap(api.GET("/api/v1/agent-evaluations/online-overview", { params: { query: { days: EVALUATION_HEALTH_DAYS } } })),
    select: healthBySource,
    staleTime: EVALUATION_HEALTH_STALE_MS,
  });
  return query.data ?? EMPTY_HEALTH;
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
 * Saves the configuration that returns a source to the scope it had before
 * the change in progress, which replaces that change.
 */
export function useRestorePreviousScope() {
  const api = useApi();
  return useSourceMutation(
    ({ sourceId, transitionId, config }: { sourceId: string; transitionId: string; config: Record<string, unknown> }) =>
      unwrap(
        api.PUT("/api/v1/sources/{source_id}", {
          params: { path: { source_id: sourceId } },
          // The server refuses the save when another change, or none, is in progress by then.
          body: { config, expected_scope_transition_id: transitionId },
        }),
      ),
  );
}

/** What saving `config` would do to the source's scope, read while `enabled`; it saves nothing. */
export function useScopeChangePreview(sourceId: string, config: Record<string, unknown> | null, enabled: boolean) {
  const api = useApi();
  return useQuery({
    // Kept apart from the source keys, which every source write refreshes.
    queryKey: ["source-scope-change-preview", sourceId, config] as const,
    queryFn: () =>
      unwrap(
        api.POST("/api/v1/sources/{source_id}/scope-change-preview", {
          params: { path: { source_id: sourceId } },
          body: { config: config! },
        }),
      ),
    enabled: enabled && config !== null,
    staleTime: 0,
    gcTime: 0,
    retry: false,
  });
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
          dry_run: false,
          force_full_sync: false,
          retry_target: retryTarget?.execution_kind === "source_sync_run" ? retryTarget : undefined,
        },
      }),
    );
  });
}

/**
 * Starts a full sync: the server, or the device of a source that runs on
 * one, lists the whole source again and processes every document, unchanged
 * ones included.
 */
export function useFullSync() {
  const api = useApi();
  return useSourceMutation((sourceId: string) =>
    unwrap(api.POST("/api/v1/sources/{source_id}/force-resync", { params: { path: { source_id: sourceId } } })),
  );
}

/**
 * Stops the source's sync in progress: a server run, or the local sync job of
 * its device. The source stays active, and its next sync runs as usual.
 */
export function useStopSync() {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (sourceId: string) =>
      unwrap(api.POST("/api/v1/sources/{source_id}/sync/stop", { params: { path: { source_id: sourceId } } })),
    onSettled: () => Promise.all([
      queryClient.invalidateQueries({ queryKey: sourceKeys.all }),
      queryClient.invalidateQueries({ queryKey: localSyncKeys.jobs }),
    ]),
  });
}

export function useDeleteSource() {
  const api = useApi();
  return useSourceMutation((sourceId: string) =>
    unwrap(api.DELETE("/api/v1/sources/{source_id}", { params: { path: { source_id: sourceId } } })),
  );
}

/**
 * Starts changing who can use a source, across its whole history. The
 * server answers at once and re-shares the source's memories in the
 * background; the source list follows it while it runs. `requestId` makes a
 * repeated request start the same change, not a second one.
 */
export function useChangeSourceAccess() {
  const api = useApi();
  return useSourceMutation(
    ({ sourceId, targetPolicy, requestId }: { sourceId: string; targetPolicy: AccessPolicy; requestId: string }) =>
      unwrap(
        api.POST("/api/v1/sources/{source_id}/access-transitions", {
          params: { path: { source_id: sourceId }, header: { "Idempotency-Key": requestId } },
          body: { target_policy: targetPolicy },
        }),
      ),
  );
}

/** How long a retry or revert is followed until the server's worker picks it up. */
const SETTLE_PICKUP_MS = 30_000;

/**
 * Settles a failed access change: `retry` runs it again, and `revert`
 * returns every memory to the policy the source had before it. The server
 * queues either one, so this follows the source's current access change
 * until the worker has picked it up; the refreshed source list then follows
 * it while it runs.
 */
export function useSettleSourceAccess() {
  const api = useApi();
  return useSourceMutation(
    async ({ source, settle }: { source: Source; settle: "retry" | "revert" }) => {
      const failed = source.access_transition;
      if (!failed) throw new Error("This source has no access change to settle.");
      const path = { source_id: source.id, operation_id: failed.operation_id };
      await unwrap(
        settle === "retry"
          ? api.POST("/api/v1/sources/{source_id}/access-transitions/{operation_id}/retry", { params: { path } })
          : api.POST("/api/v1/sources/{source_id}/access-transitions/{operation_id}/revert", { params: { path } }),
      );
      const deadline = Date.now() + SETTLE_PICKUP_MS;
      while (Date.now() < deadline) {
        const { transition } = await unwrap(
          api.GET("/api/v1/sources/{source_id}/access-transition", { params: { path: { source_id: source.id } } }),
        );
        if (transition?.status !== "failed" || transition.updated_at !== failed.updated_at) return;
        await new Promise((resolve) => window.setTimeout(resolve, ACTIVE_POLL_MS));
      }
    },
  );
}
