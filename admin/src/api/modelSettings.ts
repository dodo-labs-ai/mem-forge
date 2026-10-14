import { useQuery } from "@tanstack/react-query";
import { useApi } from "./ApiProvider";
import { unwrap } from "./client";
import type { components } from "./schema.gen";

type Schemas = components["schemas"];

/** One configured model: its provider, endpoint, model and the state of its latest validation. */
export type ModelBinding = Schemas["ModelBindingResponse"];
/** The model of one kind that is in use (`active`) and the edit that waits to be validated (`candidate`). */
export type ModelKindSettings = Schemas["ModelKindSettingsResponse"];
/** The models of both kinds, the search index, and whether this installation lets Settings change the models. */
export type ModelSettings = Schemas["ModelSettingsResponse"];
/**
 * The search index: the vectors of all memories, which belong to one
 * embedding model. It says whether it fits the model in use and how the
 * latest rebuild of it went.
 */
export type VectorIndex = Schemas["VectorIndexResponse"];
/** The two models MemForge calls: one writes memories, one embeds them for search. */
export type ModelKind = ModelBinding["kind"];

export const MODEL_KINDS: readonly ModelKind[] = ["generation", "embedding"];

/**
 * Query keys of the model settings. Settings and the notices in the app shell
 * read the one answer, so a model that passes validation or a search index
 * that is rebuilt clears its notice without another request.
 */
export const modelSettingsQueryKeys = {
  all: ["model-settings"] as const,
  settings: ["model-settings", "settings"] as const,
  providers: ["model-settings", "providers"] as const,
};

/** How often the settings are read again while the server validates a model on its own. */
const BACKGROUND_VALIDATION_REFETCH_MS = 10_000;
/** The reads after which a result is no longer expected; a validation takes about a minute at most. */
const BACKGROUND_VALIDATION_MAX_READS = 18;

/** Whether the server is still to report the first validation of a model in use. */
function awaitsBackgroundValidation(settings: ModelSettings | undefined): boolean {
  if (settings === undefined) return false;
  return MODEL_KINDS.some((kind) => {
    const active = settings[kind].active;
    return active !== null && active.state === "not_validated" && active.validation === null;
  });
}

/** How often the settings are read again while the search index is rebuilt, to follow its progress. */
const REINDEX_REFETCH_MS = 2_000;

/**
 * The model settings of the workspace. A rebuild of the search index runs
 * beside the server and ends by putting a model to use, so the answer is read
 * again every two seconds for as long as one runs. A server also validates
 * the models it starts with in the background, so the answer is read again
 * for a few minutes until each of them has a result.
 */
export function useModelSettings() {
  const api = useApi();
  return useQuery({
    queryKey: modelSettingsQueryKeys.settings,
    queryFn: () => unwrap(api.GET("/api/v1/model-settings")),
    refetchInterval: ({ state }) => {
      if (state.data?.vector_index.status === "reindexing") return REINDEX_REFETCH_MS;
      return awaitsBackgroundValidation(state.data) && state.dataUpdateCount < BACKGROUND_VALIDATION_MAX_READS
        ? BACKGROUND_VALIDATION_REFETCH_MS
        : false;
    },
  });
}
