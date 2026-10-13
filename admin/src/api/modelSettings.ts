import { useQuery } from "@tanstack/react-query";
import { useApi } from "./ApiProvider";
import { unwrap } from "./client";
import type { components } from "./schema.gen";

type Schemas = components["schemas"];

/** One configured model: its provider, endpoint, model and the state of its latest validation. */
export type ModelBinding = Schemas["ModelBindingResponse"];
/** The model of one kind that is in use (`active`) and the edit that waits to be validated (`candidate`). */
export type ModelKindSettings = Schemas["ModelKindSettingsResponse"];
/** The models of both kinds, with whether this installation lets Settings change them. */
export type ModelSettings = Schemas["ModelSettingsResponse"];
/** The two models MemForge calls: one writes memories, one embeds them for search. */
export type ModelKind = ModelBinding["kind"];

export const MODEL_KINDS: readonly ModelKind[] = ["generation", "embedding"];

/**
 * Query keys of the model settings. Settings and the notice in the app shell
 * read the one answer, so a model that passes validation clears the notice
 * without another request.
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

/**
 * The model settings of the workspace. A server validates the models it
 * starts with in the background, so the answer is read again for a few
 * minutes until each of them has a result.
 */
export function useModelSettings() {
  const api = useApi();
  return useQuery({
    queryKey: modelSettingsQueryKeys.settings,
    queryFn: () => unwrap(api.GET("/api/v1/model-settings")),
    refetchInterval: ({ state }) =>
      awaitsBackgroundValidation(state.data) && state.dataUpdateCount < BACKGROUND_VALIDATION_MAX_READS
        ? BACKGROUND_VALIDATION_REFETCH_MS
        : false,
  });
}
