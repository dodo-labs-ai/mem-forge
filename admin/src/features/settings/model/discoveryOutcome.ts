import type { ModelDiscoveryResult, ModelOption } from "./types";

/** `ok`: connected and listed models of the kind; `warn`: connected without any to offer; `danger`: failed. */
export type DiscoveryTone = "ok" | "warn" | "danger";

export interface DiscoveryOutcome {
  tone: DiscoveryTone;
  message: string;
  /** The listed models of the kind, or every listed model when the endpoint does not say which kind each is. */
  models: ModelOption[];
  /** Whether `models` holds only models of the kind. */
  kindFiltered: boolean;
  /** A base URL that reaches the host from inside the API container, when the server suggests one. */
  suggestedEndpoint: string | null;
}

export function discoveryOutcome(result: ModelDiscoveryResult): DiscoveryOutcome {
  const models = result.models ?? [];
  const tone: DiscoveryTone = !result.ok ? "danger" : result.models_supported && models.length > 0 ? "ok" : "warn";
  const latency = result.latency_ms ?? null;
  return {
    tone,
    message: latency === null ? result.message : `${result.message} (${latency} ms)`,
    models,
    kindFiltered: result.kind_filtered,
    suggestedEndpoint: result.suggested_endpoint ?? null,
  };
}

/** A request that could not reach the Admin API at all. */
export function discoveryFailure(message: string): DiscoveryOutcome {
  return { tone: "danger", message, models: [], kindFiltered: false, suggestedEndpoint: null };
}

/**
 * The model Load models puts in the field by itself: the one listed model of
 * the kind, when the endpoint says which kind its models are and lists exactly
 * one. With more to choose from, the user picks.
 */
export function onlyListedModel(outcome: DiscoveryOutcome): ModelOption | undefined {
  return outcome.kindFiltered && outcome.models.length === 1 ? outcome.models[0] : undefined;
}

/**
 * The listed model the model field names, however the name got there: picked
 * from the list, typed, pasted or filled in by the browser. Naming a listed
 * model counts as choosing it.
 */
export function namedListedModel(outcome: DiscoveryOutcome | null, typed: string): ModelOption | undefined {
  const id = typed.trim();
  return id === "" ? undefined : outcome?.models.find((model) => model.id === id);
}
