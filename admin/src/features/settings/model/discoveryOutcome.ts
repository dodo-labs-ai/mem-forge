import type { ModelDiscoveryResult } from "./types";

/** `ok`: connected and listed models; `warn`: connected without a model list; `danger`: failed. */
export type DiscoveryTone = "ok" | "warn" | "danger";

export interface DiscoveryOutcome {
  tone: DiscoveryTone;
  message: string;
  models: string[];
  /** A base URL that reaches the host from inside the API container, when the server suggests one. */
  suggestedEndpoint: string | null;
}

export function discoveryOutcome(result: ModelDiscoveryResult): DiscoveryOutcome {
  const models = (result.models ?? []).map((model) => model.id);
  const tone: DiscoveryTone = !result.ok ? "danger" : result.models_supported && models.length > 0 ? "ok" : "warn";
  const latency = result.latency_ms ?? null;
  return {
    tone,
    message: latency === null ? result.message : `${result.message} (${latency} ms)`,
    models,
    suggestedEndpoint: result.suggested_endpoint ?? null,
  };
}

/** A request that could not reach the Admin API at all. */
export function discoveryFailure(message: string): DiscoveryOutcome {
  return { tone: "danger", message, models: [], suggestedEndpoint: null };
}
