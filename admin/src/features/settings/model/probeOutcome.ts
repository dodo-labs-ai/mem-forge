import type { LlmProbeResult } from "./types";

/** `ok`: connected and listed models; `warn`: connected without a model list; `danger`: failed. */
export type ProbeTone = "ok" | "warn" | "danger";

export interface ProbeOutcome {
  tone: ProbeTone;
  message: string;
  models: string[];
  /** A base URL that reaches the host from inside the API container, when the test suggests one. */
  suggestedBaseUrl: string | null;
}

export function probeOutcome(result: LlmProbeResult): ProbeOutcome {
  const models = (result.models ?? []).map((model) => model.id);
  const tone: ProbeTone = !result.ok ? "danger" : result.models_supported && models.length > 0 ? "ok" : "warn";
  const latency = result.latency_ms ?? null;
  return {
    tone,
    message: latency === null ? result.message : `${result.message} (${latency} ms)`,
    models,
    suggestedBaseUrl: result.suggested_base_url ?? null,
  };
}

/** A test that could not reach the Admin API at all. */
export function probeFailure(message: string): ProbeOutcome {
  return { tone: "danger", message, models: [], suggestedBaseUrl: null };
}
