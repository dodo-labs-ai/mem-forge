import { pluralize } from "@/lib/format";
import type { EndpointKind } from "./types";

const EMBEDDING_MODEL_PATTERN = /embed/i;

function isEmbeddingModel(model: string): boolean {
  return EMBEDDING_MODEL_PATTERN.test(model);
}

/**
 * The returned models that fit the endpoint: embedding models for Embedding,
 * the rest for Enrichment. When none fit, every model is offered.
 */
export function suggestedModels(kind: EndpointKind, models: readonly string[]): string[] {
  const fitting = models.filter((model) => isEmbeddingModel(model) === (kind === "embedding"));
  return fitting.length > 0 ? fitting : [...models];
}

/** The model to fill in when the field is still empty after a successful test. */
export function preferredModel(kind: EndpointKind, models: readonly string[]): string | undefined {
  return suggestedModels(kind, models)[0];
}

/** The sentence above the suggestion buttons. */
export function suggestionSummary(kind: EndpointKind, shown: number, total: number): string {
  const found =
    shown === total
      ? `${pluralize(total, "model")} available from this endpoint.`
      : `${pluralize(shown, `${kind} suggestion`)} from ${pluralize(total, "returned model")}.`;
  return `${found} Pick one or type any model id.`;
}
