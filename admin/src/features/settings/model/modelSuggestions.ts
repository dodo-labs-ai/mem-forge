import { pluralize } from "@/lib/format";
import type { ModelKind } from "./types";

const EMBEDDING_MODEL_PATTERN = /embed/i;

/** The most suggestions shown at once; a provider can list hundreds of models. */
export const MAX_SHOWN_SUGGESTIONS = 24;

function isEmbeddingModel(model: string): boolean {
  return EMBEDDING_MODEL_PATTERN.test(model);
}

/**
 * The listed models that fit the kind: embedding models for the embedding
 * model, the rest for the generation model. When none fit, every model is
 * offered.
 */
export function suggestedModels(kind: ModelKind, models: readonly string[]): string[] {
  const fitting = models.filter((model) => isEmbeddingModel(model) === (kind === "embedding"));
  return fitting.length > 0 ? fitting : [...models];
}

/** The model to fill in when the field is still empty after the endpoint listed its models. */
export function preferredModel(kind: ModelKind, models: readonly string[]): string | undefined {
  return suggestedModels(kind, models)[0];
}

/** The suggestions whose name contains what is typed; all of them once a whole name, or nothing that matches, is typed. */
function narrowedByText(suggestions: readonly string[], typed: string): readonly string[] {
  const text = typed.trim().toLowerCase();
  if (text === "" || suggestions.some((model) => model.toLowerCase() === text)) return suggestions;
  const matching = suggestions.filter((model) => model.toLowerCase().includes(text));
  return matching.length > 0 ? matching : suggestions;
}

export interface ModelSuggestionList {
  /** The models offered as buttons. */
  shown: string[];
  /** The sentence above the buttons. */
  summary: string;
}

/**
 * The models to offer for a kind out of everything an endpoint listed,
 * narrowed by what is typed in the model field and cut to a readable number.
 */
export function modelSuggestionList(kind: ModelKind, models: readonly string[], typed: string): ModelSuggestionList {
  const fitting = suggestedModels(kind, models);
  const shown = narrowedByText(fitting, typed).slice(0, MAX_SHOWN_SUGGESTIONS);
  const found =
    fitting.length === models.length
      ? `${pluralize(models.length, "model")} available from this endpoint.`
      : `${pluralize(fitting.length, `${kind} suggestion`)} from ${pluralize(models.length, "listed model")}.`;
  const narrowed = shown.length < fitting.length ? ` Showing ${shown.length}; type part of a name to narrow the list.` : "";
  return { shown, summary: `${found}${narrowed} Pick one or type any model id.` };
}
