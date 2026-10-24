import type { ModelKind } from "./types";

/** The card title of each kind of model. */
export const KIND_TITLES: Record<ModelKind, string> = {
  generation: "Generation model",
  embedding: "Embedding model",
};

export const KIND_DESCRIPTIONS: Record<ModelKind, string> = {
  generation: "Reads source content and writes memories.",
  embedding: "Turns memories into vectors for search.",
};

/** What MemForge does without a model of each kind, said on the card while none is set up. */
export const WITHOUT_A_MODEL: Record<ModelKind, string> = {
  generation: "Until one is set up, a source sync stores content but writes no memories.",
  embedding: "Until one is set up, search ranks by matching words.",
};

/** The environment variable the server reads the API key of each kind of model from. */
export const KEY_VARIABLES: Record<ModelKind, string> = {
  generation: "MEMFORGE_ENRICHMENT_API_KEY",
  embedding: "MEMFORGE_EMBEDDING_API_KEY",
};

/** "Generation model", "Generation model and Embedding model". */
export function kindList(kinds: readonly ModelKind[]): string {
  return kinds.map((kind) => KIND_TITLES[kind]).join(" and ");
}
