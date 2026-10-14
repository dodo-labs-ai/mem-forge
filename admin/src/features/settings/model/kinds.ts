import type { ModelKind } from "./types";

/** The card title of each kind of model. */
export const KIND_TITLES: Record<ModelKind, string> = {
  generation: "Generation model",
  embedding: "Embedding model",
};

export const KIND_DESCRIPTIONS: Record<ModelKind, string> = {
  generation: "Reads source content and writes memories, and summarizes agent sessions.",
  embedding: "Turns memories into vectors for search.",
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
