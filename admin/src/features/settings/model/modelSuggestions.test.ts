import { expect, test } from "vitest";
import { MAX_SHOWN_SUGGESTIONS, modelSuggestionList } from "./modelSuggestions";
import type { ModelOption } from "./types";

function listed(...ids: string[]): ModelOption[] {
  return ids.map((id) => ({ id }));
}

const EMBEDDING = listed("openai/text-embedding-3-small", "openai/text-embedding-3-large", "qwen/qwen3-embedding-8b");

function ids(models: readonly ModelOption[]): string[] {
  return models.map((model) => model.id);
}

test("every listed model is offered, and the summary says how to filter them", () => {
  const list = modelSuggestionList("embedding", EMBEDDING, "", true);

  expect(ids(list.shown)).toEqual(ids(EMBEDDING));
  expect(list.summary).toBe("3 listed embedding models. Type in the Model field to filter them, or pick one.");
});

test("a listing that does not tell the kinds apart is offered as models, not as models of the kind", () => {
  expect(modelSuggestionList("embedding", listed("BAAI/bge-m3", "llama-3.1-8b"), "", false).summary).toBe(
    "2 listed models. Type in the Model field to filter them, or pick one.",
  );
});

test("typing part of an id narrows the suggestions", () => {
  const list = modelSuggestionList("embedding", EMBEDDING, "OPENAI", true);

  expect(ids(list.shown)).toEqual(["openai/text-embedding-3-small", "openai/text-embedding-3-large"]);
  expect(list.summary).toBe("2 of 3 listed embedding models match “OPENAI”. Clear the Model field to see them all.");
});

test("a whole model id still narrows the list, with that model first", () => {
  const models = listed("openai/gpt-5-mini-high", "openai/gpt-5", "openai/gpt-5-mini");

  const list = modelSuggestionList("generation", models, "openai/gpt-5-mini", true);

  expect(ids(list.shown)).toEqual(["openai/gpt-5-mini", "openai/gpt-5-mini-high"]);
  expect(list.summary).toBe("2 of 3 listed generation models match “openai/gpt-5-mini”. Clear the Model field to see them all.");
});

test("text that matches no listed model offers none and says the id is used as typed", () => {
  const list = modelSuggestionList("embedding", EMBEDDING, "my-own-model", true);

  expect(list.shown).toEqual([]);
  expect(list.summary).toBe("No listed embedding model matches “my-own-model”. The model id is used as typed.");
});

test("a long list is cut to a readable number", () => {
  const many = Array.from({ length: 300 }, (_, index) => ({ id: `model-${index}` }));

  const list = modelSuggestionList("generation", many, "", true);

  expect(list.shown).toHaveLength(MAX_SHOWN_SUGGESTIONS);
  expect(list.summary).toBe(
    `300 listed generation models. Showing the first ${MAX_SHOWN_SUGGESTIONS}. Type in the Model field to filter them, or pick one.`,
  );
  expect(modelSuggestionList("generation", many, "model-1", true).summary).toBe(
    `111 of 300 listed generation models match “model-1”. Showing the first ${MAX_SHOWN_SUGGESTIONS}. Clear the Model field to see them all.`,
  );
});
