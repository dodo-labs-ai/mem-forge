import { expect, test } from "vitest";
import { MAX_SHOWN_SUGGESTIONS, modelSuggestionList, preferredModel, suggestedModels } from "./modelSuggestions";

const MODELS = ["anthropic/claude-sonnet-4-6", "text-embedding-3-small", "openai/gpt-5.4", "nomic-embed-text"];

test("each kind is offered the models that fit it", () => {
  expect(suggestedModels("generation", MODELS)).toEqual(["anthropic/claude-sonnet-4-6", "openai/gpt-5.4"]);
  expect(suggestedModels("embedding", MODELS)).toEqual(["text-embedding-3-small", "nomic-embed-text"]);
});

test("every model is offered when none fits", () => {
  expect(suggestedModels("embedding", ["gpt-5.4"])).toEqual(["gpt-5.4"]);
});

test("the first fitting model is preferred", () => {
  expect(preferredModel("embedding", MODELS)).toBe("text-embedding-3-small");
  expect(preferredModel("embedding", [])).toBeUndefined();
});

test("the summary says whether the list was narrowed to the kind", () => {
  expect(modelSuggestionList("generation", MODELS, "").summary).toBe(
    "2 generation suggestions from 4 listed models. Pick one or type any model id.",
  );
  expect(modelSuggestionList("embedding", ["nomic-embed-text"], "").summary).toBe(
    "1 model available from this endpoint. Pick one or type any model id.",
  );
});

test("typing part of a name narrows the suggestions", () => {
  const list = modelSuggestionList("generation", MODELS, "GPT");

  expect(list.shown).toEqual(["openai/gpt-5.4"]);
  expect(list.summary).toBe(
    "2 generation suggestions from 4 listed models. Showing 1; type part of a name to narrow the list. Pick one or type any model id.",
  );
});

test("a whole model name, or text that matches nothing, keeps every suggestion", () => {
  expect(modelSuggestionList("generation", MODELS, "openai/gpt-5.4").shown).toHaveLength(2);
  expect(modelSuggestionList("generation", MODELS, "my-own-model").shown).toHaveLength(2);
});

test("a long list is cut to a readable number", () => {
  const many = Array.from({ length: 300 }, (_, index) => `model-${index}`);

  const list = modelSuggestionList("generation", many, "");

  expect(list.shown).toHaveLength(MAX_SHOWN_SUGGESTIONS);
  expect(list.summary).toContain(`Showing ${MAX_SHOWN_SUGGESTIONS}; type part of a name to narrow the list.`);
});
