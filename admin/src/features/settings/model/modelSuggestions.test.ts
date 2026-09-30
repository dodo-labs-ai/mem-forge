import { expect, test } from "vitest";
import { preferredModel, suggestedModels, suggestionSummary } from "./modelSuggestions";

const MODELS = ["anthropic/claude-sonnet-4-6", "text-embedding-3-small", "openai/gpt-5.4", "nomic-embed-text"];

test("each endpoint is offered the models that fit it", () => {
  expect(suggestedModels("enrichment", MODELS)).toEqual(["anthropic/claude-sonnet-4-6", "openai/gpt-5.4"]);
  expect(suggestedModels("embedding", MODELS)).toEqual(["text-embedding-3-small", "nomic-embed-text"]);
});

test("every model is offered when none fits", () => {
  expect(suggestedModels("embedding", ["gpt-5.4"])).toEqual(["gpt-5.4"]);
});

test("the first fitting model is preferred", () => {
  expect(preferredModel("embedding", MODELS)).toBe("text-embedding-3-small");
  expect(preferredModel("embedding", [])).toBeUndefined();
});

test("the summary says whether the list was narrowed", () => {
  expect(suggestionSummary("enrichment", 3, 14)).toBe(
    "3 enrichment suggestions from 14 returned models. Pick one or type any model id.",
  );
  expect(suggestionSummary("embedding", 1, 1)).toBe("1 model available from this endpoint. Pick one or type any model id.");
});
