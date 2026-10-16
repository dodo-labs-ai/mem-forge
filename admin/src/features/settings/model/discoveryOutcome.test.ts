import { expect, test } from "vitest";
import { discoveryFailure, discoveryOutcome, namedListedModel, onlyListedModel } from "./discoveryOutcome";

test("an endpoint that listed models of the kind succeeds and reports the latency", () => {
  expect(
    discoveryOutcome({
      ok: true,
      models_supported: true,
      kind_filtered: true,
      models: [{ id: "gpt-5.4", context_window_tokens: 400000, max_output_tokens: 128000 }],
      message: "Connected. Found 1 generation model.",
      latency_ms: 182,
    }),
  ).toEqual({
    tone: "ok",
    message: "Connected. Found 1 generation model. (182 ms)",
    models: [{ id: "gpt-5.4", context_window_tokens: 400000, max_output_tokens: 128000 }],
    kindFiltered: true,
    suggestedEndpoint: null,
  });
});

test("a connection without a model list is a warning", () => {
  const outcome = discoveryOutcome({
    ok: true,
    models_supported: false,
    kind_filtered: false,
    message: "Connected, but this endpoint did not return a model list.",
    latency_ms: 96,
  });
  expect(outcome.tone).toBe("warn");
  expect(outcome.models).toEqual([]);
});

test("a listing with no model of the kind is a warning, not a list of the other models", () => {
  const outcome = discoveryOutcome({
    ok: true,
    models_supported: true,
    kind_filtered: true,
    models: [],
    message: "Connected, but this endpoint listed no embedding models. Type the model id if you know it.",
  });
  expect(outcome.tone).toBe("warn");
  expect(outcome.models).toEqual([]);
});

test("a failure keeps the server's message and its suggested base URL", () => {
  expect(
    discoveryOutcome({
      ok: false,
      models_supported: false,
      kind_filtered: false,
      stage: "connect",
      error_code: "provider_unreachable",
      message: "Could not reach localhost.",
      suggested_endpoint: "http://host.docker.internal:11434",
    }),
  ).toEqual({
    tone: "danger",
    message: "Could not reach localhost.",
    models: [],
    kindFiltered: false,
    suggestedEndpoint: "http://host.docker.internal:11434",
  });
});

test("only the one listed model of the kind is put in the field by itself", () => {
  const listing = (kindFiltered: boolean, ...ids: string[]) =>
    discoveryOutcome({
      ok: true,
      models_supported: true,
      kind_filtered: kindFiltered,
      models: ids.map((id) => ({ id })),
      message: "Connected.",
    });

  expect(onlyListedModel(listing(true, "qwen3:14b"))).toEqual({ id: "qwen3:14b" });
  expect(onlyListedModel(listing(true, "qwen3:14b", "llama3.3"))).toBeUndefined();
  expect(onlyListedModel(listing(false, "served-model"))).toBeUndefined();
});

test("a model field names a listed model by its exact id, without surrounding spaces", () => {
  const outcome = discoveryOutcome({
    ok: true,
    models_supported: true,
    kind_filtered: true,
    models: [{ id: "nomic-embed-text" }, { id: "nomic-embed-text-v2" }],
    message: "Connected. Found 2 embedding models.",
  });

  expect(namedListedModel(outcome, " nomic-embed-text ")).toEqual({ id: "nomic-embed-text" });
  expect(namedListedModel(outcome, "nomic-embed")).toBeUndefined();
  expect(namedListedModel(outcome, "")).toBeUndefined();
  expect(namedListedModel(null, "nomic-embed-text")).toBeUndefined();
  expect(namedListedModel(discoveryFailure("Network error"), "nomic-embed-text")).toBeUndefined();
});
