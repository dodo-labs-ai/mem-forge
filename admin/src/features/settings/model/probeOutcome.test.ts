import { expect, test } from "vitest";
import { probeOutcome } from "./probeOutcome";

test("a test that listed models succeeds and reports the latency", () => {
  expect(
    probeOutcome({
      ok: true,
      models_supported: true,
      models: [{ id: "openai/gpt-5.4" }],
      message: "Connected. Found 1 model.",
      latency_ms: 182,
    }),
  ).toEqual({ tone: "ok", message: "Connected. Found 1 model. (182 ms)", models: ["openai/gpt-5.4"], suggestedBaseUrl: null });
});

test("a connection without a model list is a warning", () => {
  const outcome = probeOutcome({
    ok: true,
    models_supported: false,
    message: "Connected, but this endpoint did not return a model list.",
    latency_ms: 96,
  });
  expect(outcome.tone).toBe("warn");
  expect(outcome.models).toEqual([]);
});

test("a failed test keeps the server's message and its suggested base URL", () => {
  expect(
    probeOutcome({
      ok: false,
      models_supported: false,
      stage: "connect",
      message: "Could not reach localhost.",
      suggested_base_url: "http://host.docker.internal:11434/v1",
    }),
  ).toEqual({
    tone: "danger",
    message: "Could not reach localhost.",
    models: [],
    suggestedBaseUrl: "http://host.docker.internal:11434/v1",
  });
});
