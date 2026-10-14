import { expect, test } from "vitest";
import { discoveryOutcome } from "./discoveryOutcome";

test("an endpoint that listed models succeeds and reports the latency", () => {
  expect(
    discoveryOutcome({
      ok: true,
      models_supported: true,
      models: [{ id: "gpt-5.4" }],
      message: "Connected. Found 1 model.",
      latency_ms: 182,
    }),
  ).toEqual({ tone: "ok", message: "Connected. Found 1 model. (182 ms)", models: ["gpt-5.4"], suggestedEndpoint: null });
});

test("a connection without a model list is a warning", () => {
  const outcome = discoveryOutcome({
    ok: true,
    models_supported: false,
    message: "Connected, but this endpoint did not return a model list.",
    latency_ms: 96,
  });
  expect(outcome.tone).toBe("warn");
  expect(outcome.models).toEqual([]);
});

test("a failure keeps the server's message and its suggested base URL", () => {
  expect(
    discoveryOutcome({
      ok: false,
      models_supported: false,
      stage: "connect",
      error_code: "provider_unreachable",
      message: "Could not reach localhost.",
      suggested_endpoint: "http://host.docker.internal:11434",
    }),
  ).toEqual({
    tone: "danger",
    message: "Could not reach localhost.",
    models: [],
    suggestedEndpoint: "http://host.docker.internal:11434",
  });
});
