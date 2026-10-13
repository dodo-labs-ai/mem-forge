import { describe, expect, test } from "vitest";
import { PROVIDERS, makeBinding, makeEmbeddingBinding, makeValidation } from "@/test/modelSettingsFixtures";
import { failureReason, providerLabel, stateBadge, validationFacts, validationNotes, validationSummary } from "./modelStatus";

const NOW = new Date("2026-03-02T09:05:00Z");

test("the badge names the state of the model in use, or that there is none", () => {
  expect(stateBadge(null)).toEqual({ label: "Not set up", tone: "idle" });
  expect(stateBadge(makeBinding())).toEqual({ label: "Ready", tone: "ok" });
  expect(stateBadge(makeBinding({ state: "not_validated" }))).toEqual({ label: "Not validated", tone: "warn" });
  expect(stateBadge(makeBinding({ state: "needs_revalidation" }))).toEqual({ label: "Needs revalidation", tone: "warn" });
});

describe("validationSummary", () => {
  test("a current pass says when it passed", () => {
    expect(validationSummary(makeBinding(), "active", NOW)).toEqual({
      tone: "ok",
      headline: "Passed validation 5 min ago.",
      detail: null,
    });
  });

  test("a model in use that was never validated is used as it is", () => {
    const summary = validationSummary(makeBinding({ state: "not_validated", validation: null }), "active", NOW);

    expect(summary.tone).toBe("warn");
    expect(summary.headline).toBe("Not validated yet.");
    expect(summary.detail).toContain("validates it in the background");
  });

  test("a saved edit that was never validated says only that", () => {
    expect(validationSummary(makeBinding({ state: "not_validated", validation: null }), "candidate", NOW)).toEqual({
      tone: "warn",
      headline: "Saved, but not validated yet.",
      detail: null,
    });
  });

  test("a failure gives the reason, and says a model in use stays in use", () => {
    const failed = makeBinding({
      state: "needs_revalidation",
      validation: makeValidation({ passed: false, error_code: "authentication_failed", facts: { latency_ms: 80 } }),
    });

    expect(validationSummary(failed, "active", NOW)).toEqual({
      tone: "danger",
      headline: "Validation failed 5 min ago.",
      detail: "The provider rejected the credentials. Check the API key. MemForge keeps using this model.",
    });
    expect(validationSummary(failed, "candidate", NOW).detail).toBe("The provider rejected the credentials. Check the API key.");
  });

  test("a key the server does not have is named by where the model takes it from", () => {
    const failed = (apiKey: { is_set: boolean; source: "saved" | "environment"; last4: null }) =>
      makeBinding({
        state: "not_validated",
        validation: makeValidation({ passed: false, error_code: "credential_unavailable", facts: {} }),
        api_key: apiKey,
      });

    expect(validationSummary(failed({ is_set: false, source: "environment", last4: null }), "candidate", NOW).detail).toBe(
      "The server reads the API key of this model from MEMFORGE_ENRICHMENT_API_KEY, which is not set.",
    );
    expect(validationSummary(failed({ is_set: false, source: "saved", last4: null }), "candidate", NOW).detail).toBe(
      "The saved API key is not available to the server. Enter it again.",
    );
  });

  test("a pass from before a change asks for another validation", () => {
    const stale = makeBinding({ state: "needs_revalidation", validation: makeValidation({ current: false }) });

    expect(validationSummary(stale, "active", NOW)).toEqual({
      tone: "warn",
      headline: "Passed validation 5 min ago, before the latest change.",
      detail: "The settings or the installed model library changed since then. Validate again.",
    });
  });
});

test("every failure has a reason, also one the page has no words for", () => {
  expect(failureReason("embedding_dimension_mismatch")).toBe(
    "The model returned vectors of another size than the output dimensions set here.",
  );
  expect(failureReason("something_new")).toBe("The validation request failed.");
  expect(failureReason(null)).toBe("The validation request failed.");
});

test("validation facts are read by type, and anything else is left out", () => {
  const validation = makeValidation({ facts: { latency_ms: "fast", dimension: 768, provider_model: "", token_limits_known: false } });

  expect(validationFacts(validation)).toEqual({ dimension: 768, token_limits_known: false });
});

describe("validationNotes", () => {
  test("a generation model without known token limits needs them set on the server", () => {
    const notes = validationNotes("generation", makeValidation({ facts: { token_limits_known: false, schema_fallback: true } }));

    expect(notes).toHaveLength(2);
    expect(notes[0]).toContain("MEMFORGE_LLM_MAX_INPUT_TOKENS");
    expect(notes[1]).toContain("JSON as text");
  });

  test("there is nothing to note for a plain pass, a failure or an embedding model", () => {
    expect(validationNotes("generation", makeValidation())).toEqual([]);
    expect(validationNotes("generation", makeValidation({ passed: false, facts: { token_limits_known: false } }))).toEqual([]);
    expect(validationNotes("embedding", makeEmbeddingBinding().validation)).toEqual([]);
    expect(validationNotes("generation", null)).toEqual([]);
  });
});

test("a provider is named by its label, or by its id when the server does not list it", () => {
  expect(providerLabel("openai_compatible", PROVIDERS)).toBe("OpenAI-compatible endpoint");
  expect(providerLabel("acme", PROVIDERS)).toBe("acme");
});
