import { describe, expect, test } from "vitest";
import {
  changedEndpoints,
  endpointList,
  keyIntent,
  probeRequest,
  settingsSchema,
  settingsValues,
  updateRequest,
  type EndpointValues,
  type SettingsValues,
} from "./settingsForm";
import type { LlmConfig } from "./types";

const CONFIG: LlmConfig = {
  writable: true,
  enrichment_model: "anthropic/claude-sonnet-4-6",
  enrichment_base_url: "https://litellm.example.test/v1",
  enrichment_api_key: "********a91f",
  enrichment_api_key_set: true,
  enrichment_api_key_last4: "a91f",
  embedding_model: null,
  embedding_base_url: null,
  embedding_api_key: null,
  embedding_api_key_set: false,
  embedding_api_key_last4: null,
};

const SAVED = settingsValues(CONFIG);

function edit(kind: keyof SettingsValues, change: Partial<EndpointValues>): SettingsValues {
  return { ...SAVED, [kind]: { ...SAVED[kind], ...change } };
}

describe("settingsValues", () => {
  test("starts from the stored values with no typed key", () => {
    expect(SAVED).toEqual({
      enrichment: {
        baseUrl: "https://litellm.example.test/v1",
        model: "anthropic/claude-sonnet-4-6",
        apiKey: "",
        removeSavedKey: false,
      },
      embedding: { baseUrl: "", model: "", apiKey: "", removeSavedKey: false },
    });
  });
});

describe("keyIntent", () => {
  test("a typed key replaces the saved one, even after Remove", () => {
    expect(keyIntent({ ...SAVED.enrichment, apiKey: "sk-new", removeSavedKey: true })).toBe("replace");
  });

  test("whitespace is not a key", () => {
    expect(keyIntent({ ...SAVED.enrichment, apiKey: "   " })).toBe("keep");
  });

  test("Remove without a typed key removes the saved key", () => {
    expect(keyIntent({ ...SAVED.enrichment, removeSavedKey: true })).toBe("remove");
  });
});

describe("changedEndpoints", () => {
  test("names only the cards that differ from what is stored", () => {
    expect(changedEndpoints(SAVED, SAVED)).toEqual([]);
    expect(changedEndpoints(SAVED, edit("embedding", { model: "nomic-embed-text" }))).toEqual(["embedding"]);
  });

  test("ignores surrounding whitespace", () => {
    expect(changedEndpoints(SAVED, edit("enrichment", { baseUrl: " https://litellm.example.test/v1 " }))).toEqual([]);
  });

  test("counts a key change as a change", () => {
    expect(changedEndpoints(SAVED, edit("enrichment", { removeSavedKey: true }))).toEqual(["enrichment"]);
    expect(changedEndpoints(SAVED, edit("embedding", { apiKey: "sk-embed" }))).toEqual(["embedding"]);
  });
});

test("endpointList joins card names in a sentence", () => {
  expect(endpointList(["enrichment"])).toBe("Enrichment");
  expect(endpointList(["enrichment", "embedding"])).toBe("Enrichment and Embedding");
});

describe("updateRequest", () => {
  test("sends only the changed card, trimmed, and leaves its saved key alone", () => {
    const current = edit("enrichment", { model: "  openai/gpt-5.4 " });
    expect(updateRequest(SAVED, current)).toEqual({
      enrichment_base_url: "https://litellm.example.test/v1",
      enrichment_model: "openai/gpt-5.4",
    });
  });

  test("sends a replaced key and an empty string for a removed key", () => {
    const current: SettingsValues = {
      enrichment: { ...SAVED.enrichment, removeSavedKey: true },
      embedding: { ...SAVED.embedding, apiKey: " sk-embed " },
    };
    expect(updateRequest(SAVED, current)).toEqual({
      enrichment_base_url: "https://litellm.example.test/v1",
      enrichment_model: "anthropic/claude-sonnet-4-6",
      enrichment_api_key: "",
      embedding_base_url: "",
      embedding_model: "",
      embedding_api_key: "sk-embed",
    });
  });

  test("is empty when nothing changed", () => {
    expect(updateRequest(SAVED, SAVED)).toEqual({});
  });
});

describe("probeRequest", () => {
  test("tests with the saved key unless the user typed or removed one", () => {
    expect(probeRequest("enrichment", SAVED.enrichment)).toEqual({
      kind: "enrichment",
      base_url: "https://litellm.example.test/v1",
      api_key: null,
    });
    expect(probeRequest("enrichment", { ...SAVED.enrichment, apiKey: "sk-new" }).api_key).toBe("sk-new");
    expect(probeRequest("enrichment", { ...SAVED.enrichment, removeSavedKey: true }).api_key).toBe("");
  });
});

describe("settingsSchema", () => {
  test("accepts an empty or http(s) base URL and rejects anything else", () => {
    expect(settingsSchema.safeParse(SAVED).success).toBe(true);
    expect(settingsSchema.safeParse(edit("embedding", { baseUrl: "localhost:11434" })).success).toBe(false);
    expect(settingsSchema.safeParse(edit("embedding", { baseUrl: "ftp://models.example.test" })).success).toBe(false);
  });
});
