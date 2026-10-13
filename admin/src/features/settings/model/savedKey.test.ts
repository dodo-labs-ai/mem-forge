import { describe, expect, test } from "vitest";
import { kindSettings, makeBinding, makeEmbeddingBinding } from "@/test/modelSettingsFixtures";
import { apiKeyStatus, endpointOrigin, routeOrigin, savedKeyFor } from "./savedKey";

describe("endpointOrigin", () => {
  test("is the scheme and host, with the port only when it is not the scheme's own", () => {
    expect(endpointOrigin("https://api.openai.com/v1")).toBe("https://api.openai.com");
    expect(endpointOrigin("https://API.OpenAI.com:443/v1/")).toBe("https://api.openai.com");
    expect(endpointOrigin("http://localhost:11434")).toBe("http://localhost:11434");
    expect(endpointOrigin("http://[::1]:8080/v1")).toBe("http://[::1]:8080");
  });

  test("is null for anything that is not an HTTP(S) URL", () => {
    expect(endpointOrigin("localhost:11434")).toBeNull();
    expect(endpointOrigin("ftp://example.test")).toBeNull();
    expect(endpointOrigin("")).toBeNull();
  });
});

test("a route without a base URL belongs to its provider", () => {
  expect(routeOrigin("bedrock/claude")).toBe("provider:bedrock");
  expect(routeOrigin("")).toBeNull();
});

describe("savedKeyFor", () => {
  const inUse = makeBinding();

  test("keeps the key saved for the same origin", () => {
    expect(savedKeyFor(kindSettings(inUse), "https://api.openai.com")).toEqual({ status: "kept", source: "saved", last4: "a91f" });
  });

  test("does not offer a key saved for another origin", () => {
    expect(savedKeyFor(kindSettings(inUse), "https://proxy.example.test")).toEqual({
      status: "other_origin",
      origin: "https://api.openai.com",
    });
    expect(savedKeyFor(kindSettings(inUse), null).status).toBe("other_origin");
  });

  test("asks the saved edit before the model in use", () => {
    const edit = makeBinding({
      endpoint: "https://proxy.example.test/v1",
      origin: "https://proxy.example.test",
      api_key: { is_set: true, source: "saved", last4: "77c2" },
    });

    expect(savedKeyFor(kindSettings(inUse, edit), "https://proxy.example.test")).toMatchObject({ status: "kept", last4: "77c2" });
    expect(savedKeyFor(kindSettings(inUse, edit), "https://api.openai.com")).toMatchObject({ status: "kept", last4: "a91f" });
  });

  test("keeps no key the server does not have", () => {
    const unset = makeBinding({ api_key: { is_set: false, source: "environment", last4: null } });
    const lost = makeEmbeddingBinding({ auth_strategy: "api_key", api_key: { is_set: false, source: "saved", last4: null } });

    expect(savedKeyFor(kindSettings(unset), "https://api.openai.com")).toEqual({
      status: "unavailable",
      source: "environment",
      kind: "generation",
    });
    expect(savedKeyFor(kindSettings(lost), "http://localhost:11434")).toEqual({
      status: "unavailable",
      source: "saved",
      kind: "embedding",
    });
    expect(savedKeyFor(kindSettings(unset), "https://proxy.example.test").status).toBe("other_origin");
  });

  test("has nothing to keep when no model of the kind uses a key", () => {
    expect(savedKeyFor(kindSettings(makeEmbeddingBinding()), "http://localhost:11434")).toEqual({ status: "none" });
    expect(savedKeyFor(kindSettings(null), "https://api.openai.com")).toEqual({ status: "none" });
  });
});

describe("apiKeyStatus", () => {
  test("a kept key shows only its last four characters", () => {
    expect(apiKeyStatus({ status: "kept", source: "saved", last4: "a91f" }, false)).toEqual({
      message: "The saved key (****a91f) is used. Leave blank to keep it.",
      warning: false,
    });
    expect(apiKeyStatus({ status: "kept", source: "environment", last4: null }, false).message).toBe(
      "The key from the server environment is used. Leave blank to keep it.",
    );
  });

  test("typing over a kept key says it takes its place", () => {
    expect(apiKeyStatus({ status: "kept", source: "saved", last4: "a91f" }, true).message).toBe(
      "Takes the place of the saved key (****a91f) when you save.",
    );
  });

  test("a key saved for another origin has to be entered again", () => {
    expect(apiKeyStatus({ status: "other_origin", origin: "https://api.openai.com" }, false)).toEqual({
      message: "The saved key belongs to https://api.openai.com and is not sent anywhere else. Enter the key for this endpoint.",
      warning: true,
    });
    expect(apiKeyStatus({ status: "other_origin", origin: "https://api.openai.com" }, true).warning).toBe(false);
  });

  test("a key the server does not have has to be entered, or its variable set", () => {
    expect(apiKeyStatus({ status: "unavailable", source: "environment", kind: "generation" }, false)).toEqual({
      message:
        "MEMFORGE_ENRICHMENT_API_KEY is not set on the server. Enter a key here, or set the variable and restart the server.",
      warning: true,
    });
    expect(apiKeyStatus({ status: "unavailable", source: "environment", kind: "embedding" }, false).message).toContain(
      "MEMFORGE_EMBEDDING_API_KEY is not set",
    );
    expect(apiKeyStatus({ status: "unavailable", source: "saved", kind: "generation" }, false)).toEqual({
      message: "The saved key is not available to the server. Enter it again.",
      warning: true,
    });
    expect(apiKeyStatus({ status: "unavailable", source: "environment", kind: "generation" }, true)).toEqual({
      message: "Stored encrypted on the server and never shown again.",
      warning: false,
    });
  });

  test("a new key is stored encrypted", () => {
    expect(apiKeyStatus({ status: "none" }, false).message).toBe("Stored encrypted on the server and never shown again.");
  });
});
