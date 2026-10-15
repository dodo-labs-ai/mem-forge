import { describe, expect, test } from "vitest";
import {
  PROVIDERS,
  makeBinding,
  makeEmbeddingBinding,
  makeEnvironmentBinding,
  makeValidation,
  providerById,
} from "@/test/modelSettingsFixtures";
import {
  API_KEY_REQUIRED_MESSAGE,
  INVALID_URL_MESSAGE,
  MODEL_REQUIRED_MESSAGE,
  MODEL_ROUTE_MESSAGE,
  URL_CREDENTIALS_MESSAGE,
  WHOLE_NUMBER_MESSAGE,
  blankValues,
  candidateRequest,
  discoveryRequest,
  environmentEndpointConflict,
  environmentEndpointMessage,
  fieldProblems,
  formOrigin,
  hasTokenLimitValues,
  isChanged,
  listedTokenLimits,
  offersTokenLimits,
  savedValues,
  tokenLimitsForChoice,
  tokenLimitsForTypedModel,
  valuesForProvider,
  type ModelFormValues,
} from "./modelForm";
import type { SavedKey } from "./savedKey";

const OPENAI = providerById("openai");
const OLLAMA = providerById("ollama");
const COMPATIBLE = providerById("openai_compatible");
const LITELLM = providerById("litellm");

const SAVED = savedValues(makeBinding(), "generation", PROVIDERS);
/** A route the server's environment variables pair with a base URL and no key. */
const FROM_ENVIRONMENT = savedValues(makeEnvironmentBinding(), "generation", PROVIDERS);
const KEPT: SavedKey = { status: "kept", source: "saved", last4: "a91f" };

function edited(change: Partial<ModelFormValues>, base: ModelFormValues = SAVED): ModelFormValues {
  return { ...base, ...change };
}

describe("savedValues", () => {
  test("shows a stored model with an empty key field", () => {
    expect(SAVED).toEqual({
      provider: "openai",
      endpoint: "https://api.openai.com/v1",
      authStrategy: "api_key",
      apiKey: "",
      model: "gpt-5.4",
      options: {},
      parameters: {},
      tokenLimits: {},
    });
  });

  test("shows extra settings and token limits as the text of their fields", () => {
    const stored = makeEmbeddingBinding({ provider: "openai", options: { organization: "org-1" }, parameters: { dimensions: 256 } });
    const local = makeBinding({ token_limits: { context_window_tokens: 32768, max_output_tokens: 4096 } });

    expect(savedValues(stored, "embedding", PROVIDERS)).toMatchObject({
      options: { organization: "org-1" },
      parameters: { dimensions: "256" },
    });
    expect(savedValues(local, "generation", PROVIDERS).tokenLimits).toEqual({
      context_window_tokens: "32768",
      max_output_tokens: "4096",
    });
  });

  test("starts from the first provider of the kind when nothing is stored", () => {
    expect(savedValues(null, "generation", PROVIDERS)).toEqual(blankValues(OPENAI));
    expect(blankValues(OLLAMA)).toMatchObject({ endpoint: "http://localhost:11434", authStrategy: "none", model: "" });
  });
});

test("choosing another provider starts a blank form, and choosing the saved one restores it", () => {
  expect(valuesForProvider(COMPATIBLE, SAVED)).toEqual(blankValues(COMPATIBLE));
  expect(valuesForProvider(OPENAI, SAVED)).toBe(SAVED);
});

describe("isChanged", () => {
  test("ignores surrounding whitespace, a trailing slash and a blank base URL that means the provider's own", () => {
    expect(isChanged(SAVED, edited({ model: " gpt-5.4 " }), OPENAI, "generation")).toBe(false);
    expect(isChanged(SAVED, edited({ endpoint: "https://api.openai.com/v1/" }), OPENAI, "generation")).toBe(false);
    expect(isChanged(SAVED, edited({ endpoint: "" }), OPENAI, "generation")).toBe(false);
  });

  test("sees a new model, base URL, provider, way of authenticating or extra setting", () => {
    expect(isChanged(SAVED, edited({ model: "gpt-5.4-mini" }), OPENAI, "generation")).toBe(true);
    expect(isChanged(SAVED, edited({ endpoint: "https://proxy.example.test/v1" }), OPENAI, "generation")).toBe(true);
    expect(isChanged(SAVED, blankValues(OLLAMA), OLLAMA, "generation")).toBe(true);
    expect(isChanged(SAVED, edited({ authStrategy: "environment" }), OPENAI, "generation")).toBe(true);
    expect(isChanged(SAVED, edited({ options: { organization: "org-1" } }), OPENAI, "generation")).toBe(true);
  });

  test("counts giving up the base URL of a model with environment credentials as a change", () => {
    expect(isChanged(FROM_ENVIRONMENT, FROM_ENVIRONMENT, LITELLM, "generation")).toBe(false);
    expect(isChanged(FROM_ENVIRONMENT, edited({ endpoint: "" }, FROM_ENVIRONMENT), LITELLM, "generation")).toBe(true);
  });

  test("counts a typed key as a change, but only when a key is used", () => {
    expect(isChanged(SAVED, edited({ apiKey: "sk-new" }), OPENAI, "generation")).toBe(true);
    const keyless = blankValues(OLLAMA);
    expect(isChanged(keyless, edited({ apiKey: "sk-left-over" }, keyless), OLLAMA, "generation")).toBe(false);
  });
});

describe("candidateRequest", () => {
  test("sends a typed key and leaves a blank one out, so the saved key stays", () => {
    expect(candidateRequest(SAVED, OPENAI, "generation")).toEqual({
      provider: "openai",
      model_id: "gpt-5.4",
      endpoint: "https://api.openai.com/v1",
      auth_strategy: "api_key",
      options: {},
      parameters: {},
      token_limits: {},
    });
    expect(candidateRequest(edited({ apiKey: " sk-new " }), OPENAI, "generation").api_key).toBe("sk-new");
  });

  test("sends no base URL when the field is blank", () => {
    expect(candidateRequest(edited({ endpoint: " " }), OPENAI, "generation").endpoint).toBeNull();
    expect(candidateRequest(edited({ endpoint: "" }, FROM_ENVIRONMENT), LITELLM, "generation")).toMatchObject({
      endpoint: null,
      auth_strategy: "environment",
    });
  });

  test("sends the base URL the form shows, whatever the way of authenticating", () => {
    expect(candidateRequest(edited({ authStrategy: "environment" }), OPENAI, "generation").endpoint).toBe(
      "https://api.openai.com/v1",
    );
    expect(candidateRequest(edited({ model: "openai/gpt-5.4-mini" }, FROM_ENVIRONMENT), LITELLM, "generation").endpoint).toBe(
      "http://127.0.0.1:4000/v1",
    );
  });

  test("sends the extra settings the provider and the kind accept, with numbers as numbers", () => {
    const values = edited({ options: { organization: " org-1 " }, parameters: { dimensions: "256" } });

    expect(candidateRequest(values, OPENAI, "embedding")).toMatchObject({
      options: { organization: "org-1" },
      parameters: { dimensions: 256 },
    });
    expect(candidateRequest(values, OPENAI, "generation").parameters).toEqual({});
    expect(candidateRequest(edited({ ...values, provider: "ollama" }), OLLAMA, "embedding").options).toEqual({});
  });

  test("sends the token limits that are filled in as numbers, for a generation model only", () => {
    const values = edited({ tokenLimits: { context_window_tokens: " 32768 ", max_output_tokens: "" } });

    expect(candidateRequest(values, OPENAI, "generation").token_limits).toEqual({ context_window_tokens: 32768 });
    expect(candidateRequest(values, OPENAI, "embedding").token_limits).toEqual({});
  });
});

describe("offersTokenLimits", () => {
  test("keeps the fields out of the way for a model the server has limits for", () => {
    expect(offersTokenLimits(makeBinding())).toBe(false);
    expect(offersTokenLimits(makeBinding({ state: "not_validated", validation: null }))).toBe(false);
    expect(offersTokenLimits(null)).toBe(false);
  });

  test("offers them once a validation found that the server has none", () => {
    const refused = makeValidation({ passed: false, error_code: "token_limits_required", facts: { token_limits_known: false } });

    expect(offersTokenLimits(makeBinding({ state: "not_validated", validation: refused }))).toBe(true);
  });

  test("offers them for a model that states limits, so they can be changed", () => {
    expect(offersTokenLimits(makeBinding({ token_limits: { max_output_tokens: 4096 } }))).toBe(true);
  });
});

describe("token limits from a model list", () => {
  const NAMES = ["context_window_tokens", "max_output_tokens"];
  const HAIKU = { context_window_tokens: "1000000", max_output_tokens: "128000" };
  const EMPTY = { context_window_tokens: "", max_output_tokens: "" };

  test("are the limits the listing gives, among the fields the form has", () => {
    expect(
      listedTokenLimits({ id: "anthropic/claude-haiku-5.5", context_window_tokens: 1000000, max_output_tokens: 128000 }, NAMES),
    ).toEqual(HAIKU);
    expect(listedTokenLimits({ id: "openai/gpt-5-mini", context_window_tokens: 400000, max_output_tokens: null }, NAMES)).toEqual({
      context_window_tokens: "400000",
    });
    expect(listedTokenLimits({ id: "text-embedding-3-small", context_window_tokens: 8192 }, [])).toEqual({});
  });

  test("fill empty fields", () => {
    expect(tokenLimitsForChoice({}, NAMES, null, HAIKU)).toEqual(HAIKU);
    expect(tokenLimitsForChoice(EMPTY, NAMES, null, { context_window_tokens: "400000" })).toEqual({
      context_window_tokens: "400000",
      max_output_tokens: "",
    });
  });

  test("replace what an earlier choice filled in, and clear it for a model the listing gives none for", () => {
    const earlier = { context_window_tokens: "400000", max_output_tokens: "" };

    expect(tokenLimitsForChoice(earlier, NAMES, earlier, HAIKU)).toEqual(HAIKU);
    expect(tokenLimitsForChoice(HAIKU, NAMES, HAIKU, {})).toEqual(EMPTY);
  });

  test("leave limits the user entered alone", () => {
    const typed = { context_window_tokens: "32768", max_output_tokens: "" };

    expect(tokenLimitsForChoice(typed, NAMES, null, HAIKU)).toBeNull();
    expect(tokenLimitsForChoice(typed, NAMES, HAIKU, HAIKU)).toBeNull();
    expect(tokenLimitsForChoice(EMPTY, NAMES, null, {})).toBeNull();
  });

  test("clear when the user types another model than the one they were listed for", () => {
    const filled = { model: "anthropic/claude-haiku-5.5", limits: HAIKU };

    expect(tokenLimitsForTypedModel(HAIKU, NAMES, filled, "mistral/my-own-model")).toEqual(EMPTY);
    expect(tokenLimitsForTypedModel(HAIKU, NAMES, filled, "")).toEqual(EMPTY);
  });

  test("stay while the Model field names the listed model, or when the user entered them", () => {
    const filled = { model: "anthropic/claude-haiku-5.5", limits: HAIKU };
    const edited = { ...HAIKU, max_output_tokens: "64000" };

    expect(tokenLimitsForTypedModel(HAIKU, NAMES, filled, " anthropic/claude-haiku-5.5 ")).toBeNull();
    expect(tokenLimitsForTypedModel(edited, NAMES, filled, "mistral/my-own-model")).toBeNull();
    expect(tokenLimitsForTypedModel(HAIKU, NAMES, null, "mistral/my-own-model")).toBeNull();
  });

  test("count as values only when one is filled in", () => {
    expect(hasTokenLimitValues(EMPTY)).toBe(false);
    expect(hasTokenLimitValues({ max_output_tokens: " 4096 " })).toBe(true);
  });
});

test("listing models uses the typed key, or leaves the choice of a saved one to the server", () => {
  expect(discoveryRequest(edited({ apiKey: "sk-new" }))).toEqual({
    provider: "openai",
    endpoint: "https://api.openai.com/v1",
    auth_strategy: "api_key",
    api_key: "sk-new",
  });
  expect(discoveryRequest(SAVED).api_key).toBeNull();
  expect(discoveryRequest(blankValues(OLLAMA))).toMatchObject({ auth_strategy: "none", api_key: null });
});

describe("environmentEndpointConflict", () => {
  test("is the base URL that credentials from the server environment may not be saved with", () => {
    expect(environmentEndpointConflict(FROM_ENVIRONMENT, LITELLM)).toBe("http://127.0.0.1:4000/v1");
    expect(environmentEndpointConflict(edited({ authStrategy: "environment", endpoint: "https://proxy.example.test/v1/" }), OPENAI)).toBe(
      "https://proxy.example.test/v1",
    );
  });

  test("is nothing for the provider's own address, a blank base URL or another way of authenticating", () => {
    expect(environmentEndpointConflict(edited({ authStrategy: "environment" }), OPENAI)).toBeNull();
    expect(environmentEndpointConflict(edited({ authStrategy: "environment", endpoint: "" }), OPENAI)).toBeNull();
    expect(environmentEndpointConflict(edited({ endpoint: "" }, FROM_ENVIRONMENT), LITELLM)).toBeNull();
    expect(environmentEndpointConflict(edited({ authStrategy: "api_key" }, FROM_ENVIRONMENT), LITELLM)).toBeNull();
  });
});

describe("formOrigin", () => {
  test("is the origin of the base URL, or of the provider's own address when the field is blank", () => {
    expect(formOrigin(edited({ endpoint: "https://proxy.example.test:8443/v1" }), OPENAI)).toBe("https://proxy.example.test:8443");
    expect(formOrigin(edited({ endpoint: "" }), OPENAI)).toBe("https://api.openai.com");
  });

  test("is the route's provider for a route without a base URL", () => {
    expect(formOrigin(edited({ model: "bedrock/claude" }, blankValues(LITELLM)), LITELLM)).toBe("provider:bedrock");
  });
});

describe("fieldProblems", () => {
  function problems(values: ModelFormValues, provider = OPENAI, savedKey: SavedKey = KEPT) {
    return fieldProblems(values, { provider, kind: "embedding", savedKey });
  }

  test("accepts a stored model as it is", () => {
    expect(problems(SAVED)).toEqual([]);
  });

  test("asks for a model, and for a provider/model route where the provider takes one", () => {
    expect(problems(edited({ model: " " }))).toEqual([{ path: ["model"], message: MODEL_REQUIRED_MESSAGE }]);
    const route = (model: string) => problems(edited({ model, apiKey: "sk" }, blankValues(LITELLM)), LITELLM);
    expect(route("gpt-5.4")).toEqual([{ path: ["model"], message: MODEL_ROUTE_MESSAGE }]);
    expect(route("azure/")).toEqual([{ path: ["model"], message: MODEL_ROUTE_MESSAGE }]);
    expect(route("azure/my-deployment")).toEqual([]);
  });

  test("asks for a base URL only where the provider has no address of its own", () => {
    const blank = edited({ model: "local-model", apiKey: "sk" }, blankValues(COMPATIBLE));
    expect(problems(blank, COMPATIBLE)).toEqual([{ path: ["endpoint"], message: "Enter the base URL of the endpoint." }]);
    expect(problems(edited({ endpoint: "" }))).toEqual([]);
  });

  test("refuses a base URL that is not HTTP(S) or that carries a user name or password", () => {
    expect(problems(edited({ endpoint: "localhost:11434" }))).toContainEqual({ path: ["endpoint"], message: INVALID_URL_MESSAGE });
    expect(problems(edited({ endpoint: "https://user:secret@api.openai.com/v1" }))).toEqual([
      { path: ["endpoint"], message: URL_CREDENTIALS_MESSAGE },
    ]);
  });

  test("refuses a base URL together with credentials from the server environment until one is given up", () => {
    expect(problems(FROM_ENVIRONMENT, LITELLM)).toEqual([{ path: ["endpoint"], message: environmentEndpointMessage(LITELLM) }]);
    expect(environmentEndpointMessage(LITELLM)).toBe(
      "Credentials from the server environment cannot be saved with a base URL. Remove the base URL, or choose another way of authenticating.",
    );
    const elsewhere = edited({ authStrategy: "environment", endpoint: "https://proxy.example.test/v1" });
    expect(problems(elsewhere)).toEqual([
      {
        path: ["endpoint"],
        message:
          "Credentials from the server environment go to https://api.openai.com/v1 only. Use that address, or choose another way of authenticating.",
      },
    ]);
    expect(problems(edited({ endpoint: "" }, FROM_ENVIRONMENT), LITELLM)).toEqual([]);
    expect(problems(edited({ authStrategy: "api_key", apiKey: "sk-proxy" }, FROM_ENVIRONMENT), LITELLM)).toEqual([]);
  });

  test("asks for a key when none is saved, or when the server does not have the one the model names", () => {
    expect(problems(SAVED, OPENAI, { status: "none" })).toEqual([{ path: ["apiKey"], message: API_KEY_REQUIRED_MESSAGE }]);
    expect(problems(SAVED, OPENAI, { status: "unavailable", source: "environment", kind: "generation" })).toEqual([
      { path: ["apiKey"], message: API_KEY_REQUIRED_MESSAGE },
    ]);
    expect(problems(edited({ apiKey: "sk-new" }), OPENAI, { status: "none" })).toEqual([]);
    expect(problems(edited({ authStrategy: "environment" }), OPENAI, { status: "none" })).toEqual([]);
  });

  test("asks for the key again when the base URL moved to another origin", () => {
    const moved = edited({ endpoint: "https://proxy.example.test/v1" });

    expect(problems(moved, OPENAI, { status: "other_origin", origin: "https://api.openai.com" })).toEqual([
      {
        path: ["apiKey"],
        message: "Enter the API key for https://proxy.example.test. The saved key belongs to https://api.openai.com.",
      },
    ]);
  });

  test("takes only whole numbers greater than zero as output dimensions", () => {
    const dimensions = (value: string) => problems(edited({ parameters: { dimensions: value } }));
    expect(dimensions("256")).toEqual([]);
    expect(dimensions("")).toEqual([]);
    for (const value of ["0", "-1", "1.5", "many"]) {
      expect(dimensions(value)).toEqual([{ path: ["parameters", "dimensions"], message: WHOLE_NUMBER_MESSAGE }]);
    }
  });

  test("takes only whole numbers greater than zero as token limits", () => {
    const limit = (value: string) =>
      fieldProblems(edited({ tokenLimits: { context_window_tokens: value } }), { provider: OPENAI, kind: "generation", savedKey: KEPT });
    expect(limit("32768")).toEqual([]);
    expect(limit("")).toEqual([]);
    for (const value of ["0", "32k", "4096.5"]) {
      expect(limit(value)).toEqual([{ path: ["tokenLimits", "context_window_tokens"], message: WHOLE_NUMBER_MESSAGE }]);
    }
  });

  test("asks for a provider when the stored one is not offered", () => {
    expect(fieldProblems(SAVED, { provider: undefined, kind: "generation", savedKey: KEPT })).toEqual([
      { path: ["provider"], message: "Choose a provider." },
    ]);
  });
});
