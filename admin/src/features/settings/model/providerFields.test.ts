import { expect, test } from "vitest";
import { PROVIDERS, providerById } from "@/test/modelSettingsFixtures";
import {
  endpointField,
  keylessProviders,
  modelField,
  optionField,
  parameterField,
  parameterNames,
  providerHint,
  providersFor,
  tokenLimitField,
  tokenLimitLabel,
  tokenLimitNames,
} from "./providerFields";

test("each kind is offered the providers that serve it", () => {
  expect(providersFor("generation", PROVIDERS).map((provider) => provider.id)).toContain("anthropic");
  expect(providersFor("embedding", PROVIDERS).map((provider) => provider.id)).toEqual([
    "openai",
    "openrouter",
    "ollama",
    "openai_compatible",
    "litellm",
  ]);
});

test("the providers that send no key are the ones for a server that needs none", () => {
  expect(keylessProviders("generation", PROVIDERS).map((provider) => provider.id)).toEqual(["ollama", "openai_compatible"]);
});

test("only an embedding model takes output dimensions", () => {
  expect(parameterNames(providerById("openai"), "embedding")).toEqual(["dimensions"]);
  expect(parameterNames(providerById("openai"), "generation")).toEqual([]);
  expect(parameterNames(providerById("anthropic"), "embedding")).toEqual([]);
});

test("only a generation model states token limits, each named in tokens", () => {
  expect(tokenLimitNames(providerById("openai"), "generation")).toEqual(["context_window_tokens", "max_output_tokens"]);
  expect(tokenLimitNames(providerById("openai"), "embedding")).toEqual([]);
  expect(tokenLimitNames(providerById("anthropic"), "embedding")).toEqual([]);
  expect(tokenLimitField("context_window_tokens", { required: true })).toEqual({
    label: "Context window",
    hint: "In tokens: the most the model takes in one request, its answer included.",
  });
  expect(tokenLimitField("max_output_tokens", { required: true }).label).toBe("Maximum output");
});

test("a provider that opens the window each request asks for says the window costs the server memory", () => {
  const ollama = providerById("ollama");
  const window = tokenLimitField("context_window_tokens", {
    required: true,
    setsContextWindow: ollama.sets_context_window,
  });

  expect(window.label).toBe("Context window");
  expect(window.hint).toMatch(/the window the server opens for every request.*more of the server's memory/);
  expect(
    tokenLimitField("context_window_tokens", { required: true, setsContextWindow: providerById("openai").sets_context_window })
      .hint,
  ).toBe("In tokens: the most the model takes in one request, its answer included.");
  expect(tokenLimitField("max_output_tokens", { required: true, setsContextWindow: true }).hint).toBe(
    "In tokens: the most the model writes in one answer.",
  );
});

test("a token limit is marked optional until validation finds that the model needs it", () => {
  expect(tokenLimitField("context_window_tokens", { required: false }).label).toBe("Context window (optional)");
  expect(tokenLimitLabel("max_output_tokens")).toBe("Maximum output");
});

test("the base URL field is optional where the provider has an address of its own, and says which", () => {
  expect(endpointField(providerById("openrouter"), "api_key")).toEqual({
    label: "Base URL (optional)",
    placeholder: "https://openrouter.ai/api/v1",
    hint: "Defaults to the OpenRouter API at https://openrouter.ai/api/v1; change it only for a proxy or another server.",
    fixed: false,
  });
  expect(endpointField(providerById("litellm"), "api_key").label).toBe("Base URL (optional)");
});

test("an endpoint without an address of its own needs a base URL, and the field says what to enter", () => {
  expect(endpointField(providerById("openai_compatible"), "api_key")).toEqual({
    label: "Base URL",
    placeholder: "https://api.example.com/v1",
    hint: "The address the API's paths start from. A /chat/completions or /models path at its end is left out.",
    fixed: false,
  });
});

test("the base URL is fixed while credentials come from the server environment", () => {
  const openai = endpointField(providerById("openai"), "environment");
  expect(openai.fixed).toBe(true);
  expect(openai.hint).toContain("requests go to https://api.openai.com/v1 only");

  expect(endpointField(providerById("litellm"), "environment").hint).toBe(
    "With credentials from the server environment, the route decides where requests go.",
  );
});

test("the model field asks for a route where the provider takes one", () => {
  expect(modelField(providerById("litellm"))).toMatchObject({ label: "Model route", placeholder: "provider/model" });
  expect(modelField(providerById("openai"))).toEqual({
    label: "Model",
    placeholder: "Model id",
    hint: "Type a model id, or load the provider's models and type here to filter them.",
  });
});

test("extra settings are optional, have their own labels, and an unknown one is named after itself", () => {
  expect(optionField("organization").label).toBe("Organization ID (optional)");
  expect(parameterField("dimensions")).toEqual({
    label: "Output dimensions (optional)",
    hint: "Leave blank for the model's own vector size.",
  });
  expect(optionField("api_version")).toEqual({ label: "Api version (optional)", hint: null });
});

test("only the provider that takes any LiteLLM route says under its name what it reaches", () => {
  expect(providerHint(providerById("litellm"))).toBe(
    "Any provider LiteLLM supports that is not listed above. Enter the model as a LiteLLM route, such as bedrock/… or vertex_ai/…, and give the credentials that provider needs.",
  );
  expect(PROVIDERS.filter((provider) => providerHint(provider) !== null).map((provider) => provider.id)).toEqual(["litellm"]);
  expect(providerHint(undefined)).toBeNull();
});
