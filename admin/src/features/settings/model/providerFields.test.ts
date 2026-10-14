import { expect, test } from "vitest";
import { PROVIDERS, providerById } from "@/test/modelSettingsFixtures";
import {
  endpointField,
  keylessProviders,
  modelField,
  optionField,
  parameterField,
  parameterNames,
  providersFor,
  tokenLimitField,
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
  expect(tokenLimitField("context_window_tokens")).toEqual({
    label: "Context window",
    hint: "In tokens: the most the model takes in one request, its answer included.",
  });
  expect(tokenLimitField("max_output_tokens").label).toBe("Maximum output");
});

test("the base URL field shows the provider's own address, or says when one is needed", () => {
  expect(endpointField(providerById("openai"), "api_key")).toMatchObject({
    label: "Base URL",
    placeholder: "https://api.openai.com/v1",
    fixed: false,
  });
  expect(endpointField(providerById("openai_compatible"), "api_key").placeholder).toBe("https://api.example.com/v1");
  expect(endpointField(providerById("litellm"), "api_key").label).toBe("Base URL (optional)");
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
  expect(modelField(providerById("openai"))).toEqual({ label: "Model", placeholder: "Model id", hint: null });
});

test("extra settings have their own labels, and an unknown one is named after itself", () => {
  expect(optionField("organization").label).toBe("Organization ID");
  expect(parameterField("dimensions").label).toBe("Output dimensions");
  expect(optionField("api_version")).toEqual({ label: "Api version", hint: "Optional." });
});
