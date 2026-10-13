import type { ModelBinding, ModelKindSettings, ModelSettings, components } from "@/api";

type ModelProvider = components["schemas"]["ModelProviderResponse"];
type ModelValidation = components["schemas"]["ModelValidationResult"];

function provider(overrides: Partial<ModelProvider> & Pick<ModelProvider, "id" | "label">): ModelProvider {
  return {
    kinds: ["generation", "embedding"],
    auth_strategies: ["api_key", "environment"],
    default_endpoint: null,
    endpoint_required: false,
    model_is_route: false,
    lists_models: true,
    options: [],
    parameters: { generation: [], embedding: ["dimensions"] },
    ...overrides,
  };
}

/** The providers a server lists, in its order. */
export const PROVIDERS: ModelProvider[] = [
  provider({ id: "openai", label: "OpenAI", default_endpoint: "https://api.openai.com/v1", options: ["organization"] }),
  provider({
    id: "anthropic",
    label: "Anthropic",
    kinds: ["generation"],
    default_endpoint: "https://api.anthropic.com",
    parameters: { generation: [] },
  }),
  provider({ id: "openrouter", label: "OpenRouter", default_endpoint: "https://openrouter.ai/api/v1" }),
  provider({ id: "ollama", label: "Ollama", auth_strategies: ["none", "api_key"], default_endpoint: "http://localhost:11434" }),
  provider({
    id: "openai_compatible",
    label: "OpenAI-compatible endpoint",
    auth_strategies: ["api_key", "none"],
    endpoint_required: true,
  }),
  provider({ id: "litellm", label: "LiteLLM route (advanced)", model_is_route: true, lists_models: false }),
];

export function providerById(id: string): ModelProvider {
  const found = PROVIDERS.find((candidate) => candidate.id === id);
  if (!found) throw new Error(`No provider fixture named ${id}`);
  return found;
}

/** A validation that passed five minutes before `2026-03-02T09:05:00Z`. */
export function makeValidation(overrides: Partial<ModelValidation> = {}): ModelValidation {
  return {
    checked_at: "2026-03-02T09:00:00Z",
    passed: true,
    error_code: null,
    facts: { latency_ms: 420, structured_output_mode: "native_schema", schema_fallback: false, token_limits_known: true },
    current: true,
    ...overrides,
  };
}

/** A ready OpenAI generation model with a saved key; override what a test needs. */
export function makeBinding(overrides: Partial<ModelBinding> = {}): ModelBinding {
  return {
    kind: "generation",
    provider: "openai",
    endpoint: "https://api.openai.com/v1",
    origin: "https://api.openai.com",
    auth_strategy: "api_key",
    model_id: "gpt-5.4",
    route: "openai/gpt-5.4",
    options: {},
    parameters: {},
    config_source: "settings",
    state: "ready",
    validation: makeValidation(),
    api_key: { is_set: true, source: "saved", last4: "a91f" },
    saved_at: "2026-03-02T08:59:00Z",
    ...overrides,
  };
}

/** A ready local Ollama embedding model that needs no key. */
export function makeEmbeddingBinding(overrides: Partial<ModelBinding> = {}): ModelBinding {
  return makeBinding({
    kind: "embedding",
    provider: "ollama",
    endpoint: "http://localhost:11434",
    origin: "http://localhost:11434",
    auth_strategy: "none",
    model_id: "nomic-embed-text",
    route: "ollama/nomic-embed-text",
    validation: makeValidation({ facts: { latency_ms: 96, dimension: 768 } }),
    api_key: { is_set: false, source: null, last4: null },
    ...overrides,
  });
}

/**
 * A generation model as the server reads it from its `MEMFORGE_*` variables
 * when they name a route and a base URL but no key: a LiteLLM route sent to
 * that base URL with the provider's credentials from the server environment.
 * Such a model is not stored, so it has no time of saving.
 */
export function makeEnvironmentBinding(overrides: Partial<ModelBinding> = {}): ModelBinding {
  return makeBinding({
    provider: "litellm",
    endpoint: "http://127.0.0.1:4000/v1",
    origin: "http://127.0.0.1:4000",
    auth_strategy: "environment",
    model_id: "openai/gpt-5.4",
    route: "openai/gpt-5.4",
    config_source: "environment",
    api_key: { is_set: false, source: null, last4: null },
    saved_at: null,
    ...overrides,
  });
}

/**
 * A generation model of a deployment that manages its models: a LiteLLM route
 * without a base URL, with the key the deployment sets for it.
 */
export function makeDeploymentBinding(overrides: Partial<ModelBinding> = {}): ModelBinding {
  return makeEnvironmentBinding({
    endpoint: null,
    origin: "provider:openai",
    auth_strategy: "api_key",
    config_source: "deployment",
    api_key: { is_set: true, source: "environment", last4: "9d1e" },
    ...overrides,
  });
}

/** A workspace with both models ready and nothing waiting to be validated. */
export function makeModelSettings(overrides: Partial<ModelSettings> = {}): ModelSettings {
  return {
    writable: true,
    litellm_version: "1.86.0",
    generation: { active: makeBinding(), candidate: null },
    embedding: { active: makeEmbeddingBinding(), candidate: null },
    ...overrides,
  };
}

/** A fresh installation: nothing configured for either kind. */
export const NO_MODELS: Pick<ModelSettings, "generation" | "embedding"> = {
  generation: { active: null, candidate: null },
  embedding: { active: null, candidate: null },
};

export function kindSettings(active: ModelBinding | null, candidate: ModelBinding | null = null): ModelKindSettings {
  return { active, candidate };
}
