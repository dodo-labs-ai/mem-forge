import type { ModelBinding, ModelKindSettings, ModelSettings, VectorIndex, components } from "@/api";

type ModelProvider = components["schemas"]["ModelProviderResponse"];
type ModelValidation = components["schemas"]["ModelValidationResult"];
type EmbeddingProfile = components["schemas"]["EmbeddingProfileResponse"];
type IndexRecord = components["schemas"]["VectorIndexRecordResponse"];
type Reindex = components["schemas"]["VectorReindexResponse"];

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
    token_limits: { generation: ["context_window_tokens", "max_output_tokens"], embedding: [] },
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
    token_limits: { generation: ["context_window_tokens", "max_output_tokens"] },
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
    token_limits: {},
    config_source: "settings",
    state: "ready",
    validation: makeValidation(),
    api_key: { is_set: true, source: "saved", last4: "a91f" },
    saved_at: "2026-03-02T08:59:00Z",
    profile: null,
    waits_for_reindex: false,
    ...overrides,
  };
}

const NOMIC_FINGERPRINT = "a1".repeat(32);

/**
 * The vector space of the local Ollama embedding model: 768 numbers a vector,
 * and similarity thresholds that were tuned on another model.
 */
export function makeProfile(overrides: Partial<EmbeddingProfile> = {}): EmbeddingProfile {
  return {
    provider: "ollama",
    route: "ollama/nomic-embed-text",
    endpoint: "http://localhost:11434",
    model_id: "nomic-embed-text",
    requested_dimensions: null,
    dimension: 768,
    fingerprint: NOMIC_FINGERPRINT,
    thresholds_calibrated: false,
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
    profile: makeProfile(),
    ...overrides,
  });
}

/** The search index the local Ollama embedding model built: 1,204 memories, in use since March 1. */
export function makeIndexRecord(overrides: Partial<IndexRecord> = {}): IndexRecord {
  const profile = overrides.profile === undefined ? makeProfile() : overrides.profile;
  return {
    fingerprint: profile?.fingerprint ?? NOMIC_FINGERPRINT,
    collection: `memories-${profile?.fingerprint ?? NOMIC_FINGERPRINT}`,
    dimension: profile?.dimension ?? 768,
    adopted: false,
    vector_count: 1204,
    activated_at: "2026-03-01T10:00:00Z",
    profile,
    ...overrides,
  };
}

/** A rebuild of the search index that is running, 120 of 1,204 memories in. */
export function makeReindex(overrides: Partial<Reindex> = {}): Reindex {
  const profile = overrides.profile === undefined ? makeProfile() : overrides.profile;
  return {
    state: "running",
    target: "candidate",
    fingerprint: profile?.fingerprint ?? NOMIC_FINGERPRINT,
    profile,
    memories_total: 1204,
    memories_indexed: 120,
    started_at: "2026-03-02T09:04:00Z",
    finished_at: null,
    error_code: null,
    ...overrides,
  };
}

/** A search index that fits the embedding model in use and was never rebuilt. */
export function makeVectorIndex(overrides: Partial<VectorIndex> = {}): VectorIndex {
  const status = overrides.status ?? "ready";
  return {
    status,
    reindex_required: status === "reindex_required",
    // A rebuild replaces an index that search reads, unless it is said otherwise.
    serves_search: status === "not_managed" || ((status === "ready" || status === "reindexing") && overrides.index !== null),
    reindex_target: null,
    index: makeIndexRecord(),
    reindex: null,
    ...overrides,
  };
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

/** A workspace with both models ready, a search index that fits, and nothing waiting to be validated. */
export function makeModelSettings(overrides: Partial<ModelSettings> = {}): ModelSettings {
  return {
    writable: true,
    litellm_version: "1.86.0",
    generation: { active: makeBinding(), candidate: null },
    embedding: { active: makeEmbeddingBinding(), candidate: null },
    vector_index: makeVectorIndex(),
    ...overrides,
  };
}

/** A fresh installation: nothing configured for either kind, and no search index. */
export const NO_MODELS: Pick<ModelSettings, "generation" | "embedding" | "vector_index"> = {
  generation: { active: null, candidate: null },
  embedding: { active: null, candidate: null },
  vector_index: {
    status: "not_configured",
    reindex_required: false,
    serves_search: false,
    reindex_target: null,
    index: null,
    reindex: null,
  },
};

export function kindSettings(active: ModelBinding | null, candidate: ModelBinding | null = null): ModelKindSettings {
  return { active, candidate };
}
