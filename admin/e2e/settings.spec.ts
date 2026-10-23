import { expect, test, type Page } from "@playwright/test";
import { appPath, appUrl } from "./support/appPath";

const SETTINGS_ROUTE = "/api/v1/model-settings";

type Kind = "generation" | "embedding";

interface Binding {
  kind: Kind;
  provider: string;
  endpoint: string | null;
  origin: string;
  auth_strategy: "api_key" | "none" | "environment";
  model_id: string;
  route: string;
  options: Record<string, unknown>;
  parameters: Record<string, unknown>;
  token_limits: Record<string, number>;
  config_source: "settings" | "environment" | "deployment";
  state: "not_validated" | "ready" | "needs_revalidation";
  validation: Validation | null;
  api_key: { is_set: boolean; source: "saved" | "environment" | null; last4: string | null };
  saved_at: string | null;
  profile: Profile | null;
  waits_for_reindex: boolean;
}

/** The vector space of an embedding model. Vectors are shared only between equal fingerprints. */
interface Profile {
  provider: string;
  route: string;
  endpoint: string | null;
  model_id: string;
  requested_dimensions: number | null;
  dimension: number;
  fingerprint: string;
  thresholds_calibrated: boolean;
}

interface Reindex {
  state: "running" | "succeeded" | "failed";
  target: "active" | "candidate";
  fingerprint: string;
  profile: Profile;
  memories_total: number | null;
  memories_indexed: number;
  started_at: string;
  finished_at: string | null;
  error_code: string | null;
}

interface VectorIndex {
  status: "ready" | "reindex_required" | "reindexing" | "awaiting_validation" | "not_configured" | "not_managed";
  reindex_required: boolean;
  serves_search: boolean;
  reindex_target: "active" | "candidate" | null;
  index: {
    fingerprint: string;
    collection: string;
    dimension: number;
    adopted: boolean;
    vector_count: number;
    activated_at: string | null;
    profile: Profile | null;
  } | null;
  reindex: Reindex | null;
}

interface Validation {
  checked_at: string;
  passed: boolean;
  error_code: string | null;
  facts: Record<string, unknown>;
  current: boolean;
}

interface KindSettings {
  active: Binding | null;
  candidate: Binding | null;
}

interface Settings {
  writable: boolean;
  litellm_version: string;
  generation: KindSettings;
  embedding: KindSettings;
  vector_index: VectorIndex;
}

function provider(id: string, label: string, overrides: Record<string, unknown> = {}) {
  return {
    id,
    label,
    kinds: ["generation", "embedding"],
    auth_strategies: ["api_key", "environment"],
    default_endpoint: null,
    endpoint_required: false,
    model_is_route: false,
    lists_models: true,
    lists_token_limits: false,
    options: [],
    parameters: { generation: [], embedding: ["dimensions"] },
    token_limits: { generation: ["context_window_tokens", "max_output_tokens"], embedding: [] },
    ...overrides,
  };
}

const PROVIDERS = [
  provider("openai", "OpenAI", { default_endpoint: "https://api.openai.com/v1", options: ["organization"] }),
  provider("anthropic", "Anthropic", {
    kinds: ["generation"],
    default_endpoint: "https://api.anthropic.com",
    parameters: { generation: [] },
    token_limits: { generation: ["context_window_tokens", "max_output_tokens"] },
  }),
  provider("ollama", "Ollama", { auth_strategies: ["none", "api_key"], default_endpoint: "http://localhost:11434" }),
  provider("openai_compatible", "OpenAI-compatible endpoint", {
    auth_strategies: ["api_key", "none"],
    endpoint_required: true,
    lists_token_limits: true,
  }),
  provider("litellm", "Other provider (LiteLLM route)", { model_is_route: true, lists_models: false }),
];

function passed(facts: Record<string, unknown>): Validation {
  return { checked_at: new Date().toISOString(), passed: true, error_code: null, facts, current: true };
}

const GENERATION: Binding = {
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
  validation: passed({ latency_ms: 420, token_limits_known: true, schema_fallback: false }),
  api_key: { is_set: true, source: "saved", last4: "a91f" },
  saved_at: "2026-03-02T08:59:00Z",
  profile: null,
  waits_for_reindex: false,
};

const MEMORY_COUNT = 1204;

/** The vector space a model embeds in; two models never share a fingerprint. */
function profileOf(model: Pick<Binding, "provider" | "route" | "endpoint" | "model_id">, dimension: number): Profile {
  const hex = Array.from(`${model.route}@${dimension}`, (character) => character.charCodeAt(0).toString(16)).join("");
  return {
    provider: model.provider,
    route: model.route,
    endpoint: model.endpoint,
    model_id: model.model_id,
    requested_dimensions: null,
    dimension,
    fingerprint: hex.padEnd(64, "0").slice(0, 64),
    thresholds_calibrated: model.model_id === "text-embedding-3-small",
  };
}

/** The search index as `profile` built it, holding one vector for every memory. */
function indexOf(profile: Profile): NonNullable<VectorIndex["index"]> {
  return {
    fingerprint: profile.fingerprint,
    collection: `memories-${profile.fingerprint}`,
    dimension: profile.dimension,
    adopted: false,
    vector_count: MEMORY_COUNT,
    activated_at: "2026-03-01T10:00:00Z",
    profile,
  };
}

/** A model with the vector space it embeds in, as a validation of it observed `dimension`. */
function embeddingModel(model: Binding, dimension: number): Binding {
  return { ...model, validation: passed({ latency_ms: 96, dimension }), profile: profileOf(model, dimension) };
}

const EMBEDDING = embeddingModel(
  {
    ...GENERATION,
    kind: "embedding",
    provider: "ollama",
    endpoint: "http://localhost:11434",
    origin: "http://localhost:11434",
    auth_strategy: "none",
    model_id: "nomic-embed-text",
    route: "ollama/nomic-embed-text",
    api_key: { is_set: false, source: null, last4: null },
  },
  768,
);

const NO_INDEX: VectorIndex = {
  status: "not_configured",
  reindex_required: false,
  serves_search: false,
  reindex_target: null,
  index: null,
  reindex: null,
};

/** A search index the embedding model in use built. */
function fittingIndex(): VectorIndex {
  return {
    status: "ready",
    reindex_required: false,
    serves_search: true,
    reindex_target: null,
    index: indexOf(EMBEDDING.profile!),
    reindex: null,
  };
}

/**
 * A generation model as the server reads it from its `MEMFORGE_*` variables
 * when they name a route and a base URL but no key. It is not stored.
 */
const ENVIRONMENT_GENERATION: Binding = {
  ...GENERATION,
  provider: "litellm",
  endpoint: "http://127.0.0.1:4000/v1",
  origin: "http://127.0.0.1:4000",
  auth_strategy: "environment",
  model_id: "openai/gpt-5.4",
  route: "openai/gpt-5.4",
  config_source: "environment",
  api_key: { is_set: false, source: null, last4: null },
  saved_at: null,
};

function settings(overrides: Partial<Settings> = {}): Settings {
  return {
    writable: true,
    litellm_version: "1.86.0",
    generation: { active: GENERATION, candidate: null },
    embedding: { active: EMBEDDING, candidate: null },
    vector_index: fittingIndex(),
    ...overrides,
  };
}

const RESPONSES: Record<string, unknown> = {
  "/api/v1/sources": { data: [] },
  "/api/v1/projects": { data: [], can_manage: true },
  "/api/v1/genes": [],
  "/api/v1/source-list/preferences": { sort_mode: "name" },
  "/api/cloud/local-agent/status": { status: "offline", last_seen_at: null, checked_at: "", stale_after_seconds: 60 },
  "/api/cloud/local-agent/jobs/current": { data: [] },
};

interface CandidateBody {
  provider: string;
  model_id: string;
  endpoint: string | null;
  auth_strategy: Binding["auth_strategy"];
  token_limits?: Record<string, number>;
  api_key?: string;
}

/** What the models and the rebuilds behind the stubbed server do. */
interface Script {
  /** The code every validation fails with; a model passes when there is none. */
  errorCode?: string;
  /** True for a generation model LiteLLM has no token limits for: it passes only with both stated. */
  limitsUnknown?: boolean;
  /** How each rebuild of the search index ends, in the order they are started: `null` succeeds, a code fails. */
  rebuilds?: Array<string | null>;
}

const VALIDATED_DIMENSION = 1024;
const STATUS_ACCEPTED = 202;

function validationOf(kind: Kind, candidate: Binding, script: Script): Validation {
  if (script.errorCode) return { ...passed({ latency_ms: 70 }), passed: false, error_code: script.errorCode };
  if (kind === "embedding") return passed({ latency_ms: 88, dimension: VALIDATED_DIMENSION });
  if (script.limitsUnknown && Object.keys(candidate.token_limits).length < 2) {
    return { ...passed({ latency_ms: 310, token_limits_known: false }), passed: false, error_code: "token_limits_required" };
  }
  return passed({ latency_ms: 310, token_limits_known: true });
}

/**
 * Moves a running rebuild on by one read of the settings: the first read
 * finds it half way, the second finds it ended. One that succeeds puts its
 * model and its index to use; one that fails leaves everything as it was.
 */
function advanceRebuild(state: Settings, before: VectorIndex, errorCode: string | null) {
  const reindex = state.vector_index.reindex!;
  if (reindex.memories_total === null) {
    state.vector_index = { ...state.vector_index, reindex: { ...reindex, memories_total: MEMORY_COUNT, memories_indexed: MEMORY_COUNT / 2 } };
    return;
  }
  const finished = { ...reindex, memories_total: MEMORY_COUNT, finished_at: new Date().toISOString() };
  if (errorCode !== null) {
    state.vector_index = { ...before, reindex: { ...finished, state: "failed", error_code: errorCode } };
    return;
  }
  if (reindex.target === "candidate") {
    state.embedding = { active: { ...state.embedding.candidate!, waits_for_reindex: false }, candidate: null };
  }
  state.vector_index = {
    status: "ready",
    reindex_required: false,
    serves_search: true,
    reindex_target: null,
    index: indexOf(reindex.profile),
    reindex: { ...finished, state: "succeeded", memories_indexed: MEMORY_COUNT },
  };
}

/**
 * Stubs the API with a server that keeps model settings the way the real one
 * does: saving writes the edit, and validating puts it to use only when
 * `script` lets the model pass. An embedding model of another vector space
 * than the search index passes and waits; a rebuild of the index then moves
 * on with every read of the settings and ends as `script` says. `writes`
 * collects every write made.
 */
async function stubApi(page: Page, state: Settings, script: Script = {}) {
  const writes: Array<{ request: string; body: unknown }> = [];
  const outcomes = [...(script.rebuilds ?? [])];
  let rebuild: { before: VectorIndex; errorCode: string | null } | null = null;
  await page.route("**/api/**", (route) => {
    const request = route.request();
    const { pathname } = new URL(request.url());
    if (pathname === SETTINGS_ROUTE) {
      if (rebuild !== null) {
        advanceRebuild(state, rebuild.before, rebuild.errorCode);
        if (state.vector_index.status !== "reindexing") rebuild = null;
      }
      return route.fulfill({ json: state });
    }
    if (pathname === `${SETTINGS_ROUTE}/providers`) return route.fulfill({ json: { providers: PROVIDERS } });
    if (!pathname.startsWith(`${SETTINGS_ROUTE}/`)) {
      const body = RESPONSES[pathname];
      return body === undefined ? route.fulfill({ status: 404, json: { detail: "Not stubbed" } }) : route.fulfill({ json: body });
    }

    writes.push({ request: `${request.method()} ${pathname}`, body: request.postDataJSON() });
    const [resource, ...action] = pathname.slice(SETTINGS_ROUTE.length + 1).split("/");
    if (resource === "vector-index" && action.join("/") === "reindex") {
      const target = state.embedding.candidate?.waits_for_reindex ? "candidate" : "active";
      const profile = (target === "candidate" ? state.embedding.candidate : state.embedding.active)!.profile!;
      rebuild = { before: state.vector_index, errorCode: outcomes.shift() ?? null };
      state.vector_index = {
        ...state.vector_index,
        status: "reindexing",
        reindex_required: false,
        reindex_target: null,
        reindex: {
          state: "running",
          target,
          fingerprint: profile.fingerprint,
          profile,
          memories_total: null,
          memories_indexed: 0,
          started_at: new Date().toISOString(),
          finished_at: null,
          error_code: null,
        },
      };
      return route.fulfill({ status: STATUS_ACCEPTED, json: state.vector_index });
    }

    const kind = resource as Kind;
    const slot = state[kind];
    if (action.join("/") === "candidate" && request.method() === "PUT") {
      const body = request.postDataJSON() as CandidateBody;
      const provider = PROVIDERS.find((candidate) => candidate.id === body.provider);
      const ownAddress = provider?.default_endpoint ?? null;
      const endpoint = body.endpoint ?? ownAddress;
      if (body.auth_strategy === "environment" && endpoint !== ownAddress) {
        return route.fulfill({
          status: 422,
          json: { detail: { error: "environment_endpoint_not_allowed", message: "Leave the base URL empty." } },
        });
      }
      slot.candidate = {
        ...GENERATION,
        kind,
        provider: body.provider,
        endpoint,
        origin: endpoint === null ? `provider:${body.model_id.split("/")[0]}` : new URL(endpoint).origin,
        auth_strategy: body.auth_strategy,
        model_id: body.model_id,
        route: provider?.model_is_route ? body.model_id : `${body.provider}/${body.model_id}`,
        token_limits: body.token_limits ?? {},
        state: "not_validated",
        validation: null,
        api_key:
          body.auth_strategy === "api_key"
            ? { is_set: true, source: "saved", last4: body.api_key?.slice(-4) ?? slot.active?.api_key.last4 ?? null }
            : { is_set: false, source: null, last4: null },
        saved_at: new Date().toISOString(),
      };
      return route.fulfill({ json: slot });
    }
    if (action.join("/") === "candidate/validate" && slot.candidate) {
      const result = validationOf(kind, slot.candidate, script);
      const profile = kind === "embedding" && result.passed ? profileOf(slot.candidate, VALIDATED_DIMENSION) : null;
      const index = state.vector_index.index;
      const waits = profile !== null && index !== null && index.fingerprint !== profile.fingerprint;
      if (waits) {
        slot.candidate = { ...slot.candidate, state: "ready", validation: result, profile, waits_for_reindex: true };
        state.vector_index = { ...state.vector_index, reindex_target: "candidate" };
      } else if (result.passed) {
        slot.active = { ...slot.candidate, state: "ready", validation: result, profile };
        slot.candidate = null;
      } else {
        slot.candidate = { ...slot.candidate, validation: result };
      }
      return route.fulfill({
        json: {
          passed: result.passed,
          promoted: result.passed && !waits,
          reindex_required: waits,
          error_code: result.error_code,
          validation: result,
          settings: slot,
        },
      });
    }
    return route.fulfill({ status: 404, json: { detail: "Not stubbed" } });
  });
  return writes;
}

const mainNav = (page: Page) => page.getByRole("navigation", { name: "Main" });

test("Settings saves and validates a model, and asks before leaving with unsaved changes", async ({ page }) => {
  const writes = await stubApi(page, settings());
  await page.goto(appPath("settings"));

  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
  await expect(mainNav(page).getByRole("link", { name: "Settings" })).toHaveAttribute("aria-current", "page");
  await expect(page).toHaveTitle("Settings · MemForge");

  const generation = page.getByRole("region", { name: "Generation model" });
  await expect(generation.getByText("Ready", { exact: true })).toBeVisible();
  await generation.getByLabel("Model", { exact: true }).fill("gpt-5.4-mini");
  await expect(generation.getByText("Unsaved changes")).toBeVisible();

  await mainNav(page).getByRole("link", { name: "Sources" }).click();
  const dialog = page.getByRole("dialog", { name: "Leave without saving?" });
  await expect(dialog).toContainText("Your changes in Generation model are not saved.");
  await dialog.getByRole("button", { name: "Keep editing" }).click();
  await expect(page).toHaveURL(appUrl("settings"));

  await generation.getByRole("button", { name: "Save and validate" }).click();
  await expect(generation.getByRole("group", { name: "In use" }).getByText(/^Passed validation/)).toBeVisible();
  await expect(generation.getByRole("group", { name: "In use" })).toContainText("gpt-5.4-mini");
  expect(writes).toEqual([
    {
      request: `PUT ${SETTINGS_ROUTE}/generation/candidate`,
      body: {
        provider: "openai",
        model_id: "gpt-5.4-mini",
        endpoint: "https://api.openai.com/v1",
        auth_strategy: "api_key",
        options: {},
        parameters: {},
        token_limits: {},
      },
    },
    { request: `POST ${SETTINGS_ROUTE}/generation/candidate/validate`, body: null },
  ]);

  await mainNav(page).getByRole("link", { name: "Sources" }).click();
  await expect(page).toHaveURL(appUrl("sources"));
});

test("a model that fails validation is not put to use, and the page says which model still is", async ({ page }) => {
  await stubApi(page, settings(), { errorCode: "authentication_failed" });
  await page.goto(appPath("settings"));

  const generation = page.getByRole("region", { name: "Generation model" });
  await generation.getByRole("combobox", { name: "Provider" }).click();
  await page.getByRole("option", { name: "Anthropic" }).click();
  await expect(generation.getByLabel("Base URL")).toHaveValue("https://api.anthropic.com");
  await generation.getByLabel("Model", { exact: true }).fill("claude-sonnet-4-6");

  await generation.getByRole("button", { name: "Save and validate" }).click();
  await expect(
    generation.getByText("Enter the API key for https://api.anthropic.com. The saved key belongs to https://api.openai.com."),
  ).toBeVisible();

  await generation.getByLabel("API key", { exact: true }).fill("sk-ant-wrong-key");
  await generation.getByRole("button", { name: "Save and validate" }).click();

  const failure = generation.getByRole("status").filter({ hasText: "These settings failed validation and are not in use" });
  await expect(failure).toContainText("The provider rejected the credentials. Check the API key.");
  await expect(failure).toContainText("MemForge keeps using OpenAI gpt-5.4.");
  await expect(generation.getByText("Ready", { exact: true })).toBeVisible();
  await expect(generation.getByRole("group", { name: "In use" })).toContainText("gpt-5.4");
  await expect(page.getByText("sk-ant-wrong-key")).toHaveCount(0);
});

test("a model that pairs a base URL with environment credentials is saved only once one of the two is given up", async ({
  page,
}) => {
  const writes = await stubApi(page, settings({ generation: { active: ENVIRONMENT_GENERATION, candidate: null } }));
  await page.goto(appPath("settings"));

  const generation = page.getByRole("region", { name: "Generation model" });
  await expect(generation.getByRole("group", { name: "In use" })).toContainText("http://127.0.0.1:4000/v1");
  await expect(generation.getByLabel("Base URL")).toHaveValue("http://127.0.0.1:4000/v1");
  const conflict = generation
    .getByRole("status")
    .filter({ hasText: "This base URL cannot be saved with credentials from the server environment" });
  await expect(conflict).toContainText("This model sends its requests to http://127.0.0.1:4000/v1.");

  await generation.getByLabel("Model route").fill("openai/gpt-5.4-mini");
  await generation.getByRole("button", { name: "Save and validate" }).click();
  await expect(
    generation.getByText(
      "Credentials from the server environment cannot be saved with a base URL. Remove the base URL, or choose another way of authenticating.",
    ),
  ).toBeVisible();
  await expect(generation.getByLabel("Base URL")).toHaveValue("http://127.0.0.1:4000/v1");
  expect(writes).toEqual([]);

  await conflict.getByRole("button", { name: "Remove the base URL" }).click();
  await expect(generation.getByLabel("Base URL")).toHaveValue("");
  await expect(conflict).toHaveCount(0);
  await generation.getByRole("button", { name: "Save and validate" }).click();

  await expect(generation.getByRole("group", { name: "In use" }).getByText(/^Passed validation/)).toBeVisible();
  const inUse = generation.getByRole("group", { name: "In use" });
  await expect(inUse).toContainText("openai/gpt-5.4-mini");
  await expect(inUse).not.toContainText("http://127.0.0.1:4000/v1");
  expect(writes).toEqual([
    {
      request: `PUT ${SETTINGS_ROUTE}/generation/candidate`,
      body: {
        provider: "litellm",
        model_id: "openai/gpt-5.4-mini",
        endpoint: null,
        auth_strategy: "environment",
        options: {},
        parameters: {},
        token_limits: {},
      },
    },
    { request: `POST ${SETTINGS_ROUTE}/generation/candidate/validate`, body: null },
  ]);
});

test("Settings managed by the deployment is read-only", async ({ page }) => {
  const deployedGeneration: Binding = {
    ...ENVIRONMENT_GENERATION,
    endpoint: null,
    origin: "provider:openai",
    auth_strategy: "api_key",
    config_source: "deployment",
    api_key: { is_set: true, source: "environment", last4: "9d1e" },
  };
  const deployedEmbedding = embeddingModel(
    {
      ...EMBEDDING,
      provider: "openai_compatible",
      endpoint: "http://embeddings.internal:8080/v1",
      origin: "http://embeddings.internal:8080",
      model_id: "bge-m3",
      route: "openai/bge-m3",
      config_source: "deployment",
      saved_at: null,
    },
    1024,
  );
  await stubApi(
    page,
    settings({
      writable: false,
      generation: { active: deployedGeneration, candidate: null },
      embedding: { active: deployedEmbedding, candidate: null },
      vector_index: { ...NO_INDEX, status: "not_managed" },
    }),
  );
  await page.goto(appPath("settings"));

  await expect(page.getByText("Managed by the deployment environment.")).toBeVisible();
  const generation = page.getByRole("region", { name: "Generation model" });
  await expect(generation.getByRole("group", { name: "In use" })).toContainText("openai/gpt-5.4");
  await expect(generation.getByRole("group", { name: "In use" })).toContainText("****9d1e");
  await expect(generation.getByRole("button", { name: "Validate again" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Save and validate" })).toHaveCount(0);
  await expect(page.getByLabel("Base URL")).toHaveCount(0);
  await expect(page.getByRole("group", { name: "Search index" })).toHaveCount(0);
});

test("a fresh installation is led from the page it lands on to Settings, until a generation model is in use", async ({ page }) => {
  const writes = await stubApi(
    page,
    settings({
      generation: { active: null, candidate: null },
      embedding: { active: null, candidate: null },
      vector_index: NO_INDEX,
    }),
  );
  await page.goto(appPath());

  await expect(page).toHaveURL(appUrl("sources"));
  const notice = page.getByRole("status").filter({ hasText: "No generation model is set up" });
  await expect(notice).toContainText("MemForge cannot write memories from your sources until you choose one.");
  await notice.getByRole("link", { name: "Open Settings" }).click();

  await expect(page).toHaveURL(appUrl("settings"));
  await expect(notice).toHaveCount(0);
  const generation = page.getByRole("region", { name: "Generation model" });
  await expect(generation.getByText("Not set up")).toBeVisible();
  await expect(generation.getByRole("combobox", { name: "Provider" })).toContainText("OpenAI");
  await generation.getByLabel("API key", { exact: true }).fill("sk-first-key-77c2");
  await generation.getByLabel("Model", { exact: true }).fill("gpt-5.4");
  await generation.getByRole("button", { name: "Save and validate" }).click();

  await expect(generation.getByRole("group", { name: "In use" }).getByText(/^Passed validation/)).toBeVisible();
  await expect(generation.getByText("Ready", { exact: true })).toBeVisible();
  await expect(generation.getByRole("group", { name: "In use" })).toContainText("****77c2");
  expect(writes[0]).toEqual({
    request: `PUT ${SETTINGS_ROUTE}/generation/candidate`,
    body: {
      provider: "openai",
      model_id: "gpt-5.4",
      endpoint: "https://api.openai.com/v1",
      auth_strategy: "api_key",
      options: {},
      parameters: {},
      token_limits: {},
      api_key: "sk-first-key-77c2",
    },
  });

  await mainNav(page).getByRole("link", { name: "Sources" }).click();
  await expect(page.getByRole("heading", { name: "Sources" })).toBeVisible();
  await expect(notice).toHaveCount(0);
});

test("changing the embedding model rebuilds the search index, and the new model comes into use when the rebuild ends", async ({
  page,
}) => {
  const writes = await stubApi(page, settings());
  await page.goto(appPath("settings"));

  const embedding = page.getByRole("region", { name: "Embedding model" });
  const index = embedding.getByRole("group", { name: "Search index" });
  await expect(index).toContainText("Ollama nomic-embed-text");
  await expect(index).toContainText("1,204");
  await expect(index).toContainText("Vector search is on.");
  await expect(index).toContainText("Thresholds not calibrated");

  await embedding.getByLabel("Model", { exact: true }).fill("mxbai-embed-large");
  await embedding.getByRole("button", { name: "Save and validate" }).click();

  const waiting = embedding.getByRole("status").filter({ hasText: "These settings passed validation and wait for the search index" });
  await expect(waiting).toContainText("the search index is rebuilt before MemForge uses the model");
  await expect(waiting).toContainText("MemForge keeps using Ollama nomic-embed-text.");
  await expect(embedding.getByRole("group", { name: "In use" })).toContainText("nomic-embed-text");

  await embedding.getByRole("button", { name: "Rebuild search index" }).click();

  await expect(index).toContainText("Rebuilding the search index.");
  await expect(index).toContainText("Every memory is embedded again with Ollama mxbai-embed-large.");
  await expect(index).toContainText("Search keeps working on the current index until the new one is ready.");
  await expect(embedding.getByRole("button", { name: "Rebuilding…" })).toBeDisabled();
  await expect(embedding.getByLabel("Model", { exact: true })).toBeDisabled();
  await expect(index).toContainText("602 of 1,204 memories embedded");
  await expect(index.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "50");

  await expect(embedding.getByText("Search index rebuilt")).toBeVisible();
  await expect(embedding.getByRole("group", { name: "In use" })).toContainText("mxbai-embed-large");
  await expect(index).toContainText("Ollama mxbai-embed-large");
  await expect(index).toContainText("Vector search is on. The index was rebuilt just now.");
  await expect(index.getByRole("progressbar")).toHaveCount(0);
  await expect(embedding.getByLabel("Model", { exact: true })).toBeEnabled();
  expect(writes.map((write) => write.request)).toEqual([
    `PUT ${SETTINGS_ROUTE}/embedding/candidate`,
    `POST ${SETTINGS_ROUTE}/embedding/candidate/validate`,
    `POST ${SETTINGS_ROUTE}/vector-index/reindex`,
  ]);
});

test("a rebuild that fails says why, leaves the model in use, and succeeds when it is run again", async ({ page }) => {
  const writes = await stubApi(page, settings(), { rebuilds: ["rate_limited", null] });
  await page.goto(appPath("settings"));

  const embedding = page.getByRole("region", { name: "Embedding model" });
  const index = embedding.getByRole("group", { name: "Search index" });
  await embedding.getByLabel("Model", { exact: true }).fill("mxbai-embed-large");
  await embedding.getByRole("button", { name: "Save and validate" }).click();
  await embedding.getByRole("button", { name: "Rebuild search index" }).click();
  await expect(index).toContainText("Rebuilding the search index.");

  const failure = index.getByRole("status").filter({ hasText: "The rebuild failed" });
  await expect(failure).toContainText("The provider is limiting requests right now. Try again in a moment.");
  await expect(failure).toContainText("The model and the index in use are unchanged.");
  await expect(embedding.getByRole("group", { name: "In use" })).toContainText("nomic-embed-text");
  await expect(index).toContainText("Ollama nomic-embed-text");
  await expect(index).toContainText("Vector search is on.");

  await embedding.getByRole("button", { name: "Run the rebuild again" }).click();

  await expect(index).toContainText("Rebuilding the search index.");
  await expect(failure).toHaveCount(0);
  await expect(embedding.getByText("Search index rebuilt")).toBeVisible();
  await expect(embedding.getByRole("group", { name: "In use" })).toContainText("mxbai-embed-large");
  expect(writes.map((write) => write.request).filter((request) => request.endsWith("/reindex"))).toHaveLength(2);
});

test("an installation whose search index does not fit its embedding model is told so on every page, and rebuilds it in Settings", async ({
  page,
}) => {
  const required: VectorIndex = {
    status: "reindex_required",
    reindex_required: true,
    serves_search: false,
    reindex_target: "active",
    index: indexOf(profileOf({ ...EMBEDDING, model_id: "all-minilm", route: "ollama/all-minilm" }, 384)),
    reindex: null,
  };
  await stubApi(page, settings({ vector_index: required }));
  await page.goto(appPath("sources"));

  const notice = page.getByRole("status").filter({ hasText: "Vector search is off" });
  await expect(notice).toContainText("Search ranks by matching words until the index is rebuilt in Settings.");
  await notice.getByRole("link", { name: "Open Settings" }).click();

  await expect(page).toHaveURL(appUrl("settings"));
  const index = page.getByRole("region", { name: "Embedding model" }).getByRole("group", { name: "Search index" });
  await expect(index).toContainText(
    "This index was built with Ollama all-minilm, and MemForge now uses Ollama nomic-embed-text.",
  );
  await index.getByRole("button", { name: "Rebuild search index" }).click();

  // The index being replaced was not serving search, and the rebuild does not change that until it ends.
  await expect(index).toContainText(/602 of 1,204 memories embedded.*Search ranks by matching words until the index is ready\./);
  await expect(index).toContainText("Vector search is on. The index was rebuilt just now.");
  await expect(index).toContainText("Ollama nomic-embed-text");
  await expect(index.getByRole("button")).toHaveCount(0);

  await mainNav(page).getByRole("link", { name: "Sources" }).click();
  await expect(page.getByRole("heading", { name: "Sources" })).toBeVisible();
  await expect(notice).toHaveCount(0);
});

test("a generation model without known token limits is asked for them, and passes once they are entered", async ({ page }) => {
  const writes = await stubApi(page, settings(), { limitsUnknown: true });
  await page.goto(appPath("settings"));

  const generation = page.getByRole("region", { name: "Generation model" });
  await expect(generation.getByRole("group", { name: "Token limits" })).toHaveCount(0);
  await generation.getByRole("combobox", { name: "Provider" }).click();
  await page.getByRole("option", { name: "Ollama" }).click();
  await generation.getByLabel("Model", { exact: true }).fill("qwen3:14b");
  await generation.getByRole("button", { name: "Save and validate" }).click();

  const failure = generation.getByRole("status").filter({ hasText: "One more step: enter the token limits" });
  await expect(failure).toContainText(
    "The model answered, but MemForge has no token limits for it. Enter its context window and maximum output, then validate again.",
  );
  const limits = generation.getByRole("group", { name: "Token limits" });
  await expect(limits).toContainText("MemForge sizes every request to these limits");
  await limits.getByLabel("Context window").fill("32768");
  await limits.getByLabel("Maximum output").fill("4096");
  await generation.getByRole("button", { name: "Save and validate" }).click();

  await expect(generation.getByRole("group", { name: "In use" }).getByText(/^Passed validation/)).toBeVisible();
  const inUse = generation.getByRole("group", { name: "In use" });
  await expect(inUse).toContainText("qwen3:14b");
  await expect(inUse).toContainText("32,768 tokens");
  await expect(inUse).toContainText("4,096 tokens");
  await expect(limits.getByLabel("Context window")).toHaveValue("32768");
  expect(writes.filter((write) => write.request.startsWith("PUT")).map((write) => write.body)).toEqual([
    {
      provider: "ollama",
      model_id: "qwen3:14b",
      endpoint: "http://localhost:11434",
      auth_strategy: "none",
      options: {},
      parameters: {},
      token_limits: {},
    },
    {
      provider: "ollama",
      model_id: "qwen3:14b",
      endpoint: "http://localhost:11434",
      auth_strategy: "none",
      options: {},
      parameters: {},
      token_limits: { context_window_tokens: 32768, max_output_tokens: 4096 },
    },
  ]);
});
