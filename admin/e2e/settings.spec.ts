import { expect, test, type Page } from "@playwright/test";

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
  config_source: "settings" | "environment" | "deployment";
  state: "not_validated" | "ready" | "needs_revalidation";
  validation: Validation | null;
  api_key: { is_set: boolean; source: "saved" | "environment" | null; last4: string | null };
  saved_at: string | null;
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
    options: [],
    parameters: { generation: [], embedding: ["dimensions"] },
    ...overrides,
  };
}

const PROVIDERS = [
  provider("openai", "OpenAI", { default_endpoint: "https://api.openai.com/v1", options: ["organization"] }),
  provider("anthropic", "Anthropic", {
    kinds: ["generation"],
    default_endpoint: "https://api.anthropic.com",
    parameters: { generation: [] },
  }),
  provider("ollama", "Ollama", { auth_strategies: ["none", "api_key"], default_endpoint: "http://localhost:11434" }),
  provider("openai_compatible", "OpenAI-compatible endpoint", { auth_strategies: ["api_key", "none"], endpoint_required: true }),
  provider("litellm", "LiteLLM route (advanced)", { model_is_route: true, lists_models: false }),
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
  config_source: "settings",
  state: "ready",
  validation: passed({ latency_ms: 420, token_limits_known: true, schema_fallback: false }),
  api_key: { is_set: true, source: "saved", last4: "a91f" },
  saved_at: "2026-03-02T08:59:00Z",
};

const EMBEDDING: Binding = {
  ...GENERATION,
  kind: "embedding",
  provider: "ollama",
  endpoint: "http://localhost:11434",
  origin: "http://localhost:11434",
  auth_strategy: "none",
  model_id: "nomic-embed-text",
  route: "ollama/nomic-embed-text",
  validation: passed({ latency_ms: 96, dimension: 768 }),
  api_key: { is_set: false, source: null, last4: null },
};

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
  api_key?: string;
}

/**
 * Stubs the API with a server that keeps model settings the way the real one
 * does: saving writes the edit, and validating puts it to use only when
 * `validation` says the model passes. `writes` collects every write made.
 */
async function stubApi(page: Page, state: Settings, validation: { errorCode: string | null } = { errorCode: null }) {
  const writes: Array<{ request: string; body: unknown }> = [];
  await page.route("**/api/**", (route) => {
    const request = route.request();
    const { pathname } = new URL(request.url());
    if (pathname === SETTINGS_ROUTE) return route.fulfill({ json: state });
    if (pathname === `${SETTINGS_ROUTE}/providers`) return route.fulfill({ json: { providers: PROVIDERS } });

    const [kind, ...action] = pathname.startsWith(`${SETTINGS_ROUTE}/`)
      ? (pathname.slice(SETTINGS_ROUTE.length + 1).split("/") as [Kind, ...string[]])
      : [undefined];
    if (kind === undefined) {
      const body = RESPONSES[pathname];
      return body === undefined ? route.fulfill({ status: 404, json: { detail: "Not stubbed" } }) : route.fulfill({ json: body });
    }

    writes.push({ request: `${request.method()} ${pathname}`, body: request.postDataJSON() });
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
      const result: Validation =
        validation.errorCode === null
          ? passed(kind === "embedding" ? { latency_ms: 88, dimension: 1536 } : { latency_ms: 310, token_limits_known: true })
          : { ...passed({ latency_ms: 70 }), passed: false, error_code: validation.errorCode };
      if (result.passed) {
        slot.active = { ...slot.candidate, state: "ready", validation: result };
        slot.candidate = null;
      } else {
        slot.candidate = { ...slot.candidate, validation: result };
      }
      return route.fulfill({
        json: { passed: result.passed, promoted: result.passed, error_code: result.error_code, validation: result, settings: slot },
      });
    }
    return route.fulfill({ status: 404, json: { detail: "Not stubbed" } });
  });
  return writes;
}

const mainNav = (page: Page) => page.getByRole("navigation", { name: "Main" });

test("Settings saves and validates a model, and asks before leaving with unsaved changes", async ({ page }) => {
  const writes = await stubApi(page, settings());
  await page.goto("/v2/settings");

  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
  await expect(mainNav(page).getByRole("link", { name: "Settings" })).toHaveAttribute("aria-current", "page");
  await expect(page).toHaveTitle("Settings · MemForge");

  const embedding = page.getByRole("region", { name: "Embedding model" });
  await expect(embedding.getByText("Ready", { exact: true })).toBeVisible();
  await embedding.getByLabel("Model", { exact: true }).fill("mxbai-embed-large");
  await expect(embedding.getByText("Unsaved changes")).toBeVisible();

  await mainNav(page).getByRole("link", { name: "Sources" }).click();
  const dialog = page.getByRole("dialog", { name: "Leave without saving?" });
  await expect(dialog).toContainText("Your changes in Embedding model are not saved.");
  await dialog.getByRole("button", { name: "Keep editing" }).click();
  await expect(page).toHaveURL(/\/v2\/settings$/);

  await embedding.getByRole("button", { name: "Save and validate" }).click();
  await expect(embedding.getByText("Validation passed")).toBeVisible();
  await expect(embedding.getByRole("group", { name: "In use" })).toContainText("mxbai-embed-large");
  expect(writes).toEqual([
    {
      request: `PUT ${SETTINGS_ROUTE}/embedding/candidate`,
      body: {
        provider: "ollama",
        model_id: "mxbai-embed-large",
        endpoint: "http://localhost:11434",
        auth_strategy: "none",
        options: {},
        parameters: {},
      },
    },
    { request: `POST ${SETTINGS_ROUTE}/embedding/candidate/validate`, body: null },
  ]);

  await mainNav(page).getByRole("link", { name: "Sources" }).click();
  await expect(page).toHaveURL(/\/v2\/sources$/);
});

test("a model that fails validation is not put to use, and the page says which model still is", async ({ page }) => {
  await stubApi(page, settings(), { errorCode: "authentication_failed" });
  await page.goto("/v2/settings");

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
  await page.goto("/v2/settings");

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

  await expect(generation.getByText("Validation passed")).toBeVisible();
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
  const deployedEmbedding: Binding = {
    ...EMBEDDING,
    provider: "openai_compatible",
    endpoint: "http://embeddings.internal:8080/v1",
    origin: "http://embeddings.internal:8080",
    model_id: "bge-m3",
    route: "openai/bge-m3",
    config_source: "deployment",
    saved_at: null,
  };
  await stubApi(
    page,
    settings({
      writable: false,
      generation: { active: deployedGeneration, candidate: null },
      embedding: { active: deployedEmbedding, candidate: null },
    }),
  );
  await page.goto("/v2/settings");

  await expect(page.getByText("Managed by the deployment environment.")).toBeVisible();
  const generation = page.getByRole("region", { name: "Generation model" });
  await expect(generation.getByRole("group", { name: "In use" })).toContainText("openai/gpt-5.4");
  await expect(generation.getByRole("group", { name: "In use" })).toContainText("****9d1e");
  await expect(generation.getByRole("button", { name: "Validate again" })).toBeVisible();
  await expect(page.getByRole("button", { name: "Save and validate" })).toHaveCount(0);
  await expect(page.getByLabel("Base URL")).toHaveCount(0);
});

test("a fresh installation is led from the page it lands on to Settings, until a generation model is in use", async ({ page }) => {
  const writes = await stubApi(
    page,
    settings({ generation: { active: null, candidate: null }, embedding: { active: null, candidate: null } }),
  );
  await page.goto("/v2/");

  await expect(page).toHaveURL(/\/v2\/sources$/);
  const notice = page.getByRole("status").filter({ hasText: "No generation model is set up" });
  await expect(notice).toContainText("MemForge cannot write memories from your sources until you choose one.");
  await notice.getByRole("link", { name: "Open Settings" }).click();

  await expect(page).toHaveURL(/\/v2\/settings$/);
  await expect(notice).toHaveCount(0);
  const generation = page.getByRole("region", { name: "Generation model" });
  await expect(generation.getByText("Not set up")).toBeVisible();
  await expect(generation.getByRole("combobox", { name: "Provider" })).toContainText("OpenAI");
  await generation.getByLabel("API key", { exact: true }).fill("sk-first-key-77c2");
  await generation.getByLabel("Model", { exact: true }).fill("gpt-5.4");
  await generation.getByRole("button", { name: "Save and validate" }).click();

  await expect(generation.getByText("Validation passed")).toBeVisible();
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
      api_key: "sk-first-key-77c2",
    },
  });

  await mainNav(page).getByRole("link", { name: "Sources" }).click();
  await expect(page.getByRole("heading", { name: "Sources" })).toBeVisible();
  await expect(notice).toHaveCount(0);
});
