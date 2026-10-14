import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Link, Outlet } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";
import type { ModelBinding, ModelSettings } from "@/api";
import { errorResponse, type Route } from "@/test/fakeApi";
import {
  NO_MODELS,
  PROVIDERS,
  kindSettings,
  makeBinding,
  makeDeploymentBinding,
  makeEmbeddingBinding,
  makeEnvironmentBinding,
  makeIndexRecord,
  makeModelSettings,
  makeProfile,
  makeReindex,
  makeValidation,
  makeVectorIndex,
} from "@/test/modelSettingsFixtures";
import { renderRoutes } from "@/test/renderRoutes";
import { SettingsPage } from "./SettingsPage";

const HTTP_UNPROCESSABLE = 422;
const SETTINGS_ROUTE = "/api/v1/model-settings";

afterEach(() => {
  vi.unstubAllGlobals();
});

function Layout() {
  return (
    <>
      <Link to="/sources">Sources</Link>
      <Outlet />
    </>
  );
}

/**
 * A server that keeps its settings between requests, as the real one keeps
 * what the page writes. `change` alters them at once; the settings queued
 * with `thenAnswer` are what the next reads find, one after another, the way
 * a rebuild moves on between two reads.
 */
function settingsServer(initial: ModelSettings = makeModelSettings()) {
  let current = initial;
  const upcoming: Array<Partial<ModelSettings>> = [];
  return {
    read: () => {
      current = { ...current, ...upcoming.shift() };
      return current;
    },
    change: (next: Partial<ModelSettings>) => {
      current = { ...current, ...next };
      return current;
    },
    thenAnswer: (...next: Array<Partial<ModelSettings>>) => {
      upcoming.push(...next);
    },
  };
}

type SettingsServer = ReturnType<typeof settingsServer>;

function renderPage(settings: ModelSettings | SettingsServer = makeModelSettings(), routes: Record<string, Route> = {}) {
  const api: Record<string, Route> = {
    [`GET ${SETTINGS_ROUTE}`]: "read" in settings ? settings.read : () => settings,
    [`GET ${SETTINGS_ROUTE}/providers`]: () => ({ providers: PROVIDERS }),
    ...routes,
  };
  return renderRoutes(
    [
      {
        element: <Layout />,
        children: [
          { path: "settings", element: <SettingsPage /> },
          { path: "sources", element: <h1>Sources page</h1> },
        ],
      },
    ],
    { path: "/settings", api },
  );
}

/** How long a test waits for the page to read a running rebuild again; it does so every two seconds. */
const NEXT_READ = { timeout: 5_000 };
/** Room for a test that follows a rebuild over two or three reads. */
const REBUILD_TEST_MS = 20_000;

async function card(name: "Generation model" | "Embedding model") {
  return screen.findByRole("region", { name });
}

function inUse(region: HTMLElement) {
  return within(region).getByRole("group", { name: "In use" });
}

async function searchIndex() {
  return within(await card("Embedding model")).findByRole("group", { name: "Search index" });
}

function writes(calls: Request[]): string[] {
  return calls.filter((request) => request.method !== "GET").map((request) => `${request.method} ${new URL(request.url).pathname}`);
}

function bodyOf(calls: Request[], method: string, path: string) {
  const request = calls.find((call) => call.method === method && new URL(call.url).pathname === path);
  return request!.clone().json();
}

/** The embedding model as the form saves it: OpenAI-hosted, validated or not. */
function openAiEmbedding(overrides: Partial<ModelBinding> = {}): ModelBinding {
  return makeEmbeddingBinding({
    provider: "openai",
    endpoint: "https://api.openai.com/v1",
    origin: "https://api.openai.com",
    auth_strategy: "api_key",
    model_id: "text-embedding-3-small",
    route: "openai/text-embedding-3-small",
    api_key: { is_set: true, source: "saved", last4: "77c2" },
    ...overrides,
  });
}

test("shows the models in use with their state, and only the last characters of a saved key", async () => {
  renderPage();

  const generation = await card("Generation model");
  expect(within(generation).getByText("Ready")).toBeInTheDocument();
  const used = inUse(generation);
  expect(within(used).getByText("OpenAI")).toBeInTheDocument();
  expect(within(used).getByText("gpt-5.4")).toBeInTheDocument();
  expect(within(used).getByText("****a91f")).toBeInTheDocument();
  expect(within(used).getByText(/^Passed validation/)).toBeInTheDocument();

  expect(within(generation).getByRole("combobox", { name: "Provider" })).toHaveTextContent("OpenAI");
  expect(within(generation).getByLabelText("Base URL")).toHaveValue("https://api.openai.com/v1");
  expect(within(generation).getByLabelText("Model")).toHaveValue("gpt-5.4");
  expect(within(generation).getByLabelText("API key")).toHaveValue("");
  expect(within(generation).getByText("The saved key (****a91f) is used. Leave blank to keep it.")).toBeInTheDocument();
  expect(within(generation).getByRole("button", { name: "Save and validate" })).toBeDisabled();

  const embedding = await card("Embedding model");
  expect(within(inUse(embedding)).getByText("768")).toBeInTheDocument();
  expect(within(embedding).queryByLabelText("API key")).not.toBeInTheDocument();
});

describe("Save and validate", () => {
  test("saves the edit, validates it, and shows the model that is now in use", async () => {
    const promoted = makeEmbeddingBinding({ model_id: "mxbai-embed-large", route: "ollama/mxbai-embed-large" });
    const server = settingsServer();
    const { calls } = renderPage(server, {
      [`PUT ${SETTINGS_ROUTE}/embedding/candidate`]: () =>
        server.change({
          embedding: kindSettings(makeEmbeddingBinding(), { ...promoted, state: "not_validated", validation: null, profile: null }),
        }).embedding,
      [`POST ${SETTINGS_ROUTE}/embedding/candidate/validate`]: () => ({
        passed: true,
        promoted: true,
        reindex_required: false,
        error_code: null,
        validation: promoted.validation,
        settings: server.change({ embedding: kindSettings(promoted) }).embedding,
      }),
    });
    const embedding = await card("Embedding model");

    await userEvent.clear(within(embedding).getByLabelText("Model"));
    await userEvent.type(within(embedding).getByLabelText("Model"), "mxbai-embed-large");
    expect(within(embedding).getByText("Unsaved changes")).toBeInTheDocument();
    await userEvent.click(within(embedding).getByRole("button", { name: "Save and validate" }));

    expect(await within(embedding).findByText("Validation passed")).toBeInTheDocument();
    expect(within(embedding).getByText("MemForge now uses this model.")).toBeInTheDocument();
    expect(within(inUse(embedding)).getByText("mxbai-embed-large")).toBeInTheDocument();
    expect(within(embedding).queryByText("Unsaved changes")).not.toBeInTheDocument();
    expect(writes(calls)).toEqual([
      `PUT ${SETTINGS_ROUTE}/embedding/candidate`,
      `POST ${SETTINGS_ROUTE}/embedding/candidate/validate`,
    ]);
    expect(await bodyOf(calls, "PUT", `${SETTINGS_ROUTE}/embedding/candidate`)).toEqual({
      provider: "ollama",
      model_id: "mxbai-embed-large",
      endpoint: "http://localhost:11434",
      auth_strategy: "none",
      options: {},
      parameters: {},
      token_limits: {},
    });
  });

  test("a failed validation gives the reason and says which model stays in use", async () => {
    const failed = makeBinding({
      model_id: "gpt-9",
      route: "openai/gpt-9",
      state: "not_validated",
      validation: makeValidation({ passed: false, error_code: "model_not_found", facts: { latency_ms: 80 } }),
    });
    renderPage(makeModelSettings(), {
      [`PUT ${SETTINGS_ROUTE}/generation/candidate`]: () => kindSettings(makeBinding(), { ...failed, validation: null }),
      [`POST ${SETTINGS_ROUTE}/generation/candidate/validate`]: () => ({
        passed: false,
        promoted: false,
        error_code: "model_not_found",
        validation: failed.validation,
        settings: kindSettings(makeBinding(), failed),
      }),
    });
    const generation = await card("Generation model");

    await userEvent.clear(within(generation).getByLabelText("Model"));
    await userEvent.type(within(generation).getByLabelText("Model"), "gpt-9");
    await userEvent.click(within(generation).getByRole("button", { name: "Save and validate" }));

    const notice = await within(generation).findByText("These settings failed validation and are not in use");
    const status = notice.closest('[role="status"]') as HTMLElement;
    expect(status).toHaveTextContent("The provider does not know this model. Check the model name.");
    expect(status).toHaveTextContent("MemForge keeps using OpenAI gpt-5.4.");
    expect(within(inUse(generation)).getByText("gpt-5.4")).toBeInTheDocument();
    expect(within(generation).getByText("Ready")).toBeInTheDocument();
    expect(within(generation).getByLabelText("Model")).toHaveValue("gpt-9");
    expect(within(generation).getByRole("button", { name: "Save and validate" })).toBeEnabled();
  });

  test("a refused save shows the server's reason and keeps the edit", async () => {
    const { calls } = renderPage(makeModelSettings(), {
      [`PUT ${SETTINGS_ROUTE}/generation/candidate`]: () =>
        errorResponse(HTTP_UNPROCESSABLE, { error: "secret_storage_unavailable", message: "The API key cannot be stored." }),
    });
    const generation = await card("Generation model");

    await userEvent.type(within(generation).getByLabelText("API key"), "sk-new-key");
    await userEvent.click(within(generation).getByRole("button", { name: "Save and validate" }));

    const alert = await within(generation).findByRole("alert");
    expect(alert).toHaveTextContent("Couldn't save");
    expect(alert).toHaveTextContent("The API key cannot be stored.");
    expect(within(generation).getByLabelText("API key")).toHaveValue("sk-new-key");
    expect(within(generation).getByText("Unsaved changes")).toBeInTheDocument();
    expect(writes(calls)).toEqual([`PUT ${SETTINGS_ROUTE}/generation/candidate`]);
  });

  test("an edit that is already saved is validated without saving it again", async () => {
    const edit = openAiEmbedding({ state: "not_validated", validation: null, profile: null });
    const server = settingsServer(makeModelSettings({ embedding: kindSettings(makeEmbeddingBinding(), edit) }));
    const { calls } = renderPage(server, {
      [`POST ${SETTINGS_ROUTE}/embedding/candidate/validate`]: () => ({
        passed: true,
        promoted: true,
        validation: makeValidation(),
        settings: server.change({ embedding: kindSettings(openAiEmbedding()) }).embedding,
      }),
    });
    const embedding = await card("Embedding model");

    expect(within(embedding).getByText("These settings are saved but not in use")).toBeInTheDocument();
    expect(within(embedding).getByText("MemForge keeps using Ollama nomic-embed-text.")).toBeInTheDocument();
    expect(within(embedding).getByLabelText("Model")).toHaveValue("text-embedding-3-small");
    await userEvent.click(within(embedding).getByRole("button", { name: "Save and validate" }));

    expect(await within(embedding).findByText("Validation passed")).toBeInTheDocument();
    expect(writes(calls)).toEqual([`POST ${SETTINGS_ROUTE}/embedding/candidate/validate`]);
  });

  test("Discard changes drops a saved edit and shows the model in use again", async () => {
    const edit = openAiEmbedding({ state: "not_validated", validation: null, profile: null });
    const server = settingsServer(makeModelSettings({ embedding: kindSettings(makeEmbeddingBinding(), edit) }));
    const { calls } = renderPage(server, {
      [`DELETE ${SETTINGS_ROUTE}/embedding/candidate`]: () =>
        server.change({ embedding: kindSettings(makeEmbeddingBinding()) }).embedding,
    });
    const embedding = await card("Embedding model");

    await userEvent.click(within(embedding).getByRole("button", { name: "Discard changes" }));

    expect(await within(embedding).findByDisplayValue("nomic-embed-text")).toBeInTheDocument();
    expect(within(embedding).queryByText("These settings are saved but not in use")).not.toBeInTheDocument();
    expect(within(embedding).queryByRole("button", { name: "Discard changes" })).not.toBeInTheDocument();
    expect(writes(calls)).toEqual([`DELETE ${SETTINGS_ROUTE}/embedding/candidate`]);
  });
});

describe("the fields a provider needs", () => {
  test("a provider with its own address fills it in, and one without a key hides the key field", async () => {
    renderPage();
    const generation = await card("Generation model");

    await userEvent.click(within(generation).getByRole("combobox", { name: "Provider" }));
    await userEvent.click(await screen.findByRole("option", { name: "Ollama" }));

    expect(within(generation).getByLabelText("Base URL")).toHaveValue("http://localhost:11434");
    expect(within(generation).getByRole("combobox", { name: "Authentication" })).toHaveTextContent("No authentication");
    expect(within(generation).queryByLabelText("API key")).not.toBeInTheDocument();
    expect(within(generation).getByLabelText("Model")).toHaveValue("");
    expect(within(generation).queryByLabelText("Organization ID")).not.toBeInTheDocument();
  });

  test("an endpoint without an address of its own must be given one", async () => {
    const { calls } = renderPage();
    const generation = await card("Generation model");

    await userEvent.click(within(generation).getByRole("combobox", { name: "Provider" }));
    await userEvent.click(await screen.findByRole("option", { name: "OpenAI-compatible endpoint" }));
    await userEvent.type(within(generation).getByLabelText("Model"), "local-model");
    await userEvent.type(within(generation).getByLabelText("API key"), "sk-local");
    await userEvent.click(within(generation).getByRole("button", { name: "Save and validate" }));

    expect(await within(generation).findByText("Enter the base URL of the endpoint.")).toBeInTheDocument();
    expect(within(generation).getByLabelText("Base URL")).toHaveAttribute("aria-invalid", "true");
    expect(writes(calls)).toEqual([]);
  });

  test("credentials from the server environment fix the base URL to the provider's own", async () => {
    const { calls } = renderPage(makeModelSettings(), {
      [`PUT ${SETTINGS_ROUTE}/generation/candidate`]: () => errorResponse(HTTP_UNPROCESSABLE, { error: "x", message: "Stop here." }),
    });
    const generation = await card("Generation model");

    await userEvent.click(within(generation).getByRole("combobox", { name: "Authentication" }));
    await userEvent.click(await screen.findByRole("option", { name: "Credentials from the server environment" }));

    expect(within(generation).getByLabelText("Base URL")).toHaveAttribute("readonly");
    expect(within(generation).queryByLabelText("API key")).not.toBeInTheDocument();
    await userEvent.click(within(generation).getByRole("button", { name: "Save and validate" }));
    await within(generation).findByRole("alert");
    expect(await bodyOf(calls, "PUT", `${SETTINGS_ROUTE}/generation/candidate`)).toMatchObject({
      auth_strategy: "environment",
      endpoint: "https://api.openai.com/v1",
    });
  });

  test("choosing credentials from the server environment puts the provider's own address in place of a typed one", async () => {
    renderPage();
    const generation = await card("Generation model");
    await userEvent.clear(within(generation).getByLabelText("Base URL"));
    await userEvent.type(within(generation).getByLabelText("Base URL"), "https://proxy.example.test/v1");

    await userEvent.click(within(generation).getByRole("combobox", { name: "Authentication" }));
    await userEvent.click(await screen.findByRole("option", { name: "Credentials from the server environment" }));

    expect(within(generation).getByLabelText("Base URL")).toHaveValue("https://api.openai.com/v1");
    expect(within(generation).getByText(/requests go to https:\/\/api\.openai\.com\/v1 only/)).toBeInTheDocument();
  });

  test("an embedding model takes output dimensions as a whole number", async () => {
    const { calls } = renderPage();
    const embedding = await card("Embedding model");

    await userEvent.type(within(embedding).getByLabelText("Output dimensions"), "1.5");
    await userEvent.click(within(embedding).getByRole("button", { name: "Save and validate" }));

    expect(await within(embedding).findByText("Enter a whole number greater than zero.")).toBeInTheDocument();
    expect(within(embedding).getByLabelText("Output dimensions")).toHaveAttribute("aria-invalid", "true");
    expect(writes(calls)).toEqual([]);
    expect(within(await card("Generation model")).queryByLabelText("Output dimensions")).not.toBeInTheDocument();
  });
});

test("moving the base URL to another origin asks for the API key again before anything is sent", async () => {
  const { calls } = renderPage(makeModelSettings(), {
    [`PUT ${SETTINGS_ROUTE}/generation/candidate`]: () => errorResponse(HTTP_UNPROCESSABLE, { error: "x", message: "Stop here." }),
  });
  const generation = await card("Generation model");

  await userEvent.clear(within(generation).getByLabelText("Base URL"));
  await userEvent.type(within(generation).getByLabelText("Base URL"), "https://proxy.example.test/v1");
  expect(
    within(generation).getByText(
      "The saved key belongs to https://api.openai.com and is not sent anywhere else. Enter the key for this endpoint.",
    ),
  ).toBeInTheDocument();
  await userEvent.click(within(generation).getByRole("button", { name: "Save and validate" }));

  expect(
    await within(generation).findByText(
      "Enter the API key for https://proxy.example.test. The saved key belongs to https://api.openai.com.",
    ),
  ).toBeInTheDocument();
  expect(within(generation).getByLabelText("API key")).toHaveAttribute("aria-invalid", "true");
  expect(writes(calls)).toEqual([]);

  await userEvent.type(within(generation).getByLabelText("API key"), "sk-proxy-key");
  await userEvent.click(within(generation).getByRole("button", { name: "Save and validate" }));

  await within(generation).findByRole("alert");
  expect(await bodyOf(calls, "PUT", `${SETTINGS_ROUTE}/generation/candidate`)).toMatchObject({
    endpoint: "https://proxy.example.test/v1",
    api_key: "sk-proxy-key",
  });
});

describe("a model that pairs a base URL with credentials from the server environment", () => {
  const CONFLICT_TITLE = "This base URL cannot be saved with credentials from the server environment";
  const REFUSAL =
    "Credentials from the server environment cannot be saved with a base URL. Remove the base URL, or choose another way of authenticating.";

  function renderEnvironmentModel() {
    return renderPage(makeModelSettings({ generation: kindSettings(makeEnvironmentBinding()) }), {
      [`PUT ${SETTINGS_ROUTE}/generation/candidate`]: () => errorResponse(HTTP_UNPROCESSABLE, { error: "x", message: "Stop here." }),
    });
  }

  async function changeRoute(generation: HTMLElement) {
    await userEvent.clear(within(generation).getByLabelText("Model route"));
    await userEvent.type(within(generation).getByLabelText("Model route"), "openai/gpt-5.4-mini");
  }

  test("shows the base URL in use and says it cannot be saved that way", async () => {
    renderEnvironmentModel();
    const generation = await card("Generation model");

    expect(within(inUse(generation)).getByText("LiteLLM route (advanced)")).toBeInTheDocument();
    expect(within(inUse(generation)).getByText("http://127.0.0.1:4000/v1")).toBeInTheDocument();
    expect(within(generation).getByLabelText("Base URL")).toHaveValue("http://127.0.0.1:4000/v1");
    const notice = within(generation).getByText(CONFLICT_TITLE).closest('[role="status"]') as HTMLElement;
    expect(notice).toHaveTextContent("This model sends its requests to http://127.0.0.1:4000/v1.");
    expect(notice).toHaveTextContent("To keep the base URL, choose API key under Authentication and enter the key of that server.");
    expect(notice).toHaveTextContent("For a server that needs no key, choose Ollama or OpenAI-compatible endpoint as the provider.");
    expect(
      within(generation).queryByText("With credentials from the server environment, the route decides where requests go."),
    ).not.toBeInTheDocument();
    expect(within(generation).getByRole("button", { name: "Save and validate" })).toBeDisabled();
  });

  test("does not save another change until the base URL or the credentials are given up", async () => {
    const { calls } = renderEnvironmentModel();
    const generation = await card("Generation model");

    await changeRoute(generation);
    await userEvent.click(within(generation).getByRole("button", { name: "Save and validate" }));

    expect(await within(generation).findByText(REFUSAL)).toBeInTheDocument();
    expect(within(generation).getByLabelText("Base URL")).toHaveAttribute("aria-invalid", "true");
    expect(within(generation).getByLabelText("Base URL")).toHaveValue("http://127.0.0.1:4000/v1");
    expect(writes(calls)).toEqual([]);
  });

  test("Remove the base URL is a change of its own, and saving then sends none", async () => {
    const { calls } = renderEnvironmentModel();
    const generation = await card("Generation model");

    await userEvent.click(within(generation).getByRole("button", { name: "Remove the base URL" }));

    expect(within(generation).getByLabelText("Base URL")).toHaveValue("");
    expect(within(generation).queryByText(CONFLICT_TITLE)).not.toBeInTheDocument();
    expect(within(generation).getByText("With credentials from the server environment, the route decides where requests go.")).toBeInTheDocument();
    expect(within(generation).getByText("Unsaved changes")).toBeInTheDocument();
    await userEvent.click(within(generation).getByRole("button", { name: "Save and validate" }));

    await within(generation).findByRole("alert");
    expect(await bodyOf(calls, "PUT", `${SETTINGS_ROUTE}/generation/candidate`)).toEqual({
      provider: "litellm",
      model_id: "openai/gpt-5.4",
      endpoint: null,
      auth_strategy: "environment",
      options: {},
      parameters: {},
      token_limits: {},
    });
  });

  test("choosing API key keeps the base URL and asks for the key of that server", async () => {
    const { calls } = renderEnvironmentModel();
    const generation = await card("Generation model");
    await changeRoute(generation);
    await userEvent.click(within(generation).getByRole("button", { name: "Save and validate" }));
    await within(generation).findByText(REFUSAL);

    await userEvent.click(within(generation).getByRole("combobox", { name: "Authentication" }));
    await userEvent.click(await screen.findByRole("option", { name: "API key" }));

    expect(within(generation).queryByText(CONFLICT_TITLE)).not.toBeInTheDocument();
    expect(within(generation).queryByText(REFUSAL)).not.toBeInTheDocument();
    expect(within(generation).getByLabelText("Base URL (optional)")).toHaveValue("http://127.0.0.1:4000/v1");
    await userEvent.type(within(generation).getByLabelText("API key"), "sk-proxy-key");
    await userEvent.click(within(generation).getByRole("button", { name: "Save and validate" }));

    await within(generation).findByRole("alert");
    expect(await bodyOf(calls, "PUT", `${SETTINGS_ROUTE}/generation/candidate`)).toMatchObject({
      provider: "litellm",
      model_id: "openai/gpt-5.4-mini",
      endpoint: "http://127.0.0.1:4000/v1",
      auth_strategy: "api_key",
      api_key: "sk-proxy-key",
    });
  });
});

test("a model whose key variable is not set on the server asks for a key instead of keeping one", async () => {
  const withoutKey = makeBinding({
    provider: "anthropic",
    endpoint: "https://api.anthropic.com",
    origin: "https://api.anthropic.com",
    model_id: "claude-sonnet-4-6",
    route: "anthropic/claude-sonnet-4-6",
    state: "not_validated",
    validation: makeValidation({ passed: false, error_code: "credential_unavailable", facts: {} }),
    api_key: { is_set: false, source: "environment", last4: null },
  });
  const { calls } = renderPage(makeModelSettings({ generation: kindSettings(withoutKey) }));
  const generation = await card("Generation model");

  expect(within(inUse(generation)).getByText("Not available to the server")).toBeInTheDocument();
  expect(within(inUse(generation)).getByRole("status")).toHaveTextContent(
    "The server reads the API key of this model from MEMFORGE_ENRICHMENT_API_KEY, which is not set. MemForge keeps using this model.",
  );
  expect(
    within(generation).getByText(
      "MEMFORGE_ENRICHMENT_API_KEY is not set on the server. Enter a key here, or set the variable and restart the server.",
    ),
  ).toBeInTheDocument();
  expect(within(generation).queryByText(/Leave blank to keep it/)).not.toBeInTheDocument();

  await userEvent.type(within(generation).getByLabelText("Model"), "-latest");
  await userEvent.click(within(generation).getByRole("button", { name: "Save and validate" }));

  expect(await within(generation).findByText("Enter an API key.")).toBeInTheDocument();
  expect(writes(calls)).toEqual([]);
});

describe("Load models", () => {
  test("offers the models that fit the kind and fills an empty model field", async () => {
    const { calls } = renderPage(makeModelSettings(), {
      [`POST ${SETTINGS_ROUTE}/embedding/discover-models`]: () => ({
        ok: true,
        models_supported: true,
        models: [{ id: "mxbai-embed-large" }, { id: "llama3.3" }, { id: "nomic-embed-text" }],
        message: "Connected. Found 3 models.",
        latency_ms: 42,
      }),
    });
    const embedding = await card("Embedding model");
    await userEvent.clear(within(embedding).getByLabelText("Model"));

    await userEvent.click(within(embedding).getByRole("button", { name: "Load models" }));

    expect(await within(embedding).findByText("Connected. Found 3 models. (42 ms)")).toBeInTheDocument();
    expect(within(embedding).getByLabelText("Model")).toHaveValue("mxbai-embed-large");
    const suggestions = within(embedding).getByRole("group", { name: "Suggested models" });
    expect(within(suggestions).getAllByRole("button").map((button) => button.textContent)).toEqual([
      "mxbai-embed-large",
      "nomic-embed-text",
    ]);
    await userEvent.click(within(suggestions).getByRole("button", { name: "nomic-embed-text" }));
    expect(within(embedding).getByLabelText("Model")).toHaveValue("nomic-embed-text");
    expect(await bodyOf(calls, "POST", `${SETTINGS_ROUTE}/embedding/discover-models`)).toEqual({
      provider: "ollama",
      endpoint: "http://localhost:11434",
      auth_strategy: "none",
      api_key: null,
    });
  });

  test("applies the base URL the server suggests", async () => {
    renderPage(makeModelSettings(), {
      [`POST ${SETTINGS_ROUTE}/embedding/discover-models`]: () => ({
        ok: false,
        models_supported: false,
        stage: "connect",
        error_code: "provider_unreachable",
        message: "Could not reach localhost.",
        suggested_endpoint: "http://host.docker.internal:11434",
      }),
    });
    const embedding = await card("Embedding model");

    await userEvent.click(within(embedding).getByRole("button", { name: "Load models" }));
    await userEvent.click(await within(embedding).findByRole("button", { name: "Use host.docker.internal" }));

    expect(within(embedding).getByLabelText("Base URL")).toHaveValue("http://host.docker.internal:11434");
    expect(within(embedding).queryByText("Could not reach localhost.")).not.toBeInTheDocument();
  });

  test("a provider that cannot list models only takes a typed route", async () => {
    renderPage();
    const generation = await card("Generation model");

    await userEvent.click(within(generation).getByRole("combobox", { name: "Provider" }));
    await userEvent.click(await screen.findByRole("option", { name: "LiteLLM route (advanced)" }));

    expect(within(generation).getByLabelText("Model route")).toHaveAttribute("placeholder", "provider/model");
    expect(within(generation).getByLabelText("Base URL (optional)")).toHaveValue("");
    expect(within(generation).queryByRole("button", { name: "Load models" })).not.toBeInTheDocument();
  });
});

describe("the model in use", () => {
  test("Validate again checks it and shows why it failed, while it stays in use", async () => {
    const failed = makeBinding({
      state: "needs_revalidation",
      validation: makeValidation({ passed: false, error_code: "authentication_failed", facts: { latency_ms: 80 } }),
    });
    const { calls } = renderPage(makeModelSettings(), {
      [`POST ${SETTINGS_ROUTE}/generation/active/validate`]: () => ({
        passed: false,
        promoted: false,
        error_code: "authentication_failed",
        validation: failed.validation,
        settings: kindSettings(failed),
      }),
    });
    const generation = await card("Generation model");

    await userEvent.click(within(inUse(generation)).getByRole("button", { name: "Validate again" }));

    expect(await within(generation).findByText("Needs revalidation")).toBeInTheDocument();
    expect(within(inUse(generation)).getByRole("status")).toHaveTextContent(
      "The provider rejected the credentials. Check the API key. MemForge keeps using this model.",
    );
    expect(writes(calls)).toEqual([`POST ${SETTINGS_ROUTE}/generation/active/validate`]);
  });

  test("a model that states its token limits shows them", async () => {
    const local = makeBinding({ token_limits: { context_window_tokens: 32768, max_output_tokens: 4096 } });
    renderPage(makeModelSettings({ generation: kindSettings(local) }));

    const used = inUse(await card("Generation model"));
    expect(within(used).getByText("32,768 tokens")).toBeInTheDocument();
    expect(within(used).getByText("4,096 tokens")).toBeInTheDocument();
  });

  test("a model named by environment variables says that a saved one takes its place", async () => {
    const fromEnvironment = makeEnvironmentBinding({ state: "not_validated", validation: null });
    renderPage(makeModelSettings({ generation: kindSettings(fromEnvironment) }));

    const generation = await card("Generation model");
    expect(within(generation).getByText("Not validated")).toBeInTheDocument();
    expect(within(inUse(generation)).getByText(/A model saved here takes their place/)).toBeInTheDocument();
  });
});

test("a fresh installation says no model is in use and asks for one before saving", async () => {
  const { calls } = renderPage(makeModelSettings(NO_MODELS));
  const generation = await card("Generation model");

  expect(within(generation).getByText("Not set up")).toBeInTheDocument();
  expect(within(inUse(generation)).getByText(/No generation model is in use/)).toBeInTheDocument();
  expect(within(generation).getByText("Stored encrypted on the server and never shown again.")).toBeInTheDocument();
  await userEvent.click(within(generation).getByRole("button", { name: "Save and validate" }));

  expect(await within(generation).findByText("Enter a model.")).toBeInTheDocument();
  expect(within(generation).getByText("Enter an API key.")).toBeInTheDocument();
  expect(within(generation).getByLabelText("Model")).toHaveAttribute("aria-invalid", "true");
  expect(writes(calls)).toEqual([]);
});

test("when the deployment manages the models, the page only shows them and can still validate them", async () => {
  const deployed = makeDeploymentBinding();
  const { calls } = renderPage(
    makeModelSettings({ writable: false, generation: kindSettings(deployed), embedding: kindSettings(null) }),
    {
      [`POST ${SETTINGS_ROUTE}/generation/active/validate`]: () => ({
        passed: true,
        promoted: false,
        validation: makeValidation(),
        settings: kindSettings(deployed),
      }),
    },
  );

  expect(await screen.findByText("Managed by the deployment environment.")).toBeInTheDocument();
  const generation = await card("Generation model");
  expect(within(inUse(generation)).getByText("****9d1e")).toBeInTheDocument();
  expect(within(inUse(generation)).getByText("From the server environment")).toBeInTheDocument();
  expect(screen.queryByRole("form")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Save and validate" })).not.toBeInTheDocument();
  expect(screen.queryByLabelText("Base URL")).not.toBeInTheDocument();

  await userEvent.click(within(generation).getByRole("button", { name: "Validate again" }));
  expect(writes(calls)).toEqual([`POST ${SETTINGS_ROUTE}/generation/active/validate`]);
});

test("a settings request that fails can be tried again", async () => {
  let attempts = 0;
  renderPage(makeModelSettings(), {
    [`GET ${SETTINGS_ROUTE}`]: () => {
      attempts += 1;
      return attempts === 1 ? errorResponse(500, "The database is locked.") : makeModelSettings();
    },
  });

  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("The database is locked.");
  await userEvent.click(within(alert).getByRole("button", { name: "Try again" }));

  expect(await card("Generation model")).toBeInTheDocument();
});

describe("leaving with unsaved changes", () => {
  async function editGenerationModel() {
    const generation = await card("Generation model");
    await userEvent.type(within(generation).getByLabelText("Model"), "-mini");
  }

  test("asks first, and Keep editing stays on the page", async () => {
    const { router } = renderPage();
    await editGenerationModel();

    await userEvent.click(screen.getByRole("link", { name: "Sources" }));
    const dialog = await screen.findByRole("dialog", { name: "Leave without saving?" });
    expect(dialog).toHaveTextContent("Your changes in Generation model are not saved.");

    await userEvent.click(within(dialog).getByRole("button", { name: "Keep editing" }));
    expect(router.state.location.pathname).toBe("/settings");
    expect(within(await card("Generation model")).getByLabelText("Model")).toHaveValue("gpt-5.4-mini");
  });

  test("Leave without saving follows the link", async () => {
    renderPage();
    await editGenerationModel();

    await userEvent.click(screen.getByRole("link", { name: "Sources" }));
    await userEvent.click(await screen.findByRole("button", { name: "Leave without saving" }));

    expect(await screen.findByRole("heading", { name: "Sources page" })).toBeInTheDocument();
  });

  test("reloading or closing the tab gets the browser's prompt", async () => {
    renderPage();
    await card("Generation model");
    const cleanUnload = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(cleanUnload);
    expect(cleanUnload.defaultPrevented).toBe(false);

    await editGenerationModel();
    const dirtyUnload = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(dirtyUnload);
    expect(dirtyUnload.defaultPrevented).toBe(true);
  });

  test("Discard changes puts the form back, and the page then leaves without asking", async () => {
    renderPage();
    await editGenerationModel();

    await userEvent.click(within(await card("Generation model")).getByRole("button", { name: "Discard changes" }));
    expect(within(await card("Generation model")).getByLabelText("Model")).toHaveValue("gpt-5.4");
    await userEvent.click(screen.getByRole("link", { name: "Sources" }));

    expect(await screen.findByRole("heading", { name: "Sources page" })).toBeInTheDocument();
  });

  test("a page whose edit was saved leaves without asking, even when validation failed", async () => {
    const failed = makeBinding({
      model_id: "gpt-5.4-mini",
      state: "not_validated",
      validation: makeValidation({ passed: false, error_code: "timeout", facts: {} }),
    });
    renderPage(makeModelSettings(), {
      [`PUT ${SETTINGS_ROUTE}/generation/candidate`]: () => kindSettings(makeBinding(), { ...failed, validation: null }),
      [`POST ${SETTINGS_ROUTE}/generation/candidate/validate`]: () => ({
        passed: false,
        promoted: false,
        error_code: "timeout",
        validation: failed.validation,
        settings: kindSettings(makeBinding(), failed),
      }),
    });
    await editGenerationModel();
    await userEvent.click(within(await card("Generation model")).getByRole("button", { name: "Save and validate" }));
    await screen.findByText("These settings failed validation and are not in use");

    await userEvent.click(screen.getByRole("link", { name: "Sources" }));

    expect(await screen.findByRole("heading", { name: "Sources page" })).toBeInTheDocument();
  });
});

describe("the search index", () => {
  test("says which model built it, how many memories it holds, and what is special about it", async () => {
    renderPage(
      makeModelSettings({ vector_index: makeVectorIndex({ reindex_target: "active", index: makeIndexRecord({ adopted: true }) }) }),
    );

    const index = await searchIndex();
    expect(within(index).getByText("Ollama nomic-embed-text")).toBeInTheDocument();
    expect(within(index).getByText("1,204")).toBeInTheDocument();
    expect(within(index).getByText("768")).toBeInTheDocument();
    expect(within(index).getByRole("status")).toHaveTextContent("Vector search is on.");
    expect(within(index).getByText("Carried over")).toBeInTheDocument();
    expect(within(index).getByText(/takes them to come from the model in use/)).toBeInTheDocument();
    expect(within(index).getByText("Thresholds not calibrated")).toBeInTheDocument();
    expect(within(index).getByText(/were tuned on another model/)).toBeInTheDocument();
    expect(within(index).getByRole("button", { name: "Rebuild search index" })).toBeEnabled();
  });

  test("an index that fits its model, with thresholds tuned on it, offers no rebuild", async () => {
    const tuned = makeProfile({
      provider: "openai",
      route: "openai/text-embedding-3-small",
      endpoint: "https://api.openai.com/v1",
      model_id: "text-embedding-3-small",
      dimension: 1536,
      thresholds_calibrated: true,
    });
    renderPage(
      makeModelSettings({
        embedding: kindSettings(openAiEmbedding({ profile: tuned })),
        vector_index: makeVectorIndex({ index: makeIndexRecord({ profile: tuned }) }),
      }),
    );

    const index = await searchIndex();
    expect(within(index).getByText("OpenAI text-embedding-3-small")).toBeInTheDocument();
    expect(within(index).queryByText("Thresholds not calibrated")).not.toBeInTheDocument();
    expect(within(index).queryByText("Carried over")).not.toBeInTheDocument();
    expect(within(index).queryByRole("button")).not.toBeInTheDocument();
  });

  test(
    "an installation whose index does not fit its model rebuilds it from here, and sees the rebuild through",
    async () => {
      const required = makeVectorIndex({ status: "reindex_required", reindex_target: "active", index: null });
      const server = settingsServer(makeModelSettings({ vector_index: required }));
      const { calls } = renderPage(server, {
        [`POST ${SETTINGS_ROUTE}/vector-index/reindex`]: () =>
          server.change({
            vector_index: makeVectorIndex({
              status: "reindexing",
              index: null,
              reindex: makeReindex({ target: "active", memories_total: null, memories_indexed: 0 }),
            }),
          }).vector_index,
      });

      const index = await searchIndex();
      expect(within(index).getByRole("status")).toHaveTextContent("Vector search is off.");
      expect(within(index).getByRole("status")).toHaveTextContent("The vectors stored so far do not fit Ollama nomic-embed-text.");
      server.thenAnswer(
        {
          vector_index: makeVectorIndex({
            status: "reindexing",
            index: null,
            reindex: makeReindex({ target: "active", memories_indexed: 602 }),
          }),
        },
        {
          vector_index: makeVectorIndex({
            reindex: makeReindex({ target: "active", state: "succeeded", memories_indexed: 1204, finished_at: new Date().toISOString() }),
          }),
        },
      );
      await userEvent.click(within(index).getByRole("button", { name: "Rebuild search index" }));

      expect(await within(index).findByText("Counting the memories to embed…")).toBeInTheDocument();
      expect(within(index).getByText(/Search ranks by matching words until the index is ready\./)).toBeInTheDocument();
      expect(within(index).queryByRole("button")).not.toBeInTheDocument();
      expect(await within(index).findByText("602 of 1,204 memories embedded", undefined, NEXT_READ)).toBeInTheDocument();
      expect(within(index).getByRole("progressbar")).toHaveAttribute("aria-valuenow", "50");

      expect(await within(index).findByText("Vector search is on.", undefined, NEXT_READ)).toBeInTheDocument();
      expect(within(index).getByRole("status")).toHaveTextContent("The index was rebuilt just now.");
      expect(within(index).getByText("1,204")).toBeInTheDocument();
      expect(within(index).queryByRole("progressbar")).not.toBeInTheDocument();
      expect(writes(calls)).toEqual([`POST ${SETTINGS_ROUTE}/vector-index/reindex`]);
    },
    REBUILD_TEST_MS,
  );

  test("a rebuild that cannot start says why", async () => {
    renderPage(makeModelSettings({ vector_index: makeVectorIndex({ status: "reindex_required", reindex_target: "active" }) }), {
      [`POST ${SETTINGS_ROUTE}/vector-index/reindex`]: () =>
        errorResponse(409, {
          error: "sync_in_progress",
          message: "A Source is syncing. Rebuild the vector index once no Source is syncing.",
        }),
    });
    const index = await searchIndex();

    await userEvent.click(within(index).getByRole("button", { name: "Rebuild search index" }));

    const alert = await within(index).findByRole("alert");
    expect(alert).toHaveTextContent("Couldn't start the rebuild");
    expect(alert).toHaveTextContent("A source is syncing. Start the rebuild once no source is syncing.");
    expect(within(index).getByRole("button", { name: "Rebuild search index" })).toBeEnabled();
  });

  test("a rebuild that is running when the page opens shows its progress and holds the embedding model", async () => {
    renderPage(makeModelSettings({ vector_index: makeVectorIndex({ status: "reindexing", reindex: makeReindex() }) }));

    const index = await searchIndex();
    expect(within(index).getByText("120 of 1,204 memories embedded")).toBeInTheDocument();
    expect(within(index).getByText(/Search keeps working on the current index until the new one is ready\./)).toBeInTheDocument();
    expect(within(index).getByText(/Source syncs and changes to memories wait/)).toBeInTheDocument();

    const embedding = await card("Embedding model");
    expect(within(embedding).getByLabelText("Model")).toBeDisabled();
    expect(within(embedding).getByText("The embedding model can be changed once the search index is rebuilt.")).toBeInTheDocument();
    expect(within(await card("Generation model")).getByLabelText("Model")).toBeEnabled();
  });

  test("an installation the deployment manages can still rebuild its index", async () => {
    renderPage(
      makeModelSettings({
        writable: false,
        vector_index: makeVectorIndex({ status: "reindex_required", reindex_target: "active" }),
      }),
    );

    expect(within(await searchIndex()).getByRole("button", { name: "Rebuild search index" })).toBeEnabled();
    expect(screen.queryByRole("button", { name: "Save and validate" })).not.toBeInTheDocument();
  });

  test("a deployment that keeps the index by its own means shows nothing about it", async () => {
    renderPage(
      makeModelSettings({
        vector_index: makeVectorIndex({ status: "not_managed", index: null }),
      }),
    );

    const embedding = await card("Embedding model");
    expect(within(embedding).queryByRole("group", { name: "Search index" })).not.toBeInTheDocument();
    expect(within(embedding).queryByRole("button", { name: /rebuild/i })).not.toBeInTheDocument();
  });
});

describe("an embedding model whose vectors cannot be compared with the stored ones", () => {
  const LARGE = makeProfile({
    model_id: "mxbai-embed-large",
    route: "ollama/mxbai-embed-large",
    dimension: 1024,
    fingerprint: "c3".repeat(32),
  });
  /** The saved edit once it has passed validation: only the search index keeps it from use. */
  const waiting = makeEmbeddingBinding({
    model_id: "mxbai-embed-large",
    route: "ollama/mxbai-embed-large",
    validation: makeValidation({ facts: { latency_ms: 90, dimension: 1024 } }),
    profile: LARGE,
    waits_for_reindex: true,
  });
  const WAITING_INDEX = makeVectorIndex({ reindex_target: "candidate" });
  const REBUILDING = makeVectorIndex({
    status: "reindexing",
    reindex: makeReindex({ profile: LARGE, memories_total: null, memories_indexed: 0 }),
  });

  function waitingSettings() {
    return makeModelSettings({ embedding: kindSettings(makeEmbeddingBinding(), waiting), vector_index: WAITING_INDEX });
  }

  test(
    "passes validation, waits, and is put to use by the rebuild its action starts",
    async () => {
      const server = settingsServer();
      const { calls } = renderPage(server, {
        [`PUT ${SETTINGS_ROUTE}/embedding/candidate`]: () =>
          server.change({
            embedding: kindSettings(makeEmbeddingBinding(), {
              ...waiting,
              state: "not_validated",
              validation: null,
              profile: null,
              waits_for_reindex: false,
            }),
          }).embedding,
        [`POST ${SETTINGS_ROUTE}/embedding/candidate/validate`]: () => ({
          passed: true,
          promoted: false,
          reindex_required: true,
          error_code: null,
          validation: waiting.validation,
          settings: server.change(waitingSettings()).embedding,
        }),
        [`POST ${SETTINGS_ROUTE}/vector-index/reindex`]: () => server.change({ vector_index: REBUILDING }).vector_index,
      });
      const embedding = await card("Embedding model");

      await userEvent.clear(within(embedding).getByLabelText("Model"));
      await userEvent.type(within(embedding).getByLabelText("Model"), "mxbai-embed-large");
      await userEvent.click(within(embedding).getByRole("button", { name: "Save and validate" }));

      const notice = (await within(embedding).findByText("These settings passed validation and wait for the search index")).closest(
        '[role="status"]',
      ) as HTMLElement;
      expect(notice).toHaveTextContent("the search index is rebuilt before MemForge uses the model");
      expect(notice).toHaveTextContent("Thresholds not calibrated");
      expect(notice).toHaveTextContent("MemForge keeps using Ollama nomic-embed-text.");
      expect(within(inUse(embedding)).getByText("nomic-embed-text")).toBeInTheDocument();
      expect(within(embedding).queryByRole("button", { name: "Save and validate" })).not.toBeInTheDocument();

      server.thenAnswer(
        { vector_index: { ...REBUILDING, reindex: makeReindex({ profile: LARGE, memories_indexed: 301 }) } },
        {
          embedding: kindSettings({ ...waiting, waits_for_reindex: false }),
          vector_index: makeVectorIndex({
            index: makeIndexRecord({ profile: LARGE }),
            reindex: makeReindex({ profile: LARGE, state: "succeeded", memories_indexed: 1204, finished_at: new Date().toISOString() }),
          }),
        },
      );
      await userEvent.click(within(embedding).getByRole("button", { name: "Rebuild search index" }));

      const index = await searchIndex();
      expect(await within(index).findByText("Rebuilding the search index.")).toBeInTheDocument();
      expect(within(index).getByRole("status")).toHaveTextContent("Every memory is embedded again with Ollama mxbai-embed-large.");
      expect(within(embedding).getByRole("button", { name: "Rebuilding…" })).toBeDisabled();
      expect(within(embedding).getByLabelText("Model")).toBeDisabled();
      expect(await within(index).findByText("301 of 1,204 memories embedded", undefined, NEXT_READ)).toBeInTheDocument();
      expect(within(index).getByText(/Search keeps working on the current index/)).toBeInTheDocument();

      expect(await within(embedding).findByText("Search index rebuilt", undefined, NEXT_READ)).toBeInTheDocument();
      expect(within(embedding).getByText("MemForge now uses this model.")).toBeInTheDocument();
      expect(within(inUse(embedding)).getByText("mxbai-embed-large")).toBeInTheDocument();
      expect(within(index).getByText("Ollama mxbai-embed-large")).toBeInTheDocument();
      expect(within(index).getByRole("status")).toHaveTextContent("Vector search is on. The index was rebuilt just now.");
      expect(within(embedding).getByRole("button", { name: "Save and validate" })).toBeDisabled();
      expect(writes(calls)).toEqual([
        `PUT ${SETTINGS_ROUTE}/embedding/candidate`,
        `POST ${SETTINGS_ROUTE}/embedding/candidate/validate`,
        `POST ${SETTINGS_ROUTE}/vector-index/reindex`,
      ]);
    },
    REBUILD_TEST_MS,
  );

  test(
    "a rebuild that fails gives the reason, changes nothing in use, and can be run again",
    async () => {
      const server = settingsServer(waitingSettings());
      const { calls } = renderPage(server, {
        [`POST ${SETTINGS_ROUTE}/vector-index/reindex`]: () => server.change({ vector_index: REBUILDING }).vector_index,
      });
      const embedding = await card("Embedding model");
      const failed = makeVectorIndex({
        reindex_target: "candidate",
        reindex: makeReindex({
          profile: LARGE,
          state: "failed",
          error_code: "rate_limited",
          memories_indexed: 301,
          finished_at: new Date().toISOString(),
        }),
      });

      server.thenAnswer({ vector_index: failed });
      await userEvent.click(within(embedding).getByRole("button", { name: "Rebuild search index" }));

      const index = await searchIndex();
      const failure = (await within(index).findByText("The rebuild failed just now", undefined, NEXT_READ)).closest(
        '[role="status"]',
      ) as HTMLElement;
      expect(failure).toHaveTextContent("The provider is limiting requests right now. Try again in a moment.");
      expect(failure).toHaveTextContent("The model and the index in use are unchanged.");
      expect(within(inUse(embedding)).getByText("nomic-embed-text")).toBeInTheDocument();
      expect(within(index).getByText("Vector search is on.")).toBeInTheDocument();
      expect(within(embedding).getByText("These settings passed validation and wait for the search index")).toBeInTheDocument();

      await userEvent.click(within(embedding).getByRole("button", { name: "Run the rebuild again" }));

      expect(await within(index).findByText("Rebuilding the search index.")).toBeInTheDocument();
      expect(within(index).queryByText("The rebuild failed just now")).not.toBeInTheDocument();
      expect(writes(calls)).toEqual([
        `POST ${SETTINGS_ROUTE}/vector-index/reindex`,
        `POST ${SETTINGS_ROUTE}/vector-index/reindex`,
      ]);
    },
    REBUILD_TEST_MS,
  );

  test("says that search ranks by matching words meanwhile when the index in use does not fit the model in use", async () => {
    const off = makeVectorIndex({ status: "reindex_required", reindex_target: "candidate" });
    renderPage(makeModelSettings({ embedding: kindSettings(makeEmbeddingBinding(), waiting), vector_index: off }));
    const embedding = await card("Embedding model");

    const notice = within(embedding)
      .getByText("These settings passed validation and wait for the search index")
      .closest('[role="status"]') as HTMLElement;
    expect(notice).toHaveTextContent("meanwhile search ranks by matching words, as it does now, and source syncs and changes to memories wait.");
    expect(notice).not.toHaveTextContent("search keeps working");
  });

  test("a rebuild another process already started is followed instead of reported as a failure", async () => {
    const server = settingsServer(waitingSettings());
    renderPage(server, {
      [`POST ${SETTINGS_ROUTE}/vector-index/reindex`]: () => {
        server.change({ vector_index: { ...REBUILDING, reindex: makeReindex({ profile: LARGE }) } });
        return errorResponse(409, { error: "reindex_in_progress", message: "The Memory vector index is already being rebuilt." });
      },
    });
    const embedding = await card("Embedding model");

    await userEvent.click(within(embedding).getByRole("button", { name: "Rebuild search index" }));

    expect(await within(await searchIndex()).findByText("120 of 1,204 memories embedded")).toBeInTheDocument();
    expect(within(embedding).queryByRole("alert")).not.toBeInTheDocument();
  });

  test("changing the waiting settings makes the action save and validate again", async () => {
    renderPage(waitingSettings());
    const embedding = await card("Embedding model");
    expect(within(embedding).getByRole("button", { name: "Rebuild search index" })).toBeEnabled();

    await userEvent.type(within(embedding).getByLabelText("Output dimensions"), "512");

    expect(within(embedding).getByRole("button", { name: "Save and validate" })).toBeEnabled();
    expect(within(embedding).queryByRole("button", { name: "Rebuild search index" })).not.toBeInTheDocument();
  });
});

describe("the token limits of a generation model", () => {
  const LIMITS_UNKNOWN = makeValidation({ passed: false, error_code: "token_limits_required", facts: { latency_ms: 300, token_limits_known: false } });

  function localModel(overrides: Partial<ModelBinding> = {}): ModelBinding {
    return makeBinding({
      provider: "ollama",
      endpoint: "http://localhost:11434",
      origin: "http://localhost:11434",
      auth_strategy: "none",
      model_id: "qwen3:14b",
      route: "ollama/qwen3:14b",
      api_key: { is_set: false, source: null, last4: null },
      ...overrides,
    });
  }

  test("stay out of the way for a model the server has limits for", async () => {
    renderPage();
    const generation = await card("Generation model");

    expect(within(generation).queryByRole("group", { name: "Token limits" })).not.toBeInTheDocument();
    expect(within(generation).queryByLabelText("Context window")).not.toBeInTheDocument();
  });

  test("are asked for once validation finds that the server has none, and are saved with the model", async () => {
    const refused = localModel({ state: "not_validated", validation: LIMITS_UNKNOWN });
    const limited = localModel({ token_limits: { context_window_tokens: 32768, max_output_tokens: 4096 } });
    const { calls } = renderPage(makeModelSettings({ generation: kindSettings(null, refused) }), {
      [`PUT ${SETTINGS_ROUTE}/generation/candidate`]: () => kindSettings(null, { ...limited, state: "not_validated", validation: null }),
      [`POST ${SETTINGS_ROUTE}/generation/candidate/validate`]: () => ({
        passed: true,
        promoted: true,
        reindex_required: false,
        error_code: null,
        validation: limited.validation,
        settings: kindSettings(limited),
      }),
    });
    const generation = await card("Generation model");

    expect(within(generation).getByText("These settings failed validation and are not in use").closest('[role="status"]')).toHaveTextContent(
      "The model answered, but MemForge has no token limits for it. Enter its context window and maximum output, then validate again.",
    );
    const limits = within(generation).getByRole("group", { name: "Token limits" });
    expect(limits).toHaveAccessibleDescription(
      "MemForge sizes every request to these limits, so that the model is never sent more than it can read or asked for more than it can write. It has none for this model: take them from the model card or from your model server.",
    );
    await userEvent.type(within(limits).getByLabelText("Context window"), "32768");
    await userEvent.type(within(limits).getByLabelText("Maximum output"), "4096");
    await userEvent.click(within(generation).getByRole("button", { name: "Save and validate" }));

    expect(await within(generation).findByText("Validation passed")).toBeInTheDocument();
    expect(within(inUse(generation)).getByText("32,768 tokens")).toBeInTheDocument();
    expect(within(generation).getByLabelText("Context window")).toHaveValue("32768");
    expect(await bodyOf(calls, "PUT", `${SETTINGS_ROUTE}/generation/candidate`)).toEqual({
      provider: "ollama",
      model_id: "qwen3:14b",
      endpoint: "http://localhost:11434",
      auth_strategy: "none",
      options: {},
      parameters: {},
      token_limits: { context_window_tokens: 32768, max_output_tokens: 4096 },
    });
  });

  test("of a model that states them can be changed, as whole numbers of tokens", async () => {
    const limited = localModel({ token_limits: { context_window_tokens: 32768, max_output_tokens: 4096 } });
    const { calls } = renderPage(makeModelSettings({ generation: kindSettings(limited) }));
    const generation = await card("Generation model");
    const limits = within(generation).getByRole("group", { name: "Token limits" });

    expect(within(limits).getByLabelText("Context window")).toHaveValue("32768");
    expect(within(limits).getByLabelText("Maximum output")).toHaveValue("4096");
    expect(limits).toHaveAccessibleDescription(
      "MemForge sizes every request to these limits, so that the model is never sent more than it can read or asked for more than it can write.",
    );
    expect(within(generation).getByRole("button", { name: "Save and validate" })).toBeDisabled();

    await userEvent.clear(within(limits).getByLabelText("Maximum output"));
    await userEvent.type(within(limits).getByLabelText("Maximum output"), "8k");
    expect(within(generation).getByText("Unsaved changes")).toBeInTheDocument();
    await userEvent.click(within(generation).getByRole("button", { name: "Save and validate" }));

    expect(await within(limits).findByText("Enter a whole number greater than zero.")).toBeInTheDocument();
    expect(within(limits).getByLabelText("Maximum output")).toHaveAttribute("aria-invalid", "true");
    expect(writes(calls)).toEqual([]);
  });

  test("are not offered for an embedding model", async () => {
    renderPage();

    expect(within(await card("Embedding model")).queryByRole("group", { name: "Token limits" })).not.toBeInTheDocument();
  });
});
