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
  makeModelSettings,
  makeValidation,
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

function renderPage(settings: ModelSettings = makeModelSettings(), routes: Record<string, Route> = {}) {
  const api: Record<string, Route> = {
    [`GET ${SETTINGS_ROUTE}`]: () => settings,
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

async function card(name: "Generation model" | "Embedding model") {
  return screen.findByRole("region", { name });
}

function inUse(region: HTMLElement) {
  return within(region).getByRole("group", { name: "In use" });
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
    const { calls } = renderPage(makeModelSettings(), {
      [`PUT ${SETTINGS_ROUTE}/embedding/candidate`]: () =>
        kindSettings(makeEmbeddingBinding(), { ...promoted, state: "not_validated", validation: null }),
      [`POST ${SETTINGS_ROUTE}/embedding/candidate/validate`]: () => ({
        passed: true,
        promoted: true,
        error_code: null,
        validation: promoted.validation,
        settings: kindSettings(promoted),
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
    const edit = openAiEmbedding({ state: "not_validated", validation: null });
    const { calls } = renderPage(makeModelSettings({ embedding: kindSettings(makeEmbeddingBinding(), edit) }), {
      [`POST ${SETTINGS_ROUTE}/embedding/candidate/validate`]: () => ({
        passed: true,
        promoted: true,
        validation: makeValidation(),
        settings: kindSettings(openAiEmbedding()),
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
    const edit = openAiEmbedding({ state: "not_validated", validation: null });
    const { calls } = renderPage(makeModelSettings({ embedding: kindSettings(makeEmbeddingBinding(), edit) }), {
      [`DELETE ${SETTINGS_ROUTE}/embedding/candidate`]: () => kindSettings(makeEmbeddingBinding()),
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

  test("says what the server still needs when the model's token limits are unknown", async () => {
    const local = makeBinding({ validation: makeValidation({ facts: { latency_ms: 300, token_limits_known: false } }) });
    renderPage(makeModelSettings({ generation: kindSettings(local) }));

    expect(await screen.findByText(/set MEMFORGE_LLM_MAX_INPUT_TOKENS/)).toBeInTheDocument();
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
