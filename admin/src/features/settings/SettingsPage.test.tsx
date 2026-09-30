import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Link, Outlet } from "react-router-dom";
import { afterEach, describe, expect, test, vi } from "vitest";
import type { Route } from "@/test/fakeApi";
import { renderRoutes } from "@/test/renderRoutes";
import type { LlmConfig } from "./model/types";
import { SettingsPage } from "./SettingsPage";

const HTTP_UNPROCESSABLE = 422;

const CONFIG: LlmConfig = {
  writable: true,
  enrichment_model: "anthropic/claude-sonnet-4-6",
  enrichment_base_url: "https://litellm.example.test/v1",
  enrichment_api_key: "********a91f",
  enrichment_api_key_set: true,
  enrichment_api_key_last4: "a91f",
  embedding_model: "nomic-embed-text",
  embedding_base_url: "http://localhost:11434/v1",
  embedding_api_key: null,
  embedding_api_key_set: false,
  embedding_api_key_last4: null,
};

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

function renderPage(config: LlmConfig = CONFIG, routes: Record<string, Route> = {}) {
  const api: Record<string, Route> = {
    "GET /api/v1/llm-config": () => config,
    "PUT /api/v1/llm-config": () => config,
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

async function card(name: string) {
  return screen.findByRole("region", { name });
}

function requestsTo(calls: Request[], method: string): Request[] {
  return calls.filter((request) => request.method === method && new URL(request.url).pathname.startsWith("/api/v1/llm-config"));
}

test("shows the stored endpoints and only the last characters of a saved key", async () => {
  renderPage();

  const enrichment = await card("Enrichment");
  expect(within(enrichment).getByLabelText("Base URL")).toHaveValue("https://litellm.example.test/v1");
  expect(within(enrichment).getByLabelText("Model")).toHaveValue("anthropic/claude-sonnet-4-6");
  expect(within(enrichment).getByLabelText("API key")).toHaveValue("");
  expect(within(enrichment).getByText("Saved key (****a91f) in use. Leave blank to keep it.")).toBeInTheDocument();
  expect(screen.queryByText(/\*{8}a91f/)).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();
});

test("names the card with unsaved changes and saves only that card", async () => {
  const { calls } = renderPage();
  const embedding = await card("Embedding");

  await userEvent.clear(within(embedding).getByLabelText("Model"));
  await userEvent.type(within(embedding).getByLabelText("Model"), "text-embedding-3-small");
  expect(screen.getByText("Unsaved changes in Embedding")).toBeInTheDocument();

  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

  expect(await screen.findByText("Saved")).toBeInTheDocument();
  const [put] = requestsTo(calls, "PUT");
  expect(await put!.clone().json()).toEqual({
    embedding_base_url: "http://localhost:11434/v1",
    embedding_model: "text-embedding-3-small",
  });
});

test("a failed save shows the server's reason and keeps the edits", async () => {
  renderPage(CONFIG, {
    "PUT /api/v1/llm-config": () =>
      Response.json({ detail: "Embedding model is required." }, { status: HTTP_UNPROCESSABLE }),
  });
  const embedding = await card("Embedding");

  await userEvent.clear(within(embedding).getByLabelText("Model"));
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

  const alert = await screen.findByRole("alert");
  expect(alert).toHaveTextContent("Couldn't save");
  expect(alert).toHaveTextContent("Embedding model is required.");
  expect(within(embedding).getByLabelText("Model")).toHaveValue("");
  expect(screen.getByText("Unsaved changes in Embedding")).toBeInTheDocument();
});

test("removing a saved key sends an empty key, and Undo keeps it", async () => {
  const { calls } = renderPage();
  const enrichment = await card("Enrichment");

  await userEvent.click(within(enrichment).getByRole("button", { name: "Remove saved key" }));
  expect(within(enrichment).getByText("Saved key will be removed on save.")).toBeInTheDocument();
  await userEvent.click(within(enrichment).getByRole("button", { name: "Undo key removal" }));
  expect(screen.getByRole("button", { name: "Save changes" })).toBeDisabled();

  await userEvent.click(within(enrichment).getByRole("button", { name: "Remove saved key" }));
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

  await screen.findByText("Saved");
  const [put] = requestsTo(calls, "PUT");
  expect((await put!.clone().json()).enrichment_api_key).toBe("");
});

test("an invalid base URL is reported before anything is sent", async () => {
  const { calls } = renderPage();
  const embedding = await card("Embedding");

  await userEvent.clear(within(embedding).getByLabelText("Base URL"));
  await userEvent.type(within(embedding).getByLabelText("Base URL"), "localhost:11434");
  await userEvent.click(screen.getByRole("button", { name: "Save changes" }));

  expect(await within(embedding).findByText("Enter a URL starting with http:// or https://.")).toBeInTheDocument();
  expect(within(embedding).getByLabelText("Base URL")).toHaveAttribute("aria-invalid", "true");
  expect(requestsTo(calls, "PUT")).toEqual([]);
});

describe("Test connection", () => {
  test("offers the fitting models and fills an empty model field", async () => {
    const { calls } = renderPage(CONFIG, {
      "POST /api/v1/llm-config/probe": () => ({
        ok: true,
        models_supported: true,
        models: [{ id: "text-embedding-3-small" }, { id: "gpt-5.4" }, { id: "nomic-embed-text" }],
        message: "Connected. Found 3 models.",
        latency_ms: 42,
      }),
    });
    const embedding = await card("Embedding");
    await userEvent.clear(within(embedding).getByLabelText("Model"));

    await userEvent.click(within(embedding).getByRole("button", { name: "Test connection" }));

    expect(await within(embedding).findByText("Connected. Found 3 models. (42 ms)")).toBeInTheDocument();
    expect(within(embedding).getByLabelText("Model")).toHaveValue("text-embedding-3-small");
    const suggestions = within(embedding).getByRole("group", { name: "Suggested models" });
    expect(within(suggestions).getAllByRole("button").map((button) => button.textContent)).toEqual([
      "text-embedding-3-small",
      "nomic-embed-text",
    ]);
    await userEvent.click(within(suggestions).getByRole("button", { name: "nomic-embed-text" }));
    expect(within(embedding).getByLabelText("Model")).toHaveValue("nomic-embed-text");

    const [probe] = requestsTo(calls, "POST");
    expect(await probe!.clone().json()).toEqual({ kind: "embedding", base_url: "http://localhost:11434/v1", api_key: null });
  });

  test("applies the base URL the server suggests", async () => {
    renderPage(CONFIG, {
      "POST /api/v1/llm-config/probe": () => ({
        ok: false,
        models_supported: false,
        stage: "connect",
        message: "Could not reach localhost.",
        suggested_base_url: "http://host.docker.internal:11434/v1",
      }),
    });
    const embedding = await card("Embedding");

    await userEvent.click(within(embedding).getByRole("button", { name: "Test connection" }));
    await userEvent.click(await within(embedding).findByRole("button", { name: "Use host.docker.internal" }));

    expect(within(embedding).getByLabelText("Base URL")).toHaveValue("http://host.docker.internal:11434/v1");
    expect(within(embedding).queryByText("Could not reach localhost.")).not.toBeInTheDocument();
  });
});

test("when the environment manages the settings, the page is read-only from the start", async () => {
  renderPage({ ...CONFIG, writable: false });

  expect(await screen.findByText("Managed by the deployment environment.")).toBeInTheDocument();
  const enrichment = await card("Enrichment");
  expect(within(enrichment).getByLabelText("Base URL")).toHaveAttribute("readonly");
  expect(within(enrichment).getByLabelText("Model")).toHaveAttribute("readonly");
  expect(within(enrichment).getByLabelText("API key")).toHaveValue("****a91f");
  expect(within(enrichment).getByRole("button", { name: "Test connection" })).toBeDisabled();
  expect(screen.queryByRole("button", { name: "Save changes" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Remove saved key" })).not.toBeInTheDocument();
});

describe("leaving with unsaved changes", () => {
  async function editEnrichmentModel() {
    const enrichment = await card("Enrichment");
    await userEvent.type(within(enrichment).getByLabelText("Model"), "-latest");
  }

  test("asks first, and Keep editing stays on the page", async () => {
    const { router } = renderPage();
    await editEnrichmentModel();

    await userEvent.click(screen.getByRole("link", { name: "Sources" }));
    const dialog = await screen.findByRole("dialog", { name: "Leave without saving?" });
    expect(dialog).toHaveTextContent("Your changes in Enrichment are not saved.");

    await userEvent.click(within(dialog).getByRole("button", { name: "Keep editing" }));
    expect(router.state.location.pathname).toBe("/settings");
    expect(within(await card("Enrichment")).getByLabelText("Model")).toHaveValue("anthropic/claude-sonnet-4-6-latest");
  });

  test("Leave without saving follows the link", async () => {
    renderPage();
    await editEnrichmentModel();

    await userEvent.click(screen.getByRole("link", { name: "Sources" }));
    await userEvent.click(await screen.findByRole("button", { name: "Leave without saving" }));

    expect(await screen.findByRole("heading", { name: "Sources page" })).toBeInTheDocument();
  });

  test("reloading or closing the tab gets the browser's prompt", async () => {
    renderPage();
    await card("Enrichment");
    const cleanUnload = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(cleanUnload);
    expect(cleanUnload.defaultPrevented).toBe(false);

    await editEnrichmentModel();
    const dirtyUnload = new Event("beforeunload", { cancelable: true });
    window.dispatchEvent(dirtyUnload);
    expect(dirtyUnload.defaultPrevented).toBe(true);
  });

  test("a saved page leaves without asking", async () => {
    renderPage();
    await editEnrichmentModel();
    await userEvent.click(screen.getByRole("button", { name: "Save changes" }));
    await screen.findByText("Saved");

    await userEvent.click(screen.getByRole("link", { name: "Sources" }));

    expect(await screen.findByRole("heading", { name: "Sources page" })).toBeInTheDocument();
  });
});
