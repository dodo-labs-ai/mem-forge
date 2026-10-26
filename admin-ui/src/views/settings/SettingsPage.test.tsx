import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { resourceClient } from "@/api/client";
import type { ModelSettings } from "@/api/types";
import { SettingsPage } from "@/views/settings/SettingsPage";

vi.mock("@/api/client", () => ({
  resourceClient: { get: vi.fn() },
}));

const getMock = vi.mocked(resourceClient.get);

const SETTINGS: ModelSettings = {
  writable: true,
  generation: {
    active: {
      kind: "generation",
      provider: "openai",
      endpoint: "https://api.openai.com/v1",
      model_id: "gpt-5.4",
      state: "ready",
    },
  },
  embedding: { active: null },
  vector_index: { status: "not_configured", index: null },
};

const EMBEDDING: ModelSettings["embedding"] = {
  active: {
    kind: "embedding",
    provider: "ollama",
    endpoint: "http://localhost:11434",
    model_id: "nomic-embed-text",
    state: "ready",
  },
};

function renderPage(settings: ModelSettings) {
  getMock.mockResolvedValue({ data: settings });
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={queryClient}>
      <SettingsPage />
    </QueryClientProvider>,
  );
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

it("shows the models in use and where they are changed, with nothing to edit", async () => {
  renderPage(SETTINGS);

  const generation = await screen.findByRole("region", { name: "Generation model" });
  expect(getMock).toHaveBeenCalledWith("/model-settings");
  expect(within(generation).getByText("openai")).toBeTruthy();
  expect(within(generation).getByText("https://api.openai.com/v1")).toBeTruthy();
  expect(within(generation).getByText("gpt-5.4")).toBeTruthy();
  expect(within(generation).getByText("Ready")).toBeTruthy();
  expect(within(screen.getByRole("region", { name: "Embedding model" })).getByText("Not set up")).toBeTruthy();

  expect(screen.getByRole("link", { name: "Settings in the new admin" }).getAttribute("href")).toBe("/settings");
  expect(screen.queryByRole("textbox")).toBeNull();
  expect(screen.queryByRole("button")).toBeNull();
});

it("says how many memories the search index holds, under the embedding model that built it", async () => {
  renderPage({ ...SETTINGS, embedding: EMBEDDING, vector_index: { status: "ready", index: { vector_count: 1204 } } });

  const embedding = await screen.findByRole("region", { name: "Embedding model" });
  expect(within(embedding).getByText("Search index: ready, with 1,204 memories.")).toBeTruthy();
  expect(within(screen.getByRole("region", { name: "Generation model" })).queryByText(/Search index/)).toBeNull();
});

it("says that vector search is off until the search index is rebuilt in the new admin", async () => {
  renderPage({ ...SETTINGS, embedding: EMBEDDING, vector_index: { status: "reindex_required", index: null } });

  const embedding = await screen.findByRole("region", { name: "Embedding model" });
  expect(
    within(embedding).getByText(
      "Search index: has to be rebuilt for this model. Vector search is off until it is; start the rebuild in Settings in the new admin.",
    ),
  ).toBeTruthy();
  expect(screen.queryByRole("button")).toBeNull();
});

it("says nothing about a search index the deployment keeps by its own means", async () => {
  renderPage({ ...SETTINGS, embedding: EMBEDDING, vector_index: { status: "not_managed", index: null } });

  await screen.findByRole("region", { name: "Embedding model" });
  expect(screen.queryByText(/Search index/)).toBeNull();
});

it("says so when the deployment environment manages the models", async () => {
  renderPage({ ...SETTINGS, writable: false });

  expect(
    await screen.findByText("The deployment environment manages these models, so they cannot be changed in the admin."),
  ).toBeTruthy();
  expect(screen.queryByRole("link", { name: "Settings in the new admin" })).toBeNull();
});
