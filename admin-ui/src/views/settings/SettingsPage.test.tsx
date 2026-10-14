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

  expect(screen.getByRole("link", { name: "Settings in the new admin" }).getAttribute("href")).toBe("/v2/settings");
  expect(screen.queryByRole("textbox")).toBeNull();
  expect(screen.queryByRole("button")).toBeNull();
});

it("says so when the deployment environment manages the models", async () => {
  renderPage({ ...SETTINGS, writable: false });

  expect(
    await screen.findByText("The deployment environment manages these models, so they cannot be changed in the admin."),
  ).toBeTruthy();
  expect(screen.queryByRole("link", { name: "Settings in the new admin" })).toBeNull();
});
