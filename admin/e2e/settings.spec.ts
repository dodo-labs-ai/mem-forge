import { expect, test, type Page } from "@playwright/test";

const CONFIG = {
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

const RESPONSES: Record<string, unknown> = {
  "/api/v1/sources": { data: [] },
  "/api/v1/projects": { data: [], can_manage: true },
  "/api/v1/genes": [],
  "/api/v1/source-list/preferences": { sort_mode: "name" },
  "/api/cloud/local-agent/status": { status: "offline", last_seen_at: null, checked_at: "", stale_after_seconds: 60 },
  "/api/cloud/local-agent/jobs/current": { data: [] },
};

/** Stubs the API; `saved` collects the bodies of settings updates. */
async function stubApi(page: Page, config: typeof CONFIG) {
  const saved: unknown[] = [];
  await page.route("**/api/**", (route) => {
    const request = route.request();
    const { pathname } = new URL(request.url());
    if (pathname === "/api/v1/llm-config") {
      if (request.method() === "PUT") {
        saved.push(request.postDataJSON());
        return route.fulfill({ json: { ...config, ...(request.postDataJSON() as object) } });
      }
      return route.fulfill({ json: config });
    }
    const body = RESPONSES[pathname];
    return body === undefined ? route.fulfill({ status: 404, json: { detail: "Not stubbed" } }) : route.fulfill({ json: body });
  });
  return saved;
}

test("Settings saves the changed card and asks before leaving with unsaved changes", async ({ page }) => {
  const saved = await stubApi(page, CONFIG);
  await page.goto("/v2/settings");

  await expect(page.getByRole("heading", { name: "Settings" })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Settings" })).toHaveAttribute("aria-current", "page");
  await expect(page).toHaveTitle("Settings · MemForge");

  const embedding = page.getByRole("region", { name: "Embedding" });
  await embedding.getByLabel("Model").fill("text-embedding-3-small");
  await expect(page.getByText("Unsaved changes in Embedding")).toBeVisible();

  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Sources" }).click();
  const dialog = page.getByRole("dialog", { name: "Leave without saving?" });
  await expect(dialog).toBeVisible();
  await dialog.getByRole("button", { name: "Keep editing" }).click();
  await expect(page).toHaveURL(/\/v2\/settings$/);

  await page.getByRole("button", { name: "Save changes" }).click();
  await expect(page.getByText("Saved", { exact: true })).toBeVisible();
  expect(saved).toEqual([{ embedding_base_url: "http://localhost:11434/v1", embedding_model: "text-embedding-3-small" }]);

  await page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Sources" }).click();
  await expect(page).toHaveURL(/\/v2\/sources$/);
});

test("Settings managed by the environment is read-only", async ({ page }) => {
  await stubApi(page, { ...CONFIG, writable: false });
  await page.goto("/v2/settings");

  await expect(page.getByText("Managed by the deployment environment.")).toBeVisible();
  await expect(page.getByRole("region", { name: "Enrichment" }).getByLabel("API key")).toHaveValue("****a91f");
  await expect(page.getByRole("region", { name: "Enrichment" }).getByLabel("Base URL")).not.toBeEditable();
  await expect(page.getByRole("button", { name: "Save changes" })).toHaveCount(0);
});
