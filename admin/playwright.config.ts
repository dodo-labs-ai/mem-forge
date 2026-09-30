import { defineConfig, devices } from "@playwright/test";

const PREVIEW_PORT = 4175;

/**
 * Smoke tests run against the production build. Each test stubs the API with
 * `page.route`, so no backend is needed.
 */
export default defineConfig({
  testDir: "e2e",
  use: { baseURL: `http://localhost:${PREVIEW_PORT}`, trace: "retain-on-failure" },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: {
    command: `npm run build && npx vite preview --port ${PREVIEW_PORT} --strictPort`,
    port: PREVIEW_PORT,
    reuseExistingServer: false,
  },
});
