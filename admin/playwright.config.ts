import { defineConfig, devices } from "@playwright/test";

const DEFAULT_PREVIEW_PORT = 4175;
/** Set `ADMIN_E2E_PORT` to run the suite while the default port is taken. */
const PREVIEW_PORT = Number(process.env.ADMIN_E2E_PORT ?? DEFAULT_PREVIEW_PORT);

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
