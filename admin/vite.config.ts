import path from "node:path";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { APP_BASE_PATH } from "./src/app/basePath.ts";

const DEV_PORT = 5175;
const DEV_API_TARGET = "http://localhost:8765";
// Deprecated (admin-ui-v1): in development, every path outside /v2/ and /api/
// opens the V1 dev server (`npm run dev` in admin-ui), as the image serves V1
// beside V2. Source setup still links there. Remove with V1.
const V1_DEV_TARGET = "http://localhost:5174";
const V1_DEV_PATHS = "^/(?!v2(/|$)|api/)";

/**
 * `--mode e2e` builds what the Playwright suite runs against: the app as
 * production builds it, plus the harness pages under `e2e/harness/`, into a
 * directory of its own so `dist` never holds a harness.
 */
const E2E_MODE = "e2e";
const E2E_OUT_DIR = "node_modules/.tmp/e2e-dist";
const E2E_HARNESS_PAGES = ["e2e/harness/shared-workspace.html"];

export default defineConfig(({ mode }) => ({
  base: APP_BASE_PATH,
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(import.meta.dirname, "src") },
  },
  server: {
    port: DEV_PORT,
    strictPort: true,
    proxy: {
      "/api": { target: DEV_API_TARGET, changeOrigin: true },
      [V1_DEV_PATHS]: { target: V1_DEV_TARGET, changeOrigin: true },
    },
  },
  preview: { port: DEV_PORT },
  build:
    mode === E2E_MODE
      ? {
          outDir: E2E_OUT_DIR,
          emptyOutDir: true,
          rolldownOptions: {
            input: ["index.html", ...E2E_HARNESS_PAGES].map((page) => path.resolve(import.meta.dirname, page)),
          },
        }
      : { outDir: "dist", emptyOutDir: true },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    css: false,
  },
}));
