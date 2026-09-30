import path from "node:path";
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { APP_BASE_PATH } from "./src/app/basePath.ts";

const DEV_PORT = 5175;
const DEV_API_TARGET = "http://localhost:8765";
// Deprecated (admin-ui-v1): in development, links to pages not yet built here
// open the V1 dev server (`npm run dev` in admin-ui). Remove with V1 (ADR 0044).
const V1_DEV_TARGET = "http://localhost:5174";
const V1_DEV_PATHS = "^/(?!v2(/|$)|api/)";

export default defineConfig({
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
  build: { outDir: "dist", emptyOutDir: true },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    include: ["src/**/*.test.{ts,tsx}"],
    css: false,
  },
});
