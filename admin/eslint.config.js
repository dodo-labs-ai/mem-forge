import js from "@eslint/js";
import globals from "globals";
import reactHooks from "eslint-plugin-react-hooks";
import reactRefresh from "eslint-plugin-react-refresh";
import tseslint from "typescript-eslint";
import { defineConfig, globalIgnores } from "eslint/config";

/** Layers a module may not import, keyed by its own layer (ADR 0044). */
const LAYER_RULES = [
  { files: ["src/lib/**"], forbidden: ["@/api", "@/ui", "@/patterns", "@/features", "@/app"] },
  { files: ["src/api/**"], forbidden: ["@/ui", "@/patterns", "@/features", "@/app"] },
  { files: ["src/ui/**"], forbidden: ["@/api", "@/patterns", "@/features", "@/app"] },
  { files: ["src/patterns/**"], forbidden: ["@/api", "@/features", "@/app"] },
  { files: ["src/features/**"], forbidden: ["@/app"] },
];

const V1_IMPORT = {
  group: ["**/admin-ui/**"],
  message: "The V1 admin UI is removed after the parallel run; copy what you need (ADR 0044).",
};
const FEATURE_INTERNALS = {
  group: ["@/features/*/**"],
  message: "Import another feature through its index, for example '@/features/local-sync'.",
};

function layerRule(forbidden, extraPatterns = []) {
  return {
    "no-restricted-imports": [
      "error",
      {
        patterns: [
          V1_IMPORT,
          ...extraPatterns,
          ...forbidden.map((layer) => ({
            group: [layer, `${layer}/**`],
            message: "This layer may only import from the layers below it (ADR 0044).",
          })),
        ],
      },
    ],
  };
}

export default defineConfig([
  globalIgnores(["dist", "storybook-static", "playwright-report", "test-results", "src/api/schema.gen.ts"]),
  {
    files: ["**/*.{ts,tsx}"],
    extends: [js.configs.recommended, tseslint.configs.recommended, reactHooks.configs.flat.recommended, reactRefresh.configs.vite],
    languageOptions: { ecmaVersion: 2023, globals: globals.browser },
    rules: layerRule([], [FEATURE_INTERNALS]),
  },
  ...LAYER_RULES.map(({ files, forbidden }) => ({
    files,
    rules: layerRule(forbidden, files[0] === "src/features/**" ? [FEATURE_INTERNALS] : []),
  })),
  {
    // Generated shadcn primitives export variant helpers next to components.
    files: ["src/ui/**", "src/patterns/**", "src/app/extension/**", "src/api/ApiProvider.tsx"],
    rules: { "react-refresh/only-export-components": "off" },
  },
]);
