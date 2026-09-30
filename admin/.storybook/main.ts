import type { StorybookConfig } from "@storybook/react-vite";

/** The catalogue covers the shared layers only: primitives and patterns. */
const config: StorybookConfig = {
  stories: ["../src/ui/**/*.stories.tsx", "../src/patterns/**/*.stories.tsx"],
  framework: { name: "@storybook/react-vite", options: {} },
  core: { disableTelemetry: true },
  viteFinal: (viteConfig) => ({ ...viteConfig, base: "./" }),
};

export default config;
