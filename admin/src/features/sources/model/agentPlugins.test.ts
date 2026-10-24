/// <reference types="node" />
import { readFileSync } from "node:fs";
import path from "node:path";
import { expect, test } from "vitest";
import { AGENT_PLUGIN_SETUPS } from "./agentPlugins";

/** The plugin README of a client, at the repository root's `integrations/<client>/memforge-memory/`. */
function pluginReadme(client: string): string {
  const root = path.resolve(import.meta.dirname, "../../../../..");
  return readFileSync(path.join(root, "integrations", client, "memforge-memory", "README.md"), "utf8");
}

test.each(AGENT_PLUGIN_SETUPS.map((setup) => [setup.name, setup] as const))(
  "the %s install commands are the ones its plugin README gives",
  (_name, setup) => {
    const lines = pluginReadme(setup.client).split("\n").map((line) => line.trim());
    const start = lines.indexOf(setup.commands[0]!);
    expect(start).toBeGreaterThanOrEqual(0);
    expect(lines.slice(start, start + setup.commands.length)).toEqual(setup.commands);
  },
);
