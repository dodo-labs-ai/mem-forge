// Verifies the menu spec without booting Clack: the script source must expose
// the two-tier area -> action structure, route each action through the
// canonical `memforge` subcommand (never re-implementing them), and use the
// async `spawn` runner so spinners animate.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import path from "node:path";
import url from "node:url";

const here = path.dirname(url.fileURLToPath(import.meta.url));
const source = readFileSync(path.join(here, "..", "index.mjs"), "utf-8");

const requiredLabels = [
  // Areas
  "Connect a MemForge server",
  "Local repository",
  "GitHub repository",
  "Jira",
  "Search memory",
  "Status & diagnostics",
  // Actions
  "Connect a server",
  "Switch active server",
  "Health check",
  "Set up a repository",
  "Sync now",
  "Preview (dry run)",
  "Schedule sync",
  "Remove repository",
  "Manage repositories",
  "Set up GitHub repository",
  "Manage GitHub repositories",
  "Public internet",
  "Internal network / VPN",
  "Preview selected files",
  "Check session status",
  "Authenticate browser session",
  "Forget a session",
  "Start background refresh",
  "Run diagnostics",
  "← Back",
];

for (const label of requiredLabels) {
  assert.ok(source.includes(`label: "${label}"`), `menu must contain label ${JSON.stringify(label)}`);
}

const requiredCommands = [
  ['"target", "add"', "Connect server wraps `memforge target add`"],
  ['"target", "use"', "Connect/Switch server wraps `memforge target use`"],
  ['"target", "list"', "Switch server wraps `memforge target list`"],
  ['"target", "check"', "Health check wraps `memforge target check`"],
  ['"adapter", "kb", "add"', "Vault setup wraps `memforge adapter kb add`"],
  ['"adapter", "kb", "scan"', "Wizard instant feedback wraps `memforge adapter kb scan`"],
  ['"adapter", "kb", "preview"', "Preview wraps `memforge adapter kb preview`"],
  ['"adapter", "kb", "push"', "Sync wraps `memforge adapter kb push`"],
  ['"adapter", "kb", "schedule"', "Schedule wraps `memforge adapter kb schedule`"],
  ['"adapter", "kb", "unschedule"', "Manage schedules wraps `memforge adapter kb unschedule`"],
  ['"adapter", "kb", "remove"', "Manage vaults wraps `memforge adapter kb remove`"],
  ['"adapter", "kb", "list"', "Manage vaults / status wraps `memforge adapter kb list`"],
  ['"adapter", "github", "list"', "GitHub profile discovery wraps `memforge adapter github list`"],
  ['"adapter", "github", "preview"', "GitHub preview wraps `memforge adapter github preview`"],
  ['"adapter", "github", "push"', "GitHub sync wraps `memforge adapter github push`"],
  ['"adapter", "github", "remove"', "GitHub profile removal wraps `memforge adapter github remove`"],
  ['"adapter", "auth", "jira", "status"', "Jira status wraps `adapter auth jira status`"],
  ['"adapter", "auth", "jira", "refresh"', "Jira auth wraps `adapter auth jira refresh`"],
  ['"adapter", "auth", "jira", "forget"', "Forget wraps `adapter auth jira forget`"],
  ['"adapter", "auth", "jira", "watch"', "Background refresh wraps `adapter auth jira watch`"],
  ['"adapter", "auth", provider, "list"', "Origin picker wraps `adapter auth <provider> list`"],
  ['"memory", "search"', "Search wraps `memforge memory search`"],
  ['"adapter", "status"', "Status wraps `memforge adapter status`"],
  ['"--create-source"', "Vault setup links the server source end-to-end"],
];

for (const [needle, message] of requiredCommands) {
  assert.ok(source.includes(needle), message);
}

assert.match(
  source,
  /"adapter",\s*"github",\s*"add"/,
  "GitHub setup wraps `memforge adapter github add`",
);

assert.ok(
  !/\[\s*"auth",\s*"jira"/.test(source),
  "interactive UI must use `adapter auth jira`, never the removed `auth jira`",
);

assert.ok(!source.includes("local_push"), "interactive UI must not expose internal GitHub connection mode names");
assert.ok(!source.includes("cloud_pull"), "interactive UI must not expose internal GitHub connection mode names");

assert.ok(
  !source.includes("spawnSync"),
  "use async spawn, not spawnSync, so spinners animate during shell-outs",
);
assert.ok(/\bspawn\b/.test(source), "must shell out via spawn");

assert.ok(source.includes("MEMFORGE_NO_INTERACTIVE"), "spawn must set MEMFORGE_NO_INTERACTIVE to prevent recursion");
assert.ok(source.includes("MEMFORGE_CLI_BIN"), "must read the canonical binary path from MEMFORGE_CLI_BIN");
