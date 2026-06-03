import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { buildLocalMarkdownPushCommand } from "../src/views/sources/localMarkdownConfig.js";

assert.equal(
  buildLocalMarkdownPushCommand({ vaultId: "engineering", sourceId: "src-abc" }),
  "memforge adapter kb push engineering --source-id src-abc",
);

assert.equal(
  buildLocalMarkdownPushCommand({ vaultId: "", sourceId: null }),
  "memforge adapter kb push <vault-id> --source-id <source-id>",
);

assert.equal(
  buildLocalMarkdownPushCommand({ vaultId: "  spaces  ", sourceId: "  src-1  " }),
  "memforge adapter kb push spaces --source-id src-1",
);

const sourcesPageSource = readFileSync("src/views/sources/SourcesPage.tsx", "utf8");

assert.match(
  sourcesPageSource,
  /local_markdown:\s*{[^}]*Local Repository/,
  "SourcesPage should label local_markdown sources with a friendly name",
);

assert.match(
  sourcesPageSource,
  /local_markdown:\s*"files"/,
  "SourcesPage should describe local_markdown items as 'files'",
);

const sourceConfigDialogSource = readFileSync("src/views/sources/SourceConfigDialog.tsx", "utf8");

assert.match(
  sourceConfigDialogSource,
  /LocalMarkdownPushPanel/,
  "SourceConfigDialog should render the local-markdown push panel",
);

assert.match(
  sourceConfigDialogSource,
  /MemForge does not read your filesystem/,
  "Local-markdown push panel should explain that the service does not read local files",
);

const jiraGenePy = readFileSync("../src/memforge/genes/jira_gene.py", "utf8");
assert.match(
  jiraGenePy,
  /local CLI adapter/,
  "Jira config schema should reference the local CLI adapter for browser-session auth",
);
