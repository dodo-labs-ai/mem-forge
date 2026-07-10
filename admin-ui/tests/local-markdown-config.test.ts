import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { isPushBasedSourceType } from "../src/views/sources/managedSources.js";

assert.equal(isPushBasedSourceType("local_markdown"), true);

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

assert.doesNotMatch(
  sourcesPageSource,
  /const isPushBased = isPushBasedSourceType\(gene\.name\)/,
  "Add Source configurable cards should not turn local_markdown into a setup-only card",
);

const sourceConfigDialogSource = readFileSync("src/views/sources/SourceConfigDialog.tsx", "utf8");

assert.match(
  sourceConfigDialogSource,
  /local_markdown_preview_tree/,
  "SourceConfigDialog should preview local_markdown through the local-agent queue",
);

assert.match(
  sourceConfigDialogSource,
  /local_markdown_pick_root/,
  "SourceConfigDialog should let local_markdown choose a folder through the local-agent queue",
);

assert.match(
  sourceConfigDialogSource,
  /Choose folder/,
  "SourceConfigDialog should expose a friendly folder picker action",
);

assert.match(
  sourceConfigDialogSource,
  /pollLocalAgentPreviewJob/,
  "SourceConfigDialog should poll the local-agent preview job",
);
assert.doesNotMatch(
  sourceConfigDialogSource,
  /sourceType !== "local_markdown"/,
  "local_markdown preview should be reachable in the source dialog",
);
assert.doesNotMatch(
  sourceConfigDialogSource,
  /LocalRepoSetupInstructions/,
  "local_markdown should use the normal source form instead of the legacy CLI-only setup panel",
);

const jiraGenePy = readFileSync("../src/memforge/genes/jira_gene.py", "utf8");
assert.match(
  jiraGenePy,
  /local CLI adapter/,
  "Jira config schema should reference the local CLI adapter for browser-session auth",
);
assert.match(
  jiraGenePy,
  /local_agent/,
  "Jira config schema should expose local daemon sync mode",
);

const localMarkdownGenePy = readFileSync("../src/memforge/genes/local_markdown_gene.py", "utf8");
assert.match(
  localMarkdownGenePy,
  /Folder Path/,
  "local_markdown config schema should let the UI collect a daemon-side folder path",
);

assert.doesNotMatch(
  localMarkdownGenePy,
  /Vault ID|vault-id you set|memforge adapter kb/,
  "local_markdown config schema should not expose the internal vault id in the normal source form",
);

const githubRepoGenePy = readFileSync("../src/memforge/genes/github_repo_gene.py", "utf8");
assert.match(
  githubRepoGenePy,
  /key="repo_path"/,
  "github_repo local mode should persist the local clone path in source config",
);

assert.match(
  sourceConfigDialogSource,
  /github_repo_pick_root/,
  "SourceConfigDialog should let github_repo choose a local clone through the local-agent queue",
);

assert.doesNotMatch(
  sourceConfigDialogSource,
  /memforge adapter github/,
  "github_repo local mode should not expose legacy CLI profile commands",
);
