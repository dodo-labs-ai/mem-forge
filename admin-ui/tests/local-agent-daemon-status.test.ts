import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const componentSource = readFileSync(
  "src/views/sources/LocalAgentDaemonStatus.tsx",
  "utf8",
);
const sourcesPageSource = readFileSync("src/views/sources/SourcesPage.tsx", "utf8");
const sourceConfigDialogSource = readFileSync(
  "src/views/sources/SourceConfigDialog.tsx",
  "utf8",
);
const apiTypesSource = readFileSync("src/api/types.ts", "utf8");

// --- Component contract --------------------------------------------------

assert.match(
  componentSource,
  /"\/api\/cloud\/local-agent\/status"/,
  "LocalAgentDaemonStatus should read the cloud daemon status endpoint",
);

assert.match(
  componentSource,
  /useQuery<LocalAgentDaemonStatusResponse>/,
  "LocalAgentDaemonStatus should fetch the daemon status through TanStack Query",
);

assert.match(
  componentSource,
  /Local daemon online/,
  "LocalAgentDaemonStatus should surface an online label in product language",
);

assert.match(
  componentSource,
  /Local daemon offline/,
  "LocalAgentDaemonStatus should surface an offline label in product language",
);

assert.match(
  componentSource,
  /memforge adapter daemon run/,
  "LocalAgentDaemonStatus offline state should show the daemon command",
);

assert.doesNotMatch(
  componentSource,
  /stale_after_seconds/,
  "LocalAgentDaemonStatus should not render raw debug payload fields",
);

assert.doesNotMatch(
  componentSource,
  /checked_at/,
  "LocalAgentDaemonStatus should not render raw debug payload fields",
);

assert.match(
  componentSource,
  /timeAgo\(data\.last_seen_at\)/,
  "LocalAgentDaemonStatus should show last seen in the shared friendly format",
);

// --- Types ---------------------------------------------------------------

assert.match(
  apiTypesSource,
  /export interface LocalAgentDaemonStatusResponse\s*\{[\s\S]*?status:\s*"online"\s*\|\s*"offline"/,
  "types.ts should export the daemon status response shape used by the admin UI",
);

// --- Sources list --------------------------------------------------------

assert.match(
  sourcesPageSource,
  /import \{ LocalAgentDaemonStatus \}/,
  "SourcesPage should import the daemon status indicator",
);

assert.match(
  sourcesPageSource,
  /function sourceUsesLocalAgent\(source: Source\)/,
  "SourcesPage should have a predicate for sources that depend on the local daemon",
);

assert.match(
  sourcesPageSource,
  /hasLocalAgentSource\s*=\s*sources\.some\(sourceUsesLocalAgent\)/,
  "SourcesPage should compute whether any configured source depends on the local daemon",
);

assert.match(
  sourcesPageSource,
  /hasLocalAgentSource\s*&&\s*\(\s*<div[\s\S]*?<LocalAgentDaemonStatus \/>/,
  "SourcesPage should only surface the daemon status when at least one local-agent source is configured",
);

// --- Add Source dialog ---------------------------------------------------

assert.match(
  sourcesPageSource,
  /<SectionDivider label="Push from your local device" \/>\s*<LocalAgentDaemonStatus \/>/,
  "Add Source push-from-local section should surface the daemon status inline",
);

// --- Configure dialog ----------------------------------------------------

assert.match(
  sourceConfigDialogSource,
  /import \{ LocalAgentDaemonStatus \}/,
  "SourceConfigDialog should import the daemon status indicator",
);

assert.match(
  sourceConfigDialogSource,
  /usesLocalAgent\s*=\s*[\s\S]*?sourceType === "local_markdown"[\s\S]*?sourceType === "github_repo"[\s\S]*?"local_push"[\s\S]*?sourceType === "jira"[\s\S]*?"local_agent"/,
  "SourceConfigDialog should mark local_markdown, GitHub local_push, and Jira local_agent configurations as daemon-backed",
);

assert.match(
  sourceConfigDialogSource,
  /\{usesLocalAgent && <LocalAgentDaemonStatus \/>\}/,
  "SourceConfigDialog should render the daemon status only for daemon-backed source configurations",
);
