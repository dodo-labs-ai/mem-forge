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
const topbarSource = readFileSync("src/components/layout/Topbar.tsx", "utf8");
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
  /Local sync ready/,
  "LocalAgentDaemonStatus should surface the online label in product language",
);

assert.match(
  componentSource,
  /Local sync unavailable/,
  "LocalAgentDaemonStatus should surface the offline label in product language",
);

assert.doesNotMatch(
  componentSource,
  /Local daemon online|Local daemon offline/,
  "LocalAgentDaemonStatus should not surface the legacy daemon-oriented status labels",
);

assert.match(
  componentSource,
  /memforge adapter daemon run/,
  "LocalAgentDaemonStatus offline state should still surface the daemon start command in the Add Source flow",
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

// --- Topbar --------------------------------------------------------------

assert.doesNotMatch(
  topbarSource,
  /<span>API<\/span>/,
  "Topbar should not surface a hardcoded API status badge",
);

assert.doesNotMatch(
  topbarSource,
  /LocalAgentDaemonStatus/,
  "Topbar should not render a local sync status chip",
);

// --- Sources list --------------------------------------------------------

assert.doesNotMatch(
  sourcesPageSource,
  /hasLocalAgentSource/,
  "SourcesPage should no longer surface a prominent daemon status strip above the configured sources list",
);

// The Add Source dialog lives inside SourcesPage; the daemon status should
// only appear there, alongside the "Push from your local device" selection.
assert.match(
  sourcesPageSource,
  /import \{ LocalAgentDaemonStatus \}/,
  "SourcesPage should import the daemon status indicator for the Add Source flow",
);

assert.match(
  sourcesPageSource,
  /<SectionDivider label="Push from your local device" \/>\s*<LocalAgentDaemonStatus \/>/,
  "Add Source push-from-local section should surface the daemon status inline",
);

const localAgentStatusUsages = sourcesPageSource.match(/<LocalAgentDaemonStatus \/>/g) ?? [];
assert.equal(
  localAgentStatusUsages.length,
  1,
  "SourcesPage should render the daemon status exactly once, inside the Add Source flow",
);

// --- Configure dialog ----------------------------------------------------

assert.doesNotMatch(
  sourceConfigDialogSource,
  /LocalAgentDaemonStatus/,
  "SourceConfigDialog should not surface daemon status on the source configuration form",
);
