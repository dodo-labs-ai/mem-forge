import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const componentSource = readFileSync(
  "src/views/sources/LocalAgentDaemonStatus.tsx",
  "utf8",
);
const querySource = readFileSync("src/views/sources/localAgentDaemonStatusQuery.ts", "utf8");
const sourcesPageSource = readFileSync("src/views/sources/SourcesPage.tsx", "utf8");
const sourceRowSource = readFileSync("src/views/sources/SourceRow.tsx", "utf8");
const schemaSourceSetupSource = readFileSync(
  "src/views/sources/SchemaSourceSetup.tsx",
  "utf8",
);
const topbarSource = readFileSync("src/components/layout/Topbar.tsx", "utf8");
const apiTypesSource = readFileSync("src/api/types.ts", "utf8");

// --- Component contract --------------------------------------------------

assert.match(
  querySource,
  /getLocalAgentDaemonStatus/,
  "LocalAgentDaemonStatus should read the controller-derived local-agent host endpoint",
);

assert.match(
  querySource,
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
assert.match(
  querySource,
  /export function useLocalAgentDaemonStatus/,
  "Local-agent daemon status should expose one shared query hook for compact and full status UI",
);
assert.match(
  componentSource,
  /export function LocalAgentDaemonStatus/,
  "Local-agent daemon status should keep the detailed Add Source presentation",
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
// only appear there, alongside sources that use the local sync app.
assert.match(
  sourcesPageSource,
  /import \{ LocalAgentDaemonStatus \}/,
  "SourcesPage should import the daemon status indicator for the Add Source flow",
);

assert.match(
  sourcesPageSource,
  /Select a source to configure its connection and sync scope\.[\s\S]*<LocalAgentDaemonStatus \/>/,
  "Add Source source grid should surface the local sync status inline",
);

const localAgentStatusUsages = sourcesPageSource.match(/<LocalAgentDaemonStatus \/>/g) ?? [];
assert.equal(
  localAgentStatusUsages.length,
  1,
  "SourcesPage should render the daemon status exactly once, inside the Add Source flow",
);

assert.match(
  sourceRowSource,
  /SourceReadinessBadge/,
  "Source rows should use the shared local source readiness presenter",
);
assert.match(
  sourceRowSource,
  /const localExecution = isLocalAgentBackedSource\(source\);[\s\S]*const showReadiness = !isPaused[\s\S]*capabilities\.can_sync/,
  "Paused lifecycle and execution ownership should gate source readiness",
);
assert.match(
  sourceRowSource,
  /<SourceLifecycleBadge status=\{source\.status\} \/>[\s\S]*showReadiness &&/,
  "Source lifecycle should remain visible independently of local readiness",
);

// --- Configure dialog ----------------------------------------------------

assert.doesNotMatch(
  schemaSourceSetupSource,
  /LocalAgentDaemonStatus/,
  "source setup should not surface daemon status inside an individual source form",
);
