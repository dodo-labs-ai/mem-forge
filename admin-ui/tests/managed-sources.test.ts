import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  canConfigureSourceType,
  canDeleteSourceType,
  isManagedSourceType,
  userConfigurableGenes,
} from "../src/views/sources/managedSources.js";
import {
  presentSourceConnection,
} from "../src/views/sources/sourceConnectionPresentation.js";

const genes = [
  { name: "confluence", display_name: "Confluence", execution_kinds: ["server"] as const },
  { name: "agent_session", display_name: "Agent Session", execution_kinds: [] as const },
  { name: "jira", display_name: "Jira", execution_kinds: ["server", "local_agent"] as const },
  { name: "teams", display_name: "Teams", execution_kinds: ["local_agent"] as const },
];

assert.equal(isManagedSourceType("agent_session"), true);
assert.equal(canConfigureSourceType("agent_session"), false);
assert.equal(canDeleteSourceType("agent_session"), false);
assert.equal(canConfigureSourceType("confluence"), true);
assert.equal(canDeleteSourceType("confluence"), true);
assert.equal(canConfigureSourceType("github_repo"), true);
assert.equal(canDeleteSourceType("github_repo"), true);
assert.deepEqual(userConfigurableGenes(genes).map((gene) => gene.name), ["confluence", "jira", "teams"]);
assert.deepEqual(
  userConfigurableGenes(genes).map((gene) => presentSourceConnection(gene)),
  [
    { mode: "direct", label: "Cloud" },
    { mode: "choice", label: "Cloud or local" },
    { mode: "device", label: "Local sync" },
  ],
);
assert.throws(
  () => presentSourceConnection(genes[1]),
  /must declare at least one execution kind/,
);

const sourcesPageSource = readFileSync("src/views/sources/SourcesPage.tsx", "utf8");

assert.match(
  sourcesPageSource,
  /max-h-\[calc\(100dvh-2rem\)\] overflow-y-auto/,
  "Add Source dialog should stay scrollable inside the visible viewport",
);

assert.match(
  sourcesPageSource,
  /userConfigurableGenes\(genes\)/,
  "the add-source dialog should filter service-managed genes before rendering source cards",
);

assert.match(
  sourcesPageSource,
  /capabilities\.can_configure/,
  "source cards should gate Configure on backend-issued capabilities, not by deriving role/creator locally",
);

assert.match(
  sourcesPageSource,
  /capabilities\.can_delete/,
  "source action menus should gate Delete on backend-issued capabilities, not by deriving role/creator locally",
);

assert.match(
  sourcesPageSource,
  /capabilities\.can_subscribe/,
  "source rows should expose a per-viewer subscription affordance when the backend allows it",
);

assert.match(
  sourcesPageSource,
  /resourceClient\.put\(`\/sources\/\$\{[^}]+\}\/subscription/,
  "the page should call the per-source subscription endpoint when a viewer toggles their subscription",
);

assert.match(
  sourcesPageSource,
  /AgentSessionDetailsDialog/,
  "managed agent-session sources should expose a read-only details dialog",
);

assert.match(
  sourcesPageSource,
  /resourceClient\.get\(`\/sources\/\$\{source\.id\}\/projects/,
  "agent-session details should show project buckets from the generic source projects endpoint",
);

assert.match(
  sourcesPageSource,
  /agent_session/,
  "agent-session sources should have an explicit source-list label instead of falling through to raw gene metadata",
);

assert.match(
  sourcesPageSource,
  /Memories created/,
  "agent-session details should headline value-first metrics, not raw operational counts",
);

assert.match(
  sourcesPageSource,
  /Kept summaries/,
  "agent-session details should rename package_created to a PM-friendly 'Kept summaries' label",
);

assert.doesNotMatch(
  sourcesPageSource,
  /Drop rate/,
  "agent-session details should not surface a headline 'Drop rate' metric — most no_output windows are intentional filtering",
);

assert.match(
  sourcesPageSource,
  /Skipped low-signal/,
  "agent-session details should rename no_output to 'Skipped low-signal' inside the operational details disclosure",
);

assert.match(
  sourcesPageSource,
  /Operational details/,
  "agent-session details should expose raw counts via a progressive-disclosure section, not as headline tiles",
);

assert.match(
  sourcesPageSource,
  /latest_failure/,
  "agent-session details should consume the latest_failure timestamp so retry backlog metadata stays visible",
);

assert.doesNotMatch(
  sourcesPageSource,
  /Latest retry reason/,
  "agent-session details should not surface raw provider exception text in the product dialog",
);

assert.doesNotMatch(
  sourcesPageSource,
  /title=\{latestFailure\.reason\}/,
  "agent-session details should not leak raw retry exceptions through hover text",
);
