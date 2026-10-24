import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  canConfigureSourceType,
  hasSetupForm,
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
assert.equal(canConfigureSourceType("confluence"), true);
assert.equal(canConfigureSourceType("github_repo"), true);
assert.deepEqual(userConfigurableGenes(genes).map((gene) => gene.name), ["confluence", "jira", "teams"]);
assert.equal(hasSetupForm("teams"), true);
assert.equal(hasSetupForm("confluence"), true);
assert.equal(hasSetupForm("notion"), false);
// hides a source type without a setup form
assert.deepEqual(
  userConfigurableGenes([
    ...genes,
    { name: "notion", display_name: "Notion", execution_kinds: ["server"] as const },
  ]).map((gene) => gene.name),
  ["confluence", "jira", "teams"],
);
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

assert.doesNotMatch(
  sourcesPageSource,
  /agent-sessions\/completeness|Coding agent integrations/,
  "the page reads no agent-session completeness and offers no session-capture setup",
);

assert.match(
  sourcesPageSource,
  /agent_session/,
  "agent-session sources should have an explicit source-list label instead of falling through to raw gene metadata",
);
