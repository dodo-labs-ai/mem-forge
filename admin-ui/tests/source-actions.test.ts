import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  getSourceActionEndpoint,
  getSourceMenuPlacement,
  getSourceMenuStyle,
  sourceActionLayout,
} from "../src/views/sources/sourceActions.js";

assert.deepEqual(
  sourceActionLayout.primary.map((action) => action.id),
  ["configure", "sync"],
  "source cards should keep only Configure and Sync as visible primary actions",
);

assert.deepEqual(
  sourceActionLayout.menu.map((action) => action.id),
  ["toggle-status", "force-resync", "delete"],
  "source cards should move source lifecycle, expensive, and destructive actions into the overflow menu",
);

const toggleStatus = sourceActionLayout.menu.find((action) => action.id === "toggle-status");
assert.equal(toggleStatus?.tone, "neutral");
assert.equal(
  toggleStatus?.description,
  "Pause or resume source discovery without deleting configuration or extracted memories.",
);

const forceResync = sourceActionLayout.menu.find((action) => action.id === "force-resync");
assert.equal(forceResync?.label, "Refresh source");
assert.equal(forceResync?.tone, "neutral");
assert.equal("disabled" in (forceResync ?? {}), false);
assert.equal(
  forceResync?.description,
  "Look for new, changed, or removed documents. Existing memories are not rebuilt unless source content changed.",
);
assert.equal(getSourceActionEndpoint("src-1", "force-resync"), "/api/sources/src-1/force-resync");

const deleteSource = sourceActionLayout.menu.find((action) => action.id === "delete");
assert.equal(deleteSource?.tone, "destructive");
assert.equal(deleteSource?.requiresConfirmation, true);
assert.equal(getSourceActionEndpoint("src-1", "delete"), "/api/sources/src-1");

assert.deepEqual(
  getSourceMenuPlacement({
    triggerTop: 650,
    triggerBottom: 686,
    viewportHeight: 720,
    menuHeight: 224,
  }),
  { direction: "up", top: 418 },
  "menus near the bottom of the viewport should open upward instead of being clipped",
);

assert.deepEqual(
  getSourceMenuPlacement({
    triggerTop: 120,
    triggerBottom: 156,
    viewportHeight: 720,
    menuHeight: 224,
  }),
  { direction: "down", top: 164 },
  "menus with enough lower viewport space should open downward with an 8px gap",
);

assert.deepEqual(
  getSourceMenuStyle({
    triggerRight: 1_224,
    triggerTop: 560,
    triggerBottom: 596,
    viewportWidth: 1_280,
    viewportHeight: 720,
    menuHeight: 160,
  }),
  { position: "fixed", top: 392, left: 936, width: 288 },
  "source action menus should align to the trigger and stay within the viewport",
);

assert.deepEqual(
  getSourceMenuStyle({
    triggerRight: 240,
    triggerTop: 120,
    triggerBottom: 156,
    viewportWidth: 320,
    viewportHeight: 720,
    menuHeight: 160,
  }),
  { position: "fixed", top: 164, left: 8, width: 288 },
  "source action menus should clamp horizontally on narrow viewports",
);

const sourcesPageSource = readFileSync("src/views/sources/SourcesPage.tsx", "utf8");
const sourceRowSource = readFileSync("src/views/sources/SourceRow.tsx", "utf8");
const syncStatusBarSource = readFileSync("src/components/admin/SyncStatusBar.tsx", "utf8");

assert.match(
  sourcesPageSource,
  /setSourceStatus\s*=\s*useMutation/,
  "SourcesPage should update source lifecycle through the generic source update endpoint",
);
assert.match(
  sourcesPageSource,
  /client\.put\(`\/api\/sources\/\$\{sourceId\}`,\s*\{\s*status\s*\}\)/,
  "Pause and resume should use PUT /api/sources/{id} with a status body",
);
assert.match(
  sourcesPageSource,
  /onToggleStatus=\{\(\)\s*=>\s*\{/,
  "SourceActionsMenu should receive a pause/resume action per source row",
);
assert.match(
  sourceRowSource,
  /const isPaused = source\.status === "paused";/,
  "SourceRow should derive paused state from the source status",
);
assert.match(
  sourceRowSource,
  /disabled=\{isSyncing \|\| isDeleting \|\| isPaused\}/,
  "Paused sources should not expose an enabled primary Sync button",
);
assert.match(
  sourceRowSource,
  /onRetry=\{isPaused \? undefined : onSync\}/,
  "Paused sources should not expose retry sync from the status bar",
);
assert.match(
  sourceRowSource,
  /source\.sync_schedule\?\.enabled/,
  "SourceRow should show automatic sync metadata when a source schedule is enabled",
);
assert.match(
  sourceRowSource,
  /formatRelativeFuture\(source\.sync_schedule\.next_run_at\)/,
  "SourceRow should format the next scheduled sync as a future time instead of using the last-sync formatter",
);
assert.doesNotMatch(
  sourceRowSource,
  /New memories/,
  "last-sync details should not label extraction candidates as new durable memories",
);
assert.doesNotMatch(
  syncStatusBarSource,
  /new memories|stored memories/i,
  "sync status details should avoid memory extraction counters that can differ from durable memory counts",
);

assert.match(
  sourcesPageSource,
  /className="[^"]*cursor-pointer[^"]*disabled:cursor-not-allowed[^"]*"/,
  "enabled overflow menu actions should use a pointer cursor while disabled actions keep not-allowed",
);

const sourceConfigDialogSource = readFileSync("src/views/sources/SourceConfigDialog.tsx", "utf8");
assert.match(
  sourceConfigDialogSource,
  /const DISCOVERY_PREVIEW_LIMIT = 5;/,
  "source discovery preview should request a small bounded result set",
);
assert.match(
  sourceConfigDialogSource,
  /function discoveryPreviewGroupKey/,
  "source discovery preview placement should be centralized instead of hard-coded inline",
);
assert.match(
  sourceConfigDialogSource,
  /group\.key === "scope"/,
  "source discovery preview should appear after the scope fields when a source has a What to Sync group",
);
assert.match(
  sourceConfigDialogSource,
  /limit: DISCOVERY_PREVIEW_LIMIT/,
  "source discovery preview requests should send the bounded limit to the API",
);
assert.match(
  sourceConfigDialogSource,
  /memforge adapter auth jira refresh --base-url/,
  "Jira browser-session guidance should use the refresh subcommand that uploads the local browser session",
);
assert.match(
  sourceConfigDialogSource,
  /jiraSessionQuery\.refetch\(\)/,
  "Jira browser-session guidance should allow users to re-check after running the CLI refresh",
);
assert.match(
  sourceConfigDialogSource,
  /const payloadWithSchedule = \{/,
  "Source saves should bundle automatic sync settings into the source payload",
);
assert.match(
  sourceConfigDialogSource,
  /sync_schedule:\s*\{\s*enabled: scheduleEnabled,\s*interval_minutes: intervalMinutes,\s*\}/,
  "Source saves should send the schedule shape expected by the source API",
);
assert.doesNotMatch(
  sourceConfigDialogSource,
  /\/api\/sources\/[^`]+\/schedule/,
  "SourceConfigDialog should not split config and schedule persistence into two requests",
);
assert.match(
  sourceConfigDialogSource,
  /<span className="block text-sm font-medium">Sync on a schedule<\/span>/,
  "Source configuration should expose a clear automatic sync control",
);

const projectBindingSource = readFileSync("src/views/sources/ProjectBindingFields.tsx", "utf8");
assert.match(
  projectBindingSource,
  /focus-visible:ring-1 focus-visible:ring-ring\/40/,
  "project picker focus styling should be visible without creating a heavy shadow around the dropdown",
);
