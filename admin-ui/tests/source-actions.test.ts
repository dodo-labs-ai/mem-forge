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
  ["force-resync", "delete"],
  "source cards should move expensive and destructive actions into the overflow menu",
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
assert.match(
  sourcesPageSource,
  /className="[^"]*cursor-pointer[^"]*disabled:cursor-not-allowed[^"]*"/,
  "enabled overflow menu actions should use a pointer cursor while disabled actions keep not-allowed",
);
