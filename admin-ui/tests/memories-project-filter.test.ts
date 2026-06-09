import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const memoriesSource = readFileSync("src/views/memories/MemoriesPage.tsx", "utf8");

assert.match(
  memoriesSource,
  /const effectiveProjectKey = pageProjectOverride;/,
  "the Memories page project filter should not fall back to the topbar active project",
);

assert.match(
  memoriesSource,
  /enabled: true,/,
  "the Memories page should load the all-project result set without requiring a selected project",
);

assert.doesNotMatch(
  memoriesSource,
  /label:\s*`\$\{project\.name\} \(unmapped\)`/,
  "the Memories project filter should not show UNSORTED as a normal project option",
);

assert.match(
  memoriesSource,
  /project\.key !== UNSORTED_PROJECT_KEY/,
  "the Memories project filter should hide the UNSORTED backlog from the normal project picker",
);

assert.doesNotMatch(
  memoriesSource,
  /Pick a project to start/,
  "the Memories page should not block the list behind the global project chip",
);

console.log("memories-project-filter.test.ts: all assertions passed");
