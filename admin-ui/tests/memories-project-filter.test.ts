import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const memoriesSource = readFileSync("src/views/memories/MemoriesPage.tsx", "utf8");
const memoryFiltersSource = readFileSync(
  "src/views/memories/MemoryFiltersPopover.tsx",
  "utf8",
);

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

assert.match(
  memoriesSource,
  /include_private:\s*true,/,
  "the Memories page should include the current user's private rows, including agent-session memories",
);

assert.match(
  memoriesSource,
  /include_private:\s*"true",/,
  "the Memories list route should request current-user private rows for source filters",
);

assert.match(
  memoriesSource,
  /<SearchInput[\s\S]*?size="sm"[\s\S]*?className="min-w-0 flex-1"/,
  "the Memories search should use the same compact control treatment as Sources",
);

assert.doesNotMatch(
  memoriesSource,
  />Memory List</,
  "the Memories data surface should not repeat the page title in a tall list header",
);

assert.match(
  memoriesSource,
  /className="flex flex-col gap-2 border-b p-3 sm:flex-row sm:items-center"/,
  "the Memories data surface should use one compact command row",
);

assert.match(
  memoriesSource,
  /<MemoryFiltersPopover\b/,
  "the Memories toolbar should collapse advanced filters into one popover",
);

assert.doesNotMatch(
  memoriesSource,
  /<FilterSelect\b/,
  "the Memories toolbar should not render four always-visible filter dropdowns",
);

for (const label of ["Type", "Status", "Source", "Project"]) {
  assert.match(
    memoryFiltersSource,
    new RegExp(`label="${label}"`),
    `the filter popover should keep the ${label.toLowerCase()} filter`,
  );
}

assert.match(
  memoryFiltersSource,
  /activeFilterCount/,
  "the collapsed filter control should communicate how many filters are active",
);

assert.match(
  memoryFiltersSource,
  /Clear all/,
  "the collapsed filter control should provide one reset action",
);

assert.match(
  memoryFiltersSource,
  /Only this project/,
  "the project filter popover should preserve the narrow project scope control",
);

assert.match(
  memoryFiltersSource,
  /w-\[min\(20rem,calc\(100vw-2rem\)\)\]/,
  "the filter popover should fit within narrow viewports",
);

assert.match(
  memoryFiltersSource,
  /grid-cols-1[^"]*min-\[420px\]:grid-cols-2/,
  "the filter fields should stack before the popover has room for two columns",
);

console.log("memories-project-filter.test.ts: all assertions passed");
