import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { organizeSourceGroups } from "../src/views/sources/sourceListOrganization.js";

const sourcesPageSource = readFileSync("src/views/sources/SourcesPage.tsx", "utf8");

assert.match(
  sourcesPageSource,
  /<SearchInput[\s\S]*?ariaLabel="Search sources"[\s\S]*?size="sm"[\s\S]*?className="min-w-0 flex-1"/,
  "Source search should use the same compact height as the pinned filter button",
);
assert.doesNotMatch(
  sourcesPageSource,
  />Source List</,
  "the Sources data surface should not repeat the page title in a tall list header",
);
assert.doesNotMatch(
  sourcesPageSource,
  /configured ingestion sources\./,
  "the Sources toolbar should not repeat the configured count already shown in the stats",
);
assert.match(
  sourcesPageSource,
  /className="flex flex-col gap-2 border-b p-3 sm:flex-row sm:items-center"/,
  "the Sources data surface should use one compact command row",
);
assert.match(
  sourcesPageSource,
  /<SelectTrigger aria-label="Sort sources" className="h-7\b/,
  "Source sort should use the same compact height as the pinned filter button",
);

const groups = [
  {
    project: { name: "Payroll", key: "PAY" },
    sources: [
      { source: { id: "old", name: "Alpha Wiki", type: "confluence", created_at: "2026-01-01T00:00:00Z", last_sync: "2026-07-10T00:00:00Z", pinned_for_me: false, doc_count: 3 }, memory_count: 5 },
      { source: { id: "new", name: "Beta Chat", type: "teams", created_at: "2026-07-01T00:00:00Z", last_sync: null, pinned_for_me: false, doc_count: 7 }, memory_count: 11 },
      { source: { id: "pin", name: "Gamma Repo", type: "github_repo", created_at: "2025-01-01T00:00:00Z", last_sync: "2026-07-11T00:00:00Z", pinned_for_me: true, doc_count: 2 }, memory_count: 4 },
    ],
    docCount: 12,
    memoryCount: 20,
  },
];

const newest = organizeSourceGroups(groups, { query: "", pinnedOnly: false, sortMode: "newest" });
assert.deepEqual(newest[0].sources.map((entry) => entry.source.id), ["pin", "new", "old"]);
assert.equal(newest[0].docCount, 12);

const synced = organizeSourceGroups(groups, { query: "", pinnedOnly: false, sortMode: "recently_synced" });
assert.deepEqual(synced[0].sources.map((entry) => entry.source.id), ["pin", "old", "new"]);

const searched = organizeSourceGroups(groups, {
  query: "microsoft teams",
  pinnedOnly: false,
  sortMode: "name",
  typeLabels: { teams: "Microsoft Teams" },
});
assert.deepEqual(searched[0].sources.map((entry) => entry.source.id), ["new"]);
assert.equal(searched[0].docCount, 7);
assert.equal(searched[0].memoryCount, 11);

const projectSearch = organizeSourceGroups(groups, { query: "payroll", pinnedOnly: false, sortMode: "name" });
assert.equal(projectSearch[0].sources.length, 3);

const pinnedOnly = organizeSourceGroups(groups, { query: "", pinnedOnly: true, sortMode: "newest" });
assert.deepEqual(pinnedOnly[0].sources.map((entry) => entry.source.id), ["pin"]);

const stable = organizeSourceGroups([
  {
    project: null,
    sources: [
      { source: { id: "b", name: "Same", type: "jira", created_at: "2026-01-01T00:00:00Z", last_sync: null, pinned_for_me: false, doc_count: 1 }, memory_count: 1 },
      { source: { id: "a", name: "same", type: "jira", created_at: "2026-01-01T00:00:00Z", last_sync: null, pinned_for_me: false, doc_count: 1 }, memory_count: 1 },
    ],
    docCount: 2,
    memoryCount: 2,
  },
], { query: "", pinnedOnly: false, sortMode: "newest" });
assert.deepEqual(stable[0].sources.map((entry) => entry.source.id), ["a", "b"]);
