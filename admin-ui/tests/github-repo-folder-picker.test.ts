import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  normalizeRepoPickerPath,
  pathIsCoveredBySelection,
  repoEffectiveFiles,
  repoEffectiveFileCount,
  repoPickerItemsFromFilePaths,
  repoPickerSelectionState,
  repoPickerTreeRows,
  repoScopeSummary,
  type RepoPickerItem,
  updateRepoPathSelection,
} from "../src/views/sources/githubRepoFolderPickerUtils.js";

assert.equal(normalizeRepoPickerPath(" /Payroll Processing// "), "Payroll Processing");
assert.equal(normalizeRepoPickerPath("\\Payroll\\Processing\\README.md"), "Payroll/Processing/README.md");

const items = repoPickerItemsFromFilePaths([
  "Payroll Processing/README.md",
  "Payroll Processing/V2/Migration.md",
  "Flexible Payroll/Overview.md",
]);

assert.deepEqual(
  items.filter((item: RepoPickerItem) => item.type === "tree").map((item: RepoPickerItem) => item.path),
  ["Flexible Payroll", "Payroll Processing", "Payroll Processing/V2"],
);
assert.deepEqual(
  items.filter((item: RepoPickerItem) => item.type === "blob").map((item: RepoPickerItem) => item.path),
  [
    "Flexible Payroll/Overview.md",
    "Payroll Processing/README.md",
    "Payroll Processing/V2/Migration.md",
  ],
);

assert.deepEqual(updateRepoPathSelection([], "Payroll Processing/", true), ["Payroll Processing"]);
assert.deepEqual(
  updateRepoPathSelection(["Payroll Processing"], "Flexible Payroll", true),
  ["Flexible Payroll", "Payroll Processing"],
);
assert.deepEqual(updateRepoPathSelection(["Payroll Processing"], "Payroll Processing", false), []);
assert.deepEqual(
  updateRepoPathSelection(["Payroll Processing/V2"], "Payroll Processing", true),
  ["Payroll Processing"],
  "selecting a parent should collapse redundant descendants",
);
assert.equal(pathIsCoveredBySelection("docs/archive/old.md", ["docs/archive"]), true);
assert.equal(pathIsCoveredBySelection("docs/current.md", ["docs/archive"]), false);
assert.equal(
  repoEffectiveFileCount(items, [], ["Payroll Processing/V2"]),
  2,
  "whole-repository preview should subtract excluded files",
);
assert.equal(
  repoEffectiveFileCount(items, ["Payroll Processing"], ["Payroll Processing/V2"]),
  1,
  "selected-only preview should apply exclusions after includes",
);

const nestedItems = repoPickerItemsFromFilePaths([
  ".github/workflows/ci.yml",
  "docs/.internal/puml.md",
  "docs/archived-topics/archive.md",
  "docs/cnp-core/README.md",
  "README.md",
]);

const expandedDocsRows = repoPickerTreeRows(nestedItems, new Set(["docs"]), "");
assert.deepEqual(
  expandedDocsRows.map((row) => [row.item.path, row.depth, row.fileCount]),
  [
    [".github", 0, 1],
    ["docs", 0, 3],
    ["docs/.internal", 1, 1],
    ["docs/archived-topics", 1, 1],
    ["docs/cnp-core", 1, 1],
    ["README.md", 0, 1],
  ],
  "expanded folders should reveal indented children while collapsed siblings stay compact",
);

assert.deepEqual(
  repoPickerTreeRows(nestedItems, new Set(), "archive.md").map((row) => [row.item.path, row.depth]),
  [
    ["docs", 0],
    ["docs/archived-topics", 1],
    ["docs/archived-topics/archive.md", 2],
  ],
  "search should keep the matching path's complete ancestor chain visible",
);

assert.equal(repoPickerSelectionState("docs", ["docs/cnp-core"]), "partial");
assert.equal(repoPickerSelectionState("docs/cnp-core", ["docs/cnp-core"]), "selected");
assert.equal(repoPickerSelectionState("docs/cnp-core/README.md", ["docs/cnp-core"]), "inherited");
assert.equal(repoPickerSelectionState("docs/messaging", ["docs/cnp-core"]), "unselected");

assert.deepEqual(
  repoEffectiveFiles(nestedItems, ["docs"], ["docs/archived-topics"]).map((item) => item.path),
  ["docs/.internal/puml.md", "docs/cnp-core/README.md"],
  "preview files should use the same include-then-exclude scope contract as sync",
);
assert.deepEqual(repoScopeSummary(nestedItems, [], ["docs/archived-topics"]), {
  readyCount: 4,
  totalCount: 5,
  filteredCount: 1,
  readyLabel: "4 files ready to sync",
  detailLabel: "1 file filtered out by 1 confirmed exclusion",
});
assert.deepEqual(repoScopeSummary(nestedItems, ["docs"], ["docs/archived-topics"]), {
  readyCount: 2,
  totalCount: 5,
  filteredCount: 3,
  readyLabel: "2 files ready to sync",
  detailLabel: "3 files outside the effective scope · 1 confirmed exclusion",
});

const pickerSource = readFileSync("src/views/sources/GitHubRepoFolderPicker.tsx", "utf8");
assert.match(
  pickerSource,
  /createLocalAgentJob/,
  "Internal network GitHub folder browsing should enqueue through the target-aware local-agent helper",
);
assert.match(
  pickerSource,
  /getLocalAgentJob\(jobId\)/,
  "Internal network GitHub folder browsing should poll through the target-aware local-agent helper",
);
assert.match(
  pickerSource,
  /github_repo_preview_tree/,
  "Internal network GitHub folder browsing should request a preview-tree job",
);
assert.match(
  pickerSource,
  /pollLocalAgentJob/,
  "Internal network GitHub folder browsing should poll until the daemon completes the preview job",
);
assert.match(
  pickerSource,
  /const LOCAL_AGENT_POLL_ATTEMPTS = 180;/,
  "Folder browsing should tolerate daemon startup and one missed polling tick",
);
assert.match(pickerSource, /Sync all supported files in this repository/);
assert.match(pickerSource, /Choose exclusions/);
assert.match(pickerSource, /Sync only selected folders instead/);
assert.match(pickerSource, /Preview files/);
assert.match(pickerSource, /aria-expanded/);
assert.match(pickerSource, /tone="exclude"/);
assert.match(pickerSource, /Enter a valid HTTPS Repository URL before browsing/);
assert.doesNotMatch(pickerSource, /Choose local repository clone|github_repo_pick_root/);
