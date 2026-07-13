import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import {
  normalizeRepoPickerPath,
  pathIsCoveredBySelection,
  repoEffectiveFileCount,
  repoPickerItemsFromFilePaths,
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
assert.match(pickerSource, /Enter a valid HTTPS Repository URL before browsing/);
assert.doesNotMatch(pickerSource, /Choose local repository clone|github_repo_pick_root/);
