import assert from "node:assert/strict";

import {
  normalizeRepoPickerPath,
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
