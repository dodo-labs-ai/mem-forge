import { describe, expect, test } from "vitest";
import {
  normalizeRepoPath,
  pathIsCoveredBySelection,
  repoEffectiveFiles,
  repoItemsFromFilePaths,
  repoScopeSummary,
  repoSelectionState,
  repoTreeRows,
  suggestedExclusions,
  topLevelFolders,
  updateRepoPathSelection,
} from "./githubRepoPaths";

const PAYROLL = repoItemsFromFilePaths([
  "Payroll Processing/README.md",
  "Payroll Processing/V2/Migration.md",
  "Flexible Payroll/Overview.md",
]);

const NESTED = repoItemsFromFilePaths([
  ".github/workflows/ci.yml",
  "docs/.internal/puml.md",
  "docs/archived-topics/archive.md",
  "docs/cnp-core/README.md",
  "README.md",
]);

function paths(items: { path: string }[]): string[] {
  return items.map((item) => item.path);
}

describe("githubRepoPaths", () => {
  test("a path is stored with forward slashes and no outer slashes", () => {
    expect(normalizeRepoPath(" /Payroll Processing// ")).toBe("Payroll Processing");
    expect(normalizeRepoPath("\\Payroll\\Processing\\README.md")).toBe("Payroll/Processing/README.md");
  });

  test("a preview's file paths become their folders, then the files, each sorted", () => {
    expect(paths(PAYROLL.filter((item) => item.type === "tree"))).toEqual(["Flexible Payroll", "Payroll Processing", "Payroll Processing/V2"]);
    expect(paths(PAYROLL.filter((item) => item.type === "blob"))).toEqual([
      "Flexible Payroll/Overview.md",
      "Payroll Processing/README.md",
      "Payroll Processing/V2/Migration.md",
    ]);
  });

  test("choosing a folder keeps the selection minimal", () => {
    expect(updateRepoPathSelection([], "Payroll Processing/", true)).toEqual(["Payroll Processing"]);
    expect(updateRepoPathSelection(["Payroll Processing"], "Flexible Payroll", true)).toEqual(["Flexible Payroll", "Payroll Processing"]);
    expect(updateRepoPathSelection(["Payroll Processing"], "Payroll Processing", false)).toEqual([]);
    expect(updateRepoPathSelection(["Payroll Processing/V2"], "Payroll Processing", true)).toEqual(["Payroll Processing"]);
    expect(updateRepoPathSelection(["docs"], "docs/guide.md", true)).toEqual(["docs"]);
  });

  test("a chosen folder covers what is inside it", () => {
    expect(pathIsCoveredBySelection("docs/archive/old.md", ["docs/archive"])).toBe(true);
    expect(pathIsCoveredBySelection("docs/current.md", ["docs/archive"])).toBe(false);
    expect(repoSelectionState("docs", ["docs/cnp-core"])).toBe("partial");
    expect(repoSelectionState("docs/cnp-core", ["docs/cnp-core"])).toBe("selected");
    expect(repoSelectionState("docs/cnp-core/README.md", ["docs/cnp-core"])).toBe("inherited");
    expect(repoSelectionState("docs/messaging", ["docs/cnp-core"])).toBe("unselected");
  });

  test("the files read are under an included path and under no excluded one, as a sync reads them", () => {
    expect(repoEffectiveFiles(PAYROLL, [], ["Payroll Processing/V2"])).toHaveLength(2);
    expect(repoEffectiveFiles(PAYROLL, ["Payroll Processing"], ["Payroll Processing/V2"])).toHaveLength(1);
    expect(paths(repoEffectiveFiles(NESTED, ["docs"], ["docs/archived-topics"]))).toEqual(["docs/.internal/puml.md", "docs/cnp-core/README.md"]);
  });

  test("the summary counts the files ready to sync and says what the paths leave out", () => {
    expect(repoScopeSummary(NESTED, [], [])).toEqual({ readyLabel: "5 files ready to sync", detailLabel: "All 5 files are included." });
    expect(repoScopeSummary(NESTED, [], ["docs/archived-topics"])).toEqual({
      readyLabel: "4 files ready to sync",
      detailLabel: "1 file left out by 1 exclusion",
    });
    expect(repoScopeSummary(NESTED, ["docs"], ["docs/archived-topics"])).toEqual({
      readyLabel: "2 files ready to sync",
      detailLabel: "3 files outside the selected paths · 1 exclusion",
    });
  });

  test("an expanded folder shows its children indented, and collapsed ones stay closed", () => {
    expect(repoTreeRows(NESTED, new Set(["docs"]), "").map((row) => [row.item.path, row.depth, row.fileCount])).toEqual([
      [".github", 0, 1],
      ["docs", 0, 3],
      ["docs/.internal", 1, 1],
      ["docs/archived-topics", 1, 1],
      ["docs/cnp-core", 1, 1],
      ["README.md", 0, 1],
    ]);
    expect([...topLevelFolders(NESTED)]).toEqual([".github", "docs"]);
  });

  test("a search keeps the matching path's folders above it", () => {
    expect(repoTreeRows(NESTED, new Set(), "archive.md").map((row) => [row.item.path, row.depth])).toEqual([
      ["docs", 0],
      ["docs/archived-topics", 1],
      ["docs/archived-topics/archive.md", 2],
    ]);
  });

  test("outdated-looking folders are suggested as exclusions until they are excluded", () => {
    const items = repoItemsFromFilePaths(["docs/archive/old.md", "docs/deprecated/api.md", "docs/archived-topics/a.md", "docs/guide.md"]);
    expect(suggestedExclusions(items, [])).toEqual(["docs/archive", "docs/deprecated"]);
    expect(suggestedExclusions(items, ["docs/archive"])).toEqual(["docs/deprecated"]);
    expect(suggestedExclusions(items, ["docs"])).toEqual([]);
  });
});
