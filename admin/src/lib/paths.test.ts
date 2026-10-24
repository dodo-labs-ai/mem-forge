import { expect, test } from "vitest";
import {
  entityPath,
  entitySearchPath,
  memoryPath,
  newSourcePath,
  projectMemoriesPath,
  projectPath,
  reviewPath,
  sourceEvaluationPath,
  sourceSettingsPath,
} from "./paths";

test("links one memory, review or project by its encoded id", () => {
  expect(memoryPath("mem-1")).toBe("/memories/mem-1");
  expect(reviewPath("rev 9")).toBe("/review/rev%209");
  expect(projectPath("PAYROLL_V2")).toBe("/projects/PAYROLL_V2");
  expect(projectPath("R&D")).toBe("/projects/R%26D");
});

test("opens the Memories list filtered to a project", () => {
  expect(projectMemoriesPath("PAYROLL_V2")).toBe("/memories?project=PAYROLL_V2");
  expect(projectMemoriesPath("R&D")).toBe("/memories?project=R%26D");
});

test("opens the Evaluation page scoped to a source", () => {
  expect(sourceEvaluationPath("src-wiki")).toBe("/evaluation?source_id=src-wiki");
});

test("opens the type picker, a new Source of one type, and a Source's settings at a step", () => {
  expect(newSourcePath()).toBe("/sources/new");
  expect(newSourcePath("github_pages")).toBe("/sources/new/github_pages");
  expect(sourceSettingsPath("src-wiki")).toBe("/sources/src-wiki/settings");
  expect(sourceSettingsPath("src wiki", "project")).toBe("/sources/src%20wiki/settings?step=project");
});

test("opens one entity by its id, and the Entities list searched for a name", () => {
  expect(entityPath(42)).toBe("/entities/42");
  expect(entitySearchPath("Cut-off")).toBe("/entities?search=Cut-off");
  expect(entitySearchPath("R&D team")).toBe("/entities?search=R%26D+team");
});
