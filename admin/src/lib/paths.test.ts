import { expect, test } from "vitest";
import { memoryPath, projectMemoriesPath, projectPath, reviewPath, sourceEvaluationPath } from "./paths";

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
