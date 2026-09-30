import { expect, test } from "vitest";
import { projectBadge, projectFilterOptions } from "./projects";

const PROJECTS = [
  { key: "PAY", name: "payroll-v2" },
  { key: "SHARED", name: "Shared" },
  { key: "UNSORTED", name: "Unsorted" },
];

test("project badges name the team-wide and catch-all projects", () => {
  expect(projectBadge("PAY", PROJECTS)).toEqual({ kind: "project", label: "payroll-v2" });
  expect(projectBadge("GONE", PROJECTS)).toEqual({ kind: "project", label: "GONE" });
  expect(projectBadge("SHARED", PROJECTS)).toEqual({ kind: "shared", label: "Shared" });
  expect(projectBadge("UNSORTED", PROJECTS)).toEqual({ kind: "unsorted", label: "Unsorted" });
  expect(projectBadge(null, PROJECTS)).toBeNull();
});

test("the project filter marks the team-wide project and leaves out Unsorted", () => {
  expect(projectFilterOptions(PROJECTS)).toEqual([
    { value: "PAY", label: "payroll-v2" },
    { value: "SHARED", label: "Shared (team-wide)" },
  ]);
});
