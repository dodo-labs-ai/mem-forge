import { describe, expect, test } from "vitest";
import { makeSource } from "@/test/sourceFixtures";
import { buildSourceRowGroups } from "./sourceRows";

const NOW = new Date("2026-09-29T12:00:00Z");
const PROJECTS = [{ id: "p1", key: "PAY", name: "Payroll" }];

function build(filter: "all" | "needs_you" | "pinned", sources = [
  makeSource(),
  makeSource({ id: "src-jira", type: "jira", name: "Payroll Jira", project_binding: null, pinned_for_me: true }),
]) {
  return buildSourceRowGroups({
    sources,
    projects: PROJECTS,
    resolvedBySource: {},
    jobs: [],
    daemon: "ready",
    query: "",
    filter,
    sortMode: "name",
    typeLabels: {},
    now: NOW,
  });
}

describe("buildSourceRowGroups", () => {
  test("groups by project and puts sources without one last", () => {
    const groups = build("all");
    expect(groups.map((group) => group.title)).toEqual(["Payroll", "No project"]);
    expect(groups[1]?.description).toContain("not assigned");
  });

  test("keys rows by group so a source can appear in several groups", () => {
    expect(build("all").flatMap((group) => group.rows.map((row) => row.key))).toEqual(["PAY:src-wiki", "__unmapped__:src-jira"]);
  });

  test("the Needs you filter keeps only sources with a problem", () => {
    const rows = build("needs_you").flatMap((group) => group.rows);
    expect(rows.map((row) => row.source.id)).toEqual(["src-jira"]);
    expect(rows[0]?.attention?.reason).toBe("unmapped");
  });

  test("the Pinned filter keeps pinned sources", () => {
    expect(build("pinned").flatMap((group) => group.rows.map((row) => row.source.id))).toEqual(["src-jira"]);
  });
});
