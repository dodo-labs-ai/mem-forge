import { expect, test } from "vitest";
import { makeSource } from "@/test/sourceFixtures";
import { projectSourceRows } from "./projectSources";

test("orders sources by the memories they sent and names how each is bound", () => {
  const wiki = makeSource({ id: "src-wiki", name: "Payroll wiki" });
  const jira = makeSource({ id: "src-jira", name: "Payroll Jira", type: "jira" });
  const repo = makeSource({ id: "src-repo", name: "Payroll repo", type: "github_repo" });
  const rows = projectSourceRows([
    { source: repo, memory_count: 0, binding: { mode: "by_field", field: "repo" } },
    { source: wiki, memory_count: 96, binding: { mode: "fixed", project_key: "PAY" } },
    { source: jira, memory_count: 611, binding: { mode: "fixed", project_key: "PAY" } },
  ]);
  expect(rows.map((row) => [row.source.name, row.binding, row.memoryCount])).toEqual([
    ["Payroll Jira", "direct", 611],
    ["Payroll wiki", "direct", 96],
    ["Payroll repo", "by_field", 0],
  ]);
});
