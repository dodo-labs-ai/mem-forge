import { expect, test } from "vitest";
import { makeSource } from "@/test/sourceFixtures";
import { projectSourceRows } from "./projectSources";

test("orders sources by the memories they sent and names how each is bound", () => {
  const wiki = makeSource({ id: "src-wiki", name: "Payroll wiki" });
  const jira = makeSource({ id: "src-jira", name: "Payroll Jira", type: "jira" });
  const codex = makeSource({ id: "src-codex", name: "Codex sessions", type: "agent_session" });
  const rows = projectSourceRows([
    { source: codex, memory_count: 0, binding: { mode: "by_field", field: "repo" } },
    { source: wiki, memory_count: 96, binding: { mode: "fixed", project_key: "PAY" } },
    { source: jira, memory_count: 611, binding: { mode: "fixed", project_key: "PAY" } },
  ]);
  expect(rows.map((row) => [row.source.name, row.binding, row.memoryCount])).toEqual([
    ["Payroll Jira", "direct", 611],
    ["Payroll wiki", "direct", 96],
    ["Codex sessions", "by_field", 0],
  ]);
});
