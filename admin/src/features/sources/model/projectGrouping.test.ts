import { expect, test } from "vitest";
import { makeSource } from "@/test/sourceFixtures";
import { sourcesByProjectKey } from "./projectGrouping";

test("keys each bound source by the projects it sends memories to", () => {
  const wiki = makeSource({ id: "src-wiki", project_binding: { mode: "fixed", project_key: "PAY" }, memory_count: 96 });
  const codex = makeSource({
    id: "src-codex",
    type: "agent_session",
    project_binding: { mode: "by_field", field: "repo", default: "UNSORTED" },
  });
  const pending = makeSource({
    id: "src-pending",
    project_binding: { mode: "by_field", field: "space", default: "UNSORTED" },
    memory_count: 4,
  });
  const unbound = makeSource({ id: "src-unbound", project_binding: null });

  const byKey = sourcesByProjectKey([wiki, codex, pending, unbound], {
    "src-codex": [
      { project_key: "PAY", memory_count: 7 },
      { project_key: "RISK", memory_count: 3 },
    ],
  });

  const summary = (key: string) =>
    byKey.get(key)?.map((entry) => [entry.source.id, entry.binding.mode, entry.memory_count]);
  expect(summary("PAY")).toEqual([
    ["src-wiki", "fixed", 96],
    ["src-codex", "by_field", 7],
  ]);
  expect(summary("RISK")).toEqual([["src-codex", "by_field", 3]]);
  expect(summary("UNSORTED")).toEqual([["src-pending", "by_field", 4]]);
  expect([...byKey.keys()]).not.toContain(null);
});
