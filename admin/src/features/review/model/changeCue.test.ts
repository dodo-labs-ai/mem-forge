import { expect, test } from "vitest";
import { buildChangeCue } from "./changeCue";

test("highlights only the words that differ", () => {
  const cue = buildChangeCue(
    "The MCP Helper Tool exposes five endpoints for agents.",
    "The MCP Helper Tool exposes six endpoints for agents.",
  );
  expect(cue.current).toEqual({ before: "… MCP Helper Tool exposes", changed: "five", after: "endpoints for agents." });
  expect(cue.proposed?.changed).toBe("six");
});

test("shows a removal as a plain excerpt with nothing proposed", () => {
  const cue = buildChangeCue("This memory still has content.", null);
  expect(cue.current).toEqual({ before: "This memory still has content.", changed: "", after: "" });
  expect(cue.proposed).toBeNull();
});

test("cuts long changes and marks the cut", () => {
  const current = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen";
  const cue = buildChangeCue(current, "zero");
  expect(cue.current).toEqual({
    before: "",
    changed: "one two three four five six seven eight",
    after: "nine ten eleven twelve …",
  });
});

test("treats equal text as unchanged", () => {
  const cue = buildChangeCue("Payroll closes Friday.", "Payroll closes Friday.");
  expect(cue.current?.changed).toBe("");
  expect(cue.proposed?.changed).toBe("");
});
