import { expect, test } from "vitest";
import { builtInRole, splitProjects } from "./projectList";
import type { Project } from "./types";

function project(key: string, name: string): Project {
  return { id: `id-${key}`, key, name, memory_count: 0, created_at: null };
}

test("lists own projects by name and the built-in ones after them", () => {
  const { own, builtIn } = splitProjects([
    project("UNSORTED", "Unsorted"),
    project("PAY", "payments"),
    project("SHARED", "Shared"),
    project("MEMFORGE", "MemForge"),
  ]);
  expect(own.map((item) => item.key)).toEqual(["MEMFORGE", "PAY"]);
  expect(builtIn.map((item) => item.key)).toEqual(["SHARED", "UNSORTED"]);
});

test("names what each built-in project is for", () => {
  expect(builtInRole("SHARED")).toBe("Team-wide");
  expect(builtInRole("UNSORTED")).toBe("Catch-all");
  expect(builtInRole("PAY")).toBeUndefined();
});
