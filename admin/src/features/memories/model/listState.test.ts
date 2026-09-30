import { describe, expect, test } from "vitest";
import { projectMemoriesPath } from "@/lib/paths";
import {
  activeFilterKeys,
  changeFilters,
  clearFilters,
  memoryListRequest,
  readPageState,
  relationsPageState,
  writePageState,
} from "./listState";

function stateFrom(search: string) {
  return readPageState(new URLSearchParams(search));
}

describe("page state in the URL", () => {
  test("defaults to the first page of active memories", () => {
    expect(stateFrom("")).toEqual({
      view: "memories",
      query: "",
      filters: { type: null, status: null, source: null, project: null, projectScope: "project-first" },
      relation: null,
      page: 1,
    });
  });

  test("round-trips every value and leaves defaults out", () => {
    const search = "view=relations&q=cut-off&type=fact&status=retired&source=src-a&project=PAY&scope=only&relation=updates&page=3";
    expect(writePageState(stateFrom(search)).toString()).toBe(search);
    expect(writePageState(stateFrom("")).toString()).toBe("");
  });

  test("reads an active status as the default list", () => {
    expect(stateFrom("status=active").filters.status).toBeNull();
    expect(writePageState(stateFrom("status=active")).toString()).toBe("");
  });

  test("drops values the page does not offer", () => {
    const state = stateFrom("view=grid&type=rumor&status=deleted&relation=maybe&page=-2&source=");
    expect(state.view).toBe("memories");
    expect(state.filters).toMatchObject({ type: null, status: null, source: null });
    expect(state.relation).toBeNull();
    expect(state.page).toBe(1);
  });

  test("keeps the project scope only while a project is chosen", () => {
    const state = changeFilters(stateFrom("project=PAY&scope=only"), { project: null });
    expect(writePageState(state).has("scope")).toBe(false);
  });

  test("reads the project filter other pages link to", () => {
    expect(stateFrom(projectMemoriesPath("PAY 2").split("?")[1]!).filters.project).toBe("PAY 2");
  });
});

describe("changing filters", () => {
  test("returns to the first page", () => {
    expect(changeFilters(stateFrom("page=4"), { type: "decision" }).page).toBe(1);
    expect(clearFilters(stateFrom("type=fact&project=PAY&page=2"))).toEqual(stateFrom(""));
  });

  test("lists the active filters in toolbar order", () => {
    expect(activeFilterKeys(stateFrom("project=PAY&type=fact").filters)).toEqual(["type", "project"]);
  });
});

describe("the list request", () => {
  test("pages the admin listing with keyword search and every filter", () => {
    expect(memoryListRequest(stateFrom("q=cut-off&type=fact&status=retired&source=src-a&page=3"))).toEqual({
      kind: "list",
      query: {
        search: "cut-off",
        type: "fact",
        status: "retired",
        source: "src-a",
        include_private: true,
        limit: 50,
        offset: 100,
      },
    });
  });

  test("ranks a search within a project, sending the source as a source filter", () => {
    expect(memoryListRequest(stateFrom("q=cut-off&project=PAY&scope=only&type=procedure&source=src-a"))).toEqual({
      kind: "search",
      body: {
        query: "cut-off",
        active_project: "PAY",
        scope_mode: "project",
        include_private: true,
        include_superseded: false,
        top_k: 50,
        offset: 0,
        memory_types: ["procedure"],
        source_filter: { source_ids: ["src-a"] },
      },
    });
  });

  test("lists a project without search text", () => {
    expect(memoryListRequest(stateFrom("project=PAY")).kind).toBe("list");
    expect(memoryListRequest(stateFrom("q=%20%20&project=PAY")).kind).toBe("list");
  });
});

test("the conflicts link opens the Relations view on conflicts alone", () => {
  expect(writePageState(relationsPageState("contradicts")).toString()).toBe("view=relations&relation=contradicts");
});
