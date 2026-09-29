import { describe, expect, it } from "vitest";
import searchRequestContract from "../../../tests/fixtures/memory-search-request.json";
import { buildMemorySearchRequest } from "@/views/memories/memorySearchRequest";

// tests/test_phase4_search_request.py validates the same fixture against the
// backend MemorySearchRequest model, so the two sides cannot drift apart.
describe("buildMemorySearchRequest", () => {
  it("sends project, query, type, status, and source in the backend request shape", () => {
    const body = buildMemorySearchRequest({
      query: "payroll cutoff",
      type: "decision",
      status: "active",
      source: "src-jira",
      activeProject: "PAY",
      scopeMode: "project",
    });

    expect(body).toEqual(searchRequestContract);
  });

  it("omits filters left at all", () => {
    const body = buildMemorySearchRequest({
      query: "payroll cutoff",
      type: "all",
      status: "all",
      source: "all",
      activeProject: "PAY",
      scopeMode: "project-first",
    });

    expect(body).toEqual({
      query: "payroll cutoff",
      active_project: "PAY",
      scope_mode: "project-first",
      include_private: true,
      top_k: 50,
    });
  });
});
