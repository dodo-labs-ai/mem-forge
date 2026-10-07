import { STANDALONE_TARGET, type components } from "@/api";
import { MemoriesRoutes } from "@/features/memories";
import type { Route as FakeRoute } from "./fakeApi";
import { makeMemory } from "./memoryFixtures";
import { renderRoutes } from "./renderRoutes";

type Memory = components["schemas"]["MemoryResponse"];

/** The status the list reads when the request names none. */
const DEFAULT_LIST_STATUS = "active";

/** Relation totals per label, answered by the count requests. */
export const RELATION_TOTALS: Record<string, number> = { contradicts: 12, updates: 31, equivalent: 58 };

/** A small workspace: two active memories, one of them private, and one that waits for a review. */
export const WORKSPACE_MEMORIES: Memory[] = [
  makeMemory({ relations: [] }),
  makeMemory({ id: "mem-private", content: "Prefer merge over rebase.", visibility: "private", sources: [] }),
  makeMemory({
    id: "mem-pending",
    content: "Clear the blocker hint first.",
    status: "pending_review",
    open_review_id: "rev-pending",
  }),
];

/**
 * `GET /memories` as the server answers it: the memories in the requested
 * status, and only active ones when the request names no status.
 */
export function memoryListRoute(memories: readonly Memory[]): FakeRoute {
  return (request) => {
    const params = new URL(request.url).searchParams;
    const status = params.get("status") ?? DEFAULT_LIST_STATUS;
    const data = memories.filter((memory) => memory.status === status);
    return { data, total: data.length, limit: Number(params.get("limit")), offset: Number(params.get("offset")) };
  };
}

/** The routes every Memories page reads, with a small workspace behind them. */
export function memoriesRoutes(overrides: Record<string, FakeRoute> = {}): Record<string, FakeRoute> {
  return {
    "GET /api/v1/memories": memoryListRoute(WORKSPACE_MEMORIES),
    "GET /api/v1/memories/stats": () => ({
      by_type: [],
      by_status: [
        { key: "active", count: 3344 },
        { key: "superseded", count: 400 },
        { key: "retired", count: 12 },
        { key: "pending_review", count: 5 },
      ],
      total: 3761,
    }),
    "GET /api/v1/memory-reviews": () => ({ data: [], total: 5, limit: 1, offset: 0 }),
    "GET /api/v1/memories/relations": (request) => {
      const label = new URL(request.url).searchParams.get("label");
      return { data: [], total: label ? (RELATION_TOTALS[label] ?? 0) : 0, limit: 1, offset: 0 };
    },
    "GET /api/v1/projects": () => ({
      data: [
        { id: "p1", key: "PAY", name: "payroll-v2", kind: "normal" },
        { id: "p2", key: "SHARED", name: "Shared", kind: "shared" },
      ],
      can_manage: true,
    }),
    "GET /api/v1/sources": () => ({ data: [{ id: "src-handbook", name: "Payroll handbook" }] }),
    "GET /api/v1/genes": () => [{ name: "confluence", display_name: "Confluence" }],
    ...overrides,
  };
}

/** Renders the Memories routes at `path` against a fake API; returns the recorded requests and the router. */
export function renderMemories(path: string, routes: Record<string, FakeRoute> = memoriesRoutes()) {
  return renderRoutes(
    [
      { path: "/memories/*", element: <MemoriesRoutes /> },
      { path: "*", element: null },
    ],
    { path, api: routes, workspace: { ...STANDALONE_TARGET, label: "Mount Tai" } },
  );
}
