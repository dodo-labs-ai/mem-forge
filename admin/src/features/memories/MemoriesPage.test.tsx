import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import { makeMemory, makeRelatedMemory, makeRelation, makeRelationPair } from "@/test/memoryFixtures";
import { memoriesRoutes, memoryListRoute, RELATION_TOTALS, renderMemories, WORKSPACE_MEMORIES } from "@/test/renderMemories";

afterEach(() => {
  vi.unstubAllGlobals();
});

function requestsTo(calls: Request[], method: string, pathname: string): URL[] {
  return calls
    .filter((request) => request.method === method && new URL(request.url).pathname === pathname)
    .map((request) => new URL(request.url));
}

test("shows the overview and each memory with its project, source, access and relations", async () => {
  renderMemories(
    "/memories",
    memoriesRoutes({
      "GET /api/v1/memories": memoryListRoute([
        makeMemory({
          sources: [
            { source_id: "src-handbook", source_type: "confluence", name: "Payroll handbook" },
            { source_id: "src-board", source_type: "jira", name: "SFPAY board" },
          ],
          relations: [
            makeRelation(),
            makeRelation({ label: "updates", role: "older", counterpart: makeRelatedMemory({ memory_id: "mem-newer" }) }),
          ],
        }),
        makeMemory({ id: "mem-private", content: "Prefer merge over rebase.", visibility: "private", project_key: "SHARED" }),
      ]),
    }),
  );

  const overview = await screen.findByRole("region", { name: "Overview" });
  expect(await within(overview).findByText("3,344")).toBeInTheDocument();
  expect(within(overview).getByText("412 superseded or retired")).toBeInTheDocument();
  expect(await within(overview).findByText("Also 31 updates, 58 same knowledge")).toBeInTheDocument();
  expect(within(overview).getByRole("link", { name: "Show conflicts" })).toHaveAttribute(
    "href",
    "/memories?view=relations&relation=contradicts",
  );

  const table = await screen.findByRole("table", { name: "Memories" });
  const cutoff = within(table).getByRole("link", { name: /Cut-off runs the same way/ }).closest("tr")!;
  expect(within(cutoff).getByText("payroll-v2")).toBeInTheDocument();
  expect(within(cutoff).getByText("Payroll handbook")).toBeInTheDocument();
  expect(within(cutoff).getByText("and 1 more")).toBeInTheDocument();
  expect(within(cutoff).getByRole("link", { name: "1 conflict" })).toHaveAttribute("href", "/memories/mem-cutoff");
  expect(within(cutoff).getByRole("link", { name: "Updated by a newer memory" })).toHaveAttribute("href", "/memories/mem-newer");

  const privateRow = within(table).getByRole("link", { name: "Prefer merge over rebase." }).closest("tr")!;
  expect(within(privateRow).getByText("Only me")).toBeInTheDocument();
  expect(within(privateRow).getByText("Shared")).toBeInTheDocument();
});

test("the list shows active memories until a status filter asks for another status", async () => {
  const user = userEvent.setup();
  const { calls } = renderMemories(
    "/memories",
    memoriesRoutes({
      "GET /api/v1/memories": memoryListRoute([
        ...WORKSPACE_MEMORIES,
        makeMemory({ id: "mem-retired", content: "Scheduled syncs have their own page.", status: "retired" }),
      ]),
    }),
  );

  const table = await screen.findByRole("table", { name: "Memories" });
  expect(await within(table).findByRole("link", { name: /Cut-off runs the same way/ })).toBeInTheDocument();
  expect(within(table).queryByRole("link", { name: "Scheduled syncs have their own page." })).not.toBeInTheDocument();
  expect(requestsTo(calls, "GET", "/api/v1/memories")[0]!.searchParams.has("status")).toBe(false);

  await user.click(screen.getByRole("button", { name: "Filters" }));
  await user.click(screen.getByRole("combobox", { name: "Status" }));
  expect(screen.getByRole("option", { name: "Active" })).toHaveAttribute("aria-selected", "true");
  expect(screen.queryByRole("option", { name: "All statuses" })).not.toBeInTheDocument();
  await user.click(screen.getByRole("option", { name: "Retired" }));

  const retiredRow = (await within(table).findByRole("link", { name: "Scheduled syncs have their own page." })).closest("tr")!;
  expect(within(retiredRow).getByText("Retired")).toBeInTheDocument();
  expect(within(table).queryByRole("link", { name: /Cut-off runs the same way/ })).not.toBeInTheDocument();
  expect(requestsTo(calls, "GET", "/api/v1/memories").at(-1)!.searchParams.get("status")).toBe("retired");
  expect(screen.getByRole("button", { name: "Remove Status filter" })).toBeInTheDocument();
});

test("a memory that needs review opens its review", async () => {
  renderMemories("/memories?status=pending_review");

  const table = await screen.findByRole("table", { name: "Memories" });
  expect(await within(table).findByRole("link", { name: "Clear the blocker hint first." })).toHaveAttribute(
    "href",
    "/review/rev-pending",
  );
});

test("the review count reads only the total of the open queue", async () => {
  const { calls } = renderMemories("/memories");

  const overview = await screen.findByRole("region", { name: "Overview" });
  expect(await within(overview).findByText("5")).toBeInTheDocument();
  const reviewRequests = requestsTo(calls, "GET", "/api/v1/memory-reviews");
  expect(reviewRequests).toHaveLength(1);
  expect(reviewRequests[0]!.searchParams.get("status")).toBe("open");
  expect(reviewRequests[0]!.searchParams.get("limit")).toBe("1");
});

test("a project in the URL filters the listing", async () => {
  const { calls } = renderMemories("/memories?project=PAY");

  expect(await screen.findByRole("button", { name: "Remove Project filter" })).toBeInTheDocument();
  await vi.waitFor(() => expect(requestsTo(calls, "GET", "/api/v1/memories")).not.toHaveLength(0));
  const [listing] = requestsTo(calls, "GET", "/api/v1/memories");
  expect(listing!.searchParams.get("project")).toBe("PAY");
  expect(listing!.searchParams.get("include_private")).toBe("true");
});

test("searching within a project ranks results and sends the source as a source filter", async () => {
  const { calls } = renderMemories(
    "/memories?project=PAY&q=cut-off&source=src-handbook",
    memoriesRoutes({
      "POST /api/v1/memories/search": () => ({
        query_analysis: {},
        results: [
          {
            memory_id: "mem-hit",
            memory_type: "procedure",
            summary: "Clear the blocker hint, then trigger the cut-off.",
            relevance_score: 0.9,
            corroborated_by: 1,
            freshness: "current",
            status: "active",
            relations: [],
          },
        ],
        total_candidates: 18,
        candidate_count_kind: "windowed",
        ranking_window_size: 50,
        limit: 50,
        offset: 0,
        has_more: false,
        retrieval_time_ms: 3,
      }),
    }),
  );

  expect(await screen.findByText("Clear the blocker hint, then trigger the cut-off.")).toBeInTheDocument();
  expect(screen.getByText("18 candidates")).toBeInTheDocument();
  expect(
    screen.getByText("Search ranks results within payroll-v2 first, then team-wide. Showing the top 50 matches, so pages are not shown."),
  ).toBeInTheDocument();
  expect(screen.queryByRole("navigation", { name: "Pagination" })).not.toBeInTheDocument();

  const search = calls.find((request) => request.method === "POST")!;
  expect(await search.clone().json()).toMatchObject({
    query: "cut-off",
    active_project: "PAY",
    scope_mode: "project-first",
    source_filter: { source_ids: ["src-handbook"] },
  });
});

test("the Relations view lists pairs by label with each label's count", async () => {
  const user = userEvent.setup();
  const { calls, router } = renderMemories(
    "/memories?view=relations",
    memoriesRoutes({
      "GET /api/v1/memories/relations": (request) => {
        const params = new URL(request.url).searchParams;
        const label = params.get("label");
        if (params.get("limit") === "1") return { data: [], total: RELATION_TOTALS[label ?? ""] ?? 0, limit: 1, offset: 0 };
        const older = makeRelatedMemory({ memory_id: "mem-old", summary: "On-demand pay dates can be any date." });
        const newer = makeRelatedMemory({ memory_id: "mem-new", summary: "On-demand pay dates must fall before the next pay date." });
        const pairs = [
          makeRelationPair(),
          makeRelationPair({ label: "updates", newer_memory_id: "mem-new", decided_by: "review", memories: [older, newer] }),
        ];
        const shown = label ? pairs.filter((pair) => pair.label === label) : pairs;
        return { data: shown, total: shown.length, limit: 50, offset: 0 };
      },
    }),
  );

  const table = await screen.findByRole("table", { name: "Relations" });
  expect(await within(table).findByText("Conflict")).toBeInTheDocument();
  const update = within(table).getByText("Update").closest("tr")!;
  const [newerCell, olderCell] = within(update).getAllByText(/^(Newer|Older)$/);
  expect(newerCell).toHaveTextContent("Newer");
  expect(olderCell).toHaveTextContent("Older");
  expect(within(update).getByText("Reviewer")).toBeInTheDocument();

  const relationFilter = screen.getByRole("group", { name: "Relation" });
  expect(within(relationFilter).getByRole("button", { name: /All\s*101/ })).toBeInTheDocument();
  await user.click(within(relationFilter).getByRole("button", { name: /Updates\s*31/ }));

  expect(router.state.location.search).toBe("?view=relations&relation=updates");
  await vi.waitFor(() =>
    expect(
      requestsTo(calls, "GET", "/api/v1/memories/relations").some(
        (url) => url.searchParams.get("label") === "updates" && url.searchParams.get("limit") === "50",
      ),
    ).toBe(true),
  );
});
