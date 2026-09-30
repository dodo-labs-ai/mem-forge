import { expect, test, type Page } from "@playwright/test";

const HANDBOOK = { source_id: "src-handbook", source_type: "confluence", name: "Payroll handbook" };
const BOARD = { source_id: "src-board", source_type: "jira", name: "SFPAY board" };

const ONDEMAND = {
  memory_id: "mem-ondemand",
  summary: "The on-demand lifecycle does not support cut-off yet.",
  content_hash: "hash-ondemand",
  sources: [BOARD],
  evidence_time: "2026-09-18",
};

const CUTOFF = {
  id: "mem-cutoff",
  memory_type: "fact",
  content: "Cut-off runs the same way for default, deviating, and on-demand lifecycles.",
  content_hash: "hash-cutoff",
  visibility: "workspace",
  project_key: "PAY",
  corroboration_count: 2,
  status: "active",
  created_at: "2026-09-25T09:00:00Z",
  updated_at: "2026-09-25T09:00:00Z",
  sources: [HANDBOOK, BOARD],
  relations: [{ label: "contradicts", role: "peer", counterpart: ONDEMAND, decided_by: "classifier" }],
};

const CUTOFF_DETAIL = {
  ...CUTOFF,
  entity_refs: ["Cut-off", "PeriodLifecycle"],
  evidence: [
    {
      kind: "evidence_unit",
      support_ids: ["sup-1"],
      source_id: "src-handbook",
      source_type: "confluence",
      evidence_unit_id: "eu-1",
      current: true,
      document: { doc_id: "doc-handbook", title: "Payroll Handbook", source_updated_at: "2026-09-22T09:14:00Z" },
      items: [
        {
          authority: "revision_pinned",
          role: "primary",
          kind: "text",
          support_contribution: true,
          anchor_kind: "text_range",
          current: true,
          excerpt: "Cut-off is uniform across Default, Deviating and On-Demand lifecycle types.",
        },
      ],
    },
  ],
  dismissed_relations: [],
  source_backed: true,
};

const RELATION_TOTALS: Record<string, number> = { contradicts: 1, updates: 0, equivalent: 0 };

const RESPONSES: Record<string, (url: URL) => unknown> = {
  "/api/v1/memories": () => ({ data: [CUTOFF], total: 1, limit: 50, offset: 0 }),
  "/api/v1/memories/mem-cutoff": () => CUTOFF_DETAIL,
  "/api/v1/memories/stats": () => ({
    by_type: [],
    by_status: [
      { key: "active", count: 1 },
      { key: "superseded", count: 0 },
      { key: "retired", count: 0 },
    ],
    total: 1,
  }),
  "/api/v1/memory-reviews": () => ({ data: [], total: 0, limit: 500, offset: 0 }),
  "/api/v1/memories/relations": (url) => {
    const label = url.searchParams.get("label");
    if (url.searchParams.get("limit") === "1") return { data: [], total: RELATION_TOTALS[label ?? ""] ?? 0, limit: 1, offset: 0 };
    const pair = {
      label: "contradicts",
      decided_by: "classifier",
      decided_at: "2026-09-22T10:00:00Z",
      memories: [{ memory_id: CUTOFF.id, summary: CUTOFF.content, content_hash: CUTOFF.content_hash, sources: [HANDBOOK] }, ONDEMAND],
    };
    return { data: [pair], total: 1, limit: 50, offset: 0 };
  },
  "/api/v1/projects": () => ({ data: [{ id: "p1", key: "PAY", name: "payroll-v2", kind: "normal" }], can_manage: true }),
  "/api/v1/sources": () => ({ data: [{ id: "src-handbook", name: "Payroll handbook" }] }),
  "/api/v1/genes": () => [
    { name: "confluence", display_name: "Confluence" },
    { name: "jira", display_name: "Jira" },
  ],
  "/api/cloud/local-agent/status": () => ({ status: "offline", last_seen_at: null, checked_at: "", stale_after_seconds: 60 }),
  "/api/cloud/local-agent/jobs/current": () => ({ data: [] }),
};

async function stubApi(page: Page) {
  await page.route("**/api/**", (route) => {
    const url = new URL(route.request().url());
    const respond = RESPONSES[url.pathname];
    return respond ? route.fulfill({ json: respond(url) }) : route.fulfill({ status: 404, json: { detail: "Not stubbed" } });
  });
}

test("the Memories list opens a memory's detail", async ({ page }) => {
  await stubApi(page);
  await page.goto("/v2/memories");

  await expect(page.getByRole("heading", { name: "Memories", level: 1 })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Memories" })).toHaveAttribute(
    "aria-current",
    "page",
  );
  const table = page.getByRole("table", { name: "Memories" });
  await expect(table.getByText("payroll-v2")).toBeVisible();
  await expect(table.getByRole("link", { name: "1 conflict" })).toBeVisible();

  await table.getByRole("link", { name: /Cut-off runs the same way/ }).click();
  await expect(page).toHaveURL(/\/v2\/memories\/mem-cutoff$/);
  await expect(page.getByRole("heading", { level: 1, name: /Cut-off runs the same way/ })).toBeVisible();
  await expect(page.getByRole("region", { name: "Evidence" }).getByText("Payroll Handbook")).toBeVisible();
  await expect(page).toHaveTitle(/^Memories/);
});

test("the conflicts card opens the Relations view on conflicts", async ({ page }) => {
  await stubApi(page);
  await page.goto("/v2/memories");

  await page.getByRole("link", { name: "Show conflicts" }).click();
  await expect(page).toHaveURL(/\/v2\/memories\?view=relations&relation=contradicts$/);
  await expect(page.getByRole("table", { name: "Relations" }).getByText("Conflict", { exact: true })).toBeVisible();
});
