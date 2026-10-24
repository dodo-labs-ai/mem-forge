import { expect, test, type Page } from "@playwright/test";
import { appPath, appUrl } from "./support/appPath";

interface Alias {
  alias: string;
  alias_normalized: string;
  source: string;
  created_at: string | null;
}

interface Entity {
  id: number;
  canonical_name: string;
  display_name: string;
  created_at: string;
  aliases: Alias[];
  linked_memory_count: number;
  can_curate: boolean;
}

function workspaceEntities(): Map<number, Entity> {
  return new Map([
    [
      7,
      {
        id: 7,
        canonical_name: "cut-off",
        display_name: "Cut-off",
        created_at: "2026-07-02T09:00:00Z",
        aliases: [{ alias: "Cutoff", alias_normalized: "cutoff", source: "admin_manual", created_at: null }],
        linked_memory_count: 12,
        can_curate: true,
      },
    ],
    [
      11,
      {
        id: 11,
        canonical_name: "payroll cutoff",
        display_name: "Payroll cutoff",
        created_at: "2026-07-05T09:00:00Z",
        aliases: [],
        linked_memory_count: 30,
        can_curate: true,
      },
    ],
  ]);
}

/** A workspace with two entities behind the entity routes, changed by alias and merge requests as the server would. */
async function stubApi(page: Page) {
  const entities = workspaceEntities();
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const { pathname, searchParams } = new URL(request.url());
    const key = `${request.method()} ${pathname}`;
    if (key === "GET /api/v1/memory-reviews") return route.fulfill({ json: { data: [], total: 0 } });
    if (key === "GET /api/v1/entities") {
      const search = (searchParams.get("search") ?? "").toLowerCase();
      const rows = [...entities.values()]
        .filter((entity) => entity.display_name.toLowerCase().includes(search))
        .map(({ id, canonical_name, display_name, created_at }) => ({ id, canonical_name, display_name, created_at }));
      return route.fulfill({ json: { data: rows, total: rows.length } });
    }
    if (key === "POST /api/v1/entities/merge") {
      const { source_id, target_id } = request.postDataJSON() as { source_id: number; target_id: number };
      const source = entities.get(source_id)!;
      const target = entities.get(target_id)!;
      entities.delete(source_id);
      entities.set(target_id, { ...target, aliases: [...target.aliases, ...source.aliases], linked_memory_count: 38 });
      return route.fulfill({
        json: {
          ok: true,
          merged: { source_id, source_name: source.canonical_name, target_id, target_name: target.canonical_name },
        },
      });
    }
    const aliasMatch = /^\/api\/v1\/entities\/(\d+)\/aliases(?:\/(.+))?$/.exec(pathname);
    if (aliasMatch) {
      const entity = entities.get(Number(aliasMatch[1]))!;
      if (request.method() === "POST") {
        const { alias } = request.postDataJSON() as { alias: string };
        entity.aliases.push({ alias, alias_normalized: alias.toLowerCase(), source: "admin_manual", created_at: null });
        return route.fulfill({ json: { ok: true, alias, alias_normalized: alias.toLowerCase() } });
      }
      const removed = decodeURIComponent(aliasMatch[2]!);
      entity.aliases = entity.aliases.filter((alias) => alias.alias !== removed);
      return route.fulfill({ json: { ok: true } });
    }
    const detailMatch = /^\/api\/v1\/entities\/(\d+)$/.exec(pathname);
    const entity = detailMatch ? entities.get(Number(detailMatch[1])) : undefined;
    if (request.method() === "GET" && entity) return route.fulfill({ json: entity });
    return route.fulfill({ status: 404, json: { detail: "Not stubbed" } });
  });
}

test("the Entities page searches names and opens an entity's aliases", async ({ page }) => {
  await stubApi(page);
  await page.goto(appPath("entities"));

  await expect(page.getByRole("heading", { name: "Entities" })).toBeVisible();
  await expect(page.getByRole("navigation", { name: "Main" }).getByRole("link", { name: "Entities" })).toHaveAttribute(
    "aria-current",
    "page",
  );
  await page.getByRole("searchbox", { name: "Search entities" }).fill("payroll");
  await expect(page).toHaveURL(appUrl("entities?search=payroll"));
  await expect(page.getByRole("link", { name: "Cut-off" })).toHaveCount(0);

  await page.getByRole("searchbox", { name: "Search entities" }).fill("");
  await page.getByRole("link", { name: "Cut-off" }).click();
  await expect(page).toHaveURL(appUrl("entities/7"));
  await expect(page.getByRole("heading", { level: 1, name: "Cut-off" })).toBeVisible();

  await page.getByRole("textbox", { name: "New alias" }).fill("Cut off date");
  await page.getByRole("button", { name: "Add" }).click();
  const aliases = page.getByRole("list", { name: "Aliases" });
  await expect(aliases.getByText("Cut off date")).toBeVisible();
  await page.getByRole("button", { name: "Remove alias Cut off date" }).click();
  await expect(aliases.getByText("Cut off date")).toHaveCount(0);
});

test("merging an entity shows both counts and opens the target", async ({ page }) => {
  await stubApi(page);
  await page.goto(appPath("entities/7"));

  await page.getByRole("button", { name: "Merge into…" }).click();
  await page.getByRole("dialog", { name: "Merge Cut-off" }).getByRole("button", { name: /Payroll cutoff/ }).click();
  const dialog = page.getByRole("dialog", { name: "Merge into Payroll cutoff?" });
  await expect(dialog.getByText("12 linked memories now")).toBeVisible();
  await expect(dialog.getByText("30 linked memories now")).toBeVisible();
  await dialog.getByRole("button", { name: "Merge" }).click();

  await expect(page).toHaveURL(appUrl("entities/11"));
  await expect(page.getByRole("heading", { level: 1, name: "Payroll cutoff" })).toBeVisible();
  await expect(page.getByRole("list", { name: "Aliases" }).getByText("Cutoff")).toBeVisible();
});

test("a link with a search, as a memory's entity chip makes, opens the list searched for that name", async ({ page }) => {
  await stubApi(page);
  await page.goto(appPath("entities?search=Cut-off"));

  await expect(page.getByRole("searchbox", { name: "Search entities" })).toHaveValue("Cut-off");
  await expect(page.getByRole("link", { name: "Cut-off" })).toBeVisible();
  await expect(page).toHaveTitle(/^Entities/);
});
