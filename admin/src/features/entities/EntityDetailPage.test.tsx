import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import { entityRow, makeEntityDetail } from "@/test/entityFixtures";
import { errorResponse, type Route as FakeRoute } from "@/test/fakeApi";
import { renderRoutes } from "@/test/renderRoutes";
import { EntitiesRoutes } from "./EntitiesRoutes";

afterEach(() => {
  vi.unstubAllGlobals();
});

const CUT_OFF = makeEntityDetail();
const PAYROLL_CUT_OFF = makeEntityDetail({
  id: 11,
  canonical_name: "payroll cutoff",
  display_name: "Payroll cutoff",
  aliases: [],
  linked_memory_count: 30,
});
const FORBIDDEN = {
  error: "entity_curation_forbidden",
  message: "Only a workspace admin can change entity aliases or merge entities.",
};

function renderAt(path: string, routes: Record<string, FakeRoute> = {}) {
  return renderRoutes([{ path: "entities/*", element: <EntitiesRoutes /> }], {
    path,
    api: {
      [`GET /api/v1/entities/${CUT_OFF.id}`]: () => CUT_OFF,
      [`GET /api/v1/entities/${PAYROLL_CUT_OFF.id}`]: () => PAYROLL_CUT_OFF,
      "GET /api/v1/entities": () => ({ data: [entityRow(CUT_OFF), entityRow(PAYROLL_CUT_OFF)], total: 2 }),
      ...routes,
    },
  });
}

function requestsTo(calls: Request[], method: string, path: string): Request[] {
  return calls.filter((request) => request.method === method && new URL(request.url).pathname === path);
}

test("shows aliases and the linked memory count", async () => {
  renderAt("/entities/7");

  expect(await screen.findByRole("heading", { level: 1, name: "Cut-off" })).toBeInTheDocument();
  expect(screen.getByText("cut-off")).toBeInTheDocument();
  expect(screen.getByText("Linked memories").parentElement).toHaveTextContent("12");
  const aliases = screen.getByRole("list", { name: "Aliases" });
  const manual = within(aliases).getByText("Cutoff").closest("li")!;
  expect(within(manual).getByText("Added by an admin")).toBeInTheDocument();
  const found = within(aliases).getByText("Payroll cut-off").closest("li")!;
  expect(within(found).getByText("Found by MemForge")).toBeInTheDocument();
});

test("adds and removes an alias", async () => {
  let entity = CUT_OFF;
  const added: unknown[] = [];
  const { calls } = renderAt("/entities/7", {
    "GET /api/v1/entities/7": () => entity,
    "POST /api/v1/entities/7/aliases": async (request) => {
      const body = (await request.json()) as { alias: string };
      added.push(body);
      const alias = { alias: body.alias, alias_normalized: body.alias.toLowerCase(), source: "admin_manual", created_at: null };
      entity = { ...entity, aliases: [...entity.aliases, alias] };
      return { ok: true, alias: alias.alias, alias_normalized: alias.alias_normalized };
    },
    "DELETE /api/v1/entities/7/aliases/Payroll%20cut-off": () => {
      entity = { ...entity, aliases: entity.aliases.filter((alias) => alias.alias !== "Payroll cut-off") };
      return { ok: true };
    },
  });

  const input = await screen.findByRole("textbox", { name: "New alias" });
  await userEvent.type(input, "  Cut off date ");
  await userEvent.click(screen.getByRole("button", { name: "Add" }));
  const aliases = screen.getByRole("list", { name: "Aliases" });
  expect(await within(aliases).findByText("Cut off date")).toBeInTheDocument();
  expect(added).toEqual([{ alias: "Cut off date" }]);
  expect(input).toHaveValue("");

  await userEvent.click(screen.getByRole("button", { name: "Remove alias Payroll cut-off" }));
  await waitFor(() => expect(within(aliases).queryByText("Payroll cut-off")).not.toBeInTheDocument());
  expect(requestsTo(calls, "DELETE", "/api/v1/entities/7/aliases/Payroll%20cut-off")).toHaveLength(1);
});

/** Opens the merge dialog and chooses Payroll cutoff as the target. */
async function chooseMergeTarget() {
  await userEvent.click(await screen.findByRole("button", { name: "Merge into…" }));
  const dialog = await screen.findByRole("dialog", { name: "Merge Cut-off" });
  const candidates = await within(dialog).findByRole("list", { name: "Matching entities" });
  expect(within(candidates).queryByText("Cut-off")).not.toBeInTheDocument();
  await userEvent.click(within(candidates).getByRole("button", { name: /Payroll cutoff/ }));
  return screen.findByRole("dialog", { name: "Merge into Payroll cutoff?" });
}

test("the merge dialog shows both entities' linked memory counts", async () => {
  renderAt("/entities/7");

  const dialog = await chooseMergeTarget();
  expect(within(dialog).getByText("Merged away: Cut-off").nextElementSibling).toHaveTextContent("12 linked memories now");
  expect(await within(dialog).findByText("30 linked memories now")).toBeInTheDocument();
  expect(within(dialog).getByText(/each entity’s current links, not a total after the merge/)).toBeInTheDocument();
});

test("merge moves references and opens the target", async () => {
  let target = PAYROLL_CUT_OFF;
  const merges: unknown[] = [];
  const { calls } = renderAt("/entities/7", {
    "GET /api/v1/entities/11": () => target,
    "POST /api/v1/entities/merge": async (request) => {
      merges.push(await request.json());
      target = { ...target, aliases: CUT_OFF.aliases, linked_memory_count: 38 };
      return { ok: true, merged: { source_id: 7, source_name: "cut-off", target_id: 11, target_name: "payroll cutoff" } };
    },
  });

  const dialog = await chooseMergeTarget();
  await within(dialog).findByText("30 linked memories now");
  await userEvent.click(within(dialog).getByRole("button", { name: "Merge" }));

  expect(await screen.findByRole("heading", { level: 1, name: "Payroll cutoff" })).toBeInTheDocument();
  expect(merges).toEqual([{ source_id: 7, target_id: 11 }]);
  expect(await screen.findByText("Linked memories")).toBeInTheDocument();
  expect(screen.getByText("Linked memories").parentElement).toHaveTextContent("38");
  expect(within(screen.getByRole("list", { name: "Aliases" })).getByText("Cutoff")).toBeInTheDocument();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  expect(requestsTo(calls, "GET", "/api/v1/entities/7")).toHaveLength(1);
});

test("hides alias and merge controls when can_curate is false", async () => {
  renderAt("/entities/7", { "GET /api/v1/entities/7": () => ({ ...CUT_OFF, can_curate: false }) });

  expect(await screen.findByRole("heading", { level: 1, name: "Cut-off" })).toBeInTheDocument();
  expect(within(screen.getByRole("list", { name: "Aliases" })).getByText("Cutoff")).toBeInTheDocument();
  expect(screen.queryByRole("textbox", { name: "New alias" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /Remove alias/ })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Merge into…" })).not.toBeInTheDocument();
});

test("a 403 entity_curation_forbidden shows the server message", async () => {
  let canCurate = true;
  const { calls } = renderAt("/entities/7", {
    "GET /api/v1/entities/7": () => ({ ...CUT_OFF, can_curate: canCurate }),
    "POST /api/v1/entities/7/aliases": () => {
      canCurate = false;
      return errorResponse(403, FORBIDDEN);
    },
  });

  await userEvent.type(await screen.findByRole("textbox", { name: "New alias" }), "Cut off date");
  await userEvent.click(screen.getByRole("button", { name: "Add" }));

  const notice = await screen.findByRole("alert");
  expect(notice).toHaveTextContent("Could not change the entity");
  expect(notice).toHaveTextContent(FORBIDDEN.message);
  await waitFor(() => expect(screen.queryByRole("textbox", { name: "New alias" })).not.toBeInTheDocument());
  expect(screen.queryByRole("button", { name: "Merge into…" })).not.toBeInTheDocument();
  expect(requestsTo(calls, "GET", "/api/v1/entities/7").length).toBeGreaterThanOrEqual(2);
});

test("a 403 entity_curation_forbidden on merge closes the dialog and shows the server message", async () => {
  let canCurate = true;
  renderAt("/entities/7", {
    "GET /api/v1/entities/7": () => ({ ...CUT_OFF, can_curate: canCurate }),
    "POST /api/v1/entities/merge": () => {
      canCurate = false;
      return errorResponse(403, FORBIDDEN);
    },
  });

  const dialog = await chooseMergeTarget();
  await within(dialog).findByText("30 linked memories now");
  await userEvent.click(within(dialog).getByRole("button", { name: "Merge" }));

  expect(await screen.findByText(FORBIDDEN.message)).toBeInTheDocument();
  expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  await waitFor(() => expect(screen.queryByRole("button", { name: "Merge into…" })).not.toBeInTheDocument());
});

test("an entity that no longer exists says so", async () => {
  renderAt("/entities/99", { "GET /api/v1/entities/99": () => errorResponse(404, "Entity not found") });

  expect(await screen.findByText("Entity not found")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Back to Entities" })).toHaveAttribute("href", "/entities");
});
