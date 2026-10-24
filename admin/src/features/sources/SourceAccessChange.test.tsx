import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import { STANDALONE_TARGET, type Source, type WorkspaceTarget } from "@/api";
import type { Route } from "@/test/fakeApi";
import { makeOverview } from "@/test/evaluationFixtures";
import { renderRoutes } from "@/test/renderRoutes";
import { makeSource } from "@/test/sourceFixtures";
import { SourcesPage } from "./SourcesPage";

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

const SHARED: WorkspaceTarget = { ...STANDALONE_TARGET, shared: true };
/** How long a test waits for the page to read a source again; it does so every two seconds while access changes. */
const NEXT_READ = { timeout: 5_000 };
const POLLING_TEST_MS = 15_000;

type Transition = NonNullable<Source["access_transition"]>;

function transition(overrides: Partial<Transition> = {}): Transition {
  return {
    operation_id: "sat-1",
    source_id: "src-wiki",
    previous_policy: "private",
    target_policy: "workspace",
    status: "queued",
    total_memories: 0,
    processed_memories: 0,
    error_code: null,
    error_message: null,
    created_at: "2026-10-24T09:00:00Z",
    updated_at: "2026-10-24T09:00:00Z",
    completed_at: null,
    ...overrides,
  };
}

function renderPage(sources: () => Source[], routes: Record<string, Route> = {}, workspace: WorkspaceTarget = SHARED) {
  return renderRoutes([{ path: "/sources", element: <SourcesPage /> }], {
    path: "/sources",
    workspace,
    api: {
      "GET /api/v1/sources": () => ({ data: sources() }),
      "GET /api/v1/projects": () => ({ data: [{ id: "p1", key: "PAY", name: "Payroll", kind: "normal" }], can_manage: true }),
      "GET /api/v1/genes": () => [{ name: "confluence", display_name: "Confluence" }],
      "GET /api/v1/source-list/preferences": () => ({ sort_mode: "name" }),
      "GET /api/cloud/local-agent/status": () => ({ status: "online", last_seen_at: null, checked_at: "", stale_after_seconds: 60 }),
      "GET /api/cloud/local-agent/jobs/current": () => ({ data: [] }),
      "GET /api/v1/agent-evaluations/online-overview": () => makeOverview({ sources: [] }),
      ...routes,
    },
  });
}

async function openDrawer() {
  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByText("Payroll wiki"));
  return screen.findByRole("dialog", { name: "Payroll wiki" });
}

test("hidden in a standalone workspace", async () => {
  renderPage(() => [makeSource({ access_policy: "private" })], {}, STANDALONE_TARGET);

  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByRole("button", { name: "More actions for Payroll wiki" }));
  expect(await screen.findByRole("menuitem", { name: "View details" })).toBeInTheDocument();
  expect(screen.queryByRole("menuitem", { name: "Change who can use this source" })).not.toBeInTheDocument();
  await userEvent.keyboard("{Escape}");

  const drawer = await openDrawer();
  await userEvent.click(within(drawer).getByRole("button", { name: "More actions for Payroll wiki" }));
  expect(await screen.findByRole("menuitem", { name: /Full sync/ })).toBeInTheDocument();
  expect(screen.queryByRole("menuitem", { name: "Change who can use this source" })).not.toBeInTheDocument();
});

test("hidden from a viewer who cannot manage the source", async () => {
  const source = makeSource({ capabilities: { ...makeSource().capabilities!, can_change_access: false } });
  renderPage(() => [source]);

  const table = await screen.findByRole("table", { name: "Sources" });
  await userEvent.click(await within(table).findByRole("button", { name: "More actions for Payroll wiki" }));
  expect(await screen.findByRole("menuitem", { name: "View details" })).toBeInTheDocument();
  expect(screen.queryByRole("menuitem", { name: "Change who can use this source" })).not.toBeInTheDocument();
});

test("starts from the drawer menu over the open drawer", async () => {
  const { calls } = renderPage(() => [makeSource({ access_policy: "private" })], {
    "POST /api/v1/sources/src-wiki/access-transitions": () => transition(),
  });

  const drawer = await openDrawer();
  await userEvent.click(within(drawer).getByRole("button", { name: "More actions for Payroll wiki" }));
  await userEvent.click(await screen.findByRole("menuitem", { name: "Change who can use this source" }));
  const dialog = await screen.findByRole("dialog", { name: "Change who can use Payroll wiki" });
  await userEvent.click(within(dialog).getByRole("button", { name: "Change access" }));

  await waitFor(() => expect(screen.queryByRole("dialog", { name: "Change who can use Payroll wiki" })).not.toBeInTheDocument());
  expect(calls.some((call) => call.method === "POST" && new URL(call.url).pathname.endsWith("/access-transitions"))).toBe(true);
  expect(screen.getByRole("dialog", { name: "Payroll wiki" })).toBeInTheDocument();
});

test(
  "private to workspace polls until applied",
  async () => {
    const changing = [
      transition(),
      transition({ status: "running", total_memories: 3, processed_memories: 1 }),
    ];
    let started = false;
    let reads = 0;
    const sources = () => {
      if (!started) return [makeSource({ access_policy: "private" })];
      const step = changing[reads++];
      return [
        step
          ? makeSource({ access_policy: "private", access_state: "changing", access_transition: step })
          : makeSource({ access_policy: "workspace" }),
      ];
    };
    const { calls } = renderPage(sources, {
      "POST /api/v1/sources/src-wiki/access-transitions": () => {
        started = true;
        return transition();
      },
    });

    const table = await screen.findByRole("table", { name: "Sources" });
    expect(await within(table).findByText("Only me")).toBeInTheDocument();
    await userEvent.click(within(table).getByRole("button", { name: "More actions for Payroll wiki" }));
    await userEvent.click(await screen.findByRole("menuitem", { name: "Change who can use this source" }));

    const dialog = await screen.findByRole("dialog", { name: "Change who can use Payroll wiki" });
    expect(within(dialog).getByRole("radio", { name: "Everyone in this workspace" })).toBeChecked();
    await userEvent.click(within(dialog).getByRole("button", { name: "Change access" }));

    await waitFor(() => expect(screen.queryByRole("dialog", { name: "Change who can use Payroll wiki" })).not.toBeInTheDocument());
    const start = calls.find((call) => call.method === "POST" && new URL(call.url).pathname.endsWith("/access-transitions"))!;
    expect(await start.clone().json()).toEqual({ target_policy: "workspace" });
    expect(start.headers.get("Idempotency-Key")).toBeTruthy();

    expect(await within(table).findByText("Changing access")).toBeInTheDocument();
    const drawer = await openDrawer();
    expect(within(drawer).getByText("Changing access to Everyone in this workspace")).toBeInTheDocument();
    expect(await within(drawer).findByText("1 of 3 memories updated", undefined, NEXT_READ)).toBeInTheDocument();

    expect(await within(table).findByText("Workspace", undefined, NEXT_READ)).toBeInTheDocument();
    expect(within(drawer).queryByText("Access change")).not.toBeInTheDocument();
  },
  POLLING_TEST_MS,
);

test("a failed change offers retry and revert", async () => {
  const failed = transition({ status: "failed", total_memories: 3, processed_memories: 1, error_message: "Search index offline" });
  let current: Transition = failed;
  const { calls } = renderPage(() => [makeSource({ access_policy: "private", access_state: "changing", access_transition: current })], {
    "POST /api/v1/sources/src-wiki/access-transitions/sat-1/retry": () => {
      current = transition({ status: "running", total_memories: 3, processed_memories: 1, updated_at: "2026-10-24T09:05:00Z" });
      return { operation_id: "sat-1", status: "queued" };
    },
    "POST /api/v1/sources/src-wiki/access-transitions/sat-1/revert": () => ({ operation_id: "sat-1", status: "queued" }),
    "GET /api/v1/sources/src-wiki/access-transition": () => ({ transition: current }),
  });

  const table = await screen.findByRole("table", { name: "Sources" });
  expect(await within(table).findByText("Access change failed")).toBeInTheDocument();
  const drawer = await openDrawer();
  expect(within(drawer).getByText("The change to Everyone in this workspace failed")).toBeInTheDocument();
  expect(within(drawer).getByText("Search index offline")).toBeInTheDocument();
  expect(within(drawer).getByRole("button", { name: "Revert to Only me" })).toBeInTheDocument();

  await userEvent.click(within(drawer).getByRole("button", { name: "Retry" }));

  expect(await within(drawer).findByText("Changing access to Everyone in this workspace")).toBeInTheDocument();
  expect(within(table).getByText("Changing access")).toBeInTheDocument();
  expect(calls.some((call) => call.method === "POST" && new URL(call.url).pathname.endsWith("/sat-1/retry"))).toBe(true);
  expect(calls.some((call) => new URL(call.url).pathname.endsWith("/sat-1/revert"))).toBe(false);
});
