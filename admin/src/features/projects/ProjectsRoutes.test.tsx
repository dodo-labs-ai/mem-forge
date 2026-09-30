import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLocation } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import type { Route as FakeRoute } from "@/test/fakeApi";
import { renderRoutes } from "@/test/renderRoutes";
import { makeSource } from "@/test/sourceFixtures";
import { ProjectsRoutes } from "./ProjectsRoutes";

const PROJECTS = [
  { id: "p-shared", key: "SHARED", name: "Shared", kind: "shared", memory_count: 1441, created_at: null },
  { id: "p-unsorted", key: "UNSORTED", name: "Unsorted", kind: "normal", memory_count: 40, created_at: null },
  { id: "p-pay", key: "PAY", name: "Payroll", kind: "normal", memory_count: 708, created_at: "2026-07-02T00:00:00Z" },
];

afterEach(() => {
  vi.unstubAllGlobals();
});

function CurrentLocation() {
  const location = useLocation();
  return <output aria-label="Location">{`${location.pathname}${location.search}`}</output>;
}

const PAYROLL_DELETION_IMPACT = { memory_count: 712, source_count: 3 };
/** Longer than any test runs, so only a query that asks for fresh data refetches. */
const CACHED_FOR_THE_TEST_MS = 60_000;

function renderAt(path: string, routes: Record<string, FakeRoute> = {}, queryStaleMs?: number) {
  const api: Record<string, FakeRoute> = {
    "GET /api/v1/projects": () => ({ data: PROJECTS, can_manage: true }),
    "GET /api/v1/projects/p-pay/deletion-impact": () => PAYROLL_DELETION_IMPACT,
    "GET /api/v1/sources": () => ({
      data: [makeSource(), makeSource({ id: "src-jira", type: "jira", name: "Payroll Jira", memory_count: 611 })],
    }),
    "GET /api/v1/genes": () => [
      { name: "confluence", display_name: "Confluence" },
      { name: "jira", display_name: "Jira" },
    ],
    ...routes,
  };
  return renderRoutes(
    [
      {
        path: "projects/*",
        element: (
          <>
            <ProjectsRoutes />
            <CurrentLocation />
          </>
        ),
      },
      { path: "*", element: <CurrentLocation /> },
    ],
    { path: path, api, queryStaleMs },
  );
}

test("lists own projects with their sources and memories, and built-in projects without actions", async () => {
  renderAt("/projects");

  const table = await screen.findByRole("table", { name: "Projects" });
  const payroll = (await within(table).findByRole("link", { name: "Payroll" })).closest("tr")!;
  expect(within(payroll).getByText("PAY")).toBeInTheDocument();
  expect(await within(payroll).findByText("2")).toBeInTheDocument();
  expect(within(payroll).getByText("708")).toBeInTheDocument();
  expect(within(payroll).getByRole("button", { name: "Actions for Payroll" })).toBeInTheDocument();

  const shared = within(table).getByRole("link", { name: "Shared" }).closest("tr")!;
  expect(within(shared).getByText("Team-wide")).toBeInTheDocument();
  expect(within(shared).getByText("1,441")).toBeInTheDocument();
  expect(within(shared).queryByRole("button", { name: /Actions for/ })).not.toBeInTheDocument();
  expect(screen.queryByText("Set active")).not.toBeInTheDocument();
});

/** Opens the delete dialog from the list, passes its first step and types the project code. */
async function confirmDeleteFromList() {
  await userEvent.click(await screen.findByRole("button", { name: "Actions for Payroll" }));
  await userEvent.click(await screen.findByRole("menuitem", { name: "Delete…" }));
  const dialog = await screen.findByRole("dialog", { name: "Delete Payroll?" });
  await userEvent.click(await within(dialog).findByRole("button", { name: "Continue" }));
  await userEvent.type(within(dialog).getByLabelText(/to confirm/), "PAY");
  return dialog;
}

test("deleting from the list states the consequences, then asks for the project code", async () => {
  const { calls } = renderAt("/projects", {
    "DELETE /api/v1/projects/p-pay": () => ({
      id: "p-pay",
      rebucketed_count: 712,
      rebucketed_memory_ids: [],
      released_source_count: 3,
    }),
  });

  await userEvent.click(await screen.findByRole("button", { name: "Actions for Payroll" }));
  await userEvent.click(await screen.findByRole("menuitem", { name: "Delete…" }));

  const dialog = await screen.findByRole("dialog", { name: "Delete Payroll?" });
  expect(
    await within(dialog).findByText(
      "712 memories move to the Unsorted project. 3 sources will stop writing to this project. This cannot be undone.",
    ),
  ).toBeInTheDocument();
  expect(within(dialog).queryByRole("button", { name: "Delete project" })).not.toBeInTheDocument();

  await userEvent.click(within(dialog).getByRole("button", { name: "Continue" }));
  const deleteButton = within(dialog).getByRole("button", { name: "Delete project" });
  expect(deleteButton).toBeDisabled();
  await userEvent.type(within(dialog).getByLabelText(/to confirm/), "PA");
  expect(deleteButton).toBeDisabled();
  expect(calls.some((request) => request.method === "DELETE")).toBe(false);

  await userEvent.type(within(dialog).getByLabelText(/to confirm/), "Y");
  await userEvent.click(deleteButton);

  expect(
    await screen.findByText(
      "Project “Payroll” deleted. 712 memories moved to the Unsorted project. 3 sources stopped writing to this project.",
    ),
  ).toBeInTheDocument();
  // The released Sources' bindings changed, so the Sources list is read again.
  const deletedAt = calls.findIndex((request) => request.method === "DELETE");
  await vi.waitFor(() =>
    expect(calls.slice(deletedAt).some((request) => new URL(request.url).pathname === "/api/v1/sources")).toBe(true),
  );
});

test("reopening the delete dialog reads the counts again even while other data is cached", async () => {
  let sourceCount = 3;
  renderAt(
    "/projects",
    { "GET /api/v1/projects/p-pay/deletion-impact": () => ({ memory_count: 712, source_count: sourceCount }) },
    CACHED_FOR_THE_TEST_MS,
  );

  async function openDialog() {
    await userEvent.click(await screen.findByRole("button", { name: "Actions for Payroll" }));
    await userEvent.click(await screen.findByRole("menuitem", { name: "Delete…" }));
    return screen.findByRole("dialog", { name: "Delete Payroll?" });
  }

  const first = await openDialog();
  expect(await within(first).findByText(/3 sources will stop writing to this project\./)).toBeInTheDocument();
  await userEvent.click(within(first).getByRole("button", { name: "Cancel" }));
  await vi.waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument());

  sourceCount = 4;
  const second = await openDialog();
  expect(await within(second).findByText(/4 sources will stop writing to this project\./)).toBeInTheDocument();
});

test("the delete cannot go ahead until it knows what the delete changes", async () => {
  renderAt("/projects", {
    "GET /api/v1/projects/p-pay/deletion-impact": () => {
      throw new Error("Only a workspace admin can create, change, or delete projects.");
    },
  });

  await userEvent.click(await screen.findByRole("button", { name: "Actions for Payroll" }));
  await userEvent.click(await screen.findByRole("menuitem", { name: "Delete…" }));
  const dialog = await screen.findByRole("dialog", { name: "Delete Payroll?" });

  expect(await within(dialog).findByText(/^Could not check what the delete changes\./)).toBeInTheDocument();
  expect(within(dialog).getByRole("button", { name: "Continue" })).toBeDisabled();
});

test("a failed delete keeps the dialog open with the reason", async () => {
  renderAt("/projects", {
    "DELETE /api/v1/projects/p-pay": () => {
      throw new Error("Only a workspace admin can create, change, or delete projects.");
    },
  });

  const dialog = await confirmDeleteFromList();
  await userEvent.click(within(dialog).getByRole("button", { name: "Delete project" }));

  expect(await within(dialog).findByRole("alert")).toHaveTextContent("Only a workspace admin");
});

test("a new project refuses a built-in code and sends no kind", async () => {
  const { calls } = renderAt("/projects", {
    "POST /api/v1/projects": () => ({ id: "p-new", key: "PAYMENTS", name: "Payments", kind: "normal", created_at: null }),
  });

  await userEvent.click(await screen.findByRole("button", { name: "New project" }));
  const dialog = await screen.findByRole("dialog", { name: "New project" });
  expect(within(dialog).queryByText(/team-wide/i)).not.toBeInTheDocument();
  await userEvent.type(within(dialog).getByLabelText(/Name/), "Payments");
  await userEvent.click(within(dialog).getByRole("button", { name: "Advanced" }));
  await userEvent.type(within(dialog).getByLabelText("Code (optional)"), "shared");

  expect(await within(dialog).findByText(/That code is reserved/)).toBeInTheDocument();
  expect(within(dialog).getByRole("button", { name: "Create" })).toBeDisabled();

  await userEvent.clear(within(dialog).getByLabelText("Code (optional)"));
  await userEvent.click(within(dialog).getByRole("button", { name: "Create" }));

  await vi.waitFor(() => expect(calls.some((request) => request.method === "POST")).toBe(true));
  const create = calls.find((request) => request.method === "POST")!;
  expect(await create.clone().json()).toEqual({ name: "Payments" });
});

test("the detail page links to the project's memories and lists the sources sending them", async () => {
  renderAt("/projects/PAY");

  expect(await screen.findByRole("heading", { name: "Payroll" })).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "View memories" })).toHaveAttribute("href", "/memories?project=PAY");
  expect(screen.getByRole("button", { name: "Edit" })).toBeInTheDocument();

  const sources = screen.getByRole("region", { name: "Sources sending memories here" });
  const items = await within(sources).findAllByRole("listitem");
  expect(items.map((item) => item.textContent)).toEqual([
    expect.stringContaining("Payroll Jira"),
    expect.stringContaining("Payroll wiki"),
  ]);
  expect(within(items[0]!).getByText("Direct")).toBeInTheDocument();
  expect(within(items[0]!).getByText("611 memories")).toBeInTheDocument();
});

test("a built-in project cannot be edited or deleted", async () => {
  renderAt("/projects/SHARED");

  expect(await screen.findByRole("heading", { name: /Shared/ })).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "View memories" })).toHaveAttribute("href", "/memories?project=SHARED");
  expect(screen.queryByRole("button", { name: "Edit" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();
});

test("deleting on the detail page returns to the list with the notice", async () => {
  renderAt("/projects/PAY", {
    "DELETE /api/v1/projects/p-pay": () => ({
      id: "p-pay",
      rebucketed_count: 3,
      rebucketed_memory_ids: [],
      released_source_count: 0,
    }),
  });

  await userEvent.click(await screen.findByRole("button", { name: "Delete" }));
  const dialog = await screen.findByRole("dialog", { name: "Delete Payroll?" });
  await userEvent.click(await within(dialog).findByRole("button", { name: "Continue" }));
  await userEvent.type(within(dialog).getByLabelText(/to confirm/), "PAY");
  await userEvent.click(within(dialog).getByRole("button", { name: "Delete project" }));

  await vi.waitFor(() => expect(screen.getByRole("status", { name: "Location" })).toHaveTextContent("/projects"));
  expect(await screen.findByText("Project “Payroll” deleted. 3 memories moved to the Unsorted project.")).toBeInTheDocument();
});

test("an unknown code shows that the project was not found", async () => {
  renderAt("/projects/GONE");
  expect(await screen.findByText("Project not found")).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "Back to Projects" })).toHaveAttribute("href", "/projects");
});

test("a caller who cannot change projects sees no create, edit or delete", async () => {
  renderAt("/projects", { "GET /api/v1/projects": () => ({ data: PROJECTS, can_manage: false }) });

  const table = await screen.findByRole("table", { name: "Projects" });
  await userEvent.click(await within(table).findByRole("button", { name: "Actions for Payroll" }));
  expect(await screen.findByRole("menuitem", { name: "View memories" })).toBeInTheDocument();
  expect(screen.queryByRole("menuitem", { name: "Edit" })).not.toBeInTheDocument();
  expect(screen.queryByRole("menuitem", { name: "Delete…" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "New project" })).not.toBeInTheDocument();
});

test("the detail page hides edit and delete from a caller who cannot change projects", async () => {
  renderAt("/projects/PAY", { "GET /api/v1/projects": () => ({ data: PROJECTS, can_manage: false }) });

  expect(await screen.findByRole("heading", { name: "Payroll" })).toBeInTheDocument();
  expect(screen.getByRole("link", { name: "View memories" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Edit" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Delete" })).not.toBeInTheDocument();
});

test("without projects, a caller who cannot create one is told who can", async () => {
  renderAt("/projects", { "GET /api/v1/projects": () => ({ data: PROJECTS.slice(0, 2), can_manage: false }) });

  expect(await screen.findByText("A workspace admin creates projects for products or codebases.")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "New project" })).not.toBeInTheDocument();
});
