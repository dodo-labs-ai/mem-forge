import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLocation } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { entityRow, makeEntityDetail } from "@/test/entityFixtures";
import type { Route as FakeRoute } from "@/test/fakeApi";
import { renderRoutes } from "@/test/renderRoutes";
import { EntitiesRoutes } from "./EntitiesRoutes";

afterEach(() => {
  vi.unstubAllGlobals();
});

function CurrentLocation() {
  const location = useLocation();
  return <output aria-label="Location">{`${location.pathname}${location.search}`}</output>;
}

const CUT_OFF = makeEntityDetail();
const LEDGER = makeEntityDetail({ id: 9, canonical_name: "ledger", display_name: "ledger", aliases: [] });

function renderAt(path: string, routes: Record<string, FakeRoute> = {}) {
  return renderRoutes(
    [
      {
        path: "entities/*",
        element: (
          <>
            <EntitiesRoutes />
            <CurrentLocation />
          </>
        ),
      },
    ],
    {
      path,
      api: {
        "GET /api/v1/entities": () => ({ data: [entityRow(CUT_OFF), entityRow(LEDGER)], total: 2 }),
        [`GET /api/v1/entities/${CUT_OFF.id}`]: () => CUT_OFF,
        ...routes,
      },
    },
  );
}

function listRequests(calls: Request[]): URLSearchParams[] {
  return calls
    .filter((request) => new URL(request.url).pathname === "/api/v1/entities")
    .map((request) => new URL(request.url).searchParams);
}

test("lists entities with a canonical name that differs and opens one", async () => {
  renderAt("/entities");

  const table = await screen.findByRole("table", { name: "Entities" });
  const cutOff = (await within(table).findByRole("link", { name: "Cut-off" })).closest("tr")!;
  expect(within(cutOff).getByText("cut-off")).toBeInTheDocument();
  const ledger = within(table).getByRole("link", { name: "ledger" }).closest("tr")!;
  expect(within(ledger).getAllByText("ledger")).toHaveLength(1);
  expect(screen.getByText("2 entities")).toBeInTheDocument();

  await userEvent.click(cutOff);
  expect(await screen.findByRole("heading", { level: 1, name: "Cut-off" })).toBeInTheDocument();
  expect(screen.getByRole("status", { name: "Location" })).toHaveTextContent("/entities/7");
});

test("reads the search from the URL, searches as the user types and pages through the results", async () => {
  const { calls } = renderAt("/entities?search=Cut", {
    "GET /api/v1/entities": (request) => {
      const offset = Number(new URL(request.url).searchParams.get("offset"));
      return { data: offset === 0 ? [entityRow(CUT_OFF)] : [entityRow(LEDGER)], total: 60 };
    },
  });

  const search = await screen.findByRole("searchbox", { name: "Search entities" });
  expect(search).toHaveValue("Cut");
  await screen.findByRole("link", { name: "Cut-off" });
  expect(listRequests(calls)[0]!.get("search")).toBe("Cut");

  await userEvent.click(screen.getByRole("button", { name: "Next" }));
  await screen.findByRole("link", { name: "ledger" });
  expect(listRequests(calls).at(-1)!.get("offset")).toBe("50");
  expect(screen.getByRole("status", { name: "Location" })).toHaveTextContent("/entities?search=Cut&page=2");

  await userEvent.clear(search);
  await userEvent.type(search, "led");
  await waitFor(() => expect(listRequests(calls).at(-1)!.get("search")).toBe("led"));
  expect(listRequests(calls).at(-1)!.get("offset")).toBe("0");
  expect(screen.getByRole("status", { name: "Location" })).toHaveTextContent("/entities?search=led");
});

test("a search with no match says what search reads", async () => {
  renderAt("/entities?search=zzz", { "GET /api/v1/entities": () => ({ data: [], total: 0 }) });

  expect(await screen.findByText("No entities match")).toBeInTheDocument();
  expect(screen.getByText("Search matches display and canonical names, not aliases.")).toBeInTheDocument();
});
