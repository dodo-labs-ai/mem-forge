import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLocation } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { makeOverview } from "@/test/evaluationFixtures";
import type { Route as FakeRoute } from "@/test/fakeApi";
import { renderRoutes } from "@/test/renderRoutes";
import { EvaluationPage } from "./EvaluationPage";

const OVERVIEW_PATH = "/api/v1/agent-evaluations/online-overview";

afterEach(() => {
  vi.unstubAllGlobals();
});

function LocationProbe() {
  const location = useLocation();
  return <output data-testid="location">{location.search}</output>;
}

function renderPage(initialEntry = "/evaluation") {
  const api: Record<string, FakeRoute> = {
    [`GET ${OVERVIEW_PATH}`]: () => makeOverview(),
    "GET /api/v1/genes": () => [
      { name: "confluence", display_name: "Confluence" },
      { name: "github_pages", display_name: "GitHub Pages" },
      { name: "github_repo", display_name: "GitHub Repository" },
      { name: "jira", display_name: "Jira" },
    ],
  };
  return renderRoutes(
    [
      {
        path: "/evaluation",
        element: (
          <>
            <EvaluationPage />
            <LocationProbe />
          </>
        ),
      },
    ],
    { path: initialEntry, api },
  );
}

function overviewQueries(calls: Request[]): Record<string, string>[] {
  return calls
    .map((request) => new URL(request.url))
    .filter((url) => url.pathname === OVERVIEW_PATH)
    .map((url) => Object.fromEntries(url.searchParams));
}

test("shows the metrics and failing patterns, and copies a content-free prompt for a case", async () => {
  const user = userEvent.setup();
  const { calls } = renderPage();

  expect(await screen.findByText("2 issue groups")).toBeInTheDocument();
  expect(screen.getByText("Patterns to check", { selector: "span" })).toBeInTheDocument();
  expect(overviewQueries(calls)[0]).toEqual({ days: "1", workspace_id: "local" });

  const group = screen.getByRole("button", { name: /Evidence reference validity/ });
  expect(group).toHaveTextContent("Affects 2 sources (Confluence and GitHub Pages).");
  await user.click(group);
  await user.click(screen.getByRole("button", { name: "Investigate with agent: case from Architecture docs" }));

  const preview = await screen.findByRole("region", { name: "Copied prompt" });
  expect(preview).toHaveTextContent("It carries IDs and versions only, no source text.");
  expect(preview).toHaveTextContent("Source: Architecture docs (src-docs, github_pages)");
  expect(await navigator.clipboard.readText()).toContain("Runtime event: evt_4c19d2e7a8b1c2d3e4a01f55");
});

test("choosing a source scopes the page to it, and the chip returns to all sources", async () => {
  const user = userEvent.setup();
  const { calls } = renderPage("/evaluation?workspace=mount_tai&criterion=evidence_localization");

  await user.click(await screen.findByRole("tab", { name: "Sources 3" }));
  await user.click(screen.getByRole("button", { name: "Open evaluation for mem-inception" }));

  expect(screen.getByTestId("location")).toHaveTextContent("?workspace=mount_tai&source_id=src-repo");
  await vi.waitFor(() => expect(overviewQueries(calls).at(-1)).toMatchObject({ source_id: "src-repo" }));
  expect(screen.getByRole("tab", { name: "Needs attention 2" })).toHaveAttribute("aria-selected", "true");

  await user.click(screen.getByRole("button", { name: "Show all sources" }));
  expect(screen.getByTestId("location")).toHaveTextContent("?workspace=mount_tai");
  expect(screen.getByTestId("location")).not.toHaveTextContent("source_id");
});

test("the status filter narrows the lists, and All checks keeps raw reason codes", async () => {
  const user = userEvent.setup();
  renderPage("/evaluation?label=pass");

  expect(await screen.findByRole("tab", { name: "Needs attention 0" })).toBeInTheDocument();
  expect(screen.getByText("No live-traffic failures in this window")).toBeInTheDocument();

  await user.click(screen.getByRole("tab", { name: "All checks" }));
  const checks = screen.getByRole("tabpanel");
  expect(within(checks).getByText("Extraction completion")).toBeInTheDocument();
  expect(within(checks).getByText("candidates_extracted")).toBeInTheDocument();
  expect(within(checks).queryByText("Evidence reference validity")).not.toBeInTheDocument();
});
