import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithApi } from "@/test/renderWithApi";
import { setupJobRoutes, type FakeSetupJob } from "@/test/setupJobs";
import TeamsSessionPanel from "./TeamsSessionPanel";

afterEach(() => {
  vi.unstubAllGlobals();
});

function check(authenticated: boolean): FakeSetupJob {
  return {
    status: authenticated ? "succeeded" : "failed",
    result: {
      operation: "teams_auth_check",
      region: "amer",
      authenticated,
      expires_in_minutes: authenticated ? 50 : null,
      error: authenticated ? null : "No Teams session found: tokens missing",
    },
  };
}

test("an active session says so and can be checked again", async () => {
  const created: Parameters<typeof setupJobRoutes>[1] = [];
  renderWithApi(<TeamsSessionPanel sourceType="teams" config={{ region: "amer" }} />, setupJobRoutes({ teams_auth_check: check(true) }, created));

  expect(await screen.findByText("Teams session active")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Check again" })).toBeEnabled();
  expect(created).toEqual([{ source_id: "", source_type: "teams", operation: "teams_auth_check", payload: { region: "amer" } }]);
});

test("without a session Connect signs in on the computer, then checks again", async () => {
  const created: Parameters<typeof setupJobRoutes>[1] = [];
  renderWithApi(
    <TeamsSessionPanel sourceType="teams" sourceId="src-teams" config={{ region: "amer" }} />,
    setupJobRoutes(
      {
        teams_auth_check: (reads) => check(reads > 1),
        teams_auth: { status: "succeeded", result: { operation: "teams_auth", region: "amer", authenticated: true, token_count: 3, error: null } },
      },
      created,
    ),
  );

  expect(await screen.findByText("Connect Microsoft Teams")).toBeInTheDocument();
  expect(screen.getByText("No active Teams session was found. Sign in to Teams in Chrome on your computer, then connect.")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Connect" }));

  expect(await screen.findByText("Teams session active")).toBeInTheDocument();
  expect(created.map((job) => job.operation)).toEqual(["teams_auth_check", "teams_auth", "teams_auth_check"]);
  expect(created[1]).toMatchObject({ source_id: "src-teams", payload: { region: "amer" } });
});
