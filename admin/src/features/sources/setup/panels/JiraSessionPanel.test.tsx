import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithApi } from "@/test/renderWithApi";
import { setupJobRoutes } from "@/test/setupJobs";
import JiraSessionPanel from "./JiraSessionPanel";

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

afterEach(() => {
  vi.unstubAllGlobals();
});

const BASE_URL = "https://jira.example.com";
const ACTIVE = { origin: BASE_URL, status: "active", principal_name: "Ada Lovelace", browser: "chrome", principal_changed: false, provider: "jira", sources_reset: [] };

test("shows the stored session and the command that captures one", async () => {
  const writeText = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
  const { calls } = renderWithApi(<JiraSessionPanel sourceType="jira" config={{ base_url: BASE_URL }} />, {
    "GET /api/v1/auth/jira-session": () => ACTIVE,
  });

  expect(await screen.findByText("Active")).toBeInTheDocument();
  expect(screen.getByText("Signed in as Ada Lovelace through chrome.")).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Sign in again" })).not.toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Copy" }));
  expect(writeText).toHaveBeenCalledWith(`memforge adapter auth jira refresh --base-url ${BASE_URL}`);
  expect(new URL(calls[0]!.url).searchParams.get("base_url")).toBe(BASE_URL);
});

test("an existing source signs in again through local sync, then reads the session again", async () => {
  let reads = 0;
  const created: Parameters<typeof setupJobRoutes>[1] = [];
  renderWithApi(<JiraSessionPanel sourceType="jira" sourceId="src-jira" config={{ base_url: BASE_URL }} />, {
    "GET /api/v1/auth/jira-session": () => (++reads === 1 ? { ...ACTIVE, status: "expired" } : ACTIVE),
    ...setupJobRoutes({ jira_auth: { status: "succeeded", result: { operation: "jira_auth", authenticated: true, origin: BASE_URL } } }, created),
  });

  expect(await screen.findByText("Expired")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Sign in again" }));

  expect(await screen.findByText("Active")).toBeInTheDocument();
  expect(created).toEqual([
    { source_id: "src-jira", source_type: "jira", operation: "jira_auth", payload: { base_url: BASE_URL, auth_mode: "browser_cookie" } },
  ]);
});
