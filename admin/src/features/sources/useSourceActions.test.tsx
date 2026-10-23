import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLocation } from "react-router-dom";
import { toast } from "sonner";
import { afterEach, expect, test, vi } from "vitest";
import type { Source } from "@/api";
import { renderRoutes } from "@/test/renderRoutes";
import { setupJobRoutes } from "@/test/setupJobs";
import { makeSource } from "@/test/sourceFixtures";
import { SOURCE_TYPE_FORMS, sourceTypeSchema } from "@/test/sourceTypeForms";
import type { SourceAttentionAction } from "./model/sourceAttention";
import { useSourceActions } from "./useSourceActions";

vi.mock("sonner", () => ({ toast: { success: vi.fn(), error: vi.fn() } }));

afterEach(() => {
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

const ACTIONS: SourceAttentionAction[] = [
  "start_local_sync",
  "sign_in",
  "configure",
  "configure_scope",
  "retry",
  "view_details",
  "sync_now",
  "assign_project",
];

function Harness({ source, action }: { source: Source; action: SourceAttentionAction }) {
  const actions = useSourceActions({ onViewDetails: () => undefined });
  const location = useLocation();
  const control = actions.attentionControl(action, source, "Act");
  return (
    <>
      <button type="button" disabled={!control.enabled} onClick={() => actions.runAttention(action, source)}>
        {control.label}
      </button>
      <output aria-label="Location">{`${location.pathname}${location.search}`}</output>
    </>
  );
}

function renderAction(source: Source, action: SourceAttentionAction, routes: Parameters<typeof renderRoutes>[1]["api"] = {}) {
  return renderRoutes([{ path: "*", element: <Harness source={source} action={action} /> }], {
    path: "/sources",
    api: {
      ...Object.fromEntries(
        Object.keys(SOURCE_TYPE_FORMS.schemas).map((type) => [`GET /api/v1/genes/${type}/config-schema`, () => sourceTypeSchema(type)]),
      ),
      "POST /api/v1/sources/src-wiki/sync": () => ({ status: "queued" }),
      ...routes,
    },
  });
}

const TEAMS = makeSource({
  id: "src-teams",
  type: "teams",
  name: "Team chats",
  config: { region: "amer", conversation_ids: ["19:a@thread.v2"] },
  connection_status: { state: "action_required", reason: "authentication" },
});

test("Teams sign-in renews in place", async () => {
  const created: Parameters<typeof setupJobRoutes>[1] = [];
  const { router } = renderAction(
    TEAMS,
    "sign_in",
    setupJobRoutes({ teams_auth: { status: "succeeded", result: { operation: "teams_auth", region: "amer", authenticated: true } } }, created),
  );

  await userEvent.click(screen.getByRole("button", { name: "Act" }));

  await waitFor(() => expect(toast.success).toHaveBeenCalledWith("Signed in to Teams for Team chats"));
  expect(created).toEqual([{ source_id: "src-teams", source_type: "teams", operation: "teams_auth", payload: { region: "amer" } }]);
  expect(router.state.location.pathname).toBe("/sources");
});

test("a stored PAT Jira source without auth_mode opens the connection step", async () => {
  const jira = makeSource({ id: "src-jira", type: "jira", config: { base_url: "https://jira.example.com", pat_configured: true } });
  const { calls } = renderAction(jira, "sign_in");

  await userEvent.click(screen.getByRole("button", { name: "Act" }));

  expect(screen.getByRole("status", { name: "Location" })).toHaveTextContent("/sources/src-jira/settings?step=connection");
  expect(calls.some((request) => request.method === "POST")).toBe(false);
});

test("a Confluence sign-in opens the connection step, where a token is entered", async () => {
  renderAction(makeSource({ connection_status: { state: "action_required", reason: "authentication" } }), "sign_in");

  await userEvent.click(screen.getByRole("button", { name: "Act" }));

  expect(screen.getByRole("status", { name: "Location" })).toHaveTextContent("/sources/src-wiki/settings?step=connection");
});

/** Records each time the page tries to leave the app, which jsdom reports as a navigation it cannot carry out. */
function watchPageNavigation(): string[] {
  const attempts: string[] = [];
  const dom = (globalThis as { jsdom?: { virtualConsole: { on: (event: string, listener: (error: Error) => void) => void } } }).jsdom;
  dom?.virtualConsole.on("jsdomError", (error) => {
    if (error.message.includes("navigation")) attempts.push(error.message);
  });
  return attempts;
}

test.each(ACTIONS)("no action navigates outside the app (%s)", async (action) => {
  vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText: vi.fn().mockResolvedValue(undefined) } });
  const attempts = watchPageNavigation();
  const { router } = renderAction(makeSource(), action);

  await userEvent.click(screen.getByRole("button", { name: "Act" }));

  await waitFor(() => expect(router.state.location.pathname.startsWith("/sources")).toBe(true));
  expect(attempts).toEqual([]);
});
