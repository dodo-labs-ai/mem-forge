import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Outlet } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import type { ModelSettings } from "@/api";
import { errorResponse, type Route } from "@/test/fakeApi";
import { NO_MODELS, makeModelSettings, makeVectorIndex } from "@/test/modelSettingsFixtures";
import { renderRoutes } from "@/test/renderRoutes";
import type { AdminExtension } from "../extension/contract";
import { ExtensionProvider } from "../extension/ExtensionProvider";
import { ModelSetupNotice } from "./ModelSetupNotice";

const SETTINGS_ROUTE = "GET /api/v1/model-settings";

afterEach(() => {
  vi.unstubAllGlobals();
});

function Shell({ extension }: { extension?: AdminExtension }) {
  return (
    <ExtensionProvider extension={extension}>
      <ModelSetupNotice />
      <Outlet />
    </ExtensionProvider>
  );
}

function openApp(path: string, settings: ModelSettings | Route, extension?: AdminExtension) {
  return renderRoutes(
    [
      {
        element: <Shell extension={extension} />,
        children: [
          { path: "sources", element: <h1>Sources page</h1> },
          { path: "settings", element: <h1>Settings page</h1> },
          { path: "cloud/workspaces", element: <h1>Workspaces page</h1> },
        ],
      },
    ],
    { path, api: { [SETTINGS_ROUTE]: typeof settings === "function" ? settings : () => settings } },
  );
}

test("a fresh installation is told, on the page it lands on, to set up a generation model", async () => {
  openApp("/sources", makeModelSettings(NO_MODELS));

  const notice = await screen.findByRole("status");
  expect(notice).toHaveTextContent("No generation model is set up");
  expect(notice).toHaveTextContent("MemForge cannot write memories from your sources until you choose one.");

  await userEvent.click(within(notice).getByRole("link", { name: "Open Settings" }));

  expect(await screen.findByRole("heading", { name: "Settings page" })).toBeInTheDocument();
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});

test("an installation whose search index has to be rebuilt is told that vector search is off, with the way to Settings", async () => {
  openApp("/sources", makeModelSettings({ vector_index: makeVectorIndex({ status: "reindex_required", reindex_target: "active" }) }));

  const notice = await screen.findByRole("status");
  expect(notice).toHaveTextContent("Vector search is off");
  expect(notice).toHaveTextContent("Search ranks by matching words until the index is rebuilt in Settings.");

  await userEvent.click(within(notice).getByRole("link", { name: "Open Settings" }));

  expect(await screen.findByRole("heading", { name: "Settings page" })).toBeInTheDocument();
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});

test("nothing is shown while the generation model is ready and the search index fits", async () => {
  const { calls } = openApp("/sources", makeModelSettings());

  await waitFor(() => expect(calls).toHaveLength(1));
  expect(await screen.findByRole("heading", { name: "Sources page" })).toBeInTheDocument();
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});

test("nothing is shown when the settings cannot be read", async () => {
  const { calls } = openApp("/sources", () => errorResponse(500, "The database is locked."));

  await waitFor(() => expect(calls).toHaveLength(1));
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});

test("an extension page, which is not about the workspace, asks for nothing", async () => {
  const { calls } = openApp("/cloud/workspaces", makeModelSettings(NO_MODELS));

  expect(await screen.findByRole("heading", { name: "Workspaces page" })).toBeInTheDocument();
  expect(calls).toEqual([]);
});

test("the link opens the extension's settings page when it redirects Settings", async () => {
  openApp("/sources", makeModelSettings({ ...NO_MODELS, writable: false }), {
    id: "cloud",
    reservedRouteRedirects: [{ from: "settings", to: "/cloud/settings" }],
  });

  expect(await screen.findByRole("link", { name: "Open Settings" })).toHaveAttribute("href", "/cloud/settings");
});

test("a user who cannot open the extension's settings page gets the notice without a link", async () => {
  openApp("/sources", makeModelSettings({ ...NO_MODELS, writable: false }), {
    id: "cloud",
    reservedRouteRedirects: [{ from: "settings", to: "/cloud/settings", visibleWhen: () => false }],
  });

  expect(await screen.findByRole("status")).toHaveTextContent("until the deployment environment names one");
  expect(screen.queryByRole("link", { name: "Open Settings" })).not.toBeInTheDocument();
});
