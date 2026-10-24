import { screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useLocation } from "react-router-dom";
import { afterEach, expect, test, vi } from "vitest";
import { renderRoutes } from "@/test/renderRoutes";
import { SOURCE_TYPE_FORMS } from "@/test/sourceTypeForms";
import { AddSourcePage } from "./AddSourcePage";

afterEach(() => {
  vi.unstubAllGlobals();
});

function CurrentLocation() {
  const location = useLocation();
  return <output aria-label="Location">{location.pathname}</output>;
}

function renderPicker(localSync: "online" | "offline" = "online") {
  return renderRoutes(
    [
      { path: "/sources/new", element: <AddSourcePage /> },
      { path: "*", element: <CurrentLocation /> },
    ],
    {
      path: "/sources/new",
      api: {
        "GET /api/v1/genes": () => SOURCE_TYPE_FORMS.genes,
        "GET /api/cloud/local-agent/status": () => ({ status: localSync, last_seen_at: null, checked_at: "", stale_after_seconds: 60 }),
      },
    },
  );
}

test("lists configurable types and hides managed ones", async () => {
  renderPicker();

  const types = await screen.findByRole("list", { name: "Source types" });
  const names = within(types)
    .getAllByRole("link")
    .map((link) => link.textContent);
  expect(names.some((name) => name?.includes("Confluence"))).toBe(true);
  expect(names.some((name) => name?.includes("Microsoft Teams"))).toBe(true);
  expect(within(types).queryByText("Agent Session")).not.toBeInTheDocument();

  await userEvent.click(within(types).getByRole("link", { name: /GitHub Pages/ }));
  expect(await screen.findByRole("status", { name: "Location" })).toHaveTextContent("/sources/new/github_pages");
});

test("a local-only type shows the Local sync badge", async () => {
  renderPicker("offline");

  const types = await screen.findByRole("list", { name: "Source types" });
  const local = within(types).getByRole("link", { name: /Local Repository/ });
  expect(within(local).getByText("Local sync")).toBeInTheDocument();
  expect(await within(local).findByText("Local sync offline")).toBeInTheDocument();
  const confluence = within(types).getByRole("link", { name: /Confluence/ });
  expect(within(confluence).getByText("Cloud")).toBeInTheDocument();
  expect(within(confluence).queryByText(/Local sync/)).not.toBeInTheDocument();
  expect(within(within(types).getByRole("link", { name: /Jira/ })).getByText("Cloud or local sync")).toBeInTheDocument();
});

test("a coding agent shows the commands that install its memory-tools plugin", async () => {
  const writeText = vi.fn().mockResolvedValue(undefined);
  vi.stubGlobal("navigator", { ...navigator, clipboard: { writeText } });
  renderPicker();

  const agents = await screen.findByRole("list", { name: "Coding agents" });
  await userEvent.click(within(agents).getByRole("button", { name: /Claude Code/ }));

  const dialog = await screen.findByRole("dialog", { name: "Set up Claude Code" });
  expect(within(dialog).getByText(/records nothing on its own/)).toBeInTheDocument();
  expect(within(dialog).queryByText(/session windows/)).not.toBeInTheDocument();
  expect(within(dialog).getByText("/plugin marketplace add dodo-labs-ai/mem-forge")).toBeInTheDocument();
  expect(within(dialog).getByText("/plugin install memory@memforge")).toBeInTheDocument();
  await userEvent.click(within(dialog).getByRole("button", { name: "Copy /plugin install memory@memforge" }));
  expect(writeText).toHaveBeenCalledWith("/plugin install memory@memforge");
});
