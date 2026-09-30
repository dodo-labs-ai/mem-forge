import { expect, test } from "vitest";
import type { AdminExtension } from "../extension/contract";
import { documentTitle, pageLabel } from "./pageTitle";

const CLOUD: AdminExtension = {
  id: "cloud",
  navItems: [
    { to: "/cloud/workspaces", label: "Workspaces", group: "Cloud admin" },
    { to: "/cloud/workspaces/archived", label: "Archived workspaces", group: "Cloud admin" },
  ],
  reservedRouteRedirects: [{ from: "settings", to: "/cloud/settings" }],
};

test("names product pages built here, including their subpaths", () => {
  expect(pageLabel("/sources", undefined)).toBe("Sources");
  expect(pageLabel("/sources/src-wiki", undefined)).toBe("Sources");
  expect(pageLabel("/sourcesx", undefined)).toBeUndefined();
});

test("does not name pages that V1 still serves", () => {
  expect(pageLabel("/memories", undefined)).toBeUndefined();
  expect(pageLabel("/settings", undefined)).toBeUndefined();
});

test("names a redirected product page at the extension path it opens", () => {
  expect(pageLabel("/cloud/settings", CLOUD)).toBe("Settings");
});

test("names extension pages by the most specific navigation item", () => {
  expect(pageLabel("/cloud/workspaces/ws-a", CLOUD)).toBe("Workspaces");
  expect(pageLabel("/cloud/workspaces/archived", CLOUD)).toBe("Archived workspaces");
});

test("joins the known parts of the title", () => {
  expect(documentTitle("Sources", "Payroll")).toBe("Sources · Payroll · MemForge");
  expect(documentTitle("Sources", undefined)).toBe("Sources · MemForge");
  expect(documentTitle(undefined, undefined)).toBe("MemForge");
});
