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

test("names every product page, including its subpaths", () => {
  expect(pageLabel("/sources", undefined)).toBe("Sources");
  expect(pageLabel("/sources/src-wiki", undefined)).toBe("Sources");
  expect(pageLabel("/sourcesx", undefined)).toBeUndefined();
  expect(pageLabel("/settings", undefined)).toBe("Settings");
  expect(pageLabel("/memories/mem-1", undefined)).toBe("Memories");
  expect(pageLabel("/review/rev-1", undefined)).toBe("Review");
  expect(pageLabel("/evaluation", undefined)).toBe("Evaluation");
  expect(pageLabel("/projects/PAY", undefined)).toBe("Projects");
});

test("names a redirected product page at the extension path it opens", () => {
  expect(pageLabel("/cloud/settings", CLOUD)).toBe("Settings");
});

test("names extension pages by the most specific navigation item", () => {
  expect(pageLabel("/cloud/workspaces/ws-a", CLOUD)).toBe("Workspaces");
  expect(pageLabel("/cloud/workspaces/archived", CLOUD)).toBe("Archived workspaces");
});

test("names the workspace on product pages", () => {
  expect(documentTitle("/sources", undefined, "Payroll")).toBe("Sources · Payroll · MemForge");
  expect(documentTitle("/sources/src-wiki", CLOUD, "Payroll")).toBe("Sources · Payroll · MemForge");
  expect(documentTitle("/sources", undefined, undefined)).toBe("Sources · MemForge");
});

test("leaves the workspace out on extension pages", () => {
  expect(documentTitle("/cloud/workspaces", CLOUD, "Payroll")).toBe("Workspaces · MemForge");
  expect(documentTitle("/cloud/settings", CLOUD, "Payroll")).toBe("Settings · MemForge");
  expect(documentTitle("/cloud/account", CLOUD, "Payroll")).toBe("MemForge");
});

test("falls back to the app name on pages it cannot name", () => {
  expect(documentTitle("/missing", undefined, "Payroll")).toBe("MemForge");
  expect(documentTitle("/missing", undefined, undefined)).toBe("MemForge");
});
