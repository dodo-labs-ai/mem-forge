import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const sidebarSource = readFileSync("src/components/layout/Sidebar.tsx", "utf8");

assert.match(
  sidebarSource,
  /className="[^"]*sticky top-0[^"]*h-screen[^"]*lg:flex lg:flex-col"/,
  "desktop sidebar should stay fixed to the viewport so AccountFooter remains visible",
);

assert.match(
  sidebarSource,
  /className="[^"]*flex-1[^"]*overflow-y-auto[^"]*"/,
  "sidebar navigation should scroll instead of pushing AccountFooter below the viewport",
);
