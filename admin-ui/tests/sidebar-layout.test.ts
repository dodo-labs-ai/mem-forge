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

assert.match(
  sidebarSource,
  /getExtensionAccountSurface/,
  "sidebar footer should allow an extension-owned account surface",
);

assert.doesNotMatch(
  sidebarSource,
  /function AccountFooter\(\)[\s\S]*?<button[\s\S]*?<\/button>/,
  "standalone AccountFooter should not render a fake clickable account button",
);

assert.doesNotMatch(
  sidebarSource,
  /Signed-in identity/,
  "standalone AccountFooter should not describe the static workspace card as a signed-in account",
);

const topbarSource = readFileSync("src/components/layout/Topbar.tsx", "utf8");

assert.match(
  topbarSource,
  /getExtensionAccountSurface/,
  "topbar should allow an extension-owned account surface",
);

assert.match(
  topbarSource,
  /function DefaultAccountBadge\(\)[\s\S]*aria-hidden="true"/,
  "standalone topbar account badge should be decorative when no extension owns it",
);
