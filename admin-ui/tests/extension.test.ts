import assert from "node:assert/strict";

import {
  __resetCloudExtensionForTests,
  getExtensionNavItems,
  getExtensionRoutes,
  getExtensionShell,
  getExtensionTopbarSlots,
  mountCloudExtension,
  RESERVED_OSS_ROUTE_SEGMENTS,
} from "../src/extension.js";

// Default state: nothing is mounted, every accessor returns the empty/no-op
// shape so the OSS shell renders unchanged.
__resetCloudExtensionForTests();
assert.deepEqual(getExtensionRoutes(), []);
assert.deepEqual(getExtensionNavItems(), []);
assert.deepEqual(getExtensionTopbarSlots(), []);
assert.equal(getExtensionShell(), null);

// The reserved-route set is the OSS-owned product-memory surface plus settings.
assert.deepEqual(
  [...RESERVED_OSS_ROUTE_SEGMENTS].sort(),
  ["entities", "memories", "projects", "review", "settings", "sources"],
);

// First mount succeeds and exposes the additive contributions.
const placeholderElement = null;
const registered = mountCloudExtension({
  id: "ext-addon",
  routes: [
    { path: "/extension/usage", element: placeholderElement },
    { path: "/extension/billing", element: placeholderElement },
  ],
  navItems: [
    { to: "/extension/usage", label: "Usage" },
    {
      to: "/extension/billing",
      label: "Billing",
      group: "Extension",
      visibleWhen: () => false,
    },
  ],
  topbarSlots: [
    { id: "principal-menu", render: () => null },
    { id: "principal-menu-2", render: () => null, placement: "before-account" },
  ],
  shell: { Wrapper: ({ children }) => children },
});
assert.equal(registered, true);
assert.equal(getExtensionRoutes().length, 2);
assert.equal(getExtensionNavItems().length, 2);
assert.equal(getExtensionNavItems()[1]!.visibleWhen?.(), false);
// `visibleWhen` is the render-time consumer contract: items default to visible,
// and a predicate returning `false` hides the item from the nav.
const visibleNavItems = getExtensionNavItems().filter((item) => item.visibleWhen?.() ?? true);
assert.equal(visibleNavItems.length, 1);
assert.equal(visibleNavItems[0]!.to, "/extension/usage");
assert.equal(getExtensionTopbarSlots().length, 2);
assert.notEqual(getExtensionShell(), null);

// Second mount is rejected: the shell stays deterministic for the lifetime of
// the bundle.
const reMount = mountCloudExtension({
  id: "ext-other",
  routes: [{ path: "/other", element: placeholderElement }],
});
assert.equal(reMount, false);
assert.equal(getExtensionRoutes().length, 2);
assert.equal(
  getExtensionRoutes().every((r) => r.path !== "/other"),
  true,
);

// Routes/nav items that try to shadow reserved OSS-owned product-memory
// surfaces are dropped at registration. This is the structural invariant the
// extension contract exists to protect.
__resetCloudExtensionForTests();
mountCloudExtension({
  id: "ext-reserved",
  routes: [
    { path: "/memories", element: placeholderElement }, // dropped
    { path: "/memories/:id/edit", element: placeholderElement }, // dropped (segment)
    { path: "/review", element: placeholderElement }, // dropped
    { path: "/sources", element: placeholderElement }, // dropped
    { path: "/projects/:key", element: placeholderElement }, // dropped (segment)
    { path: "/settings", element: placeholderElement }, // dropped
    { path: "/entities", element: placeholderElement }, // dropped
    { path: "/", element: placeholderElement }, // dropped (root)
    { path: "/extension/safe", element: placeholderElement }, // kept
  ],
  navItems: [
    { to: "/memories", label: "Hijacked Memories" }, // dropped
    { to: "/extension/safe", label: "Extension Safe" }, // kept
  ],
});
const safeRoutes = getExtensionRoutes();
assert.equal(safeRoutes.length, 1);
assert.equal(safeRoutes[0]!.path, "/extension/safe");

const safeNav = getExtensionNavItems();
assert.equal(safeNav.length, 1);
assert.equal(safeNav[0]!.to, "/extension/safe");

// Empty/missing id is rejected so the registration is intentional.
__resetCloudExtensionForTests();
const missingId = mountCloudExtension({ id: "" });
assert.equal(missingId, false);
assert.deepEqual(getExtensionRoutes(), []);

// Mounting with no contributions still succeeds (a shell-only wrapper is a
// valid extension shape).
__resetCloudExtensionForTests();
const wrapperOnly = mountCloudExtension({
  id: "ext-wrapper-only",
  shell: { Wrapper: ({ children }) => children },
});
assert.equal(wrapperOnly, true);
assert.deepEqual(getExtensionRoutes(), []);
assert.deepEqual(getExtensionNavItems(), []);
assert.deepEqual(getExtensionTopbarSlots(), []);
assert.notEqual(getExtensionShell(), null);

console.log("extension.test.ts: all assertions passed");
