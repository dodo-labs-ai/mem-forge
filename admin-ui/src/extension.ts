/**
 * Admin UI extension hook.
 *
 * The OSS admin UI is the single host shell. Downstream packagings can register
 * additive shell contributions through `mountCloudExtension` at module init,
 * and this module exposes those contributions to the shell at render time.
 *
 * Boundaries enforced here:
 *   - Exactly one extension can be mounted; second `mountCloudExtension` calls
 *     are rejected so the shell stays deterministic.
 *   - Routes that collide with reserved OSS-owned product-memory paths are
 *     dropped (they cannot shadow built-in views). The reserved set is the
 *     authoritative list of OSS-owned surfaces.
 *   - When no extension is mounted, the shell renders unchanged: the consumer
 *     APIs return empty arrays and a pass-through wrapper.
 *
 * The contract is intentionally narrow: routes, nav items, topbar slots, and
 * an optional shell wrapper/provider. Anything richer should land here as a
 * new typed slot, not as escape hatches.
 */
import type { ComponentType, ReactNode } from "react";
import type { RouteObject } from "react-router-dom";

/**
 * Routes that the OSS admin UI owns. Extensions cannot register routes whose
 * top-level segment matches one of these prefixes; doing so would shadow a
 * product-memory surface.
 */
const RESERVED_ROUTE_SEGMENTS = Object.freeze([
  "memories",
  "review",
  "entities",
  "sources",
  "projects",
  "settings",
] as const);

export type ReservedRouteSegment = (typeof RESERVED_ROUTE_SEGMENTS)[number];

export interface ExtensionNavItem {
  /** Absolute path (e.g. `/extension/usage`). Must not collide with reserved OSS routes. */
  to: string;
  /** Display label. */
  label: string;
  /** Optional `lucide-react`-shaped icon component. */
  icon?: ComponentType<{ className?: string }>;
  /** Optional group label; defaults to "Extension". */
  group?: string;
}

export interface ExtensionTopbarSlot {
  /** Stable id for React keys and replacement detection. */
  id: string;
  /** Render function for the slot. */
  render: () => ReactNode;
  /**
   * Where in the topbar this slot should land. Today only `before-account` is
   * defined; new positions land here as new literal members.
   */
  placement?: "before-account";
}

export interface ExtensionShellWrapper {
  /** Wraps the entire admin shell; useful for context providers (e.g. auth, capability flags). */
  Wrapper: ComponentType<{ children: ReactNode }>;
}

export interface CloudExtension {
  /** Stable id for telemetry and duplicate detection. */
  id: string;
  /** Additive routes; reserved OSS paths are dropped at registration time. */
  routes?: RouteObject[];
  /** Additive nav items rendered after the OSS nav groups. */
  navItems?: ExtensionNavItem[];
  /** Additive topbar slots. */
  topbarSlots?: ExtensionTopbarSlot[];
  /** Optional shell wrapper; rendered around the entire app. */
  shell?: ExtensionShellWrapper;
}

interface RegisteredExtension extends CloudExtension {
  routes: RouteObject[];
  navItems: ExtensionNavItem[];
  topbarSlots: ExtensionTopbarSlot[];
}

let mounted: RegisteredExtension | null = null;

function isReservedPath(path: string): boolean {
  if (typeof path !== "string" || path.length === 0) return false;
  const trimmed = path.replace(/^\/+/, "");
  if (trimmed.length === 0) return true; // root belongs to OSS redirect
  const head = trimmed.split("/", 1)[0];
  return (RESERVED_ROUTE_SEGMENTS as readonly string[]).includes(head);
}

function filterRoutes(routes: RouteObject[] | undefined): RouteObject[] {
  if (!routes || routes.length === 0) return [];
  return routes.filter((route) => {
    if (typeof route.path !== "string") return false;
    if (isReservedPath(route.path)) {
      // Reserved OSS surface: silently drop so a misconfigured extension
      // cannot shadow built-in views even at runtime.
      return false;
    }
    return true;
  });
}

function filterNavItems(items: ExtensionNavItem[] | undefined): ExtensionNavItem[] {
  if (!items || items.length === 0) return [];
  return items.filter((item) => !isReservedPath(item.to));
}

/**
 * Register an extension at module init. Subsequent calls are no-ops; the
 * first registration wins and the shell stays deterministic.
 *
 * Returns `true` if the extension was registered, `false` if a previous
 * registration already won.
 */
export function mountCloudExtension(extension: CloudExtension): boolean {
  if (mounted !== null) return false;
  if (!extension || typeof extension.id !== "string" || extension.id.length === 0) {
    return false;
  }
  mounted = {
    ...extension,
    routes: filterRoutes(extension.routes),
    navItems: filterNavItems(extension.navItems),
    topbarSlots: extension.topbarSlots ?? [],
  };
  return true;
}

/**
 * Test-only reset hook. Not part of the public surface; tests import this
 * directly from `@/extension` to start each case from a clean slate.
 */
export function __resetCloudExtensionForTests(): void {
  mounted = null;
}

export function getExtensionRoutes(): RouteObject[] {
  return mounted?.routes ?? [];
}

export function getExtensionNavItems(): ExtensionNavItem[] {
  return mounted?.navItems ?? [];
}

export function getExtensionTopbarSlots(): ExtensionTopbarSlot[] {
  return mounted?.topbarSlots ?? [];
}

export function getExtensionShell(): ExtensionShellWrapper | null {
  return mounted?.shell ?? null;
}

export const RESERVED_OSS_ROUTE_SEGMENTS = RESERVED_ROUTE_SEGMENTS;
