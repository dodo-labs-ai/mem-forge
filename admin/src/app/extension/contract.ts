import type { ComponentType, ReactNode } from "react";
import type { RouteObject } from "react-router-dom";
import type { WorkspaceController } from "@/api/workspace";

/**
 * Top-level route segments the admin UI owns. An extension cannot register
 * routes under them; it can only redirect one of them to its own page with
 * `reservedRouteRedirects`.
 */
export const RESERVED_ROUTE_SEGMENTS = [
  "memories",
  "review",
  "evaluation",
  "entities",
  "sources",
  "projects",
  "settings",
] as const;

export type ReservedRouteSegment = (typeof RESERVED_ROUTE_SEGMENTS)[number];

export interface ExtensionNavItem {
  /** Path inside the app, such as `/cloud/workspaces`. */
  to: string;
  label: string;
  icon?: ComponentType<{ className?: string }>;
  /** Heading of the sidebar group; items with the same group render together. */
  group: string;
  /** Hides the item for users who cannot use it. The server still enforces access. */
  visibleWhen?: () => boolean;
}

export interface ExtensionTopbarSlot {
  id: string;
  /** `start` sits after the brand (workspace switcher); `end` holds the account menu. */
  placement: "start" | "end";
  render: () => ReactNode;
}

export interface ExtensionReservedRouteRedirect {
  from: ReservedRouteSegment;
  /** Absolute path owned by the extension. */
  to: string;
  /** Hides the product navigation item for users who cannot open `to`. The server still enforces access. */
  visibleWhen?: () => boolean;
}

/**
 * What a packaging such as MemForge Cloud adds to the admin UI. Everything is
 * additive: extension routes cannot shadow product pages, and extension
 * navigation renders at the bottom of the sidebar below a divider.
 */
export interface AdminExtension {
  id: string;
  /** Routes relative to the app root, such as `cloud/workspaces`. */
  routes?: RouteObject[];
  navItems?: ExtensionNavItem[];
  topbarSlots?: ExtensionTopbarSlot[];
  /** Wraps the whole app, for example to provide sign-in state and select the workspace. */
  Wrapper?: ComponentType<{ children: ReactNode; workspace: WorkspaceController }>;
  reservedRouteRedirects?: ExtensionReservedRouteRedirect[];
}

function topSegment(path: string | undefined): string {
  return (path ?? "").replace(/^\/+/, "").split("/")[0] ?? "";
}

export function isReservedSegment(segment: string): segment is ReservedRouteSegment {
  return (RESERVED_ROUTE_SEGMENTS as readonly string[]).includes(segment);
}

/** Drops extension routes that would shadow a product page. */
export function extensionRoutes(extension: AdminExtension | undefined): RouteObject[] {
  return (extension?.routes ?? []).filter((route) => !isReservedSegment(topSegment(route.path)));
}

export function findReservedRedirect(
  extension: AdminExtension | undefined,
  segment: ReservedRouteSegment,
): ExtensionReservedRouteRedirect | undefined {
  return extension?.reservedRouteRedirects?.find((redirect) => redirect.from === segment);
}

export function reservedRedirect(
  extension: AdminExtension | undefined,
  segment: ReservedRouteSegment,
): string | undefined {
  return findReservedRedirect(extension, segment)?.to;
}
