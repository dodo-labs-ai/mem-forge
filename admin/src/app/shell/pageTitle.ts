import { isProductPath, reservedRedirect, type AdminExtension } from "../extension/contract";
import { NAV_GROUPS, SETTINGS_NAV_ITEM } from "../navigation";

export const APP_NAME = "MemForge";
const TITLE_SEPARATOR = " · ";

interface TitledPath {
  path: string;
  label: string;
}

/** Pages this app renders, named by their navigation label. A redirected product page is named at its target. */
function titledPaths(extension: AdminExtension | undefined): TitledPath[] {
  const product = [...NAV_GROUPS.flatMap((group) => group.items), SETTINGS_NAV_ITEM].map((item) => ({
    path: reservedRedirect(extension, item.segment) ?? `/${item.segment}`,
    label: item.label,
  }));
  const extensionPages = (extension?.navItems ?? []).map((item) => ({ path: item.to, label: item.label }));
  return [...product, ...extensionPages];
}

function isAtOrBelow(pathname: string, path: string): boolean {
  return pathname === path || pathname.startsWith(`${path}/`);
}

/** The label of the navigation item whose path is the longest prefix of `pathname`. */
export function pageLabel(pathname: string, extension: AdminExtension | undefined): string | undefined {
  let match: TitledPath | undefined;
  for (const candidate of titledPaths(extension)) {
    if (isAtOrBelow(pathname, candidate.path) && candidate.path.length > (match?.path.length ?? -1)) match = candidate;
  }
  return match?.label;
}

/**
 * The browser tab title for `pathname`, leaving out the parts that are unknown:
 * `Page · Workspace · MemForge` on product pages and `Page · MemForge` on
 * extension pages. The workspace target governs product requests only, so the
 * workspace names nothing on an extension page.
 */
export function documentTitle(
  pathname: string,
  extension: AdminExtension | undefined,
  workspace: string | undefined,
): string {
  const parts = [pageLabel(pathname, extension), isProductPath(pathname) ? workspace : undefined, APP_NAME];
  return parts.filter((part) => part !== undefined && part !== "").join(TITLE_SEPARATOR);
}
