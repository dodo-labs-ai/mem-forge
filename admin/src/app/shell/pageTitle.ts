import { reservedRedirect, type AdminExtension } from "../extension/contract";
import { NAV_GROUPS, SETTINGS_NAV_ITEM } from "../navigation";

export const APP_NAME = "MemForge";
const TITLE_SEPARATOR = " · ";

interface TitledPath {
  path: string;
  label: string;
}

/** Pages this app renders, named by their navigation label. */
function titledPaths(extension: AdminExtension | undefined): TitledPath[] {
  const product = [...NAV_GROUPS.flatMap((group) => group.items), SETTINGS_NAV_ITEM].flatMap((item) => {
    const redirect = reservedRedirect(extension, item.segment);
    if (redirect !== undefined) return [{ path: redirect, label: item.label }];
    // A page still served by V1 is not a route here, so it has no title here either.
    return item.v1Path === undefined ? [{ path: `/${item.segment}`, label: item.label }] : [];
  });
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

/** `Page · Workspace · MemForge`, leaving out the parts that are unknown. */
export function documentTitle(page: string | undefined, workspace: string | undefined): string {
  return [page, workspace, APP_NAME].filter((part) => part !== undefined && part !== "").join(TITLE_SEPARATOR);
}
