import { APP_BASE_PATH } from "../../src/app/basePath.ts";

/**
 * The path of an admin page under the build's base path. Every e2e URL is
 * built here, so the suite follows the app when its base path changes.
 *
 * `appPath("sources")` is `"/v2/sources"` while the app is served at `/v2/`;
 * `appPath()` is the app's root.
 */
export function appPath(page = ""): string {
  return APP_BASE_PATH + page.replace(/^\/+/, "");
}

/** Matches a page URL that ends with `appPath(page)`, for `expect(page).toHaveURL`. */
export function appUrl(page = ""): RegExp {
  return new RegExp(`${appPath(page).replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}$`);
}
