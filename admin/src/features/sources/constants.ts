/**
 * The V1 Sources page, where signing in to a source other than a Jira
 * browser session still happens until this app has its own sign-in panels.
 *
 * @deprecated admin-ui-v1: V2's own sign-in panels replace this link.
 */
export const V1_SOURCES_PATH = "/sources";

/** Poll interval while any source is syncing or changing access. */
export const ACTIVE_POLL_MS = 2_000;
