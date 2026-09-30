/**
 * The V1 Sources page, where setup, configuration and sign-in still happen
 * until this app has its own dialogs for them. Remove with those dialogs
 * (ADR 0044).
 */
export const V1_SOURCES_PATH = "/sources";

/** Poll interval while any source is syncing or changing access. */
export const ACTIVE_POLL_MS = 2_000;
