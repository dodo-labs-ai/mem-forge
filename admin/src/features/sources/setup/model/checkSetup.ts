/**
 * Types whose content step has no "Check setup": a repository, whose folder
 * tree shows the content, and Teams, whose conversation Gene cannot discover
 * before it is bound.
 */
const NO_CHECK_SETUP = new Set(["github_repo", "teams"]);

/** Whether "Check setup" lists, through the server's preview-discovery, the first items the configuration would sync (Admin source setup §1.1). */
export function offersCheckSetup(sourceType: string): boolean {
  return !NO_CHECK_SETUP.has(sourceType);
}
