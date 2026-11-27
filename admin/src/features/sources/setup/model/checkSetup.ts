import { comparisonString, effectiveValue, type ConfigFormState, type ConfigSchema } from "./configForm";

/**
 * Types whose content step has no "Check setup": a repository, whose folder
 * tree shows the content, and Teams, whose conversation Gene cannot discover
 * before it is bound.
 */
const NO_CHECK_SETUP = new Set(["github_repo", "teams"]);

/** Jira's local sync reads through the person's browser on their device, which the server cannot preview. */
function isJiraLocalSync(sourceType: string, schema: ConfigSchema, state: ConfigFormState): boolean {
  const syncMode = schema.fields.find((field) => field.key === "sync_mode");
  return sourceType === "jira" && syncMode !== undefined && comparisonString(effectiveValue(syncMode, state)) === "local_agent";
}

/** Whether "Check setup" lists, through the server's preview-discovery, the first items the configuration would sync (Admin source setup §1.1). */
export function offersCheckSetup(sourceType: string, schema: ConfigSchema, state: ConfigFormState): boolean {
  return !NO_CHECK_SETUP.has(sourceType) && !isJiraLocalSync(sourceType, schema, state);
}
