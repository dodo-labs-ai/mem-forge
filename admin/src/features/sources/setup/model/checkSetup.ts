import { comparisonString, effectiveValue, type ConfigFormState, type ConfigSchema } from "./configForm";

/**
 * Types whose content step has no "Check setup" through preview-discovery: a
 * repository, whose folder tree shows the content; Teams, whose conversation
 * Gene cannot discover before it is bound; and a local folder, whose preview
 * runs as a local setup job on the device instead.
 */
const NO_DISCOVERY_PREVIEW = new Set(["github_repo", "teams", "local_markdown"]);

/** Jira's local sync reads through the person's browser on their device, which the server cannot preview. */
function isJiraLocalSync(sourceType: string, schema: ConfigSchema, state: ConfigFormState): boolean {
  const syncMode = schema.fields.find((field) => field.key === "sync_mode");
  return sourceType === "jira" && syncMode !== undefined && comparisonString(effectiveValue(syncMode, state)) === "local_agent";
}

/** Whether "Check setup" lists the first items the configuration would sync (Admin source setup §1.1). */
export function offersDiscoveryPreview(sourceType: string, schema: ConfigSchema, state: ConfigFormState): boolean {
  return !NO_DISCOVERY_PREVIEW.has(sourceType) && !isJiraLocalSync(sourceType, schema, state);
}
