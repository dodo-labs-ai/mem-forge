import { comparisonString, effectiveValue, type ConfigFormState, type ConfigSchema } from "./configForm";

/**
 * Types whose content step has no "Check setup": a repository, whose folder
 * tree shows the content, and Teams, whose conversation Gene cannot discover
 * before it is bound.
 */
const NO_CHECK_SETUP = new Set(["github_repo", "teams"]);

/** A local folder is listed by the local sync daemon on the person's computer, not by the server. */
const LOCAL_FOLDER_TYPE = "local_markdown";

/** How "Check setup" lists the first items: through the server's preview-discovery, or through local sync's folder preview. */
export type CheckSetupKind = "discovery" | "local_folder";

/** Jira's local sync reads through the person's browser on their device, which the server cannot preview. */
function isJiraLocalSync(sourceType: string, schema: ConfigSchema, state: ConfigFormState): boolean {
  const syncMode = schema.fields.find((field) => field.key === "sync_mode");
  return sourceType === "jira" && syncMode !== undefined && comparisonString(effectiveValue(syncMode, state)) === "local_agent";
}

/** How "Check setup" lists the first items the configuration would sync, or null when it has none (Admin source setup §1.1). */
export function checkSetupKind(sourceType: string, schema: ConfigSchema, state: ConfigFormState): CheckSetupKind | null {
  if (NO_CHECK_SETUP.has(sourceType) || isJiraLocalSync(sourceType, schema, state)) return null;
  return sourceType === LOCAL_FOLDER_TYPE ? "local_folder" : "discovery";
}
