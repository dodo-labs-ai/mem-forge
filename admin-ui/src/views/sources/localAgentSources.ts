import type { Source } from "../../api/types.js";

function isInternalGitHubSource(source: Source): boolean {
  return source.type === "github_repo" && String(source.config.connection_mode ?? "cloud_pull") === "local_push";
}

export function localAgentSyncOperation(source: Source): string | null {
  if (isInternalGitHubSource(source)) return "github_repo_sync";
  if (source.type === "local_markdown" && String(source.config.root ?? "").trim().length > 0) {
    return "local_markdown_sync";
  }
  if (source.type === "jira" && String(source.config.sync_mode ?? "cloud") === "local_agent") return "jira_sync";
  if (source.type === "teams") return "teams_sync";
  return null;
}

export function isLocalAgentBackedSource(source: Source): boolean {
  return localAgentSyncOperation(source) !== null;
}
