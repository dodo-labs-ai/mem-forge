export function currentWorkspaceId(): string | undefined {
  if (typeof window === "undefined") return undefined;
  const workspaceId = new URLSearchParams(window.location.search).get("workspace")?.trim();
  return workspaceId || undefined;
}

export function requireCurrentWorkspaceId(): string {
  const workspaceId = currentWorkspaceId();
  if (!workspaceId) {
    throw new Error("Select a workspace before starting local sync.");
  }
  return workspaceId;
}
