export function currentWorkspaceId(): string | undefined {
  if (typeof window === "undefined") return undefined;
  const workspaceId = new URLSearchParams(window.location.search).get("workspace")?.trim();
  return workspaceId || undefined;
}

