/**
 * Where API requests go. The standalone build talks to the local server; the
 * Cloud extension points the target at the workspace the user selected, or at
 * nothing while no workspace is selected.
 */
export interface WorkspaceTarget {
  /** Replaces the `/api/v1` prefix of resource routes. */
  resourceBaseUrl: string;
  /** Replaces the `/api/cloud/local-agent` prefix of local sync routes. */
  localAgentBaseUrl: string;
  /** Sent as the `workspace_id` query parameter on resource routes. */
  workspaceId: string | null;
}

export const RESOURCE_PREFIX = "/api/v1";
export const LOCAL_AGENT_PREFIX = "/api/cloud/local-agent";

export const STANDALONE_TARGET: WorkspaceTarget = Object.freeze({
  resourceBaseUrl: RESOURCE_PREFIX,
  localAgentBaseUrl: LOCAL_AGENT_PREFIX,
  workspaceId: "local",
});

export interface WorkspaceController {
  current(): WorkspaceTarget | null;
  setTarget(target: WorkspaceTarget | null): void;
  subscribe(listener: () => void): () => void;
}

function trimTrailingSlash(value: string): string {
  return value.replace(/\/+$/, "");
}

function sameTarget(a: WorkspaceTarget | null, b: WorkspaceTarget | null): boolean {
  if (a === null || b === null) return a === b;
  return (
    a.resourceBaseUrl === b.resourceBaseUrl &&
    a.localAgentBaseUrl === b.localAgentBaseUrl &&
    a.workspaceId === b.workspaceId
  );
}

/**
 * Creates the workspace target holder. `onChange` runs after every real
 * change, so the caller can drop data cached for the previous workspace.
 */
export function createWorkspaceController(
  onChange: () => void,
  initial: WorkspaceTarget | null = STANDALONE_TARGET,
): WorkspaceController {
  let target = initial;
  const listeners = new Set<() => void>();
  return {
    current: () => target,
    setTarget(next) {
      const normalized = next && {
        resourceBaseUrl: trimTrailingSlash(next.resourceBaseUrl),
        localAgentBaseUrl: trimTrailingSlash(next.localAgentBaseUrl),
        workspaceId: next.workspaceId,
      };
      if (sameTarget(target, normalized)) return;
      target = normalized;
      onChange();
      listeners.forEach((listener) => listener());
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
