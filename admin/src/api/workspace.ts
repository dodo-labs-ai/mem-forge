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
  /** Name of the workspace, shown in the browser tab title. */
  label?: string;
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

function normalize(target: WorkspaceTarget | null): WorkspaceTarget | null {
  return (
    target && {
      ...target,
      resourceBaseUrl: trimTrailingSlash(target.resourceBaseUrl),
      localAgentBaseUrl: trimTrailingSlash(target.localAgentBaseUrl),
    }
  );
}

/** Whether both targets send requests to the same workspace. */
function sameRequests(a: WorkspaceTarget | null, b: WorkspaceTarget | null): boolean {
  if (a === null || b === null) return a === b;
  return (
    a.resourceBaseUrl === b.resourceBaseUrl &&
    a.localAgentBaseUrl === b.localAgentBaseUrl &&
    a.workspaceId === b.workspaceId
  );
}

/**
 * Creates the workspace target holder. `onChange` runs whenever requests go to
 * a different workspace, so the caller can drop data cached for the previous
 * one; a new label alone only notifies subscribers.
 */
export function createWorkspaceController(
  onChange: () => void,
  initial: WorkspaceTarget | null = STANDALONE_TARGET,
): WorkspaceController {
  let target = normalize(initial);
  const listeners = new Set<() => void>();
  return {
    current: () => target,
    setTarget(next) {
      const normalized = normalize(next);
      const requestsChanged = !sameRequests(target, normalized);
      if (!requestsChanged && target?.label === normalized?.label) return;
      target = normalized;
      if (requestsChanged) onChange();
      listeners.forEach((listener) => listener());
    },
    subscribe(listener) {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
  };
}
