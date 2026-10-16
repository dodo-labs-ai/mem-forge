import axios from "axios";
import type { QueryClient } from "@tanstack/react-query";

/** Host-level local-agent job control route served by the MemForge server. */
export const LOCAL_AGENT_HOST_BASE_URL = "/api/cloud/local-agent";

const STANDALONE_TARGET: WorkspaceApiTarget = Object.freeze({
  resourceBaseUrl: "/api/v1",
  localAgentBaseUrl: LOCAL_AGENT_HOST_BASE_URL,
  workspaceId: "local",
  shared: false,
});

export interface WorkspaceApiTarget {
  resourceBaseUrl: string;
  localAgentBaseUrl: string;
  workspaceId: string | null;
  /**
   * Whether other people use this workspace too. Only then do Sources and
   * Memories say who can use them; in a workspace of one, everything belongs
   * to its one user.
   */
  shared: boolean;
}

export interface WorkspaceApiController {
  current(): WorkspaceApiTarget | null;
  setTarget(target: WorkspaceApiTarget | null): void;
}

export const resourceClient = axios.create({
  baseURL: STANDALONE_TARGET.resourceBaseUrl,
  headers: { "Content-Type": "application/json" },
});

export const hostClient = axios.create({
  baseURL: "",
  headers: { "Content-Type": "application/json" },
});

let currentTarget: Readonly<WorkspaceApiTarget> | null = null;
const targetListeners = new Set<() => void>();

function normalizeBaseUrl(baseUrl: string): string {
  const normalized = baseUrl.trim().replace(/\/+$/, "");
  return normalized || "/";
}

function normalizeTarget(target: WorkspaceApiTarget): Readonly<WorkspaceApiTarget> {
  return Object.freeze({
    resourceBaseUrl: normalizeBaseUrl(target.resourceBaseUrl),
    localAgentBaseUrl: normalizeBaseUrl(target.localAgentBaseUrl),
    workspaceId: target.workspaceId?.trim() || null,
    // Only a target that says it has one user hides who can use what.
    shared: target.shared !== false,
  });
}

/** Whether both targets send requests to the same workspace. */
function sameRequests(
  left: Readonly<WorkspaceApiTarget> | null,
  right: Readonly<WorkspaceApiTarget> | null,
): boolean {
  if (left === null || right === null) return left === right;
  return (
    left.resourceBaseUrl === right.resourceBaseUrl &&
    left.localAgentBaseUrl === right.localAgentBaseUrl &&
    left.workspaceId === right.workspaceId
  );
}

export function createWorkspaceApiController(
  queryClient: Pick<QueryClient, "clear">,
): WorkspaceApiController {
  return Object.freeze({
    current: () => currentTarget,
    setTarget: (target: WorkspaceApiTarget | null) => {
      const nextTarget = target === null ? null : normalizeTarget(target);
      const requestsChanged = !sameRequests(currentTarget, nextTarget);
      if (!requestsChanged && currentTarget?.shared === nextTarget?.shared) return;

      currentTarget = nextTarget;
      if (requestsChanged) {
        resourceClient.defaults.baseURL =
          nextTarget?.resourceBaseUrl ?? STANDALONE_TARGET.resourceBaseUrl;
        resourceClient.defaults.params = nextTarget?.workspaceId
          ? { workspace_id: nextTarget.workspaceId }
          : {};
        queryClient.clear();
      }
      targetListeners.forEach((listener) => listener());
    },
  });
}

/** Whether other people use the selected workspace; the standalone workspace has one user. */
export function currentWorkspaceIsShared(): boolean {
  return (currentTarget ?? STANDALONE_TARGET).shared;
}

/** Calls `listener` whenever the selected workspace target changes. */
export function subscribeWorkspaceTarget(listener: () => void): () => void {
  targetListeners.add(listener);
  return () => targetListeners.delete(listener);
}

export function currentLocalAgentBaseUrl(): string {
  return currentTarget?.localAgentBaseUrl ?? STANDALONE_TARGET.localAgentBaseUrl;
}

/** Applies the resource client's selected workspace to a same-origin resource link. */
export function workspaceResourceUrl(href: string, origin = window.location.origin): string {
  const url = new URL(href, origin);
  if (url.origin !== new URL(origin).origin || !(url.pathname === "/api/v1" || url.pathname.startsWith("/api/v1/"))) {
    return href;
  }
  const target = currentTarget ?? STANDALONE_TARGET;
  url.pathname = target.resourceBaseUrl + url.pathname.slice("/api/v1".length);
  if (target.workspaceId) url.searchParams.set("workspace_id", target.workspaceId);
  return url.href;
}
