import axios from "axios";
import type { QueryClient } from "@tanstack/react-query";

const STANDALONE_TARGET: WorkspaceApiTarget = Object.freeze({
  resourceBaseUrl: "/api",
  localAgentBaseUrl: "/api/cloud/local-agent",
});

export interface WorkspaceApiTarget {
  resourceBaseUrl: string;
  localAgentBaseUrl: string;
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

function normalizeBaseUrl(baseUrl: string): string {
  const normalized = baseUrl.trim().replace(/\/+$/, "");
  return normalized || "/";
}

function normalizeTarget(target: WorkspaceApiTarget): Readonly<WorkspaceApiTarget> {
  return Object.freeze({
    resourceBaseUrl: normalizeBaseUrl(target.resourceBaseUrl),
    localAgentBaseUrl: normalizeBaseUrl(target.localAgentBaseUrl),
  });
}

function sameTarget(
  left: Readonly<WorkspaceApiTarget> | null,
  right: Readonly<WorkspaceApiTarget> | null,
): boolean {
  if (left === null || right === null) return left === right;
  return (
    left.resourceBaseUrl === right.resourceBaseUrl &&
    left.localAgentBaseUrl === right.localAgentBaseUrl
  );
}

export function createWorkspaceApiController(
  queryClient: Pick<QueryClient, "clear">,
): WorkspaceApiController {
  return Object.freeze({
    current: () => currentTarget,
    setTarget: (target: WorkspaceApiTarget | null) => {
      const nextTarget = target === null ? null : normalizeTarget(target);
      if (sameTarget(currentTarget, nextTarget)) return;

      currentTarget = nextTarget;
      resourceClient.defaults.baseURL =
        nextTarget?.resourceBaseUrl ?? STANDALONE_TARGET.resourceBaseUrl;
      queryClient.clear();
    },
  });
}

export function currentLocalAgentBaseUrl(): string {
  return currentTarget?.localAgentBaseUrl ?? STANDALONE_TARGET.localAgentBaseUrl;
}
