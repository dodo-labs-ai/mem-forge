import { SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY } from "@/api";
import type { Project } from "./types";

export type ProjectBadgeKind = "project" | "shared" | "unsorted";

export interface ProjectBadge {
  kind: ProjectBadgeKind;
  label: string;
}

const SHARED_LABEL = "Shared";
const UNSORTED_LABEL = "Unsorted";

/** How a Memory's project reads: the team-wide and catch-all projects get their own labels. */
export function projectBadge(projectKey: string | null | undefined, projects: readonly Project[]): ProjectBadge | null {
  if (!projectKey) return null;
  if (projectKey === SHARED_PROJECT_KEY) return { kind: "shared", label: SHARED_LABEL };
  if (projectKey === UNSORTED_PROJECT_KEY) return { kind: "unsorted", label: UNSORTED_LABEL };
  return { kind: "project", label: projectName(projectKey, projects) };
}

export function projectName(projectKey: string, projects: readonly Project[]): string {
  return projects.find((project) => project.key === projectKey)?.name ?? projectKey;
}

export interface ProjectOption {
  value: string;
  label: string;
}

/**
 * Projects a user can filter by. The team-wide project says so; the
 * catch-all Unsorted project is reached through "All projects" and managed
 * from Sources, so it is not a choice here.
 */
export function projectFilterOptions(projects: readonly Project[]): ProjectOption[] {
  return projects
    .filter((project) => project.key !== UNSORTED_PROJECT_KEY)
    .map((project) => ({
      value: project.key,
      label: project.key === SHARED_PROJECT_KEY ? `${project.name} (team-wide)` : project.name,
    }));
}
