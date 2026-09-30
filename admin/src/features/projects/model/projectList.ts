import { RESERVED_PROJECT_KEYS, SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY, isReservedProjectKey } from "@/api";
import type { Project } from "./types";

/** What each built-in project is for, shown next to its name. */
const BUILT_IN_ROLES: Readonly<Record<string, string>> = {
  [SHARED_PROJECT_KEY]: "Team-wide",
  [UNSORTED_PROJECT_KEY]: "Catch-all",
};

export function builtInRole(key: string): string | undefined {
  return BUILT_IN_ROLES[key];
}

export interface ProjectSections {
  /** Projects people created, by name. */
  own: Project[];
  /** The built-in projects, SHARED first. */
  builtIn: Project[];
}

export function splitProjects(projects: readonly Project[]): ProjectSections {
  const own = projects
    .filter((project) => !isReservedProjectKey(project.key))
    .sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: "base" }) || a.key.localeCompare(b.key));
  const builtIn = RESERVED_PROJECT_KEYS.flatMap((key) => projects.filter((project) => project.key === key));
  return { own, builtIn };
}
