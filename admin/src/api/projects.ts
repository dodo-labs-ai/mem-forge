import { useQuery } from "@tanstack/react-query";
import { useApi } from "./ApiProvider";
import { unwrap, type ApiClient } from "./client";
import type { components } from "./schema.gen";

/**
 * The built-in projects every workspace has. The server creates them, refuses
 * to rename or delete them, and ranks SHARED memories with every project.
 */
export const SHARED_PROJECT_KEY = "SHARED";
export const UNSORTED_PROJECT_KEY = "UNSORTED";
export const RESERVED_PROJECT_KEYS: readonly string[] = [SHARED_PROJECT_KEY, UNSORTED_PROJECT_KEY];

export function isReservedProjectKey(key: string): boolean {
  return RESERVED_PROJECT_KEYS.includes(key);
}

/** A project as the list returns it, with the memories the caller can see in it. */
export type ProjectListItem = components["schemas"]["ProjectListItemResponse"];
/** The project list, with whether the caller may create, rename and delete projects. */
export type ProjectList = components["schemas"]["ProjectListResponse"];

/**
 * Query keys of the workspace's projects. Sources, Memories and Projects all
 * read the one list, so a change made on the Projects page reaches every page.
 */
export const projectQueryKeys = {
  all: ["projects"] as const,
  list: ["projects", "list"] as const,
  deletionImpact: (projectId: string) => ["projects", "deletion-impact", projectId] as const,
};

function projectListQuery(api: ApiClient) {
  return { queryKey: projectQueryKeys.list, queryFn: () => unwrap(api.GET("/api/v1/projects")) };
}

function projectsOf(list: ProjectList): ProjectListItem[] {
  return list.data;
}

/** The project list with the caller's authority, for the pages that change projects. */
export function useProjectListQuery() {
  return useQuery(projectListQuery(useApi()));
}

/** The workspace's projects, for the pages that name or filter by them. */
export function useProjects() {
  return useQuery({ ...projectListQuery(useApi()), select: projectsOf });
}
