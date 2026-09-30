import { skipToken, useMutation, useQuery, useQueryClient, type QueryKey } from "@tanstack/react-query";
import { projectQueryKeys, unwrap, useApi, useProjectListQuery } from "@/api";
import { memoryKeys } from "@/features/memories";
import { sourceKeys } from "@/features/sources";
import type { CreateProjectBody } from "./model/projectForm";
import type { Project } from "./model/types";

const EMPTY_PROJECTS: Project[] = [];

/** The projects, and whether the caller may create, rename and delete them (false until the list loads). */
export function useProjectList() {
  const query = useProjectListQuery();
  return { ...query, projects: query.data?.data ?? EMPTY_PROJECTS, canManage: query.data?.can_manage ?? false };
}

/** Creating or renaming a project changes only the project list, which every page reads for names. */
const PROJECT_LIST_AFFECTED: readonly QueryKey[] = [projectQueryKeys.all];
/** Deleting a project also moves its memories to UNSORTED and stops its Sources writing to it. */
const PROJECT_DELETION_AFFECTED: readonly QueryKey[] = [projectQueryKeys.all, memoryKeys.all, sourceKeys.all];

/** Runs one project change, then refetches every cached query in `affected`. */
function useProjectMutation<TVariables, TResult>(
  affected: readonly QueryKey[],
  run: (variables: TVariables) => Promise<TResult>,
) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: run,
    onSettled: () => Promise.all(affected.map((queryKey) => queryClient.invalidateQueries({ queryKey }))),
  });
}

export function useCreateProject() {
  const api = useApi();
  return useProjectMutation(PROJECT_LIST_AFFECTED, (body: CreateProjectBody) =>
    unwrap(api.POST("/api/v1/projects", { body })),
  );
}

export function useRenameProject() {
  const api = useApi();
  return useProjectMutation(PROJECT_LIST_AFFECTED, ({ projectId, name }: { projectId: string; name: string }) =>
    unwrap(api.PATCH("/api/v1/projects/{project_id}", { params: { path: { project_id: projectId } }, body: { name } })),
  );
}

export function useDeleteProject() {
  const api = useApi();
  return useProjectMutation(PROJECT_DELETION_AFFECTED, (projectId: string) =>
    unwrap(api.DELETE("/api/v1/projects/{project_id}", { params: { path: { project_id: projectId } } })),
  );
}

/**
 * What deleting the project changes across the workspace; idle while no
 * project is picked. Read fresh each time the dialog opens, because the
 * counts are what the admin confirms.
 */
export function useProjectDeletionImpact(projectId: string | null) {
  const api = useApi();
  return useQuery({
    queryKey: projectQueryKeys.deletionImpact(projectId ?? ""),
    staleTime: 0,
    queryFn:
      projectId === null
        ? skipToken
        : () =>
            unwrap(
              api.GET("/api/v1/projects/{project_id}/deletion-impact", { params: { path: { project_id: projectId } } }),
            ),
  });
}
