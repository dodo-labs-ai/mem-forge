import { useQuery } from "@tanstack/react-query";
import { unwrap, useApi } from "@/api";
import { useLocalSetupJob } from "../useLocalSetupJob";
import { normalizeRepoPath, repoItemsFromFilePaths, type RepoItem } from "./model/githubRepoPaths";
import type { ServerConfig } from "./model/configForm";

/** How many files the tree lists at most; a larger repository is narrowed by search. */
export const REPO_TREE_LIMIT = 2_000;

export interface RepoTree {
  items: RepoItem[];
  /** The repository holds more files than the tree lists. */
  truncated: boolean;
}

function text(config: ServerConfig, key: string): string {
  return typeof config[key] === "string" ? (config[key] as string).trim() : "";
}

/**
 * A GitHub repository's folders and files for the path picker, loaded when
 * `requested` is true. Cloud access asks the server (`browse-tree`); local
 * sync lists the remote repository through the daemon's Git access
 * (`github_repo_preview_tree`), whose job carries only the repository, ref
 * and file types it reads. Both pickers of the form, include and exclude,
 * share one tree for the repository, ref and file types.
 */
export function useRepoTree(config: ServerConfig, sourceId: string | undefined, requested: boolean) {
  const api = useApi();
  const previewTree = useLocalSetupJob("github_repo_preview_tree");
  const local = text(config, "connection_mode") === "local_push";
  const repoUrl = text(config, "repo_url");
  const ref = text(config, "ref");
  const includeExtensions = config.include_extensions;
  return useQuery({
    queryKey: ["github-repo-tree", sourceId ?? null, local, repoUrl, ref, includeExtensions] as const,
    queryFn: async (): Promise<RepoTree> => {
      if (local) {
        const result = await previewTree({ sourceId, payload: { repo_url: repoUrl, ref, include_extensions: includeExtensions, limit: REPO_TREE_LIMIT } });
        const items = repoItemsFromFilePaths(result.items.map((item) => normalizeRepoPath(item.relative_path)));
        return { items, truncated: result.counts.included > result.items.length };
      }
      const tree = await unwrap(
        api.POST("/api/v1/genes/github_repo/browse-tree", {
          body: { config: { ...config }, limit: REPO_TREE_LIMIT, source_id: sourceId ?? null },
        }),
      );
      return { items: tree.items, truncated: tree.truncated };
    },
    enabled: requested && repoUrl !== "",
    retry: false,
    staleTime: Infinity,
  });
}
