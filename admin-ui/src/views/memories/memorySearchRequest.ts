import { LIST_PAGE_SIZE } from "@/lib/constants";

// The filter value that means "no filter" for type, status, and source.
const FILTER_ALL = "all";

export type MemorySearchScopeMode = "project" | "project-first";

export interface MemorySearchFilters {
  query: string;
  type: string;
  status: string;
  source: string;
  activeProject: string;
  scopeMode: MemorySearchScopeMode;
}

/**
 * Body for `POST /memories/search`. Mirrors `MemorySearchRequest`
 * (memforge.server.admin_api), which forbids unknown fields, so only fields
 * that model declares may appear here. The source facet is nested under
 * `source_filter.source_ids`, as in `SourceFacetFilterRequest`.
 */
export interface MemorySearchRequestBody {
  query: string;
  memory_types?: string[];
  status?: string;
  source_filter?: { source_ids: string[] };
  active_project: string;
  scope_mode: MemorySearchScopeMode;
  include_private: boolean;
  top_k: number;
}

export function buildMemorySearchRequest(filters: MemorySearchFilters): MemorySearchRequestBody {
  const body: MemorySearchRequestBody = {
    query: filters.query,
    active_project: filters.activeProject,
    scope_mode: filters.scopeMode,
    include_private: true,
    top_k: LIST_PAGE_SIZE,
  };
  if (filters.type !== FILTER_ALL) body.memory_types = [filters.type];
  if (filters.status !== FILTER_ALL) body.status = filters.status;
  if (filters.source !== FILTER_ALL) body.source_filter = { source_ids: [filters.source] };
  return body;
}
