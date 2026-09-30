import { MEMORIES_PROJECT_PARAM } from "@/lib/paths";
import { LIST_PAGE_SIZE, SEARCH_RESULT_LIMIT } from "../constants";
import { MEMORY_TYPES, STATUS_FILTERS } from "./memoryPresentation";
import { isRelationLabel } from "./relations";
import type { RelationLabel } from "./types";

export type MemoriesView = "memories" | "relations";

/**
 * How ranked search treats the chosen project: `project-first` ranks it
 * first and keeps other projects visible, `project` keeps only that project
 * and team-wide memories.
 */
export type ProjectScope = "project-first" | "project";

export interface MemoryFilters {
  type: string | null;
  /** A lifecycle status other than active; without one the list shows active Memories. */
  status: string | null;
  source: string | null;
  project: string | null;
  projectScope: ProjectScope;
}

export type FilterKey = "type" | "status" | "source" | "project";

export interface MemoriesPageState {
  view: MemoriesView;
  query: string;
  filters: MemoryFilters;
  /** The relation label the Relations view shows, or every label. */
  relation: RelationLabel | null;
  /** One-based page of the current list. */
  page: number;
}

const PARAM = {
  view: "view",
  query: "q",
  type: "type",
  status: "status",
  source: "source",
  project: MEMORIES_PROJECT_PARAM,
  scope: "scope",
  relation: "relation",
  page: "page",
} as const;

/** The `scope` URL value that keeps ranked search inside the project. */
const ONLY_PROJECT_SCOPE = "only";
const FIRST_PAGE = 1;
const RELATIONS_VIEW: MemoriesView = "relations";
const FILTER_KEYS: readonly FilterKey[] = ["type", "status", "source", "project"];

const NO_FILTERS: MemoryFilters = { type: null, status: null, source: null, project: null, projectScope: "project-first" };

function oneOf(value: string | null, allowed: readonly string[]): string | null {
  return value !== null && allowed.includes(value) ? value : null;
}

function nonEmpty(value: string | null): string | null {
  return value !== null && value.trim() !== "" ? value : null;
}

function pageNumber(value: string | null): number {
  const page = Number(value);
  return Number.isInteger(page) && page >= FIRST_PAGE ? page : FIRST_PAGE;
}

/** Reads the page state from the URL, dropping values the page does not offer. */
export function readPageState(params: URLSearchParams): MemoriesPageState {
  const relation = params.get(PARAM.relation);
  return {
    view: params.get(PARAM.view) === RELATIONS_VIEW ? RELATIONS_VIEW : "memories",
    query: params.get(PARAM.query) ?? "",
    filters: {
      type: oneOf(params.get(PARAM.type), MEMORY_TYPES),
      status: oneOf(params.get(PARAM.status), STATUS_FILTERS),
      source: nonEmpty(params.get(PARAM.source)),
      project: nonEmpty(params.get(PARAM.project)),
      projectScope: params.get(PARAM.scope) === ONLY_PROJECT_SCOPE ? "project" : "project-first",
    },
    relation: isRelationLabel(relation) ? relation : null,
    page: pageNumber(params.get(PARAM.page)),
  };
}

/** Writes the page state to URL parameters, leaving out every default. */
export function writePageState(state: MemoriesPageState): URLSearchParams {
  const params = new URLSearchParams();
  if (state.view === RELATIONS_VIEW) params.set(PARAM.view, RELATIONS_VIEW);
  if (state.query.trim() !== "") params.set(PARAM.query, state.query);
  for (const key of FILTER_KEYS) {
    const value = state.filters[key];
    if (value !== null) params.set(PARAM[key], value);
  }
  if (state.filters.project !== null && state.filters.projectScope === "project") params.set(PARAM.scope, ONLY_PROJECT_SCOPE);
  if (state.relation !== null) params.set(PARAM.relation, state.relation);
  if (state.page !== FIRST_PAGE) params.set(PARAM.page, String(state.page));
  return params;
}

/** The offset of a one-based page in the server's list. */
export function pageOffset(page: number): number {
  return (page - FIRST_PAGE) * LIST_PAGE_SIZE;
}

/** The Relations view showing one label, or every label. */
export function relationsPageState(relation: RelationLabel | null): MemoriesPageState {
  return { ...readPageState(new URLSearchParams()), view: RELATIONS_VIEW, relation };
}

/** Changing what the list shows starts it again from the first page. */
export function changePageState(state: MemoriesPageState, change: Partial<Omit<MemoriesPageState, "page">>): MemoriesPageState {
  return { ...state, ...change, page: FIRST_PAGE };
}

export function changeFilters(state: MemoriesPageState, change: Partial<MemoryFilters>): MemoriesPageState {
  return changePageState(state, { filters: { ...state.filters, ...change } });
}

export function clearFilters(state: MemoriesPageState): MemoriesPageState {
  return changePageState(state, { filters: NO_FILTERS });
}

/** The filters that narrow the list, in the order the toolbar shows them. */
export function activeFilterKeys(filters: MemoryFilters): FilterKey[] {
  return FILTER_KEYS.filter((key) => filters[key] !== null);
}

export interface MemoryListQuery {
  search?: string;
  type?: string;
  status?: string;
  source?: string;
  project?: string;
  include_private: true;
  limit: number;
  offset: number;
}

/**
 * Body for `POST /memories/search`. The request model forbids unknown fields,
 * and the source facet is nested under `source_filter.source_ids`.
 */
export interface MemorySearchBody {
  query: string;
  active_project: string;
  scope_mode: ProjectScope;
  include_private: true;
  /** The status filter decides which lifecycle states appear. */
  include_superseded: false;
  top_k: number;
  offset: 0;
  memory_types?: string[];
  status?: string;
  source_filter?: { source_ids: string[] };
}

export type MemoryListRequest = { kind: "list"; query: MemoryListQuery } | { kind: "search"; body: MemorySearchBody };

/**
 * Ranked search runs when the user searches within a project, where the
 * ranker can weigh project affinity. Every other list is the paginated
 * admin listing, which filters by keyword.
 */
export function memoryListRequest(state: MemoriesPageState): MemoryListRequest {
  const { filters } = state;
  const query = state.query.trim();
  if (query !== "" && filters.project !== null) {
    return {
      kind: "search",
      body: {
        query,
        active_project: filters.project,
        scope_mode: filters.projectScope,
        include_private: true,
        include_superseded: false,
        top_k: SEARCH_RESULT_LIMIT,
        offset: 0,
        ...(filters.type !== null ? { memory_types: [filters.type] } : {}),
        ...(filters.status !== null ? { status: filters.status } : {}),
        ...(filters.source !== null ? { source_filter: { source_ids: [filters.source] } } : {}),
      },
    };
  }
  return {
    kind: "list",
    query: {
      ...(query !== "" ? { search: query } : {}),
      ...(filters.type !== null ? { type: filters.type } : {}),
      ...(filters.status !== null ? { status: filters.status } : {}),
      ...(filters.source !== null ? { source: filters.source } : {}),
      ...(filters.project !== null ? { project: filters.project } : {}),
      include_private: true,
      limit: LIST_PAGE_SIZE,
      offset: pageOffset(state.page),
    },
  };
}
