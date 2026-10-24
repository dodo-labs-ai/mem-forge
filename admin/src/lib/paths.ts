/**
 * Paths of the product pages, for links from one page to another. Each page
 * reads its own URL parameters with the names declared here, so a link and
 * the page it opens cannot disagree.
 */
export const MEMORIES_PATH = "/memories";
export const REVIEW_PATH = "/review";
export const EVALUATION_PATH = "/evaluation";
export const SOURCES_PATH = "/sources";
export const PROJECTS_PATH = "/projects";

/** The Memories parameter that narrows the list to one project, by its key. */
export const MEMORIES_PROJECT_PARAM = "project";
/** The Evaluation parameter that scopes the page to one source, by its id. */
export const EVALUATION_SOURCE_PARAM = "source_id";
/** The Source settings parameter that opens one step, by its key, such as `connection` or `project`. */
export const SOURCE_SETTINGS_STEP_PARAM = "step";

export function memoryPath(memoryId: string): string {
  return `${MEMORIES_PATH}/${encodeURIComponent(memoryId)}`;
}

export function reviewPath(reviewId: string): string {
  return `${REVIEW_PATH}/${encodeURIComponent(reviewId)}`;
}

export function projectPath(projectKey: string): string {
  return `${PROJECTS_PATH}/${encodeURIComponent(projectKey)}`;
}

/** The Memories list narrowed to one project. */
export function projectMemoriesPath(projectKey: string): string {
  return `${MEMORIES_PATH}?${new URLSearchParams({ [MEMORIES_PROJECT_PARAM]: projectKey })}`;
}

/** The Evaluation page scoped to one source. */
export function sourceEvaluationPath(sourceId: string): string {
  return `${EVALUATION_PATH}?${new URLSearchParams({ [EVALUATION_SOURCE_PARAM]: sourceId })}`;
}

/** The Add source type picker, or with a type, the page that creates a Source of that type. */
export function newSourcePath(sourceType?: string): string {
  return sourceType === undefined ? `${SOURCES_PATH}/new` : `${SOURCES_PATH}/new/${encodeURIComponent(sourceType)}`;
}

/** One Source's settings, opened at a step when one is named. */
export function sourceSettingsPath(sourceId: string, step?: string): string {
  const path = `${SOURCES_PATH}/${encodeURIComponent(sourceId)}/settings`;
  return step === undefined ? path : `${path}?${new URLSearchParams({ [SOURCE_SETTINGS_STEP_PARAM]: step })}`;
}
