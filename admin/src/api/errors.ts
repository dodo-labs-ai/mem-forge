/** A failed API call, with the server's own explanation when it gave one. */
export class ApiError extends Error {
  readonly status: number;
  /** The code of a coded refusal, such as `reindex_in_progress`; `null` when the server named none. */
  readonly code: string | null;

  constructor(status: number, message: string, code: string | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

/** The HTTP statuses whose failures the admin UI explains differently from others. */
export const HTTP_STATUS = {
  forbidden: 403,
  notFound: 404,
  conflict: 409,
} as const;

/** Whether `error` is an API failure with the given HTTP status. */
export function isApiErrorStatus(error: unknown, status: (typeof HTTP_STATUS)[keyof typeof HTTP_STATUS]): error is ApiError {
  return error instanceof ApiError && error.status === status;
}

const REINDEX_IN_PROGRESS = "reindex_in_progress";

/**
 * The codes of refusals that are about the search index, not about the
 * request: a change to memories is refused while the index is rebuilt, and
 * one that needs a new vector while the index does not fit the embedding
 * model in use. Each can be repeated as it is once the index is rebuilt.
 */
const SEARCH_INDEX_REFUSALS: ReadonlySet<string> = new Set([REINDEX_IN_PROGRESS, "reindex_required", "vector_index_missing"]);

/**
 * What every page says to a change that is refused while the search index is
 * rebuilt. Any change to memories can meet this refusal, so it is worded
 * once, with the place that shows how far the rebuild is.
 */
const REINDEX_IN_PROGRESS_MESSAGE =
  "The search index is being rebuilt, so this change has to wait. Try again when the rebuild has finished. Settings shows its progress.";

/**
 * Whether a change was refused because of the state of the search index. What
 * the change was about is as the user left it, so there is nothing to reload.
 */
export function isSearchIndexRefusal(error: unknown): error is ApiError {
  return error instanceof ApiError && error.code !== null && SEARCH_INDEX_REFUSALS.has(error.code);
}

/** Whether a request was refused because the search index is being rebuilt. */
export function isReindexInProgress(error: unknown): error is ApiError {
  return error instanceof ApiError && error.code === REINDEX_IN_PROGRESS;
}

/** Raised when a request is made while no workspace is selected. */
export class NoWorkspaceError extends Error {
  constructor() {
    super("Select a workspace to continue.");
    this.name = "NoWorkspaceError";
  }
}

interface ValidationIssue {
  loc?: unknown[];
  msg?: string;
}

/** The code of a coded refusal, which a FastAPI error body carries as `{ detail: { error, message } }`. */
export function errorCodeOf(body: unknown): string | null {
  if (!body || typeof body !== "object" || !("detail" in body)) return null;
  const detail = (body as { detail: unknown }).detail;
  if (!detail || typeof detail !== "object" || !("error" in detail)) return null;
  const code = (detail as { error: unknown }).error;
  return typeof code === "string" && code !== "" ? code : null;
}

/** Turns a FastAPI error body into one readable sentence. */
export function describeErrorBody(body: unknown, status: number): string {
  if (errorCodeOf(body) === REINDEX_IN_PROGRESS) return REINDEX_IN_PROGRESS_MESSAGE;
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    // Coded refusals carry the sentence for people next to the machine code: `{ error, message }`.
    if (detail && typeof detail === "object" && "message" in detail) {
      const message = (detail as { message: unknown }).message;
      if (typeof message === "string" && message !== "") return message;
    }
    if (Array.isArray(detail)) {
      const messages = (detail as ValidationIssue[])
        .map((issue) => issue.msg)
        .filter((message): message is string => Boolean(message));
      if (messages.length > 0) return messages.join(" ");
    }
  }
  return `The request failed with status ${status}.`;
}
