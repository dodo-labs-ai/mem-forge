/** A failed API call, with the server's own explanation when it gave one. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
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

/** Turns a FastAPI error body into one readable sentence. */
export function describeErrorBody(body: unknown, status: number): string {
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
