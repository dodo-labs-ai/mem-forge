/** A failed API call, with the server's own explanation when it gave one. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
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
    if (Array.isArray(detail)) {
      const messages = (detail as ValidationIssue[])
        .map((issue) => issue.msg)
        .filter((message): message is string => Boolean(message));
      if (messages.length > 0) return messages.join(" ");
    }
  }
  return `The request failed with status ${status}.`;
}
