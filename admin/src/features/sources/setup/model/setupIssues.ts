import { ApiError } from "@/api";

/** One thing to correct in a configuration, keyed by field where the server knows it. */
export interface FieldIssue {
  key: string | null;
  message: string;
}

/** Issues by field key, and those that name no field, as the setup page shows them. */
export interface IssueView {
  byField: Record<string, string>;
  general: string[];
}

/**
 * The field issues of a configuration the server refused: 400
 * `config_invalid` with `detail.fields`. Null for any other error.
 */
export function configIssuesOf(error: unknown): FieldIssue[] | null {
  if (!(error instanceof ApiError) || error.code !== "config_invalid") return null;
  const fields = Array.isArray(error.detail.fields) ? (error.detail.fields as unknown[]) : [];
  const issues = fields.flatMap((item): FieldIssue[] => {
    if (!item || typeof item !== "object") return [];
    const { key, message } = item as { key?: unknown; message?: unknown };
    if (typeof message !== "string" || message === "") return [];
    return [{ key: typeof key === "string" && key !== "" ? key : null, message }];
  });
  return issues.length > 0 ? issues : [{ key: null, message: error.message }];
}

/** Issues grouped for display: one sentence per field, several joined, and the rest as general lines. */
export function issueView(issues: readonly FieldIssue[]): IssueView {
  const byField: Record<string, string> = {};
  const general: string[] = [];
  for (const { key, message } of issues) {
    if (key === null) general.push(message);
    else byField[key] = byField[key] ? `${byField[key]} ${message}` : message;
  }
  return { byField, general };
}
