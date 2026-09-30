/**
 * Reads the user-facing message from an Admin API error response.
 *
 * The API reports failures as `detail`, either a plain string or a
 * structured `{ error, message }` object. Returns null when the error
 * carries neither, so callers can fall back to their own copy.
 */
export function apiErrorMessage(error: unknown): string | null {
  if (typeof error !== "object" || error === null) return null;
  const candidate = error as { response?: { data?: { detail?: unknown } } };
  const detail = candidate.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (typeof detail === "object" && detail !== null && "message" in detail) {
    const message = (detail as { message: unknown }).message;
    return typeof message === "string" ? message : null;
  }
  return null;
}
