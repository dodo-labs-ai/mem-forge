export interface PageMeta {
  /** Total number of pages for the result set (0 when empty). */
  totalPages: number;
  /** 1-based index of the first row shown on the current page (0 when empty). */
  pageStart: number;
  /** 1-based index of the last row shown on the current page (0 when empty). */
  pageEnd: number;
  /** Whether a previous page exists. */
  hasPrev: boolean;
  /** Whether a next page exists. */
  hasNext: boolean;
}

/**
 * Derive the display range and navigation state for a zero-based page index.
 *
 * `page` is clamped to a non-negative value; the returned range stays within
 * `[1, total]` so an out-of-range page never reports rows past the result set.
 */
export function getPageMeta(page: number, pageSize: number, total: number): PageMeta {
  const current = Math.max(0, page);
  const totalPages = total > 0 ? Math.ceil(total / pageSize) : 0;
  const pageStart = total === 0 ? 0 : Math.min(current * pageSize + 1, total);
  const pageEnd = Math.min(total, (current + 1) * pageSize);
  return {
    totalPages,
    pageStart,
    pageEnd,
    hasPrev: current > 0,
    hasNext: current + 1 < totalPages,
  };
}
