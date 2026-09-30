/** Rows per page in the memory and relation lists. */
export const LIST_PAGE_SIZE = 50;

/** Ranked search returns one window of the best matches; the API caps it at 50. */
export const SEARCH_RESULT_LIMIT = 50;

/**
 * Open reviews read to find the review behind each "Needs review" memory.
 * The API caps one page at 500.
 */
export const OPEN_REVIEW_LOOKUP_LIMIT = 500;

/** A dismissal note is a short explanation; the API refuses longer notes. */
export const DISMISSAL_NOTE_MAX_CHARS = 2000;

/** Delay before typed search text reaches the URL and the server. */
export const SEARCH_DEBOUNCE_MS = 300;
