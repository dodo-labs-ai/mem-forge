/** Rows per page in the memory and relation lists. */
export const LIST_PAGE_SIZE = 50;

/** Ranked search returns one window of the best matches; the API caps it at 50. */
export const SEARCH_RESULT_LIMIT = 50;

/** A dismissal note is a short explanation; the API refuses longer notes. */
export const DISMISSAL_NOTE_MAX_CHARS = 2000;

/** Delay before typed search text reaches the URL and the server. */
export const SEARCH_DEBOUNCE_MS = 300;
