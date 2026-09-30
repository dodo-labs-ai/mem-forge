/** Router state a queue row passes to the review page, so its back link returns to the same queue view. */
export interface ReviewLinkState {
  queueSearch: string;
}

export function queueSearchFrom(state: unknown): string {
  if (state && typeof state === "object" && "queueSearch" in state && typeof state.queueSearch === "string") {
    return state.queueSearch;
  }
  return "";
}
