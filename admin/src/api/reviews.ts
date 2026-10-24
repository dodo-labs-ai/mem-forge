import { useQuery } from "@tanstack/react-query";
import { useApi } from "./ApiProvider";
import { unwrap } from "./client";

/** How often the count of reviews waiting for a decision is read again. */
const OPEN_REVIEW_COUNT_POLL_MS = 30_000;

/**
 * Query keys of memory reviews. The Review pages build theirs under `all`,
 * so a decision refreshes the waiting count the navigation shows as well.
 */
export const reviewQueryKeys = {
  all: ["reviews"] as const,
  openCount: ["reviews", "open-count"] as const,
};

/** How many reviews wait for a decision, or null until the count is known. */
export function useOpenReviewCount(): number | null {
  const api = useApi();
  const query = useQuery({
    queryKey: reviewQueryKeys.openCount,
    queryFn: () => unwrap(api.GET("/api/v1/memory-reviews", { params: { query: { status: "open", limit: 1 } } })),
    select: (page) => page.total,
    refetchInterval: OPEN_REVIEW_COUNT_POLL_MS,
  });
  return query.data ?? null;
}
