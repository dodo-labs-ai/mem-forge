import { formatRelative, pluralize } from "@/lib/format";
import { reviewState } from "./reviewDecision";
import type { ReviewListItem } from "./types";

/** The queue tabs, each one backend status filter. */
export type ReviewTab = "waiting" | "expired" | "applied" | "kept";
/** Which kind of decision the queue shows, each one backend origin filter. */
export type ReviewKind = "all" | "lifecycle" | "memory";

export const REVIEW_TABS: ReadonlyArray<{ value: ReviewTab; label: string; status: string }> = [
  { value: "waiting", label: "Waiting", status: "open" },
  { value: "expired", label: "Expired", status: "stale" },
  { value: "applied", label: "Applied", status: "approved" },
  { value: "kept", label: "Kept current", status: "rejected" },
];

export const REVIEW_KINDS: ReadonlyArray<{ value: ReviewKind; label: string; origin?: ReviewListItem["review_origin"] }> = [
  { value: "all", label: "All decisions" },
  { value: "lifecycle", label: "Source lifecycle", origin: "lifecycle" },
  { value: "memory", label: "Memory updates", origin: "memory" },
];

export const REVIEW_QUEUE_PAGE_SIZE = 25;

const DEFAULT_TAB: ReviewTab = "waiting";
const DEFAULT_KIND: ReviewKind = "all";
const FIRST_PAGE = 1;

export interface ReviewQueueView {
  tab: ReviewTab;
  kind: ReviewKind;
  /** One-based, as shown to the user and kept in the URL. */
  page: number;
}

/** The queue view a URL describes; unknown or missing values fall back to the first page of waiting reviews. */
export function queueViewFromSearch(search: URLSearchParams): ReviewQueueView {
  const tab = REVIEW_TABS.find((option) => option.value === search.get("tab"))?.value ?? DEFAULT_TAB;
  const kind = REVIEW_KINDS.find((option) => option.value === search.get("kind"))?.value ?? DEFAULT_KIND;
  const page = Number.parseInt(search.get("page") ?? "", 10);
  return { tab, kind, page: Number.isInteger(page) && page > FIRST_PAGE ? page : FIRST_PAGE };
}

/** The URL search for a queue view, leaving out values that are the default. */
export function searchForQueueView(view: ReviewQueueView): URLSearchParams {
  const search = new URLSearchParams();
  if (view.tab !== DEFAULT_TAB) search.set("tab", view.tab);
  if (view.kind !== DEFAULT_KIND) search.set("kind", view.kind);
  if (view.page !== FIRST_PAGE) search.set("page", String(view.page));
  return search;
}

/** The last page of a queue that holds `total` reviews; the first page when it is empty. */
export function lastQueuePage(total: number): number {
  return Math.max(FIRST_PAGE, Math.ceil(total / REVIEW_QUEUE_PAGE_SIZE));
}

/** The list endpoint's query for a queue view. */
export function queueQuery(view: ReviewQueueView) {
  return {
    status: REVIEW_TABS.find((option) => option.value === view.tab)!.status,
    origin: REVIEW_KINDS.find((option) => option.value === view.kind)!.origin,
    limit: REVIEW_QUEUE_PAGE_SIZE,
    offset: (view.page - FIRST_PAGE) * REVIEW_QUEUE_PAGE_SIZE,
  };
}

export interface QueueSummary {
  title: string;
  description: string;
}

/** The heading above the rows of one tab. */
export function queueSummary(tab: ReviewTab, total: number): QueueSummary {
  switch (tab) {
    case "waiting":
      return {
        title: "Decisions waiting",
        description: `${pluralize(total, "review")} ${total === 1 ? "needs" : "need"} your input. Current memories stay active and searchable until you decide.`,
      };
    case "expired":
      return {
        title: "Expired proposals",
        description: `${pluralize(total, "proposal")} expired because the memory changed before anyone decided. Open one to recheck it.`,
      };
    case "applied":
      return {
        title: "Latest state used",
        description: `${pluralize(total, "decision")} applied the proposed state.`,
      };
    case "kept":
      return {
        title: "Current state kept",
        description: `${pluralize(total, "decision")} kept the current memory and discarded the proposal.`,
      };
  }
}

/** What the queue says when a tab has no reviews. */
export function emptyQueueMessage(tab: ReviewTab): QueueSummary {
  switch (tab) {
    case "waiting":
      return { title: "All clear", description: "No current memory decisions need your attention." };
    case "expired":
      return { title: "No expired proposals", description: "Every proposal was decided before its memory changed." };
    case "applied":
      return { title: "Nothing applied yet", description: "Reviews where someone used the latest state appear here." };
    case "kept":
      return { title: "Nothing kept yet", description: "Reviews where someone kept the current state appear here." };
  }
}

/** The source a review came from, or what kind of decision it is when no single source is known. */
export function reviewSourceLabel(review: Pick<ReviewListItem, "source_name" | "review_origin">): string {
  if (review.source_name) return review.source_name;
  return review.review_origin === "lifecycle" ? "Source sync" : "Memory update";
}

export interface ReviewOrigin {
  sourceType: string;
  client: string | null;
}

/**
 * Where the proposal came from, for the source mark on a queue row: the
 * proposed memory's origin, or the current memory's when nothing replaces it.
 */
export function reviewOrigin(review: Pick<ReviewListItem, "incumbent" | "challenger">): ReviewOrigin | null {
  const memory = review.challenger ?? review.incumbent;
  if (!memory?.origin_source_type) return null;
  return { sourceType: memory.origin_source_type, client: memory.origin_client ?? null };
}

/** When a queue row's review opened, expired or was decided, relative to now. */
export function rowTimeLabel(
  review: Pick<ReviewListItem, "status" | "is_stale" | "created_at" | "resolved_at">,
  now: Date = new Date(),
): string | null {
  const state = reviewState(review);
  if (state === "waiting") return review.created_at ? formatRelative(review.created_at, now) : null;
  if (review.resolved_at) {
    return `${state === "expired" ? "Expired" : "Decided"} ${formatRelative(review.resolved_at, now)}`;
  }
  return review.created_at ? `Opened ${formatRelative(review.created_at, now)}` : null;
}
