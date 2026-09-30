import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { unwrap, useApi, type ApiClient } from "@/api";
import { inBatches } from "./model/batchDecisions";
import { decisionNote } from "./model/reviewDecision";
import { queueQuery, type ReviewQueueView } from "./model/reviewQueue";
import type { DecisionManifestItem, DecisionResult, ReviewDecision, ReviewDetail } from "./model/types";

const CHECK_ROOT = "review-decision-check";

export const reviewKeys = {
  all: ["reviews"] as const,
  queue: (view: ReviewQueueView) => ["reviews", "queue", view.tab, view.kind, view.page] as const,
  detail: (reviewId: string) => ["reviews", "detail", reviewId] as const,
  /** Outside `all`: a check describes the moment it ran, so a decision never refetches it. */
  check: (decision: ReviewDecision, note: string, reviewIds: string[]) =>
    [CHECK_ROOT, decision, note, reviewIds] as const,
};

export function useReviewQueue(view: ReviewQueueView) {
  const api = useApi();
  return useQuery({
    queryKey: reviewKeys.queue(view),
    queryFn: () => unwrap(api.GET("/api/v1/memory-reviews", { params: { query: queueQuery(view) } })),
    placeholderData: keepPreviousData,
  });
}

export function useReviewDetail(reviewId: string) {
  const api = useApi();
  return useQuery({
    queryKey: reviewKeys.detail(reviewId),
    queryFn: () =>
      unwrap(api.GET("/api/v1/memory-reviews/{review_id}", { params: { path: { review_id: reviewId } } })),
  });
}

/**
 * A decision changes memories that other pages list, so every cached query
 * except decision checks refetches when it is next shown.
 */
function useRefreshAfterDecision() {
  const queryClient = useQueryClient();
  return () => queryClient.invalidateQueries({ predicate: (query) => query.queryKey[0] !== CHECK_ROOT });
}

/** Uses the latest state or keeps the current state of one review. */
export function useDecideReview(review: ReviewDetail) {
  const api = useApi();
  const queryClient = useQueryClient();
  const refresh = useRefreshAfterDecision();
  return useMutation({
    mutationFn: ({ decision, note }: { decision: ReviewDecision; note: string }) => {
      const params = { params: { path: { review_id: review.id } } };
      const body = { expected_fingerprint: review.decision_fingerprint, note: decisionNote(note) };
      return unwrap(
        decision === "approve"
          ? api.POST("/api/v1/memory-reviews/{review_id}/approve", { ...params, body })
          : api.POST("/api/v1/memory-reviews/{review_id}/reject", { ...params, body }),
      );
    },
    onSuccess: (decided) => {
      queryClient.setQueryData(reviewKeys.detail(review.id), decided);
      return refresh();
    },
  });
}

/** Asks for a new proposal from the current state of an expired review; resolves to the new review. */
export function useRecheckReview(review: ReviewDetail) {
  const api = useApi();
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () =>
      unwrap(
        api.POST("/api/v1/memory-reviews/{review_id}/refresh", {
          params: { path: { review_id: review.id } },
          body: { expected_fingerprint: review.decision_fingerprint },
        }),
      ),
    onSuccess: (rechecked) => {
      queryClient.setQueryData(reviewKeys.detail(rechecked.id), rechecked);
      return queryClient.invalidateQueries({ queryKey: reviewKeys.all });
    },
  });
}

async function validateBatch(api: ApiClient, decisions: DecisionManifestItem[]) {
  return unwrap(api.POST("/api/v1/memory-reviews/decisions/validate", { body: { decisions } }));
}

/** Checks each decision against the current state of its review, without changing anything. */
export async function checkDecisions(api: ApiClient, decisions: DecisionManifestItem[]): Promise<DecisionResult[]> {
  const checked = await Promise.all(inBatches(decisions).map((batch) => validateBatch(api, batch)));
  return checked.flatMap((response) => response.results);
}

/** What a bulk decision did to each review it sent. */
export interface AppliedDecisions {
  results: DecisionResult[];
  /**
   * Why a later batch failed, or null when every batch succeeded. The reviews
   * of the failed batch and of the batches after it have no result.
   */
  error: unknown;
}

/**
 * Validates each batch and applies it with the receipt the validation
 * returned. The server applies only the decisions that are still ready and
 * reports every other one with its reason. Batches run one after another;
 * when one fails after others were applied, the mutation still succeeds with
 * their results and the failure, so the user sees what already changed.
 */
export function useApplyDecisions() {
  const api = useApi();
  const refresh = useRefreshAfterDecision();
  return useMutation({
    mutationFn: async (decisions: DecisionManifestItem[]): Promise<AppliedDecisions> => {
      const results: DecisionResult[] = [];
      for (const batch of inBatches(decisions)) {
        try {
          const validated = await validateBatch(api, batch);
          const applied = await unwrap(
            api.POST("/api/v1/memory-reviews/decisions/apply", {
              body: { decisions: batch, validation_receipt: validated.validation_receipt },
            }),
          );
          results.push(...applied.results);
        } catch (error) {
          if (results.length === 0) throw error;
          return { results, error };
        }
      }
      return { results, error: null };
    },
    onSettled: refresh,
  });
}
