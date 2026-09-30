import type { components } from "@/api/schema.gen";

type Schemas = components["schemas"];
type ReviewDetail = Schemas["MemoryReviewDetailResponse"];
type ReviewListItem = Schemas["MemoryReviewListItemResponse"];
type ReviewMemory = Schemas["MemoryReviewMemorySummary"];

const PRESENTATION: ReviewDetail["presentation"] = {
  decision_label: "Updated",
  summary: "Use the proposed source state or keep the current memory?",
  why_human: "Applying the proposal would change active memory state and requires your decision.",
  current_label: "Current memory",
  proposed_label: "Proposed memory",
  proposed_empty_text: "No replacement memory; this proposal removes source support.",
  actions: [
    {
      key: "use_latest_state",
      decision: "approve",
      label: "Use latest state",
      consequence: "Use the proposed state going forward and keep the previous state in audit history.",
      requires_note: false,
    },
    {
      key: "keep_current_state",
      decision: "reject",
      label: "Keep current state",
      consequence: "Keep the current memory active and discard this proposal.",
      requires_note: true,
    },
  ],
  technical_reason: "candidate_supersede_vs_audit_keep",
};

/** An active memory; override what a test needs. */
export function makeReviewMemory(overrides: Partial<ReviewMemory> = {}): ReviewMemory {
  return {
    id: "mem-current",
    memory_type: "fact",
    content: "On-demand pay dates can be any date in the open period.",
    corroboration_count: 1,
    status: "active",
    entity_refs: [],
    evidence: [],
    created_at: "2026-09-28T09:00:00Z",
    updated_at: "2026-09-28T09:00:00Z",
    ...overrides,
  };
}

const INCUMBENT = makeReviewMemory();
const CHALLENGER = makeReviewMemory({
  id: "mem-proposed",
  content: "On-demand pay dates must fall before the next regular pay date.",
  status: "pending_review",
});

/** A queue row for a waiting lifecycle review the caller can decide; override what a test needs. */
export function makeReviewListItem(overrides: Partial<ReviewListItem> = {}): ReviewListItem {
  return {
    id: "review-1",
    kind: "lifecycle",
    status: "pending",
    review_origin: "lifecycle",
    source_id: "src-jira",
    source_name: "SFPAY board",
    incumbent_memory_id: INCUMBENT.id,
    challenger_memory_id: CHALLENGER.id,
    reason: "candidate_supersede_vs_audit_keep",
    review_note: null,
    reviewer: null,
    replacement_kind: "supersession",
    created_at: "2026-09-29T09:00:00Z",
    resolved_at: null,
    is_stale: false,
    refreshable: false,
    decision_fingerprint: "review-decision-v1:current",
    presentation: PRESENTATION,
    can_decide: true,
    incumbent: INCUMBENT,
    challenger: CHALLENGER,
    ...overrides,
  };
}

/** A waiting lifecycle review the caller can decide; override what a test needs. */
export function makeReviewDetail(overrides: Partial<ReviewDetail> = {}): ReviewDetail {
  return { ...makeReviewListItem(), related_challengers: [], ...overrides };
}
