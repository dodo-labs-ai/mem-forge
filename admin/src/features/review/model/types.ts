import type { components } from "@/api/schema.gen";

type Schemas = components["schemas"];

export type ReviewListItem = Schemas["MemoryReviewListItemResponse"];
export type ReviewDetail = Schemas["MemoryReviewDetailResponse"];
export type ReviewMemory = Schemas["MemoryReviewMemorySummary"];
export type ReviewEvidenceGroup = Schemas["MemoryEvidenceGroupDetail"];
export type ReviewPresentation = Schemas["ReviewPresentationResponse"];
export type ReviewAction = Schemas["ReviewActionPresentationResponse"];
export type ReviewDecision = ReviewAction["decision"];
export type DecisionManifestItem = Schemas["MemoryReviewManifestDecision"];
export type DecisionResult = Schemas["MemoryReviewDecisionResult"];
export type DecisionOutcome = DecisionResult["outcome"];
