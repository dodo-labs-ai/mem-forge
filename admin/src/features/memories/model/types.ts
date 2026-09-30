import type { components } from "@/api/schema.gen";

type Schemas = components["schemas"];

export type Memory = Schemas["MemoryResponse"];
export type MemoryDetail = Schemas["MemoryDetailResponse"];
export type MemorySearchHit = Schemas["MemorySearchResultDetail"];
export type MemorySourceRef = Schemas["MemorySourceRefDetail"];
export type MemoryRelation = Schemas["MemoryRelationDetail"];
export type RelatedMemory = Schemas["RelatedMemoryDetail"];
export type DismissedRelation = Schemas["DismissedRelationDetail"];
export type RelationPair = Schemas["RelationPairDetail"];
export type RelationLabel = RelationPair["label"];
export type EvidenceGroup = Schemas["MemoryEvidenceGroupDetail"];
export type EvidenceItem = Schemas["MemoryEvidenceItemDetail"];
export type MemoryStats = Schemas["MemoryStatsResponse"];
export type ReviewListItem = Schemas["MemoryReviewListItemResponse"];
export type CorrectionKind = NonNullable<Schemas["MemoryCorrectionProposalRequest"]["replacement_kind"]>;

/** The project fields the Memories pages show. */
export type Project = Pick<Schemas["ProjectResponse"], "key" | "name">;

/** The source fields the Memories filter offers. */
export type SourceOption = Pick<Schemas["SourceResponse"], "id" | "name">;
