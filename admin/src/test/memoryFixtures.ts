import type { components } from "@/api/schema.gen";

type Schemas = components["schemas"];

/** An active workspace Memory from one Confluence source; override what a test needs. */
export function makeMemory(overrides: Partial<Schemas["MemoryResponse"]> = {}): Schemas["MemoryResponse"] {
  return {
    id: "mem-cutoff",
    memory_type: "fact",
    content: "Cut-off runs the same way for default, deviating, and on-demand lifecycles.",
    content_hash: "hash-cutoff",
    visibility: "workspace",
    owner_user_id: null,
    project_key: "PAY",
    corroboration_count: 2,
    status: "active",
    created_at: "2026-09-25T09:00:00Z",
    updated_at: "2026-09-25T09:00:00Z",
    origin_source_type: "confluence",
    origin_client: null,
    sources: [{ source_id: "src-handbook", source_type: "confluence", name: "Payroll handbook" }],
    relations: [],
    ...overrides,
  };
}

export function makeRelatedMemory(overrides: Partial<Schemas["RelatedMemoryDetail"]> = {}): Schemas["RelatedMemoryDetail"] {
  return {
    memory_id: "mem-ondemand",
    summary: "The on-demand lifecycle does not support cut-off yet.",
    content_hash: "hash-ondemand",
    sources: [{ source_id: "src-board", source_type: "jira", name: "SFPAY board" }],
    evidence_time: "2026-09-18",
    ...overrides,
  };
}

export function makeRelation(overrides: Partial<Schemas["MemoryRelationDetail"]> = {}): Schemas["MemoryRelationDetail"] {
  return {
    label: "contradicts",
    role: "peer",
    counterpart: makeRelatedMemory(),
    decided_by: "classifier",
    ...overrides,
  };
}

export function makeMemoryDetail(overrides: Partial<Schemas["MemoryDetailResponse"]> = {}): Schemas["MemoryDetailResponse"] {
  return {
    ...makeMemory(),
    entity_refs: ["Cut-off", "PeriodLifecycle"],
    evidence: [
      {
        kind: "evidence_unit",
        support_ids: ["sup-1"],
        source_id: "src-handbook",
        source_type: "confluence",
        evidence_unit_id: "eu-1",
        doc_id: "doc-handbook",
        current: true,
        document: {
          doc_id: "doc-handbook",
          title: "Payroll Handbook",
          source_url: "https://wiki.example.test/payroll",
          content_url: "/api/v1/source-units/su-1/content",
          pdf_url: null,
          source_updated_at: "2026-09-22T09:14:00Z",
        },
        items: [
          {
            authority: "revision_pinned",
            role: "primary",
            kind: "text",
            support_contribution: true,
            interpretation_available: true,
            anchor_kind: "text_range",
            current: true,
            excerpt: "Cut-off is uniform across Default, Deviating and On-Demand lifecycle types.",
          },
        ],
      },
    ],
    relations: [makeRelation()],
    dismissed_relations: [],
    source_backed: true,
    ...overrides,
  };
}

export function makeRelationPair(overrides: Partial<Schemas["RelationPairDetail"]> = {}): Schemas["RelationPairDetail"] {
  return {
    label: "contradicts",
    decided_by: "classifier",
    decided_at: "2026-09-22T10:00:00Z",
    newer_memory_id: null,
    memories: [
      makeRelatedMemory({
        memory_id: "mem-cutoff",
        summary: "Cut-off runs the same way for default, deviating, and on-demand lifecycles.",
        sources: [{ source_id: "src-handbook", source_type: "confluence", name: "Payroll handbook" }],
        evidence_time: "2026-09-22",
      }),
      makeRelatedMemory(),
    ],
    ...overrides,
  };
}
