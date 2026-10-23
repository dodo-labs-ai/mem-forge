import type { components } from "@/api";

type EntityDetail = components["schemas"]["EntityDetailResponse"];

/** An entity a workspace admin sees: two aliases, twelve linked memories, and curation allowed. */
export function makeEntityDetail(overrides: Partial<EntityDetail> = {}): EntityDetail {
  return {
    id: 7,
    canonical_name: "cut-off",
    display_name: "Cut-off",
    created_at: "2026-07-02T09:00:00Z",
    aliases: [
      { alias: "Cutoff", alias_normalized: "cutoff", source: "admin_manual", created_at: "2026-07-03T09:00:00Z" },
      { alias: "Payroll cut-off", alias_normalized: "payroll cut-off", source: "llm_extracted", created_at: null },
    ],
    linked_memory_count: 12,
    can_curate: true,
    ...overrides,
  };
}

/** The list row of an entity detail. */
export function entityRow({ id, canonical_name, display_name, created_at }: EntityDetail) {
  return { id, canonical_name, display_name, created_at };
}
