import assert from "node:assert/strict";

import type { Memory } from "../src/api/types.js";
import { getLifecycleDetail } from "../src/views/memories/lifecycle.js";

const baseMemory: Memory = {
  id: "mem-1",
  memory_type: "fact",
  content: "A remembered fact",
  content_hash: "hash",
  scope: "global",
  project_key: null,
  tags: [],
  confidence: 0.9,
  corroboration_count: 1,
  contradiction_count: 0,
  status: "active",
  retirement_reason: null,
  retired_at: null,
  superseded_at: null,
  superseded_by: null,
  replacement_reason: null,
  valid_from: null,
  valid_until: null,
  created_at: "2026-05-26T19:44:56Z",
  updated_at: "2026-05-26T19:44:56Z",
  extraction_context: null,
  entity_refs: [],
  sources: [],
};

assert.equal(getLifecycleDetail(baseMemory), null);

assert.deepEqual(
  getLifecycleDetail({
    ...baseMemory,
    status: "retired",
    retirement_reason: "source_deleted",
    retired_at: "2026-05-26T19:44:56Z",
  }),
  {
    status: "Retired",
    reason: "Source document removed from current indexed source",
    occurredLabel: "Retired",
    occurredAt: "2026-05-26T19:44:56Z",
    technicalReason: "source_deleted",
  },
);

assert.deepEqual(
  getLifecycleDetail({
    ...baseMemory,
    status: "superseded",
    replacement_reason: "newer source",
    superseded_at: "2026-05-26T20:00:00Z",
    superseded_by: "mem-new",
  }),
  {
    status: "Superseded",
    reason: "newer source",
    occurredLabel: "Superseded",
    occurredAt: "2026-05-26T20:00:00Z",
    replacedBy: "mem-new",
  },
);

assert.deepEqual(
  getLifecycleDetail({ ...baseMemory, status: "pending_review" }),
  {
    status: "Needs Review",
    reason: "Quarantined pending review",
  },
);

assert.equal(
  getLifecycleDetail({ ...baseMemory, status: "retired" })?.reason,
  "Not recorded",
);
