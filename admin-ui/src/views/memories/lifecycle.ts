import type { Memory } from "../../api/types.js";

const STATUS_LABELS: Record<string, string> = {
  active: "Active",
  superseded: "Superseded",
  retired: "Retired",
  decayed: "Retired",
  pending_review: "Needs Review",
};

const RETIREMENT_REASON_LABELS: Record<string, string> = {
  source_deleted: "Source document removed from current indexed source",
  admin_hidden: "Manually hidden by admin",
  review_rejected: "Rejected during review",
};

export interface LifecycleDetail {
  status: string;
  reason: string;
  occurredLabel?: string;
  occurredAt?: string;
  technicalReason?: string;
  replacedBy?: string;
}

export function getLifecycleDetail(memory: Memory): LifecycleDetail | null {
  if (memory.status === "retired" || memory.status === "decayed") {
    return {
      status: STATUS_LABELS[memory.status] ?? memory.status,
      reason: labelRetirementReason(memory.retirement_reason),
      occurredLabel: "Retired",
      occurredAt: memory.retired_at ?? undefined,
      technicalReason: memory.retirement_reason ?? undefined,
    };
  }

  if (memory.status === "superseded") {
    return {
      status: STATUS_LABELS[memory.status] ?? memory.status,
      reason: memory.replacement_reason || "Not recorded",
      occurredLabel: "Superseded",
      occurredAt: memory.superseded_at ?? undefined,
      replacedBy: memory.superseded_by ?? undefined,
    };
  }

  if (memory.status === "pending_review") {
    return {
      status: STATUS_LABELS[memory.status] ?? memory.status,
      reason: "Quarantined pending review",
    };
  }

  if (hasLifecycleMetadata(memory)) {
    return {
      status: STATUS_LABELS[memory.status] ?? memory.status,
      reason:
        memory.replacement_reason ||
        labelRetirementReason(memory.retirement_reason),
      occurredLabel: memory.retired_at ? "Retired" : "Superseded",
      occurredAt: memory.retired_at ?? memory.superseded_at ?? undefined,
      technicalReason: memory.retirement_reason ?? undefined,
      replacedBy: memory.superseded_by ?? undefined,
    };
  }

  return null;
}

function labelRetirementReason(reason: string | null): string {
  if (!reason) return "Not recorded";
  return RETIREMENT_REASON_LABELS[reason] ?? reason;
}

function hasLifecycleMetadata(memory: Memory): boolean {
  return Boolean(
    memory.retirement_reason ||
      memory.retired_at ||
      memory.replacement_reason ||
      memory.superseded_at ||
      memory.superseded_by,
  );
}
