import { formatDate } from "@/lib/format";
import type { Tone } from "@/patterns";
import { memoryStatus, memoryStatusLabel } from "./memoryPresentation";
import type { Memory } from "./types";

type LifecycleFields = Pick<
  Memory,
  "status" | "retirement_reason" | "retired_at" | "replacement_reason" | "superseded_at" | "superseded_by"
>;

/** Retirement reasons MemForge records itself, in words. A person's own reason is shown as written. */
const RETIREMENT_REASON_LABELS: Readonly<Record<string, string>> = {
  source_deleted: "Source document removed from current indexed source",
  admin_hidden: "Manually hidden by admin",
  review_rejected: "Rejected during review",
};

const NOT_RECORDED = "Not recorded";
const PENDING_REVIEW_REASON = "Quarantined pending review";

export interface LifecycleDetail {
  status: string;
  reason: string;
  /** When the Memory left search, and what that step is called. */
  occurred?: { label: string; at: string };
  /** The Memory that replaced this one. */
  replacedBy?: string;
  /** The recorded reason code, when the reason above translates it. */
  technicalReason?: string;
}

function retirementReason(code: string | null | undefined): { reason: string; technicalReason?: string } {
  if (!code) return { reason: NOT_RECORDED };
  const label = RETIREMENT_REASON_LABELS[code];
  return label ? { reason: label, technicalReason: code } : { reason: code };
}

function hasLifecycleMetadata(memory: LifecycleFields): boolean {
  return Boolean(
    memory.retirement_reason || memory.retired_at || memory.replacement_reason || memory.superseded_at || memory.superseded_by,
  );
}

/**
 * What happened to a Memory that left search. An active Memory has a
 * lifecycle only when metadata from an earlier state is left over.
 */
export function lifecycleDetail(memory: LifecycleFields): LifecycleDetail | null {
  const status = memoryStatus(memory.status);
  const statusLabel = memoryStatusLabel(memory.status);
  if (status === "retired") {
    return {
      status: statusLabel,
      ...retirementReason(memory.retirement_reason),
      occurred: memory.retired_at ? { label: "Retired", at: memory.retired_at } : undefined,
    };
  }
  if (status === "superseded") {
    return {
      status: statusLabel,
      reason: memory.replacement_reason || NOT_RECORDED,
      occurred: memory.superseded_at ? { label: "Superseded", at: memory.superseded_at } : undefined,
      replacedBy: memory.superseded_by ?? undefined,
    };
  }
  if (status === "pending_review") return { status: statusLabel, reason: PENDING_REVIEW_REASON };
  if (!hasLifecycleMetadata(memory)) return null;
  const retired = retirementReason(memory.retirement_reason);
  const occurredAt = memory.retired_at ?? memory.superseded_at;
  return {
    status: statusLabel,
    reason: memory.replacement_reason || retired.reason,
    technicalReason: retired.technicalReason,
    occurred: occurredAt ? { label: memory.retired_at ? "Retired" : "Superseded", at: occurredAt } : undefined,
    replacedBy: memory.superseded_by ?? undefined,
  };
}

export interface StatusBanner {
  tone: Tone;
  title: string;
  message: string;
  /** The newer Memory a superseded one points to. */
  newerMemoryId?: string;
}

/** What a Memory's status means for search, for any status other than active. */
export function statusBanner(memory: LifecycleFields): StatusBanner | null {
  const status = memoryStatus(memory.status);
  if (status === "superseded") {
    const replaced = memory.superseded_at
      ? `A newer memory replaced this one on ${formatDate(memory.superseded_at)}.`
      : "A newer memory replaced this one.";
    return {
      tone: "live",
      title: "Superseded.",
      message: `${replaced} Superseded memories are kept for history and left out of search.`,
      newerMemoryId: memory.superseded_by ?? undefined,
    };
  }
  if (status === "retired") {
    const { reason } = retirementReason(memory.retirement_reason);
    const cause = memory.retirement_reason ? `${reason.replace(/\.$/, "")}. ` : "";
    return { tone: "idle", title: "Retired.", message: `${cause}Retired memories are left out of search.` };
  }
  if (status === "pending_review") {
    return {
      tone: "warn",
      title: "Needs review.",
      message: `${PENDING_REVIEW_REASON}. It stays out of search until a reviewer decides.`,
    };
  }
  return null;
}
