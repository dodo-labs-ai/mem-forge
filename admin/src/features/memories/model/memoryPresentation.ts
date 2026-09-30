import type { Tone } from "@/patterns";

export const MEMORY_TYPES = ["fact", "decision", "convention", "procedure"] as const;
export type MemoryType = (typeof MEMORY_TYPES)[number];

export const MEMORY_TYPE_LABELS: Record<MemoryType, string> = {
  fact: "Fact",
  decision: "Decision",
  convention: "Convention",
  procedure: "Procedure",
};

const MEMORY_TYPE_TONES: Record<MemoryType, Tone> = {
  fact: "live",
  decision: "idle",
  convention: "ok",
  procedure: "warn",
};

export const MEMORY_STATUSES = ["active", "pending_review", "superseded", "retired"] as const;
export type MemoryStatus = (typeof MEMORY_STATUSES)[number];

/** What the list shows unless a status filter asks for another lifecycle status. */
export const DEFAULT_LIST_STATUS: MemoryStatus = "active";

/** The statuses the list filters by; the list shows active Memories without one. */
export const STATUS_FILTERS: readonly MemoryStatus[] = MEMORY_STATUSES.filter((status) => status !== DEFAULT_LIST_STATUS);

export const MEMORY_STATUS_LABELS: Record<MemoryStatus, string> = {
  active: "Active",
  pending_review: "Needs review",
  superseded: "Superseded",
  retired: "Retired",
};

const MEMORY_STATUS_TONES: Record<MemoryStatus, Tone> = {
  active: "ok",
  pending_review: "warn",
  superseded: "live",
  retired: "idle",
};

/** Storage statuses that read as another status: a decayed Memory is retired. */
const STATUS_ALIASES: Record<string, MemoryStatus> = { decayed: "retired" };

function isMemoryType(value: string): value is MemoryType {
  return (MEMORY_TYPES as readonly string[]).includes(value);
}

export function memoryStatus(value: string): MemoryStatus | null {
  if ((MEMORY_STATUSES as readonly string[]).includes(value)) return value as MemoryStatus;
  return STATUS_ALIASES[value] ?? null;
}

export function memoryTypeLabel(value: string): string {
  return isMemoryType(value) ? MEMORY_TYPE_LABELS[value] : value;
}

export function memoryTypeTone(value: string): Tone {
  return isMemoryType(value) ? MEMORY_TYPE_TONES[value] : "idle";
}

export function memoryStatusLabel(value: string): string {
  const status = memoryStatus(value);
  return status ? MEMORY_STATUS_LABELS[status] : value;
}

export function memoryStatusTone(value: string): Tone {
  const status = memoryStatus(value);
  return status ? MEMORY_STATUS_TONES[status] : "idle";
}

/** Superseded and retired Memories are kept for history and left out of search. */
export function isHistorical(value: string): boolean {
  const status = memoryStatus(value);
  return status === "superseded" || status === "retired";
}
