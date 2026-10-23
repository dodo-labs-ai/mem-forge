import { ApiError } from "@/api";
import { errorMessage } from "@/lib/errors";
import { formatCount, pluralize } from "@/lib/format";
import { recordedFailedItems, type SourceFailedItems } from "./sourceFailedItems";
import type { SourceSyncActivity } from "./sourceSyncActivity";
import type { Source } from "./types";

export type SourceScopeTransition = NonNullable<Source["scope_transition"]>;

/** One scope setting the change moves, from its previous value to its new one. */
export interface ScopeFieldChange {
  field: string;
  label: string;
  from: string;
  to: string;
}

/** What the Source page shows about the scope change a source is applying. */
export interface SourceScopeTransitionView {
  /** Where the change stands, in a few words. */
  status: string;
  /** Why the last sync left it open, as the server recorded it; null when there is nothing to add. */
  reason: string | null;
  changes: ScopeFieldChange[];
  /** Listed items synced under the change out of the items its new scope lists, or what comes next. */
  progressLabel: string;
  /** The items whose failure keeps the change open. */
  failedItems: SourceFailedItems | null;
  /** The configuration that restores the previous scope; null when this viewer cannot restore it. */
  restoreConfig: Record<string, unknown> | null;
}

const STATUS_LABELS: Record<SourceScopeTransition["status"], string> = {
  pending: "Waiting for the next sync",
  running: "A sync is working on it",
  failed: "The last sync left it open",
};

/**
 * How a person reads a scope setting's name. The source type's own label is
 * used when its configuration schema names the field.
 */
export function scopeFieldLabel(field: string, labels: Readonly<Record<string, string>> = {}): string {
  const known = labels[field];
  if (known) return known;
  const words = field.replace(/_/g, " ").trim();
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : field;
}

/** How a person reads a scope setting's value. */
export function scopeValueLabel(value: unknown): string {
  if (value === undefined || value === null || value === "") return "Not set";
  if (Array.isArray(value)) return value.length === 0 ? "Not set" : value.map((item) => scopeValueLabel(item)).join(", ");
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

/** The scope settings that differ between the previous scope and the new one, in a stable order. */
export function scopeFieldChanges(
  transition: Pick<SourceScopeTransition, "previous_scope" | "target_scope">,
  labels: Readonly<Record<string, string>> = {},
): ScopeFieldChange[] {
  const fields = [...new Set([...Object.keys(transition.previous_scope), ...Object.keys(transition.target_scope)])];
  return fields
    .map((field) => ({
      field,
      label: scopeFieldLabel(field, labels),
      from: scopeValueLabel(transition.previous_scope[field]),
      to: scopeValueLabel(transition.target_scope[field]),
    }))
    .filter((change) => change.from !== change.to)
    .sort((a, b) => a.label.localeCompare(b.label));
}

export function transitionFailedItemsTitle(count: number): string {
  return `${pluralize(count, "failed item")} ${count === 1 ? "keeps" : "keep"} the change open`;
}

function progressLabel(transition: SourceScopeTransition): string {
  if (transition.excluded_scopes_due) {
    return "Every listed item is synced. The next sync checks what the previous scope held, then completes the change.";
  }
  if (transition.listed_count === null) return "The next sync lists what the new scope holds.";
  const synced = Math.min(transition.members_committed, transition.listed_count);
  return `${formatCount(synced)} of ${pluralize(transition.listed_count, "listed item")} synced under the new scope`;
}

/** The scope change a source is applying, as its detail shows it; null when none is in progress. */
export function presentSourceScopeTransition(
  source: Source,
  activity: SourceSyncActivity | undefined,
  labels: Readonly<Record<string, string>> = {},
): SourceScopeTransitionView | null {
  const transition = source.scope_transition;
  if (!transition) return null;
  return {
    status: STATUS_LABELS[transition.status],
    reason: transition.status === "failed" && !transition.excluded_scopes_due ? transition.error : null,
    changes: scopeFieldChanges(transition, labels),
    progressLabel: progressLabel(transition),
    failedItems: recordedFailedItems(transition.failing_items, activity),
    restoreConfig: transition.restore_config,
  };
}

/** Refusals of a scope change, in words the user can act on. */
const SCOPE_CHANGE_ERROR_MESSAGES: Readonly<Record<string, string>> = {
  source_sync_running: "A sync of this source is running. Stop the sync, then try again.",
  projection_scope_transition_stale:
    "The scope change in progress changed while you were looking at it. Reload the source, then try again.",
  projection_scope_transition_active:
    "This source finishes its scope change in progress before it accepts another one. Try again once it completes.",
  source_provider_namespace_immutable:
    "This change points the source at another site or project. Create a new source for it instead.",
  source_execution_mode_immutable:
    "This change moves the source between the server and a device. Create a new source for it instead.",
};

/**
 * The message for a refused scope change. A refusal for a mode that cannot
 * list keeps the server's words, which name the mode and what to choose
 * instead.
 */
export function scopeChangeErrorMessage(error: unknown): string {
  const message = errorMessage(error);
  const code = error instanceof ApiError ? (error.code ?? message) : message;
  return SCOPE_CHANGE_ERROR_MESSAGES[code] ?? message;
}
