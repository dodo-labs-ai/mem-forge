import { ApiError } from "@/api";
import type { components } from "@/api/schema.gen";
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
  /** The change returns to the previous scope, as a restore made it. */
  returning: boolean;
  /** Where the change stands, in a few words. */
  status: string;
  /** What the change removes, and when. */
  summary: string;
  /** Why the last sync left it open, as the server recorded it; null when there is nothing to add. */
  reason: string | null;
  /** The settings the change moves; empty when it returns to the previous scope or the viewer cannot read them. */
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
  const { previous_scope: previous, target_scope: target } = transition;
  if (!previous || !target) return [];
  const fields = [...new Set([...Object.keys(previous), ...Object.keys(target)])];
  return fields
    .map((field) => ({
      field,
      label: scopeFieldLabel(field, labels),
      from: scopeValueLabel(previous[field]),
      to: scopeValueLabel(target[field]),
    }))
    .filter((change) => change.from !== change.to)
    .sort((a, b) => a.label.localeCompare(b.label));
}

/**
 * The failed items not yet synced under the change. One the new scope lists
 * keeps the change open; one only the previous scope held is settled when it
 * completes. Which is which is not known here, so the title names neither.
 */
export function transitionFailedItemsTitle(count: number): string {
  return `${pluralize(count, "failed item")} not yet synced under the change`;
}

/** Whether the change returns to the previous scope, as restoring it does. */
export function scopeTransitionReturns(transition: Pick<SourceScopeTransition, "previous_scope" | "target_scope">): boolean {
  const { previous_scope: previous, target_scope: target } = transition;
  return previous !== null && target !== null && stableJson(previous) === stableJson(target);
}

function stableJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(stableJson).join(",")}]`;
  if (value && typeof value === "object") {
    const entries = Object.entries(value as Record<string, unknown>).sort(([a], [b]) => a.localeCompare(b));
    return `{${entries.map(([key, item]) => `${JSON.stringify(key)}:${stableJson(item)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

/**
 * What a conversation source's change removes, and when it completes. Its next
 * sync removes the deselected conversations, and nothing is listed.
 */
const CONVERSATION_REMOVAL = "The next sync removes the conversations the new selection leaves out.";
const CONVERSATION_COMPLETION = "The change completes once every selected conversation is synced.";

function summary(transition: SourceScopeTransition, returning: boolean): string {
  if (transition.removes === "next_sync") return CONVERSATION_REMOVAL;
  return returning
    ? "Nothing the previous scope synced is removed. Once a sync lists it in full, items only the replaced change added are removed."
    : "Nothing is removed until every item the new scope lists is synced. Each sync retries the items that failed.";
}

function progressLabel(transition: SourceScopeTransition, returning: boolean): string {
  if (transition.removes === "next_sync") return CONVERSATION_COMPLETION;
  if (transition.excluded_scopes_due) {
    return "Every listed item is synced. The next sync checks what the previous scope held, then completes the change.";
  }
  const scope = returning ? "the previous scope" : "the new scope";
  if (transition.listed_count === null) return `The next sync lists what ${scope} holds.`;
  const synced = Math.min(transition.members_committed, transition.listed_count);
  return `${formatCount(synced)} of ${pluralize(transition.listed_count, "listed item")} synced under ${scope}`;
}

/** The scope change a source is applying, as its detail shows it; null when none is in progress. */
export function presentSourceScopeTransition(
  source: Source,
  activity: SourceSyncActivity | undefined,
  labels: Readonly<Record<string, string>> = {},
): SourceScopeTransitionView | null {
  const transition = source.scope_transition;
  if (!transition) return null;
  const returning = scopeTransitionReturns(transition);
  return {
    returning,
    status: STATUS_LABELS[transition.status],
    summary: summary(transition, returning),
    reason: transition.status === "failed" && !transition.excluded_scopes_due ? transition.error : null,
    changes: scopeFieldChanges(transition, labels),
    progressLabel: progressLabel(transition, returning),
    failedItems: recordedFailedItems(transition.failing_items, activity),
    restoreConfig: transition.restore_config,
  };
}

/** Refusals of a scope change, in words the user can act on. */
const SCOPE_CHANGE_ERROR_MESSAGES: Readonly<Record<string, string>> = {
  source_sync_running: "A sync of this source is running. Stop the sync, then try again.",
  projection_scope_transition_stale:
    "The scope change in progress changed while you were looking at it. Reload the source, then try again.",
  source_provider_namespace_immutable:
    "This change points the source at another site or project. Create a new source for it instead.",
  source_execution_mode_immutable:
    "This change moves the source between the server and a device. Create a new source for it instead.",
};

/**
 * The message for a refused scope change, by its code. A refusal the table
 * does not word, such as one for a mode that cannot list, keeps the server's
 * words, which name the mode and what to choose instead.
 */
export function scopeChangeRefusalMessage(code: string, serverMessage?: string | null): string {
  return SCOPE_CHANGE_ERROR_MESSAGES[code] ?? serverMessage ?? code;
}

/** The message for a scope change the server refused. */
export function scopeChangeErrorMessage(error: unknown): string {
  const message = errorMessage(error);
  const code = error instanceof ApiError ? (error.code ?? message) : message;
  return scopeChangeRefusalMessage(code, message);
}

export type ScopeChangePreview = components["schemas"]["SourceScopeChangePreviewResponse"];

/** What the restore confirmation says, from the server's preview of saving the previous scope. */
export interface RestorePresentation {
  description: string;
  /** False while the preview is on its way, or when restoring would not do what the dialog offers. */
  ready: boolean;
}

/**
 * The restore confirmation for a source, from the preview of saving its
 * restore configuration. A configuration that cannot list closes the change
 * and removes nothing. One that lists opens a change back to the previous
 * scope, which removes what only the replaced change added: once a sync lists
 * the previous scope in full for a document source, and in the next sync for a
 * conversation source.
 */
export function presentRestore(
  sourceName: string,
  preview: ScopeChangePreview | undefined,
  previewFailed: boolean,
): RestorePresentation {
  if (previewFailed) {
    return {
      description: `This undoes the scope change in progress and saves ${sourceName}'s previous scope again, keeping its other settings. Nothing the previous scope synced is removed.`,
      ready: true,
    };
  }
  if (!preview) return { description: "Checking what restoring the previous scope does…", ready: false };
  switch (preview.outcome) {
    case "restores_previous_scope":
      return {
        description: `This closes the scope change in progress and saves ${sourceName}'s previous scope again, keeping its other settings. Nothing is removed.`,
        ready: true,
      };
    case "returns_to_previous_scope":
      return {
        description:
          `This replaces the scope change in progress with a change back to ${sourceName}'s previous scope, keeping its other settings. `
          + (preview.removes === "next_sync"
            ? `${CONVERSATION_REMOVAL} ${CONVERSATION_COMPLETION}`
            : "Nothing the previous scope synced is removed. Once a sync lists the previous scope in full, items only the replaced change added are removed."),
        ready: true,
      };
    case "refused":
      return {
        description: preview.refusal?.message
          ?? SCOPE_CHANGE_ERROR_MESSAGES[preview.refusal?.code ?? ""]
          ?? "The previous scope cannot be restored now.",
        ready: false,
      };
    default:
      return { description: SCOPE_CHANGE_ERROR_MESSAGES.projection_scope_transition_stale!, ready: false };
  }
}
