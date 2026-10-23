import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { resourceClient } from "@/api/client";
import type { AbsenceCheckSkippedReason, SourceScopeChangePreview, SourceScopeTransition, SyncStatus } from "@/api/types";

/** How long the form waits after the last edit before it asks what saving would do. */
const PREVIEW_DELAY_MS = 400;

/** Refusals of a Source change, in words the user can act on. */
const REFUSAL_MESSAGES: Readonly<Record<string, string>> = {
  projection_scope_transition_stale:
    "The scope change in progress changed while you were editing. Close this dialog, open the source again, and save.",
  projection_scope_transition_active:
    "This source finishes its scope change in progress before it accepts another one. Save again once it completes.",
  source_provider_namespace_immutable:
    "This change points the source at another site or project. Create a new source for it instead.",
  source_execution_mode_immutable:
    "This change moves the source between the server and a device. Create a new source for it instead.",
};

/** The code of a refused save, which the server sends as `{ detail: { code } }` or as `{ detail: "<code>" }`. */
function refusalCode(error: unknown): string | null {
  if (typeof error !== "object" || error === null) return null;
  const detail = (error as { response?: { data?: { detail?: unknown } } }).response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (typeof detail === "object" && detail !== null && "code" in detail) {
    const code = (detail as { code: unknown }).code;
    return typeof code === "string" ? code : null;
  }
  return null;
}

/**
 * The message for a save the server refused because of the Source's scope
 * or identity; null for any other error. A refusal for a mode that cannot
 * list keeps the server's words, which name the mode and what to choose
 * instead.
 */
export function refusedSaveMessage(error: unknown): string | null {
  const code = refusalCode(error);
  return code ? REFUSAL_MESSAGES[code] ?? null : null;
}

/** A note the configuration form shows before saving a change of what the source syncs. */
export interface ScopeChangeWarning {
  /** `warning` when saving would be refused, `info` when it replaces the change in progress. */
  tone: "warning" | "info";
  title: string;
  message: string;
}

/** What the form says before saving, from the server's preview of the change; null when there is nothing to say. */
export function scopeChangeWarning(preview: SourceScopeChangePreview | undefined): ScopeChangeWarning | null {
  if (!preview) return null;
  switch (preview.outcome) {
    case "refused": {
      const refusal = preview.refusal;
      const code = refusal?.code ?? "";
      if (code === "scope_change_requires_listing") {
        return {
          tone: "warning",
          title: "This mode cannot change what the source syncs",
          message: refusal?.message
            ?? "Choose a mode that lists the whole scope, or create a new source for this scope.",
        };
      }
      if (code === "transition_requires_listing") {
        return {
          tone: "warning",
          title: "This would keep the scope change in progress from finishing",
          message: refusal?.message ?? "Keep a mode that lists the whole scope until the change completes.",
        };
      }
      return {
        tone: "warning",
        title: "This change cannot be saved",
        message: REFUSAL_MESSAGES[code] ?? refusal?.message ?? code,
      };
    }
    case "supersedes_transition":
      return {
        tone: "info",
        title: "This replaces the scope change in progress",
        message:
          "The source changes from the scope it had before that change to this one. Nothing is removed until every item of the new scope is synced.",
      };
    case "restores_previous_scope":
      return {
        tone: "info",
        title: "This replaces the scope change in progress",
        message: "The source returns to its previous scope. Nothing the previous scope synced is removed.",
      };
    default:
      return null;
  }
}

function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setDebounced(value), delayMs);
    return () => clearTimeout(timer);
  }, [value, delayMs]);
  return debounced;
}

/**
 * Asks the server what saving `config` would do to the Source's scope, a
 * moment after the last edit. `config` is null while there is nothing to ask
 * about, such as for a new source or an incomplete form. A configuration the
 * server rejects yields no preview; saving reports it.
 */
export function useScopeChangePreview(sourceId: string | undefined, config: Record<string, unknown> | null) {
  const serialized = useDebouncedValue(config ? JSON.stringify(config) : null, PREVIEW_DELAY_MS);
  const preview = useQuery<SourceScopeChangePreview>({
    queryKey: ["scope-change-preview", sourceId, serialized],
    queryFn: () =>
      resourceClient
        .post(`/sources/${sourceId}/scope-change-preview`, { config: JSON.parse(serialized!) })
        .then((response) => response.data as SourceScopeChangePreview),
    enabled: Boolean(sourceId && serialized),
    retry: false,
    staleTime: 0,
  });
  return sourceId && serialized ? preview.data : undefined;
}

const ABSENCE_CHECK_SKIPPED_NOTES: Record<AbsenceCheckSkippedReason, string> = {
  scope_listing_failed: "the list of items in scope could not be read in full",
  absence_confirmation_failed: "the source could not confirm which missing items were deleted",
  absence_confirmation_missing: "the local agent did not confirm which missing items were deleted",
};

/** What a finished sync's details add about deleted and kept items. */
export function lastSyncAbsenceNotes(sync: Pick<SyncStatus, "absence_check_skipped_reason" | "unlisted_present_count">): string[] {
  const notes: string[] = [];
  const reason = sync.absence_check_skipped_reason;
  if (reason) {
    notes.push(`Deletion check skipped: ${ABSENCE_CHECK_SKIPPED_NOTES[reason]}. Nothing was removed; the next sync checks again.`);
  }
  const kept = sync.unlisted_present_count ?? 0;
  if (kept > 0) {
    notes.push(
      kept === 1
        ? "1 item is outside the sync scope but still in the source, so it is kept and checked again each sync."
        : `${kept} items are outside the sync scope but still in the source, so they are kept and checked again each sync.`,
    );
  }
  return notes;
}

/** One line on the scope change a source is applying, for its sync details. */
export function scopeTransitionSummary(transition: SourceScopeTransition): string {
  if (transition.excluded_scopes_due) {
    return "Scope change in progress: every listed item is synced, and the next sync completes it.";
  }
  if (transition.listed_count === null) {
    return "Scope change in progress: the next sync lists what the new scope holds.";
  }
  const synced = Math.min(transition.members_committed, transition.listed_count);
  const failing = transition.failing_items.count;
  return `Scope change in progress: ${synced} of ${transition.listed_count} listed items synced${
    failing > 0 ? `, ${failing} failed` : ""
  }. Nothing is removed until it completes.`;
}
