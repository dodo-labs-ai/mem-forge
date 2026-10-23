import { scopeChangeRefusalMessage } from "../../model/sourceScopeTransition";
import type { ScopeChangePreview } from "../api";

/** What the settings page says before saving a change of what the Source syncs. */
export interface ScopeChangeNotice {
  /** `danger` when saving would be refused, which blocks save; `info` when it replaces the change in progress. */
  tone: "danger" | "info";
  title: string;
  message: string;
}

/** The notice for the server's preview of a scope change; null when there is nothing to say. */
export function scopeChangeNotice(preview: ScopeChangePreview | undefined): ScopeChangeNotice | null {
  if (!preview) return null;
  switch (preview.outcome) {
    case "refused": {
      const code = preview.refusal?.code ?? "";
      const message = preview.refusal?.message;
      if (code === "scope_change_requires_listing") {
        return {
          tone: "danger",
          title: "This mode cannot change what the source syncs",
          message: message ?? "Choose a mode that lists the whole scope, or create a new source for this scope.",
        };
      }
      if (code === "transition_requires_listing") {
        return {
          tone: "danger",
          title: "This would keep the scope change in progress from finishing",
          message: message ?? "Keep a mode that lists the whole scope until the change completes.",
        };
      }
      return { tone: "danger", title: "This change cannot be saved", message: scopeChangeRefusalMessage(code, message) };
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
        message: "The source returns to its previous scope, and the change in progress closes. Nothing is removed.",
      };
    case "returns_to_previous_scope":
      return {
        tone: "info",
        title: "This replaces the scope change in progress",
        message:
          "The source returns to its previous scope. Nothing the previous scope synced is removed. Once a sync lists the previous scope in full, items only the replaced change added are removed.",
      };
    default:
      return null;
  }
}
