import type { KeyIntent } from "./settingsForm";

/** The action offered next to the status line. */
export type ApiKeyAction = "remove" | "undo";

export interface ApiKeyStatus {
  message: string;
  /** True when saving will delete the stored key. */
  warning: boolean;
  action: ApiKeyAction | null;
}

export interface SavedKey {
  set: boolean;
  last4: string | null;
}

/** "****a91f": enough to recognise a key without revealing it. */
export function maskedKey(last4: string): string {
  return `****${last4}`;
}

function savedKeyName(saved: SavedKey): string {
  return saved.last4 ? `saved key (${maskedKey(saved.last4)})` : "saved key";
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

/** What will happen to an endpoint's API key on save, and how to change that. */
export function apiKeyStatus(saved: SavedKey, intent: KeyIntent): ApiKeyStatus {
  if (intent === "remove") {
    return { message: "Saved key will be removed on save.", warning: true, action: "undo" };
  }
  if (intent === "replace") {
    return {
      message: saved.set ? `Will replace ${savedKeyName(saved)} on save.` : "New key will be saved.",
      warning: false,
      action: null,
    };
  }
  if (saved.set) {
    return { message: `${capitalize(savedKeyName(saved))} in use. Leave blank to keep it.`, warning: false, action: "remove" };
  }
  return { message: "Leave blank only for local endpoints that do not require auth.", warning: false, action: null };
}
