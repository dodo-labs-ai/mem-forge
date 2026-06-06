/**
 * Sticky global UI state for the active project.
 *
 * The active project is the project the user is currently working in. The
 * Memories list ranks its memories above cross-project hits, the topbar chip
 * surfaces it, and other surfaces use it to anchor their default view.
 *
 * The store is a tiny external store consumed via `useSyncExternalStore`,
 * persisted to `localStorage` so the choice survives a reload. A separate
 * `crossProjectMode` flag captures the explicit "show me everything" view;
 * picking a project clears it. A `lastActiveProjectKey` is also persisted so
 * the cross-project banner's Exit link can restore the project the user was
 * working in before they entered the admin view.
 *
 * The mutator surface is intentionally minimal: callers either set the active
 * project (passing `null` to leave it unchosen) or enable cross-project mode.
 * Restoring a remembered project is just `setActiveProjectKey(lastActiveProjectKey)`.
 */

import { useSyncExternalStore } from "react";

const STORAGE_KEY_ACTIVE_PROJECT = "memforge.activeProjectKey";
const STORAGE_KEY_CROSS_PROJECT_MODE = "memforge.crossProjectMode";
const STORAGE_KEY_LAST_ACTIVE_PROJECT = "memforge.lastActiveProjectKey";

interface ActiveProjectState {
  activeProjectKey: string | null;
  crossProjectMode: boolean;
  lastActiveProjectKey: string | null;
}

type Listener = () => void;

function readInitial(): ActiveProjectState {
  if (typeof window === "undefined") {
    return {
      activeProjectKey: null,
      crossProjectMode: false,
      lastActiveProjectKey: null,
    };
  }
  try {
    const key = window.localStorage.getItem(STORAGE_KEY_ACTIVE_PROJECT);
    const cross = window.localStorage.getItem(STORAGE_KEY_CROSS_PROJECT_MODE);
    const last = window.localStorage.getItem(STORAGE_KEY_LAST_ACTIVE_PROJECT);
    return {
      activeProjectKey: key && key.length > 0 ? key : null,
      crossProjectMode: cross === "1",
      lastActiveProjectKey: last && last.length > 0 ? last : null,
    };
  } catch {
    return {
      activeProjectKey: null,
      crossProjectMode: false,
      lastActiveProjectKey: null,
    };
  }
}

let state: ActiveProjectState = readInitial();
const listeners = new Set<Listener>();

function emit() {
  for (const listener of listeners) listener();
}

function persist(next: ActiveProjectState) {
  if (typeof window === "undefined") return;
  try {
    if (next.activeProjectKey) {
      window.localStorage.setItem(STORAGE_KEY_ACTIVE_PROJECT, next.activeProjectKey);
    } else {
      window.localStorage.removeItem(STORAGE_KEY_ACTIVE_PROJECT);
    }
    if (next.crossProjectMode) {
      window.localStorage.setItem(STORAGE_KEY_CROSS_PROJECT_MODE, "1");
    } else {
      window.localStorage.removeItem(STORAGE_KEY_CROSS_PROJECT_MODE);
    }
    if (next.lastActiveProjectKey) {
      window.localStorage.setItem(
        STORAGE_KEY_LAST_ACTIVE_PROJECT,
        next.lastActiveProjectKey,
      );
    } else {
      window.localStorage.removeItem(STORAGE_KEY_LAST_ACTIVE_PROJECT);
    }
  } catch {
    // Storage may be unavailable (private mode, quota); the in-memory store
    // still reflects the change for the rest of the session.
  }
}

function subscribe(listener: Listener): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function getSnapshot(): ActiveProjectState {
  return state;
}

function setActiveProjectKey(key: string | null): void {
  const normalized = key && key.length > 0 ? key : null;
  if (
    state.activeProjectKey === normalized &&
    !state.crossProjectMode &&
    (normalized === null || state.lastActiveProjectKey === normalized)
  ) {
    return;
  }
  state = {
    activeProjectKey: normalized,
    crossProjectMode: false,
    lastActiveProjectKey: normalized ?? state.lastActiveProjectKey,
  };
  persist(state);
  emit();
}

function enableCrossProjectMode(): void {
  if (state.crossProjectMode && state.activeProjectKey === null) return;
  state = {
    activeProjectKey: null,
    crossProjectMode: true,
    // The most-recently-used real project is preserved here so the cross-
    // project banner Exit link can restore it without a reload.
    lastActiveProjectKey: state.activeProjectKey ?? state.lastActiveProjectKey,
  };
  persist(state);
  emit();
}

export interface UseActiveProjectResult {
  activeProjectKey: string | null;
  crossProjectMode: boolean;
  lastActiveProjectKey: string | null;
  setActiveProjectKey: (key: string | null) => void;
  enableCrossProjectMode: () => void;
}

export function useActiveProject(): UseActiveProjectResult {
  const snapshot = useSyncExternalStore(subscribe, getSnapshot, getSnapshot);
  return {
    activeProjectKey: snapshot.activeProjectKey,
    crossProjectMode: snapshot.crossProjectMode,
    lastActiveProjectKey: snapshot.lastActiveProjectKey,
    setActiveProjectKey,
    enableCrossProjectMode,
  };
}
