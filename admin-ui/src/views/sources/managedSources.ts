import { isSchemaSourceType } from "./sourceSetupAdapters.js";

type NamedSourceType = {
  name: string;
};

// Both the legacy singleton type and the per-client split ids are managed:
// they are populated automatically by plugins and cannot be user-configured.
const MANAGED_SOURCE_TYPES = new Set(["agent_session"]);
const MANAGED_SOURCE_ID_PREFIX = "src-agent-sessions-";

export function isManagedSourceType(sourceType: string): boolean {
  return MANAGED_SOURCE_TYPES.has(sourceType);
}

/** Returns true for per-client agent-session source ids (e.g. "src-agent-sessions-codex"). */
export function isManagedSourceId(sourceId: string): boolean {
  return sourceId.startsWith(MANAGED_SOURCE_ID_PREFIX);
}

export function canConfigureSourceType(sourceType: string): boolean {
  return !isManagedSourceType(sourceType);
}

/** Whether this UI has a setup form for the source type. */
export function hasSetupForm(sourceType: string): boolean {
  return sourceType === "teams" || isSchemaSourceType(sourceType);
}

/**
 * The source types the Add Source dialog offers: configurable ones this UI
 * has a setup form for. A type set up only in the newer admin UI is left out.
 */
export function userConfigurableGenes<T extends NamedSourceType>(genes: readonly T[]): T[] {
  return genes.filter((gene) => canConfigureSourceType(gene.name) && hasSetupForm(gene.name));
}

/**
 * Returns the managed (info-only) source types from the gene list. These are
 * push-based sources where the service cannot be configured by the user; the
 * Add Source dialog shows them as read-only information cards instead.
 */
export function infoOnlyGenes<T extends NamedSourceType>(genes: readonly T[]): T[] {
  return genes.filter((gene) => isManagedSourceType(gene.name));
}
