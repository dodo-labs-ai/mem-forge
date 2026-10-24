import { isSchemaSourceType } from "./sourceSetupAdapters.js";

type NamedSourceType = {
  name: string;
};

// Agent-session Sources, by type or by their per-client id prefix, are managed:
// they hold memories kept from earlier Codex or Claude Code sessions and take
// in no new content, so they have no sync runs, schedule or configuration.
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
