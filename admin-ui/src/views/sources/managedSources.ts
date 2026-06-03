type NamedSourceType = {
  name: string;
};

// Both the legacy singleton type and the per-client split ids are managed:
// they are populated automatically by plugins and cannot be user-configured.
const MANAGED_SOURCE_TYPES = new Set(["agent_session"]);
const MANAGED_SOURCE_ID_PREFIX = "src-agent-sessions-";

// Source types whose data lives on the user's machine and is delivered to
// MemForge by a local CLI or plugin push, rather than pulled from a remote
// service. The Add Source dialog groups these into a "Push from your local
// agent" section because they share that delivery model.
const PUSH_BASED_SOURCE_TYPES = new Set(["local_markdown", "agent_session"]);

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

export function canDeleteSourceType(sourceType: string): boolean {
  return !isManagedSourceType(sourceType);
}

export function isPushBasedSourceType(sourceType: string): boolean {
  return PUSH_BASED_SOURCE_TYPES.has(sourceType);
}

export function userConfigurableGenes<T extends NamedSourceType>(genes: readonly T[]): T[] {
  return genes.filter((gene) => canConfigureSourceType(gene.name));
}

/**
 * Returns the managed (info-only) source types from the gene list. These are
 * push-based sources where the service cannot be configured by the user; the
 * Add Source dialog shows them as read-only information cards instead.
 */
export function infoOnlyGenes<T extends NamedSourceType>(genes: readonly T[]): T[] {
  return genes.filter((gene) => isManagedSourceType(gene.name));
}
