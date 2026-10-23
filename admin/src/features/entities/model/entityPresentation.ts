import { ApiError } from "@/api";
import type { Entity, EntityAlias } from "./types";

/** The name people read: the display name, or the canonical name when there is none. */
export function entityName(entity: Entity): string {
  return entity.display_name || entity.canonical_name;
}

/** The canonical name, shown under the display name only when the two differ. */
export function canonicalNameHint(entity: Entity): string | null {
  return entity.display_name && entity.display_name !== entity.canonical_name ? entity.canonical_name : null;
}

/** Alias sources a workspace admin wrote; every other source is an alias MemForge recorded while resolving names. */
const MANUAL_ALIAS_SOURCES: ReadonlySet<string> = new Set(["admin_manual", "manual"]);

/** Where an alias came from, in words. */
export function aliasOrigin(alias: EntityAlias): string {
  return MANUAL_ALIAS_SOURCES.has(alias.source) ? "Added by an admin" : "Found by MemForge";
}

const ENTITY_CURATION_FORBIDDEN = "entity_curation_forbidden";

/** Whether a change was refused because only a workspace admin may change aliases or merge entities. */
export function isCurationForbidden(error: unknown): error is ApiError {
  return error instanceof ApiError && error.code === ENTITY_CURATION_FORBIDDEN;
}
