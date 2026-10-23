import type { components } from "@/api";

/** An entity as the list returns it. */
export type Entity = components["schemas"]["EntityResponse"];
/** One entity with its aliases, linked memory count and whether the caller may curate it. */
export type EntityDetail = components["schemas"]["EntityDetailResponse"];
export type EntityAlias = components["schemas"]["EntityAliasResponse"];
