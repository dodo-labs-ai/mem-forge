import type { ProjectListItem } from "@/api";

/** A project as the list returns it. `kind` only labels projects in V1 and is not read here. */
export type Project = Omit<ProjectListItem, "kind">;
