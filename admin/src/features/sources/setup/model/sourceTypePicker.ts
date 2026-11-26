import type { components } from "@/api";

type GeneMetadata = components["schemas"]["GeneMetadataResponse"];

/**
 * Whether a type's syncs can run on the server and on this machine: a document
 * type's executors, or a conversation type's execution kinds.
 */
function runPlaces(type: GeneMetadata): { server: boolean; local: boolean } {
  if (type.executors) return { server: type.executors.includes("server"), local: type.executors.includes("device") };
  return { server: type.execution_kinds.includes("server"), local: type.execution_kinds.includes("local_agent") };
}

/** Where a type's syncs can run, as the type picker's badge names it: on the server, through local sync, or either. */
export function runBadge(type: GeneMetadata): string {
  const { server, local } = runPlaces(type);
  if (server && local) return "Cloud or local sync";
  return local ? "Local sync" : "Cloud";
}

/** Whether a type can sync through local sync, so its card shows whether local sync is running. */
export function canRunOnLocalSync(type: GeneMetadata): boolean {
  return runPlaces(type).local;
}
