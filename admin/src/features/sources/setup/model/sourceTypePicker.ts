import type { components } from "@/api";

type GeneMetadata = components["schemas"]["GeneMetadataResponse"];
type ExecutionKind = GeneMetadata["execution_kinds"][number];

/** The types a person can add: every registered type except those the service manages. */
export function addableSourceTypes(genes: readonly GeneMetadata[]): GeneMetadata[] {
  return genes.filter((gene) => !gene.managed);
}

/** Where a type's syncs can run, as the type picker's badge names it: on the server, through local sync, or either. */
export function runBadge(executionKinds: readonly ExecutionKind[]): string {
  const server = executionKinds.includes("server");
  const local = executionKinds.includes("local_agent");
  if (server && local) return "Cloud or local sync";
  return local ? "Local sync" : "Cloud";
}

/** Whether a type can sync through local sync, so its card shows whether local sync is running. */
export function canRunOnLocalSync(executionKinds: readonly ExecutionKind[]): boolean {
  return executionKinds.includes("local_agent");
}
