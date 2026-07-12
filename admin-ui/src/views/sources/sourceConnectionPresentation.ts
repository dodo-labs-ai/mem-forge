import type { SourceExecutionKind } from "../../api/types.js";

export type SourceConnectionMode = "direct" | "device" | "choice";

type ExecutionAwareGene = {
  execution_kinds: readonly SourceExecutionKind[];
};

export interface SourceConnectionPresentation {
  mode: SourceConnectionMode;
  label: "Cloud" | "Local sync" | "Cloud or local";
}

export function presentSourceConnection(gene: ExecutionAwareGene): SourceConnectionPresentation {
  const server = gene.execution_kinds.includes("server");
  const localAgent = gene.execution_kinds.includes("local_agent");
  if (server && localAgent) return { mode: "choice", label: "Cloud or local" };
  if (localAgent) return { mode: "device", label: "Local sync" };
  if (server) return { mode: "direct", label: "Cloud" };
  throw new Error("Configurable source must declare at least one execution kind");
}
