import type { Source } from "./types";

export function localAgentSyncOperation(source: Source): string | null {
  return source.execution?.operation ?? null;
}

export function isLocalAgentBackedSource(source: Source): boolean {
  return source.execution?.kind === "local_agent";
}

/** Where a Source's acquisition runs: on the server, or on its execution owner's device. */
export type SourceExecutor = "server" | "device";

/** The local-agent operation a device Source's runs go through. */
const DOCUMENT_ACQUISITION_OPERATION = "document_acquisition";

/** A Source runs on a device when its runs are document acquisitions, whether or not it has an execution device yet. */
export function sourceExecutor(source: Source): SourceExecutor {
  return source.execution?.operation === DOCUMENT_ACQUISITION_OPERATION ? "device" : "server";
}
