import type { Source } from "./types";

export function localAgentSyncOperation(source: Source): string | null {
  return source.execution?.operation ?? null;
}

export function isLocalAgentBackedSource(source: Source): boolean {
  return source.execution?.kind === "local_agent";
}

/** Where a Source's acquisition runs: on the server, or on its execution owner's device. */
export type SourceExecutor = "server" | "device";

/** A Source runs on a device exactly when the source API names its device. */
export function sourceExecutor(source: Source): SourceExecutor {
  return source.execution?.device ? "device" : "server";
}
