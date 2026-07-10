import type { Source } from "../../api/types.js";

export function localAgentSyncOperation(source: Source): string | null {
  return source.execution?.operation ?? null;
}

export function isLocalAgentBackedSource(source: Source): boolean {
  return source.execution?.kind === "local_agent";
}

export function isImmutableExecutionModeField(source: Source, fieldKey: string): boolean {
  return source.execution?.immutable_config_fields.includes(fieldKey) ?? false;
}
