import type { SourceConnectionStatus } from "../../api/types.js";

export type LocalDaemonReadiness = "checking" | "ready" | "unavailable";
export type SourceReadiness =
  | "checking_local_sync"
  | "local_sync_ready"
  | "local_sync_unavailable"
  | "connection_ready"
  | "sign_in_required"
  | "configuration_required"
  | "account_mismatch";

export function resolveSourceReadiness({
  localExecution,
  daemon,
  connectionStatus,
}: {
  localExecution: boolean;
  daemon?: LocalDaemonReadiness;
  connectionStatus?: SourceConnectionStatus | null;
}): SourceReadiness | null {
  if (localExecution) {
    if (daemon === "checking" || daemon === undefined) return "checking_local_sync";
    if (daemon === "unavailable") return "local_sync_unavailable";
  }

  if (connectionStatus?.state === "action_required") {
    if (connectionStatus.reason === "identity_conflict") return "account_mismatch";
    if (connectionStatus.reason === "configuration") return "configuration_required";
    return "sign_in_required";
  }

  if (localExecution) return "local_sync_ready";
  if (connectionStatus?.state === "ready") return "connection_ready";
  return null;
}
