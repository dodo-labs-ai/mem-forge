import type { SourceConnectionStatus } from "../../api/types.js";

export type LocalDaemonReadiness = "checking" | "ready" | "unavailable";
export type LocalSourceReadiness =
  | "checking"
  | "ready"
  | "unavailable"
  | "sign_in_required"
  | "configuration_required"
  | "account_mismatch";

export function resolveLocalSourceReadiness({
  daemon,
  connectionStatus,
}: {
  daemon: LocalDaemonReadiness;
  connectionStatus?: SourceConnectionStatus | null;
}): LocalSourceReadiness {
  if (daemon !== "ready") return daemon;
  if (connectionStatus?.state !== "action_required") return "ready";
  if (connectionStatus.reason === "identity_conflict") return "account_mismatch";
  if (connectionStatus.reason === "configuration") return "configuration_required";
  return "sign_in_required";
}
