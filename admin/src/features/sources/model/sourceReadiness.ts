import { sourceExecutor } from "./localAgentSources";
import type { Source, SourceConnectionStatus } from "./types";

/** Whether the viewer's own local sync daemon answers: still asking, online, or offline. */
export type LocalDaemonReadiness = "checking" | "ready" | "unavailable";

/**
 * Whether the device that collects a locally executed Source can take its
 * work: the viewer's daemon states, plus `update_required` when the device
 * is online but its acquisition contract for the Source's type differs from
 * the server's.
 */
export type DeviceReadiness = LocalDaemonReadiness | "update_required";

export type SourceReadiness =
  | "checking_local_sync"
  | "local_sync_ready"
  | "local_sync_unavailable"
  | "update_local_sync"
  | "sign_in_required"
  | "configuration_required"
  | "account_mismatch";

const DEVICE_STATES: Readonly<Record<NonNullable<NonNullable<Source["execution"]>["device"]>["state"], DeviceReadiness>> = {
  online: "ready",
  offline: "unavailable",
  update_required: "update_required",
};

/**
 * The device readiness of one Source. A Source whose acquisition runs on its
 * execution owner's device reads it from the source API's `execution.device`,
 * which comes from that owner's device reports; while it has no execution
 * device, no device stands in the way and its connection shows Finish
 * setup. Any other locally executed Source runs on the viewer's own daemon.
 */
export function sourceDeviceReadiness(source: Source, daemon: LocalDaemonReadiness): DeviceReadiness {
  const device = source.execution?.device;
  if (device) return DEVICE_STATES[device.state];
  return sourceExecutor(source) === "device" ? "ready" : daemon;
}

/**
 * The one compact readiness of a Source, by the shared precedence: device
 * readiness first when collection runs locally (checking, offline, update
 * required), because a device that cannot run the collection cannot repair
 * a connection either; then connection blockers; then Local sync ready.
 */
export function resolveSourceReadiness({
  localExecution,
  daemon,
  connectionStatus,
}: {
  localExecution: boolean;
  daemon?: DeviceReadiness;
  connectionStatus?: SourceConnectionStatus | null;
}): SourceReadiness | null {
  if (localExecution) {
    if (daemon === "checking" || daemon === undefined) return "checking_local_sync";
    if (daemon === "unavailable") return "local_sync_unavailable";
    if (daemon === "update_required") return "update_local_sync";
  }

  if (connectionStatus?.state === "action_required") {
    if (connectionStatus.reason === "identity_conflict") return "account_mismatch";
    if (connectionStatus.reason === "configuration") return "configuration_required";
    return "sign_in_required";
  }

  if (localExecution) return "local_sync_ready";
  return null;
}
